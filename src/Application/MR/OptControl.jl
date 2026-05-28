"""
    MR/OptControl.jl

High-level optimal control interface for Pulsar magnetic resonance applications.

Defines `MRControl` (a single struct bundling drifts, control operators, target
states, time grid, and optimiser settings) and `optimcon` (a one-call driver
that validates the problem, builds penalties, and runs the chosen optimiser).

Workflow:
    1. Build drift Hamiltonians and operators via `hamiltonian` / `spin_op`.
    2. Describe the problem in an `MRControl` struct.
    3. Call `optimcon(ctrl, guess)` — validates, configures, and runs optimisation.
    4. Inspect `result.controls` (optimised waveform) and `result.fidelity`.
"""

using LinearAlgebra
using Statistics
using Printf

# ─── Abstract supertype ───────────────────────────────────────────────────────

"""
    AbstractMRControl

Abstract supertype for all MR optimal control problem specifications.
Concrete subtypes: `MRControl` (Hilbert-space / state-transfer) and
`LindbladMRControl` (Liouville-space / open-system).
"""
abstract type AbstractMRControl <: AbstractOptimizationContext end

# ─── MRControl ────────────────────────────────────────────────────────────────

"""
    MRControl

Complete specification of a magnetic resonance optimal control problem —
drifts, control operators, target states, time grid, and optimiser settings
bundled in a single struct.

## Fields — problem definition

- `drifts::Vector{Matrix{ComplexF64}}` — drift Hamiltonians in rad/s, one per
  frequency-offset / field-strength ensemble member. For on-resonance: one element.
  Build with [`hamiltonian`](@ref).

- `operators::Vector{Matrix{ComplexF64}}` — dimensionless control operators
  (e.g. `Ix = 0.5σx`, **without** 2π). The physical RF contribution at step `n`
  for power level `p` is `p × waveform[k,n] × operators[k]` rad/s.
  Build with [`spin_op`](@ref).

- `rho_init::Vector{Vector{ComplexF64}}` — normalised initial state vectors.
  Single state transfer: `[spin_state(sys, :Iz)]`.
  Multiple (UR-type): `[ψ_a, ψ_b, ψ_c]`.

- `rho_targ::Vector{Vector{ComplexF64}}` — normalised target state vectors,
  one per element of `rho_init`.

## Fields — power levels

- `pwr_levels::Vector{Float64}` — RF power levels in rad/s. Single nominal level
  `[2π × RF_MAX_HZ]` for standard pulses. Multiple levels model B₁ inhomogeneity:
  the fidelity is averaged over all power levels and all drift Hamiltonians.

## Fields — time grid

- `pulse_dt::Vector{Float64}` — duration of each piecewise-constant step in seconds.
  Total pulse duration = `sum(pulse_dt)`. Uniform steps: `fill(dt, n_steps)`.

## Fields — penalties

- `penalties::Vector{Symbol}` — active penalty types. Options:
  - `:none`  — no penalty (default)
  - `:NS`    — norm-square  Σ w[k,n]²  (RF power regularisation)
  - `:SNS`   — spillout norm-square; penalises |w| > `u_bound` or < `l_bound`
  - `:DNS`   — derivative norm-square Σ (Δw)²  (smoothness)
- `p_weights::Vector{Float64}` — weight for each penalty (same length as `penalties`).
- `l_bound::Float64` — lower waveform bound (default −1.0 for normalised waveform).
- `u_bound::Float64` — upper waveform bound (default +1.0).

## Fields — optimisation

- `method::Symbol`     — `:lbfgs` (default, recommended) or `:grape`.
- `max_iter::Int`      — maximum iterations (default 500).
- `grad_tol::Float64`  — gradient norm convergence threshold (default 1e-7).
  L-BFGS stops early when the normalised gradient norm drops below this value.
- `fidelity::Symbol`   — state fidelity type passed to [`state_fidelity`](@ref).
  Default `:real`. GRAPE-compatible types: `:real`, `:square`, `:modulus`.
- `lbfgs_memory::Int`  — L-BFGS history pairs (default 10).

## Fields — output

- `verbose::Bool`        — print progress (default `true`).
- `print_interval::Int`  — print every N iterations (default 1).

## Fields — trajectory tracking

- `tracking::Vector{TrackingPoint}` — intermediate-time checkpoints for
  tracking GRAPE. Default `TrackingPoint[]` (empty = terminal-only GRAPE).
  When non-empty, `optimcon` automatically dispatches to `grape_tracking_kernel`
  regardless of `ctrl.method` (`:lbfgs` or `:grape`).

  Each [`TrackingPoint`](@ref) specifies a time-step index, a target density
  matrix, and a weight.  The fidelity is a weighted sum of `Tr(ρ_tar ρ(t_k))`
  over all checkpoints and ensemble members.

  Example — oscillate between ±Iz every 100 µs (2 µs steps, 50 per segment):
  ```julia
  n_seg = 50
  psi_pz = [1.0+0im, 0.0+0im]
  psi_mz = [0.0+0im, 1.0+0im]
  N_CK = 10
  checkpoints = [
      TrackingPoint(k * n_seg,
                    isodd(k) ? psi_mz * psi_mz' : psi_pz * psi_pz',
                    1.0 / N_CK)
      for k in 1:N_CK
  ]
  ctrl = MRControl(drifts=..., ..., tracking=checkpoints)
  ```

## Fields — backend

- `backend::Symbol` — computation backend for ensemble averaging:
  - `:cpu`   — multi-threaded CPU (default). Uses all Julia threads
                (`julia -t N` or `JULIA_NUM_THREADS=N`). Scales linearly
                with the number of available threads.
  - `:metal` — Apple Silicon GPU via Metal.jl. Effective for large spin
                systems (matrix dim ≥ 32). Requires `Metal.jl` in the
                environment. Uses Float32 arithmetic.
  - `:cuda`  — NVIDIA GPU via CUDA.jl. Effective for large spin systems.
                Requires `CUDA.jl` and a functional NVIDIA driver.
                Uses Float64 arithmetic.

Construct via the keyword constructor `MRControl(; ...)`.
"""
struct MRControl <: AbstractMRControl
    drifts         :: Vector{Matrix{ComplexF64}}
    operators      :: Vector{Matrix{ComplexF64}}
    rho_init       :: Vector{Vector{ComplexF64}}
    rho_targ       :: Vector{Vector{ComplexF64}}
    pwr_levels     :: Vector{Float64}
    pulse_dt       :: Vector{Float64}
    penalties      :: Vector{Symbol}
    p_weights      :: Vector{Float64}
    l_bound        :: Float64
    u_bound        :: Float64
    method         :: Symbol
    max_iter       :: Int
    grad_tol       :: Float64
    fidelity       :: Symbol
    lbfgs_memory   :: Int
    verbose        :: Bool
    print_interval :: Int
    backend        :: Symbol
    tracking       :: Vector{TrackingPoint}
    callback       :: Union{Nothing, Function}
    parameterization :: AbstractControlParameterization
    # Single-spin (dim=2) closed-system state-transfer fast path:
    #   nothing → auto-detect (default)
    #   true    → force on (errors if ineligible)
    #   false   → force off (always use generic kernel)
    fast_path      :: Union{Nothing, Bool}
end

"""
    MRControl(; drifts, operators, rho_init, rho_targ, pwr_levels, pulse_dt, kwargs...)

Keyword constructor for `MRControl` with sensible defaults.

# Required keyword arguments
- `drifts`, `operators` — see struct docs
- `rho_init`, `rho_targ` — accepts either a single `Vector{ComplexF64}` (wrapped
  automatically) or a `Vector{Vector{ComplexF64}}` for multiple state pairs
- `pwr_levels`, `pulse_dt`

# Optional keyword arguments (all have defaults)
```julia
penalties      = [:none]      # no constraints
p_weights      = [0.0]
l_bound        = -1.0
u_bound        = +1.0
method         = :lbfgs
max_iter       = 500
grad_tol       = 1e-7
fidelity       = :real
lbfgs_memory   = 10
verbose        = true
print_interval = 1
backend        = get_device() # :cpu | :metal | :cuda  (follows Pulsar.set_device!)
```

# Example
```julia
sys = mr_system("1H")
ctrl = MRControl(
    drifts     = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-6000, 6000, 25)],
    operators  = [spin_op(sys, :Ix), spin_op(sys, :Iy)],
    rho_init   = [spin_state(sys, :Iz)],
    rho_targ   = [spin_state(sys, :mIy)],
    pwr_levels = [2π * 10_000.0],          # 10 kHz max RF
    pulse_dt   = fill(2e-6, 250),          # 250 × 2 µs = 500 µs
    method     = :lbfgs,
    max_iter   = 500,
)
```
"""
function MRControl(;
    drifts         :: Vector{<:Matrix{<:Number}},
    operators      :: Vector{<:Matrix{<:Number}},
    rho_init,                                       # flexible: vec or vec-of-vecs
    rho_targ,
    pwr_levels     :: Vector{Float64},
    pulse_dt       :: Vector{Float64},
    penalties      :: Vector{Symbol}  = [:none],
    p_weights      :: Vector{Float64} = [0.0],
    l_bound        :: Float64         = -1.0,
    u_bound        :: Float64         = +1.0,
    method         :: Symbol          = :lbfgs,
    max_iter       :: Int             = 500,
    grad_tol       :: Float64         = 1e-7,
    fidelity       :: Symbol          = :real,
    lbfgs_memory   :: Int             = 10,
    verbose        :: Bool            = true,
    print_interval :: Int             = 1,
    backend        :: Symbol          = get_device(),
    tracking                          = TrackingPoint[],
    callback       :: Union{Nothing, Function} = nothing,
    parameterization :: AbstractControlParameterization = PiecewiseConstant(),
    fast_path      :: Union{Nothing, Bool} = nothing,
)
    # Normalise rho_init / rho_targ to Vector{Vector{ComplexF64}}
    _wrap(x::Vector{<:Number})           = [ComplexF64.(x)]
    _wrap(x::Vector{<:AbstractVector})   = [ComplexF64.(v) for v in x]
    rho_i = _wrap(rho_init)
    rho_t = _wrap(rho_targ)

    length(rho_i) == length(rho_t) ||
        throw(ArgumentError("rho_init and rho_targ must have the same length " *
                            "(got $(length(rho_i)) and $(length(rho_t)))"))
    length(penalties) == length(p_weights) ||
        throw(ArgumentError("penalties and p_weights must have the same length"))

    backend ∈ (:cpu, :metal, :cuda) ||
        throw(ArgumentError("backend must be :cpu, :metal, or :cuda (got :$backend)"))

    # Normalise tracking to Vector{TrackingPoint}
    ck = tracking isa Vector{TrackingPoint} ? tracking :
         [TrackingPoint(tp.step, tp.target_dm, tp.weight) for tp in tracking]

    return MRControl(
        Matrix{ComplexF64}.(drifts),
        Matrix{ComplexF64}.(operators),
        rho_i, rho_t,
        pwr_levels, pulse_dt,
        penalties, p_weights,
        l_bound, u_bound,
        method, max_iter, grad_tol, fidelity,
        lbfgs_memory, verbose, print_interval,
        backend,
        ck,
        callback,
        parameterization,
        fast_path,
    )
end

# ─── optimcon ─────────────────────────────────────────────────────────────────

"""
    optimcon(ctrl::MRControl, guess::Matrix{Float64}) → OptimizationResult

Configure and run GRAPE pulse optimisation using the `MRControl` interface.
Validates the problem, then runs the optimiser specified by `ctrl.method`.

# Arguments
- `ctrl`  — `MRControl` struct with all problem and optimiser settings.
- `guess` — initial waveform, shape `[n_ctrl × n_t]` where
            `n_ctrl = length(ctrl.operators)`, `n_t = length(ctrl.pulse_dt)`.
            Values should be in `[l_bound, u_bound]` (normalised amplitudes).
            Actual RF = `pwr_levels[j] × guess[k,n] × operators[k]` (rad/s).

# Returns
`OptimizationResult` with fields:
- `controls` — optimised waveform (same shape as `guess`, normalised)
- `fidelity` — best ensemble-averaged fidelity achieved
- `fidelity_history`, `gradient_norm_history` — per-iteration arrays
- `n_iterations`, `converged`, `termination_reason`, `total_time`

# Methods
| `ctrl.method`  | Description                                                    |
|----------------|----------------------------------------------------------------|
| `:lbfgs`       | L-BFGS with Armijo backtracking (default, most efficient)      |
| `:grape`       | Normalised gradient ascent (simple GRAPE)                      |
| `:lbfgsb`      | L-BFGS-B with Wolfe line search (gradient-based)               |
| `:cg`          | Nonlinear CG with PR+ direction (gradient-based)               |
| `:cmaes`       | CMA-ES — full-covariance evolutionary (derivative-free)        |
| `:pscmaes`     | PS-CMA-ES — parallel island CMA-ES (derivative-free)           |
| `:nelder_mead` | Nelder-Mead simplex (derivative-free; best for low dim)        |
| `:pso`         | Particle Swarm Optimisation (derivative-free)                  |
| `:de`          | Differential Evolution DE/rand/1/bin (derivative-free)         |

# Example
```julia
sys    = mr_system("1H")
drifts = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-6000, 6000, 25)]
ctrl   = MRControl(
    drifts=drifts, operators=[spin_op(sys,:Ix), spin_op(sys,:Iy)],
    rho_init=[spin_state(sys,:Iz)], rho_targ=[spin_state(sys,:mIy)],
    pwr_levels=[2π*10_000.0], pulse_dt=fill(2e-6,250),
)
guess  = 0.05 .* randn(2, 250)
result = optimcon(ctrl, guess)
@printf("Best fidelity: %.6f\\n", result.fidelity)
```
"""
function optimcon(ctrl::MRControl, guess::Matrix{Float64})::OptimizationResult
    _validate_mr_control(ctrl, guess)

    n_ctrl, n_t = size(guess)

    if ctrl.verbose
        T_us    = sum(ctrl.pulse_dt) * 1e6
        pwr_khz = maximum(ctrl.pwr_levels) / (2π * 1e3)
        n_ens   = length(ctrl.drifts) * length(ctrl.pwr_levels) * length(ctrl.rho_init)
        @printf("[optimcon] Pulse: n_ctrl=%d  n_t=%d  T=%.1f µs  pwr_max=%.2f kHz\n",
                n_ctrl, n_t, T_us, pwr_khz)
        @printf("[optimcon] Ensemble: %d drift(s) × %d power(s) × %d state pair(s) = %d members\n",
                length(ctrl.drifts), length(ctrl.pwr_levels),
                length(ctrl.rho_init), n_ens)
        @printf("[optimcon] Method: %s   max_iter: %d   fidelity: %s   backend: %s\n\n",
                ctrl.method, ctrl.max_iter, ctrl.fidelity, ctrl.backend)
    end

    # Tracking GRAPE: non-empty ctrl.tracking overrides the kernel but keeps
    # the same outer optimiser (lbfgs / grape) requested by ctrl.method.
    if !isempty(ctrl.tracking)
        if !(ctrl.parameterization isa PiecewiseConstant)
            throw(ArgumentError(
                "Tracking mode does not yet support non-default parameterization " *
                "(got $(typeof(ctrl.parameterization))). Use the default " *
                "PiecewiseConstant() with tracking, or run without tracking."))
        end
        if ctrl.verbose
            @printf("[optimcon] Tracking mode: %d checkpoint(s)\n",
                    length(ctrl.tracking))
        end
        return if ctrl.method == :lbfgs || ctrl.method in _MR_GENERIC_METHODS
            _optimcon_lbfgs_tracking(ctrl, guess)
        else
            _optimcon_grape_tracking(ctrl, guess)
        end
    end

    # Non-trivial parameterization: route :lbfgs / :grape through the generic
    # closure-based path with :lbfgsb substitution (the closest analog).
    if !(ctrl.parameterization isa PiecewiseConstant) &&
       ctrl.method in (:lbfgs, :grape)
        if ctrl.verbose
            @printf("[optimcon] parameterization=%s: routing :%s → :lbfgsb\n",
                    nameof(typeof(ctrl.parameterization)), ctrl.method)
        end
        return _optimcon_generic(ctrl, guess; method = :lbfgsb)
    end

    return if ctrl.method == :lbfgs
        _optimcon_lbfgs(ctrl, guess)
    elseif ctrl.method == :grape
        _optimcon_grape(ctrl, guess)
    elseif ctrl.method in _MR_GENERIC_METHODS
        _optimcon_generic(ctrl, guess)
    else
        throw(ArgumentError(
            "Unknown method ':$(ctrl.method)'. Supported: " *
            ":lbfgs, :grape, :lbfgsb, :cg, :cmaes, :pscmaes, :nelder_mead, :pso, :de"))
    end
end

# ─── L-BFGS optimiser ─────────────────────────────────────────────────────────
#
# Single implementation for both `MRControl` (Hilbert) and `LindbladMRControl`
# (Liouville). The physics kernel is selected by Julia dispatch through
# `_mr_kernel(w, ctrl)`. Per-context log prefix and metadata are produced by
# `_optimcon_log_prefix(ctrl)` / `_optimcon_lbfgs_metadata(ctrl)` /
# `_optimcon_grape_metadata(ctrl)` (defined alongside `_mr_kernel`).

function _optimcon_lbfgs(ctrl::AbstractMRControl,
                          guess::Matrix{Float64})::OptimizationResult
    t_start = time()

    w = copy(guess)
    clamp!(w, ctrl.l_bound, ctrl.u_bound)

    F_curr, G_curr = _mr_kernel(w, ctrl)
    best_w  = copy(w)
    best_F  = F_curr
    n_fid   = 1
    n_grad  = 1

    fidelity_history      = Float64[F_curr]
    gradient_norm_history = Float64[norm(G_curr)]

    s_list = Vector{Vector{Float64}}()
    y_list = Vector{Vector{Float64}}()

    converged = false
    reason    = "max iterations reached"

    if ctrl.verbose
        @printf("  %6s  %-14s  %-12s  %s\n", "Iter", "F_ensemble", "ΔF", "|∇F|")
        println("  " * "─"^52)
        @printf("  %6s  %.8f  %12s  %.3e\n", "init", F_curr, "—", norm(G_curr))
    end

    for iter in 1:ctrl.max_iter
        g_flat = vec(G_curr)

        d_flat = _lbfgs_direction(g_flat, s_list, y_list)

        d_max = maximum(abs, d_flat)
        d_max < 1e-14 && (converged = true; reason = "zero gradient direction"; break)
        d_flat ./= d_max
        d_mat = reshape(d_flat, size(w))

        α0 = 0.05 * (ctrl.u_bound - ctrl.l_bound)
        α, w_new, F_new, n_ls = _line_search(w, d_mat, F_curr, α0, ctrl)
        n_fid += n_ls

        F_new2, G_new = _mr_kernel(w_new, ctrl)
        n_fid  += 1
        n_grad += 1

        s_vec = vec(w_new .- w)
        y_vec = vec(G_new) .- g_flat
        if dot(y_vec, s_vec) > 1e-14 * norm(s_vec)^2
            push!(s_list, s_vec)
            push!(y_list, y_vec)
            if length(s_list) > ctrl.lbfgs_memory
                popfirst!(s_list); popfirst!(y_list)
            end
        end

        w      = w_new
        G_curr = G_new
        F_prev = F_curr
        F_curr = F_new2

        push!(fidelity_history, F_curr)
        gnorm = norm(G_curr)
        push!(gradient_norm_history, gnorm)

        if F_curr > best_F
            best_F = F_curr
            best_w .= w
        end

        if ctrl.verbose && (iter % ctrl.print_interval == 0 || iter == 1)
            @printf("  %6d  %.8f  %+.3e  %.3e\n",
                    iter, F_curr, F_curr - F_prev, gnorm)
        end

        ctrl.max_iter > 0 && F_curr >= 1.0 - 1e-8 &&
            (converged = true; reason = "near-perfect fidelity"; break)
        gnorm < ctrl.grad_tol &&
            (converged = true; reason = "gradient norm < $(ctrl.grad_tol)"; break)
        length(fidelity_history) >= 2 &&
            abs(F_curr - F_prev) < 1e-9 &&
            (converged = true; reason = "fidelity stalled"; break)
    end

    t_elapsed = time() - t_start

    if ctrl.verbose
        println()
        @printf("%s Done: F=%.8f  iter=%d  time=%.2f s  %s\n",
                _optimcon_log_prefix(ctrl), best_F,
                length(fidelity_history) - 1, t_elapsed,
                converged ? "✓ " * reason : reason)
    end

    return OptimizationResult(
        best_w, best_F,
        fidelity_history, gradient_norm_history,
        length(fidelity_history) - 1,
        converged, reason, t_elapsed,
        n_fid, n_grad,
        _optimcon_lbfgs_metadata(ctrl),
    )
end

# ─── Simple GRAPE ─────────────────────────────────────────────────────────────

function _optimcon_grape(ctrl::AbstractMRControl,
                          guess::Matrix{Float64})::OptimizationResult
    t_start = time()

    w = copy(guess)
    clamp!(w, ctrl.l_bound, ctrl.u_bound)

    F_curr, G_curr = _mr_kernel(w, ctrl)
    best_w = copy(w)
    best_F = F_curr
    n_fid  = 1
    n_grad = 1

    fidelity_history      = Float64[F_curr]
    gradient_norm_history = Float64[norm(G_curr)]

    converged = false
    reason    = "max iterations reached"
    step_size = 0.05

    if ctrl.verbose
        @printf("  %6s  %-14s  %-12s  %s\n", "Iter", "F_ensemble", "ΔF", "|∇F|")
        println("  " * "─"^52)
        @printf("  %6s  %.8f  %12s  %.3e\n", "init", F_curr, "—", norm(G_curr))
    end

    for iter in 1:ctrl.max_iter
        g_max = maximum(abs, G_curr)
        if g_max > 1e-14
            step = step_size * (ctrl.u_bound - ctrl.l_bound) / g_max
            w .+= step .* G_curr
            clamp!(w, ctrl.l_bound, ctrl.u_bound)
        end

        F_prev = F_curr
        F_curr, G_curr = _mr_kernel(w, ctrl)
        n_fid  += 1
        n_grad += 1

        push!(fidelity_history, F_curr)
        gnorm = norm(G_curr)
        push!(gradient_norm_history, gnorm)

        if F_curr > best_F
            best_F = F_curr
            best_w .= w
        end

        if ctrl.verbose && (iter % ctrl.print_interval == 0 || iter == 1)
            @printf("  %6d  %.8f  %+.3e  %.3e\n",
                    iter, F_curr, F_curr - F_prev, gnorm)
        end

        F_curr >= 1.0 - 1e-8 && (converged = true; reason = "near-perfect fidelity"; break)
        gnorm < ctrl.grad_tol && (converged = true; reason = "gradient norm < $(ctrl.grad_tol)"; break)
    end

    t_elapsed = time() - t_start

    if ctrl.verbose
        println()
        @printf("%s Done: F=%.8f  iter=%d  time=%.2f s  %s\n",
                _optimcon_log_prefix(ctrl), best_F,
                length(fidelity_history) - 1, t_elapsed,
                converged ? "✓ " * reason : reason)
    end

    return OptimizationResult(
        best_w, best_F,
        fidelity_history, gradient_norm_history,
        length(fidelity_history) - 1,
        converged, reason, t_elapsed,
        n_fid, n_grad,
        _optimcon_grape_metadata(ctrl),
    )
end

# ─── L-BFGS two-loop recursion ────────────────────────────────────────────────

function _lbfgs_direction(g::Vector{Float64},
                           s_list::Vector{Vector{Float64}},
                           y_list::Vector{Vector{Float64}})::Vector{Float64}
    k = length(s_list)
    k == 0 && return copy(g)    # first step: steepest ascent

    q = copy(g)
    α = zeros(k)
    ρ = [1.0 / dot(y_list[i], s_list[i]) for i in 1:k]

    for i in k:-1:1
        α[i] = ρ[i] * dot(s_list[i], q)
        q   .-= α[i] .* y_list[i]
    end

    # Hessian initialisation: γ = (s·y)/(y·y)
    γ = dot(s_list[end], y_list[end]) / dot(y_list[end], y_list[end])
    r = γ .* q

    for i in 1:k
        β  = ρ[i] * dot(y_list[i], r)
        r .+= (α[i] - β) .* s_list[i]
    end

    return r   # ascent direction (same sign as gradient)
end

# ─── Armijo backtracking line search ─────────────────────────────────────────

function _line_search(w::Matrix{Float64}, d::Matrix{Float64},
                      F0::Float64, α0::Float64, ctrl)
    α    = α0
    c1   = 1e-4          # Armijo sufficient-increase parameter
    max_halvings = 20

    for halving in 1:max_halvings
        w_new = clamp.(w .+ α .* d, ctrl.l_bound, ctrl.u_bound)
        F_new, _ = _mr_kernel(w_new, ctrl)
        F_new >= F0 - c1 * abs(α) && return α, w_new, F_new, halving + 1
        α *= 0.5
    end

    # Fall back: return the step (or zero step) that was best
    w_new = clamp.(w .+ α .* d, ctrl.l_bound, ctrl.u_bound)
    F_new, _ = _mr_kernel(w_new, ctrl)
    return α, F_new > F0 ? w_new : copy(w), max(F_new, F0), max_halvings + 1
end

# ─── Validation ───────────────────────────────────────────────────────────────

function _validate_mr_control(ctrl::MRControl, guess::Matrix{Float64})
    isempty(ctrl.drifts)    && throw(ArgumentError("ctrl.drifts must be non-empty"))
    isempty(ctrl.operators) && throw(ArgumentError("ctrl.operators must be non-empty"))
    isempty(ctrl.rho_init)  && throw(ArgumentError("ctrl.rho_init must be non-empty"))
    isempty(ctrl.pwr_levels)&& throw(ArgumentError("ctrl.pwr_levels must be non-empty"))
    isempty(ctrl.pulse_dt)  && throw(ArgumentError("ctrl.pulse_dt must be non-empty"))

    n_ctrl, n_t = size(guess)
    n_ctrl == length(ctrl.operators) ||
        throw(ArgumentError("guess has $(n_ctrl) rows but ctrl.operators has " *
                            "$(length(ctrl.operators)) entries"))
    n_t == length(ctrl.pulse_dt) ||
        throw(ArgumentError("guess has $(n_t) columns but ctrl.pulse_dt has " *
                            "$(length(ctrl.pulse_dt)) entries"))

    dim = size(ctrl.drifts[1], 1)
    for (i, op) in enumerate(ctrl.operators)
        size(op) == (dim, dim) ||
            throw(ArgumentError("ctrl.operators[$i] has wrong size $(size(op)); " *
                                "expected ($dim, $dim)"))
    end
    for (i, ψ) in enumerate(ctrl.rho_init)
        length(ψ) == dim ||
            throw(ArgumentError("ctrl.rho_init[$i] has length $(length(ψ)); " *
                                "expected $dim"))
    end
    for (i, ψ) in enumerate(ctrl.rho_targ)
        length(ψ) == dim ||
            throw(ArgumentError("ctrl.rho_targ[$i] has length $(length(ψ)); " *
                                "expected $dim"))
    end

    _grape_fid_types = (:real, :square, :modulus)
    ctrl.fidelity ∈ _grape_fid_types ||
        throw(ArgumentError(
            "ctrl.fidelity ':$(ctrl.fidelity)' is not GRAPE-compatible. " *
            "Use one of: " * join(string.(_grape_fid_types), ", ")))
    ctrl.l_bound < ctrl.u_bound ||
        throw(ArgumentError("ctrl.l_bound must be < ctrl.u_bound"))
end

# ═══════════════════════════════════════════════════════════════════════════════
# LindbladMRControl — open-system optimal control (Lindblad / Liouville space)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Parallel to MRControl but for open quantum systems governed by the Lindblad
# master equation. The user-facing API is identical except for two extra fields:
#   jump_ops    — Lindblad jump operators L_k (N×N each)
#   decay_rates — rates γ_k in rad/s
# These are built by `mr_relaxation(sys; T1, T2star)`.
#
# Switching between closed and open system:
#   Closed:  ctrl = MRControl(drifts=..., ...)
#   Open:    ctrl = LindbladMRControl(drifts=..., jump_ops=..., decay_rates=..., ...)
# Then call `optimcon(ctrl, guess)` identically for both — Julia dispatch routes
# to the correct kernel (grape_state_kernel vs grape_lindblad_kernel).
#
# Internal precomputation (done once in the constructor):
#   _L_drifts   — one N²×N² Liouvillian per drift member (coherent + dissipative)
#   _L_controls — one N²×N² coherent Liouvillian per control operator
#   _sigma_init — vec(ρ_init[s]) for each initial state (length N²)
#   _sigma_targ — vec(ρ_targ[s]) for each target state  (length N²)
# These are marked with a leading underscore to signal they are not user-facing.

"""
    LindbladMRControl

Complete specification of an open-system MR optimal control problem.
Extends `MRControl` with Lindblad jump operators for T1/T2 relaxation.

## User-facing fields (same as `MRControl`)

- `drifts`, `operators`, `rho_init`, `rho_targ`, `pwr_levels`, `pulse_dt`
- `penalties`, `p_weights`, `l_bound`, `u_bound`
- `method`, `max_iter`, `fidelity`, `lbfgs_memory`, `verbose`, `print_interval`

## Additional fields — relaxation

- `jump_ops::Vector{Matrix{ComplexF64}}` — Lindblad operators L_k (N×N each).
  Build with [`mr_relaxation`](@ref).
- `decay_rates::Vector{Float64}` — rates γ_k in rad/s. Same length as `jump_ops`.

## Internal precomputed fields (not user-facing)

- `_hilbert_dim`   — N (Hilbert space dimension)
- `_liouville_dim` — N² (Liouville space dimension)
- `_L_drifts`      — N²×N² drift Liouvillians (one per ensemble member)
- `_L_controls`    — N²×N² control Liouvillians (one per control operator)
- `_sigma_init`    — vec(ρ_init[s]) (length N²)
- `_sigma_targ`    — vec(ρ_targ[s]) (length N²)

Construct via the keyword constructor `LindbladMRControl(; ...)`.

## Example
```julia
sys = mr_system("13C")
drifts = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-6000, 6000, 25)]
Lx, Ly = spin_op(sys, :Ix), spin_op(sys, :Iy)

# T1 = 2 s, T2* = 50 ms (typical ¹³C at 800 MHz)
jump_ops, rates = mr_relaxation(sys; T1=2.0, T2star=0.05)

ctrl = LindbladMRControl(
    drifts      = drifts,
    operators   = [Lx, Ly],
    rho_init    = [spin_state(sys, :Iz)],
    rho_targ    = [spin_state(sys, :mIz)],
    jump_ops    = jump_ops,
    decay_rates = rates,
    pwr_levels  = [2π * 600.0],
    pulse_dt    = fill(dt, N_TS),
    fidelity    = :square,
    max_iter    = 500,
)
guess  = 0.05 .* randn(2, N_TS)
result = optimcon(ctrl, guess)
```
"""
struct LindbladMRControl <: AbstractMRControl
    # ── User-facing problem definition ────────────────────────────────────────
    drifts         :: Vector{Matrix{ComplexF64}}
    operators      :: Vector{Matrix{ComplexF64}}
    jump_ops       :: Vector{Matrix{ComplexF64}}
    decay_rates    :: Vector{Float64}
    rho_init       :: Vector{Vector{ComplexF64}}   # pure-state vecs (length N)
    rho_targ       :: Vector{Vector{ComplexF64}}   # pure-state vecs (length N)
    pwr_levels     :: Vector{Float64}
    pulse_dt       :: Vector{Float64}
    # ── Penalties and bounds ──────────────────────────────────────────────────
    penalties      :: Vector{Symbol}
    p_weights      :: Vector{Float64}
    l_bound        :: Float64
    u_bound        :: Float64
    # ── Optimisation settings ─────────────────────────────────────────────────
    method         :: Symbol
    max_iter       :: Int
    grad_tol       :: Float64
    fidelity       :: Symbol
    lbfgs_memory   :: Int
    verbose        :: Bool
    print_interval :: Int
    # ── Backend / precision ───────────────────────────────────────────────────
    backend        :: Symbol   # :cpu (default) | :metal | :cuda
    precision      :: Symbol   # :f64 (default) | :f32  (GPU only; Metal always :f32)
    # ── Precomputed Liouville-space objects (built in constructor) ─────────────
    _hilbert_dim   :: Int
    _liouville_dim :: Int
    _L_drifts      :: Vector{Matrix{ComplexF64}}   # 𝓛_drift[j] (N²×N²)
    _L_controls    :: Vector{Matrix{ComplexF64}}   # 𝓛_ctrl[k]  (N²×N²)
    _sigma_init    :: Vector{Vector{ComplexF64}}   # vec(ρ_init[s]) length N²
    _sigma_targ    :: Vector{Vector{ComplexF64}}   # vec(ρ_targ[s]) length N²
    # Single-spin (dim=2) closed-system Liouville fast path:
    #   nothing → auto-detect (default)
    #   true    → force on (errors if ineligible)
    #   false   → force off (always use generic 4×4 kernel)
    fast_path      :: Union{Nothing, Bool}
end

"""
    LindbladMRControl(; drifts, operators, jump_ops, decay_rates,
                        rho_init, rho_targ, pwr_levels, pulse_dt, kwargs...)

Keyword constructor for `LindbladMRControl`. Precomputes Liouvillians and
vectorised density matrices at construction time.

# Required keyword arguments
- `drifts`, `operators` — same as `MRControl`
- `jump_ops`    — Lindblad operators (from `mr_relaxation`); pass `[]` for closed
- `decay_rates` — matching rates in rad/s; pass `[]` for closed system
- `rho_init`, `rho_targ`, `pwr_levels`, `pulse_dt`

# Optional keyword arguments (all have defaults)
```julia
penalties      = [:none]
p_weights      = [0.0]
l_bound        = -1.0
u_bound        = +1.0
method         = :lbfgs
max_iter       = 500
grad_tol       = 1e-7
fidelity       = :square    # :square recommended for open systems
lbfgs_memory   = 10
verbose        = true
print_interval = 1
backend        = get_device()   # :cpu | :metal | :cuda
precision      = :f64           # :f64 | :f32  (GPU only; Metal always uses :f32)
```

`precision = :f32` halves GPU memory usage at the cost of single-precision arithmetic
(usually negligible for NMR gradient accuracy). Metal always uses Float32 regardless.
GPU support is most beneficial for N ≥ 8 (3+ spins, Liouville dim ≥ 64).
"""
function LindbladMRControl(;
    drifts         :: Vector{<:Matrix{<:Number}},
    operators      :: Vector{<:Matrix{<:Number}},
    jump_ops       :: Vector{<:Matrix{<:Number}} = Matrix{ComplexF64}[],
    decay_rates    :: Vector{Float64}            = Float64[],
    rho_init,
    rho_targ,
    pwr_levels     :: Vector{Float64},
    pulse_dt       :: Vector{Float64},
    penalties      :: Vector{Symbol}  = [:none],
    p_weights      :: Vector{Float64} = [0.0],
    l_bound        :: Float64         = -1.0,
    u_bound        :: Float64         = +1.0,
    method         :: Symbol          = :lbfgs,
    max_iter       :: Int             = 500,
    grad_tol       :: Float64         = 1e-7,
    fidelity       :: Symbol          = :square,
    lbfgs_memory   :: Int             = 10,
    verbose        :: Bool            = true,
    print_interval :: Int             = 1,
    backend        :: Symbol          = get_device(),
    precision      :: Symbol          = :f64,
    fast_path      :: Union{Nothing, Bool} = nothing,
)
    length(jump_ops) == length(decay_rates) ||
        throw(ArgumentError("jump_ops and decay_rates must have the same length " *
                            "(got $(length(jump_ops)) and $(length(decay_rates)))"))
    fidelity ∈ (:real, :square) ||
        throw(ArgumentError(
            "LindbladMRControl fidelity must be :square (recommended) or :real " *
            "(got :$fidelity). Modulus fidelity is not supported in Liouville space."))
    l_bound < u_bound ||
        throw(ArgumentError("l_bound must be < u_bound"))
    backend ∈ (:cpu, :metal, :cuda) ||
        throw(ArgumentError("backend must be :cpu, :metal, or :cuda (got :$backend)"))
    precision ∈ (:f32, :f64) ||
        throw(ArgumentError("precision must be :f32 or :f64 (got :$precision)"))

    # Normalise rho_init / rho_targ to Vector{Vector{ComplexF64}} (pure states)
    _wrap(x::Vector{<:Number})          = [ComplexF64.(x)]
    _wrap(x::Vector{<:AbstractVector})  = [ComplexF64.(v) for v in x]
    rho_i = _wrap(rho_init)
    rho_t = _wrap(rho_targ)
    length(rho_i) == length(rho_t) ||
        throw(ArgumentError("rho_init and rho_targ must have the same length"))

    drifts_cf    = Matrix{ComplexF64}.(drifts)
    operators_cf = Matrix{ComplexF64}.(operators)
    jump_ops_cf  = Matrix{ComplexF64}.(jump_ops)

    N  = size(drifts_cf[1], 1)      # Hilbert space dim
    N2 = N * N                       # Liouville space dim

    # ── Precompute Liouvillians ────────────────────────────────────────────────
    # Drift Liouvillians (one per ensemble member, coherent + dissipative)
    _L_drifts = [build_drift_liouvillian(H, jump_ops_cf, decay_rates)
                  for H in drifts_cf]
    # Control Liouvillians (one per operator, coherent only)
    _L_controls = [build_control_liouvillian(op) for op in operators_cf]

    # ── Vectorise initial and target states ────────────────────────────────────
    # Accept pure-state vectors (length N) → vec(|ψ⟩⟨ψ|) (length N²)
    # Also accept length-N² vectors directly (already vectorised density matrix)
    _to_sigma(ψ::Vector{ComplexF64}) =
        length(ψ) == N  ? pure_state_to_vec_rho(ψ)  :
        length(ψ) == N2 ? ψ :
        throw(ArgumentError("State vector has length $(length(ψ)); " *
                            "expected N=$N (pure state) or N²=$N2 (vec(ρ))"))
    sigma_i = [_to_sigma(ψ) for ψ in rho_i]
    sigma_t = [_to_sigma(ψ) for ψ in rho_t]

    return LindbladMRControl(
        drifts_cf, operators_cf, jump_ops_cf, decay_rates,
        rho_i, rho_t,
        pwr_levels, pulse_dt,
        penalties, p_weights,
        l_bound, u_bound,
        method, max_iter, grad_tol, fidelity,
        lbfgs_memory, verbose, print_interval,
        backend, precision,
        N, N2,
        _L_drifts, _L_controls,
        sigma_i, sigma_t,
        fast_path,
    )
end

# ─── optimcon for LindbladMRControl ──────────────────────────────────────────

"""
    optimcon(ctrl::LindbladMRControl, guess::Matrix{Float64}) → OptimizationResult

Configure and run GRAPE pulse optimisation for an open quantum system using the
Lindblad master equation. Identical call signature to `optimcon(ctrl::MRControl, ...)`;
Julia dispatch routes to this method automatically when `ctrl` is `LindbladMRControl`.

Internally calls `grape_lindblad_kernel` (Liouville-space GRAPE) instead of
`grape_state_kernel` (Hilbert-space GRAPE). All other logic (L-BFGS, line search,
penalty handling, output) is identical.

# Example
```julia
# Closed system
ctrl_closed = MRControl(drifts=drifts, operators=[Lx,Ly], ...)
result_closed = optimcon(ctrl_closed, guess)

# Open system — same call, different ctrl type
jump_ops, rates = mr_relaxation(sys; T1=2.0, T2star=0.05)
ctrl_open   = LindbladMRControl(drifts=drifts, operators=[Lx,Ly],
                                 jump_ops=jump_ops, decay_rates=rates, ...)
result_open = optimcon(ctrl_open, guess)    # ← same function call
```
"""
function optimcon(ctrl::LindbladMRControl, guess::Matrix{Float64})::OptimizationResult
    _validate_lindblad_control(ctrl, guess)

    n_ctrl, n_t = size(guess)

    if ctrl.verbose
        T_us    = sum(ctrl.pulse_dt) * 1e6
        pwr_khz = maximum(ctrl.pwr_levels) / (2π * 1e3)
        n_ens   = length(ctrl._L_drifts) * length(ctrl.pwr_levels) *
                  length(ctrl._sigma_init)
        @printf("[optimcon/Lindblad] Pulse: n_ctrl=%d  n_t=%d  T=%.1f µs  pwr_max=%.2f kHz\n",
                n_ctrl, n_t, T_us, pwr_khz)
        @printf("[optimcon/Lindblad] Hilbert dim N=%d  Liouville dim N²=%d\n",
                ctrl._hilbert_dim, ctrl._liouville_dim)
        @printf("[optimcon/Lindblad] Ensemble: %d drift × %d pwr × %d pair = %d members\n",
                length(ctrl._L_drifts), length(ctrl.pwr_levels),
                length(ctrl._sigma_init), n_ens)
        @printf("[optimcon/Lindblad] Jump operators: %d  (T1/T2 channels)\n",
                length(ctrl.jump_ops))
        prec_str = ctrl.backend != :cpu ? "  precision: $(ctrl.precision)" : ""
        @printf("[optimcon/Lindblad] Method: %s   max_iter: %d   fidelity: %s   backend: %s%s\n\n",
                ctrl.method, ctrl.max_iter, ctrl.fidelity, ctrl.backend, prec_str)
    end

    return if ctrl.method == :lbfgs
        _optimcon_lbfgs(ctrl, guess)
    elseif ctrl.method == :grape
        _optimcon_grape(ctrl, guess)
    elseif ctrl.method in _MR_GENERIC_METHODS
        _optimcon_generic(ctrl, guess)
    else
        throw(ArgumentError(
            "Unknown method ':$(ctrl.method)'. Supported: " *
            ":lbfgs, :grape, :lbfgsb, :cg, :cmaes, :pscmaes, :nelder_mead, :pso, :de"))
    end
end

# NOTE: The previously-separate `_optimcon_lindblad_lbfgs` and
# `_optimcon_lindblad_grape` functions were removed in Theme 6b. The unified
# `_optimcon_lbfgs` / `_optimcon_grape` (defined further up, dispatched on
# `AbstractMRControl`) now serve both Hilbert and Liouville contexts via
# `_mr_kernel(w, ctrl)`. Per-context metadata is provided by
# `_optimcon_lbfgs_metadata` / `_optimcon_grape_metadata`.

# ─── Validation ───────────────────────────────────────────────────────────────

function _validate_lindblad_control(ctrl::LindbladMRControl, guess::Matrix{Float64})
    isempty(ctrl.drifts)     && throw(ArgumentError("ctrl.drifts must be non-empty"))
    isempty(ctrl.operators)  && throw(ArgumentError("ctrl.operators must be non-empty"))
    isempty(ctrl.pwr_levels) && throw(ArgumentError("ctrl.pwr_levels must be non-empty"))
    isempty(ctrl.pulse_dt)   && throw(ArgumentError("ctrl.pulse_dt must be non-empty"))

    n_ctrl, n_t = size(guess)
    n_ctrl == length(ctrl.operators) ||
        throw(ArgumentError("guess rows $(n_ctrl) ≠ length(ctrl.operators) " *
                            "$(length(ctrl.operators))"))
    n_t == length(ctrl.pulse_dt) ||
        throw(ArgumentError("guess columns $n_t ≠ length(ctrl.pulse_dt) " *
                            "$(length(ctrl.pulse_dt))"))
    ctrl.l_bound < ctrl.u_bound ||
        throw(ArgumentError("ctrl.l_bound must be < ctrl.u_bound"))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Generic optimizer support — works with MRControl and LindbladMRControl
# ═══════════════════════════════════════════════════════════════════════════════
#
# All Pulsar optimizers (CMA-ES, PSO, Nelder-Mead, L-BFGS-B, CG, DE) can be
# used with either Hilbert-space or Liouville-space problems by setting
# ctrl.method to the appropriate symbol. Julia dispatch selects the correct
# physics kernel (grape_state_kernel vs grape_lindblad_kernel) automatically.
#
# Gradient-based methods (:lbfgsb, :cg) use the GRAPE gradient via GRAPEFamily.jl.
# Derivative-free methods (:cmaes, :pscmaes, :nelder_mead, :pso, :de) use only
# the forward fidelity; no gradient computation is required.
#
# Note: for gradient-based methods, :lbfgs (built-in) remains more efficient
# than :lbfgsb/:cg because it avoids the flat-vector round-trip.

# Set of method symbols routed through _optimcon_generic
const _MR_GENERIC_METHODS = (:lbfgsb, :cg, :cmaes, :pscmaes, :nelder_mead, :pso, :de)

# ─── Physics kernel dispatch ───────────────────────────────────────────────────
# Single-dispatch bridge: call the correct GRAPE kernel for each control type.

_mr_kernel(w::Matrix{Float64}, ctrl::MRControl)         = grape_state_kernel(w, ctrl)
_mr_kernel(w::Matrix{Float64}, ctrl::LindbladMRControl) = grape_lindblad_kernel(w, ctrl)

# When tracking checkpoints are present, route through the tracking kernel.
_mr_kernel_tracking(w::Matrix{Float64}, ctrl::MRControl) =
    grape_tracking_kernel(w, ctrl)

# ─── Per-context log prefix and OptimizationResult metadata ───────────────────
# Used by the unified `_optimcon_lbfgs` / `_optimcon_grape` loops so the only
# thing that varies between Hilbert and Liouville contexts is the kernel call
# and the diagnostic strings.

_optimcon_log_prefix(::MRControl)         = "[optimcon]"
_optimcon_log_prefix(::LindbladMRControl) = "[optimcon/Lindblad]"

_optimcon_lbfgs_metadata(ctrl::MRControl) = Dict{String,Any}(
    "algorithm"    => "MR L-BFGS-GRAPE",
    "n_ensemble"   => length(ctrl.drifts) * length(ctrl.pwr_levels) *
                       length(ctrl.rho_init),
    "lbfgs_memory" => ctrl.lbfgs_memory,
)

_optimcon_lbfgs_metadata(ctrl::LindbladMRControl) = Dict{String,Any}(
    "algorithm"      => "Lindblad L-BFGS-GRAPE",
    "hilbert_dim"    => ctrl._hilbert_dim,
    "liouville_dim"  => ctrl._liouville_dim,
    "n_jump_ops"     => length(ctrl.jump_ops),
    "n_ensemble"     => length(ctrl._L_drifts) * length(ctrl.pwr_levels) *
                         length(ctrl._sigma_init),
    "lbfgs_memory"   => ctrl.lbfgs_memory,
)

_optimcon_grape_metadata(::MRControl) =
    Dict{String,Any}("algorithm" => "MR GRAPE (gradient ascent)")

_optimcon_grape_metadata(::LindbladMRControl) =
    Dict{String,Any}("algorithm" => "Lindblad GRAPE (gradient ascent)")

# ─── Tracking optimisers ──────────────────────────────────────────────────────
# Identical loop structure to _optimcon_lbfgs / _optimcon_grape, but call
# grape_tracking_kernel instead of grape_state_kernel.

function _optimcon_lbfgs_tracking(ctrl::MRControl, guess::Matrix{Float64})::OptimizationResult
    t_start = time()

    w = copy(guess)
    clamp!(w, ctrl.l_bound, ctrl.u_bound)

    F_curr, G_curr = grape_tracking_kernel(w, ctrl)
    best_w  = copy(w)
    best_F  = F_curr
    n_fid   = 1
    n_grad  = 1

    fidelity_history      = Float64[F_curr]
    gradient_norm_history = Float64[norm(G_curr)]

    s_list = Vector{Vector{Float64}}()
    y_list = Vector{Vector{Float64}}()

    converged = false
    reason    = "max iterations reached"

    if ctrl.verbose
        @printf("  %6s  %-14s  %-12s  %s\n", "Iter", "F_tracking", "ΔF", "|∇F|")
        println("  " * "─"^52)
        @printf("  %6s  %.8f  %12s  %.3e\n", "init", F_curr, "—", norm(G_curr))
    end

    for iter in 1:ctrl.max_iter
        g_flat = vec(G_curr)
        d_flat = _lbfgs_direction(g_flat, s_list, y_list)
        d_max  = maximum(abs, d_flat)
        d_max < 1e-14 && (converged = true; reason = "zero gradient direction"; break)
        d_flat ./= d_max
        d_mat = reshape(d_flat, size(w))

        α0 = 0.05 * (ctrl.u_bound - ctrl.l_bound)
        α, w_new, F_new, n_ls = _line_search_tracking(w, d_mat, F_curr, α0, ctrl)
        n_fid += n_ls

        F_new2, G_new = grape_tracking_kernel(w_new, ctrl)
        n_fid  += 1
        n_grad += 1

        s_vec = vec(w_new .- w)
        y_vec = vec(G_new) .- g_flat
        if dot(y_vec, s_vec) > 1e-14 * norm(s_vec)^2
            push!(s_list, s_vec)
            push!(y_list, y_vec)
            if length(s_list) > ctrl.lbfgs_memory
                popfirst!(s_list); popfirst!(y_list)
            end
        end

        w      = w_new
        G_curr = G_new
        F_prev = F_curr
        F_curr = F_new2

        push!(fidelity_history, F_curr)
        gnorm = norm(G_curr)
        push!(gradient_norm_history, gnorm)

        if F_curr > best_F
            best_F = F_curr
            best_w .= w
        end

        if ctrl.verbose && (iter % ctrl.print_interval == 0 || iter == 1)
            @printf("  %6d  %.8f  %+.3e  %.3e\n",
                    iter, F_curr, F_curr - F_prev, gnorm)
        end

        F_curr >= 1.0 - 1e-8 && (converged = true; reason = "near-perfect fidelity"; break)
        gnorm < ctrl.grad_tol && (converged = true; reason = "gradient norm < $(ctrl.grad_tol)"; break)
        length(fidelity_history) >= 2 && abs(F_curr - F_prev) < 1e-9 &&
            (converged = true; reason = "fidelity stalled"; break)
    end

    t_elapsed = time() - t_start

    if ctrl.verbose
        println()
        @printf("[optimcon] Done: F=%.8f  iter=%d  time=%.2f s  %s\n",
                best_F, length(fidelity_history) - 1, t_elapsed,
                converged ? "✓ " * reason : reason)
    end

    return OptimizationResult(
        best_w, best_F,
        fidelity_history, gradient_norm_history,
        length(fidelity_history) - 1,
        converged, reason, t_elapsed,
        n_fid, n_grad,
        Dict{String,Any}(
            "algorithm"    => "MR Tracking L-BFGS-GRAPE",
            "n_checkpoints"=> length(ctrl.tracking),
            "lbfgs_memory" => ctrl.lbfgs_memory,
        )
    )
end

function _optimcon_grape_tracking(ctrl::MRControl, guess::Matrix{Float64})::OptimizationResult
    t_start = time()

    w = copy(guess)
    clamp!(w, ctrl.l_bound, ctrl.u_bound)

    F_curr, G_curr = grape_tracking_kernel(w, ctrl)
    best_w = copy(w)
    best_F = F_curr
    n_fid  = 1
    n_grad = 1

    fidelity_history      = Float64[F_curr]
    gradient_norm_history = Float64[norm(G_curr)]

    converged  = false
    reason     = "max iterations reached"
    step_size  = 0.05

    if ctrl.verbose
        @printf("  %6s  %-14s  %-12s  %s\n", "Iter", "F_tracking", "ΔF", "|∇F|")
        println("  " * "─"^52)
        @printf("  %6s  %.8f  %12s  %.3e\n", "init", F_curr, "—", norm(G_curr))
    end

    for iter in 1:ctrl.max_iter
        g_max = maximum(abs, G_curr)
        if g_max > 1e-14
            step = step_size * (ctrl.u_bound - ctrl.l_bound) / g_max
            w .+= step .* G_curr
            clamp!(w, ctrl.l_bound, ctrl.u_bound)
        end

        F_prev = F_curr
        F_curr, G_curr = grape_tracking_kernel(w, ctrl)
        n_fid  += 1
        n_grad += 1

        push!(fidelity_history, F_curr)
        gnorm = norm(G_curr)
        push!(gradient_norm_history, gnorm)

        if F_curr > best_F
            best_F = F_curr
            best_w .= w
        end

        if ctrl.verbose && (iter % ctrl.print_interval == 0 || iter == 1)
            @printf("  %6d  %.8f  %+.3e  %.3e\n",
                    iter, F_curr, F_curr - F_prev, gnorm)
        end

        F_curr >= 1.0 - 1e-8 && (converged = true; reason = "near-perfect fidelity"; break)
        gnorm < ctrl.grad_tol && (converged = true; reason = "gradient norm < $(ctrl.grad_tol)"; break)
    end

    t_elapsed = time() - t_start

    if ctrl.verbose
        println()
        @printf("[optimcon] Done: F=%.8f  iter=%d  time=%.2f s  %s\n",
                best_F, length(fidelity_history) - 1, t_elapsed,
                converged ? "✓ " * reason : reason)
    end

    return OptimizationResult(
        best_w, best_F,
        fidelity_history, gradient_norm_history,
        length(fidelity_history) - 1,
        converged, reason, t_elapsed,
        n_fid, n_grad,
        Dict{String,Any}(
            "algorithm"    => "MR Tracking GRAPE (gradient ascent)",
            "n_checkpoints"=> length(ctrl.tracking),
        )
    )
end

# Armijo line search for tracking kernel
function _line_search_tracking(w::Matrix{Float64}, d::Matrix{Float64},
                               F0::Float64, α0::Float64, ctrl)
    α    = α0
    c1   = 1e-4
    max_halvings = 20

    for halving in 1:max_halvings
        w_new = clamp.(w .+ α .* d, ctrl.l_bound, ctrl.u_bound)
        F_new, _ = grape_tracking_kernel(w_new, ctrl)
        F_new >= F0 - c1 * abs(α) && return α, w_new, F_new, halving + 1
        α *= 0.5
    end

    w_new = clamp.(w .+ α .* d, ctrl.l_bound, ctrl.u_bound)
    F_new, _ = grape_tracking_kernel(w_new, ctrl)
    return α, F_new > F0 ? w_new : copy(w), max(F_new, F0), max_halvings + 1
end

# ─── Cached f / grad! closures ────────────────────────────────────────────────
# Returns a minimisation objective f(θ_flat) = -fidelity and an in-place
# gradient grad!(g_out, θ_flat) = -∇fidelity. Both share a one-step cache so
# that gradient-based optimizers calling f and grad! at the same point incur
# only one kernel evaluation.

_get_parameterization(ctrl) =
    hasproperty(ctrl, :parameterization) ? ctrl.parameterization : PiecewiseConstant()

# Compute initial θ vector and bounds for a given parameterization.
# Returns (θ0::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}).
# For PiecewiseConstant: θ = vec(w), bounds = [l_bound, u_bound] uniformly.
# For PhaseOnlyParam: phase rows are unbounded (±Inf), free rows keep waveform bounds.
# For other diagonal types: θ = from_waveform(w), bounds inferred per-row by inverse map.
function _param_θ0_and_bounds(p::AbstractControlParameterization,
                              guess::Matrix{Float64},
                              l_bound::Float64, u_bound::Float64)
    n_ctrl, n_t = size(guess)
    n = n_ctrl * n_t
    if p isa PiecewiseConstant
        θ0 = vec(clamp.(guess, l_bound, u_bound))
        return θ0, fill(l_bound, n), fill(u_bound, n)
    end
    θ0 = from_waveform(clamp.(guess, l_bound, u_bound), p)
    if p isa PhaseOnlyParam
        n_p = length(p.phase_pairs)
        n_free = n_ctrl - 2 * n_p
        n_θ = (n_p + n_free) * n_t
        lb = Vector{Float64}(undef, n_θ)
        ub = Vector{Float64}(undef, n_θ)
        @inbounds for k in 1:n_t
            base = (k - 1) * (n_p + n_free)
            for i in 1:n_p
                lb[base + i] = -Inf
                ub[base + i] = +Inf
            end
            for j in 1:n_free
                lb[base + n_p + j] = l_bound
                ub[base + n_p + j] = u_bound
            end
        end
        return θ0, lb, ub
    end
    return θ0, fill(-Inf, length(θ0)), fill(+Inf, length(θ0))
end

function _make_fg_closures(ctrl, n_ctrl::Int, n_t::Int,
                            param::AbstractControlParameterization = PiecewiseConstant())
    n_θ = param isa PiecewiseConstant ? n_ctrl * n_t : begin
        # Derive θ length by trial: from_waveform on a zero matrix
        length(from_waveform(zeros(n_ctrl, n_t), param))
    end
    last_θ = fill(NaN, n_θ)
    last_F = Ref(0.0)
    last_g_θ = zeros(n_θ)
    g_w_buf  = zeros(n_ctrl, n_t)
    w_buf    = zeros(n_ctrl, n_t)

    function _refresh!(θ_flat::AbstractVector{<:Real})
        if θ_flat != last_θ
            if param isa PiecewiseConstant
                w_buf .= clamp.(reshape(Float64.(θ_flat), n_ctrl, n_t),
                                ctrl.l_bound, ctrl.u_bound)
            else
                w_buf .= to_waveform(θ_flat, param, n_ctrl, n_t)
            end
            F, G_w = _mr_kernel(w_buf, ctrl)
            last_F[] = F
            if param isa PiecewiseConstant
                last_g_θ .= vec(G_w)
            else
                g_w_buf .= G_w
                apply_jacobian_transpose!(last_g_θ, g_w_buf, θ_flat, param, n_ctrl, n_t)
            end
            last_θ .= θ_flat
        end
    end

    f(θ_flat)            = (_refresh!(θ_flat); -last_F[])
    grad!(g_out, θ_flat) = (_refresh!(θ_flat); g_out .= -last_g_θ)
    return f, grad!
end

# Stateless forward-only objective for derivative-free population methods
# (:cmaes, :pscmaes, :nelder_mead, :pso, :de). These evaluate many candidates
# in parallel via Threads.@threads; a shared memoization cache would race.
function _make_f_only_closure(ctrl, n_ctrl::Int, n_t::Int,
                               param::AbstractControlParameterization = PiecewiseConstant())
    return function (θ_flat::AbstractVector{<:Real})
        w = if param isa PiecewiseConstant
            clamp.(reshape(Float64.(θ_flat), n_ctrl, n_t),
                   ctrl.l_bound, ctrl.u_bound)
        else
            to_waveform(θ_flat, param, n_ctrl, n_t)
        end
        F, _ = _mr_kernel(w, ctrl)
        return -F
    end
end

# ─── Generic optimizer driver ──────────────────────────────────────────────────

function _optimcon_generic(ctrl, guess::Matrix{Float64};
                            method::Symbol = ctrl.method)::OptimizationResult
    t_start = time()
    n_ctrl, n_t = size(guess)
    param = _get_parameterization(ctrl)
    θ0, lb, ub = _param_θ0_and_bounds(param, guess, ctrl.l_bound, ctrl.u_bound)
    n_θ = length(θ0)

    # Heuristic max_evals: enough for the population-based methods to explore
    # the landscape meaningfully within max_iter "generations".
    max_ev_pop   = ctrl.max_iter * max(50, n_θ ÷ 2)
    max_ev_large = ctrl.max_iter * max(20, n_θ ÷ 4)

    f, grad! = if method in (:cmaes, :pscmaes, :nelder_mead, :pso, :de)
        (_make_f_only_closure(ctrl, n_ctrl, n_t, param),
         (_, __) -> error("gradient not used by method :$method"))
    else
        _make_fg_closures(ctrl, n_ctrl, n_t, param)
    end

    if ctrl.verbose
        label = String(method)
        if param isa PiecewiseConstant
            @printf("[optimcon/%s] Problem: %d params (n_ctrl=%d × n_t=%d)\n",
                    label, n_θ, n_ctrl, n_t)
        else
            param_label = nameof(typeof(param))
            @printf("[optimcon/%s] Problem: %d θ-params via %s (waveform: n_ctrl=%d × n_t=%d)\n",
                    label, n_θ, param_label, n_ctrl, n_t)
        end
        method in (:cmaes, :pscmaes, :nelder_mead, :pso, :de) &&
            @printf("[optimcon/%s] Derivative-free — gradient not used.\n", label)
    end

    θ_best, neg_f_best, stats =
        if method == :cmaes
            cmaes_optimize(f, θ0;
                lower     = lb, upper    = ub,
                max_iters = ctrl.max_iter,
                max_evals = max_ev_pop)
        elseif method == :pscmaes
            pscmaes_optimize(f, θ0;
                lower     = lb, upper    = ub,
                max_iters = ctrl.max_iter,
                max_evals = max_ev_pop)
        elseif method == :nelder_mead
            nelder_mead_optimize(f, θ0;
                lower     = lb, upper    = ub,
                max_iters = ctrl.max_iter,
                max_evals = max_ev_pop,
                step      = 0.05 * (ctrl.u_bound - ctrl.l_bound))
        elseif method == :pso
            pso_optimize(f, θ0;
                lower     = lb, upper    = ub,
                max_iters = ctrl.max_iter,
                max_evals = max_ev_large)
        elseif method == :de
            de_optimize(f, θ0;
                lower     = lb, upper    = ub,
                max_iters = ctrl.max_iter,
                max_evals = max_ev_large)
        elseif method == :lbfgsb
            cb = hasproperty(ctrl, :callback) ? ctrl.callback : nothing
            let r = grape_lbfgsb_optimize(f, grad!, θ0;
                        lower          = lb, upper    = ub,
                        memory         = ctrl.lbfgs_memory,
                        max_iter       = ctrl.max_iter,
                        verbose        = ctrl.verbose,
                        print_interval = ctrl.print_interval,
                        callback       = cb)
                (vec(r.controls), -r.fidelity,
                 (evals    = r.n_fidelity_evaluations + r.n_gradient_evaluations,
                  iters    = r.n_iterations,
                  converged= r.converged,
                  history  = [-x for x in r.fidelity_history]))
            end
        elseif method == :cg
            let r = grape_cg_optimize(f, grad!, θ0;
                        lower          = lb, upper    = ub,
                        max_iter       = ctrl.max_iter,
                        verbose        = ctrl.verbose,
                        print_interval = ctrl.print_interval)
                (vec(r.controls), -r.fidelity,
                 (evals    = r.n_fidelity_evaluations + r.n_gradient_evaluations,
                  iters    = r.n_iterations,
                  converged= r.converged,
                  history  = [-x for x in r.fidelity_history]))
            end
        else
            throw(ArgumentError(
                "Unknown method ':$method'. Supported: " *
                ":lbfgs, :grape, :lbfgsb, :cg, :cmaes, :pscmaes, :nelder_mead, :pso, :de"))
        end

    best_F    = -neg_f_best
    best_w    = if param isa PiecewiseConstant
        reshape(clamp.(θ_best, ctrl.l_bound, ctrl.u_bound), n_ctrl, n_t)
    else
        to_waveform(θ_best, param, n_ctrl, n_t)
    end
    t_elapsed = time() - t_start

    # Convert stored minimisation history back to fidelity
    fid_hist = if hasproperty(stats, :history) && !isempty(stats.history)
        [-x for x in stats.history]
    else
        [best_F]
    end
    grad_hist = zeros(length(fid_hist))

    converged = stats.converged
    reason    = converged ? "optimizer converged" : "max iterations/evaluations reached"

    if ctrl.verbose
        label = String(method)
        @printf("[optimcon/%s] Done: F=%.8f  evals=%d  time=%.2f s  %s\n",
                label, best_F, stats.evals, t_elapsed,
                converged ? "✓ " * reason : reason)
    end

    n_ens = if ctrl isa LindbladMRControl
        length(ctrl._L_drifts) * length(ctrl.pwr_levels) * length(ctrl._sigma_init)
    else
        length(ctrl.drifts) * length(ctrl.pwr_levels) * length(ctrl.rho_init)
    end

    return OptimizationResult(
        best_w, best_F,
        fid_hist, grad_hist,
        stats.iters,
        converged, reason, t_elapsed,
        stats.evals, 0,
        Dict{String,Any}(
            "algorithm"  => "MR " * uppercase(String(method)),
            "n_ensemble" => n_ens,
            "n_ctrl"     => n_ctrl,
            "n_t"        => n_t,
        )
    )
end  # end of optimcon function

# New optimcon overloads are in Application/MR/OptControlExtensions.jl
# (loaded after Types/HeteronuclearSystem.jl and Physics/MRPhysics.jl)
