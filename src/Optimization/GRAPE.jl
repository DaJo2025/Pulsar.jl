"""
    GRAPE.jl

GRadient Ascent Pulse Engineering (GRAPE) optimizer for Pulsar.

Implements the canonical GRAPE algorithm (Khaneja et al., 2005) for optimizing
piecewise-constant control sequences that steer a quantum system toward a target
state or unitary operation.

The algorithm iterates:
  1. Compute forward propagators P[k] and backward propagators Q[k].
  2. Evaluate fidelity F and analytical gradient ∂F/∂u_j[k].
  3. Update controls: u_j[k] += α * ∂F/∂u_j[k].
  4. Adapt step size α based on fidelity history.
  5. Check convergence.

Reference:
  Khaneja et al., "Optimal control of coupled spin dynamics", J. Magn. Reson.
  172 (2005) 296–305. DOI: 10.1016/j.jmr.2004.11.004
"""

using LinearAlgebra
using Printf

# ============================================================================
# Configuration types
# ============================================================================

"""
    GRAPEConfig

Configuration for the GRAPE optimization algorithm.

# Fields
- `max_iter::Int` — maximum number of gradient-ascent iterations (default 1000)
- `convergence_tol::Float64` — fidelity change convergence threshold; declare
  convergence when |F_new - F_old| < tol (default 1e-8)
- `step_size::Float64` — initial gradient-ascent step size α (default 0.01)
- `adapt_step_size::Bool` — whether to adapt step size based on progress (default true)
- `min_step_size::Float64` — minimum allowed step size (default 1e-10)
- `max_step_size::Float64` — maximum allowed step size (default 10.0)
- `gradient_norm_tol::Float64` — gradient norm convergence threshold (default 1e-6)
- `verbose::Bool` — print progress every `print_interval` iterations (default false)
- `print_interval::Int` — print frequency in iterations (default 100)
- `callback::Union{Function,Nothing}` — optional callback invoked at each iteration
  with signature `callback(iter, fidelity, gradient_norm, controls)` (default nothing)

# Example
```julia
config = GRAPEConfig(
    max_iter          = 2000,
    convergence_tol   = 1e-10,
    step_size         = 0.005,
    adapt_step_size   = true,
    verbose           = true,
    print_interval    = 50
)
```
"""
struct GRAPEConfig
    max_iter::Int
    convergence_tol::Float64
    step_size::Float64
    adapt_step_size::Bool
    min_step_size::Float64
    max_step_size::Float64
    gradient_norm_tol::Float64
    verbose::Bool
    print_interval::Int
    callback::Union{Function, Nothing}
    use_threads::Bool
    backend::Symbol
    parameterization::AbstractControlParameterization
end

"""
    GRAPEConfig(; kwargs...) -> GRAPEConfig

Construct a `GRAPEConfig` with keyword arguments and default values.

# Keyword Arguments
- `max_iter = 1000`
- `convergence_tol = 1e-8`
- `step_size = 0.01`
- `adapt_step_size = true`
- `min_step_size = 1e-10`
- `max_step_size = 10.0`
- `gradient_norm_tol = 1e-6`
- `verbose = false`
- `print_interval = 100`
- `callback = nothing`
"""
function GRAPEConfig(;
    max_iter::Int                        = 1000,
    convergence_tol::Float64             = 1e-8,
    step_size::Float64                   = 0.01,
    adapt_step_size::Bool                = true,
    min_step_size::Float64               = 1e-10,
    max_step_size::Float64               = 10.0,
    gradient_norm_tol::Float64           = 1e-6,
    verbose::Bool                        = false,
    print_interval::Int                  = 100,
    callback::Union{Function, Nothing}   = nothing,
    use_threads::Bool                    = true,
    backend::Symbol                      = :auto,
    parameterization::AbstractControlParameterization = PiecewiseConstant(),
)::GRAPEConfig
    max_iter > 0           || throw(ArgumentError("max_iter must be positive"))
    convergence_tol > 0    || throw(ArgumentError("convergence_tol must be positive"))
    step_size > 0          || throw(ArgumentError("step_size must be positive"))
    min_step_size > 0      || throw(ArgumentError("min_step_size must be positive"))
    max_step_size >= step_size ||
        throw(ArgumentError("max_step_size must be >= step_size"))
    min_step_size <= step_size ||
        throw(ArgumentError("min_step_size must be <= step_size"))
    gradient_norm_tol > 0  || throw(ArgumentError("gradient_norm_tol must be positive"))
    print_interval > 0     || throw(ArgumentError("print_interval must be positive"))
    backend in (:auto, :cpu, :gpu, :hybrid) ||
        throw(ArgumentError("backend must be :auto, :cpu, :gpu, or :hybrid"))

    return GRAPEConfig(
        max_iter, convergence_tol, step_size, adapt_step_size,
        min_step_size, max_step_size, gradient_norm_tol,
        verbose, print_interval, callback, use_threads, backend,
        parameterization,
    )
end

# ============================================================================
# Result type
# ============================================================================

"""
    OptimizationResult

Container for the result of any Pulsar optimization run.

# Fields
- `controls::Matrix{Float64}` — optimal control amplitudes, shape `[n_controls × n_timesteps]`
- `fidelity::Float64` — best achieved fidelity ∈ [0, 1]
- `fidelity_history::Vector{Float64}` — fidelity at each iteration
- `gradient_norm_history::Vector{Float64}` — gradient Frobenius norm at each iteration
- `n_iterations::Int` — actual number of iterations performed
- `converged::Bool` — whether a convergence criterion was met
- `termination_reason::String` — human-readable termination reason
- `total_time::Float64` — wall-clock optimization time in seconds
- `n_fidelity_evaluations::Int` — total fidelity function evaluations
- `n_gradient_evaluations::Int` — total gradient evaluations
- `metadata::Dict{String,Any}` — algorithm-specific diagnostics

# Example
```julia
result = grape_optimize(sys, target, seq)
println("Fidelity: ", result.fidelity)
println("Converged: ", result.converged, " — ", result.termination_reason)
```
"""
struct OptimizationResult
    controls::Matrix{Float64}
    fidelity::Float64
    fidelity_history::Vector{Float64}
    gradient_norm_history::Vector{Float64}
    n_iterations::Int
    converged::Bool
    termination_reason::String
    total_time::Float64
    n_fidelity_evaluations::Int
    n_gradient_evaluations::Int
    metadata::Dict{String, Any}
end

function Base.show(io::IO, r::OptimizationResult)
    @printf(io, "OptimizationResult:\n")
    @printf(io, "  fidelity       = %.8f\n", r.fidelity)
    @printf(io, "  converged      = %s (%s)\n", r.converged, r.termination_reason)
    @printf(io, "  iterations     = %d\n", r.n_iterations)
    @printf(io, "  wall time      = %.3f s\n", r.total_time)
    @printf(io, "  ∇F evaluations = %d\n", r.n_gradient_evaluations)
end

# Back-compat: expose optional diagnostic histories through `getproperty`.
# When the optimiser has not recorded an entry under this name, return an
# empty vector so legacy tests that simply iterate the field do not error.
const _OPT_RESULT_LEGACY_HISTORIES = (
    :line_search_history,
    :hessian_history,
    :step_size_history,
    :search_direction_history,
    :hessian_eigenvalues_history,
)

const _OPT_RESULT_METADATA_FIELDS = (:dt, :n_controls, :n_timesteps,
                                      :step_size_final, :algorithm)

function Base.getproperty(r::OptimizationResult, name::Symbol)
    if name in _OPT_RESULT_LEGACY_HISTORIES
        md = getfield(r, :metadata)
        return get(md, String(name), Any[])
    end
    if name in _OPT_RESULT_METADATA_FIELDS
        md = getfield(r, :metadata)
        haskey(md, String(name)) && return md[String(name)]
    end
    return getfield(r, name)
end

function Base.propertynames(r::OptimizationResult, private::Bool=false)
    return (fieldnames(OptimizationResult)...,
            _OPT_RESULT_LEGACY_HISTORIES...,
            _OPT_RESULT_METADATA_FIELDS...)
end

# ============================================================================
# Helper: fidelity evaluation from controls matrix
# ============================================================================

"""
    _grape_fidelity(system, target, controls_mat, dt) -> Float64

Compute fidelity for a `controls_mat` matrix `[n_controls × n_timesteps]`.
Internal helper that avoids re-constructing a `ControlSequence` at every call.
"""
function _grape_fidelity(system::AbstractQuantumSystem,
                          target::QuantumTarget,
                          controls_mat::Matrix{Float64},
                          dt::Float64)::Float64
    n_t = size(controls_mat, 2)
    total_time = dt * n_t
    seq = ControlSequence(controls_mat, dt, total_time, n_t)
    H_total = build_total_hamiltonian(system, seq)
    U_steps = compute_propagators(H_total, dt)
    U_total = compute_total_propagator(U_steps)
    return compute_fidelity(U_total, target)
end

# ============================================================================
# Core per-iteration step
# ============================================================================

"""
    grape_step!(controls::Matrix{Float64}, gradient::Matrix{Float64},
                step_size::Float64)

Apply a single GRAPE gradient-ascent update in-place.

Updates `controls` by adding `step_size * gradient` element-wise.

# Arguments
- `controls`   — current control amplitudes `[n_controls × n_timesteps]`, modified in-place
- `gradient`   — GRAPE gradient `∂F/∂u_j[k]`, same shape as `controls`
- `step_size`  — scalar step size α > 0

# Notes
The update is u_j[k] += α * ∂F/∂u_j[k], which is gradient *ascent* on fidelity.

# Example
```julia
grape_step!(controls, gradient, 0.01)
```
"""
function grape_step!(controls::Matrix{Float64},
                     gradient::Matrix{Float64},
                     step_size::Float64)
    @inbounds for i in eachindex(controls)
        controls[i] += step_size * gradient[i]
    end
    return controls
end

# ============================================================================
# Adaptive step size
# ============================================================================

"""
    adapt_grape_step_size(fidelity_history::Vector{Float64},
                           current_step::Float64;
                           min_step::Float64 = 1e-10,
                           max_step::Float64 = 10.0,
                           growth_factor::Float64 = 1.05,
                           shrink_factor::Float64 = 0.5) -> Float64

Compute an adapted GRAPE step size based on recent fidelity history.

# Strategy
- If the fidelity improved in both of the last two iterations (monotone ascent):
  increase the step size by `growth_factor`.
- If the fidelity decreased (backtrack occurred or oscillation detected):
  decrease the step size by `shrink_factor`.
- Otherwise keep the step size unchanged.

Step size is clamped to `[min_step, max_step]`.

# Arguments
- `fidelity_history` — recorded fidelity values; must have at least 2 entries
- `current_step`     — step size at the current iteration
- `min_step`         — minimum allowed step size (default 1e-10)
- `max_step`         — maximum allowed step size (default 10.0)
- `growth_factor`    — multiplicative increase when improving (default 1.05)
- `shrink_factor`    — multiplicative decrease when oscillating (default 0.5)

# Returns
New step size clamped to `[min_step, max_step]`.

# Example
```julia
α_new = adapt_grape_step_size(fidelity_history, α)
```
"""
function adapt_grape_step_size(fidelity_history::Vector{Float64},
                                current_step::Float64;
                                min_step::Float64    = 1e-10,
                                max_step::Float64    = 10.0,
                                growth_factor::Float64  = 1.05,
                                shrink_factor::Float64  = 0.5)::Float64
    n = length(fidelity_history)
    if n < 2
        return current_step
    end

    delta_last = fidelity_history[n] - fidelity_history[n-1]

    new_step = if delta_last > 0.0
        # Fidelity improved — try a slightly larger step
        current_step * growth_factor
    elseif delta_last < -1e-14
        # Fidelity decreased — backtrack with a smaller step
        current_step * shrink_factor
    else
        # Flat — keep step unchanged
        current_step
    end

    return clamp(new_step, min_step, max_step)
end

# ============================================================================
# Convergence check
# ============================================================================

"""
    check_grape_convergence(fidelity_history::Vector{Float64},
                             gradient_norm::Float64,
                             config::GRAPEConfig) -> Tuple{Bool, String}

Check whether the GRAPE optimization has converged.

Convergence is declared when any of the following hold:

1. **Fidelity stall**: |F_new - F_old| < `config.convergence_tol` (checked over
   the last two iterations).
2. **Gradient norm**: ‖∇F‖_F < `config.gradient_norm_tol`.
3. **Perfect fidelity**: F ≥ 1 - 1e-12 (machine-precision convergence).

# Arguments
- `fidelity_history` — recorded fidelity values (most recent is last)
- `gradient_norm`    — Frobenius norm of the current gradient matrix
- `config`           — `GRAPEConfig` containing tolerance thresholds

# Returns
Tuple `(converged::Bool, reason::String)`.  When `converged == false`,
`reason` is an empty string.

# Example
```julia
conv, reason = check_grape_convergence(fidelity_history, gnorm, config)
if conv
    println("Converged: ", reason)
end
```
"""
function check_grape_convergence(fidelity_history::Vector{Float64},
                                  gradient_norm::Float64,
                                  config::GRAPEConfig)::Tuple{Bool, String}
    n = length(fidelity_history)
    if n == 0
        return (false, "")
    end

    current_fidelity = fidelity_history[n]

    # Check perfect convergence
    if current_fidelity >= 1.0 - 1e-12
        return (true, "perfect fidelity (F ≥ 1 - 1e-12)")
    end

    # Check gradient norm
    if gradient_norm < config.gradient_norm_tol
        return (true, "gradient norm $(gradient_norm) < tol $(config.gradient_norm_tol)")
    end

    # Check fidelity stall (require n >= 3 because the first loop pass
    # re-computes F on the still-initial controls, so history[1:2] are
    # always identical and would trigger spurious early convergence.)
    if n >= 3
        delta_F = abs(fidelity_history[n] - fidelity_history[n-1])
        if delta_F < config.convergence_tol
            return (true, "fidelity change $(delta_F) < tol $(config.convergence_tol)")
        end
    end

    return (false, "")
end

# ============================================================================
# Main GRAPE optimizer
# ============================================================================

"""
    grape_optimize(system::AbstractQuantumSystem,
                   target::QuantumTarget,
                   controls_init::ControlSequence;
                   config::GRAPEConfig = GRAPEConfig()) -> OptimizationResult

Run the GRAPE (GRadient Ascent Pulse Engineering) algorithm to find a
piecewise-constant control sequence that maximizes fidelity with `target`.

# Arguments
- `system`        — quantum system defining H_drift and H_controls
- `target`        — optimization target (unitary gate or state transfer)
- `controls_init` — initial control sequence; the optimizer starts from these
  amplitudes and returns a `ControlSequence` with the same `dt` and `n_timesteps`
- `config`        — algorithm configuration (default `GRAPEConfig()`)

# Returns
An [`OptimizationResult`](@ref) containing:
- `controls` — optimized control amplitudes `[n_controls × n_timesteps]`
- `fidelity` — best fidelity achieved
- `fidelity_history`, `gradient_norm_history` — convergence traces
- `converged`, `termination_reason` — convergence status
- `total_time`, `n_fidelity_evaluations`, `n_gradient_evaluations` — diagnostics

# Algorithm
1. Copy initial controls into a working matrix.
2. For each iteration:
   a. Build H_total[k] = H_drift + Σ_j u_j[k] * H_controls[j].
   b. Compute step propagators U[k] = exp(-i H[k] dt).
   c. Accumulate forward propagators P[k] and backward propagators Q[k].
   d. Compute total propagator U_total = U[n]⋯U[1] and fidelity F.
   e. Compute GRAPE gradient G = ∂F/∂u_j[k] via forward/backward pass.
   f. Update: u_j[k] += α * G[j,k].
   g. Adapt step size α based on fidelity history (if enabled).
   h. Check convergence criteria.
3. Return the best controls found (highest fidelity).

# Convergence Criteria
- |F_new - F_old| < `config.convergence_tol`
- ‖∇F‖_F < `config.gradient_norm_tol`
- Iteration count reaches `config.max_iter`

# Example
```julia
sys    = qubit_system(1, 0.5*pauli_z(), [0.5*pauli_x(), 0.5*pauli_y()])
target = unitary_target(ComplexF64[0 1; 1 0])   # X gate
seq    = random_controls(sys, π, 50; amplitude=1.0)

result = grape_optimize(sys, target, seq;
    config = GRAPEConfig(max_iter=500, step_size=0.05, verbose=true))

println("Fidelity: ", result.fidelity)
```

# Notes
- Step size adaptation uses a simple monotone scheme; for more sophisticated
  line search see `AdaptiveStepSize.jl`.
- The gradient is the analytical GRAPE gradient from `compute_grape_gradient`.
- Callback signature: `callback(iter::Int, F::Float64, gnorm::Float64,
  controls::Matrix{Float64}) -> nothing`.
- `ensemble = ens` (an [`EnsembleAveraging`](@ref)) optimises the ensemble
  average `ensemble_averaging(ens, system, controls, target)` instead of the
  single-system fidelity — offsets, couplings, B1, drive phase, timing, any
  added term, under any of its aggregators. When the ensemble's probe pairs are
  the targets, call `grape_optimize(system, ens, controls)`.
"""
function grape_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::GRAPEConfig = GRAPEConfig(),
    ensemble::Union{Nothing,EnsembleAveraging} = nothing,
    fidelity_fn::Union{Nothing,Function}              = nothing,
    gradient_fn::Union{Nothing,Function}              = nothing,
    fidelity_and_gradient_fn::Union{Nothing,Function} = nothing,
    penalty_fns::Vector{Function}      = Function[],
    penalty_grad_fns::Vector{Function} = Function[],
)::OptimizationResult

    # An ensemble average is supplied to the generic hook driver as its fidelity
    # hooks.
    if ensemble !== nothing
        (fidelity_fn === nothing && gradient_fn === nothing &&
         fidelity_and_gradient_fn === nothing) || throw(ArgumentError(
            "grape_optimize: `ensemble` defines the fidelity; do not also pass " *
            "fidelity_fn / gradient_fn / fidelity_and_gradient_fn"))
        ens_target = if ensemble.probes === nothing
            target.type in ("unitary", "state") || throw(ArgumentError(
                "grape_optimize: an ensemble without probe pairs needs a unitary or state target"))
            target
        else
            target.type == "none" || throw(ArgumentError(
                "grape_optimize: the ensemble's probe pairs are its targets; " *
                "call grape_optimize(system, ensemble, controls) without a target"))
            nothing
        end
        fidelity_fn = (s, c, _) -> ensemble_averaging(ensemble, s, c, ens_target; gradient = false)
        gradient_fn = (s, c, _) -> ensemble_averaging(ensemble, s, c, ens_target)[2]
        fidelity_and_gradient_fn = (s, c, _) -> ensemble_averaging(ensemble, s, c, ens_target)
    end

    # Physics-hook path: when any custom hook/penalty is supplied, dispatch to the
    # generic hook driver.  This typed method is MORE specific than the untyped
    # hook overload, so without this delegation those kwargs would be rejected with
    # a MethodError for every concrete AbstractQuantumSystem (MAS/EPR/DNP/Bloch) —
    # the historical grape_optimize "dispatch overlap" bug.
    if fidelity_fn !== nothing || gradient_fn !== nothing ||
       fidelity_and_gradient_fn !== nothing ||
       !isempty(penalty_fns) || !isempty(penalty_grad_fns)
        return _grape_optimize_hooks(system, target, controls_init;
            fidelity_fn              = fidelity_fn,
            gradient_fn              = gradient_fn,
            fidelity_and_gradient_fn = fidelity_and_gradient_fn,
            penalty_fns              = penalty_fns,
            penalty_grad_fns         = penalty_grad_fns,
            config                   = config)
    end

    if !(config.parameterization isa PiecewiseConstant)
        return _grape_optimize_param(system, target, controls_init, config)
    end

    t_start = time()

    # ---- unpack dimensions --------------------------------------------------
    dt         = controls_init.dt
    n_t        = controls_init.n_timesteps
    n_c        = system.n_controls
    total_time = controls_init.total_time

    if size(controls_init.controls) != (n_c, n_t)
        throw(DimensionMismatch(
            "controls_init.controls size $(size(controls_init.controls)) does not " *
            "match ($(n_c), $(n_t)) expected from system and n_timesteps"))
    end

    # ---- working state -------------------------------------------------------
    controls      = copy(controls_init.controls)   # n_c × n_t, mutable working copy
    best_controls = copy(controls)

    fidelity_history      = Float64[]
    gradient_norm_history = Float64[]
    n_fidelity_evals      = 0
    n_gradient_evals      = 0

    current_step = config.step_size
    converged    = false
    reason       = "maximum iterations reached"
    n_iter       = 0   # count of gradient-ascent steps actually taken

    # ---- pre-allocated workspace (mirrors GRAPE.jl `GrapeWrk`) ----------------
    dim     = system.dim
    H_total = Array{ComplexF64,3}(undef, n_t, dim, dim)
    U_steps = Array{ComplexF64,3}(undef, n_t, dim, dim)
    P_buf   = Array{ComplexF64,3}(undef, n_t + 1, dim, dim)
    Q_buf   = Array{ComplexF64,3}(undef, n_t + 1, dim, dim)
    G       = zeros(Float64, n_c, n_t)
    H_buf   = Matrix{ComplexF64}(undef, dim, dim)
    H_tmp   = Matrix{ComplexF64}(undef, dim, dim)
    seq_iter = ControlSequence(controls, dt, total_time, n_t)

    # ---- initial fidelity ---------------------------------------------------
    build_total_hamiltonian!(H_total, system, seq_iter)
    compute_propagators!(U_steps, H_total, dt; H_buf=H_buf, tmp=H_tmp)
    compute_forward_propagators!(P_buf, U_steps)
    @views U_total = Matrix{ComplexF64}(P_buf[n_t + 1, :, :])
    F_curr = compute_fidelity(U_total, target)
    n_fidelity_evals += 1

    push!(fidelity_history, F_curr)
    best_fidelity = F_curr

    if config.verbose
        @printf("[GRAPE] Starting: F0 = %.8f, n_params = %d\n",
                F_curr, n_c * n_t)
    end

    # ---- main loop -----------------------------------------------------------
    for iter in 1:config.max_iter
        n_iter = iter

        # Re-bind the working ControlSequence to the current `controls` buffer.
        # (ControlSequence stores `controls` by reference; mutation in-place is fine
        #  but a fresh struct is needed only if dimensions change — they don't here.)
        # The mutable `controls` matrix is shared with `seq_iter.controls`.
        seq_iter = ControlSequence(controls, dt, total_time, n_t)

        # --- build total Hamiltonians and step propagators (in-place) --------
        build_total_hamiltonian!(H_total, system, seq_iter)
        compute_propagators!(U_steps, H_total, dt; H_buf=H_buf, tmp=H_tmp)

        # --- forward and backward propagators (in-place) ---------------------
        compute_forward_propagators!(P_buf, U_steps)
        compute_backward_propagators!(Q_buf, U_steps)
        @views copyto!(U_total, P_buf[n_t + 1, :, :])

        # --- fidelity --------------------------------------------------------
        F_curr = compute_fidelity(U_total, target)
        n_fidelity_evals += 1

        # --- GRAPE gradient (uses pre-computed P, Q, U_total) ---------------
        compute_grape_gradient_with!(G, system, target, P_buf, Q_buf, U_total, dt)
        n_gradient_evals += 1

        gnorm = norm(G)
        push!(fidelity_history, F_curr)
        push!(gradient_norm_history, gnorm)

        # --- track best solution ---------------------------------------------
        if F_curr > best_fidelity
            best_fidelity = F_curr
            best_controls .= controls
        end

        # --- adaptive step size (before update, based on history) -----------
        if config.adapt_step_size && length(fidelity_history) >= 2
            current_step = adapt_grape_step_size(
                fidelity_history, current_step;
                min_step = config.min_step_size,
                max_step = config.max_step_size
            )
        end

        # --- gradient ascent update -----------------------------------------
        grape_step!(controls, G, current_step)

        # --- logging ---------------------------------------------------------
        if config.verbose && (iter % config.print_interval == 0 || iter == 1)
            @printf("[GRAPE] iter=%5d  F=%.8f  |∇F|=%.3e  α=%.3e\n",
                    iter, F_curr, gnorm, current_step)
        end

        # --- user callback ---------------------------------------------------
        if config.callback !== nothing
            config.callback(iter, F_curr, gnorm, controls)
        end

        # --- convergence check -----------------------------------------------
        conv, conv_reason = check_grape_convergence(fidelity_history, gnorm, config)
        if conv
            converged = true
            reason    = conv_reason
            # Record the post-update fidelity in history
            F_post = _grape_fidelity(system, target, controls, dt)
            push!(fidelity_history, F_post)
            n_fidelity_evals += 1
            if F_post > best_fidelity
                best_fidelity = F_post
                best_controls .= controls
            end
            break
        end
    end

    # ---- final evaluation at best controls ----------------------------------
    F_final = _grape_fidelity(system, target, best_controls, dt)
    n_fidelity_evals += 1

    t_elapsed = time() - t_start

    if config.verbose
        @printf("[GRAPE] Done: F=%.8f  iters=%d  time=%.3f s  converged=%s\n",
                F_final, n_iter, t_elapsed, converged)
    end

    return OptimizationResult(
        best_controls,
        F_final,
        fidelity_history,
        gradient_norm_history,
        n_iter,
        converged,
        reason,
        t_elapsed,
        n_fidelity_evals,
        n_gradient_evals,
        Dict{String, Any}(
            "step_size_final" => current_step,
            "algorithm"       => "GRAPE",
            "n_controls"      => n_c,
            "n_timesteps"     => n_t,
            "dt"              => dt,
            "total_time"      => total_time,
        )
    )
end

# ============================================================================
# Convenience overloads
# ============================================================================

"""
    grape_optimize(system::AbstractQuantumSystem,
                   target::QuantumTarget,
                   controls_init::ControlSequence,
                   max_iter::Int;
                   step_size::Float64 = 0.01,
                   verbose::Bool = false) -> OptimizationResult

Convenience overload for `grape_optimize` with positional `max_iter` and
keyword step size and verbosity.

# Example
```julia
result = grape_optimize(sys, target, seq, 500; step_size=0.02, verbose=true)
```
"""
function grape_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence,
    max_iter::Int;
    step_size::Float64 = 0.01,
    verbose::Bool      = false
)::OptimizationResult
    config = GRAPEConfig(max_iter=max_iter, step_size=step_size, verbose=verbose)
    return grape_optimize(system, target, controls_init; config=config)
end

# ============================================================================
# Multi-start GRAPE
# ============================================================================

"""
    grape_multistart(system::AbstractQuantumSystem,
                     target::QuantumTarget,
                     n_starts::Int,
                     total_time::Float64,
                     n_timesteps::Int;
                     amplitude::Float64 = 1.0,
                     config::GRAPEConfig = GRAPEConfig(),
                     rng::AbstractRNG = Random.GLOBAL_RNG) -> OptimizationResult

Run GRAPE from `n_starts` random initial conditions and return the best result.

# Arguments
- `system`      — quantum system
- `target`      — optimization target
- `n_starts`    — number of random restarts
- `total_time`  — total pulse duration (seconds)
- `n_timesteps` — number of time slices
- `amplitude`   — half-range of random initial amplitudes (default 1.0)
- `config`      — GRAPE configuration applied to every start
- `rng`         — random number generator for reproducibility

# Returns
The `OptimizationResult` with the highest fidelity among all starts.

# Notes
Runs are sequential (no parallelism). For GPU-parallel multi-start see the
`HybridExecution` module.

# Example
```julia
result = grape_multistart(sys, target, 10, 1e-3, 100;
             amplitude = 2π * 500.0,
             config    = GRAPEConfig(max_iter=300))
```
"""
function grape_multistart(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    n_starts::Int,
    total_time::Float64,
    n_timesteps::Int;
    amplitude::Float64             = 1.0,
    config::GRAPEConfig            = GRAPEConfig(),
    rng::Random.AbstractRNG        = Random.GLOBAL_RNG
)::OptimizationResult

    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1"))

    best_result = nothing

    for s in 1:n_starts
        seq = random_controls(system, total_time, n_timesteps;
                              amplitude=amplitude, rng=rng)
        result = grape_optimize(system, target, seq; config=config)
        if best_result === nothing || result.fidelity > best_result.fidelity
            best_result = result
        end
        if config.verbose
            @printf("[GRAPE multi-start %d/%d] best F = %.8f\n",
                    s, n_starts, best_result.fidelity)
        end
    end

    return best_result
end

# ============================================================================
# GRAPE with momentum (Heavy-ball method)
# ============================================================================

"""
    grape_momentum_optimize(system::AbstractQuantumSystem,
                             target::QuantumTarget,
                             controls_init::ControlSequence;
                             config::GRAPEConfig = GRAPEConfig(),
                             momentum::Float64 = 0.9) -> OptimizationResult

Run GRAPE with Polyak heavy-ball momentum.

The update rule is:

    v_{k+1} = β * v_k + α * ∇F(u_k)
    u_{k+1} = u_k + v_{k+1}

where β is the momentum coefficient (typically 0.9) and α is the step size.
Momentum tends to accelerate convergence in flat or noisy fidelity landscapes.

# Arguments
- `system`, `target`, `controls_init`, `config` — as in `grape_optimize`
- `momentum` — momentum coefficient β ∈ [0, 1) (default 0.9)

# Returns
`OptimizationResult` (same structure as `grape_optimize`).

# Example
```julia
result = grape_momentum_optimize(sys, target, seq;
             config   = GRAPEConfig(max_iter=500, step_size=0.005),
             momentum = 0.9)
```
"""
function grape_momentum_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::GRAPEConfig = GRAPEConfig(),
    momentum::Float64   = 0.9
)::OptimizationResult

    0.0 <= momentum < 1.0 ||
        throw(ArgumentError("momentum must be in [0, 1), got $momentum"))

    t_start = time()

    dt         = controls_init.dt
    n_t        = controls_init.n_timesteps
    total_time = controls_init.total_time
    n_c        = system.n_controls

    controls      = copy(controls_init.controls)
    best_controls = copy(controls)
    velocity      = zeros(Float64, n_c, n_t)   # momentum buffer

    fidelity_history      = Float64[]
    gradient_norm_history = Float64[]
    n_fidelity_evals      = 0
    n_gradient_evals      = 0

    current_step  = config.step_size
    best_fidelity = -Inf
    converged     = false
    reason        = "maximum iterations reached"

    for iter in 1:config.max_iter
        seq_iter = ControlSequence(controls, dt, total_time, n_t)

        # Fidelity
        F_curr = _grape_fidelity(system, target, controls, dt)
        n_fidelity_evals += 1

        # Gradient
        G = compute_grape_gradient(system, seq_iter, target)
        n_gradient_evals += 1
        gnorm = norm(G)

        push!(fidelity_history, F_curr)
        push!(gradient_norm_history, gnorm)

        if F_curr > best_fidelity
            best_fidelity = F_curr
            best_controls .= controls
        end

        # Momentum update
        @inbounds for i in eachindex(velocity)
            velocity[i]  = momentum * velocity[i] + current_step * G[i]
            controls[i] += velocity[i]
        end

        if config.adapt_step_size && length(fidelity_history) >= 2
            current_step = adapt_grape_step_size(
                fidelity_history, current_step;
                min_step = config.min_step_size,
                max_step = config.max_step_size
            )
        end

        if config.verbose && (iter % config.print_interval == 0 || iter == 1)
            @printf("[GRAPE-mom] iter=%5d  F=%.8f  |∇F|=%.3e  α=%.3e\n",
                    iter, F_curr, gnorm, current_step)
        end

        conv, conv_reason = check_grape_convergence(fidelity_history, gnorm, config)
        if conv
            converged = true
            reason    = conv_reason
            break
        end
    end

    F_final = _grape_fidelity(system, target, best_controls, dt)
    n_fidelity_evals += 1
    t_elapsed = time() - t_start

    return OptimizationResult(
        best_controls,
        F_final,
        fidelity_history,
        gradient_norm_history,
        length(fidelity_history),
        converged,
        reason,
        t_elapsed,
        n_fidelity_evals,
        n_gradient_evals,
        Dict{String, Any}(
            "algorithm"       => "GRAPE-momentum",
            "momentum"        => momentum,
            "step_size_final" => current_step,
        )
    )
end

# ============================================================================
# Diagnostic utilities
# ============================================================================

"""
    grape_gradient_check(system::AbstractQuantumSystem,
                          target::QuantumTarget,
                          controls::ControlSequence;
                          eps::Float64 = 1e-5) -> NamedTuple

Compare the analytical GRAPE gradient with a central finite-difference
approximation for a random subset of (j, k) pairs.

Returns a named tuple:
- `max_error::Float64` — maximum absolute error over tested pairs
- `max_relative_error::Float64` — maximum relative error
- `passed::Bool` — true if max_error < 1e-4

Useful for verifying correctness of system Hamiltonians.

# Example
```julia
check = grape_gradient_check(sys, target, seq; eps=1e-6)
@assert check.passed "Gradient check failed: max error = \$(check.max_error)"
```
"""
function grape_gradient_check(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls::ControlSequence;
    eps::Float64 = 1e-5
)

    # Analytical gradient
    G_ana = compute_grape_gradient(system, controls, target)

    # Numerical gradient via central differences
    n_c, n_t = size(controls.controls)
    G_num = zeros(Float64, n_c, n_t)

    for j in 1:n_c
        for k in 1:n_t
            u_p = copy(controls.controls)
            u_m = copy(controls.controls)
            u_p[j, k] += eps
            u_m[j, k] -= eps

            F_p = _grape_fidelity(system, target, u_p, controls.dt)
            F_m = _grape_fidelity(system, target, u_m, controls.dt)

            G_num[j, k] = (F_p - F_m) / (2 * eps)
        end
    end

    diff     = abs.(G_ana .- G_num)
    max_err  = maximum(diff)
    scale    = max(1.0, maximum(abs.(G_num)))
    max_rel  = max_err / scale

    return (
        analytical_gradient  = G_ana,
        numerical_gradient   = G_num,
        max_error            = max_err,
        max_relative_error   = max_rel,
        passed               = max_err < 1e-4
    )
end

# ============================================================================
# Ensemble (broadband) L-BFGS-GRAPE
# ============================================================================

"""
    grape_optimize_ensemble(systems, target, controls_init;
                             config, lbfgs_memory, amplitude_limit)
                             -> OptimizationResult

Broadband GRAPE optimization over an ensemble of quantum systems using
L-BFGS with backtracking line search (Armijo sufficient-increase condition).

Unlike `grape_optimize`, which takes a single system, this function accepts
a **vector of systems** and maximises the ensemble-averaged gate fidelity:

    F_bb(u) = (1/N) ∑ᵢ F(u; sysᵢ)

The gradient is similarly ensemble-averaged:

    ∇F_bb = (1/N) ∑ᵢ ∇F(u; sysᵢ)

computed via the analytic GRAPE formula for each system.

L-BFGS replaces the scalar gradient-ascent step with a quasi-Newton
search direction that incorporates curvature information from the last
`lbfgs_memory` iterations, giving superlinear convergence near the
optimum. The search direction is normalised so that `config.step_size`
is the maximum control change per step as a fraction of `amplitude_limit`
(e.g. `step_size=0.05` with `amplitude_limit=10000` → up to 500 Hz/step).

# Arguments
- `systems`         — vector of quantum systems (one per ensemble member)
- `target`          — common `QuantumTarget` for all systems
- `controls_init`   — initial `ControlSequence`
- `config`          — `GRAPEConfig` controlling `max_iter`, `verbose`,
                      `print_interval`, `convergence_tol`, `gradient_norm_tol`,
                      `callback`, and `step_size`
- `lbfgs_memory`    — number of (s, y) curvature pairs stored (default 10)
- `amplitude_limit` — per-element amplitude clipping applied after each step;
                      pass `Inf` to disable (default `Inf`)

# Returns
`OptimizationResult` with the best broadband fidelity found.

# Example
```julia
using Pulsar

Iz = 0.5 .* ComplexF64[1 0; 0 -1];  Ix = 0.5 .* ComplexF64[0 1; 1 0]
systems = [quantum_system(2π * Δf .* Iz, [2π .* Ix])
           for Δf in range(-5000, 5000; length=21)]
target  = unitary_target(ComplexF64[0 -im; -im 0])   # Rx(π)
cs_init = random_controls(systems[11], 500e-6, 250; amplitude=1000.0)

cfg = GRAPEConfig(max_iter=300, step_size=0.05, verbose=true, print_interval=1)
result = grape_optimize_ensemble(systems, target, cs_init;
                                  config=cfg, amplitude_limit=10_000.0)
```
"""
function grape_optimize_ensemble(
    systems::Vector{<:AbstractQuantumSystem},
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::GRAPEConfig      = GRAPEConfig(),
    lbfgs_memory::Int        = 10,
    amplitude_limit::Float64 = Inf
)::OptimizationResult

    t_start  = time()
    n_sys    = length(systems)
    n_sys > 0 || throw(ArgumentError("systems must be non-empty"))
    lbfgs_memory > 0 || throw(ArgumentError("lbfgs_memory must be positive"))

    controls   = copy(controls_init.controls)
    dt         = controls_init.dt
    total_time = controls_init.total_time
    n_t        = controls_init.n_timesteps

    fidelity_history      = Float64[]
    gradient_norm_history = Float64[]
    n_fid_evals           = 0
    n_grad_evals          = 0

    best_controls = copy(controls)
    best_fidelity = 0.0

    # L-BFGS curvature storage (flat vectors over all control parameters)
    s_list = Vector{Vector{Float64}}()   # control steps Δu
    y_list = Vector{Vector{Float64}}()   # gradient differences Δg

    # ── Delegate ensemble evaluation to the generic wrapper ─────────
    # `build_ensemble_from_systems` produces closures whose (F, ∇F) output is
    # bit-for-bit identical to the hand-rolled `_ens_F`/`_ens_G` that lived
    # here previously (verified against pinned-seed broadband examples).
    # The outer L-BFGS / line-search / amplitude-clipping logic below is
    # preserved verbatim so `OptimizationResult`, logging format, and
    # user-facing behaviour are unchanged.
    _grape_ens_obj = build_ensemble_from_systems(systems, target, controls_init;
                                                  aggregator = :mean)

    function _ens_F(u::Matrix{Float64})::Float64
        return ensemble_value(_grape_ens_obj, vec(u))
    end

    function _ens_G(u::Matrix{Float64})::Matrix{Float64}
        _, g = ensemble_value_and_grad(_grape_ens_obj, vec(u))
        return reshape(g, size(u))
    end

    # ── L-BFGS two-loop recursion (Nocedal & Wright Algorithm 7.4) ──
    # Returns the ascent direction (same sign convention as gradient).
    function _lbfgs_dir(g_flat::Vector{Float64})::Vector{Float64}
        k = length(s_list)
        k == 0 && return copy(g_flat)   # first step: steepest ascent

        q  = copy(g_flat)
        αs = zeros(k)
        for i in k:-1:1
            ρ     = 1.0 / max(dot(y_list[i], s_list[i]), 1e-14)
            αs[i] = ρ * dot(s_list[i], q)
            q   .-= αs[i] .* y_list[i]
        end
        # Hessian scaling by most-recent curvature pair
        γ = dot(s_list[k], y_list[k]) / max(dot(y_list[k], y_list[k]), 1e-14)
        q .*= γ
        for i in 1:k
            ρ = 1.0 / max(dot(y_list[i], s_list[i]), 1e-14)
            β = ρ * dot(y_list[i], q)
            q .+= (αs[i] - β) .* s_list[i]
        end
        return q   # L-BFGS ascent direction
    end

    # ── Backtracking line search (sufficient increase) ──────────────
    # Accept step if F_new ≥ F0; otherwise halve α up to max_bt times.
    function _line_search(u::Matrix{Float64}, d_mat::Matrix{Float64},
                          F0::Float64, α0::Float64; max_bt::Int = 20)
        α     = α0
        n_ls  = 0
        for _ in 1:max_bt
            u_new = u .+ α .* d_mat
            isfinite(amplitude_limit) && clamp!(u_new, -amplitude_limit, amplitude_limit)
            F_new  = _ens_F(u_new)
            n_ls  += 1
            F_new >= F0 && return α, u_new, F_new, n_ls
            α *= 0.5
        end
        u_fin = u .+ α .* d_mat
        isfinite(amplitude_limit) && clamp!(u_fin, -amplitude_limit, amplitude_limit)
        F_fin = _ens_F(u_fin)
        return α, u_fin, F_fin, max_bt + 1
    end

    # ── Initial evaluation ──────────────────────────────────────────
    F_curr = _ens_F(controls)
    n_fid_evals += 1
    push!(fidelity_history, F_curr)
    best_fidelity = F_curr
    best_controls .= controls

    if config.verbose
        @printf("[Ensemble L-BFGS-GRAPE] Start: F_bb = %.8f  N_sys = %d\n",
                F_curr, n_sys)
    end

    converged = false
    reason    = "maximum iterations reached"

    for iter in 1:config.max_iter

        # ── Gradient ─────────────────────────────────────────────────
        G_mat  = _ens_G(controls)
        n_grad_evals += 1
        g_flat = vec(G_mat)
        gnorm  = norm(g_flat)
        push!(gradient_norm_history, gnorm)

        # ── L-BFGS direction, then normalise ─────────────────────────
        # Normalisation makes step_size interpretable regardless of
        # gradient magnitude (which scales as dt·‖H_ctrl‖, very small
        # for NMR-scale Hz controls with 2π·Ix operators).
        d_flat = _lbfgs_dir(g_flat)
        d_max  = maximum(abs, d_flat)
        if d_max < 1e-14
            converged = true
            reason    = "zero gradient direction"
            break
        end
        d_flat ./= d_max   # max element = 1

        d_mat  = reshape(d_flat, size(controls))

        # Initial step: step_size × amplitude_limit  (or step_size if Inf)
        α0 = isfinite(amplitude_limit) ? (config.step_size * amplitude_limit) : config.step_size

        # ── Line search ──────────────────────────────────────────────
        α, controls_new, F_new, n_ls = _line_search(controls, d_mat, F_curr, α0)
        n_fid_evals += n_ls

        # ── Update L-BFGS curvature history ──────────────────────────
        G_new  = _ens_G(controls_new)
        n_grad_evals += 1
        s_vec  = α .* d_flat
        y_vec  = vec(G_new) .- g_flat

        # Curvature condition: only store if y·s > 0 (positive definiteness)
        if dot(y_vec, s_vec) > 1e-14 * norm(s_vec)^2
            push!(s_list, s_vec)
            push!(y_list, y_vec)
            if length(s_list) > lbfgs_memory
                popfirst!(s_list)
                popfirst!(y_list)
            end
        end

        # ── Accept step ───────────────────────────────────────────────
        controls = controls_new
        F_curr   = F_new
        push!(fidelity_history, F_curr)

        if F_curr > best_fidelity
            best_fidelity = F_curr
            best_controls .= controls
        end

        # ── Logging ───────────────────────────────────────────────────
        if config.verbose && (iter % config.print_interval == 0 || iter == 1)
            @printf("[Ensemble L-BFGS-GRAPE] iter=%4d  F_bb=%.8f  α=%.3e  |∇F|=%.3e\n",
                    iter, F_curr, α, gnorm)
        end

        config.callback !== nothing && config.callback(iter, F_curr, gnorm, controls)

        # ── Convergence checks ────────────────────────────────────────
        if gnorm < config.gradient_norm_tol
            converged = true
            reason    = "gradient norm < $(config.gradient_norm_tol)"
            break
        end
        if F_curr >= 1.0 - 1e-8
            converged = true
            reason    = "near-perfect fidelity"
            break
        end
        if length(fidelity_history) >= 2 &&
           abs(fidelity_history[end] - fidelity_history[end-1]) < config.convergence_tol
            converged = true
            reason    = "fidelity change < $(config.convergence_tol)"
            break
        end
    end

    t_elapsed = time() - t_start

    if config.verbose
        @printf("[Ensemble L-BFGS-GRAPE] Done: F_bb=%.8f  iters=%d  time=%.2f s  %s\n",
                best_fidelity, length(fidelity_history), t_elapsed,
                converged ? reason : "max iters")
    end

    return OptimizationResult(
        best_controls,
        best_fidelity,
        fidelity_history,
        gradient_norm_history,
        length(fidelity_history),
        converged,
        reason,
        t_elapsed,
        n_fid_evals,
        n_grad_evals,
        Dict{String,Any}(
            "algorithm"    => "Ensemble L-BFGS-GRAPE",
            "n_systems"    => n_sys,
            "lbfgs_memory" => lbfgs_memory,
        )
    )
end

# ============================================================================
# Physics-hook GRAPE overload (Task 1)
# ============================================================================

"""
    grape_optimize(system, target, controls_init;
                   fidelity_fn, gradient_fn, fidelity_and_gradient_fn,
                   penalty_fns, penalty_grad_fns, config)

Physics-agnostic GRAPE. Pass custom fidelity/gradient functions to optimize
any physical system — Bloch, Lindblad, MAS, etc.

With default arguments behaviour is identical to the standard method.
"""
grape_optimize(system, target, controls_init::ControlSequence; kwargs...) =
    _grape_optimize_hooks(system, target, controls_init; kwargs...)

# Generic hook-based GRAPE driver.  Reached via the untyped `grape_optimize`
# overload above, and via the typed (AbstractQuantumSystem, QuantumTarget) method
# when custom hooks/penalties are supplied (see the delegation there).
function _grape_optimize_hooks(
    system,
    target,
    controls_init::ControlSequence;
    fidelity_fn::Union{Nothing,Function} = nothing,
    gradient_fn::Union{Nothing,Function} = nothing,
    fidelity_and_gradient_fn::Union{Nothing,Function} = nothing,
    penalty_fns::Vector{Function}   = Function[],
    penalty_grad_fns::Vector{Function} = Function[],
    config::GRAPEConfig = GRAPEConfig()
)::OptimizationResult

    fidelity_fn = fidelity_fn === nothing ? (s,c,t) -> begin
        H_total = build_total_hamiltonian(s, c)
        U_steps = compute_propagators(H_total, c.dt)
        U_total = compute_total_propagator(U_steps)
        compute_fidelity(U_total, t)
    end : fidelity_fn
    gradient_fn = gradient_fn === nothing ? (s,c,t) -> compute_grape_gradient(s,c,t) : gradient_fn

    if !(config.parameterization isa PiecewiseConstant)
        return _grape_optimize_param(system, target, controls_init, config;
            fidelity_fn      = fidelity_fn,
            gradient_fn      = gradient_fn,
            penalty_fns      = penalty_fns,
            penalty_grad_fns = penalty_grad_fns)
    end

    t_start = time()

    dt         = controls_init.dt
    n_t        = controls_init.n_timesteps
    total_time = controls_init.total_time
    n_c        = size(controls_init.controls, 1)

    controls      = copy(controls_init.controls)
    best_controls = copy(controls)

    fidelity_history      = Float64[]
    gradient_norm_history = Float64[]
    n_fidelity_evals      = 0
    n_gradient_evals      = 0

    current_step = config.step_size
    converged    = false
    reason       = "maximum iterations reached"
    n_iter       = 0   # count of gradient-ascent steps actually taken

    # ---- initial fidelity ---------------------------------------------------
    seq_curr = ControlSequence(controls, dt, total_time, n_t)
    if fidelity_and_gradient_fn !== nothing
        F_curr, _ = fidelity_and_gradient_fn(system, seq_curr, target)
    else
        F_curr = fidelity_fn(system, seq_curr, target)
    end
    for pf in penalty_fns; F_curr -= pf(seq_curr); end
    n_fidelity_evals += 1

    push!(fidelity_history, F_curr)
    best_fidelity = F_curr

    if config.verbose
        @printf("[GRAPE-hook] Starting: F0 = %.8f, n_params = %d\n",
                F_curr, n_c * n_t)
    end

    # ---- main loop -----------------------------------------------------------
    for iter in 1:config.max_iter
        n_iter = iter

        seq_iter = ControlSequence(controls, dt, total_time, n_t)

        # --- fidelity and gradient -------------------------------------------
        if fidelity_and_gradient_fn !== nothing
            F_curr, G = fidelity_and_gradient_fn(system, seq_iter, target)
        else
            F_curr = fidelity_fn(system, seq_iter, target)
            G      = gradient_fn(system, seq_iter, target)
        end
        n_fidelity_evals += 1
        n_gradient_evals += 1

        # --- penalties -------------------------------------------------------
        for pf in penalty_fns
            F_curr -= pf(seq_iter)
        end
        for pgf in penalty_grad_fns
            G .-= pgf(seq_iter)
        end

        gnorm = norm(G)
        push!(fidelity_history, F_curr)
        push!(gradient_norm_history, gnorm)

        # --- track best solution ---------------------------------------------
        if F_curr > best_fidelity
            best_fidelity = F_curr
            best_controls .= controls
        end

        # --- adaptive step size ----------------------------------------------
        if config.adapt_step_size && length(fidelity_history) >= 2
            current_step = adapt_grape_step_size(
                fidelity_history, current_step;
                min_step = config.min_step_size,
                max_step = config.max_step_size
            )
        end

        # --- gradient ascent update ------------------------------------------
        grape_step!(controls, G, current_step)

        # --- logging ---------------------------------------------------------
        if config.verbose && (iter % config.print_interval == 0 || iter == 1)
            @printf("[GRAPE-hook] iter=%5d  F=%.8f  |∇F|=%.3e  α=%.3e\n",
                    iter, F_curr, gnorm, current_step)
        end

        # --- user callback ---------------------------------------------------
        if config.callback !== nothing
            config.callback(iter, F_curr, gnorm, controls)
        end

        # --- convergence check -----------------------------------------------
        conv, conv_reason = check_grape_convergence(fidelity_history, gnorm, config)
        if conv
            converged = true
            reason    = conv_reason
            # Post-update fidelity
            seq_post = ControlSequence(controls, dt, total_time, n_t)
            if fidelity_and_gradient_fn !== nothing
                F_post, _ = fidelity_and_gradient_fn(system, seq_post, target)
            else
                F_post = fidelity_fn(system, seq_post, target)
            end
            for pf in penalty_fns; F_post -= pf(seq_post); end
            push!(fidelity_history, F_post)
            n_fidelity_evals += 1
            if F_post > best_fidelity
                best_fidelity = F_post
                best_controls .= controls
            end
            break
        end
    end

    # ---- final evaluation ---------------------------------------------------
    seq_final = ControlSequence(best_controls, dt, total_time, n_t)
    if fidelity_and_gradient_fn !== nothing
        F_final, _ = fidelity_and_gradient_fn(system, seq_final, target)
    else
        F_final = fidelity_fn(system, seq_final, target)
    end
    for pf in penalty_fns; F_final -= pf(seq_final); end
    n_fidelity_evals += 1

    t_elapsed = time() - t_start

    if config.verbose
        @printf("[GRAPE-hook] Done: F=%.8f  iters=%d  time=%.3f s  converged=%s\n",
                F_final, n_iter, t_elapsed, converged)
    end

    return OptimizationResult(
        best_controls,
        F_final,
        fidelity_history,
        gradient_norm_history,
        n_iter,
        converged,
        reason,
        t_elapsed,
        n_fidelity_evals,
        n_gradient_evals,
        Dict{String, Any}(
            "step_size_final" => current_step,
            "algorithm"       => "GRAPE-hook",
            "n_controls"      => n_c,
            "n_timesteps"     => n_t,
            "dt"              => dt,
            "total_time"      => total_time,
        )
    )
end

"""
    grape_optimize(system, ensemble::EnsembleAveraging, controls_init; kwargs...)

GRAPE on an ensemble whose probe pairs (`ensemble.probes`) are the targets — e.g.
`EnsembleAveraging(axes...; probes = state_probes(U_target; set = :axes))`.
Keywords as for the target form; see `ensemble` there.
"""
function grape_optimize(system::AbstractQuantumSystem, ensemble::EnsembleAveraging,
                        controls_init::ControlSequence; kwargs...)
    ensemble.probes === nothing && throw(ArgumentError(
        "grape_optimize(system, ensemble, controls) is for an ensemble with probe pairs; " *
        "otherwise pass the target: grape_optimize(system, target, controls; ensemble)"))
    return grape_optimize(system, QuantumTarget("none", nothing, nothing, nothing, 0),
                          controls_init; ensemble = ensemble, kwargs...)
end

"""
    grape_optimize(system, ::Nothing, controls_init::ControlSequence; kwargs...)

Convenience overload for DNP/MRI where target is implicit in fidelity_fn.
"""
function grape_optimize(system, ::Nothing, controls_init::ControlSequence; kwargs...)
    dummy_target = QuantumTarget("none", nothing, nothing, nothing, 0)
    return grape_optimize(system, dummy_target, controls_init; kwargs...)
end

# ============================================================================
# Backward-compatible Matrix + dt overload
# ============================================================================
# Accepts a [n_controls × n_timesteps] amplitude matrix together with the time
# step `dt`, builds a `ControlSequence`, and forwards to the canonical method.
# Used by the legacy unit-test calling convention.

function grape_optimize(system::AbstractQuantumSystem,
                        target::QuantumTarget,
                        u_init::AbstractMatrix{<:Real},
                        dt::Real;
                        kwargs...)::OptimizationResult
    n_c, n_t = size(u_init)
    seq = ControlSequence(Matrix{Float64}(u_init), Float64(dt),
                          Float64(dt) * n_t, n_t)
    return grape_optimize(system, target, seq; kwargs...)
end

"""
    grape_gradient(system, target, u::AbstractMatrix, dt; parallel=false)

Backward-compatible wrapper: build a `ControlSequence` from `u`/`dt` and call
`compute_grape_gradient`. Returned shape: `[n_controls × n_timesteps]`.
"""
function grape_gradient(system::AbstractQuantumSystem,
                        target::QuantumTarget,
                        u::AbstractMatrix{<:Real},
                        dt::Real;
                        parallel::Bool = false)
    n_c, n_t = size(u)
    seq = ControlSequence(Matrix{Float64}(u), Float64(dt),
                          Float64(dt) * n_t, n_t)
    return compute_grape_gradient(system, seq, target)
end

"""
    evaluate_fidelity(system, target, u::AbstractMatrix, dt) -> Float64

Backward-compatible wrapper: build a `ControlSequence` from `u`/`dt`,
propagate, and return `compute_fidelity(U_total, target)`.
"""
function evaluate_fidelity(system::AbstractQuantumSystem,
                           target::QuantumTarget,
                           u::AbstractMatrix{<:Real},
                           dt::Real)::Float64
    n_c, n_t = size(u)
    seq = ControlSequence(Matrix{Float64}(u), Float64(dt),
                          Float64(dt) * n_t, n_t)
    H_total = build_total_hamiltonian(system, seq)
    U_steps = compute_propagators(H_total, seq.dt)
    U_total = compute_total_propagator(U_steps)
    return compute_fidelity(U_total, target)
end

# ============================================================================
# Parameterization-aware routing
# ============================================================================
# When config.parameterization is non-trivial (e.g. PhaseOnlyParam), the inner
# optimizer iterates on θ instead of the raw waveform. Closures wrap the
# physics kernel: θ → w via to_waveform, ∇_w F → ∇_θ F via apply_jacobian_transpose!.

function _grape_param_θ0_and_bounds(p::AbstractControlParameterization,
                                     guess::AbstractMatrix{<:Real})
    n_ctrl, n_t = size(guess)
    if p isa PiecewiseConstant
        return vec(Float64.(guess)), fill(-Inf, n_ctrl * n_t), fill(Inf, n_ctrl * n_t)
    end
    θ0 = from_waveform(Float64.(guess), p)
    if p isa PhaseOnlyParam
        n_p = length(p.phase_pairs)
        n_free = n_ctrl - 2 * n_p
        n_θ = (n_p + n_free) * n_t
        return θ0, fill(-Inf, n_θ), fill(Inf, n_θ)
    end
    return θ0, fill(-Inf, length(θ0)), fill(Inf, length(θ0))
end

function _grape_optimize_param(
        system::AbstractQuantumSystem,
        target::QuantumTarget,
        controls_init::ControlSequence,
        config::GRAPEConfig;
        fidelity_fn      = nothing,
        gradient_fn      = nothing,
        penalty_fns::Vector{<:Function}      = Function[],
        penalty_grad_fns::Vector{<:Function} = Function[])::OptimizationResult
    t_start = time()
    p = config.parameterization
    dt         = controls_init.dt
    n_t        = controls_init.n_timesteps
    total_time = controls_init.total_time
    n_c        = system.n_controls
    size(controls_init.controls) == (n_c, n_t) || throw(DimensionMismatch(
        "controls_init.controls $(size(controls_init.controls)) ≠ ($(n_c), $(n_t))"))

    θ0, lb, ub = _grape_param_θ0_and_bounds(p, controls_init.controls)

    # Default physics closures (mirror line-1242 defaults)
    _fid = fidelity_fn === nothing ?
        (s, c, t) -> begin
            H_total = build_total_hamiltonian(s, c)
            U_steps = compute_propagators(H_total, c.dt)
            U_total = compute_total_propagator(U_steps)
            compute_fidelity(U_total, t)
        end : fidelity_fn
    _grad = gradient_fn === nothing ?
        (s, c, t) -> compute_grape_gradient(s, c, t) : gradient_fn

    fidelity_history      = Float64[]
    gradient_norm_history = Float64[]
    n_fid_evals           = Ref(0)
    n_grad_evals          = Ref(0)

    function _w_from_θ(θ)
        if p isa PiecewiseConstant
            return reshape(Float64.(θ), n_c, n_t)
        else
            return to_waveform(θ, p, n_c, n_t)
        end
    end

    function f(θ)
        w = _w_from_θ(θ)
        seq = ControlSequence(w, dt, total_time, n_t)
        F_phys = _fid(system, seq, target)
        P = 0.0
        for pf in penalty_fns
            P += pf(system, seq, target)
        end
        n_fid_evals[] += 1
        push!(fidelity_history, F_phys)
        return -(F_phys - P)
    end

    function grad!(g_out, θ)
        w = _w_from_θ(θ)
        seq = ControlSequence(w, dt, total_time, n_t)
        G_w = _grad(system, seq, target)::Matrix{Float64}
        for pgf in penalty_grad_fns
            G_w .-= pgf(system, seq, target)
        end
        # ∇F (maximize) = G_w; ∇(-F) = -G_w
        G_w .*= -1
        if p isa PiecewiseConstant
            copyto!(g_out, vec(G_w))
        else
            apply_jacobian_transpose!(g_out, G_w, θ, p, n_c, n_t)
        end
        n_grad_evals[] += 1
        push!(gradient_norm_history, norm(g_out))
        return g_out
    end

    r = grape_lbfgsb_optimize(f, grad!, θ0;
            lower    = lb, upper = ub,
            memory   = 10,
            max_iter = config.max_iter,
            verbose  = config.verbose,
            print_interval = config.print_interval,
            callback = config.callback)

    θ_best = vec(r.controls)
    w_best = _w_from_θ(θ_best)
    F_best = r.fidelity   # grape_lbfgsb_optimize returns -minimised f; sign convention:
    # r.fidelity from the wrapper is -(neg f best) = (F_phys - P) at best
    # But our f returned -(F_phys - P), so r.fidelity = (F_phys - P).
    # For consistency, we report F_phys - P as the result fidelity.
    t_elapsed = time() - t_start

    return OptimizationResult(
        w_best, F_best,
        isempty(fidelity_history) ? Float64[F_best] : fidelity_history,
        isempty(gradient_norm_history) ? Float64[] : gradient_norm_history,
        r.n_iterations,
        r.converged, r.termination_reason, t_elapsed,
        n_fid_evals[], n_grad_evals[],
        Dict{String,Any}(
            "algorithm"        => "GRAPE-LBFGSB (parameterization=$(nameof(typeof(p))))",
            "n_controls"       => n_c,
            "n_timesteps"      => n_t,
            "n_theta"          => length(θ0),
            "dt"               => dt,
            "total_time"       => total_time,
        )
    )
end

"""
    finite_diff_gradient(system, target, u::AbstractMatrix, dt; ε=1e-6)

Backward-compatible alias for `finite_difference_gradient`. Wraps
`(u, dt)` into a `ControlSequence`. Accepts either `eps` or `ε` for the
finite-difference step.
"""
function finite_diff_gradient(system::AbstractQuantumSystem,
                              target::QuantumTarget,
                              u::AbstractMatrix{<:Real},
                              dt::Real;
                              ε::Float64 = 1e-6,
                              eps::Float64 = ε)
    n_c, n_t = size(u)
    seq = ControlSequence(Matrix{Float64}(u), Float64(dt),
                          Float64(dt) * n_t, n_t)
    return finite_difference_gradient(system, seq, target; eps = eps)
end
