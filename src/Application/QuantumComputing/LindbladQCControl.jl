# ============================================================
# Pulsar.jl — LindbladQCControl (Theme 6)
# ============================================================
#
# Open-system optimal-control context for the QC application layer,
# analogous to `LindbladMRControl <: AbstractMRControl` on the MR side.
#
# Today, the QC platform files (Superconducting / TrappedIon / NeutralAtom
# / SpinQubit / NVCenter) build a Liouville-space `QuantumSystem` via
# `_build_platform_qs(use_lindblad = true)` (see Common.jl) and let
# `grape_optimize` treat the result as Hilbert space — there is no
# explicit Lindblad gradient path.  This file establishes the missing
# dispatch handle.
#
# Phase-2 skeleton scope (this file)
# ──────────────────────────────────
#   • `LindbladQCControl` struct + keyword constructor: mirrors
#     `LindbladMRControl` shape with QC-domain field names, pre-computes
#     `_L_drifts`, `_L_controls`, `_sigma_init`, `_sigma_targ` exactly the
#     same way.
#   • `optimcon(ctrl::LindbladQCControl)` dispatch is **stubbed**:
#     it throws an `ErrorException` pointing callers to LindbladMRControl
#     (which is functionally identical for closed-form Lindblad) until
#     `grape_lindblad_kernel` is lifted from `Application/MR/` into
#     `src/Physics/LindbladGRAPE.jl` (Phase 2b).
#
# Once the kernel lift lands, this file gains the actual L-BFGS / GRAPE
# entry points without changing the public type signature.
# ============================================================

using LinearAlgebra

"""
    LindbladQCControl <: AbstractOptimizationContext

Open-system optimal-control context for the QC application layer.

Mirrors [`LindbladMRControl`](@ref) but with QC-domain field names and a
single primary `system` field (rather than separate `drifts` /
`operators` lists), matching the shape of [`QCControl`](@ref).

# Fields — user-facing
- `system`         :: QuantumSystem (drift + controls in Hilbert space)
- `target`         :: QuantumTarget
- `ctrl`           :: ControlSequence (initial waveform)
- `jump_ops`       :: Vector{Matrix{ComplexF64}} — Lindblad collapse Lₖ
- `decay_rates`    :: Vector{Float64}             — non-negative rates γₖ
- `method`         :: Symbol                       (`:lbfgs`, `:grape`, …)
- `grape_config`   :: GRAPEConfig
- `penalty_fns`    :: Vector{Function}
- `penalty_grad_fns` :: Vector{Function}
- `verbose`        :: Bool
- `metadata`       :: Dict{Symbol,Any}

# Fields — precomputed (set by the constructor)
- `_hilbert_dim`   :: Int                          (= `system.dim`)
- `_liouville_dim` :: Int                          (= `dim²`)
- `_L_drift`       :: Matrix{ComplexF64}           (single drift Liouvillian)
- `_L_controls`    :: Vector{Matrix{ComplexF64}}   (one per control operator)
- `_sigma_init`    :: Vector{ComplexF64}           (vec(ρ_init))
- `_sigma_targ`    :: Vector{ComplexF64}           (vec(ρ_targ))

# Status
The type and constructor are wired up.  `optimcon(ctrl::LindbladQCControl)`
throws an `ErrorException` until [`grape_lindblad_kernel`](@ref) is
lifted from the MR application layer to the Physics layer (Phase 2b).
For closed-form Lindblad workflows today, use [`LindbladMRControl`](@ref).
"""
struct LindbladQCControl <: AbstractOptimizationContext
    # ── User-facing problem definition ──────────────────────────────────
    system           :: QuantumSystem
    target           :: QuantumTarget
    ctrl             :: ControlSequence
    jump_ops         :: Vector{Matrix{ComplexF64}}
    decay_rates      :: Vector{Float64}
    # ── Optimisation settings ───────────────────────────────────────────
    method           :: Symbol
    grape_config     :: GRAPEConfig
    penalty_fns      :: Vector{Function}
    penalty_grad_fns :: Vector{Function}
    verbose          :: Bool
    metadata         :: Dict{Symbol,Any}
    # ── Precomputed Liouville-space objects ─────────────────────────────
    _hilbert_dim     :: Int
    _liouville_dim   :: Int
    _L_drift         :: Matrix{ComplexF64}
    _L_controls      :: Vector{Matrix{ComplexF64}}
    _sigma_init      :: Vector{ComplexF64}
    _sigma_targ      :: Vector{ComplexF64}
end

"""
    LindbladQCControl(sys, target, ctrl; jump_ops, decay_rates, kwargs...)
        -> LindbladQCControl

Keyword constructor.  Pre-computes the drift + control Liouvillians and
the vectorised initial / target density matrices using the existing
[`build_drift_liouvillian`](@ref) / [`build_control_liouvillian`](@ref)
machinery so the type is ready for the kernel-side Phase-2b wiring.

# Required arguments
- `sys`         — `AbstractQuantumSystem` (drift + controls)
- `target`      — `QuantumTarget` of `type == "state"` (state-transfer target).
                   Unitary-gate targets are not yet supported on the open-system
                   QC path; use [`LindbladMRControl`](@ref) directly.
- `ctrl`        — `ControlSequence` carrying the initial waveform
- `jump_ops`    — `Vector{Matrix{ComplexF64}}` of Lindblad collapse ops
- `decay_rates` — matching non-negative rates (rad/s)

# Keyword arguments
- `rho_init` — initial state.  Either a length-`N` pure-state ket or a
   length-`N²` `vec(ρ)`.  Defaults to the computational ground |0…0⟩.
- Same shape as [`QCControl`](@ref): `method`, `max_iter`, `convergence_tol`,
  `gradient_norm_tol`, `step_size`, `verbose`, `print_interval`,
  `penalty_fns`, `penalty_grad_fns`, `metadata`.

# Example
```julia
sys     = transmon_system(5.0e9, -200e6)
target  = state_target([0; 1] .+ 0im)        # |0⟩ → |1⟩
ctrl    = ControlSequence(0.01 .* randn(2, 100), 5e-9, 5e-7, 100)
T1, T2  = 50e-6, 30e-6
jumps, rates = mr_relaxation(sys, T1, T2)
ctx = LindbladQCControl(sys, target, ctrl; jump_ops = jumps, decay_rates = rates)
```
"""
function LindbladQCControl(
    sys              :: AbstractQuantumSystem,
    target           :: QuantumTarget,
    ctrl             :: ControlSequence;
    jump_ops         :: Vector{<:AbstractMatrix} = Matrix{ComplexF64}[],
    decay_rates      :: Vector{Float64}          = Float64[],
    rho_init                                     = nothing,
    method           :: Symbol           = :lbfgs,
    max_iter         :: Int              = 500,
    convergence_tol  :: Float64          = 1e-8,
    gradient_norm_tol :: Float64         = 1e-6,
    step_size        :: Float64          = 0.01,
    verbose          :: Bool             = false,
    print_interval   :: Int              = 100,
    penalty_fns      :: Vector{Function} = Function[],
    penalty_grad_fns :: Vector{Function} = Function[],
    metadata         :: Dict{Symbol,Any} = Dict{Symbol,Any}(),
)
    length(jump_ops) == length(decay_rates) ||
        throw(ArgumentError(
            "jump_ops and decay_rates must have the same length " *
            "(got $(length(jump_ops)) and $(length(decay_rates)))"))
    all(>=(0.0), decay_rates) ||
        throw(ArgumentError("decay_rates must be non-negative"))
    target.type == "state" ||
        throw(ArgumentError(
            "LindbladQCControl currently supports only state-transfer targets; " *
            "got target.type = \"$(target.type)\".  Use LindbladMRControl for " *
            "unitary-gate Liouville optimisation."))

    # Promote inputs to canonical types
    jumps_cf = Matrix{ComplexF64}[Matrix{ComplexF64}(L) for L in jump_ops]
    H_drift  = Matrix{ComplexF64}(sys.H_drift)
    H_ctrls  = Matrix{ComplexF64}[Matrix{ComplexF64}(H) for H in sys.H_controls]
    N        = size(H_drift, 1)

    # ── Precompute Liouvillians ────────────────────────────────────────
    L_drift    = build_drift_liouvillian(H_drift, jumps_cf, decay_rates)
    L_controls = [build_control_liouvillian(op) for op in H_ctrls]

    # ── Vectorise initial / target states ──────────────────────────────
    if rho_init === nothing
        ψ0      = zeros(ComplexF64, N); ψ0[1] = 1.0
        σ_init  = pure_state_to_vec_rho(ψ0)
    else
        σ_init  = _qc_vectorise(rho_init, N)
    end
    σ_targ = _qc_vectorise(target.target_state, N)

    grape_cfg = GRAPEConfig(
        max_iter          = max_iter,
        convergence_tol   = convergence_tol,
        gradient_norm_tol = gradient_norm_tol,
        step_size         = step_size,
        verbose           = verbose,
        print_interval    = print_interval,
    )

    qs_canonical = QuantumSystem(H_drift, H_ctrls, N, length(H_ctrls),
                                 Dict{String,Any}())

    return LindbladQCControl(
        qs_canonical,
        target, ctrl,
        jumps_cf, copy(decay_rates),
        method, grape_cfg,
        penalty_fns, penalty_grad_fns,
        verbose, metadata,
        N, N * N,
        L_drift, L_controls,
        σ_init, σ_targ,
    )
end

"""
    _qc_vectorise(ψ_or_ρ, N) -> Vector{ComplexF64}

Internal: accept a pure state of length `N` *or* a vectorised density
matrix of length `N²` and return the canonical vec(ρ) representation.
Mirrors the `_to_sigma` helper inside `LindbladMRControl`.
"""
function _qc_vectorise(ψ_or_ρ, N::Int)::Vector{ComplexF64}
    if ψ_or_ρ === nothing
        return ComplexF64[]
    end
    v = Vector{ComplexF64}(ψ_or_ρ)
    if length(v) == N
        return pure_state_to_vec_rho(v)
    elseif length(v) == N * N
        return v
    else
        throw(ArgumentError(
            "LindbladQCControl: state vector length $(length(v)) does not " *
            "match Hilbert dim $N (pure) or Liouville dim $(N*N) (vec ρ)."))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Theme 6b — Lindblad QC kernel via MR-shaped adapter
# ─────────────────────────────────────────────────────────────────────────────
#
# `grape_lindblad_kernel` lives in `Application/MR/GRAPELindblad.jl` and
# accesses fields with MR-domain names and shapes (`_L_drifts` plural,
# `pwr_levels`, `pulse_dt`, `_sigma_init`/`_sigma_targ` as `Vector{Vector}`,
# plus `backend`, `precision`, `fidelity`, and the MR penalty interface).
# Rather than fork the kernel or do a physical file move (deferred to
# Phase 2c), we present a `LindbladQCControl` to the kernel through a thin
# adapter struct whose `Base.getproperty` translates QC field shapes into
# the MR shape on the fly.  The kernel itself is unchanged.

struct _LindbladQCAdapter
    qc :: LindbladQCControl
end

function Base.getproperty(a::_LindbladQCAdapter, name::Symbol)
    qc = getfield(a, :qc)
    if name === :backend
        return :cpu
    elseif name === :precision
        return :f64
    elseif name === :fidelity
        return :dm_squared
    elseif name === :_L_drifts
        return [qc._L_drift]
    elseif name === :pwr_levels
        return [1.0]
    elseif name === :_sigma_init
        return [qc._sigma_init]
    elseif name === :_sigma_targ
        return [qc._sigma_targ]
    elseif name === :pulse_dt
        return fill(qc.ctrl.dt, qc.ctrl.n_steps)
    else
        return getfield(qc, name)
    end
end

# Override the MR-side `_apply_penalties!` for the QC adapter so that
# QC-shaped `(penalty_fns, penalty_grad_fns)` callables are honoured.
function _apply_penalties!(fidelity::Float64, grad::Matrix{Float64},
                            waveform, a::_LindbladQCAdapter)
    qc = getfield(a, :qc)
    for (pf, pg) in zip(qc.penalty_fns, qc.penalty_grad_fns)
        fidelity -= pf(waveform)
        grad     .-= pg(waveform)
    end
    return fidelity
end

"""
    _qc_kernel(w, ctx::LindbladQCControl) -> (F, grad)

Liouville-space kernel for the open-system QC context.  Wraps `ctx` in a
`_LindbladQCAdapter` so the existing `grape_lindblad_kernel` from
`Application/MR/GRAPELindblad.jl` can be reused without modification.
"""
_qc_kernel(w::Matrix{Float64}, ctx::LindbladQCControl) =
    grape_lindblad_kernel(w, _LindbladQCAdapter(ctx))

# ─────────────────────────────────────────────────────────────────────────────
# optimcon dispatch — closure-based, mirrors QCControl
# ─────────────────────────────────────────────────────────────────────────────

"""
    optimcon(ctx::LindbladQCControl) -> OptimizationResult

Run open-system optimal control on the QC application layer.  All gradient
and metaheuristic methods route through `_qc_kernel(w, ctx)` (which calls
`grape_lindblad_kernel` via the `_LindbladQCAdapter`); no per-method
Lindblad code lives here.

# Supported methods
| Symbol         | Algorithm                          |
|----------------|------------------------------------|
| `:lbfgs`       | L-BFGS-B (alias of `:lbfgsb`)      |
| `:lbfgsb`      | Bounded L-BFGS                     |
| `:cg`          | Polak-Ribière conjugate gradient   |
| `:grape`       | First-order gradient ascent        |
| `:cmaes`       | Covariance-matrix adaptation ES    |
| `:pso`         | Particle Swarm                     |
| `:nelder_mead` | Simplex search                     |
"""
function optimcon(ctx::LindbladQCControl)::OptimizationResult
    cfg     = ctx.grape_config
    nc, nt  = ctx.system.n_controls, ctx.ctrl.n_steps
    u0      = vec(copy(ctx.ctrl.controls))
    method  = ctx.method === :lbfgs ? :lbfgsb : ctx.method

    if method === :lbfgsb
        f_neg, grad_neg! = _qc_make_fg_closures(ctx, nc, nt)
        t_start = time()
        θ_best, f_best, stats = lbfgsb_optimize(f_neg, grad_neg!, u0;
            max_iter = cfg.max_iter,
            tol      = cfg.gradient_norm_tol,
            verbose  = ctx.verbose)
        reason = stats.converged ? "gradient norm < tol" : "maximum iterations reached"
        return OptimizationResult(
            reshape(θ_best, nc, nt), -f_best,
            Float64[], Float64[],
            stats.iters, stats.converged, reason,
            time() - t_start, stats.evals, stats.evals,
            Dict{String,Any}(
                "algorithm"     => "Lindblad QC L-BFGS-B",
                "hilbert_dim"   => ctx._hilbert_dim,
                "liouville_dim" => ctx._liouville_dim,
                "n_jump_ops"    => length(ctx.jump_ops),
            ),
        )

    elseif method === :cg
        f_neg, grad_neg! = _qc_make_fg_closures(ctx, nc, nt)
        t_start = time()
        θ_best, f_best, stats = cg_optimize(f_neg, grad_neg!, u0;
            max_iter = cfg.max_iter,
            tol      = cfg.gradient_norm_tol,
            verbose  = ctx.verbose)
        reason = stats.converged ? "gradient norm < tol" : "maximum iterations reached"
        return OptimizationResult(
            reshape(θ_best, nc, nt), -f_best,
            Float64[], Float64[],
            stats.iters, stats.converged, reason,
            time() - t_start, stats.evals, stats.evals,
            Dict{String,Any}(
                "algorithm"     => "Lindblad QC CG",
                "hilbert_dim"   => ctx._hilbert_dim,
                "liouville_dim" => ctx._liouville_dim,
                "n_jump_ops"    => length(ctx.jump_ops),
            ),
        )

    elseif method === :grape
        # First-order gradient ascent on the Liouville-space kernel.
        t_start = time()
        w       = reshape(copy(u0), nc, nt)
        step    = cfg.step_size
        F_hist  = Float64[]
        gnorm_hist = Float64[]
        F_prev  = -Inf
        F_best  = -Inf
        w_best  = copy(w)
        n_evals = 0
        n_grads = 0
        converged = false
        reason  = "maximum iterations reached"
        for it in 1:cfg.max_iter
            F, G = _qc_kernel(w, ctx)
            n_evals += 1; n_grads += 1
            push!(F_hist, F)
            push!(gnorm_hist, norm(G))
            if F > F_best
                F_best = F
                w_best .= w
            end
            if norm(G) < cfg.gradient_norm_tol
                converged = true
                reason    = "gradient norm < tol"
                break
            end
            if abs(F - F_prev) < cfg.convergence_tol && it > 1
                converged = true
                reason    = "fidelity change < tol"
                break
            end
            F_prev = F
            w     .+= step .* G
        end
        return OptimizationResult(
            w_best, F_best, F_hist, gnorm_hist,
            length(F_hist), converged, reason,
            time() - t_start, n_evals, n_grads,
            Dict{String,Any}(
                "algorithm"     => "Lindblad QC GRAPE (gradient ascent)",
                "hilbert_dim"   => ctx._hilbert_dim,
                "liouville_dim" => ctx._liouville_dim,
                "n_jump_ops"    => length(ctx.jump_ops),
            ),
        )

    elseif method ∈ (:cmaes, :pso, :nelder_mead)
        f_only = _qc_make_f_only_closure(ctx, nc, nt)
        t_start = time()
        θ_best, neg_F_best, stats = if method === :cmaes
            cmaes_optimize(f_only, u0;
                max_iters = cfg.max_iter, seed = 2025)
        elseif method === :pso
            pso_optimize(f_only, u0;
                max_iters = cfg.max_iter, seed = 2025)
        else
            nelder_mead_optimize(f_only, u0;
                max_iters = cfg.max_iter)
        end
        n_evals = hasproperty(stats, :evals) ? stats.evals :
                   hasproperty(stats, :n_evals) ? stats.n_evals : 0
        return OptimizationResult(
            reshape(θ_best, nc, nt), -neg_F_best,
            Float64[], Float64[],
            hasproperty(stats, :iters) ? stats.iters :
                hasproperty(stats, :n_iters) ? stats.n_iters : 0,
            hasproperty(stats, :converged) ? stats.converged : false,
            "metaheuristic terminated",
            time() - t_start, n_evals, 0,
            Dict{String,Any}(
                "algorithm"     => "Lindblad QC $(uppercase(string(method)))",
                "hilbert_dim"   => ctx._hilbert_dim,
                "liouville_dim" => ctx._liouville_dim,
                "n_jump_ops"    => length(ctx.jump_ops),
            ),
        )

    else
        throw(ArgumentError(
            "Unknown LindbladQCControl method :$(ctx.method).  " *
            "Supported: :lbfgs, :lbfgsb, :cg, :grape, :cmaes, :pso, :nelder_mead."))
    end
end
