# ============================================================================
# Gradient/QOC/Krotov.jl
# Krotov optimal-control methods for state-transfer and gate synthesis.
#
# Exports:
#   krotov_optimize(system, target, controls; ...)
#     First-order Sklarz-Tannor / Reich-Palao-Koch update with backward
#     co-state propagation and sequential forward re-propagation between
#     per-timestep control updates.  Monotonic for the real-overlap (state)
#     and real-gate (unitary) functionals.
#
#   krotov_second_order_optimize(system, target, controls; σ, ...)
#     Second-order Krotov with σ-term (Reich-Palao-Koch 2012, eq. 26).
#     σ=0 reduces to first-order.
#
# References:
#   Sklarz S.E., Tannor D.J., "Loading a Bose-Einstein condensate onto an
#     optical lattice: An application of optimal control theory to the
#     nonlinear Schrödinger equation", Phys. Rev. A 66, 053619 (2002).
#   Reich D.M., Ndong M., Koch C.P., "Monotonically convergent optimization
#     in quantum control using Krotov's method", J. Chem. Phys. 136, 104103
#     (2012).
#   Somlói J., Kazakov V.A., Tannor D.J., "Controlled dissociation of I2 via
#     optical transitions between the X and B electronic states", Chem.
#     Phys. 172, 85 (1993).
# ============================================================================

using LinearAlgebra
using Statistics: mean

# Resolve the χ-boundary constructor.  Precedence:
#   1. explicit chi_constructor wins
#   2. otherwise, if `metric` is given, build one from `make_chi(metric, target, ψ_T)`
#   3. otherwise, return nothing (driver falls back to target.target_state /
#      target.target_unitary, the linear/StateReal default).
# Mirrors Krotov.jl `make_chi` (workspace.jl:171-176).
function _resolve_chi_constructor(chi_constructor, metric)
    chi_constructor !== nothing && return chi_constructor
    metric           === nothing && return nothing
    metric isa AbstractFidelityMetric ||
        throw(ArgumentError("metric must be an AbstractFidelityMetric, got $(typeof(metric))"))
    return (target, ψ_T) -> make_chi(metric, target, ψ_T)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _krotov_step_propagator(system, amps_row, dt) → Matrix{ComplexF64}

Build U_k = exp(-i H_k dt) for one time slice with amplitudes `amps_row`
(a view of length `n_controls`).
"""
@inline function _krotov_step_propagator(system::AbstractQuantumSystem,
                                         amps_row::AbstractVector{<:Real},
                                         dt::Float64,
                                         Hk_buf::Union{Nothing,Matrix{ComplexF64}}=nothing)
    if Hk_buf === nothing
        Hk = copy(system.H_drift)
    else
        Hk = Hk_buf
        copyto!(Hk, system.H_drift)
    end
    @inbounds for j in 1:system.n_controls
        LinearAlgebra.axpy!(Float64(amps_row[j]), system.H_controls[j], Hk)
    end
    return compute_propagator(Hk, dt)
end

"""
    _krotov_fidelity(target, ψ_or_U) → Float64

Real-overlap / real-gate fidelity used as the Krotov objective.  Both are
linear in ψ(T) (resp. U(T)), which is what makes Krotov monotonic.
"""
@inline function _krotov_fidelity(target::QuantumTarget, ψ::AbstractVector)
    return real(dot(target.target_state, ψ))
end

@inline function _krotov_fidelity(target::QuantumTarget, U::AbstractMatrix)
    return real(tr(target.target_unitary' * U)) / target.dim
end

# ---------------------------------------------------------------------------
# krotov_optimize — first-order monotonic Krotov
# ---------------------------------------------------------------------------

"""
    krotov_optimize(system, target, controls; λ_a, max_iter, tol,
                    shape, verbose, check_invariants, callback)
        → (controls_opt, F_opt, stats)

Monotonically convergent first-order Krotov method.

Supports `target.type ∈ {"state", "unitary"}`.  The figure of merit is the
**real overlap** (linear in the final state / unitary), which is the
functional for which Krotov's update rule is guaranteed monotonic:

    F_state = Re⟨ψ_target | ψ(T)⟩              (state)
    F_gate  = Re Tr(U_target† U(T)) / dim      (unitary)

!!! note "Unitary targets and the linear functional"
    The real-trace gate functional `Re Tr(U_tgt† U(T))` has zero-gradient
    fixed points at unitaries U that are orthogonal to U_tgt under the
    Hilbert-Schmidt inner product (e.g. U ∝ iU_tgt).  Near such points
    Krotov stalls even when the absolute gate fidelity
    `|Tr(U_tgt† U)|² / dim²` is far from optimal.  For robust gate
    synthesis, prefer a state-transfer formulation with a basis of initial
    states (see the plan's §6 "Out of scope" note), or pass a non-degenerate
    initial guess.  The square-modulus functional requires the bilinear
    Krotov variant (Goerz, Reich, Koch, 2015) which is out of scope here.

Every iteration performs

    1. Backward propagate χ with OLD controls from χ(T) := target boundary.
    2. Forward re-propagate ψ with NEW controls, sequentially updating each
       timestep.  At step k the new control is
         u_j^{new}[k] = u_j^{old}[k] + (s_k / λ_a) Im⟨χ_k | H_j | ψ_k^{new}⟩
       (state target; analogous trace expression for unitary target).
    3. Record new fidelity; monotonicity is enforced by construction.

# Arguments
- `system`   — [`AbstractQuantumSystem`](@ref)
- `target`   — [`QuantumTarget`](@ref) (`"state"` or `"unitary"`)
- `controls` — [`ControlSequence`](@ref) used as the initial guess

# Keyword arguments
- `λ_a`              — step damping factor (> 0).  Smaller → larger steps
                       but may lose monotonicity for badly-scaled problems.
                       Default `1.0`.
- `max_iter`         — maximum outer iterations (default 100).
- `tol`              — stop when fidelity gain between successive iterations
                       falls below `tol` (default `1e-6`).
- `shape`            — update-shape function `s(k, N)` returning the per-step
                       weight applied to the update.  Default `(k, N) -> 1.0`.
                       Use a window (e.g. Blackman) to enforce smooth on/off.
- `enforce_monotonic`— on a non-monotonic iteration, roll back to the previous
                       controls and double `λ_a` before retrying.  True Krotov
                       monotonicity is a continuous-time property; finite-step
                       updates can overshoot near the optimum at the ε-level.
                       Default `true`.
- `verbose`          — print per-iteration summary (default `false`).
- `check_invariants` — enable runtime monotonicity check (default `false`).
- `callback`         — optional `(iter, F; grad=nothing, evals=iter) -> nothing`.

# Returns
A named tuple of the form `(controls_opt::ControlSequence, F_opt::Float64,
stats::NamedTuple)`.  `stats` fields: `iters`, `converged`, `history` (vector
of per-iteration fidelities).
"""
function krotov_optimize(system::AbstractQuantumSystem,
                         target::QuantumTarget,
                         controls::ControlSequence;
                         λ_a::Float64             = 1.0,
                         max_iter::Int            = 100,
                         tol::Float64             = 1e-6,
                         shape::Function          = (k, N) -> 1.0,
                         update_shapes            = nothing,
                         chi_constructor          = nothing,
                         metric                   = nothing,
                         enforce_monotonic::Bool  = true,
                         verbose::Bool            = false,
                         check_invariants::Bool   = false,
                         callback                 = nothing)
    chi_constructor = _resolve_chi_constructor(chi_constructor, metric)
    return _krotov_driver(system, target, controls;
                          λ_a = λ_a, σ = 0.0, order = 1,
                          max_iter = max_iter, tol = tol, shape = shape,
                          update_shapes = update_shapes,
                          chi_constructor = chi_constructor,
                          σ_adaptive = false,
                          enforce_monotonic = enforce_monotonic,
                          verbose = verbose,
                          check_invariants = check_invariants,
                          callback = callback, banner = "krotov")
end

# ---------------------------------------------------------------------------
# krotov_second_order_optimize — Reich-Palao-Koch second-order method
# ---------------------------------------------------------------------------

"""
    krotov_second_order_optimize(system, target, controls; λ_a, σ, max_iter,
                                  tol, shape, verbose, check_invariants, callback)
        → (controls_opt, F_opt, stats)

Second-order Krotov method (Reich, Palao, Koch, JCP 2012, eq. 26).  Adds a
correction term proportional to σ·Δψ = σ·(ψ^{new} − ψ^{old}) to the
first-order Sklarz-Tannor update:

    Δu_j[k] = (s_k / λ_a) · [ Im⟨χ_k^{old} | H_j | ψ_k^{new}⟩
                              + (σ/2)·Im⟨Δψ_k | H_j | ψ_k^{new}⟩ ]

σ controls the second-order curvature correction.  σ = 0 recovers
first-order Krotov (equivalent to [`krotov_optimize`](@ref)).  Positive σ
can accelerate convergence near the optimum but at the cost of a stricter
monotonicity constraint (λ_a must typically be larger).

Arguments, kwargs, and return shape are identical to
[`krotov_optimize`](@ref) except for the additional `σ::Float64` parameter
(default `0.1`).
"""
function krotov_second_order_optimize(system::AbstractQuantumSystem,
                                       target::QuantumTarget,
                                       controls::ControlSequence;
                                       λ_a::Float64             = 1.0,
                                       σ::Float64               = 0.1,
                                       σ_adaptive::Bool         = false,
                                       max_iter::Int            = 100,
                                       tol::Float64             = 1e-6,
                                       shape::Function          = (k, N) -> 1.0,
                                       update_shapes            = nothing,
                                       chi_constructor          = nothing,
                                       metric                   = nothing,
                                       enforce_monotonic::Bool  = true,
                                       verbose::Bool            = false,
                                       check_invariants::Bool   = false,
                                       callback                 = nothing)
    chi_constructor = _resolve_chi_constructor(chi_constructor, metric)
    return _krotov_driver(system, target, controls;
                          λ_a = λ_a, σ = σ, order = 2,
                          max_iter = max_iter, tol = tol, shape = shape,
                          update_shapes = update_shapes,
                          chi_constructor = chi_constructor,
                          σ_adaptive = σ_adaptive,
                          enforce_monotonic = enforce_monotonic,
                          verbose = verbose,
                          check_invariants = check_invariants,
                          callback = callback, banner = "krotov-2nd")
end

# ---------------------------------------------------------------------------
# Unified Krotov driver — shared between first- and second-order variants
# ---------------------------------------------------------------------------

function _krotov_driver(system::AbstractQuantumSystem,
                        target::QuantumTarget,
                        controls::ControlSequence;
                        λ_a::Float64,
                        σ::Float64,
                        order::Int,
                        max_iter::Int,
                        tol::Float64,
                        shape::Function,
                        update_shapes,
                        chi_constructor,
                        σ_adaptive::Bool,
                        enforce_monotonic::Bool,
                        verbose::Bool,
                        check_invariants::Bool,
                        callback,
                        banner::String)
    target.type ∈ ("state", "unitary") ||
        throw(ArgumentError("krotov requires target.type ∈ (\"state\", \"unitary\"), " *
                            "got \"$(target.type)\""))
    target.type == "state" && target.target_state === nothing &&
        throw(ArgumentError("state target has no target_state vector"))
    target.type == "unitary" && target.target_unitary === nothing &&
        throw(ArgumentError("unitary target has no target_unitary matrix"))

    dim    = system.dim
    n_ctrl = system.n_controls
    N      = controls.n_steps
    dt     = controls.dt

    # ─── Theme 7 — resolve update_shapes into a per-control vector ─────────
    # Accept: nothing (use single `shape`), Function (broadcast to all controls),
    # or Vector{Function} (per-control). Length must match n_ctrl.
    shapes_per_ctrl::Vector{Function} = if update_shapes === nothing
        Function[shape for _ in 1:n_ctrl]
    elseif update_shapes isa Function
        Function[update_shapes for _ in 1:n_ctrl]
    elseif update_shapes isa AbstractVector
        length(update_shapes) == n_ctrl ||
            throw(ArgumentError(
                "update_shapes length $(length(update_shapes)) ≠ n_controls $n_ctrl"))
        Function[f for f in update_shapes]
    else
        throw(ArgumentError(
            "update_shapes must be nothing, Function, or Vector{Function}"))
    end

    # Working copy of the control matrix [N × n_ctrl]
    amps = copy(controls.amplitudes)

    # ─── Persistent per-step workspace buffers (reused every iter/step) ─────
    Hk_buf = Matrix{ComplexF64}(undef, dim, dim)
    hψ_buf = Vector{ComplexF64}(undef, dim)
    Δψ_buf = Vector{ComplexF64}(undef, dim)

    # ─── Evaluate initial fidelity ──────────────────────────────────────────
    ψ_prev_traj, U_prev_traj, F_cur =
        _krotov_forward(system, target, amps, dt; Hk_buf = Hk_buf)
    F_best, amps_best, history = F_cur, copy(amps), Float64[F_cur]
    converged = false

    if verbose
        @printf("  %s iter  %4d  F=%+.6e\n", banner, 0, F_cur)
    end

    for iter in 1:max_iter
        # ─── Backward propagation of the co-state (old controls) ────────────
        χ_traj = _krotov_backward(system, target, amps, dt, ψ_prev_traj, U_prev_traj;
                                   Hk_buf = Hk_buf,
                                   chi_constructor = chi_constructor)

        # ─── Forward sweep with sequential per-step update ──────────────────
        amps_trial = copy(amps)
        F_new, ψ_new_traj, U_new_traj, amps_trial = _krotov_forward_update!(
            system, target, amps_trial, dt, χ_traj, ψ_prev_traj;
            λ_a = λ_a, σ = σ, order = order, shape = shape,
            shapes_per_ctrl = shapes_per_ctrl,
            Hk_buf = Hk_buf, hψ_buf = hψ_buf, Δψ_buf = Δψ_buf)

        # Rollback on regression: finite-step Krotov can overshoot near the
        # optimum (continuous-time monotonicity does not guarantee discrete
        # monotonicity).  Restore previous controls and double λ_a (and σ
        # if σ_adaptive=true so the second-order curvature term grows
        # alongside the damping).
        if enforce_monotonic && F_new < F_cur
            λ_a *= 2.0
            σ_adaptive && order ≥ 2 && (σ *= 2.0)
            verbose && @printf("  %s iter  %4d  rollback (F_new=%+.6e < F_cur=%+.6e) → λ_a=%.3e  σ=%.3e\n",
                               banner, iter, F_new, F_cur, λ_a, σ)
            continue
        end

        amps .= amps_trial
        push!(history, F_new)
        ΔF = F_new - F_cur

        if check_invariants
            ok, msg = check_monotone_ascent(history; tol = 1e-10)
            _assert_invariant(ok, msg, :krotov_monotone_ascent,
                              (; iter = iter, ΔF = ΔF))
        end

        if F_new > F_best
            F_best = F_new
            amps_best .= amps
        end

        verbose && @printf("  %s iter  %4d  F=%+.6e  ΔF=%+.3e\n",
                           banner, iter, F_new, ΔF)
        isnothing(callback) ||
            callback(iter, F_new; grad = nothing, evals = iter + 1)

        if abs(ΔF) < tol && iter > 1
            converged = true
            F_cur = F_new
            break
        end

        F_cur        = F_new
        ψ_prev_traj  = ψ_new_traj
        U_prev_traj  = U_new_traj
    end

    ctrl_opt = ControlSequence(amps_best, dt, N)
    stats    = (iters = length(history) - 1, converged = converged,
                history = history)
    return ctrl_opt, F_best, stats
end

# ---------------------------------------------------------------------------
# Forward propagation: fills trajectory ψ_k (state) or U_k (unitary)
# ---------------------------------------------------------------------------

function _krotov_forward(system::AbstractQuantumSystem,
                         target::QuantumTarget,
                         amps::AbstractMatrix{Float64},
                         dt::Float64;
                         Hk_buf::Union{Nothing,Matrix{ComplexF64}}=nothing)
    dim = system.dim
    N   = size(amps, 1)

    if target.type == "state"
        ψ = Array{ComplexF64, 2}(undef, dim, N + 1)
        ψ[:, 1] = target.target_state   # placeholder reshape
        # Actual initial state: for Krotov state transfer the user usually
        # supplies a separate initial state; we assume the convention that
        # `amps` is applied to the initial state stored as a field on the
        # system OR we take target.target_state as both boundary conditions
        # (same as state_fidelity does with the identity initial state
        # ordering in many QOC contexts).  To match Pulsar's state_fidelity
        # convention, the initial state is the first basis vector unless the
        # user wraps this call in grape_optimize_ensemble; here we require
        # the user to pass it via a field called `init_state` on the system
        # if they have one, else default to |0⟩.
        ψ[:, 1] .= _krotov_init_state(system, dim, target)
        U_traj = zeros(ComplexF64, 0, 0, 0)
        @inbounds for k in 1:N
            Uk = _krotov_step_propagator(system, view(amps, k, :), dt, Hk_buf)
            mul!(view(ψ, :, k + 1), Uk, view(ψ, :, k))
        end
        F = _krotov_fidelity(target, view(ψ, :, N + 1))
        return ψ, U_traj, F
    else  # unitary
        U = Array{ComplexF64, 3}(undef, dim, dim, N + 1)
        U[:, :, 1] = Matrix{ComplexF64}(I, dim, dim)
        @inbounds for k in 1:N
            Uk = _krotov_step_propagator(system, view(amps, k, :), dt, Hk_buf)
            mul!(view(U, :, :, k + 1), Uk, view(U, :, :, k))
        end
        ψ_traj = zeros(ComplexF64, 0, 0)
        F = _krotov_fidelity(target, view(U, :, :, N + 1))
        return ψ_traj, U, F
    end
end

# A system-level initial state used when the system doesn't carry its own
# `init_state`.  For explicit state-transfer problems the caller should
# prefer to encode ψ_init in their QuantumTarget workflow directly; Pulsar's
# current QuantumTarget only stores the target, so we adopt the convention
# that the first basis vector |0⟩ is the initial state unless overridden by
# a `metadata["init_state"]` entry on the system.
function _krotov_init_state(system::AbstractQuantumSystem, dim::Int,
                              target::Union{QuantumTarget, Nothing} = nothing)
    if target !== nothing && target.initial_state !== nothing
        v = target.initial_state
        length(v) == dim || throw(ArgumentError(
            "target.initial_state length $(length(v)) ≠ system.dim $dim"))
        return v
    end
    md = hasproperty(system, :metadata) ? getproperty(system, :metadata) : nothing
    if md isa AbstractDict && haskey(md, "init_state")
        v = ComplexF64.(md["init_state"])
        length(v) == dim || throw(ArgumentError(
            "metadata[\"init_state\"] length $(length(v)) ≠ system.dim $dim"))
        return v
    end
    ψ0 = zeros(ComplexF64, dim); ψ0[1] = 1.0
    return ψ0
end

# ---------------------------------------------------------------------------
# Backward co-state propagation (old controls)
# ---------------------------------------------------------------------------
#
# State target:    χ_N := target.target_state              (Re-overlap boundary)
#                  χ_k := U_k^† χ_{k+1}                    (unitary backward)
#
# Unitary target:  χ_N := target.target_unitary
#                  χ_k := U_k^† χ_{k+1}

function _krotov_backward(system::AbstractQuantumSystem,
                          target::QuantumTarget,
                          amps::AbstractMatrix{Float64},
                          dt::Float64,
                          ψ_prev_traj, U_prev_traj;
                          Hk_buf::Union{Nothing,Matrix{ComplexF64}}=nothing,
                          chi_constructor = nothing)
    dim = system.dim
    N   = size(amps, 1)

    if target.type == "state"
        χ = Array{ComplexF64, 2}(undef, dim, N + 1)
        # Theme 7 — pluggable χ-constructor (QuTiP-style chis_re / chis_sm
        # variants). Falls back to the default boundary χ_T = ψ_target,
        # which is the seeding for the linear real-overlap functional.
        χ_T = if chi_constructor === nothing
            target.target_state
        else
            v = chi_constructor(target,
                                size(ψ_prev_traj, 2) > 0 ? view(ψ_prev_traj, :, N + 1) : nothing)
            length(v) == dim || throw(ArgumentError(
                "chi_constructor returned vector of length $(length(v)) ≠ system.dim $dim"))
            ComplexF64.(v)
        end
        χ[:, N + 1] = χ_T
        @inbounds for k in N:-1:1
            Uk = _krotov_step_propagator(system, view(amps, k, :), dt, Hk_buf)
            mul!(view(χ, :, k), adjoint(Uk), view(χ, :, k + 1))
        end
        return χ
    else  # unitary
        χ = Array{ComplexF64, 3}(undef, dim, dim, N + 1)
        χ_T = if chi_constructor === nothing
            target.target_unitary
        else
            M = chi_constructor(target,
                                size(U_prev_traj, 3) > 0 ? view(U_prev_traj, :, :, N + 1) : nothing)
            size(M) == (dim, dim) || throw(ArgumentError(
                "chi_constructor returned matrix of size $(size(M)) ≠ ($dim, $dim)"))
            ComplexF64.(M)
        end
        χ[:, :, N + 1] = χ_T
        @inbounds for k in N:-1:1
            Uk = _krotov_step_propagator(system, view(amps, k, :), dt, Hk_buf)
            mul!(view(χ, :, :, k), adjoint(Uk), view(χ, :, :, k + 1))
        end
        return χ
    end
end

# ---------------------------------------------------------------------------
# Forward sweep with sequential control update
# ---------------------------------------------------------------------------

function _krotov_forward_update!(system::AbstractQuantumSystem,
                                 target::QuantumTarget,
                                 amps::AbstractMatrix{Float64},
                                 dt::Float64,
                                 χ_traj,
                                 ψ_prev_traj;
                                 λ_a::Float64,
                                 σ::Float64,
                                 order::Int,
                                 shape::Function,
                                 shapes_per_ctrl::Vector{Function} = Function[],
                                 Hk_buf::Union{Nothing,Matrix{ComplexF64}}=nothing,
                                 hψ_buf::Union{Nothing,Vector{ComplexF64}}=nothing,
                                 Δψ_buf::Union{Nothing,Vector{ComplexF64}}=nothing)
    dim    = system.dim
    n_ctrl = system.n_controls
    N      = size(amps, 1)
    # Theme 7 — per-control update shapes. Empty vector falls back to the
    # single shared `shape`.
    has_per_ctrl = length(shapes_per_ctrl) == n_ctrl
    @inline _shape_kj(k, j) = has_per_ctrl ?
        Float64(shapes_per_ctrl[j](k, N)) : Float64(shape(k, N))

    if target.type == "state"
        ψ_new = Array{ComplexF64, 2}(undef, dim, N + 1)
        ψ_new[:, 1] = _krotov_init_state(system, dim, target)
        hψ = hψ_buf === nothing ? Vector{ComplexF64}(undef, dim) : hψ_buf
        Δψ = Δψ_buf === nothing ? Vector{ComplexF64}(undef, dim) : Δψ_buf
        @inbounds for k in 1:N
            ψ_k_new  = view(ψ_new, :, k)
            χ_k      = view(χ_traj, :, k + 1)   # evaluate gradient at χ_{k+1}
            has_prev = order ≥ 2 && σ != 0.0 && k ≤ size(ψ_prev_traj, 2)
            if has_prev
                ψ_prev_k = view(ψ_prev_traj, :, k)
                @. Δψ = ψ_k_new - ψ_prev_k
            end
            # First-order update per control j
            for j in 1:n_ctrl
                s_kj     = _shape_kj(k, j)
                Hj       = system.H_controls[j]
                mul!(hψ, Hj, ψ_k_new)                # ⟨χ_{k+1}|H_j|ψ_k^new⟩
                inner    = dot(χ_k, hψ)
                Δu       = (s_kj / λ_a) * imag(inner)
                if has_prev
                    inner_δ = dot(Δψ, hψ)
                    Δu     += (s_kj * σ / (2 * λ_a)) * imag(inner_δ)
                end
                amps[k, j] += Δu
            end
            # Propagate with the just-updated control
            Uk = _krotov_step_propagator(system, view(amps, k, :), dt, Hk_buf)
            mul!(view(ψ_new, :, k + 1), Uk, view(ψ_new, :, k))
        end
        F = _krotov_fidelity(target, view(ψ_new, :, N + 1))
        return F, ψ_new, zeros(ComplexF64, 0, 0, 0), amps
    else  # unitary
        U_new = Array{ComplexF64, 3}(undef, dim, dim, N + 1)
        U_new[:, :, 1] = Matrix{ComplexF64}(I, dim, dim)
        @inbounds for k in 1:N
            U_k_new = view(U_new, :, :, k)
            χ_k     = view(χ_traj, :, :, k + 1)
            for j in 1:n_ctrl
                s_kj    = _shape_kj(k, j)
                Hj      = system.H_controls[j]
                # Im Tr(χ^† H_j U_k^{new}) / dim — the gate-target analogue
                inner   = tr(χ_k' * Hj * U_k_new) / dim
                Δu      = (s_kj / λ_a) * imag(inner)
                if order ≥ 2 && σ != 0.0 && size(ψ_prev_traj, 3) > 0
                    ΔU     = U_k_new .- view(ψ_prev_traj, :, :, k)
                    inner_δ = tr(ΔU' * Hj * U_k_new) / dim
                    Δu    += (s_kj * σ / (2 * λ_a)) * imag(inner_δ)
                end
                amps[k, j] += Δu
            end
            Uk_prop = _krotov_step_propagator(system, view(amps, k, :), dt, Hk_buf)
            mul!(view(U_new, :, :, k + 1), Uk_prop, view(U_new, :, :, k))
        end
        F = _krotov_fidelity(target, view(U_new, :, :, N + 1))
        return F, zeros(ComplexF64, 0, 0), U_new, amps
    end
end

# ---------------------------------------------------------------------------
# Ensemble Krotov — Vector{<:AbstractQuantumSystem} dispatch
# ---------------------------------------------------------------------------

"""
    krotov_optimize(systems::Vector{<:AbstractQuantumSystem}, target, controls;
                     aggregator = :mean, λ_a, max_iter, tol, shape, verbose,
                     enforce_monotonic, check_invariants, callback)
        → (controls_opt, F_opt, stats)

Ensemble-averaged first-order Krotov. Krotov's update rule is linear in the
co-state overlap `⟨χ_i | H_j | ψ_i⟩`, so the mean-aggregator update is simply
the sample-mean of per-system overlaps:

    Δu_j[k] = (s_k / λ_a) · mean_i Im⟨χ_{i,k+1} | H_j^(i) | ψ_{i,k}^new⟩

Every sample in `systems` must share the same control count
(`system.n_controls`) and dimension (`system.dim`). State-target mode only
(`target.type == "state"`); unitary ensembles are not currently supported —
use [`build_ensemble_from_systems`](@ref) + any generic optimizer instead.

# Restrictions on `aggregator`
- `:mean` — supported (averaged χ-overlap update, as above).
- `:worst_case` / `:cvar` — not a natural Krotov update. Use
  `build_ensemble_from_systems(systems, target, controls; aggregator=...)` +
  [`ensemble_wrap`](@ref) + [`lbfgs_optimize`](@ref) / any gradient optimizer.

# Other arguments
Same as the single-system [`krotov_optimize`](@ref). `stats.history` contains
the per-iteration *ensemble-mean* fidelity.
"""
function krotov_optimize(systems::Vector{<:AbstractQuantumSystem},
                         target::QuantumTarget,
                         controls::ControlSequence;
                         aggregator::Symbol       = :mean,
                         λ_a::Float64             = 1.0,
                         max_iter::Int            = 100,
                         tol::Float64             = 1e-6,
                         shape::Function          = (k, N) -> 1.0,
                         enforce_monotonic::Bool  = true,
                         verbose::Bool            = false,
                         check_invariants::Bool   = false,
                         callback                 = nothing)
    aggregator === :mean ||
        throw(ArgumentError(
            "krotov_optimize(::Vector, ...) supports only aggregator=:mean. " *
            "For :worst_case or :cvar, use build_ensemble_from_systems + " *
            "ensemble_wrap + any generic optimizer (L-BFGS, CG, Adam, ...)."))
    target.type == "state" ||
        throw(ArgumentError(
            "Ensemble Krotov currently supports only state targets " *
            "(target.type == \"state\"); got \"$(target.type)\"."))
    isempty(systems) && throw(ArgumentError("systems must be non-empty"))

    n_ctrl = systems[1].n_controls
    dim    = systems[1].dim
    for (i, s) in enumerate(systems)
        s.n_controls == n_ctrl ||
            throw(ArgumentError("systems[$i].n_controls=$(s.n_controls) ≠ $n_ctrl"))
        s.dim == dim ||
            throw(ArgumentError("systems[$i].dim=$(s.dim) ≠ $dim"))
    end

    N  = controls.n_steps
    dt = controls.dt
    N_s = length(systems)

    amps = copy(controls.amplitudes)

    # Per-sample forward trajectory
    function forward_all(amps)
        ψs = [Array{ComplexF64,2}(undef, dim, N + 1) for _ in 1:N_s]
        Fs = zeros(Float64, N_s)
        for i in 1:N_s
            ψs[i][:, 1] .= _krotov_init_state(systems[i], dim, target)
            @inbounds for k in 1:N
                Uk = _krotov_step_propagator(systems[i], view(amps, k, :), dt)
                ψs[i][:, k + 1] = Uk * ψs[i][:, k]
            end
            Fs[i] = _krotov_fidelity(target, ψs[i][:, N + 1])
        end
        return ψs, Fs
    end

    # Per-sample backward co-state trajectory (old controls)
    function backward_all(amps)
        χs = [Array{ComplexF64,2}(undef, dim, N + 1) for _ in 1:N_s]
        for i in 1:N_s
            χs[i][:, N + 1] .= target.target_state
            @inbounds for k in N:-1:1
                Uk = _krotov_step_propagator(systems[i], view(amps, k, :), dt)
                χs[i][:, k] = Uk' * χs[i][:, k + 1]
            end
        end
        return χs
    end

    ψs_prev, Fs_cur = forward_all(amps)
    F_cur = mean(Fs_cur)
    F_best, amps_best = F_cur, copy(amps)
    history = Float64[F_cur]
    converged = false

    verbose && @printf("  krotov-ens iter  %4d  F=%+.6e\n", 0, F_cur)

    for iter in 1:max_iter
        χs = backward_all(amps)

        amps_trial = copy(amps)
        ψs_new = [Array{ComplexF64,2}(undef, dim, N + 1) for _ in 1:N_s]
        for i in 1:N_s
            ψs_new[i][:, 1] .= _krotov_init_state(systems[i], dim, target)
        end

        @inbounds for k in 1:N
            s_k = Float64(shape(k, N))
            for j in 1:n_ctrl
                acc = 0.0
                for i in 1:N_s
                    Hj      = systems[i].H_controls[j]
                    χ_k1    = view(χs[i], :, k + 1)
                    ψ_k     = view(ψs_new[i], :, k)
                    acc    += imag(dot(χ_k1, Hj * ψ_k))
                end
                amps_trial[k, j] += (s_k / λ_a) * (acc / N_s)
            end
            for i in 1:N_s
                Uk = _krotov_step_propagator(systems[i], view(amps_trial, k, :), dt)
                ψs_new[i][:, k + 1] = Uk * ψs_new[i][:, k]
            end
        end

        Fs_new = [_krotov_fidelity(target, ψs_new[i][:, N + 1]) for i in 1:N_s]
        F_new  = mean(Fs_new)

        if enforce_monotonic && F_new < F_cur
            λ_a *= 2.0
            verbose && @printf("  krotov-ens iter  %4d  rollback (F_new=%+.6e < F_cur=%+.6e) → λ_a=%.3e\n",
                               iter, F_new, F_cur, λ_a)
            continue
        end

        amps .= amps_trial
        push!(history, F_new)
        ΔF = F_new - F_cur

        if check_invariants
            ok, msg = check_monotone_ascent(history; tol = 1e-10)
            _assert_invariant(ok, msg, :krotov_ensemble_monotone_ascent,
                              (; iter = iter, ΔF = ΔF))
        end

        if F_new > F_best
            F_best = F_new
            amps_best .= amps
        end

        verbose && @printf("  krotov-ens iter  %4d  F=%+.6e  ΔF=%+.3e\n",
                           iter, F_new, ΔF)
        isnothing(callback) || callback(iter, F_new; grad = nothing, evals = iter + 1)

        if abs(ΔF) < tol && iter > 1
            converged = true
            F_cur = F_new
            break
        end

        F_cur   = F_new
        ψs_prev = ψs_new
    end

    ctrl_opt = ControlSequence(amps_best, dt, N)
    stats    = (iters = length(history) - 1, converged = converged,
                history = history)
    return ctrl_opt, F_best, stats
end

