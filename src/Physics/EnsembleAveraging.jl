# Physics/EnsembleAveraging.jl
#
# Evaluation of an EnsembleAveraging (spec in Types/EnsembleAveraging.jl):
#
#     F = aggregate over members k of  ω_k · F_k,   F_k = Σ_p ν_p f_m(z_kp) / Σ_p ν_p
#
# with members k from the axes, probe pairs p from the input-state axis (or the
# target itself when there are no probes), and the metric arithmetic f_m / w_m
# from Computation/FidelityReductions.jl. The gradient is
#
#     ∂F/∂u = Σ_k a_k · ∂F_k/∂u,   a_k = ∂aggregate/∂F_k,
#
# where each ∂F_k/∂u is the first-order GRAPE gradient of the member (the same
# code compute_grape_gradient uses), so an ensemble with one member reproduces
# compute_fidelity / compute_grape_gradient exactly.

using LinearAlgebra

# ─── Building one member ─────────────────────────────────────────────────────

function _apply_axis(a::TermAxis, i, Hd, Hc, dt)
    size(a.operator) == size(Hd) || throw(DimensionMismatch(
        "TermAxis :$(a.label) operator $(size(a.operator)) ≠ drift $(size(Hd))"))
    return Hd .+ (2π * a.values[i]) .* a.operator, Hc, dt
end
function _apply_axis(a::DriveScaleAxis, i, Hd, Hc, dt)
    s = a.values[i]
    chans = a.channels === :all ? eachindex(Hc) : a.channels
    Hc2 = copy(Hc)
    for c in chans; Hc2[c] = s .* Hc[c]; end
    return Hd, Hc2, dt
end
function _apply_axis(a::DrivePhaseAxis, i, Hd, Hc, dt)
    c, s = cos(a.values[i]), sin(a.values[i])
    Hc2 = copy(Hc)
    for (x, y) in a.pairs
        Hc2[x] = c .* Hc[x] .+ s .* Hc[y]
        Hc2[y] = c .* Hc[y] .- s .* Hc[x]
    end
    return Hd, Hc2, dt
end
_apply_axis(a::TimeScaleAxis, i, Hd, Hc, dt) = (Hd, Hc, dt * a.values[i])
function _apply_axis(a::DriftListAxis, i, Hd, Hc, dt)
    size(a.values[i]) == size(Hd) || throw(DimensionMismatch(
        "DriftListAxis drift $(size(a.values[i])) ≠ nominal drift $(size(Hd))"))
    return a.values[i], Hc, dt
end
function _apply_axis(a::FunctionAxis, i, Hd, Hc, dt)
    Hd2, Hc2 = a.f(Hd, Hc, a.values[i])
    return Matrix{ComplexF64}(Hd2), Matrix{ComplexF64}[Matrix{ComplexF64}(H) for H in Hc2], dt
end

function _member_problem(ens::EnsembleAveraging, index::Tuple, Hd, Hc, dt)
    for (a, i) in zip(ens.axes, index)
        Hd, Hc, dt = _apply_axis(a, i, Hd, Hc, dt)
    end
    return Hd, Hc, dt
end

# ─── Metric of the ensemble ──────────────────────────────────────────────────

function _ensemble_metric(probes, target, metric::Symbol)
    if probes === nothing
        target isa QuantumTarget ||
            throw(ArgumentError("ensemble_averaging needs a QuantumTarget or probe pairs"))
        metric === :auto && return target.metric
        return target.type == "unitary" ?
            canonical_metric(metric, VALID_GATE_METRICS;  context = "ensemble_averaging") :
            canonical_metric(metric, VALID_STATE_METRICS; context = "ensemble_averaging")
    end
    target === nothing || throw(ArgumentError(
        "ensemble_averaging: give either a target or probe pairs, not both"))
    if probes isa StateProbes
        return canonical_metric(metric === :auto ? :state_squared : metric,
                                (STATE_METRICS..., DM_METRICS...); context = "state probe pairs")
    end
    return canonical_metric(metric === :auto ? :dm_real : metric, DM_METRICS;
                            context = "operator probe pairs")
end

_with_metric(t::QuantumTarget, m::Symbol) = m === t.metric ? t :
    QuantumTarget(t.type, t.target_state, t.target_unitary, t.initial_state, t.dim, m)

# ─── One member's fidelity and gradient ──────────────────────────────────────

function _member_fidelity(probes, target, metric, Hd, Hc, controls, dt, want_grad)
    sys = quantum_system(Hd, Hc)
    cs  = dt == controls.dt ? controls : ControlSequence(controls.amplitudes, dt, controls.n_steps)
    H_total = build_total_hamiltonian(sys, cs)
    U_steps = compute_propagators(H_total, dt)
    P = compute_forward_propagators(U_steps)
    n_t = size(U_steps, 1)
    U   = P[n_t + 1, :, :]
    if probes === nothing
        tgt = _with_metric(target, metric)
        F = compute_fidelity(U, tgt)
        want_grad || return F, nothing
        Q = compute_backward_propagators(U_steps)
        return F, _target_gradient(U, P, Q, sys.H_controls, tgt, dt)
    end
    Q = want_grad ? compute_backward_propagators(U_steps) : nothing
    ν = probes.weights
    F = 0.0
    G = want_grad ? zeros(Float64, length(Hc), n_t) : nothing
    for p in eachindex(probes.inputs)
        Fp, Gp = _probe_pair(probes, p, metric, U, P, Q, sys.H_controls, dt, want_grad)
        F += ν[p] * Fp
        want_grad && (G .+= ν[p] .* Gp)
    end
    S = sum(ν)
    want_grad && (G ./= S)
    return F / S, G
end

# Same dispatch as compute_grape_gradient (fast squared paths, typed otherwise),
# so a one-member ensemble reproduces it exactly.
function _target_gradient(U, P, Q, Hc, tgt::QuantumTarget, dt)
    if tgt.type == "unitary"
        sym = tgt.metric === :auto ? :prop_squared : tgt.metric
        sym === :prop_squared &&
            return compute_gradient_gate(U, P, Q, Hc, tgt.target_unitary, dt)
        return compute_gradient_gate(U, P, Q, Hc, tgt.target_unitary, dt, _gate_symbol_to_metric(sym))
    end
    ψi = tgt.initial_state === nothing ? tgt.target_state : tgt.initial_state
    sym = tgt.metric === :auto ? :state_squared : tgt.metric
    sym === :state_squared &&
        return compute_gradient_state(U, P, Q, Hc, ψi, tgt.target_state, dt)
    return compute_gradient_state(U, P, Q, Hc, ψi, tgt.target_state, dt, _state_symbol_to_metric(sym))
end

function _probe_pair(probes::StateProbes, p, metric, U, P, Q, Hc, dt, want_grad)
    ψi, ψt = probes.inputs[p], probes.targets[p]
    m = _state_symbol_to_metric(metric)
    F = state_fidelity(ψt, U * ψi, m)
    want_grad || return F, nothing
    return F, compute_gradient_state(U, P, Q, Hc, ψi, ψt, dt, m)
end

# Operator pair A → B: z = Tr(B† U A U†)/(‖A‖‖B‖). With the first-order GRAPE
# step derivative dU = −i·dt·Q_k†·H_j·P_k used by every gradient here,
#   dz = (−i·dt·Tr(H_j M1_k) + i·dt·Tr(H_j M2_k)) / (‖A‖‖B‖),
#   M1_k = P_k A U† B† Q_k†,   M2_k = Q_k B† U A P_k†,
# and dF = Re(w·dz) with the metric's chain weight w.
function _probe_pair(probes::OperatorProbes, p, metric, U, P, Q, Hc, dt, want_grad)
    A, B = probes.inputs[p], probes.targets[p]
    nrm = norm(A) * norm(B)
    UA  = U * A
    z   = tr(B' * UA * U') / nrm
    F   = overlap_fidelity(metric, z)
    want_grad || return F, nothing
    w   = overlap_weight(metric, z)
    n_t = size(P, 1) - 1
    G   = zeros(Float64, length(Hc), n_t)
    AUBd = A * U' * B'          # A U† B†
    BdUA = B' * UA              # B† U A
    for k in 1:n_t
        Pk = P[k, :, :]; Qk = Q[k, :, :]
        M1 = Pk * AUBd * Qk'
        M2 = Qk * BdUA * Pk'
        for (j, Hj) in enumerate(Hc)
            dz = (-im * dt * tr(Hj * M1) + im * dt * tr(Hj * M2)) / nrm
            G[j, k] = real(w * dz)
        end
    end
    return F, G
end

# ─── Aggregation ─────────────────────────────────────────────────────────────

# Returns (F, a) with a_k = ∂F/∂F_k.
function _ensemble_aggregate(ens::EnsembleAveraging, Fk::Vector{Float64}, ω::Vector{Float64})
    agg = ens.aggregate
    if agg === :sum
        return sum(ω .* Fk), copy(ω)
    end
    all(>=(0), ω) || throw(ArgumentError(
        "aggregate = :$agg needs non-negative weights (use :sum for signed weights)"))
    S = sum(ω)
    S > 0 || throw(ArgumentError("ensemble weights sum to zero"))
    ŵ = ω ./ S
    if agg === :mean
        return sum(ŵ .* Fk), ŵ
    elseif agg === :worst_case
        k = argmin(Fk); a = zeros(length(Fk)); a[k] = 1.0
        return Fk[k], a
    elseif agg === :cvar
        α = ens.cvar_alpha; a = zeros(length(Fk)); cum = 0.0
        for k in sortperm(Fk)
            take = min(ŵ[k], α - cum)
            take <= 0 && break
            a[k] = take / α; cum += take
        end
        return sum(a .* Fk), a
    else  # :softmin
        β = ens.softmin_beta; m = minimum(Fk)
        e = ŵ .* exp.(-β .* (Fk .- m)); s = sum(e)
        return m - log(s) / β, e ./ s
    end
end

# ─── Public entry point ──────────────────────────────────────────────────────

"""
    ensemble_averaging(ens, system, controls, target = nothing;
                       metric = :auto, gradient = true) -> (F, G) or F

Evaluate the ensemble average `ens` (an [`EnsembleAveraging`](@ref)) of the
fidelity of `controls` on `system` (anything with `H_drift` and `H_controls`,
e.g. a `QuantumSystem` or `SpinSystem`). Every member is the system with its
axes applied; each member's fidelity is

- with a `target` (unitary or state `QuantumTarget`): the target's metric, or
  `metric` if given — exactly what `compute_fidelity` returns for that member;
- with probe pairs (`ens.probes`, and `target = nothing`): the weighted mean
  over pairs of `metric` — `:state_squared` by default for `StateProbes`
  (state or, on the pure states, DM row), `:dm_real` for `OperatorProbes`.

`metric` defaults to `ens.metric` (`:auto` = the defaults above).

The member fidelities are combined by `ens.aggregate`. `G` is ∂F/∂controls in
the `[n_controls × n_timesteps]` layout of `compute_grape_gradient` (first-order
GRAPE, as there). Members run in parallel threads.
"""
function ensemble_averaging(ens::EnsembleAveraging, system,
                            controls::ControlSequence, target = nothing;
                            metric::Symbol = ens.metric, gradient::Bool = true)
    (hasproperty(system, :H_drift) && hasproperty(system, :H_controls)) || throw(ArgumentError(
        "ensemble_averaging: $(typeof(system)) has no H_drift / H_controls; " *
        "build the ensemble on a QuantumSystem, or use the MR front ends (MRControl(…; ensemble))"))
    m = _ensemble_metric(ens.probes, target, metric)
    members = ensemble_members(ens)
    M  = length(members)
    Fk = Vector{Float64}(undef, M)
    Gk = Vector{Union{Nothing, Matrix{Float64}}}(nothing, M)
    Hd0 = Matrix{ComplexF64}(system.H_drift)
    Hc0 = Matrix{ComplexF64}[Matrix{ComplexF64}(H) for H in system.H_controls]
    Threads.@threads :dynamic for k in 1:M
        Hd, Hc, dt = _member_problem(ens, members[k].index, Hd0, Hc0, controls.dt)
        Fk[k], Gk[k] = _member_fidelity(ens.probes, target, m, Hd, Hc, controls, dt, gradient)
    end
    F, a = _ensemble_aggregate(ens, Fk, Float64[mb.weight for mb in members])
    gradient || return F
    G = zeros(Float64, size(Gk[1]))
    for k in 1:M
        a[k] == 0 || (G .+= a[k] .* Gk[k])
    end
    return F, G
end
