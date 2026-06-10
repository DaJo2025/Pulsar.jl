"""
    Gradients.jl

Analytical GRAPE gradient computation for quantum optimal control.

The GRAPE (GRadient Ascent Pulse Engineering) algorithm, introduced by
Khaneja et al. (2005), computes exact analytical gradients of a fidelity
functional with respect to piecewise-constant control amplitudes.

For a control sequence {u_j[k] : j = 1…n_c, k = 1…n_t}, the GRAPE gradient is:

    ∂F / ∂u_j[k]

computed in O(n_t * dim³) time after the forward and backward propagators are
pre-computed.

Reference:
  Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse
  sequences by gradient ascent algorithms", J. Magn. Reson. 172 (2005) 296–305.
  DOI: 10.1016/j.jmr.2004.11.004
"""

using LinearAlgebra

# ============================================================================
# Main GRAPE gradient entry point
# ============================================================================

"""
    compute_grape_gradient(system::AbstractQuantumSystem,
                            controls::ControlSequence,
                            target::QuantumTarget) -> Matrix{Float64}

Compute the analytical GRAPE gradient ∂F/∂u_j[k] for all controls and timesteps.

# Arguments
- `system`   — quantum system with H_drift and H_controls
- `controls` — current control sequence
- `target`   — optimization target (unitary or state)

# Returns
Gradient matrix `G` of shape `[n_controls × n_timesteps]` where
`G[j, k] = ∂F/∂u_j[k]`.

Positive gradient entries indicate that increasing u_j[k] increases fidelity.

# Algorithm
1. Build H_total[k] = H_drift + Σ_j u_j[k] H_j for each time step k.
2. Compute step propagators U[k] = exp(-i H_total[k] dt).
3. Compute forward propagators P[k] via `compute_forward_propagators`.
4. Compute backward propagators Q[k] via `compute_backward_propagators`.
5. Compute total propagator U_total = P[n_t+1].
6. Dispatch to `compute_gradient_gate` or `compute_gradient_state` based on
   `target.type`.

# Mathematical formulas
For gate fidelity F = |Φ|² with Φ = Tr(U_target† U_total) / dim:

    ∂F/∂u_j[k] = 2 Re[ conj(Φ) * Tr(Q[k]† (-i H_j dt) P[k]) / dim ]

where P[k] = U[k-1]…U[1] and Q[k] = (U[n_t]…U[k+1])†, so that
Q[k]† P[k+1] = U[n_t]…U[1] = U_total and
Tr(Q[k]† U[k] P[k]) = Tr(U_total) (approximately, by trace cyclicity).

For state transfer fidelity F = |⟨ψ_t|U|ψ_i⟩|² with χ = ⟨ψ_t|U|ψ_i⟩:

    ∂F/∂u_j[k] = 2 Re[ conj(χ) * ⟨ψ_t| Q[k]† (-i H_j dt) P[k] |ψ_i⟩ ]

# Throws
- `ArgumentError` if target.type is not `"unitary"` or `"state"`.
- Propagates errors from propagator and fidelity routines.

# Example
```julia
G = compute_grape_gradient(sys, seq, tgt)
# Gradient ascent update:
seq_new = ControlSequence(seq.controls .+ α .* G, seq.dt, seq.total_time, seq.n_timesteps)
```
"""
function compute_grape_gradient(system::AbstractQuantumSystem,
                                 controls::ControlSequence,
                                 target::QuantumTarget)::Matrix{Float64}
    # Step 1 & 2: build total Hamiltonians and step propagators
    H_total = build_total_hamiltonian(system, controls)
    U_steps = _maybe_batched_propagators(H_total, controls.dt)

    # Step 3 & 4: forward and backward propagators
    P = compute_forward_propagators(U_steps)   # (n_t+1) × dim × dim
    Q = compute_backward_propagators(U_steps)  # (n_t+1) × dim × dim

    # Step 5: total propagator = P[n_t + 1]
    n_t = controls.n_timesteps
    U_total = P[n_t + 1, :, :]

    # Step 6: dispatch (honour target.metric; :auto → the per-type default)
    if target.type == "unitary"
        if target.target_unitary === nothing
            throw(ArgumentError(
                "target.type is \"unitary\" but target.target_unitary is nothing"))
        end
        sym = target.metric === :auto ? :normalized : target.metric
        if sym === :normalized
            return compute_gradient_gate(U_total, P, Q, system.H_controls,
                                         target.target_unitary, controls.dt)
        else
            return compute_gradient_gate(U_total, P, Q, system.H_controls,
                                         target.target_unitary, controls.dt,
                                         _gate_symbol_to_metric(sym))
        end

    elseif target.type == "state"
        if target.target_state === nothing
            throw(ArgumentError(
                "target.type is \"state\" but target.target_state is nothing"))
        end
        # If an explicit initial_state is provided on the target, use it and
        # optimise the proper state-transfer fidelity |⟨ψ_t|U|ψ_i⟩|².
        # Otherwise fall back to the legacy convention (target_state used as
        # both initial and final, i.e. fixed-point fidelity).
        psi_init   = target.initial_state === nothing ?
                        target.target_state : target.initial_state
        psi_target = target.target_state
        sym = target.metric === :auto ? :square : target.metric
        if sym === :square
            return compute_gradient_state(U_total, P, Q, system.H_controls,
                                          psi_init, psi_target, controls.dt)
        else
            return compute_gradient_state(U_total, P, Q, system.H_controls,
                                          psi_init, psi_target, controls.dt,
                                          _state_symbol_to_metric(sym))
        end

    else
        throw(ArgumentError(
            "Unknown target type \"$(target.type)\"; expected \"unitary\" or \"state\""))
    end
end

"""
    _maybe_batched_propagators(H_total, dt) -> U_steps

Build step propagators `U[k] = exp(-i H[k] dt)` for `H_total::(n_t, dim, dim)`,
using the batched GPU primitive when the active device is a GPU **and** the
batch volume warrants it (see [`plan_hybrid_execution`]); otherwise the standard
threaded CPU `compute_propagators`.  The GPU path permutes to the primitive's
`(dim, dim, N)` layout, runs one batched eigensolve, and permutes back — so the
result is identical to the CPU path (validated) regardless of device.
"""
function _maybe_batched_propagators(H_total::Array{ComplexF64,3}, dt::Real)
    n_t, dim, _ = size(H_total)
    dev = get_device()
    if dev !== :cpu && _gpu_available()
        planner  = HybridExecutionPlanner()
        decision = plan_hybrid_execution(dim, n_t, true, planner;
                                         op = "propagator", batch_count = n_t)
        if decision === :gpu
            be = resolve_backend(dev)
            Hb = permutedims(H_total, (2, 3, 1))                 # (dim, dim, n_t)
            Ub = Array(batched_herm_propagators(be, Hb, Float64(dt)))
            return permutedims(Ub, (3, 1, 2))                    # (n_t, dim, dim)
        end
    end
    return compute_propagators(H_total, dt)
end

"""
    compute_grape_gradient_with!(G, system, target, P, Q, U_total, dt)

Pre-allocated GRAPE gradient: writes the gradient into the supplied buffer `G`
(shape `[n_controls × n_timesteps]`) using *already-computed* forward (`P`),
backward (`Q`), and total (`U_total`) propagators. Skips the redundant
recomputation that the immutable `compute_grape_gradient` performs internally.

This is the entry point used by `grape_optimize` once propagators have been
filled in-place by `compute_propagators!`, `compute_forward_propagators!`, and
`compute_backward_propagators!`.
"""
function compute_grape_gradient_with!(G::Matrix{Float64},
                                       system::AbstractQuantumSystem,
                                       target::QuantumTarget,
                                       P::Array{ComplexF64,3},
                                       Q::Array{ComplexF64,3},
                                       U_total::Matrix{ComplexF64},
                                       dt::Real)::Matrix{Float64}
    if target.type == "unitary"
        target.target_unitary === nothing &&
            throw(ArgumentError("target.type is \"unitary\" but target.target_unitary is nothing"))
        sym = target.metric === :auto ? :normalized : target.metric
        if sym === :normalized
            _gate_gradient_into!(G, U_total, P, Q, system.H_controls,
                                 target.target_unitary, dt)
        else
            # Non-default metric: the metric-parameterised gradient is allocating;
            # copy it into the caller's buffer.  (Default :auto keeps the fast
            # in-place path, so there is no regression for the common case.)
            G .= compute_gradient_gate(U_total, P, Q, system.H_controls,
                                       target.target_unitary, dt,
                                       _gate_symbol_to_metric(sym))
        end
    elseif target.type == "state"
        target.target_state === nothing &&
            throw(ArgumentError("target.type is \"state\" but target.target_state is nothing"))
        sym = target.metric === :auto ? :square : target.metric
        if sym === :square
            _state_gradient_into!(G, U_total, P, Q, system.H_controls,
                                  target.target_state, target.target_state, dt)
        else
            G .= compute_gradient_state(U_total, P, Q, system.H_controls,
                                        target.target_state, target.target_state, dt,
                                        _state_symbol_to_metric(sym))
        end
    else
        throw(ArgumentError(
            "Unknown target type \"$(target.type)\"; expected \"unitary\" or \"state\""))
    end
    return G
end

# ============================================================================
# Gate fidelity gradient
# ============================================================================

"""
    compute_gradient_gate(U_total::Matrix{ComplexF64},
                           P::Array{ComplexF64,3},
                           Q::Array{ComplexF64,3},
                           H_controls::Vector{Matrix{ComplexF64}},
                           U_target::Matrix{ComplexF64},
                           dt::Real) -> Matrix{Float64}

Compute the gate fidelity gradient ∂F_gate/∂u_j[k].

# Arguments
- `U_total`    — total propagator, shape `dim × dim`
- `P`          — forward propagators, shape `(n_t+1) × dim × dim`;
  `P[k, :, :] = U[k-1]…U[1]` (identity at k=1)
- `Q`          — backward propagators, shape `(n_t+1) × dim × dim`;
  `Q[k, :, :] = (U[n_t]…U[k+1])†` (identity at k=n_t+1)
- `H_controls` — list of n_c control Hamiltonians, each `dim × dim`
- `U_target`   — target unitary, `dim × dim`
- `dt`         — time step (seconds)

# Returns
Gradient matrix `G[j, k] = ∂F_gate/∂u_j[k]`, shape `[n_c × n_t]`.

# Derivation
Let Φ = Tr(U_target† U_total) / dim.  Then F_gate = |Φ|².

The first-order variation of Φ when u_j[k] changes by δ is:

    δΦ = Tr(U_target† * U[n_t]…(U[k] + δU[k])…U[1]) / dim

where δU[k] = exp(-i(H_total[k] + δ*u_j*H_j)dt) - exp(-i H_total[k] dt)
            ≈ -i dt δ * U[k] H_j   (to first order in δ)

Wait — more carefully, using the identity for the derivative of the matrix
exponential with respect to a linear parameter:

    d/dε exp(-i(H_0 + ε H_j)dt)|_{ε=0} = -i dt * exp(-i H_0 dt) H_j
                                         + higher-order commutator terms

For piecewise-constant GRAPE with exact propagators, we use the exact first-order
perturbation result for matrix exponentials of Hermitian matrices (which holds
exactly in the limit that H_j and H_total commute, and is treated as the standard
GRAPE approximation for the general non-commuting case):

    δU[k] ≈ -i dt δ * H_j * U[k]   (left action convention)

Then:

    δΦ / δ = Tr(U_target† Q[k]† (-i dt H_j) U[k] P[k]) / dim
           = Tr(U_target† Q[k]† (-i dt H_j) P[k+1]) / dim

and using the cyclic property of the trace and the definition Λ = U_target† U_total:

    δF / δ = 2 Re( conj(Φ) * Tr(Q[k]† (-i dt H_j) P[k+1]) / dim )
           = 2 Re( conj(Φ) * Tr((-i dt H_j) P[k+1] Q[k]†) / dim )

Note that P[k+1] Q[k]† = U[k] * P[k] * Q[k]† = U_total (because
P[k+1] = U[k] P[k] and Q[k]† P is related to U_total by trace cyclicity).

In the implementation we compute:

    ∂F/∂u_j[k] = 2 Re[ conj(Φ) * Tr( Q[k]† * (-i dt H_j) * P[k] ) / dim ]

where we rely on P[k] = forward propagator *before* step k and Q[k] is the
backward propagator from steps k+1 through n_t, which together satisfy
Q[k] U[k] P[k] = U_total, so Q[k]† = U_total * P[k]† * U[k]†.
Substituting gives the standard GRAPE formula.

# Example
```julia
G = compute_gradient_gate(U_total, P, Q, sys.H_controls, U_target, dt)
```
"""
function compute_gradient_gate(U_total::Matrix{ComplexF64},
                                P::Array{ComplexF64,3},
                                Q::Array{ComplexF64,3},
                                H_controls::Vector{Matrix{ComplexF64}},
                                U_target::Matrix{ComplexF64},
                                dt::Real)::Matrix{Float64}
    n_t = size(P, 1) - 1
    n_c = length(H_controls)
    G   = zeros(Float64, n_c, n_t)
    return _gate_gradient_into!(G, U_total, P, Q, H_controls, U_target, dt)
end

function _gate_gradient_into!(G::Matrix{Float64},
                               U_total::Matrix{ComplexF64},
                               P::Array{ComplexF64,3},
                               Q::Array{ComplexF64,3},
                               H_controls::Vector{Matrix{ComplexF64}},
                               U_target::Matrix{ComplexF64},
                               dt::Real)::Matrix{Float64}
    dt  = Float64(dt)
    n_t = size(P, 1) - 1
    n_c = length(H_controls)
    dim = size(U_total, 1)

    # Complex overlap Φ = Tr(U_target† U_total) / dim
    Phi = tr(U_target' * U_total) / dim

    U_t_dag = U_target'
    tmp     = Matrix{ComplexF64}(undef, dim, dim)
    A_k     = Matrix{ComplexF64}(undef, dim, dim)
    Pk_buf  = Matrix{ComplexF64}(undef, dim, dim)
    Qk_buf  = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_t
        @views copyto!(Pk_buf, P[k, :, :])
        @views copyto!(Qk_buf, Q[k, :, :])
        mul!(tmp, Pk_buf, U_t_dag)
        mul!(A_k, tmp, Qk_buf')

        for j in 1:n_c
            Hj = H_controls[j]
            s  = zero(ComplexF64)
            @inbounds for q in 1:dim, p in 1:dim
                s += A_k[p, q] * Hj[q, p]
            end
            inner   = -im * dt * s / dim
            G[j, k] = 2.0 * real(conj(Phi) * inner)
        end
    end
    return G
end

# ----------------------------------------------------------------------------
# Metric-aware gate-fidelity gradient
# ----------------------------------------------------------------------------
#
# Same loop kernel as `_gate_gradient_into!`, but the chain-rule factor is
# selected at compile time via dispatch on `metric::AbstractFidelityMetric`.
# Used when the user wants a non-`:square` gate metric — `ModulusGate`,
# `RealGate`, etc. — without rewriting the propagator setup.
#
# The "bare" inner product passed to `fidelity_grad_prefactor` is
#     s_bar = Tr(U_target† Q[k]† H_j P[k]) / dim
# (no `-i·dt` factor; the prefactor folds in `dt`). For the existing
# `NormalizedGate` metric this reproduces the `_gate_gradient_into!` formula
# exactly (verified algebraically: the existing code stores
# `inner = -i·dt·s/dim` and returns `2 Re(conj(Φ)·inner)`, while the prefactor
# dispatch returns `2·dt·Im(conj(Φ)·s_bar)` — equal up to sign of i).
function compute_gradient_gate(U_total::Matrix{ComplexF64},
                                P::Array{ComplexF64,3},
                                Q::Array{ComplexF64,3},
                                H_controls::Vector{Matrix{ComplexF64}},
                                U_target::Matrix{ComplexF64},
                                dt::Real,
                                metric::AbstractFidelityMetric)::Matrix{Float64}
    dt  = Float64(dt)
    n_t = size(P, 1) - 1
    n_c = length(H_controls)
    dim = size(U_total, 1)
    G   = zeros(Float64, n_c, n_t)

    Phi = tr(U_target' * U_total) / dim

    U_t_dag = U_target'
    tmp     = Matrix{ComplexF64}(undef, dim, dim)
    A_k     = Matrix{ComplexF64}(undef, dim, dim)
    Pk_buf  = Matrix{ComplexF64}(undef, dim, dim)
    Qk_buf  = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_t
        @views copyto!(Pk_buf, P[k, :, :])
        @views copyto!(Qk_buf, Q[k, :, :])
        mul!(tmp, Pk_buf, U_t_dag)
        mul!(A_k, tmp, Qk_buf')

        for j in 1:n_c
            Hj = H_controls[j]
            s  = zero(ComplexF64)
            @inbounds for q in 1:dim, p in 1:dim
                s += A_k[p, q] * Hj[q, p]
            end
            inner_bare = s / dim
            G[j, k] = fidelity_grad_prefactor(Phi, inner_bare, dt, metric)
        end
    end
    # AverageGate's ∂F differs from the process-fidelity ∂F by a constant d/(d+1).
    if metric isa AverageGate
        G .*= dim / (dim + 1)
    end
    return G
end

# ============================================================================
# State transfer fidelity gradient
# ============================================================================

"""
    compute_gradient_state(U_total::Matrix{ComplexF64},
                            P::Array{ComplexF64,3},
                            Q::Array{ComplexF64,3},
                            H_controls::Vector{Matrix{ComplexF64}},
                            psi_init::Vector{ComplexF64},
                            psi_target::Vector{ComplexF64},
                            dt::Real) -> Matrix{Float64}

Compute the state-transfer fidelity gradient ∂F_state/∂u_j[k].

# Arguments
- `U_total`    — total propagator
- `P`          — forward propagators `(n_t+1) × dim × dim`
- `Q`          — backward propagators `(n_t+1) × dim × dim`
- `H_controls` — control Hamiltonians
- `psi_init`   — initial state (normalized internally)
- `psi_target` — target state (normalized internally)
- `dt`         — time step (seconds)

# Returns
Gradient matrix `G[j, k] = ∂F_state/∂u_j[k]`, shape `[n_c × n_t]`.

# Derivation
Let χ = ⟨ψ_target | U_total | ψ_init⟩.  Then F_state = |χ|².

The first-order variation:

    ∂F/∂u_j[k] = 2 Re( conj(χ) * ∂χ/∂u_j[k] )

where:

    ∂χ/∂u_j[k] = ⟨ψ_target | Q[k]† (-i dt H_j) P[k] | ψ_init⟩

This follows from the same matrix-exponential differentiation as the gate case,
applied to the state vector rather than the full propagator.

# Notes
Using state vectors (dim-vectors) instead of full dim×dim matrices reduces
computation cost from O(dim³) to O(dim²) per gradient element.

# Example
```julia
G = compute_gradient_state(U_total, P, Q, sys.H_controls, psi_0, psi_1, dt)
```
"""
function compute_gradient_state(U_total::Matrix{ComplexF64},
                                 P::Array{ComplexF64,3},
                                 Q::Array{ComplexF64,3},
                                 H_controls::Vector{Matrix{ComplexF64}},
                                 psi_init::Vector{ComplexF64},
                                 psi_target::Vector{ComplexF64},
                                 dt::Real)::Matrix{Float64}
    n_t = size(P, 1) - 1
    n_c = length(H_controls)
    G   = zeros(Float64, n_c, n_t)
    return _state_gradient_into!(G, U_total, P, Q, H_controls,
                                 psi_init, psi_target, dt)
end

function _state_gradient_into!(G::Matrix{Float64},
                                U_total::Matrix{ComplexF64},
                                P::Array{ComplexF64,3},
                                Q::Array{ComplexF64,3},
                                H_controls::Vector{Matrix{ComplexF64}},
                                psi_init::Vector{ComplexF64},
                                psi_target::Vector{ComplexF64},
                                dt::Real)::Matrix{Float64}
    dt  = Float64(dt)
    n_t = size(P, 1) - 1
    n_c = length(H_controls)

    nrm_i = norm(psi_init)
    nrm_t = norm(psi_target)
    nrm_i < eps(Float64) && throw(ArgumentError("psi_init has zero norm"))
    nrm_t < eps(Float64) && throw(ArgumentError("psi_target has zero norm"))
    psi_i = psi_init   / nrm_i
    psi_t = psi_target / nrm_t

    dim      = length(psi_i)
    tmp_vec  = Vector{ComplexF64}(undef, dim)
    mul!(tmp_vec, U_total, psi_i)
    chi = dot(psi_t, tmp_vec)

    phi_k    = Vector{ComplexF64}(undef, dim)
    lambda_k = Vector{ComplexF64}(undef, dim)
    Pk_buf   = Matrix{ComplexF64}(undef, dim, dim)
    Qk_buf   = Matrix{ComplexF64}(undef, dim, dim)
    minus_i_dt = -im * dt
    @inbounds for k in 1:n_t
        @views copyto!(Pk_buf, P[k, :, :])
        @views copyto!(Qk_buf, Q[k, :, :])
        mul!(phi_k,    Pk_buf, psi_i)
        mul!(lambda_k, Qk_buf, psi_t)

        for j in 1:n_c
            Hj = H_controls[j]
            inner   = minus_i_dt * dot(lambda_k, Hj, phi_k)
            G[j, k] = 2.0 * real(conj(chi) * inner)
        end
    end
    return G
end

# ----------------------------------------------------------------------------
# Metric-aware state-transfer gradient
# ----------------------------------------------------------------------------
#
# Same kernel as `_state_gradient_into!`, but the chain-rule factor is selected
# by dispatch on `metric`. For the default `SquaredOverlap` this reproduces
# `_state_gradient_into!` exactly. For `RealOverlap`, `ModulusOverlap`,
# `LinearDMFidelity`, `ModulusDMFidelity`, `SquaredDMFidelity` the corresponding
# typed `fidelity_grad_prefactor` is used.
function compute_gradient_state(U_total::Matrix{ComplexF64},
                                 P::Array{ComplexF64,3},
                                 Q::Array{ComplexF64,3},
                                 H_controls::Vector{Matrix{ComplexF64}},
                                 psi_init::Vector{ComplexF64},
                                 psi_target::Vector{ComplexF64},
                                 dt::Real,
                                 metric::AbstractFidelityMetric)::Matrix{Float64}
    dt  = Float64(dt)
    n_t = size(P, 1) - 1
    n_c = length(H_controls)

    nrm_i = norm(psi_init)
    nrm_t = norm(psi_target)
    nrm_i < eps(Float64) && throw(ArgumentError("psi_init has zero norm"))
    nrm_t < eps(Float64) && throw(ArgumentError("psi_target has zero norm"))
    psi_i = psi_init   / nrm_i
    psi_t = psi_target / nrm_t

    dim     = length(psi_i)
    G       = zeros(Float64, n_c, n_t)
    tmp_vec = Vector{ComplexF64}(undef, dim)
    mul!(tmp_vec, U_total, psi_i)
    chi = dot(psi_t, tmp_vec)

    phi_k    = Vector{ComplexF64}(undef, dim)
    lambda_k = Vector{ComplexF64}(undef, dim)
    Pk_buf   = Matrix{ComplexF64}(undef, dim, dim)
    Qk_buf   = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_t
        @views copyto!(Pk_buf, P[k, :, :])
        @views copyto!(Qk_buf, Q[k, :, :])
        mul!(phi_k,    Pk_buf, psi_i)
        mul!(lambda_k, Qk_buf, psi_t)

        for j in 1:n_c
            Hj    = H_controls[j]
            inner = dot(lambda_k, Hj, phi_k)
            G[j, k] = fidelity_grad_prefactor(chi, inner, dt, metric)
        end
    end
    return G
end

# ============================================================================
# Finite-difference gradient (for verification)
# ============================================================================

"""
    finite_difference_gradient(system::AbstractQuantumSystem,
                                controls::ControlSequence,
                                target::QuantumTarget;
                                eps::Float64=1e-6) -> Matrix{Float64}

Compute the gradient by central finite differences (for verification of the
analytical GRAPE gradient).

# Arguments
- `system`   — quantum system
- `controls` — control sequence at which to evaluate the gradient
- `target`   — optimization target
- `eps`      — finite difference step size (default 1e-6)

# Returns
Numerical gradient matrix `G_num[j, k] ≈ ∂F/∂u_j[k]`, shape `[n_c × n_t]`.

# Formula
    G_num[j, k] = (F(u + eps * e_{jk}) - F(u - eps * e_{jk})) / (2 * eps)

where e_{jk} is the unit vector with a 1 at position (j, k) and zeros elsewhere.
This is the central difference approximation of order O(eps²).

# Notes
- Cost: 2 * n_controls * n_timesteps full propagations, each O(n_t * dim³).
  Total cost is O(n_c * n_t² * dim³) — only practical for small systems or
  gradient verification.
- Use `verify_gradient` to compare with the analytical result.

# Example
```julia
G_num = finite_difference_gradient(sys, seq, tgt; eps=1e-6)
G_ana = compute_grape_gradient(sys, seq, tgt)
verify_gradient(G_ana, G_num)
```
"""
function finite_difference_gradient(system::AbstractQuantumSystem,
                                     controls::ControlSequence,
                                     target::QuantumTarget;
                                     eps::Float64=1e-6)::Matrix{Float64}
    if eps <= 0
        throw(ArgumentError("eps must be positive, got $eps"))
    end

    n_c  = system.n_controls
    n_t  = controls.n_timesteps
    G    = zeros(Float64, n_c, n_t)
    u    = copy(controls.controls)   # n_c × n_t

    for j in 1:n_c
        for k in 1:n_t
            # Perturb forward: u_{jk} + eps
            u_plus = copy(u)
            u_plus[j, k] += eps
            seq_plus = ControlSequence(u_plus, controls.dt,
                                       controls.total_time, controls.n_timesteps)
            H_plus  = build_total_hamiltonian(system, seq_plus)
            U_plus  = compute_total_propagator(compute_propagators(H_plus, controls.dt))
            F_plus  = compute_fidelity(U_plus, target)

            # Perturb backward: u_{jk} - eps
            u_minus = copy(u)
            u_minus[j, k] -= eps
            seq_minus = ControlSequence(u_minus, controls.dt,
                                        controls.total_time, controls.n_timesteps)
            H_minus  = build_total_hamiltonian(system, seq_minus)
            U_minus  = compute_total_propagator(compute_propagators(H_minus, controls.dt))
            F_minus  = compute_fidelity(U_minus, target)

            G[j, k] = (F_plus - F_minus) / (2 * eps)
        end
    end
    return G
end

# ============================================================================
# Gradient verification
# ============================================================================

"""
    verify_gradient(analytical_grad::Matrix{Float64},
                    numerical_grad::Matrix{Float64};
                    tol::Float64=1e-5,
                    relative::Bool=true) -> Bool

Verify that an analytical gradient matches a numerical (finite-difference) gradient.

# Arguments
- `analytical_grad` — gradient from `compute_grape_gradient`, shape `[n_c × n_t]`
- `numerical_grad`  — gradient from `finite_difference_gradient`, shape `[n_c × n_t]`
- `tol`             — tolerance for agreement (default 1e-5)
- `relative`        — if `true` (default), use relative error normalized by
  `max(1, ‖numerical_grad‖_∞)`; if `false`, use absolute error

# Returns
`true` if the gradients agree within `tol`.

# Error metric
If `relative == true`:

    err = ‖G_ana - G_num‖_∞ / max(1, ‖G_num‖_∞)

If `relative == false`:

    err = ‖G_ana - G_num‖_∞

# Notes
Typical agreement for well-conditioned problems:
  - eps = 1e-6 (central diff):   err ~ 1e-6 to 1e-9
  - eps = 1e-4 (central diff):   err ~ 1e-4 (truncation dominates)

If `err > tol` the function prints a diagnostic message and returns `false`.

# Throws
- `DimensionMismatch` if the two gradient matrices have different sizes.

# Example
```julia
ok = verify_gradient(G_ana, G_num; tol=1e-5)
@assert ok "GRAPE gradient verification failed"
```
"""
function verify_gradient(analytical_grad::Matrix{Float64},
                          numerical_grad::Matrix{Float64};
                          tol::Float64=1e-5,
                          relative::Bool=true)::Bool
    if size(analytical_grad) != size(numerical_grad)
        throw(DimensionMismatch(
            "analytical_grad size $(size(analytical_grad)) ≠ " *
            "numerical_grad size $(size(numerical_grad))"))
    end

    diff = analytical_grad .- numerical_grad
    abs_err = maximum(abs.(diff))

    if relative
        scale = max(1.0, maximum(abs.(numerical_grad)))
        err = abs_err / scale
    else
        err = abs_err
    end

    if err > tol
        @warn "Gradient verification failed: error = $err > tol = $tol" *
              " (max |G_ana - G_num| = $abs_err)"
        return false
    end
    return true
end

# ============================================================================
# Higher-level gradient with optional regularization
# ============================================================================

"""
    compute_grape_gradient_regularized(system::AbstractQuantumSystem,
                                        controls::ControlSequence,
                                        target::QuantumTarget;
                                        lambda_bandwidth::Float64=0.0,
                                        lambda_amplitude::Float64=0.0)
    -> Matrix{Float64}

Compute the GRAPE gradient with optional regularization penalties.

# Arguments
- `system`, `controls`, `target` — as in `compute_grape_gradient`
- `lambda_bandwidth` — weight of bandwidth penalty term (default 0 = disabled)
- `lambda_amplitude` — weight of amplitude penalty term (default 0 = disabled)

# Returns
Total gradient `G_total = G_fidelity + G_bandwidth + G_amplitude`.

# Regularization terms
Bandwidth penalty (smoothness):

    R_bw = (λ_bw / 2) Σ_{j,k} (u_j[k+1] - u_j[k])²
    ∂R_bw/∂u_j[k] = λ_bw * ( -(u_j[k+1] - u_j[k]) + (u_j[k] - u_j[k-1]) )
                   = λ_bw * ( 2 u_j[k] - u_j[k+1] - u_j[k-1] )
    (with Neumann boundary: u_j[0] = u_j[1], u_j[n_t+1] = u_j[n_t])

Gradient of the penalty is *subtracted* (since we maximise fidelity):

    G_total = G_fidelity - ∂R_bw/∂u - ∂R_amp/∂u

Amplitude penalty (L2 regularization):

    R_amp = (λ_amp / 2) Σ_{j,k} u_j[k]²
    ∂R_amp/∂u_j[k] = λ_amp * u_j[k]

# Example
```julia
G = compute_grape_gradient_regularized(sys, seq, tgt;
        lambda_bandwidth=1e-3, lambda_amplitude=1e-4)
```
"""
function compute_grape_gradient_regularized(system::AbstractQuantumSystem,
                                             controls::ControlSequence,
                                             target::QuantumTarget;
                                             lambda_bandwidth::Float64=0.0,
                                             lambda_amplitude::Float64=0.0)::Matrix{Float64}
    G = compute_grape_gradient(system, controls, target)
    n_c, n_t = size(G)
    u = controls.controls

    # Bandwidth penalty gradient (discrete second derivative term)
    if lambda_bandwidth != 0.0
        for j in 1:n_c
            for k in 1:n_t
                u_prev = k > 1  ? u[j, k-1] : u[j, 1]
                u_next = k < n_t ? u[j, k+1] : u[j, n_t]
                # Subtract penalty gradient (penalty is minimized, fidelity maximized)
                G[j, k] -= lambda_bandwidth * (2*u[j,k] - u_next - u_prev)
            end
        end
    end

    # Amplitude L2 penalty gradient
    if lambda_amplitude != 0.0
        G .-= lambda_amplitude .* u
    end

    return G
end

# ============================================================================
# Gradient norm and diagnostics
# ============================================================================

"""
    gradient_norm(G::Matrix{Float64}) -> Float64

Compute the Frobenius norm of the gradient matrix.

    ‖G‖_F = √(Σ_{j,k} G[j,k]²)

A small gradient norm (close to zero) indicates convergence of the optimization.

# Example
```julia
gnorm = gradient_norm(G)
println("Gradient norm: \$gnorm")
```
"""
function gradient_norm(G::Matrix{Float64})::Float64
    return norm(G)
end

"""
    gradient_max_abs(G::Matrix{Float64}) -> Float64

Return the maximum absolute value entry of the gradient matrix (L∞ norm).

    ‖G‖_∞ = max_{j,k} |G[j,k]|

Useful as a convergence criterion: optimization is converged when this is below
a threshold.

# Example
```julia
if gradient_max_abs(G) < 1e-8
    println("Converged!")
end
```
"""
function gradient_max_abs(G::Matrix{Float64})::Float64
    return maximum(abs.(G))
end

# ============================================================================
# Band-selective gradient
# ============================================================================

"""
    band_selective_gradient(sys, ctrl, target, band_weights) -> Matrix{Float64}

Gradient of band_selective_fidelity:
    ∇F_band = Σ_k w_k ∇F(sys+Δk, ctrl, target)
"""
function band_selective_gradient(sys::AbstractQuantumSystem,
                                  ctrl::ControlSequence,
                                  target::QuantumTarget,
                                  band_weights::Vector{BandWeight})::Matrix{Float64}
    G = zeros(Float64, size(ctrl.controls))
    for bw in band_weights
        sys_δ = shift_system(sys, bw.offset_hz)
        G .+= bw.weight .* compute_grape_gradient(sys_δ, ctrl, target)
    end
    return G
end

# MAS/DNP/Bloch gradient functions are in Physics/MRPhysics.jl
