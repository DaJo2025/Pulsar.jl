# TrotterPropagator.jl — Strang-split Trotter propagator for N coupled spin-1/2
#
# For N coupled spin-1/2 with phase-only control (single RF phase φ_n), the per-step
# propagator is approximated by the merged Strang (symmetric) splitting:
#
#   q[0]      = U_d2 · ψ_init
#   q[n]      = U_d  · U_ctrl(φ_n) · q[n-1]     n = 1 .. N_TS-1
#   q_final   = U_d2 · U_ctrl(φ_N) · q[N_TS-1]
#
# where:
#   U_d2 = exp(-i H_drift dt/2)   precomputed once per ensemble member
#   U_d  = exp(-i H_drift dt)     = U_d2²  (also precomputed)
#   U_ctrl(φ_n) = ⊗_k U_k(φ_n)  factors as tensor product of 2×2 SU(2) matrices
#                                  applied in O(n_spins × 2^n_spins) per step
#
# Trotter error: O(J · Ω_RF · dt³) per step.
# For J=100 Hz, Ω_RF=10 kHz, dt=1 µs: ~10⁻¹⁰ per step — negligible.
#
# Speedup vs standard GRAPE: O(2^N) per step (eliminates LAPACK eigendecomposition).
#
# Depends on: Computation/Propagators.jl (compute_propagator)

using LinearAlgebra: mul!, dot

# ─────────────────────────────────────────────────────────────────────────────
# F_z eigenvalues
# ─────────────────────────────────────────────────────────────────────────────

"""
    _trotter_fz_eigenvalues(n_spins) → Vector{Float64}

Eigenvalues of F_z = Σ_k Iz^{(k)} for each computational basis state of an
n-spin-1/2 system.  State ordering: bit k of integer index j (0-based) encodes
spin k, with 0 = spin-up (+1/2) and 1 = spin-down (-1/2), spin 0 = MSB.
"""
function _trotter_fz_eigenvalues(n_spins::Int)::Vector{Float64}
    dim = 2^n_spins
    f   = Vector{Float64}(undef, dim)
    for j in 0:(dim - 1)
        fj = 0.0
        for k in 0:(n_spins - 1)
            fj += ((j >> (n_spins - 1 - k)) & 1) == 0 ? 0.5 : -0.5
        end
        f[j + 1] = fj
    end
    return f
end

# ─────────────────────────────────────────────────────────────────────────────
# In-place 2×2 SU(2) application to a single spin-k subspace
# ─────────────────────────────────────────────────────────────────────────────

"""
    _trotter_apply_2x2!(ψ, alpha_k, beta_n, k, n_spins)

Apply [[α, −β*], [β, α*]] to spin k (0-indexed, 0=MSB) of the 2^n_spins-dim
vector ψ, in place.  Cost: O(2^n_spins).  Uses stride-based index arithmetic
so no temporaries are needed beyond the two scalars a, b per pair.
"""
@inline function _trotter_apply_2x2!(ψ       :: AbstractVector{ComplexF64},
                                     alpha_k :: ComplexF64,
                                     beta_n  :: ComplexF64,
                                     k       :: Int,
                                     n_spins :: Int)
    stride  = 2^(n_spins - 1 - k)
    n_pairs = 2^(n_spins - 1)
    bconj   = conj(beta_n)
    aconj   = conj(alpha_k)
    @inbounds for i in 0:(n_pairs - 1)
        inner = i % stride
        outer = i ÷ stride
        i0 = outer * 2 * stride + inner + 1   # spin-up  index (1-based)
        i1 = i0 + stride                        # spin-down index
        a  = ψ[i0];  b = ψ[i1]
        ψ[i0] = alpha_k * a - bconj * b
        ψ[i1] = beta_n  * a + aconj * b
    end
end

"""
    _trotter_ctrl_apply!(ψ_out, ψ_in, phi_n, ctrl_alpha, ctrl_betas, n_spins)

Apply U_ctrl(φ_n) = ⊗_k U_k(φ_n) to ψ_in, writing into ψ_out.
U_k(φ_n) = [[α_k, −β_k* e^{−iφ}], [β_k e^{iφ}, α_k*]] for each spin k.
ψ_out may alias ψ_in (copy is taken first).
"""
function _trotter_ctrl_apply!(ψ_out      :: AbstractVector{ComplexF64},
                              ψ_in       :: AbstractVector{ComplexF64},
                              phi_n      :: Float64,
                              ctrl_alpha :: AbstractVector{ComplexF64},
                              ctrl_betas :: AbstractVector{ComplexF64},
                              n_spins    :: Int)
    copyto!(ψ_out, ψ_in)
    expphi = cis(phi_n)
    for k in 0:(n_spins - 1)
        _trotter_apply_2x2!(ψ_out, ctrl_alpha[k + 1], expphi * ctrl_betas[k + 1],
                             k, n_spins)
    end
end

"""
    _trotter_ctrl_adj_apply!(ψ_out, ψ_in, phi_n, ctrl_alpha, ctrl_betas, n_spins)

Apply U_ctrl(φ_n)† = ⊗_k U_k(φ_n)† to ψ_in, writing into ψ_out.
Adjoint of [[α, −β*], [β, α*]] is [[α*, β*], [−β, α]], equivalent to
calling `_trotter_apply_2x2!` with alpha' = conj(α), beta' = −β_n.
"""
function _trotter_ctrl_adj_apply!(ψ_out      :: AbstractVector{ComplexF64},
                                  ψ_in       :: AbstractVector{ComplexF64},
                                  phi_n      :: Float64,
                                  ctrl_alpha :: AbstractVector{ComplexF64},
                                  ctrl_betas :: AbstractVector{ComplexF64},
                                  n_spins    :: Int)
    copyto!(ψ_out, ψ_in)
    expphi = cis(phi_n)
    for k in 0:(n_spins - 1)
        _trotter_apply_2x2!(ψ_out, conj(ctrl_alpha[k + 1]),
                             -(expphi * ctrl_betas[k + 1]), k, n_spins)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Precomputation
# ─────────────────────────────────────────────────────────────────────────────

"""
    precompute_trotter_params(drifts, Omega_RF_per_spin, dt, n_spins) → NamedTuple

Precompute all time-independent quantities for the Strang-Trotter GRAPE kernel.
Call once per `(drifts, Omega_RF_per_spin, dt)` combination; result is valid for
all pulse durations sharing the same `dt`.

## Arguments
- `drifts`            : `AbstractVector` of length N_ens — drift Hamiltonians (dim×dim, rad/s).
                        Build these from `hamiltonian(sys)` for each ensemble member.
- `Omega_RF_per_spin` : `AbstractVector{<:Real}` of length n_spins — peak Rabi (rad/s)
                        per spin channel.  All equal for homonuclear; distinct values
                        supported for heteronuclear single-phase control.
- `dt`                : time step in seconds
- `n_spins`           : number of coupled spin-1/2; `dim = 2^n_spins`

## Returns NamedTuple
- `U_drift_halves` : `Vector{Matrix{ComplexF64}}` — `exp(-i H_k dt/2)`, one per ensemble member
- `U_drift_full`   : `Vector{Matrix{ComplexF64}}` — `exp(-i H_k dt)`,   one per ensemble member
- `ctrl_alpha`     : `Vector{ComplexF64}` length n_spins — `cos(Ω_k dt/2)`
- `ctrl_betas`     : `Vector{ComplexF64}` length n_spins — `−i sin(Ω_k dt/2)`
- `F_z_eigs`       : `Vector{Float64}` length dim — eigenvalues of `Σ_k Iz^{(k)}`
- `n_spins`, `dim` : spin count and Hilbert-space dimension
"""
function precompute_trotter_params(drifts            :: AbstractVector,
                                   Omega_RF_per_spin :: AbstractVector{<:Real},
                                   dt                :: Real,
                                   n_spins           :: Int)
    @assert length(Omega_RF_per_spin) == n_spins "Omega_RF_per_spin must have length n_spins=$n_spins"
    N_ens = length(drifts)
    dim   = 2^n_spins
    dt_f  = Float64(dt)

    U_drift_halves = Vector{Matrix{ComplexF64}}(undef, N_ens)
    U_drift_full   = Vector{Matrix{ComplexF64}}(undef, N_ens)
    for k in 1:N_ens
        H = ComplexF64.(drifts[k])
        U_drift_halves[k] = compute_propagator(H, dt_f / 2)
        U_drift_full[k]   = compute_propagator(H, dt_f)
    end

    ctrl_alpha = Vector{ComplexF64}(undef, n_spins)
    ctrl_betas = Vector{ComplexF64}(undef, n_spins)
    for k in 1:n_spins
        Ω = Float64(Omega_RF_per_spin[k])
        ctrl_alpha[k] = ComplexF64(cos(Ω * dt_f / 2))
        ctrl_betas[k] = ComplexF64(-im * sin(Ω * dt_f / 2))
    end

    F_z_eigs = _trotter_fz_eigenvalues(n_spins)

    return (; U_drift_halves, U_drift_full, ctrl_alpha, ctrl_betas, F_z_eigs, n_spins, dim)
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward pass (merged Strang, trajectory storage for adjoint)
# ─────────────────────────────────────────────────────────────────────────────

"""
    trotter_forward_pass!(Psi_pre, Psi_post, phi, tp, psi_init)

Merged Strang-split forward pass storing states before and after each control step.

- `Psi_pre[k, :, n]`  : state of ensemble member k BEFORE U_ctrl(φ_n)
- `Psi_post[k, :, n]` : state of ensemble member k AFTER  U_ctrl(φ_n), BEFORE drift

Both arrays have shape `[N_ens, dim, N_TS]` and must be pre-allocated.
`psi_init` is the same initial state for all ensemble members.
"""
function trotter_forward_pass!(Psi_pre  :: Array{ComplexF64,3},
                               Psi_post :: Array{ComplexF64,3},
                               phi      :: AbstractVector{Float64},
                               tp       :: NamedTuple,
                               psi_init :: AbstractVector{ComplexF64})
    N_TS  = length(phi)
    N_ens = length(tp.U_drift_halves)
    dim   = tp.dim
    ψ_tmp  = zeros(ComplexF64, dim)
    ψ_ctrl = zeros(ComplexF64, dim)

    for k in 1:N_ens
        Ud2_k = tp.U_drift_halves[k]
        Ud_k  = tp.U_drift_full[k]

        mul!(ψ_tmp, Ud2_k, psi_init)            # q[0] = U_d2 · ψ_init
        Psi_pre[k, :, 1] .= ψ_tmp

        for n in 1:N_TS
            _trotter_ctrl_apply!(ψ_ctrl, @view(Psi_pre[k, :, n]),
                                 phi[n], tp.ctrl_alpha, tp.ctrl_betas, tp.n_spins)
            Psi_post[k, :, n] .= ψ_ctrl

            if n < N_TS
                mul!(ψ_tmp, Ud_k, ψ_ctrl)       # q[n] = U_d · q_c[n]
                Psi_pre[k, :, n + 1] .= ψ_tmp
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Fidelity + gradient (exact adjoint of merged Strang Trotter)
# ─────────────────────────────────────────────────────────────────────────────

"""
    trotter_fidelity_gradient(phi, tp, psi_init, psi_targ; metric=:state) → (F, GJ)

Strang-Trotter ensemble fidelity and exact adjoint gradient w.r.t. `phi`.

`metric=:state` (the only supported option) computes ensemble state-transfer fidelity:
```
F = (1/N_ens) Σ_k |⟨ψ_targ | U_k(T) | ψ_init⟩|²
```
where each U_k(T) is the merged Strang-split propagator for ensemble member k.

Returns `(F, GJ)` where `GJ` is the gradient of `J = 1 − F` w.r.t. `phi`.
Pass `GJ` directly as the `grad!` output to L-BFGS-B.
"""
function trotter_fidelity_gradient(phi      :: AbstractVector{Float64},
                                   tp       :: NamedTuple,
                                   psi_init :: AbstractVector{ComplexF64},
                                   psi_targ :: AbstractVector{ComplexF64};
                                   metric   :: Symbol = :state)
    metric === :state || error("trotter_fidelity_gradient: only :state metric supported")

    N_TS    = length(phi)
    N_ens   = length(tp.U_drift_halves)
    dim     = tp.dim
    N_ens_f = Float64(N_ens)

    # ── Forward pass ─────────────────────────────────────────────────────────
    Psi_pre  = zeros(ComplexF64, N_ens, dim, N_TS)
    Psi_post = zeros(ComplexF64, N_ens, dim, N_TS)
    trotter_forward_pass!(Psi_pre, Psi_post, phi, tp, psi_init)

    # Final states: ψ_final[k] = U_d2 · Psi_post[k,:,N_TS]
    ψ_fin_k = zeros(ComplexF64, dim)
    z        = Vector{ComplexF64}(undef, N_ens)
    for k in 1:N_ens
        mul!(ψ_fin_k, tp.U_drift_halves[k], @view(Psi_post[k, :, N_TS]))
        z[k] = dot(psi_targ, ψ_fin_k)
    end
    F = sum(abs2, z) / N_ens_f

    # ── Initialize co-states: χ_k = U_d2† · μ_k ─────────────────────────────
    Chi  = zeros(ComplexF64, N_ens, dim)
    μ_k  = zeros(ComplexF64, dim)
    for k in 1:N_ens
        @. μ_k = -(conj(z[k]) / N_ens_f) * psi_targ
        mul!(@view(Chi[k, :]), tp.U_drift_halves[k]', μ_k)
    end

    # ── Backward adjoint pass ─────────────────────────────────────────────────
    GJ    = zeros(Float64, N_TS)
    χ_tmp = zeros(ComplexF64, dim)
    fz    = tp.F_z_eigs

    for n in N_TS:-1:1
        # term1: Im⟨χ | F_z | ψ_after_ctrl[n]⟩  (χ before backward through ctrl)
        term1 = 0.0
        @inbounds for k in 1:N_ens
            χ_k = @view Chi[k, :]
            pc  = @view Psi_post[k, :, n]
            for j in 1:dim
                term1 += imag(conj(χ_k[j]) * fz[j] * pc[j])
            end
        end

        # Backward through U_ctrl(φ_n)†
        for k in 1:N_ens
            _trotter_ctrl_adj_apply!(χ_tmp, @view(Chi[k, :]),
                                     phi[n], tp.ctrl_alpha, tp.ctrl_betas, tp.n_spins)
            Chi[k, :] .= χ_tmp
        end

        # term2: Im⟨χ | F_z | ψ_before_ctrl[n]⟩  (χ after backward through ctrl)
        term2 = 0.0
        @inbounds for k in 1:N_ens
            χ_k = @view Chi[k, :]
            pp  = @view Psi_pre[k, :, n]
            for j in 1:dim
                term2 += imag(conj(χ_k[j]) * fz[j] * pp[j])
            end
        end

        GJ[n] = term2 - term1

        # Backward through U_drift (full step), except at n=1
        if n > 1
            for k in 1:N_ens
                mul!(χ_tmp, tp.U_drift_full[k]', @view(Chi[k, :]))
                Chi[k, :] .= χ_tmp
            end
        end
    end

    return F, GJ
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward-only (no gradient — for derivative-free optimizers)
# ─────────────────────────────────────────────────────────────────────────────

"""
    trotter_fidelity_forward(phi, tp, psi_init, psi_targ; metric=:state) → F

Strang-Trotter ensemble fidelity without gradient.
Suitable for CMA-ES, PSO, basin-hopping, and other derivative-free optimizers.
"""
function trotter_fidelity_forward(phi      :: AbstractVector{Float64},
                                  tp       :: NamedTuple,
                                  psi_init :: AbstractVector{ComplexF64},
                                  psi_targ :: AbstractVector{ComplexF64};
                                  metric   :: Symbol = :state)
    metric === :state || error("trotter_fidelity_forward: only :state metric supported")

    N_TS    = length(phi)
    N_ens   = length(tp.U_drift_halves)
    N_ens_f = Float64(N_ens)
    dim     = tp.dim

    ψ     = zeros(ComplexF64, dim)
    ψ_tmp = zeros(ComplexF64, dim)
    F     = 0.0

    for k in 1:N_ens
        Ud2_k = tp.U_drift_halves[k]
        Ud_k  = tp.U_drift_full[k]

        mul!(ψ, Ud2_k, psi_init)                   # q[0] = U_d2 · ψ_init

        for n in 1:N_TS
            _trotter_ctrl_apply!(ψ_tmp, ψ, phi[n],
                                 tp.ctrl_alpha, tp.ctrl_betas, tp.n_spins)
            if n < N_TS
                mul!(ψ, Ud_k, ψ_tmp)               # full drift
            else
                mul!(ψ, Ud2_k, ψ_tmp)              # half drift at end
            end
        end

        F += abs2(dot(psi_targ, ψ))
    end

    return F / N_ens_f
end
