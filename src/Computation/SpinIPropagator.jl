# SpinIPropagator.jl — Phase-only GRAPE kernel for arbitrary spin-I systems
#
# Generalizes SU2Propagator.jl from spin-1/2 (dim=2) to any spin-I (dim = 2I+1).
#
# Physics:
#   For phase-only control with constant Rabi, the per-step propagator factors as
#     U_k(φ_n) = exp(+i φ_n Iz) · D_k · exp(-i φ_n Iz)
#   where D_k = exp(-i [δ_k Iz + Ω_RF Ix] dt) is precomputed once per ensemble member.
#
#   Matrix-vector action per step:
#     [U_k(φ_n) ψ]_m = exp(+i φ_n m) · Σ_m' D_k[m,m'] · exp(-i φ_n m') · ψ_{m'}
#
#   Per-step cost:  O(N_ens × (2I+1)²)  mat-vec with precomputed D_k
#   vs standard:   O(N_ens × (2I+1)³)  full eigendecomposition at every step
#
# Supported fidelity metrics:
#   :state      — ensemble state-transfer  (1/N_ens) Σ_k |⟨ψ_targ|ψ_k(T)⟩|²
#   :real_trace — ensemble gate fidelity   (1/(dim·N_ens)) Σ_k Re Tr(U_targ† U_k(T))
#
# Reduces exactly to SU2Propagator.jl for spin_I = 0.5.

using LinearAlgebra: mul!, dot, I

# ─────────────────────────────────────────────────────────────────────────────
# m-value helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    _spin_m_values(spin_I) → Vector{Float64}

Return the magnetic quantum numbers [+I, +I−1, …, −I] for spin quantum number `spin_I`.
These are the diagonal entries of Iz in the Zeeman basis (same ordering as `spin_Sz`).
"""
function _spin_m_values(spin_I::Real)::Vector{Float64}
    d = Int(2 * spin_I + 1)
    return [spin_I - i for i in 0:(d-1)]
end

# ─────────────────────────────────────────────────────────────────────────────
# Precomputation
# ─────────────────────────────────────────────────────────────────────────────

"""
    precompute_spinI_params(offsets_rad, RF_max_rad, dt, spin_I)
      → (Dk_matrices, m_values)

Precompute D_k = exp(-i [δ_k Iz + Ω_RF Ix] dt) for each ensemble member.
Call this ONCE before starting the optimizer.

Returns:
- `Dk_matrices` : `Array{ComplexF64,3}` of shape `[N_ens, dim, dim]`
- `m_values`    : `Vector{Float64}` of length `dim = 2*spin_I+1`, values +I to -I
"""
function precompute_spinI_params(offsets_rad::AbstractVector{<:Real},
                                 RF_max_rad::Real,
                                 dt::Real,
                                 spin_I::Real)
    N_ens = length(offsets_rad)
    dim   = Int(2 * spin_I + 1)
    m_vals = _spin_m_values(spin_I)

    # Build Ix and Iz for this spin-I (reuse QuantumSystem.jl spin operators)
    Iz_mat = ComplexF64.(diagm(m_vals))             # diagonal, m from +I to -I
    Ix_mat = spin_Sx(spin_I)                         # (S+ + S-)/2

    Dk = zeros(ComplexF64, N_ens, dim, dim)
    for k in 1:N_ens
        H_k = Float64(offsets_rad[k]) .* Iz_mat .+ RF_max_rad .* Ix_mat
        Dk[k, :, :] .= compute_propagator(ComplexF64.(H_k), Float64(dt))
    end

    return Dk, m_vals
end

"""
    precompute_spinI_params_ensemble(offsets_rad, amp_factors, RF_max_rad, dt, spin_I)
      → (Dk_matrices, m_values)

Extend precomputation to a `(offset, B1_factor)` 2-D ensemble.
`amp_factors` are dimensionless scale factors on `RF_max_rad`.
Returns Dk of shape `[N_off * N_amp, dim, dim]` (offset-major order).
"""
function precompute_spinI_params_ensemble(offsets_rad::AbstractVector{<:Real},
                                          amp_factors::AbstractVector{<:Real},
                                          RF_max_rad::Real,
                                          dt::Real,
                                          spin_I::Real)
    N_off = length(offsets_rad)
    N_amp = length(amp_factors)
    flat_offsets = repeat(collect(Float64, offsets_rad); inner=N_amp)
    flat_amps    = repeat(collect(Float64, amp_factors), N_off)
    flat_RF      = RF_max_rad .* flat_amps
    return precompute_spinI_params(flat_offsets ./ flat_amps, flat_RF, dt, spin_I)
    # Note: offset is already in rad/s; amplitude factor scales RF_max
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

"""
    spinI_forward_pass!(Psi, phi, Dk_matrices, m_values)

In-place forward propagation of an ensemble of spin-I states.

- `Psi[k, m, n]`  : state of ensemble member `k` at time step `n` (1-indexed, so n=1 is initial)
                     shape `[N_ens, dim, N_TS+1]`, pre-initialised with `Psi[:, :, 1]`.
- `phi`           : phase vector, length N_TS
- `Dk_matrices`   : precomputed propagators, shape `[N_ens, dim, dim]`
- `m_values`      : quantum numbers `[+I, ..., -I]`, length dim

After the call, `Psi[k, :, n+1]` = U_k(φ_n) · Psi[k, :, n].
"""
function spinI_forward_pass!(Psi       :: Array{ComplexF64,3},
                             phi       :: AbstractVector{Float64},
                             Dk_matrices :: Array{ComplexF64,3},
                             m_values  :: AbstractVector{Float64})
    N_TS = length(phi)
    N_ens, dim, _ = size(Dk_matrices)
    psi_rot = zeros(ComplexF64, dim)
    psi_d   = zeros(ComplexF64, dim)

    for n in 1:N_TS
        exp_neg = cis.(-phi[n] .* m_values)   # exp(-i φ_n m), length dim
        exp_pos = cis.( phi[n] .* m_values)   # exp(+i φ_n m), length dim
        for k in 1:N_ens
            psi_prev = @view Psi[k, :, n]
            @. psi_rot = exp_neg * psi_prev                          # phase
            Dk_k = @view Dk_matrices[k, :, :]
            mul!(psi_d, Dk_k, psi_rot)                               # mat-vec
            @. Psi[k, :, n+1] = exp_pos * psi_d                     # phase
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Fidelity + gradient (exact adjoint)
# ─────────────────────────────────────────────────────────────────────────────

"""
    spinI_fidelity_gradient(phi, Dk_matrices, target, m_values;
                            metric=:state, psi_init=nothing) → (F, GJ)

Compute ensemble-averaged spin-I fidelity and exact gradient.

# Arguments
- `phi`         : phase vector, length N_TS
- `Dk_matrices` : shape `[N_ens, dim, dim]` from `precompute_spinI_params`
- `target`      : for `:state` — a `(; psi_init, psi_targ)` NamedTuple of `Vector{ComplexF64}`;
                  for `:real_trace` — `Matrix{ComplexF64}` (dim × dim unitary target U_targ)
- `m_values`    : quantum numbers from `precompute_spinI_params`
- `metric`      : `:state` (state-transfer |z|²) or `:real_trace` (Re Tr(U†U)/dim)

# Returns
- `F`  : ensemble fidelity in [0, 1]
- `GJ` : gradient of J = 1 − F w.r.t. phi, length N_TS

For `metric=:state` with `spin_I=0.5` and a single-member ensemble, this reduces
to the SU(2) forward-adjoint formula in SU2Propagator.jl.
"""
function spinI_fidelity_gradient(phi        :: AbstractVector{Float64},
                                 Dk_matrices :: Array{ComplexF64,3},
                                 target,
                                 m_values   :: AbstractVector{Float64};
                                 metric      :: Symbol = :state)
    N_TS  = length(phi)
    N_ens, dim, _ = size(Dk_matrices)

    if metric === :state
        return _spinI_grad_state(phi, Dk_matrices, target.psi_init, target.psi_targ,
                                 m_values, N_TS, N_ens, dim)
    elseif metric === :real_trace
        return _spinI_grad_real_trace(phi, Dk_matrices, target, m_values, N_TS, N_ens, dim)
    else
        error("Unknown metric: $metric.  Choose :state or :real_trace.")
    end
end

# ─── :state — state-transfer gradient ────────────────────────────────────────

function _spinI_grad_state(phi, Dk_matrices, psi_init, psi_targ, m_values, N_TS, N_ens, dim)
    # Forward pass: store full trajectory Psi[k, m, n], n=1..N_TS+1
    Psi = zeros(ComplexF64, N_ens, dim, N_TS + 1)
    for k in 1:N_ens
        Psi[k, :, 1] .= psi_init
    end
    spinI_forward_pass!(Psi, phi, Dk_matrices, m_values)

    # Fidelity and terminal co-states
    N_ens_f = Float64(N_ens)
    z   = [dot(psi_targ, @view Psi[k, :, N_TS+1]) for k in 1:N_ens]
    F   = sum(abs2.(z)) / N_ens_f

    # Lambda[k, :] = terminal co-state for ensemble member k
    Lambda = zeros(ComplexF64, N_ens, dim)
    for k in 1:N_ens
        @. Lambda[k, :] = -(conj(z[k]) / N_ens_f) * psi_targ
    end

    # Precompute D_k†: conj transpose, shape [N_ens, dim, dim]
    Dk_adj = zeros(ComplexF64, N_ens, dim, dim)
    for k in 1:N_ens
        Dk_adj[k, :, :] .= Dk_matrices[k, :, :]'
    end

    GJ       = zeros(Float64, N_TS)
    lam_rot  = zeros(ComplexF64, dim)
    lam_d    = zeros(ComplexF64, dim)

    for n in N_TS:-1:1
        exp_neg = cis.(-phi[n] .* m_values)
        exp_pos = cis.( phi[n] .* m_values)

        psi_curr = @view Psi[:, :, n+1]   # ψ_k[n], after step n
        psi_prev = @view Psi[:, :, n]     # ψ_k[n-1], before step n

        # GJ[n] += Σ_k (-Im(dot(λ_k[n], m .* ψ_k[n])) + Im(dot(λ_k[n-1], m .* ψ_k[n-1])))
        # First compute term from current lambda (before backward step)
        term1 = 0.0
        for k in 1:N_ens
            term1 += imag(dot(@view(Lambda[k, :]), m_values .* @view(psi_curr[k, :])))
        end

        # Backward propagate Lambda: λ[n-1] = U_k(φ_n)† λ[n]
        # U† = exp(+i φ_n Iz) D_k† exp(-i φ_n Iz)
        for k in 1:N_ens
            lam = @view Lambda[k, :]
            @. lam_rot = exp_neg * lam                 # exp(-i φ_n Iz) λ
            Dk_adj_k = @view Dk_adj[k, :, :]
            mul!(lam_d, Dk_adj_k, lam_rot)             # D_k† (lam_rot)
            @. Lambda[k, :] = exp_pos * lam_d          # exp(+i φ_n Iz) (lam_d)
        end

        # Compute term from propagated lambda (= λ_k[n-1])
        term2 = 0.0
        for k in 1:N_ens
            term2 += imag(dot(@view(Lambda[k, :]), m_values .* @view(psi_prev[k, :])))
        end

        GJ[n] = term2 - term1
    end

    return F, GJ
end

# ─── :real_trace — gate-fidelity gradient ─────────────────────────────────────
# Run dim state-transfer problems simultaneously (propagate identity → columns of U_targ)

function _spinI_grad_real_trace(phi, Dk_matrices, U_targ, m_values, N_TS, N_ens, dim)
    N_ens_f = Float64(N_ens)
    # Forward pass: Phi_k[:, :, n] = propagator after n steps (starts from I_dim)
    # shape [N_ens, dim, dim, N_TS+1]
    Phi = zeros(ComplexF64, N_ens, dim, dim, N_TS + 1)
    for k in 1:N_ens
        Phi[k, :, :, 1] .= Matrix{ComplexF64}(I, dim, dim)
    end

    # Forward pass: same logic as spinI_forward_pass! but applied to matrices
    D_mat = zeros(ComplexF64, dim, dim)
    for n in 1:N_TS
        exp_neg = cis.(-phi[n] .* m_values)
        exp_pos = cis.( phi[n] .* m_values)
        for k in 1:N_ens
            Phi_prev = @view Phi[k, :, :, n]
            # Phase: scale each row m by exp(-i φ m) → (Iz-rotation on ket)
            for m in 1:dim
                D_mat[m, :] .= exp_neg[m] .* @view(Phi_prev[m, :])
            end
            Dk_k = @view Dk_matrices[k, :, :]
            # D_k * D_mat → replace rows of Phi_next
            mul!(@view(Phi[k, :, :, n+1]), Dk_k, D_mat)
            # Phase: scale each row m by exp(+i φ m)
            for m in 1:dim
                Phi[k, m, :, n+1] .*= exp_pos[m]
            end
        end
    end

    # Fidelity: (1/(dim*N_ens)) Σ_k Re Tr(U_targ† Phi_k[:,:,N_TS+1])
    F = 0.0
    for k in 1:N_ens
        Phi_final = @view Phi[k, :, :, N_TS + 1]
        F += real(tr(U_targ' * Phi_final))
    end
    F /= (dim * N_ens_f)

    # Terminal co-state matrices: Chi_k = -U_targ / (dim * N_ens) (same for all k)
    Chi_targ = -(U_targ ./ (dim * N_ens_f))   # dim × dim, same for all k

    # Per-ensemble co-state (broadcasted)
    Chi = zeros(ComplexF64, N_ens, dim, dim)
    for k in 1:N_ens
        Chi[k, :, :] .= Chi_targ
    end

    # Precompute D_k†
    Dk_adj = zeros(ComplexF64, N_ens, dim, dim)
    for k in 1:N_ens
        Dk_adj[k, :, :] .= Dk_matrices[k, :, :]'
    end

    GJ = zeros(Float64, N_TS)
    chi_rot = zeros(ComplexF64, dim, dim)
    chi_d   = zeros(ComplexF64, dim, dim)

    for n in N_TS:-1:1
        exp_neg = cis.(-phi[n] .* m_values)
        exp_pos = cis.( phi[n] .* m_values)

        Phi_curr = @view Phi[:, :, :, n+1]  # propagator after step n
        Phi_prev = @view Phi[:, :, :, n]    # propagator before step n

        # GJ contribution (gate analog of the state formula — trace over columns)
        # Σ_k Σ_j (-Im(dot(Chi_j_k[n], m .* Phi_j_k[n])) + Im(dot(...)))
        term1 = 0.0
        for k in 1:N_ens
            for m in 1:dim
                # column m contribution: dot of Chi column m with m_vals .* Phi column m
                chi_col = @view Chi[k, :, m]
                phi_col = @view Phi_curr[k, :, m]
                term1  += imag(dot(chi_col, m_values .* phi_col))
            end
        end

        # Backward propagate Chi: Chi[n-1] = U_k(φ_n)† Chi[n]
        # Apply U† = exp(+iφIz) D† exp(-iφIz) to each column of Chi
        for k in 1:N_ens
            Chi_k = @view Chi[k, :, :]
            # Phase each row by exp(-i φ m)
            for m in 1:dim
                chi_rot[m, :] .= exp_neg[m] .* @view(Chi_k[m, :])
            end
            Dk_adj_k = @view Dk_adj[k, :, :]
            mul!(chi_d, Dk_adj_k, chi_rot)
            # Phase each row by exp(+i φ m)
            for m in 1:dim
                Chi[k, m, :] .= exp_pos[m] .* @view(chi_d[m, :])
            end
        end

        term2 = 0.0
        for k in 1:N_ens
            for m in 1:dim
                chi_col = @view Chi[k, :, m]
                phi_col = @view Phi_prev[k, :, m]
                term2  += imag(dot(chi_col, m_values .* phi_col))
            end
        end

        GJ[n] = term2 - term1
    end

    return F, GJ
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward-only (no gradient — for derivative-free optimizers)
# ─────────────────────────────────────────────────────────────────────────────

"""
    spinI_fidelity_forward(phi, Dk_matrices, target, m_values; metric=:state) → F

Evaluate spin-I ensemble fidelity without computing the gradient.
Suitable for CMA-ES, PSO, DE, and basin-hopping optimizers.
"""
function spinI_fidelity_forward(phi        :: AbstractVector{Float64},
                                Dk_matrices :: Array{ComplexF64,3},
                                target,
                                m_values   :: AbstractVector{Float64};
                                metric      :: Symbol = :state)
    N_TS  = length(phi)
    N_ens, dim, _ = size(Dk_matrices)
    N_ens_f = Float64(N_ens)

    if metric === :state
        Psi = zeros(ComplexF64, N_ens, dim, N_TS + 1)
        for k in 1:N_ens; Psi[k, :, 1] .= target.psi_init; end
        spinI_forward_pass!(Psi, phi, Dk_matrices, m_values)
        psi_targ = target.psi_targ
        return sum(k -> abs2(dot(psi_targ, @view(Psi[k, :, N_TS+1]))), 1:N_ens) / N_ens_f

    elseif metric === :real_trace
        U_targ = target  # Matrix{ComplexF64}
        # Propagate identity matrix
        Phi = zeros(ComplexF64, N_ens, dim, dim, N_TS + 1)
        for k in 1:N_ens; Phi[k, :, :, 1] .= Matrix{ComplexF64}(I, dim, dim); end
        D_mat = zeros(ComplexF64, dim, dim)
        for n in 1:N_TS
            ep = cis.(-phi[n] .* m_values)
            eq = cis.( phi[n] .* m_values)
            for k in 1:N_ens
                Pp = @view Phi[k, :, :, n]
                for m in 1:dim; D_mat[m, :] .= ep[m] .* @view(Pp[m, :]); end
                mul!(@view(Phi[k, :, :, n+1]), @view(Dk_matrices[k, :, :]), D_mat)
                for m in 1:dim; Phi[k, m, :, n+1] .*= eq[m]; end
            end
        end
        F = 0.0
        for k in 1:N_ens
            F += real(tr(U_targ' * @view(Phi[k, :, :, N_TS+1])))
        end
        return F / (dim * N_ens_f)
    else
        error("Unknown metric: $metric.  Choose :state or :real_trace.")
    end
end
