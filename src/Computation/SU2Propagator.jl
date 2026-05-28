# SU2Propagator.jl — Cayley-Klein scalar propagator for spin-1/2 phase-only pulses
#
# For constant Rabi amplitude Ω_RF and varying phase φ_n, the per-step SU(2) propagator
# factors as P_k(φ_n) = [[α_k, -β_n_k*], [β_n_k, α_k*]] where α_k and β_s_k are
# precomputed once per optimization (offset-dependent, time-independent) and
# β_n_k = β_s_k * exp(i φ_n) costs only 4 complex muls per step per offset.
#
# Per-step cost: O(N_ens × 4) complex multiplications (vs O(dim³) for matrix-exp GRAPE).

using LinearAlgebra: dot

# ─────────────────────────────────────────────────────────────────────────────
# Precomputation
# ─────────────────────────────────────────────────────────────────────────────

"""
    precompute_su2_params(offsets_rad, RF_max_rad, dt) → (; alpha, betas)

Precompute the time-independent SU(2) scalar factors for a constant-Rabi phase-only
ensemble.  Call this once per optimization; result is valid for all pulse durations
sharing the same `dt`, `RF_max_rad`, and `offsets_rad`.

- `offsets_rad` : resonance offsets in rad/s (length N_ens)
- `RF_max_rad`  : constant peak RF amplitude in rad/s
- `dt`          : time step in seconds

Returns a NamedTuple with:
- `alpha[k]  = cos(Ω_k dt/2) − i(δ_k/Ω_k) sin(Ω_k dt/2)`
- `betas[k]  = −i(Ω_RF/Ω_k) sin(Ω_k dt/2)`

where `Ω_k = √(δ_k² + Ω_RF²)`.  Satisfies `|alpha[k]|² + |betas[k]|² = 1` for all k.
"""
function precompute_su2_params(offsets_rad::AbstractVector{<:Real},
                               RF_max_rad::Real,
                               dt::Real)
    delta = Float64.(offsets_rad)
    Omega = @. sqrt(delta^2 + RF_max_rad^2)
    half  = dt / 2
    sinO  = @. sin(Omega * half)
    cosO  = @. cos(Omega * half)
    alpha = @. ComplexF64(cosO - im * (delta / Omega) * sinO)
    betas = @. ComplexF64(-im  * (RF_max_rad / Omega) * sinO)
    return (; alpha, betas)
end

"""
    precompute_su2_params_ensemble(offsets_rad, amp_factors, RF_max_rad, dt) → (; alpha, betas)

Extend the scalar SU(2) precomputation to a 2-D ensemble of `(offset, B1_factor)` pairs.
`amp_factors` are dimensionless multiplicative factors on `RF_max_rad` (e.g. `[0.9, 1.0, 1.1]`
for ±10% B1 inhomogeneity).

The returned `alpha` and `betas` vectors have length `N_off × N_amp`, laid out in
column-major order: index `(i_off-1)*N_amp + i_amp` corresponds to offset `i_off`
and amplitude factor `i_amp`.
"""
function precompute_su2_params_ensemble(offsets_rad::AbstractVector{<:Real},
                                        amp_factors::AbstractVector{<:Real},
                                        RF_max_rad::Real,
                                        dt::Real)
    N_off = length(offsets_rad)
    N_amp = length(amp_factors)
    alpha = Vector{ComplexF64}(undef, N_off * N_amp)
    betas = Vector{ComplexF64}(undef, N_off * N_amp)
    half  = dt / 2
    idx   = 1
    for i_off in 1:N_off
        delta = Float64(offsets_rad[i_off])
        for i_amp in 1:N_amp
            Omega_RF = RF_max_rad * Float64(amp_factors[i_amp])
            Omega    = sqrt(delta^2 + Omega_RF^2)
            sinO     = sin(Omega * half)
            cosO     = cos(Omega * half)
            alpha[idx] = cosO - im * (delta / Omega) * sinO
            betas[idx] = -im  * (Omega_RF / Omega) * sinO
            idx += 1
        end
    end
    return (; alpha, betas)
end

"""
    validate_su2_params(alpha, betas; atol=1e-12) → Bool

Assert that `|alpha[k]|² + |betas[k]|² ≈ 1` for all ensemble members.
Throws `AssertionError` if the maximum residual exceeds `atol`.
"""
function validate_su2_params(alpha::AbstractVector{ComplexF64},
                             betas::AbstractVector{ComplexF64};
                             atol::Float64 = 1e-12)
    @assert length(alpha) == length(betas) "alpha and betas must have the same length"
    residuals = @. abs2(alpha) + abs2(betas) - 1.0
    max_res   = maximum(abs, residuals)
    @assert max_res < atol "SU(2) unitarity violated: max |α|²+|β|²-1 = $max_res (tol $atol)"
    return true
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

"""
    su2_forward_pass!(A, B, phi, alpha, betas)

In-place forward propagation of the Cayley-Klein amplitudes.

- `A`, `B`  : pre-allocated `[N_ens × (N_TS+1)]` ComplexF64 matrices.
              On entry `A[:,1] = 1`, `B[:,1] = 0` (identity initial condition).
- `phi`     : phase vector, length `N_TS`
- `alpha`, `betas` : precomputed scalar SU(2) factors, length `N_ens`

On return, `A[:,n+1]` and `B[:,n+1]` hold the propagated Cayley-Klein amplitudes
after `n` time steps.
"""
function su2_forward_pass!(A::Matrix{ComplexF64},
                           B::Matrix{ComplexF64},
                           phi::AbstractVector{Float64},
                           alpha::AbstractVector{ComplexF64},
                           betas::AbstractVector{ComplexF64})
    N_TS = length(phi)
    for n in 1:N_TS
        expphi = cis(phi[n])           # exp(i φ_n), scalar
        beta_n = expphi .* betas       # per-ensemble β_n_k, length N_ens
        A_prev = @view A[:, n]
        B_prev = @view B[:, n]
        @. A[:, n+1] = alpha * A_prev - conj(beta_n) * B_prev
        @. B[:, n+1] = beta_n * A_prev + conj(alpha) * B_prev
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Fidelity + gradient (exact adjoint)
# ─────────────────────────────────────────────────────────────────────────────

"""
    su2_fidelity_gradient(phi, alpha, betas, AT, BT; metric=:real_trace,
                          workspace=nothing) → (F, GJ)

Compute the ensemble-averaged SU(2) fidelity and its exact gradient with respect
to the phase vector `phi`, using the co-state (adjoint) method.

**Fidelity metrics:**
- `:real_trace` (default): `F = mean_k Re(AT* Ak(T) + BT* Bk(T))`
  Equivalent to `Re Tr(U_targ† U) / dim` for a spin-1/2 universal-rotation target.
- `:squared`: `F = mean_k |AT* Ak(T) + BT* Bk(T)|²`

**Returns:**
- `F`  : fidelity in [0, 1] (for `:real_trace`) or [0, 1] (for `:squared`)
- `GJ` : gradient of `J = 1 − F` w.r.t. `phi`; length `N_TS`.
  Pass as `grad!` argument to L-BFGS-B (it is already the gradient of the cost
  to minimise, not the negated fidelity gradient).

**`workspace`**: optional `NamedTuple(; A, B)` of pre-allocated `[N_ens × (N_TS+1)]`
ComplexF64 matrices.  Avoids allocation in tight optimizer loops.  If `nothing`,
arrays are allocated internally.
"""
function su2_fidelity_gradient(phi::AbstractVector{Float64},
                               alpha::AbstractVector{ComplexF64},
                               betas::AbstractVector{ComplexF64},
                               AT::ComplexF64,
                               BT::ComplexF64;
                               metric::Symbol = :real_trace,
                               workspace = nothing)
    N_TS  = length(phi)
    N_ens = length(alpha)

    # Allocate or reuse workspace
    if workspace === nothing
        A = zeros(ComplexF64, N_ens, N_TS + 1)
        B = zeros(ComplexF64, N_ens, N_TS + 1)
    else
        A = workspace.A
        B = workspace.B
        fill!(A, zero(ComplexF64))
        fill!(B, zero(ComplexF64))
    end
    A[:, 1] .= one(ComplexF64)

    # Forward pass
    su2_forward_pass!(A, B, phi, alpha, betas)

    Af = @view A[:, N_TS + 1]
    Bf = @view B[:, N_TS + 1]

    # Fidelity and terminal co-state (metric-dispatched)
    N_ens_f = Float64(N_ens)
    F, pA, pB = _su2_terminal(Af, Bf, AT, BT, N_ens_f, metric)

    # Backward adjoint pass → gradient GJ of J = 1 − F
    GJ = zeros(Float64, N_TS)
    _su2_backward!(GJ, A, B, phi, alpha, betas, pA, pB, N_TS)

    return F, GJ
end

# ─── metric-specific terminal fidelity and initial co-state ──────────────────

function _su2_terminal(Af, Bf, AT, BT, N_ens_f, metric::Symbol)
    if metric === :real_trace
        overlaps = @. real(conj(AT) * Af + conj(BT) * Bf)
        F  = sum(overlaps) / N_ens_f
        pA = fill(-conj(AT) / N_ens_f, length(Af))
        pB = fill(-conj(BT) / N_ens_f, length(Bf))
    elseif metric === :squared
        z  = @. conj(AT) * Af + conj(BT) * Bf
        sq = @. abs2(z)
        F  = sum(sq) / N_ens_f
        # dJ/d(A_f*) = −(1/N)(d/dA_f*)|z|² = −(1/N) z_k * conj(AT)
        pA = @. -(z * conj(AT)) / N_ens_f
        pB = @. -(z * conj(BT)) / N_ens_f
    else
        error("Unknown SU(2) fidelity metric: $metric.  Choose :real_trace or :squared.")
    end
    return F, pA, pB
end

# ─── backward co-state sweep (shared by both metrics) ────────────────────────

function _su2_backward!(GJ, A, B, phi, alpha, betas, pA, pB, N_TS)
    for n in N_TS:-1:1
        expphi = cis(phi[n])
        beta_n = expphi .* betas

        A_prev = @view A[:, n]
        B_prev = @view B[:, n]

        dA = @. -im * beta_n * conj(B_prev)
        dB = @.  im * beta_n * A_prev

        # GJ[n] = Re(sum_k conj(pA_k)*dA_k + pB_k*dB_k)
        # dot(pA, dA) = sum_k conj(pA_k)*dA_k  (Julia LinearAlgebra.dot semantics)
        GJ[n] = real(dot(pA, dA) + sum(pB .* dB))

        pA_new = @. alpha * pA + beta_n * pB
        pB_new = @. -conj(beta_n) * pA + conj(alpha) * pB
        pA .= pA_new
        pB .= pB_new
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward-only evaluation (no gradient — for derivative-free optimizers)
# ─────────────────────────────────────────────────────────────────────────────

"""
    su2_fidelity_forward(phi, alpha, betas, AT, BT; metric=:real_trace) → F

Evaluate the SU(2) ensemble fidelity for `phi` without computing the gradient.
Cheaper than `su2_fidelity_gradient` when no gradient is needed (e.g. CMA-ES, PSO, DE).
"""
function su2_fidelity_forward(phi::AbstractVector{Float64},
                              alpha::AbstractVector{ComplexF64},
                              betas::AbstractVector{ComplexF64},
                              AT::ComplexF64,
                              BT::ComplexF64;
                              metric::Symbol = :real_trace)
    N_TS  = length(phi)
    N_ens = length(alpha)
    A = zeros(ComplexF64, N_ens, N_TS + 1)
    B = zeros(ComplexF64, N_ens, N_TS + 1)
    A[:, 1] .= one(ComplexF64)
    su2_forward_pass!(A, B, phi, alpha, betas)
    Af = @view A[:, N_TS + 1]
    Bf = @view B[:, N_TS + 1]
    N_ens_f = Float64(N_ens)
    if metric === :real_trace
        return sum(@. real(conj(AT) * Af + conj(BT) * Bf)) / N_ens_f
    elseif metric === :squared
        return sum(@. abs2(conj(AT) * Af + conj(BT) * Bf)) / N_ens_f
    else
        error("Unknown SU(2) fidelity metric: $metric.  Choose :real_trace or :squared.")
    end
end
