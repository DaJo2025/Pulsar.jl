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

# Metrics the SU(2) kernel implements (see `su2_fidelity_gradient`): the
# propagator row (z_P = Tr(U_t†U)/2 = Re z exactly in SU(2)) and the state row
# of the |↑⟩ probe (z itself).
const SU2_METRICS = (:prop_real, :prop_modulus, :prop_squared, :prop_average,
                     :prop_real_quad, :state_real, :state_modulus, :state_squared)

_su2_metric(metric::Symbol) =
    metric in SU2_METRICS ? metric :
        canonical_metric(metric, SU2_METRICS; context = "the SU(2) kernel")

"""
    su2_fidelity_gradient(phi, alpha, betas, AT, BT; metric=:prop_real,
                          workspace=nothing) → (F, GJ)

Compute the ensemble-averaged SU(2) fidelity and its exact gradient with respect
to the phase vector `phi`, using the co-state (adjoint) method.

**Fidelity metrics** (`z_k = AT* Ak(T) + BT* Bk(T) = ⟨↑|U_targ† U_k|↑⟩`). In
SU(2) the propagator overlap is exactly `Tr(U_targ† U_k)/2 = Re z_k`, so:
- propagator row on `Re z_k`: `:prop_real` (Φ^P_R, default), `:prop_modulus`
  (Φ^P_M), `:prop_squared` (Φ^P_Q = (Re z_k)²), `:prop_average`
  ((2Φ^P_Q + 1)/3), `:prop_real_quad` (1 − (1 − Φ^P_R)², the Kobzar form);
- state row of the |↑⟩ probe on `z_k`: `:state_real` (= Φ^P_R here),
  `:state_modulus`, `:state_squared` (|⟨U_targ ↑|U_k ↑⟩|², not Φ^P_Q).
`F` is the ensemble mean; every metric goes through `overlap_fidelity` /
`overlap_weight`, so the adjoint is exact for all of them.

The pre-Φ^X_Y names `:real_trace` and `:squared` are accepted with a
deprecation warning.

**Returns:**
- `F`  : fidelity — in [−1, 1] for the real metrics, [0, 1] otherwise
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
                               metric::Symbol = :prop_real,
                               workspace = nothing)
    metric = _su2_metric(metric)
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
    if has_constant_weight(metric)          # :prop_real, :state_real (w ≡ 1)
        overlaps = @. real(conj(AT) * Af + conj(BT) * Bf)
        F  = sum(overlaps) / N_ens_f
        pA = fill(-conj(AT) / N_ens_f, length(Af))
        pB = fill(-conj(BT) / N_ens_f, length(Bf))
    else
        # Any other metric (:state_squared included — its co-state used to read
        # −z·conj(AT)/N, conjugating z the wrong way and dropping the factor 2,
        # so its gradient disagreed with finite differences at O(1); the weight
        # 2·conj(z) from overlap_weight is the fix):
        # dF = Re(w · dz_m) with w = overlap_weight(metric, z_m).
        # On the propagator row z_m = Re z, dz_m = Re(dz) and w is real, so
        # Re(w · Re(dz)) = Re(w · dz) and the same co-state p = −(w/N)·conj(T)
        # serves both rows.
        z  = @. conj(AT) * Af + conj(BT) * Bf
        _metric_level(metric) === :prop && (z = complex.(real.(z)))
        F  = sum(overlap_fidelity.(metric, z; dim = 2)) / N_ens_f
        cz = overlap_weight.(metric, z; dim = 2) ./ N_ens_f
        pA = @. -cz * conj(AT)
        pB = @. -cz * conj(BT)
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
    su2_fidelity_forward(phi, alpha, betas, AT, BT; metric=:prop_real) → F

Evaluate the SU(2) ensemble fidelity for `phi` without computing the gradient.
Cheaper than `su2_fidelity_gradient` when no gradient is needed (e.g. CMA-ES, PSO, DE).
"""
function su2_fidelity_forward(phi::AbstractVector{Float64},
                              alpha::AbstractVector{ComplexF64},
                              betas::AbstractVector{ComplexF64},
                              AT::ComplexF64,
                              BT::ComplexF64;
                              metric::Symbol = :prop_real)
    metric = _su2_metric(metric)
    N_TS  = length(phi)
    N_ens = length(alpha)
    A = zeros(ComplexF64, N_ens, N_TS + 1)
    B = zeros(ComplexF64, N_ens, N_TS + 1)
    A[:, 1] .= one(ComplexF64)
    su2_forward_pass!(A, B, phi, alpha, betas)
    Af = @view A[:, N_TS + 1]
    Bf = @view B[:, N_TS + 1]
    N_ens_f = Float64(N_ens)
    if has_constant_weight(metric)          # :prop_real, :state_real
        return sum(@. real(conj(AT) * Af + conj(BT) * Bf)) / N_ens_f
    elseif metric === :state_squared
        return sum(@. abs2(conj(AT) * Af + conj(BT) * Bf)) / N_ens_f
    else
        z = @. conj(AT) * Af + conj(BT) * Bf
        _metric_level(metric) === :prop && (z = complex.(real.(z)))
        return sum(overlap_fidelity.(metric, z; dim = 2)) / N_ens_f
    end
end
