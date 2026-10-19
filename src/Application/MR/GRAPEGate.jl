# ═══════════════════════════════════════════════════════════════════════════
# GRAPEGate.jl — propagator-level GATE fidelity kernel for the MR layer.
#
# Motivation: until now the MR layer optimized unitary targets by expanding
# them into a 3-probe universal-rotation state ensemble (|0⟩, |1⟩, |+⟩ →
# U|ψ⟩ pairs through grape_state_kernel). That is mathematically exact for
# SU(2) up to global phase, but (a) costs 3 forward/backward propagations per
# ensemble member instead of 1, (b) hides the global-phase profile Φ₀(ν)
# that phase-conditioned pulse design (e.g. refocusing) relies on, and
# (c) can't express the Kobzar-style quadratic objective used there.
#
# This kernel scores the gate directly on the accumulated propagator:
#
#     Φ₀(k) = Re Tr(U_F† · U_k(T)) / dim          (the "real gate fidelity")
#     z(k)  =    Tr(U_F† · U_k(T)) / dim
#
# per ensemble member k (drifts × pwr_levels), with four metrics:
#
#     :prop_real       F = mean_k Φ₀(k)            Φ^P_R, phase-SENSITIVE, max 1
#     :prop_real_quad  F = 1 − mean_k (1 − Φ₀(k))² Kobzar target-referenced
#                                                  quadratic form of Φ^P_R
#                                                  (shifted so max = 1 and
#                                                  optimcon-style convergence
#                                                  checks work)
#     :prop_squared    F = mean_k |z(k)|²          Φ^P_Q, phase-INVARIANT, max 1
#     :prop_modulus    F = mean_k |z(k)|           Φ^P_M, phase-INVARIANT, max 1
#
# Gradient: standard first-order GRAPE (same accuracy class as
# grape_state_kernel — see its docstring). With B_n = P_N⋯P_{n+1} and
# U_{n+1} = P_n⋯P_1:
#
#     ∂U(T)/∂w[c,n] ≈ B_n · (−i·dt_n·pwr·Op_c) · U_{n+1}
#     ∂F/∂w[c,n]    = dt_n·pwr · Im( χ_k · Tr(Λ_n† · Op_c · U_{n+1}) ) / N_ens
#
# where Λ_n = B_n† U_F is the backward-propagated target and χ_k is the
# metric-specific terminal weight:
#
#     :prop_real       χ = 1/dim
#     :prop_real_quad  χ = 2(1 − Φ₀(k))/dim
#     :prop_squared    χ = 2·conj(z(k))/dim
#     :prop_modulus    χ = conj(z(k))/(|z(k)|·dim)      (0 at the |z| = 0 kink)
#
# The GLOBAL PHASE BRANCH (±U_F both lie in SU(2) for det-corrected targets)
# is the caller's responsibility: pass −U_F for the NEG branch. Phase-
# invariant :prop_squared is identical for both.
# ═══════════════════════════════════════════════════════════════════════════

export grape_gate_kernel

# Metrics the gate kernel implements (see the file header).
const GATE_KERNEL_METRICS = (:prop_real, :prop_real_quad, :prop_squared, :prop_modulus)

"""
    grape_gate_kernel(waveform, drifts, operators, pwr_levels, pulse_dt,
                      U_target; metric = :prop_real) → (F, grad)

Direct propagator-level gate-fidelity objective + first-order GRAPE gradient
for the MR layer. `waveform` is `[n_ctrl × n_t]` normalized amplitudes (1.0 =
full drive); the physical Hamiltonian per step is
`H_n = drift + pwr · Σ_c waveform[c,n] · operators[c]`.

The ensemble is `drifts × pwr_levels` (same convention as `MRControl`);
`F` and `grad` are ensemble means. `metric` is `:prop_real` (Φ^P_R, default),
`:prop_real_quad`, `:prop_squared` (Φ^P_Q) or `:prop_modulus` (Φ^P_M); see the
file header for definitions.
"""
function grape_gate_kernel(waveform::Matrix{Float64},
                           drifts::Vector{<:AbstractMatrix},
                           operators::Vector{<:AbstractMatrix},
                           pwr_levels::Vector{Float64},
                           pulse_dt::Vector{Float64},
                           U_target::Matrix{ComplexF64};
                           metric::Symbol = :prop_real)
    metric = canonical_metric(metric, GATE_KERNEL_METRICS;
                              context = "grape_gate_kernel")
    n_ctrl, n_t = size(waveform)
    n_ctrl == length(operators) ||
        throw(ArgumentError("waveform rows $(n_ctrl) ≠ length(operators) $(length(operators))"))
    n_t == length(pulse_dt) ||
        throw(ArgumentError("waveform cols $(n_t) ≠ length(pulse_dt) $(length(pulse_dt))"))
    dim = size(U_target, 1)
    size(U_target) == (dim, dim) ||
        throw(ArgumentError("U_target must be square"))
    all(d -> size(d) == (dim, dim), drifts) ||
        throw(ArgumentError("every drift must match U_target dim $dim"))

    ops   = [Matrix{ComplexF64}(op) for op in operators]
    U_F   = Matrix{ComplexF64}(U_target)
    ens   = [(Matrix{ComplexF64}(d), p) for d in drifts for p in pwr_levels]
    N_ens = length(ens)

    fid_buf  = zeros(Float64, N_ens)
    grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:N_ens]

    # Threading over ensemble members; single-threaded BLAS inside (mirrors
    # grape_state_kernel's strategy so the two kernels co-exist cleanly).
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads :static for idx in 1:N_ens
            H_drift, pwr = ens[idx]
            grad_local   = grad_buf[idx]

            # ── Propagators per step ─────────────────────────────────────
            Ps    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_t]
            H_buf = Matrix{ComplexF64}(undef, dim, dim)
            VDbuf = Matrix{ComplexF64}(undef, dim, dim)
            for n in 1:n_t
                H_buf .= H_drift
                for c in 1:n_ctrl
                    @. H_buf += pwr * waveform[c, n] * ops[c]
                end
                _expm_neg_i_into!(Ps[n], H_buf, pulse_dt[n], VDbuf)
            end

            # ── Forward accumulated propagators U_{n+1} = P_n ⋯ P_1 ──────
            Us = Vector{Matrix{ComplexF64}}(undef, n_t + 1)
            Us[1] = Matrix{ComplexF64}(I, dim, dim)
            for n in 1:n_t
                Us[n + 1] = Ps[n] * Us[n]
            end
            U_T = Us[n_t + 1]

            # ── Terminal overlap + metric ────────────────────────────────
            z    = tr(U_F' * U_T) / dim
            Φ₀   = real(z)
            F_k, χ = if metric === :prop_real
                (Φ₀,                ComplexF64(1.0 / dim))
            elseif metric === :prop_real_quad
                (1.0 - (1.0 - Φ₀)^2, ComplexF64(2.0 * (1.0 - Φ₀) / dim))
            elseif metric === :prop_modulus
                az = abs(z)
                (az, az > 1e-12 ? conj(z) / (az * dim) : ComplexF64(0))
            else # :prop_squared
                (abs2(z),           2.0 * conj(z) / dim)
            end
            fid_buf[idx] = F_k

            # ── Backward-propagated target Λ_n = (P_N⋯P_{n+1})† U_F ──────
            Λ = copy(U_F)                       # Λ at n = n_t
            tmp = Matrix{ComplexF64}(undef, dim, dim)
            for n in n_t:-1:1
                dtp = pulse_dt[n] * pwr
                Uin = Us[n + 1]
                for c in 1:n_ctrl
                    mul!(tmp, ops[c], Uin)      # tmp = Op_c · U_{n+1}
                    # Tr(Λ† · tmp) = dot(vec(Λ), vec(tmp))
                    inner = dot(Λ, tmp)
                    grad_local[c, n] += dtp * imag(χ * inner)
                end
                # Λ_{n-1} = P_n† Λ_n
                if n > 1
                    Λ = Ps[n]' * Λ
                end
            end
        end
    finally
        BLAS.set_num_threads(old_blas)
    end

    F    = sum(fid_buf) / N_ens
    grad = zeros(Float64, n_ctrl, n_t)
    for g in grad_buf
        grad .+= g
    end
    grad ./= N_ens
    return F, grad
end
