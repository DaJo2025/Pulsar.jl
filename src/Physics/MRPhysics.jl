# Physics/MRPhysics.jl
# Physics functions for MRI (Bloch), DNP, MAS solid-state NMR, and EPR systems.
# Loaded after the new type definitions (BlochSystem, DNPSpinSystem, MASSpinSystem).
using LinearAlgebra

# ============================================================================
# Bloch / MRI fidelities
# ============================================================================

"""
    bloch_fidelity(sys::BlochSystem, M_final::Matrix{Float64},
                   M_target::Matrix{Float64}) -> Float64

Weighted dot product of final and target magnetization over all isochromats.
M_final and M_target are [3 × n_iso].
"""
function bloch_fidelity(sys::BlochSystem,
                         M_final::Matrix{Float64},
                         M_target::Matrix{Float64})::Float64
    n_iso = sys.n_isochromats
    F = 0.0
    norm_sq = 0.0
    for i in 1:n_iso
        ρ = sys.isochromats[i].rho_0
        F      += ρ * dot(M_target[:,i], M_final[:,i])
        norm_sq += ρ * dot(M_target[:,i], M_target[:,i])
    end
    return norm_sq > 0 ? F / norm_sq : 0.0
end

"""
    slice_profile_fidelity(sys::BlochSystem, M_final::Matrix{Float64},
                           M_target::Matrix{Float64}) -> Float64

1 − normalized RMS error of Mz across the slice profile.
"""
function slice_profile_fidelity(sys::BlochSystem,
                                 M_final::Matrix{Float64},
                                 M_target::Matrix{Float64})::Float64
    n_iso = sys.n_isochromats
    rms = 0.0; norm_sq = 0.0
    for i in 1:n_iso
        diff = M_final[3,i] - M_target[3,i]
        rms     += diff^2
        norm_sq += M_target[3,i]^2
    end
    return 1.0 - sqrt(rms / max(norm_sq, 1e-15))
end

# ============================================================================
# MRI-specific penalties
# ============================================================================

"""
    sar_penalty(ctrl::MRIControlSequence, sigma_Sm::Float64, rho_kgm3::Float64) -> Float64

Specific Absorption Rate penalty: SAR ≈ (σ/2ρ) * (dt/T) * Σ_k |B1(t_k)|²
"""
function sar_penalty(ctrl::MRIControlSequence,
                     sigma_Sm::Float64,
                     rho_kgm3::Float64)::Float64
    return (sigma_Sm / (2*rho_kgm3)) * (ctrl.dt / ctrl.total_time) *
           sum(ctrl.B1 .^ 2)
end

"""
    sar_gradient(ctrl::MRIControlSequence, sigma_Sm, rho_kgm3) -> Matrix{Float64}

Gradient of SAR penalty w.r.t. B1 waveform.
"""
function sar_gradient(ctrl::MRIControlSequence,
                      sigma_Sm::Float64,
                      rho_kgm3::Float64)::Matrix{Float64}
    return (sigma_Sm / rho_kgm3) * (ctrl.dt / ctrl.total_time) .* ctrl.B1
end

"""
    slew_rate_penalty(G::Matrix{Float64}, dt::Float64, G_max_slew::Float64) -> Float64

Penalty on gradient slew rate exceeding G_max_slew (T/m/s).
"""
function slew_rate_penalty(G::Matrix{Float64},
                            dt::Float64,
                            G_max_slew::Float64)::Float64
    P = 0.0
    n = size(G, 2)
    for k in 1:(n-1)
        ΔG = sqrt(sum((G[j,k+1]-G[j,k])^2 for j in 1:3))
        excess = max(0.0, ΔG/dt - G_max_slew)
        P += excess^2
    end
    return P
end

"""
    slew_rate_gradient(G::Matrix{Float64}, dt::Float64, G_max_slew::Float64) -> Matrix{Float64}

Gradient of slew_rate_penalty w.r.t. G.
"""
function slew_rate_gradient(G::Matrix{Float64},
                             dt::Float64,
                             G_max_slew::Float64)::Matrix{Float64}
    grad = zeros(size(G))
    n = size(G, 2)
    for k in 1:(n-1)
        dG = G[:,k+1] .- G[:,k]
        nrm = norm(dG)
        excess = max(0.0, nrm/dt - G_max_slew)
        if excess > 0.0 && nrm > 1e-14
            c = 2.0 * excess / (nrm * dt)
            grad[:,k]   .-= c .* dG
            grad[:,k+1] .+= c .* dG
        end
    end
    return grad
end

# ============================================================================
# DNP objective
# ============================================================================

"""
    dnp_polarization_fidelity(sys::DNPSpinSystem, ctrl::ControlSequence) -> Float64

Compute normalized nuclear polarization:
    F_DNP = Tr[O_I ρ(T)] / Tr[O_I²]
"""
function dnp_polarization_fidelity(sys::DNPSpinSystem,
                                    ctrl::ControlSequence)::Float64
    ρ0 = electron_polarized_state(sys)
    OI = nuclear_polarization_operator(sys)
    den = real(tr(OI * OI))
    den < 1e-15 && return 0.0

    Us = compute_propagators(sys, ctrl)
    ρ = copy(ρ0)
    n_t = ctrl.n_timesteps
    for k in 1:n_t
        Uk = Us[k, :, :]
        ρ = Uk * ρ * Uk'
    end
    return real(tr(OI * ρ)) / den
end

# ============================================================================
# MAS GRAPE gradient
# ============================================================================

"""
    compute_grape_gradient(sys::MASSpinSystem, ctrl::ControlSequence,
                           target::QuantumTarget;
                           rotor_phase_0::Float64=0.0) -> Matrix{Float64}

GRAPE gradient for MAS solid-state NMR.
"""
function compute_grape_gradient(sys::MASSpinSystem,
                                 ctrl::ControlSequence,
                                 target::QuantumTarget;
                                 rotor_phase_0::Float64=0.0)::Matrix{Float64}
    Us = compute_propagators(sys, ctrl; rotor_phase_0=rotor_phase_0)
    return _grape_gradient_from_propagators(Us, sys.H_controls, target, ctrl.dt)
end

"""
    compute_grape_gradient_powder(sys::MASSpinSystem, ctrl, target,
                                  orientations) -> Matrix{Float64}

Powder-averaged GRAPE gradient (thread-parallel over orientations).
"""
function compute_grape_gradient_powder(sys::MASSpinSystem,
                                        ctrl::ControlSequence,
                                        target::QuantumTarget,
                                        orientations::Vector{NTuple{4,Float64}})::Matrix{Float64}
    G_total = zeros(Float64, size(ctrl.controls))
    lk = ReentrantLock()
    # Lesson 2: powder average over orientations — `@threadsif` gains BLAS-thread
    # guard around the per-orientation gradient (matches Spinach's parfor pattern
    # but without BLAS oversubscription on multi-threaded LAPACK builds).
    @threadsif true for ori in orientations
        α, β, γ, w = ori
        sys_Ω = rotate_spin_system(sys, α, β, γ)
        G_Ω   = compute_grape_gradient(sys_Ω, ctrl, target)
        lock(lk) do
            G_total .+= w .* G_Ω
        end
    end
    return G_total
end

"""
    compute_powder_fidelity(sys::MASSpinSystem, ctrl, target, orientations) -> Float64

Powder-averaged fidelity `Σ_Ω wΩ FΩ` over the **same** crystallite grid used by
[`compute_grape_gradient_powder`].  For each orientation the crystallite is
rotated (`rotate_spin_system`), the rotor-modulated step propagators are built
(`compute_propagators(::MASSpinSystem, …)`), and the total propagator is scored
with [`compute_fidelity`] (handles both unitary and state targets).

This is the fidelity partner of the powder gradient.  Pairing it as the
optimiser's `fidelity_fn` is essential: a single-orientation fidelity paired
with a powder-averaged gradient makes best-solution tracking, convergence, and
the reported fidelity inconsistent with the direction actually being optimised.
"""
function compute_powder_fidelity(sys::MASSpinSystem,
                                  ctrl::ControlSequence,
                                  target::QuantumTarget,
                                  orientations::Vector{NTuple{4,Float64}})::Float64
    return powder_average_scalar(orientations) do α, β, γ
        # rotate_spin_system preserves sys.H_controls (it no longer rebuilds via
        # mas_spin_system), so the oriented system carries the caller's scaled drive.
        sys_o = rotate_spin_system(sys, α, β, γ)
        compute_fidelity(compute_total_propagator(compute_propagators(sys_o, ctrl)), target)
    end
end

# ============================================================================
# DNP GRAPE gradient
# ============================================================================

"""
    compute_grape_gradient(sys::DNPSpinSystem, ctrl::ControlSequence,
                           ::Type{Val{:DNP}}) -> Matrix{Float64}

GRAPE gradient for DNP nuclear polarization (Pontryagin co-state recursion).
"""
function compute_grape_gradient(sys::DNPSpinSystem,
                                 ctrl::ControlSequence,
                                 ::Type{Val{:DNP}})::Matrix{Float64}
    ρ0 = electron_polarized_state(sys)
    OI = nuclear_polarization_operator(sys)
    den = real(tr(OI * OI))
    den < 1e-15 && return zeros(size(ctrl.controls))

    λN = OI / den

    Us = compute_propagators(sys, ctrl)
    n  = ctrl.n_timesteps

    ρs = Vector{Matrix{ComplexF64}}(undef, n+1)
    ρs[1] = ρ0
    for k in 1:n
        Uk = Us[k, :, :]
        ρs[k+1] = Uk * ρs[k] * Uk'
    end

    λs = Vector{Matrix{ComplexF64}}(undef, n+1)
    λs[n+1] = λN
    for k in n:-1:1
        Uk = Us[k, :, :]
        λs[k] = Uk' * λs[k+1] * Uk
    end

    G = zeros(Float64, size(ctrl.controls))
    dt = ctrl.dt
    for k in 1:n
        ρk = ρs[k]
        λk = λs[k+1]
        for (j, Hj) in enumerate(sys.H_controls)
            comm = Hj * ρk - ρk * Hj
            G[j,k] = 2 * dt * imag(tr(λk' * (-im * comm)))
        end
    end
    return G
end

# ============================================================================
# Bloch adjoint gradient
# ============================================================================

"""
    bloch_adjoint_pass(sys::BlochSystem, ctrl::MRIControlSequence,
                       M_traj::Array{Float64,3}, M_target::Matrix{Float64},
                       rho::Vector{Float64}) -> (dF_dB1, dF_dG)

Adjoint gradient of bloch_fidelity w.r.t. B1 [2×n_steps] and G [3×n_steps].
"""
function bloch_adjoint_pass(sys::BlochSystem,
                              ctrl::MRIControlSequence,
                              M_traj::Array{Float64,3},
                              M_target::Matrix{Float64},
                              rho::Vector{Float64})
    n_iso  = sys.n_isochromats
    n_t    = ctrl.n_steps
    γ      = sys.gamma
    dt     = ctrl.dt

    norm_sq = sum(rho[i] * dot(M_target[:,i], M_target[:,i]) for i in 1:n_iso)
    norm_sq = max(norm_sq, 1e-15)

    dB1 = zeros(Float64, 2, n_t)
    dG  = zeros(Float64, 3, n_t)

    for i in 1:n_iso
        iso = sys.isochromats[i]
        ρ_i = rho[i] / norm_sq
        P = ρ_i .* M_target[:,i]

        for k in n_t:-1:1
            Mx, My, Mz = M_traj[1,i,k], M_traj[2,i,k], M_traj[3,i,k]
            Px, Py, Pz = P[1], P[2], P[3]

            dB1[1,k] += -γ * dt * (Py*Mz - Pz*My)
            dB1[2,k] += -γ * dt * (Pz*Mx - Px*Mz)
            rxcross = Px*My - Py*Mx
            dG[1,k] += -γ * dt * iso.position[1] * rxcross
            dG[2,k] += -γ * dt * iso.position[2] * rxcross
            dG[3,k] += -γ * dt * iso.position[3] * rxcross

            B1x = ctrl.B1[1,k]; B1y = ctrl.B1[2,k]
            Gz_k = ctrl.G[1,k]*iso.position[1] + ctrl.G[2,k]*iso.position[2] +
                   ctrl.G[3,k]*iso.position[3] + iso.delta_B0
            B_norm = sqrt(B1x^2 + B1y^2 + Gz_k^2)
            if B_norm > 1e-15
                θ = γ * B_norm * dt
                sinθ, cosθ = sin(θ), cos(θ)
                nx, ny, nz = B1x/B_norm, B1y/B_norm, Gz_k/B_norm
                ndotP = nx*Px + ny*Py + nz*Pz
                P[1] = Px*cosθ + (ny*Pz - nz*Py)*sinθ + nx*ndotP*(1-cosθ)
                P[2] = Py*cosθ + (nz*Px - nx*Pz)*sinθ + ny*ndotP*(1-cosθ)
                P[3] = Pz*cosθ + (nx*Py - ny*Px)*sinθ + nz*ndotP*(1-cosθ)
            end
            e1 = exp(-dt/iso.T1); e2 = exp(-dt/iso.T2)
            P[1] *= e2; P[2] *= e2; P[3] *= e1
        end
    end
    return dB1, dG
end

# ============================================================================
# Internal helper: GRAPE gradient from 3D propagator array
# ============================================================================

"""
    _grape_gradient_from_propagators(Us, H_controls, target, dt) -> Matrix{Float64}

GRAPE adjoint gradient from pre-computed step propagators (3D array [n_t × dim × dim]).
"""
function _grape_gradient_from_propagators(Us::Array{ComplexF64,3},
                                           H_controls::Vector{Matrix{ComplexF64}},
                                           target::QuantumTarget,
                                           dt::Float64)::Matrix{Float64}
    n_t = size(Us, 1)
    n_c = length(H_controls)
    G   = zeros(Float64, n_c, n_t)

    P = compute_forward_propagators(Us)
    Q = compute_backward_propagators(Us)
    U_total = compute_total_propagator(Us)

    dim = size(U_total, 1)

    if target.type == "unitary" && target.target_unitary !== nothing
        U_targ = target.target_unitary
        Φ = tr(U_targ' * U_total) / dim
        for k in 1:n_t
            Pk = P[k, :, :]
            Qk = Q[k, :, :]
            for (j, Hj) in enumerate(H_controls)
                inner = tr(Qk' * (-im * dt * Hj) * Pk)
                G[j,k] = 2 * real(conj(Φ) * inner) / dim^2
            end
        end
    elseif target.type == "state" && target.target_state !== nothing
        ψ_targ = target.target_state
        ψ_init = target.initial_state === nothing ? ψ_targ : target.initial_state
        ψ_final = U_total * ψ_init
        ov = dot(ψ_targ, ψ_final)
        for k in 1:n_t
            Pk = P[k, :, :]
            Qk = Q[k, :, :]
            psi_k    = Pk * ψ_init
            lambda_k = Qk * ψ_targ
            for (j, Hj) in enumerate(H_controls)
                inner = dot(lambda_k, (-im * dt * Hj) * psi_k)
                G[j,k] = 2 * imag(conj(ov) * inner)
            end
        end
    end
    return G
end
