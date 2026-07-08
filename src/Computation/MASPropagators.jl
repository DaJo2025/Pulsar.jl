# Computation/MASPropagators.jl
# Propagator overloads for MASSpinSystem and DNPSpinSystem.
# Loaded after Types/MASSpinSystem.jl and Types/DNPSpinSystem.jl.
using LinearAlgebra

"""
    compute_propagators(sys::MASSpinSystem, ctrl::ControlSequence;
                        rotor_phase_0::Float64 = 0.0) -> Array{ComplexF64,3}

Compute step propagators for a MAS spin system.
The drift Hamiltonian is time-dependent: H_drift(t_k) = Σ_m H_m exp(im ωr t_k).
Returns 3D array of shape [n_timesteps × dim × dim].
"""
function compute_propagators(sys::MASSpinSystem,
                              ctrl::ControlSequence;
                              rotor_phase_0::Float64 = 0.0)::Array{ComplexF64,3}
    n = ctrl.n_timesteps
    dim = sys.dim
    Us = Array{ComplexF64,3}(undef, n, dim, dim)
    H_fourier = sys.H_fourier
    omega_r   = sys.omega_r
    H_tot = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n
        t_k = (k - 0.5)*ctrl.dt + rotor_phase_0/omega_r
        fill!(H_tot, zero(ComplexF64))
        for m in -2:2
            LinearAlgebra.axpy!(exp(im*m*omega_r*t_k), H_fourier[m], H_tot)
        end
        for j in eachindex(sys.H_controls)
            LinearAlgebra.axpy!(ctrl.controls[j,k], sys.H_controls[j], H_tot)
        end
        Us[k, :, :] = _expm_neg_i(H_tot, ctrl.dt)
    end
    return Us
end

"""
    compute_propagators(sys::DNPSpinSystem, ctrl::ControlSequence;
                        rotor_phase_0::Float64 = 0.0) -> Array{ComplexF64,3}

Compute step propagators for a DNP spin system.
Returns 3D array of shape [n_timesteps × dim × dim].
"""
function compute_propagators(sys::DNPSpinSystem,
                              ctrl::ControlSequence;
                              rotor_phase_0::Float64 = 0.0)::Array{ComplexF64,3}
    n = ctrl.n_timesteps
    dim = sys.dim
    Us = Array{ComplexF64,3}(undef, n, dim, dim)
    H_fourier = sys.H_fourier
    omega_r   = sys.omega_r
    # Static case (ω_r = 0, e.g. non-spinning DNP): no rotor phase, and every
    # exp(i·m·ω_r·t) = 1, so H(t) = Σ_m H_fourier[m] is time-independent.  Guard
    # the rotor-phase offset so `rotor_phase_0/omega_r` never evaluates 0/0 = NaN.
    phase0 = omega_r == 0 ? 0.0 : rotor_phase_0/omega_r
    H = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n
        t_k = (k - 0.5)*ctrl.dt + phase0
        fill!(H, zero(ComplexF64))
        for m in -2:2
            LinearAlgebra.axpy!(exp(im*m*omega_r*t_k), H_fourier[m], H)
        end
        for j in eachindex(sys.H_controls)
            LinearAlgebra.axpy!(ctrl.controls[j,k], sys.H_controls[j], H)
        end
        Us[k, :, :] = _expm_neg_i(H, ctrl.dt)
    end
    return Us
end
