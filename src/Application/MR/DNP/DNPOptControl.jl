# Application/MR/DNP/DNPOptControl.jl
# Optimal control for Dynamic Nuclear Polarization (DNP).
using LinearAlgebra

"""
    optimcon_dnp(sys::DNPSpinSystem, ctrl::ControlSequence;
                 config::GRAPEConfig,
                 orientations=nothing) -> OptimizationResult

Run GRAPE to maximize nuclear polarization via DNP.
"""
function optimcon_dnp(sys::DNPSpinSystem,
                      ctrl::ControlSequence;
                      config::GRAPEConfig = GRAPEConfig(),
                      orientations::Union{Nothing,Vector{NTuple{4,Float64}}} = nothing,
                      use_lindblad::Bool = false,
                      T1_electron_s::Float64 = Inf,
                      T2_electron_s::Float64 = Inf,
                      T1_nuclear_s::Union{Float64,Vector{Float64}} = Inf,
                      cross_relax_rate::Float64 = 0.0,
                      )::OptimizationResult
    use_lindblad && return grape_dnp_lindblad_kernel(sys, ctrl;
        T1_electron_s   = T1_electron_s,
        T2_electron_s   = T2_electron_s,
        T1_nuclear_s    = T1_nuclear_s,
        cross_relax_rate = cross_relax_rate,
        config          = config,
    )
    rotate_fn = (α, β, γ) -> _rotate_dnp_system(sys, α, β, γ)
    base_fid  = (s, c, _) -> dnp_polarization_fidelity(s, c)
    base_grad = (s, c, _) -> compute_grape_gradient(s, c, Val{:DNP})

    fid_fn  = _wrap_orient_fidelity(base_fid,  orientations, rotate_fn)
    grad_fn = _wrap_orient_gradient(base_grad, orientations, rotate_fn)

    return grape_optimize(sys, nothing, ctrl;
                          fidelity_fn = fid_fn,
                          gradient_fn = grad_fn,
                          config      = config)
end

"""
    _rotate_dnp_system(sys::DNPSpinSystem, α, β, γ) -> DNPSpinSystem

Return a copy of `sys` with its rotor-modulated Fourier components rebuilt for
crystallite orientation (α,β,γ) via the shared spherical-tensor engine
(`dnp_oriented_fourier`).  The crystallite orientation is applied as the common
molecular→rotor rotation for ALL anisotropic interactions — the physically
correct operation — rather than shifting each interaction's PAS Euler angles
independently (which is only valid for a single shared-axis interaction and was
why the previous implementation produced an orientation-independent, no-op
powder average).
"""
function _rotate_dnp_system(sys::DNPSpinSystem, α::Float64, β::Float64, γ::Float64)::DNPSpinSystem
    Hf = dnp_oriented_fourier(sys, α, β, γ)
    return DNPSpinSystem(sys.S_electron, sys.I_nuclei, sys.g_vals, sys.g_euler,
                         sys.A_vals, sys.A_euler, sys.csa_electron, sys.csa_nuclei,
                         sys.dipolar, sys.omega_r, sys.rotor_period_s, Hf,
                         sys.B0_tesla, sys.mw_freq_hz, sys.rf_freq_hz,
                         sys.dim, sys.S_ops, sys.I_ops, sys.H_ctrl_mw, sys.H_ctrl_rf,
                         copy(Hf[0]), sys.H_controls, sys.n_controls,
                         sys.interactions, sys.H0)
end
