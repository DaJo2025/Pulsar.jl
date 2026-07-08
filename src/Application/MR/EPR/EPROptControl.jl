# Application/MR/EPR/EPROptControl.jl
# Optimal control for pulsed EPR via the generic GRAPE interface.
using LinearAlgebra

"""
    optimcon(sys::EPRSpinSystem, target, ctrl;
             config, bands, orientations) -> OptimizationResult

Run band-selective GRAPE for EPR, optionally powder-averaged.
"""
function optimcon(sys::EPRSpinSystem,
                  target::QuantumTarget,
                  ctrl::ControlSequence;
                  config::GRAPEConfig        = GRAPEConfig(),
                  bands::Vector{BandWeight}   = BandWeight[],
                  orientations::Union{Nothing,Vector{NTuple{4,Float64}}} = nothing)::OptimizationResult

    # Default path: no bands, no orientations — let grape_optimize use its built-ins.
    isempty(bands) && orientations === nothing &&
        return grape_optimize(sys, target, ctrl; config=config)

    rotate_fn = (α, β, γ) -> _epr_oriented_sys(sys, α, β, γ)

    # Pick base (non-orientation-averaged) fidelity / gradient hooks.  Reaching
    # here means bands and/or orientations are set (the no-op case returned above),
    # so when orientations !== nothing the gradient is powder-averaged.  The
    # fidelity MUST be averaged over the SAME grid, else best-solution tracking and
    # convergence judge progress from a single crystallite while the step follows
    # the powder mean.  When bands are empty we therefore supply the built-in
    # single-orientation fidelity as the base closure (matching grape_optimize's
    # own default) so _wrap_orient_fidelity averages it over the grid.
    if isempty(bands)
        base_fid  = (s, c, t) -> begin
            H_total = build_total_hamiltonian(s, c)
            U_steps = compute_propagators(H_total, c.dt)
            compute_fidelity(compute_total_propagator(U_steps), t)
        end
        base_grad = (s, c, t) -> compute_grape_gradient(s, c, t)
    else
        base_fid  = (s, c, t) -> band_selective_fidelity(s, c, t, bands)
        base_grad = (s, c, t) -> band_selective_gradient(s, c, t, bands)
    end

    grad_fn = _wrap_orient_gradient(base_grad, orientations, rotate_fn)
    fid_fn  = _wrap_orient_fidelity(base_fid, orientations, rotate_fn)
    return grape_optimize(sys, target, ctrl;
                          fidelity_fn = fid_fn,
                          gradient_fn = grad_fn,
                          config      = config)
end

function _epr_oriented_hamiltonian(sys::EPRSpinSystem, α::Float64, β::Float64, γ::Float64)::Matrix{ComplexF64}
    # Static secular EPR Hamiltonian via the shared spherical-tensor engine
    # (theta_tilt = 0 ⇒ non-spinning powder: only the time-independent m=0 lab
    # component survives).  For default g_euler/A_euler this reproduces the
    # closed-form `gzz_lab = Σ g_i n_i²` secular projection exactly, and it now
    # additionally honours non-trivial g-/A-tensor molecular-frame orientations.
    Sz = sys.S_ops[3]
    ints = SphericalInteraction[]
    ωg = ntuple(i -> _EPR_μ_B_SI * sys.B0_tesla * sys.g_vals[i] / _EPR_hbar_SI, 3)
    δg, ηg = principal_to_delta_eta(ωg)
    push!(ints, shift_interaction(sum(ωg)/3 - 2π*sys.mw_freq_hz, δg, ηg, sys.g_euler, Sz))
    for (k, Ak) in enumerate(sys.A_vals)
        a = ntuple(i -> Ak[i] * _MHZ_TO_RADS, 3)
        δA, ηA = principal_to_delta_eta(a)
        push!(ints, shift_interaction(sum(a)/3, δA, ηA, sys.A_euler[k],
                                      Matrix{ComplexF64}(Sz * sys.I_ops[k][3])))
    end
    return build_orientation_fourier(ints, (α, β, γ), sys.dim; theta_tilt = 0.0)[0]
end

function _epr_oriented_sys(sys::EPRSpinSystem, α::Float64, β::Float64, γ::Float64)::EPRSpinSystem
    H_new = _epr_oriented_hamiltonian(sys, α, β, γ)
    return EPRSpinSystem(sys.S_electron, sys.I_nuclei, sys.dim, sys.B0_tesla,
                         sys.mw_freq_hz, sys.g_vals, sys.g_euler,
                         sys.A_vals, sys.A_euler, sys.D_mhz, sys.E_mhz,
                         sys.S_ops, sys.I_ops, H_new, sys.H_controls, sys.n_controls)
end
