# Application/MR/MRI/MRIOptControl.jl
# Optimal control for MRI RF+gradient co-optimization.
using LinearAlgebra

"""
    optimcon(sys::BlochSystem, M_target::Matrix{Float64},
             ctrl::MRIControlSequence;
             config, sigma_Sm, rho_kgm3, lambda_SAR, lambda_slew) -> OptimizationResult

Run GRAPE for MRI pulse design with optional SAR and slew rate penalties.
"""
function optimcon(sys::BlochSystem,
                  M_target::Matrix{Float64},
                  ctrl::MRIControlSequence;
                  config::GRAPEConfig   = GRAPEConfig(),
                  sigma_Sm::Float64     = 0.5,
                  rho_kgm3::Float64     = 1000.0,
                  lambda_SAR::Float64   = 0.0,
                  lambda_slew::Float64  = 0.0)::OptimizationResult

    pen_fns      = Function[]
    pen_grad_fns = Function[]

    if lambda_SAR > 0
        push!(pen_fns,      c -> lambda_SAR * sar_penalty(ctrl, sigma_Sm, rho_kgm3))
        push!(pen_grad_fns, c -> begin
            G_pen = zeros(5, ctrl.n_steps)
            G_pen[1:2,:] .= lambda_SAR .* sar_gradient(ctrl, sigma_Sm, rho_kgm3)
            G_pen
        end)
    end

    if lambda_slew > 0 && sys.gradient_system.max_slew_Tms < Inf
        G_max = sys.gradient_system.max_slew_Tms
        push!(pen_fns,      c -> lambda_slew * slew_rate_penalty(ctrl.G, ctrl.dt, G_max))
        push!(pen_grad_fns, c -> begin
            G_pen = zeros(5, ctrl.n_steps)
            G_pen[3:5,:] .= lambda_slew .* slew_rate_gradient(ctrl.G, ctrl.dt, G_max)
            G_pen
        end)
    end

    rho_vec = [iso.rho_0 for iso in sys.isochromats]

    fidelity_and_gradient_fn = (s, c, _) -> begin
        M_traj = bloch_forward_pass(s, ctrl)
        F      = bloch_fidelity(s, M_traj[:,:,end], M_target)
        dB1, dG = bloch_adjoint_pass(s, ctrl, M_traj, M_target, rho_vec)
        G_flat = vcat(dB1, dG)
        return F, G_flat
    end

    # Wrap the MRIControlSequence amplitudes into a ControlSequence for the hook
    # interface.  ctrl.amplitudes is [5 × n_steps] (B1x,B1y,Gx,Gy,Gz stacked),
    # i.e. the [n_controls × n_timesteps] layout — so use the 4-arg backward-compat
    # constructor, which transposes it into the internal [n_steps × n_controls]
    # layout.  The 3-arg constructor would reject it (it validates rows == n_steps).
    ctrl_seq = ControlSequence(ctrl.amplitudes, ctrl.dt, ctrl.total_time, ctrl.n_steps)

    grape_optimize(sys, nothing, ctrl_seq;
        fidelity_and_gradient_fn = fidelity_and_gradient_fn,
        penalty_fns      = pen_fns,
        penalty_grad_fns = pen_grad_fns,
        config           = config)
end
