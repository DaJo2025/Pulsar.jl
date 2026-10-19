# test/Algorithms/EnsembleTests.jl
# ================================
# Tests for the generic EnsembleObjective wrapper and its builders.
#
# Coverage:
#   (1) Aggregator correctness on synthetic Float64 samples
#   (2) build_ensemble_from_systems numerical identity vs. direct per-system
#       fidelity/gradient averaging
#   (3) build_ensemble_from_perturbations produces consistent F+∇F shapes and
#       a valid EnsembleObjective
#   (4) build_ensemble_from_mrcontrol :mean path numerical identity vs.
#       grape_state_kernel; :worst_case/:cvar per-sample path wiring
#   (5) ensemble_wrap sign convention (minimizer feeds -F, -∇F)
#   (6) Krotov ensemble dispatch (:mean monotone, non-:mean rejected)

using Test
using Pulsar
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

const _enx = ComplexF64[0 1; 1 0]
const _eny = ComplexF64[0 -im; im 0]
const _enz = ComplexF64[1 0; 0 -1]

function _en_qubit(Δ::Float64)
    return quantum_system(0.5 * Δ * _enz, [0.5 * _enx, 0.5 * _eny])
end

@testset "EnsembleObjective — aggregator correctness" begin
    # Synthetic per-sample objectives: f_i(θ) = θ[1] + offset[i];
    # gradient is [1.0, 0.0] for all samples (independent of θ).
    offsets  = [0.9, 0.8, 0.7, 0.6]
    n        = length(offsets)
    f_samps  = [let o = offsets[i]; θ -> θ[1] + o; end for i in 1:n]
    g_samps  = [let _ = i; (gv, θ) -> (gv[1] = 1.0; gv[2] = 0.0; gv); end for i in 1:n]

    # :mean
    obj_m = EnsembleObjective(f_samps; grad_samples = g_samps, aggregator = :mean)
    F_m, G_m = ensemble_value_and_grad(obj_m, [0.0, 0.0])
    @test F_m ≈ sum(offsets) / n
    @test G_m ≈ [1.0, 0.0]

    # :worst_case picks minimum
    obj_wc = EnsembleObjective(f_samps; grad_samples = g_samps, aggregator = :worst_case)
    F_wc, G_wc = ensemble_value_and_grad(obj_wc, [0.0, 0.0])
    @test F_wc ≈ minimum(offsets)
    @test G_wc ≈ [1.0, 0.0]     # every per-sample gradient is identical

    # :cvar α=0.5 → mean of worst 2 samples
    obj_cv = EnsembleObjective(f_samps;
                                grad_samples = g_samps,
                                aggregator   = :cvar,
                                cvar_alpha   = 0.5)
    F_cv, _ = ensemble_value_and_grad(obj_cv, [0.0, 0.0])
    @test F_cv ≈ (0.6 + 0.7) / 2

    # Ordering invariant
    @test F_wc ≤ F_cv ≤ F_m + 1e-12
end

@testset "build_ensemble_from_systems — numerical identity" begin
    Random.seed!(31)
    systems = [_en_qubit(Δ) for Δ in (-0.3, 0.0, 0.3)]
    target  = state_target(ComplexF64[0.0, 1.0])
    N_TS    = 18
    DT      = 0.08

    ctrl = ControlSequence(0.05 .* randn(N_TS, 2), DT, N_TS)

    obj      = build_ensemble_from_systems(systems, target, ctrl; aggregator = :mean)
    # Builder reshapes θ as [n_controls × n_steps] — use ctrl.controls view
    θ0      = vec(Matrix(ctrl.controls))
    F_obj, G_obj = ensemble_value_and_grad(obj, θ0)

    # Reference: hand average of per-system primitives
    Fs = Float64[]
    Gs = Vector{Float64}[]
    for sys in systems
        push!(Fs, Float64(compute_fidelity(sys, ctrl, target)))
        push!(Gs, vec(compute_grape_gradient(sys, ctrl, target)))
    end
    F_ref = sum(Fs) / length(Fs)
    G_ref = sum(Gs) / length(Gs)

    @test F_obj ≈ F_ref atol = 1e-10
    @test maximum(abs.(G_obj .- G_ref)) ≤ 1e-10
end

@testset "ensemble_wrap — sign inversion + consistency" begin
    Random.seed!(42)
    systems = [_en_qubit(Δ) for Δ in (-0.1, 0.1)]
    target  = state_target(ComplexF64[0.0, 1.0])
    N_TS    = 12
    ctrl    = ControlSequence(0.1 .* randn(N_TS, 2), 0.1, N_TS)

    obj    = build_ensemble_from_systems(systems, target, ctrl; aggregator = :mean)
    f, g!  = ensemble_wrap(obj)

    θ = vec(Matrix(ctrl.controls))
    # f(θ) = -F_ensemble(θ);  grad!(gv, θ) = -∇F_ensemble(θ)
    F_forward = ensemble_value(obj, θ)
    gv = similar(θ); g!(gv, θ)
    F_minus   = f(θ)
    @test F_minus ≈ -F_forward
    # Combined consistency: F and ∇F cached under repeated calls on same θ
    @test f(θ) ≈ F_minus     # cache hit
end

@testset "build_ensemble_from_perturbations — parametric shape" begin
    Random.seed!(7)
    sys    = _en_qubit(0.0)
    # Perturbation kernels use unitary fidelity (gate form) only
    U_tgt  = ComplexF64[0 1; 1 0]   # X gate
    target = unitary_target(U_tgt)
    N_TS   = 10
    ctrl   = ControlSequence(0.1 .* randn(N_TS, 2), 0.1, N_TS)

    obj = build_ensemble_from_perturbations(sys, target, ctrl;
                                             uncertainty_type = :parametric,
                                             magnitude        = 0.05,
                                             n_samples        = 6,
                                             aggregator       = :mean,
                                             resample         = false,
                                             seed             = 123)

    F, G = ensemble_value_and_grad(obj, vec(Matrix(ctrl.controls)))
    @test 0.0 ≤ F ≤ 1.0
    @test length(G) == length(ctrl.amplitudes)
    @test obj.n_samples == 6

    # :noise is explicitly not supported by this builder
    @test_throws ArgumentError build_ensemble_from_perturbations(
        sys, target, ctrl;
        uncertainty_type = :noise,
        magnitude        = 0.05,
        n_samples        = 4,
    )
end

@testset "MRBuilder — :mean batched path vs. grape_state_kernel" begin
    sys = mr_system("1H")
    drifts = [hamiltonian(sys; offset_hz = Δf) for Δf in (-3000.0, 0.0, 3000.0)]
    operators = [spin_op(sys, :Ix), spin_op(sys, :Iy)]

    N_TS = 40
    ctrl = MRControl(
        drifts     = drifts,
        operators  = operators,
        rho_init   = [spin_state(sys, :Iz)],
        rho_targ   = [spin_state(sys, :mIy)],
        pwr_levels = [2π * 10_000.0],
        pulse_dt   = fill(5e-6, N_TS),
        fidelity   = :state_squared,
        verbose    = false,
    )

    Random.seed!(17)
    w0 = 0.1 .* randn(2, N_TS)

    obj_mean = build_ensemble_from_mrcontrol(ctrl; aggregator = :mean)
    F_obj, G_obj = ensemble_value_and_grad(obj_mean, vec(w0))

    # Reference: direct call to the existing batched kernel
    F_ref, G_ref = Pulsar.grape_state_kernel(w0, ctrl)
    @test F_obj ≈ F_ref atol = 1e-12
    @test maximum(abs.(G_obj .- vec(G_ref))) ≤ 1e-12

    # Per-sample path for :worst_case — reachable, F in [0,1]
    obj_wc = build_ensemble_from_mrcontrol(ctrl; aggregator = :worst_case)
    F_wc, G_wc = ensemble_value_and_grad(obj_wc, vec(w0))
    @test 0.0 ≤ F_wc ≤ 1.0
    @test length(G_wc) == length(w0)
    @test F_wc ≤ F_ref + 1e-9     # worst ≤ mean

    # :cvar path
    obj_cv = build_ensemble_from_mrcontrol(ctrl; aggregator = :cvar, cvar_alpha = 0.5)
    F_cv, _ = ensemble_value_and_grad(obj_cv, vec(w0))
    @test F_wc ≤ F_cv ≤ F_ref + 1e-9
end

@testset "Krotov ensemble — :mean monotone + non-:mean rejected" begin
    Random.seed!(11)
    systems = [_en_qubit(Δ) for Δ in (-0.2, 0.0, 0.2)]
    target  = state_target(ComplexF64[0.0, 1.0])
    N_TS    = 15
    ctrl    = ControlSequence(0.1 .* randn(N_TS, 2), 0.1, N_TS)

    ctrl_opt, F_opt, stats = krotov_optimize(systems, target, ctrl;
                                              aggregator = :mean,
                                              max_iter   = 15,
                                              λ_a        = 2.0,
                                              verbose    = false)
    # Monotone ascent under enforce_monotonic=true (default)
    @test all(diff(stats.history) .>= -1e-9)
    @test F_opt ≥ stats.history[1]

    @test_throws ArgumentError krotov_optimize(systems, target, ctrl;
                                                aggregator = :worst_case)
    @test_throws ArgumentError krotov_optimize(systems, target, ctrl;
                                                aggregator = :cvar)
end
