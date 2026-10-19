# test/Optimization/EnsembleAveragingFrontEndTests.jl
# ===========================================
# Front ends of the ensemble averager. Each is pinned against the mechanism it
# generalises, on the same pulse:
#   - band_selective_fidelity / _gradient  = OffsetAxis with signed weights, :sum
#   - build_ensemble_from_systems           = DriftListAxis through ensemble_objective
#   - MRControl(…; ensemble)                = MRControl with the drifts / pwr_levels /
#                                             state pairs written out, and
#                                             ensemble_averaging on a QuantumSystem
#   - LindbladMRControl(…; ensemble)        = the same, with operator probes as
#                                             normalised vec(A) → vec(B) pairs
#   - SU2Kernel(ens, …), SpinIKernel(ens, …) = their offsets × amp_factors lists,
#                                             and ensemble_averaging
#   - grape_optimize(…; ensemble)           reports the ensemble fidelity it optimised
# plus the refusals for ensembles a front end cannot represent.

using Test
using Pulsar
using LinearAlgebra
using Random

@testset "Ensemble averaging — front ends" begin

    σx = ComplexF64[0 1; 1 0]; σy = ComplexF64[0 -im; im 0]; σz = ComplexF64[1 0; 0 -1]
    Ix, Iy, Iz = σx / 2, σy / 2, σz / 2
    rand_herm(rng, d) = (A = randn(rng, ComplexF64, d, d); (A + A') / 2)

    @testset "ensemble metric" begin
        e = EnsembleAveraging(OffsetAxis([0.0, 1.0]; operator = Iz); metric = :prop_real)
        @test e.metric === :prop_real
        @test occursin("metric = :prop_real", sprint(show, e))
        @test EnsembleAveraging().metric === :auto
        @test_throws ArgumentError EnsembleAveraging(; metric = :fancy)
        sys = quantum_system(2π * 50.0 .* Iz, [2π .* Ix, 2π .* Iy])
        ctrl = ControlSequence(200 .* randn(MersenneTwister(1), 10, 2), 1e-4, 10)
        V = exp(-im * π / 2 * σx)
        @test ensemble_averaging(e, sys, ctrl, unitary_target(V); gradient = false) ≈
              ensemble_averaging(EnsembleAveraging(OffsetAxis([0.0, 1.0]; operator = Iz)), sys, ctrl,
                                 unitary_target(V; metric = :prop_real); gradient = false) atol = 1e-14
        # the metric must exist at the level of what is scored
        @test_throws ArgumentError ensemble_averaging(
            EnsembleAveraging(; probes = state_probes(V), metric = :prop_real), sys, ctrl)
    end

    @testset "band_selective_fidelity = OffsetAxis, signed weights, :sum" begin
        rng = MersenneTwister(3)
        ss = spin_system([0.5, 0.5], [0.0 9.0; 9.0 0.0], [20.0, -35.0])
        n_t = 20
        ctrl = ControlSequence(400 .* randn(rng, n_t, ss.n_controls), 1e-3 / n_t, n_t)
        tgt = unitary_target(exp(-im * π / 2 * kron(σx, I(2))); metric = :prop_squared)
        bws = [BandWeight(0.0, 1.0), BandWeight(800.0, 0.5), BandWeight(4000.0, -0.7)]
        ens = EnsembleAveraging(OffsetAxis([b.offset_hz for b in bws]; operator = offset_operator(ss),
                                           weights = [b.weight for b in bws]); aggregate = :sum)
        F, G = ensemble_averaging(ens, ss, ctrl, tgt)
        @test F ≈ band_selective_fidelity(ss, ctrl, tgt, bws) atol = 1e-13
        @test G ≈ band_selective_gradient(ss, ctrl, tgt, bws) rtol = 1e-12
    end

    @testset "build_ensemble_from_systems = DriftListAxis via ensemble_objective" begin
        rng = MersenneTwister(4)
        Hc = [2π .* rand_herm(rng, 3) for _ in 1:2]
        Hds = [2π * 60.0 .* rand_herm(rng, 3) for _ in 1:4]
        systems = [quantum_system(H, Hc) for H in Hds]
        n_t = 12
        ctrl = ControlSequence(300 .* randn(rng, n_t, 2), 5e-4 / n_t, n_t)
        tgt = unitary_target(Matrix{ComplexF64}(qr(randn(rng, ComplexF64, 3, 3)).Q); metric = :prop_real)
        θ = vec(Matrix(ctrl.controls))
        for agg in (:mean, :worst_case, :cvar)        # cvar: α·N = 2 members exactly
            old = build_ensemble_from_systems(systems, tgt, ctrl; aggregator = agg, cvar_alpha = 0.5)
            new = ensemble_objective(EnsembleAveraging(DriftListAxis(Hds); aggregate = agg,
                                                       cvar_alpha = 0.5), systems[1], ctrl, tgt)
            Fo, Go = ensemble_value_and_grad(old, θ)
            Fn, Gn = ensemble_value_and_grad(new, θ)
            @test Fn ≈ Fo atol = 1e-13
            @test Gn ≈ Go rtol = 1e-12
            @test ensemble_value(new, θ) ≈ Fn atol = 1e-14
        end
        f, grad! = ensemble_wrap(ensemble_objective(EnsembleAveraging(DriftListAxis(Hds)), systems[1], ctrl, tgt))
        g = similar(θ); grad!(g, θ)
        F, G = ensemble_averaging(EnsembleAveraging(DriftListAxis(Hds)), systems[1], ctrl, tgt)
        @test f(θ) ≈ -F && g ≈ -vec(G)
        @test_throws ArgumentError ensemble_objective(EnsembleAveraging(), systems[1], ctrl)   # nothing to score
    end

    @testset "grape_optimize(…; ensemble)" begin
        rng = MersenneTwister(9)
        sys = quantum_system(zeros(ComplexF64, 2, 2), [Ix, Iy])            # rad/s amplitudes
        n_t = 20
        ctrl = ControlSequence(0.3 .* randn(rng, n_t, 2), 0.25, n_t)
        V = exp(-im * π / 2 * σx)
        ens = EnsembleAveraging(OffsetAxis([-0.15, 0.0, 0.15]; operator = Iz),
                                DriveScaleAxis([0.9, 1.1]))
        cfg = GRAPEConfig(max_iter = 60, step_size = 0.5)
        for tgt in (unitary_target(V; metric = :prop_squared), unitary_target(V; metric = :state_squared_haar))
            F0 = ensemble_averaging(ens, sys, ctrl, tgt; gradient = false)
            r = grape_optimize(sys, tgt, ctrl; config = cfg, ensemble = ens)
            seq = ControlSequence(Matrix(r.controls'), ctrl.dt, n_t)
            @test r.fidelity ≈ ensemble_averaging(ens, sys, seq, tgt; gradient = false) atol = 1e-12
            @test r.fidelity > F0 + 0.1
        end
        # probe pairs as the targets: the three axis states reproduce ⟨Φ^S_Q⟩_Haar
        ensp = EnsembleAveraging(ens.axes...; probes = state_probes(V; set = :axes))
        r = grape_optimize(sys, ensp, ctrl; config = cfg)
        seq = ControlSequence(Matrix(r.controls'), ctrl.dt, n_t)
        @test r.fidelity ≈ ensemble_averaging(ens, sys, seq, unitary_target(V; metric = :state_squared_haar);
                                              gradient = false) atol = 1e-12
        # refusals
        @test_throws ArgumentError grape_optimize(sys, unitary_target(V), ctrl; ensemble = ensp)
        @test_throws ArgumentError grape_optimize(sys, ens, ctrl)
        @test_throws ArgumentError grape_optimize(sys, unitary_target(V), ctrl; ensemble = ens,
                                                  fidelity_fn = (s, c, t) -> 0.0)
    end

    @testset "MRControl(…; ensemble)" begin
        rng = MersenneTwister(12)
        sys = mr_system("1H")
        Iz_, Ix_, Iy_ = spin_op(sys, :Iz), spin_op(sys, :Ix), spin_op(sys, :Iy)
        offs = [-2000.0, 0.0, 1500.0]; scales = [0.9, 1.1]; pwr = 2π * 10e3
        n_t = 40; dt = 2e-6
        common = (operators = [Ix_, Iy_], pulse_dt = fill(dt, n_t), verbose = false, backend = :cpu)
        ψi, ψt = spin_state(sys, :Iz), spin_state(sys, :mIy)
        ens = EnsembleAveraging(OffsetAxis(offs; operator = Iz_), DriveScaleAxis(scales))
        c_ens = MRControl(; drifts = [hamiltonian(sys)], rho_init = [ψi], rho_targ = [ψt],
                          pwr_levels = [pwr], ensemble = ens, common...)
        c_man = MRControl(; drifts = [hamiltonian(sys) .+ 2π * δ .* Iz_ for δ in offs],
                          rho_init = [ψi], rho_targ = [ψt], pwr_levels = pwr .* scales, common...)
        @test c_ens.drifts ≈ c_man.drifts && c_ens.pwr_levels ≈ c_man.pwr_levels
        @test c_ens.fidelity === :state_real
        w = 0.6 .* randn(rng, 2, n_t)
        Fe, Ge = grape_state_kernel(w, c_ens)
        Fm, Gm = grape_state_kernel(w, c_man)
        @test Fe ≈ Fm atol = 1e-14
        @test Ge ≈ Gm rtol = 1e-12
        # … and it is the ensemble average of the same physics on a QuantumSystem
        qs = quantum_system(hamiltonian(sys), [pwr .* Ix_, pwr .* Iy_])
        seq = ControlSequence(Matrix(w'), dt, n_t)
        Fq, Gq = ensemble_averaging(EnsembleAveraging(ens.axes...; probes = StateProbes([ψi], [ψt]),
                                                      metric = :state_real), qs, seq)
        @test Fe ≈ Fq atol = 1e-12
        # ∂F/∂w on both sides (Hc = pwr·Op). Both are first-order GRAPE gradients
        # with O(dt·‖H‖) error; the MR kernel's constant is ~4× larger (on a smooth
        # pulse at dt·Ω ≈ 0.1: 3.5% vs 0.8% against finite differences, both → 0
        # linearly in dt). So compare them at a step where that error is small.
        dtf = dt / 100
        c_f = MRControl(; drifts = [hamiltonian(sys)], rho_init = [ψi], rho_targ = [ψt],
                        pwr_levels = [pwr], ensemble = ens, common..., pulse_dt = fill(dtf, n_t))
        Gef = grape_state_kernel(w, c_f)[2]
        Gqf = ensemble_averaging(EnsembleAveraging(ens.axes...; probes = StateProbes([ψi], [ψt]),
                                                   metric = :state_real), qs,
                                 ControlSequence(Matrix(w'), dtf, n_t))[2]
        @test maximum(abs.(Gef .- Gqf)) / maximum(abs, Gqf) < 5e-3
        # StateProbes supply the pairs; the ensemble metric sets `fidelity`
        c_p = MRControl(; drifts = [hamiltonian(sys)], pwr_levels = [pwr], common...,
                        ensemble = EnsembleAveraging(ens.axes...; probes = StateProbes([ψi, ψt], [ψt, ψi]),
                                                     metric = :state_squared))
        @test length(c_p.rho_init) == 2 && c_p.fidelity === :state_squared
        # refusals
        base = (drifts = [hamiltonian(sys)], rho_init = [ψi], rho_targ = [ψt], pwr_levels = [pwr])
        mk(e; kw...) = MRControl(; base..., common..., ensemble = e, kw...)
        @test_throws ArgumentError mk(EnsembleAveraging(OffsetAxis(offs; operator = Iz_, weights = [1, 2, 1])))
        @test_throws ArgumentError mk(EnsembleAveraging(OffsetAxis(offs; operator = Iz_); aggregate = :worst_case))
        @test_throws ArgumentError mk(EnsembleAveraging(DrivePhaseAxis([0.0, 0.1])))
        @test_throws ArgumentError mk(EnsembleAveraging(OffsetAxis(offs; operator = Iz_); metric = :state_squared);
                                      fidelity = :state_real)
        @test_throws ArgumentError mk(EnsembleAveraging(; probes = StateProbes([ψi], [ψt])))   # pairs twice
        @test_throws ArgumentError mk(EnsembleAveraging(; probes = OperatorProbes([Iz_], [Iz_])))
        @test_throws ArgumentError MRControl(; drifts = [hamiltonian(sys)], pwr_levels = [pwr], common...)
    end

    @testset "LindbladMRControl(…; ensemble)" begin
        rng = MersenneTwister(14)
        sys = mr_system("1H")
        Iz_, Ix_, Iy_ = spin_op(sys, :Iz), spin_op(sys, :Ix), spin_op(sys, :Iy)
        offs = [-1500.0, 0.0, 2000.0]; scales = [0.95, 1.05]; pwr = 2π * 8e3
        n_t = 30; dt = 2e-6
        common = (operators = [Ix_, Iy_], pulse_dt = fill(dt, n_t), verbose = false, backend = :cpu)
        U = exp(-im * π .* Ix_)                                  # 180°x as a universal rotation
        probes = operator_probes(U; set = :pauli)
        ens = EnsembleAveraging(OffsetAxis(offs; operator = Iz_), DriveScaleAxis(scales);
                                probes, metric = :dm_real)
        c = LindbladMRControl(; drifts = [hamiltonian(sys)], pwr_levels = [pwr], ensemble = ens, common...)
        @test length(c.drifts) == 3 && c.pwr_levels ≈ pwr .* scales
        @test length(c.rho_init) == 3 && c.fidelity === :dm_real
        w = 0.6 .* randn(rng, 2, n_t)
        F, _ = grape_lindblad_kernel(w, c)
        qs  = quantum_system(hamiltonian(sys), [pwr .* Ix_, pwr .* Iy_])
        seq = ControlSequence(Matrix(w'), dt, n_t)
        # closed system: the Liouville kernel scores exactly the probe average
        @test F ≈ ensemble_averaging(ens, qs, seq; gradient = false) atol = 1e-12
        # with relaxation: identical to the problem written out by hand
        jops, rates = mr_relaxation(sys; T1 = 1e-3, T2star = 2e-4)
        c_r = LindbladMRControl(; drifts = [hamiltonian(sys)], pwr_levels = [pwr], ensemble = ens,
                                jump_ops = jops, decay_rates = rates, common...)
        c_h = LindbladMRControl(; drifts = [hamiltonian(sys) .+ 2π * δ .* Iz_ for δ in offs],
                                pwr_levels = pwr .* scales, fidelity = :dm_real,
                                rho_init = [vec(A ./ norm(A)) for A in probes.inputs],
                                rho_targ = [vec(B ./ norm(B)) for B in probes.targets],
                                jump_ops = jops, decay_rates = rates, common...)
        Fr, Gr = grape_lindblad_kernel(w, c_r)
        Fh, Gh = grape_lindblad_kernel(w, c_h)
        @test Fr ≈ Fh atol = 1e-14
        @test Gr ≈ Gh rtol = 1e-12
        @test abs(Fr - F) > 1e-3                                 # the jump operators act
        # StateProbes become pure states; :auto keeps LindbladMRControl's :dm_squared
        sp = StateProbes([spin_state(sys, :Iz)], [spin_state(sys, :mIz)])
        e_s = EnsembleAveraging(OffsetAxis(offs; operator = Iz_); probes = sp)
        c_s = LindbladMRControl(; drifts = [hamiltonian(sys)], pwr_levels = [pwr], ensemble = e_s, common...)
        @test c_s.fidelity === :dm_squared && length(c_s.rho_init[1]) == 2
        @test grape_lindblad_kernel(w, c_s)[1] ≈
              ensemble_averaging(e_s, qs, seq; metric = :dm_squared, gradient = false) atol = 1e-12
        # refusals
        base = (drifts = [hamiltonian(sys)], pwr_levels = [pwr])
        mk(e; kw...) = LindbladMRControl(; base..., common..., ensemble = e, kw...)
        @test_throws ArgumentError mk(EnsembleAveraging(OffsetAxis(offs; operator = Iz_); probes,
                                                         aggregate = :worst_case))
        @test_throws ArgumentError mk(EnsembleAveraging(DrivePhaseAxis([0.1]); probes))
        @test_throws ArgumentError mk(ens; fidelity = :dm_squared)                  # metric conflict
        @test_throws ArgumentError mk(ens; rho_init = [spin_state(sys, :Iz)],
                                      rho_targ = [spin_state(sys, :mIz)])             # pairs twice
        @test_throws ArgumentError mk(EnsembleAveraging(OffsetAxis(offs; operator = Iz_)))   # no pairs
    end

    @testset "SU2Kernel / SpinIKernel (ens, …)" begin
        rng = MersenneTwister(21)
        offs = [-3000.0, -500.0, 0.0, 2500.0]; scales = [0.92, 1.0, 1.08]
        RF = 8e3; n_t = 60; dt = 1e-6
        ens = EnsembleAveraging(OffsetAxis(offs; operator = Iz), DriveScaleAxis(scales))
        φ = 2π .* rand(rng, n_t)
        k_ens = SU2Kernel(ens, RF, dt, π, π / 2, 0.0)
        k_lst = SU2Kernel(offs, RF, dt, π, π / 2, 0.0; amp_factors = scales)
        @test k_ens(φ) == k_lst(φ)
        @test su2_fidelity_and_grad(k_ens, φ)[2] == su2_fidelity_and_grad(k_lst, φ)[2]
        @test SU2Kernel(EnsembleAveraging(OffsetAxis(offs; operator = Iz); metric = :prop_squared),
                        RF, dt, π, π / 2, 0.0).metric === :prop_squared
        # the SU(2) kernel is the RF field Ω(Ix cos φ + Iy sin φ) — the same physics
        qs  = quantum_system(zeros(ComplexF64, 2, 2), [2π * RF .* Ix, 2π * RF .* Iy])
        seq = ControlSequence(hcat(cos.(φ), sin.(φ)), dt, n_t)
        UT  = exp(-im * π / 2 * σx)
        @test k_ens(φ) ≈ ensemble_averaging(ens, qs, seq, unitary_target(UT; metric = :prop_real);
                                            gradient = false) atol = 1e-10
        # spin-1
        S = 1.0; Sz = Pulsar.spin_Sz(S); Sx = Pulsar.spin_Sx(S); Sy = Pulsar.spin_Sy(S)
        ensS = EnsembleAveraging(OffsetAxis(offs; operator = Sz), DriveScaleAxis(scales))
        ψ0 = ComplexF64[1, 0, 0]; ψ1 = ComplexF64[0, 0, 1]
        tgtS = (psi_init = ψ0, psi_targ = ψ1)
        kS = SpinIKernel(ensS, RF, dt, S, tgtS)
        @test kS(φ) == SpinIKernel(offs, RF, dt, S, tgtS; amp_factors = scales)(φ)
        qsS = quantum_system(zeros(ComplexF64, 3, 3), [2π * RF .* Sx, 2π * RF .* Sy])
        @test kS(φ) ≈ ensemble_averaging(ensS, qsS, seq, state_target(ψ1; psi_init = ψ0);
                                         gradient = false) atol = 1e-10
        # refusals
        @test_throws ArgumentError SU2Kernel(EnsembleAveraging(OffsetAxis(offs; operator = Ix)), RF, dt, π, π / 2, 0.0)
        @test_throws ArgumentError SU2Kernel(EnsembleAveraging(DrivePhaseAxis([0.1])), RF, dt, π, π / 2, 0.0)
        @test_throws ArgumentError SU2Kernel(EnsembleAveraging(; probes = state_probes(UT)), RF, dt, π, π / 2, 0.0)
        @test_throws ArgumentError SU2Kernel(EnsembleAveraging(OffsetAxis(offs; operator = Iz); metric = :prop_squared),
                                             RF, dt, π, π / 2, 0.0; metric = :prop_real)
        @test_throws DimensionMismatch SpinIKernel(ens, RF, dt, S, tgtS)    # spin-½ operator for spin 1
    end
end
