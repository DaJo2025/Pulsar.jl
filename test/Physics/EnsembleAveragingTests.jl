# test/Physics/EnsembleAveragingTests.jl
# ===========================================
# The global ensemble averager (Types/EnsembleAveraging.jl, Computation/
# EnsembleProbes.jl, Physics/EnsembleAveraging.jl). Pins:
#   (a) the spec: members, weights, :product / :zip, validation;
#   (b) the input-state axis identities that relate probe averages to
#       propagator-level metrics (2-designs, orthonormal bases, operator bases)
#       and the phase blindness of the computational basis;
#   (c) a one-member ensemble reproduces compute_fidelity / compute_grape_gradient
#       for every gate and state metric;
#   (d) every axis against members built by hand, and every aggregator's
#       derivative against finite differences;
#   (e) the full gradient (axes × probes × aggregators) against central finite
#       differences, at a dt small enough for first-order GRAPE.

using Test
using Pulsar
using LinearAlgebra
using Random
using Statistics

@testset "Ensemble averaging" begin

    σx = ComplexF64[0 1; 1 0]; σy = ComplexF64[0 -im; im 0]; σz = ComplexF64[1 0; 0 -1]
    Ix, Iy, Iz = σx / 2, σy / 2, σz / 2

    function fd(f, x; h = 1e-6)
        g = similar(x, Float64)
        for i in eachindex(x)
            xp = copy(x); xm = copy(x); xp[i] += h; xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2h)
        end
        return g
    end
    relerr(G, Gref) = maximum(abs.(G .- Gref)) / max(maximum(abs, Gref), 1e-14)
    # :basis probes warn (maxlog = 1). Base.CoreLogging is what Logging
    # re-exports — Logging itself is not a test dependency.
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

    rand_unitary(rng, d) = (Q = qr(randn(rng, ComplexF64, d, d)).Q; Matrix{ComplexF64}(Q))
    rand_herm(rng, d)    = (A = randn(rng, ComplexF64, d, d); (A + A') / 2)

    # A random pulse on a d-level system with n_c random Hermitian controls;
    # amplitudes in Hz, controls carry the 2π.
    function random_problem(rng, d, n_c, n_t; T = 1e-3, amp = 400.0)
        Hd = 2π * 100.0 .* rand_herm(rng, d)
        Hc = [2π .* rand_herm(rng, d) for _ in 1:n_c]
        sys = quantum_system(Hd, Hc)
        ctrl = ControlSequence(amp .* randn(rng, n_t, n_c), T / n_t, n_t)
        return sys, ctrl
    end
    total_U(sys, ctrl) = Pulsar.compute_total_propagator(
        compute_propagators(build_total_hamiltonian(sys, ctrl), ctrl.dt))
    with_amps(ctrl, x) = ControlSequence(reshape(x, size(ctrl.amplitudes)), ctrl.dt, ctrl.n_steps)

    @testset "spec: members, weights, combine" begin
        a = OffsetAxis([-1.0, 0.0, 1.0]; operator = Iz)
        b = DriveScaleAxis([0.9, 1.1]; weights = [0.25, 0.75])
        ens = EnsembleAveraging(a, b)
        mb = ensemble_members(ens)
        @test length(ens) == 6 == length(mb)
        @test sum(m.weight for m in mb) ≈ 1
        @test Set(m.index for m in mb) == Set((i, j) for i in 1:3, j in 1:2)
        @test only(m.weight for m in mb if m.index == (2, 2)) ≈ 0.75 / 3
        z = EnsembleAveraging(OffsetAxis([1.0, 2.0]; operator = Iz),
                              DriveScaleAxis([0.9, 1.1]); combine = :zip)
        @test [m.index for m in ensemble_members(z)] == [(1, 1), (2, 2)]
        @test only(ensemble_members(EnsembleAveraging())) == (index = (), weight = 1.0)
        @test_throws ArgumentError EnsembleAveraging(a, b; combine = :zip)
        @test_throws ArgumentError EnsembleAveraging(a; aggregate = :median)
        @test_throws ArgumentError EnsembleAveraging(a; cvar_alpha = 0.0)
        @test_throws ArgumentError OffsetAxis([1.0, 2.0]; operator = Iz, weights = [1.0])
        @test_throws ArgumentError TermAxis(ComplexF64[0 1; 0 0], [1.0])      # not Hermitian
        @test_throws ArgumentError TimeScaleAxis([1.0, -0.1])
        @test_throws ArgumentError StateProbes([ComplexF64[1, 0]], ComplexF64[])
        @test occursin("6 members", sprint(show, ens))
    end

    @testset "operator builders" begin
        ss = spin_system([0.5, 0.5], [0.0 7.0; 7.0 0.0], [0.0, 0.0])
        @test offset_operator(ss) ≈ kron(Iz, I(2)) + kron(I(2), Iz)
        @test offset_operator(ss; spins = [2]) ≈ kron(I(2), Iz)
        @test coupling_operator(ss, 1, 2; kind = :zz) ≈ kron(Iz, Iz)
        @test coupling_operator(ss, 1, 2) ≈ kron(Ix, Ix) + kron(Iy, Iy) + kron(Iz, Iz)
        # the J part of the drift is exactly 2π·J·(I1·I2)
        @test ss.H_drift ≈ 2π * 7.0 .* coupling_operator(ss, 1, 2)
        mr = mr_system(["1H", "13C"])
        @test offset_operator(mr) ≈ spin_op(mr, :Iz, 1) + spin_op(mr, :Iz, 2)
        @test coupling_operator(mr, 1, 2; kind = :zz) ≈ spin_op(mr, :Iz, 1) * spin_op(mr, :Iz, 2)
        @test_throws ArgumentError coupling_operator(mr, 1, 2; kind = :dipolar)
    end

    @testset "input-state axis: probe averages vs propagator metrics" begin
        rng = MersenneTwister(7)
        no_axes(probes) = EnsembleAveraging(; probes)
        for d in (2, 3, 4)
            sys, ctrl = random_problem(rng, d, 2, 12)
            U = total_U(sys, ctrl)
            V = rand_unitary(rng, d)
            zP = tr(V' * U) / d
            ΦPQ = abs2(zP)
            # State-Squared over a 2-design = average gate fidelity
            if d == 2 || d == 3
                set = d == 2 ? :axes : :mub
                F = ensemble_averaging(no_axes(state_probes(V; set)), sys, ctrl; gradient = false)
                @test F ≈ (d * ΦPQ + 1) / (d + 1) atol = 1e-12
                @test F ≈ gate_fidelity(U, V; type = :state_squared_haar) atol = 1e-12
            end
            d == 2 && @test ensemble_averaging(no_axes(state_probes(V; set = :axes6)), sys, ctrl;
                                               gradient = false) ≈ (2ΦPQ + 1) / 3 atol = 1e-12
            # State-Real over any orthonormal basis = Φ^P_R
            F = quiet(() -> ensemble_averaging(no_axes(state_probes(V; set = :basis)), sys, ctrl;
                                               metric = :state_real, gradient = false))
            @test F ≈ real(zP) atol = 1e-12
            # DM-Real over a complete traceless operator basis
            set = ispow2(d) ? :pauli : :gell_mann
            F = ensemble_averaging(no_axes(operator_probes(V; set)), sys, ctrl; gradient = false)
            @test F ≈ (d^2 * ΦPQ - 1) / (d^2 - 1) atol = 1e-12
            Fg = ensemble_averaging(no_axes(operator_probes(V; set = :gell_mann)), sys, ctrl;
                                    gradient = false)
            @test Fg ≈ (d^2 * ΦPQ - 1) / (d^2 - 1) atol = 1e-12
        end
        # The computational basis alone is phase-blind: Z vs the identity.
        sysI = quantum_system(zeros(ComplexF64, 2, 2), [σx]); ctrl0 = ControlSequence(zeros(4, 1), 1e-6, 4)
        Fb = quiet(() -> ensemble_averaging(no_axes(state_probes(σz; set = :basis)), sysI, ctrl0;
                                            gradient = false))
        @test Fb ≈ 1
        @test ensemble_averaging(no_axes(state_probes(σz; set = :axes)), sysI, ctrl0;
                                 gradient = false) ≈ 1 / 3
        @test_throws ArgumentError state_probes(rand_unitary(rng, 4); set = :mub)    # d = 4 not prime
        @test_throws ArgumentError state_probes(rand_unitary(rng, 3); set = :axes)
        @test_throws ArgumentError operator_probes(rand_unitary(rng, 3); set = :pauli)
        @test length(state_probes(rand_unitary(rng, 5); set = :mub).inputs) == 30
        # projector pairs: DM-Real of |ψ⟩⟨ψ| → |φ⟩⟨φ| is State-Squared of ψ → φ,
        # value and gradient
        sys, ctrl = random_problem(rng, 3, 2, 10)
        ψ = normalize(randn(rng, ComplexF64, 3)); φ = normalize(randn(rng, ComplexF64, 3))
        Fs, Gs = ensemble_averaging(no_axes(StateProbes([ψ], [φ])), sys, ctrl)
        Fo, Go = ensemble_averaging(no_axes(OperatorProbes([ψ * ψ'], [φ * φ'])), sys, ctrl)
        @test Fo ≈ Fs atol = 1e-12
        @test Go ≈ Gs rtol = 1e-10
    end

    @testset "one member = compute_fidelity / compute_grape_gradient" begin
        rng = MersenneTwister(11)
        sys, ctrl = random_problem(rng, 3, 2, 15)
        V = rand_unitary(rng, 3)
        ens = EnsembleAveraging()
        for m in (:auto, :prop_real, :prop_modulus, :prop_squared, :state_squared_haar)
            tgt = unitary_target(V; metric = m)
            F, G = ensemble_averaging(ens, sys, ctrl, tgt)
            @test F ≈ compute_fidelity(sys, ctrl, tgt) atol = 1e-13
            @test G ≈ compute_grape_gradient(sys, ctrl, tgt) rtol = 1e-12
        end
        ψi = normalize(randn(rng, ComplexF64, 3)); ψt = normalize(randn(rng, ComplexF64, 3))
        for m in (:auto, :state_real, :state_modulus, :state_squared, :dm_real, :dm_modulus, :dm_squared)
            tgt = state_target(ψt; psi_init = ψi, metric = m)
            F, G = ensemble_averaging(ens, sys, ctrl, tgt)
            @test F ≈ compute_fidelity(sys, ctrl, tgt) atol = 1e-13
            @test G ≈ compute_grape_gradient(sys, ctrl, tgt) rtol = 1e-12
        end
        # metric override on the target, and target / probe exclusivity
        tgt = unitary_target(V)
        @test ensemble_averaging(ens, sys, ctrl, tgt; metric = :prop_real, gradient = false) ≈
              compute_fidelity(sys, ctrl, unitary_target(V; metric = :prop_real)) atol = 1e-13
        @test_throws ArgumentError ensemble_averaging(EnsembleAveraging(; probes = state_probes(V; set = :mub)),
                                                      sys, ctrl, tgt)
        @test_throws ArgumentError ensemble_averaging(ens, sys, ctrl)          # nothing to score
        @test_throws ArgumentError ensemble_averaging(EnsembleAveraging(; probes = operator_probes(V; set = :gell_mann)),
                                                      sys, ctrl; metric = :state_squared)
    end

    @testset "every axis against members built by hand" begin
        rng = MersenneTwister(23)
        ss = spin_system([0.5, 0.5], [0.0 12.0; 12.0 0.0], [30.0, -45.0])
        n_t = 16
        ctrl = ControlSequence(300 .* randn(rng, n_t, ss.n_controls), 2e-3 / n_t, n_t)
        V = rand_unitary(rng, 4)
        tgt = unitary_target(V; metric = :prop_real)
        Hd, Hc = ss.H_drift, ss.H_controls
        K = offset_operator(ss); J = coupling_operator(ss, 1, 2)
        Hd_alt = [Hd .+ 2π * 5.0 .* kron(Iz, I(2)), Hd .- 2π * 8.0 .* kron(I(2), Iz)]
        fswap(Hd, Hc, v) = (Hd .+ v .* kron(σx, σx), [Hc[2], Hc[1], Hc[3], Hc[4]])
        cases = [
            (OffsetAxis([-20.0, 0.0, 35.0]; operator = K),
             v -> (Hd .+ 2π * v .* K, Hc, ctrl.dt), [-20.0, 0.0, 35.0]),
            (CouplingAxis([-2.0, 3.0]; operator = J),
             v -> (Hd .+ 2π * v .* J, Hc, ctrl.dt), [-2.0, 3.0]),
            (DriveScaleAxis([0.8, 1.15]),
             v -> (Hd, v .* Hc, ctrl.dt), [0.8, 1.15]),
            (DriveScaleAxis([0.8, 1.15]; channels = [3, 4]),
             v -> (Hd, [Hc[1], Hc[2], v .* Hc[3], v .* Hc[4]], ctrl.dt), [0.8, 1.15]),
            (DrivePhaseAxis([-0.1, 0.25]; pairs = [(1, 2), (3, 4)]),
             ε -> (Hd, [cos(ε) .* Hc[1] .+ sin(ε) .* Hc[2], cos(ε) .* Hc[2] .- sin(ε) .* Hc[1],
                        cos(ε) .* Hc[3] .+ sin(ε) .* Hc[4], cos(ε) .* Hc[4] .- sin(ε) .* Hc[3]],
                   ctrl.dt), [-0.1, 0.25]),
            (TimeScaleAxis([0.97, 1.04]),
             v -> (Hd, Hc, v * ctrl.dt), [0.97, 1.04]),
            (DriftListAxis(Hd_alt),
             i -> (Hd_alt[i], Hc, ctrl.dt), [1, 2]),
            (FunctionAxis(fswap, [0.0, 2π * 10.0]),
             v -> (fswap(Hd, Hc, v)..., ctrl.dt), [0.0, 2π * 10.0]),
        ]
        for (axis, build, vals) in cases
            F, G = ensemble_averaging(EnsembleAveraging(axis), ss, ctrl, tgt)
            Fref = 0.0; Gref = zeros(size(G))
            for v in vals
                Hd_k, Hc_k, dt_k = build(v)
                sys_k = quantum_system(Hd_k, Hc_k)
                ctrl_k = ControlSequence(ctrl.amplitudes, dt_k, n_t)
                Fref += compute_fidelity(sys_k, ctrl_k, tgt) / length(vals)
                Gref .+= compute_grape_gradient(sys_k, ctrl_k, tgt) ./ length(vals)
            end
            @test F ≈ Fref atol = 1e-13
            @test G ≈ Gref rtol = 1e-12
        end
        # the phase axis is the RF field turned by ε about z: u_x Ix + u_y Iy → e^{−iεIz}(…)e^{+iεIz}
        ε = 0.3; R = exp(-im * ε * Iz)
        ax = DrivePhaseAxis([ε])
        _, Hc2, _ = Pulsar._apply_axis(ax, 1, Iz, [Ix, Iy], 1.0)
        @test Hc2[1] ≈ R * Ix * R' && Hc2[2] ≈ R * Iy * R'
        # TermAxis applied after DriftListAxis adds to the replaced drift
        e2 = EnsembleAveraging(DriftListAxis(Hd_alt), OffsetAxis([10.0]; operator = K))
        F2 = ensemble_averaging(e2, ss, ctrl, tgt; gradient = false)
        Fm = mean(compute_fidelity(quantum_system(H .+ 2π * 10.0 .* K, Hc), ctrl, tgt) for H in Hd_alt)
        @test F2 ≈ Fm atol = 1e-13
        # a drive-only ensemble works on a non-spin QuantumSystem; a system
        # without H_drift / H_controls is refused with a pointer to the MR front ends
        @test_throws ArgumentError ensemble_averaging(EnsembleAveraging(), mr_system(["1H"]), ctrl, tgt)
    end

    @testset "aggregators: value and ∂F/∂F_k" begin
        rng = MersenneTwister(5)
        Fk = 0.5 .+ 0.4 .* rand(rng, 9)
        ω  = rand(rng, 9) .+ 0.1
        for agg in (:mean, :sum, :worst_case, :cvar, :softmin)
            ens = EnsembleAveraging(OffsetAxis(1:9; operator = Iz); aggregate = agg,
                                    cvar_alpha = 0.3, softmin_beta = 20.0)
            F, a = Pulsar._ensemble_aggregate(ens, Fk, ω)
            f(x) = Pulsar._ensemble_aggregate(ens, x, ω)[1]
            @test a ≈ fd(f, Fk; h = 1e-7) atol = 1e-6
        end
        ens(agg) = EnsembleAveraging(OffsetAxis(1:9; operator = Iz); aggregate = agg, cvar_alpha = 0.3)
        @test Pulsar._ensemble_aggregate(ens(:mean), Fk, ω)[1] ≈ sum(ω .* Fk) / sum(ω)
        @test Pulsar._ensemble_aggregate(ens(:sum), Fk, ω)[1] ≈ sum(ω .* Fk)
        @test Pulsar._ensemble_aggregate(ens(:worst_case), Fk, ω)[1] == minimum(Fk)
        Fc = Pulsar._ensemble_aggregate(ens(:cvar), Fk, ω)[1]
        @test minimum(Fk) <= Fc <= sum(ω .* Fk) / sum(ω)
        @test Pulsar._ensemble_aggregate(EnsembleAveraging(OffsetAxis(1:9; operator = Iz); aggregate = :cvar,
                                                  cvar_alpha = 1.0), Fk, ω)[1] ≈ sum(ω .* Fk) / sum(ω)
        # large β → worst case; signed weights only with :sum
        @test Pulsar._ensemble_aggregate(EnsembleAveraging(OffsetAxis(1:9; operator = Iz); aggregate = :softmin,
                                                  softmin_beta = 1e4), Fk, ω)[1] ≈ minimum(Fk) atol = 1e-3
        ωs = copy(ω); ωs[2] = -0.5
        @test Pulsar._ensemble_aggregate(ens(:sum), Fk, ωs)[1] ≈ sum(ωs .* Fk)
        @test_throws ArgumentError Pulsar._ensemble_aggregate(ens(:mean), Fk, ωs)
    end

    @testset "full gradient vs finite differences" begin
        rng = MersenneTwister(31)
        # dt = 2.5 µs (dt·‖H‖ ≈ 1e-2) keeps the first-order GRAPE error under the tolerance
        sys, ctrl = random_problem(rng, 2, 2, 120; T = 0.3e-3, amp = 300.0)
        V = exp(-im * π / 2 * σx)
        axes = (OffsetAxis([-200.0, 150.0]; operator = Iz), DriveScaleAxis([0.9, 1.05]))
        for (tgt, probes, metric) in (
                (unitary_target(V; metric = :prop_real), nothing, :auto),
                (unitary_target(V; metric = :state_squared_haar), nothing, :auto),
                (state_target(ComplexF64[0, 1]; psi_init = ComplexF64[1, 0], metric = :state_modulus), nothing, :auto),
                (nothing, state_probes(V; set = :axes), :state_squared),
                (nothing, state_probes(V; set = :axes), :dm_modulus),
                (nothing, operator_probes(V; set = :pauli), :dm_real),
                (nothing, operator_probes(V; set = :pauli), :dm_squared))
            for agg in (:mean, :worst_case, :cvar, :softmin)
                ens = EnsembleAveraging(axes...; probes, aggregate = agg, cvar_alpha = 0.5,
                                        softmin_beta = 10.0)
                F, G = ensemble_averaging(ens, sys, ctrl, tgt; metric)
                f(x) = ensemble_averaging(ens, sys, with_amps(ctrl, x), tgt; metric, gradient = false)
                Gfd = fd(f, vec(copy(ctrl.amplitudes)))
                @test relerr(vec(permutedims(G)), Gfd) < 5e-3
            end
        end
        # signed weights (pass band minus stop band) with :sum
        ens = EnsembleAveraging(OffsetAxis([0.0, 600.0]; operator = Iz, weights = [1.0, -0.5]);
                                aggregate = :sum)
        tgt = unitary_target(V; metric = :prop_squared)
        F, G = ensemble_averaging(ens, sys, ctrl, tgt)
        fs(x) = ensemble_averaging(ens, sys, with_amps(ctrl, x), tgt; gradient = false)
        @test relerr(vec(permutedims(G)), fd(fs, vec(copy(ctrl.amplitudes)))) < 5e-3
    end
end
