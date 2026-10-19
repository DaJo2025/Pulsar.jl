# ============================================================
# Pulsar.jl — Fidelity-metric coverage tests
#
# Verifies that fidelity metrics can be SELECTED (via target.metric) and drive
# the shared gradient chokepoint `compute_grape_gradient`, so every gradient-based
# algorithm and every application that funnels through it gets the chosen metric.
# Also locks in the PropAverage gradient and the :prop_modulus phase-factor fix.
# ============================================================

using Test, LinearAlgebra

@testset "Fidelity-metric coverage" begin

    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -1im; 1im 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = quantum_system(0.3 * σz, [σx, σy])

    # Finite-difference reference of compute_fidelity for a given target.
    function fd(sys, u, dt, nt, tgt; ε = 1e-6)
        g = zeros(size(u))
        for j in axes(u, 1), k in axes(u, 2)
            up = copy(u); up[j, k] += ε
            um = copy(u); um[j, k] -= ε
            Hp = build_total_hamiltonian(sys, ControlSequence(up, dt, dt*nt, nt))
            Hm = build_total_hamiltonian(sys, ControlSequence(um, dt, dt*nt, nt))
            Up = Matrix{ComplexF64}(I, 2, 2); Um = Matrix{ComplexF64}(I, 2, 2)
            for t in 1:nt
                Up = compute_propagator(Hp[t, :, :], dt) * Up
                Um = compute_propagator(Hm[t, :, :], dt) * Um
            end
            g[j, k] = (compute_fidelity(Up, tgt) - compute_fidelity(Um, tgt)) / (2ε)
        end
        return g
    end

    @testset "metric kwarg + validation + backward-compat default" begin
        @test unitary_target(σx).metric === :auto          # default
        @test unitary_target(σx; metric = :prop_modulus).metric === :prop_modulus
        @test state_target(ComplexF64[0, 1]; metric = :state_real).metric === :state_real
        @test_throws ArgumentError unitary_target(σx; metric = :not_a_metric)
        @test_throws ArgumentError state_target(ComplexF64[0, 1]; metric = :prop_squared)
        # Existing 5-arg positional constructor still works (metric → :auto).
        @test QuantumTarget("state", ComplexF64[0, 1], nothing, nothing, 2).metric === :auto
    end

    @testset "gate metrics drive compute_grape_gradient (match FD, small dt)" begin
        dt, nt = 1e-3, 12
        u = zeros(2, nt); u[1, :] .= 5.0; u[2, :] .= 1.5
        cs = ControlSequence(u, dt, dt*nt, nt)
        for m in (:auto, :prop_squared, :prop_real, :prop_average, :prop_modulus)
            tgt = unitary_target(ComplexF64[0 1; 1 0]; metric = m)
            g  = compute_grape_gradient(sys, cs, tgt)
            gf = fd(sys, u, dt, nt, tgt)
            gmax = max(maximum(abs, g), 1e-12)
            @test maximum(abs.(g .- gf)) / gmax < 5e-2   # first-order GRAPE vs FD
        end
    end

    @testset "state metrics drive compute_grape_gradient (match FD, small dt)" begin
        dt, nt = 1e-3, 12
        u = zeros(2, nt); u[1, :] .= 5.0; u[2, :] .= 1.5
        cs = ControlSequence(u, dt, dt*nt, nt)
        for m in (:auto, :state_real, :state_squared, :state_modulus, :dm_real, :dm_squared)
            tgt = state_target(ComplexF64[0, 1]; psi_init = ComplexF64[1, 0], metric = m)
            g  = compute_grape_gradient(sys, cs, tgt)
            gf = fd(sys, u, dt, nt, tgt)
            gmax = max(maximum(abs, g), 1e-12)
            @test maximum(abs.(g .- gf)) / gmax < 5e-2
        end
    end

    @testset ":prop_average gradient = (d/(d+1)) · process-fidelity gradient" begin
        dt, nt = 1e-3, 12
        u = zeros(2, nt); u[1, :] .= 5.0; u[2, :] .= 1.5
        cs = ControlSequence(u, dt, dt*nt, nt)
        g_norm = compute_grape_gradient(sys, cs, unitary_target(ComplexF64[0 1; 1 0]; metric = :prop_squared))
        g_avg  = compute_grape_gradient(sys, cs, unitary_target(ComplexF64[0 1; 1 0]; metric = :prop_average))
        @test maximum(abs.(g_avg .- (2/3) .* g_norm)) < 1e-12   # d=2 → d/(d+1)=2/3
    end

    @testset ":prop_modulus gradient is non-zero with a complex (imaginary) overlap" begin
        # Regression for the phase-factor bug: with Tr(U†_t U) purely imaginary the
        # old Im(inner/|z|) formula returned exactly 0; the fixed Im(z̄·inner)/|z|
        # does not.
        dt, nt = 1e-2, 12
        u = zeros(2, nt); u[1, :] .= 40.0; u[2, :] .= 12.0
        cs = ControlSequence(u, dt, dt*nt, nt)
        g = compute_grape_gradient(sys, cs, unitary_target(ComplexF64[0 1; 1 0]; metric = :prop_modulus))
        @test maximum(abs, g) > 1e-6
    end

    @testset "density-matrix-only metric (Uhlmann) rejected for pure-state target" begin
        # :dm_uhlmann needs a density matrix; it is not a valid metric for a
        # pure-state vector target (that is the open-system / Lindblad domain).
        @test :dm_uhlmann ∉ Pulsar.VALID_STATE_METRICS
        @test_throws ArgumentError state_target(ComplexF64[0, 1]; metric = :dm_uhlmann)
    end

    @testset "Φ^X_Y names: grid, labels, legacy aliases, level checks" begin
        @test PROP_METRICS  == (:prop_real,  :prop_modulus,  :prop_squared)
        @test STATE_METRICS == (:state_real, :state_modulus, :state_squared)
        @test DM_METRICS    == (:dm_real,    :dm_modulus,    :dm_squared)
        @test all(m -> m in FIDELITY_METRICS, (PROP_METRICS..., STATE_METRICS..., DM_METRICS...))
        @test metric_label(:dm_squared) == "Φ^D_Q (DM-Squared)"

        # The same old symbol resolves per level — the ambiguity the rename removes.
        @test canonical_metric(:square, STATE_METRICS; context = "t-state") === :state_squared
        @test canonical_metric(:square, (:dm_real, :dm_squared); context = "t-dm") === :dm_squared
        @test canonical_metric(:real, PROP_METRICS; context = "t-prop") === :prop_real
        @test canonical_metric(:squared, Pulsar.SU2_METRICS; context = "t-su2") === :state_squared
        @test canonical_metric(:real_trace, Pulsar.SPINI_METRICS; context = "t-spinI") === :prop_real
        @test canonical_metric(:state, Pulsar.SPINI_METRICS; context = "t-spinI") === :state_squared
        @test_logs (:warn, r"deprecated.*use :state_squared") canonical_metric(
            :square, STATE_METRICS; context = "t-warn")

        # Old names still work through the public entry points.
        @test unitary_target(σx; metric = :normalized).metric === :prop_squared
        @test state_target(ComplexF64[0, 1]; metric = :dm_linear).metric === :dm_real
        @test state_fidelity(ComplexF64[1, 0], ComplexF64[1, 0]; type = :square) == 1.0
        @test gate_fidelity(σx, σx; type = :normalized) ≈ 1.0
        @test Pulsar.RealOverlap === StateReal
        @test Pulsar.NORMALIZED_GATE === PROP_SQUARED

        # A valid name at the wrong level throws instead of being reinterpreted.
        @test_throws ArgumentError canonical_metric(:dm_squared, STATE_METRICS)
        @test_throws ArgumentError state_target(ComplexF64[0, 1]; metric = :prop_real)
        @test_throws ArgumentError unitary_target(σx; metric = :state_squared)
        @test_throws ArgumentError canonical_metric(:not_a_metric, STATE_METRICS)

        # MRControl propagates state vectors; LindbladMRControl density matrices.
        mr = (; drifts = [zeros(ComplexF64, 2, 2)], operators = [σx / 2],
                rho_init = ComplexF64[1, 0], rho_targ = ComplexF64[0, 1],
                pwr_levels = [1.0], pulse_dt = fill(1e-3, 4), verbose = false)
        @test MRControl(; mr...).fidelity === :state_real
        @test MRControl(; mr..., fidelity = :square).fidelity === :state_squared
        @test_throws ArgumentError MRControl(; mr..., fidelity = :dm_squared)
        @test LindbladMRControl(; mr...).fidelity === :dm_squared
        @test LindbladMRControl(; mr..., fidelity = :square).fidelity === :dm_squared
        @test LindbladMRControl(; mr..., fidelity = :real).fidelity === :dm_real
        @test_throws ArgumentError LindbladMRControl(; mr..., fidelity = :state_squared)
        @test LindbladMRControl(; mr..., fidelity = :dm_modulus).fidelity === :dm_modulus
    end

    @testset "non-differentiable metric errors clearly on gradient (catch-all)" begin
        # The fidelity_grad_prefactor catch-all gives a helpful message rather
        # than a MethodError for any metric without a closed-form GRAPE gradient.
        @test_throws ArgumentError Pulsar.fidelity_grad_prefactor(
            ComplexF64(1, 0), ComplexF64(0, 1), 1.0, Pulsar.DMUhlmann())
    end
end
