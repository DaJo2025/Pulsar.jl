# ============================================================
# Pulsar.jl — AutoDiff Tests
# Pulse Design Library for Spin Control Algorithms and Rollout
# ============================================================

using Test, LinearAlgebra

@testset "Automatic Differentiation" begin

    # ── Shared setup: single qubit ─────────────────────────────
    function setup_qubit(; dt=5e-9, n_ts=20)
        σ_x = ComplexF64[0 1; 1 0]
        σ_y = ComplexF64[0 -1im; 1im 0]
        σ_z = ComplexF64[1 0; 0 -1]
        H_drift = 2π * 1e3 * σ_z
        system  = quantum_system(H_drift, [σ_x, σ_y])
        target  = state_target(ComplexF64[0.0, 1.0])
        u       = zeros(2, n_ts) .+ 0.05
        controls = ControlSequence(u, dt, dt * n_ts, n_ts)
        return system, target, controls
    end

    @testset "Finite-difference backend" begin
        system, target, controls = setup_qubit()
        cfg  = AutoDiffConfig(backend=:finite_diff, verify_against_numerical=false)
        grad = compute_gradient_autodiff(system, controls, target; config=cfg)

        @test size(grad) == (system.n_controls, controls.n_timesteps)
        @test !any(isnan, grad)
        @test !any(isinf, grad)
    end

    @testset "Finite-diff gradient matches GRAPE gradient" begin
        system, target, controls = setup_qubit()
        cfg     = AutoDiffConfig(backend=:finite_diff)
        grad_ad = compute_gradient_autodiff(system, controls, target; config=cfg)
        grad_grape = compute_grape_gradient(system, controls, target)

        max_diff = maximum(abs.(grad_ad .- grad_grape))
        @test max_diff < 1e-5  # Central differences vs exact gradient
    end

    @testset "Backend selection returns valid symbol" begin
        system, target, controls = setup_qubit()
        cfg = AutoDiffConfig()
        be  = Pulsar.select_autodiff_backend(system, controls, cfg)
        @test be in (:forward, :reverse, :finite_diff)
    end

    @testset "Gradient shape consistency" begin
        # Various system sizes
        for (nc, nt) in [(2, 10), (4, 30), (1, 50)]
            σ_x = ComplexF64[0 1; 1 0]
            σ_y = ComplexF64[0 -1im; 1im 0]
            Hcs = fill(σ_x, nc)
            sys = quantum_system(zeros(ComplexF64, 2, 2), Hcs)
            tgt = state_target(ComplexF64[0.0, 1.0])
            u   = zeros(nc, nt)
            cs  = ControlSequence(u, 5e-9, 5e-9*nt, nt)

            cfg  = AutoDiffConfig(backend=:finite_diff)
            grad = compute_gradient_autodiff(sys, cs, tgt; config=cfg)
            @test size(grad) == (nc, nt)
        end
    end

    @testset "Gradient non-zero for non-trivial system" begin
        # With non-zero controls, gradient should not be identically zero.
        # Use an EXPLICIT initial state different from the target, plus
        # controls strong enough to actually move the state. The original
        # test was passing under the v0.1.0 eigen propagator because of
        # floating-point noise in a degenerate setup (tiny controls vs
        # large drift, plus initial_state defaulting to target_state),
        # which gave a barely-detectable FD gradient of ~1e-10. The v0.2.0
        # closed-form Pauli propagator is more accurate, so it correctly
        # returns 0 in that regime. Fixing the test to be physically
        # meaningful (true |0⟩→|1⟩ transfer, controls ≳ drift) makes the
        # gradient genuinely informative regardless of propagator path.
        import Random
        Random.seed!(20260513)
        σ_x = ComplexF64[0 1; 1 0]
        σ_y = ComplexF64[0 -1im; 1im 0]
        σ_z = ComplexF64[1 0; 0 -1]
        dt, n_ts = 5e-9, 20
        H_drift  = 2π * 1e3 * σ_z
        system   = quantum_system(H_drift, [σ_x, σ_y])
        # Explicit |0⟩ → |1⟩ transfer (psi_init differs from target_state)
        target   = state_target(ComplexF64[0.0, 1.0]; psi_init=ComplexF64[1.0, 0.0])
        u        = 2π * 1e6 .* randn(2, n_ts)
        cs       = ControlSequence(u, dt, dt*n_ts, n_ts)
        cfg      = AutoDiffConfig(backend=:finite_diff)
        grad     = compute_gradient_autodiff(system, cs, target; config=cfg)
        @test maximum(abs, grad) > 1e-10
    end

    @testset "verify_gradient_autodiff runs without error" begin
        system, target, controls = setup_qubit()
        # Should run and return Bool (true or false)
        ok = verify_gradient_autodiff(system, controls, target; tol=1e-4, verbose=false)
        @test ok isa Bool
    end

    @testset "Auto backend compute_gradient_autodiff" begin
        system, target, controls = setup_qubit()
        # Default config — should pick any available backend
        grad = compute_gradient_autodiff(system, controls, target)
        @test size(grad) == (system.n_controls, controls.n_timesteps)
        @test !any(isnan, grad)
    end

end
