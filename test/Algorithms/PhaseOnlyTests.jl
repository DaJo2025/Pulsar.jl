# PhaseOnlyParam end-to-end integration tests.
# Exercises the parameterization wired into MR optimcon and the QC GRAPE path.

using Test
using Pulsar
using LinearAlgebra
using Statistics
using Random

@testset "PhaseOnlyParam — integration" begin

    # ── 1. NMR ¹H broadband 90° via MR optimcon (PhaseOnlyParam) ───────────
    @testset "NMR broadband 90° — phase-only" begin
        sys = mr_system("1H")
        N = 50
        T = 200e-6
        RF_MAX = 2π * 10_000.0
        drifts = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-3000, 3000, 5)]
        ctrl = MRControl(
            drifts     = drifts,
            operators  = [spin_op(sys, :Ix), spin_op(sys, :Iy)],
            rho_init   = [spin_state(sys, :Iz)],
            rho_targ   = [spin_state(sys, :mIy)],
            pwr_levels = [RF_MAX],
            pulse_dt   = fill(T / N, N),
            method     = :lbfgs,
            max_iter   = 80,
            verbose    = false,
            parameterization = PhaseOnlyParam(0.99, [(1, 2)]),
        )
        Random.seed!(2026_04_29)
        guess = 0.05 .* randn(2, N)
        result = optimcon(ctrl, guess)
        @test result.fidelity > 0.95
        # geometric proof of phase-only: amplitudes constant
        amps = sqrt.(result.controls[1, :].^2 .+ result.controls[2, :].^2)
        @test maximum(amps) - minimum(amps) < 1e-10
        @test isapprox(amps[1], 0.99; atol=1e-10)
    end

    # ── 2. NMR Cartesian regression (PiecewiseConstant default) ───────────
    @testset "NMR Cartesian — default unchanged" begin
        sys = mr_system("1H")
        N = 50
        T = 200e-6
        RF_MAX = 2π * 10_000.0
        drifts = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-3000, 3000, 5)]
        ctrl = MRControl(
            drifts     = drifts,
            operators  = [spin_op(sys, :Ix), spin_op(sys, :Iy)],
            rho_init   = [spin_state(sys, :Iz)],
            rho_targ   = [spin_state(sys, :mIy)],
            pwr_levels = [RF_MAX],
            pulse_dt   = fill(T / N, N),
            method     = :lbfgs,
            max_iter   = 80,
            verbose    = false,
        )
        Random.seed!(2026_04_29)
        guess = 0.05 .* randn(2, N)
        result = optimcon(ctrl, guess)
        @test result.fidelity > 0.95   # default Cartesian path
    end

    # ── 3. Heteronuclear: 4-control with two phase pairs ───────────────────
    @testset "Heteronuclear 2-pair phase-only" begin
        # Toy 2-spin system (¹H + ¹³C) with 4 controls: Hx, Hy, Cx, Cy
        # Use plain matrices so we don't depend on heteronuclear_system internals.
        # 2-qubit Hilbert space with controls on each spin.
        Ix = [0 1; 1 0] / 2
        Iy = ComplexF64[0 -im; im 0] / 2
        Iz = [1 0; 0 -1] / 2
        I2 = [1 0; 0 1]
        H_drift = zeros(ComplexF64, 4, 4)
        Hx = kron(Ix, I2);  Hy = kron(Iy, I2)
        Cx = kron(I2, Ix);  Cy = kron(I2, Iy)
        N = 30
        T = 200e-6
        ctrl = MRControl(
            drifts     = [H_drift],
            operators  = [Hx, Hy, Cx, Cy],
            rho_init   = [ComplexF64[1, 0, 0, 0]],
            rho_targ   = [ComplexF64[0, 1, 0, 0]],   # flip carrier-1
            pwr_levels = [2π * 10_000.0],
            pulse_dt   = fill(T / N, N),
            method     = :lbfgs,
            max_iter   = 60,
            verbose    = false,
            parameterization = PhaseOnlyParam(1.0, [(1, 2), (3, 4)]),
        )
        Random.seed!(2026_04_29)
        guess = 0.05 .* randn(4, N)
        result = optimcon(ctrl, guess)
        # Both pairs must have constant amplitude
        amps_H = sqrt.(result.controls[1, :].^2 .+ result.controls[2, :].^2)
        amps_C = sqrt.(result.controls[3, :].^2 .+ result.controls[4, :].^2)
        @test maximum(amps_H) - minimum(amps_H) < 1e-10
        @test maximum(amps_C) - minimum(amps_C) < 1e-10
        @test isapprox(amps_H[1], 1.0; atol=1e-10)
        @test isapprox(amps_C[1], 1.0; atol=1e-10)
    end

    # ── 4. QC sanity: 1-qubit X-gate via grape_optimize + PhaseOnlyParam ───
    @testset "QC 1-qubit X-gate phase-only" begin
        σx = ComplexF64[0 1; 1 0]
        σy = ComplexF64[0 -im; im 0]
        H_drift = zeros(ComplexF64, 2, 2)
        H_controls = [σx / 2, σy / 2]
        sys = QuantumSystem(H_drift, H_controls, 2, 2, Dict{String,Any}())
        target = unitary_target(σx)
        N = 80
        dt = 0.05
        ctrl = ControlSequence(0.05 .* randn(MersenneTwister(2026_04_29), 2, N),
                               dt, N * dt, N)
        cfg = GRAPEConfig(max_iter=200, verbose=false,
                          parameterization=PhaseOnlyParam(0.8, [(1, 2)]))
        result = grape_optimize(sys, target, ctrl; config=cfg)
        @test result.fidelity > 0.99
        amps = sqrt.(result.controls[1, :].^2 .+ result.controls[2, :].^2)
        @test maximum(amps) - minimum(amps) < 1e-10
        @test isapprox(amps[1], 0.8; atol=1e-10)
    end

    # ── 5. QC default GRAPEConfig regression (PiecewiseConstant) ───────────
    @testset "QC GRAPEConfig default unchanged" begin
        σx = ComplexF64[0 1; 1 0]
        σy = ComplexF64[0 -im; im 0]
        H_drift = zeros(ComplexF64, 2, 2)
        H_controls = [σx / 2, σy / 2]
        sys = QuantumSystem(H_drift, H_controls, 2, 2, Dict{String,Any}())
        target = unitary_target(σx)
        N = 80
        dt = 0.05
        # Deterministic initial controls — Julia 1.12 changed the output of
        # randn(MersenneTwister(seed), ...) (the stdlib does not promise
        # cross-version RNG stability for non-default seeds). A hand-crafted
        # starting point keeps the test version-independent. Magnitude is in
        # the right ballpark for the X-gate (total angle ≈ Σ u·dt ≈ π) so
        # GRAPE converges in well under 200 iterations.
        u = Matrix{Float64}(undef, 2, N)
        @inbounds for k in 1:N
            t = (k - 0.5) / N
            u[1, k] = 0.5 + 0.1 * sin(2π * t)
            u[2, k] = 0.1 * cos(2π * t)
        end
        ctrl = ControlSequence(u, dt, N * dt, N)
        cfg = GRAPEConfig(max_iter=200, verbose=false)
        result = grape_optimize(sys, target, ctrl; config=cfg)
        @test result.fidelity > 0.99
    end

end
