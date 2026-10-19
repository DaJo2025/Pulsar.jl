# test/Physics/FidelityMetricsTheme4Tests.jl
# ==========================================
# Theme 4 — extended fidelity metrics:
#   * EssentialSubspaceGate
#   * CooperativeTargetFidelity
#   * ProcessTomographyFidelity
#
# Each metric ships with a hand-computed reference value plus a sanity
# identity (e.g. process-tomography fidelity equals the standard normalised
# gate fidelity for unitary inputs).

using Test
using Pulsar
using LinearAlgebra

@testset "Theme 4 — extended fidelity metrics" begin

    @testset "EssentialSubspaceGate" begin
        # Construct a 3×3 "transmon-style" propagator: qubit subspace |0⟩,|1⟩
        # rotates by σ_x; |2⟩ is a guard level with arbitrary leakage phase.
        U_qubit = ComplexF64[0 1; 1 0]                       # X gate
        leak    = exp(im * 0.7)
        U_full  = ComplexF64[
            0      1      0;
            1      0      0;
            0      0      leak
        ]
        U_target_full = ComplexF64[
            0 1 0;
            1 0 0;
            0 0 1
        ]

        m = EssentialSubspaceGate([1, 2])

        # Full-dim target: should match qubit-only gate fidelity = 1.0.
        F = gate_fidelity(U_full, U_target_full, m)
        @test F ≈ 1.0 atol = 1e-12

        # Essential-dim target also accepted.
        F2 = gate_fidelity(U_full, U_qubit, m)
        @test F2 ≈ 1.0 atol = 1e-12

        # Non-trivial scaled overlap on the essential block: replace U[1:2,1:2]
        # with cos(θ) * X — fidelity should be cos²(θ).
        θ = 0.4
        U_full2 = copy(U_full); U_full2[1:2, 1:2] .= cos(θ) .* ComplexF64[0 1; 1 0]
        F3 = gate_fidelity(U_full2, U_target_full, m)
        @test F3 ≈ cos(θ)^2 atol = 1e-12

        # Index validation
        @test_throws ArgumentError gate_fidelity(U_full, U_target_full,
                                                 EssentialSubspaceGate(Int[]))
        @test_throws ArgumentError gate_fidelity(U_full, U_target_full,
                                                 EssentialSubspaceGate([1, 5]))
    end

    @testset "ProcessTomographyFidelity" begin
        # For unitary U and U_target, F_proc = |Tr(U†_t U)|²/d², matching
        # the standard normalised gate fidelity.
        d = 3
        U_target = Matrix{ComplexF64}(I, d, d)
        # Random unitary via QR
        A = randn(ComplexF64, d, d)
        Q, R = qr(A); U = Q * Diagonal(sign.(diag(R)))

        m = ProcessTomographyFidelity(d)
        F_proc = gate_fidelity(U, U_target, m)
        F_norm = gate_fidelity(U, U_target, PROP_SQUARED)
        @test F_proc ≈ F_norm atol = 1e-12

        # Identity returns exactly 1.
        @test gate_fidelity(U_target, U_target, m) ≈ 1.0 atol = 1e-12

        # Dimension validation
        m_bad = ProcessTomographyFidelity(4)
        @test_throws DimensionMismatch gate_fidelity(U, U_target, m_bad)
    end

    @testset "CooperativeTargetFidelity" begin
        d        = 2
        U        = Matrix{ComplexF64}(I, d, d)
        U_target = Matrix{ComplexF64}(I, d, d)
        ψ_init   = ComplexF64[1, 0]
        ψ_target = ComplexF64[1, 0]                            # already prepared

        m = CooperativeTargetFidelity(PROP_SQUARED, STATE_SQUARED;
                                       α = 0.7, β = 0.3)
        F = cooperative_fidelity(U, U_target, ψ_init, ψ_target, m)
        # Both branches return 1, so F = 0.7 + 0.3 = 1.0
        @test F ≈ 1.0 atol = 1e-12

        # Half-rotation case: state arrives at |+⟩ but target is |1⟩.
        H_gate   = ComplexF64[1 1; 1 -1] / sqrt(2)             # Hadamard
        U_target_X = ComplexF64[0 1; 1 0]                      # X gate target
        F_g = gate_fidelity(H_gate, U_target_X, PROP_SQUARED)
        ψ_f = H_gate * ψ_init
        F_s = state_fidelity(ψ_target, ψ_f, STATE_SQUARED)
        m2 = CooperativeTargetFidelity(PROP_SQUARED, STATE_SQUARED;
                                        α = 0.5, β = 0.5)
        F2 = cooperative_fidelity(H_gate, U_target_X, ψ_init, ψ_target, m2)
        @test F2 ≈ 0.5 * F_g + 0.5 * F_s atol = 1e-12

        # α + β need not equal 1 (this is just a weighted sum).
        m3 = CooperativeTargetFidelity(PROP_SQUARED, STATE_SQUARED;
                                        α = 1.0, β = 1.0)
        F3 = cooperative_fidelity(H_gate, U_target_X, ψ_init, ψ_target, m3)
        @test F3 ≈ F_g + F_s atol = 1e-12
    end

    @testset "Type stability — methods return Float64" begin
        d = 2
        U  = Matrix{ComplexF64}(I, d, d)
        Ut = Matrix{ComplexF64}(I, d, d)
        ψi = ComplexF64[1, 0]; ψt = ComplexF64[1, 0]
        @test gate_fidelity(U, Ut, EssentialSubspaceGate([1])) isa Float64
        @test gate_fidelity(U, Ut, ProcessTomographyFidelity(d)) isa Float64
        @test cooperative_fidelity(U, Ut, ψi, ψt,
                  CooperativeTargetFidelity(PROP_SQUARED, STATE_SQUARED)) isa Float64
    end
end
