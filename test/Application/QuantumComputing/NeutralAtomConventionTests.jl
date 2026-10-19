# test/Application/QuantumComputing/NeutralAtomConventionTests.jl
# ===========================================
# NeutralAtomSystem uses |g⟩ = |0⟩, |r⟩ = |1⟩ (Rydberg = upper level) in the
# detuning, the blockade and the decay alike. Until 2026-10-19 the detuning
# (+Δω/2 σz) and the T1 operator (|r⟩⟨g|) treated |0⟩ as the upper level while
# the blockade used |1⟩, so decay pumped into the blockaded state and the
# detuning had the wrong sign relative to V.

using Test
using Pulsar
using LinearAlgebra

@testset "NeutralAtomSystem: |r⟩ = |1⟩ throughout" begin
    fc = 1.0e6
    Δ  = 3.0e5                                    # atom above the carrier (Hz)
    V  = 2.0e6

    # one atom: |r⟩ sits +2πΔ above |g⟩ in the rotating frame
    a1 = neutral_atom_system([fc + Δ], zeros(1, 1); carrier_hz = fc)
    @test real(diag(a1.H_drift)) ≈ [0.0, 2π * Δ]

    # two atoms: |gg⟩, |gr⟩, |rg⟩, |rr⟩ = 0, Δ, Δ, 2Δ + V (blockade on |rr⟩)
    a2 = neutral_atom_system([fc + Δ, fc + Δ], [0.0 V; V 0.0]; carrier_hz = fc)
    @test real(diag(a2.H_drift)) ≈ 2π .* [0.0, Δ, Δ, 2Δ + V]
    # facilitation (anti-blockade): |rr⟩ degenerate with |gg⟩ at Δ = −V/2
    af = neutral_atom_system([fc - V / 2, fc - V / 2], [0.0 V; V 0.0]; carrier_hz = fc)
    @test abs(real(af.H_drift[4, 4])) < 1e-6 * 2π * V

    # T1 takes |r⟩ → |g⟩ (and annihilates |g⟩)
    d1 = neutral_atom_system([fc], zeros(1, 1); carrier_hz = fc, T1_s = 1e-4)
    L = only(d1.collapse_ops)
    @test L * ComplexF64[0, 1] ≈ sqrt(1e4) .* ComplexF64[1, 0]
    @test norm(L * ComplexF64[1, 0]) == 0
    # … so under T1 alone the Rydberg population decays to the ground state
    ρ = ComplexF64[0 0; 0 1]
    dρ = L * ρ * L' - (L' * L * ρ + ρ * L' * L) / 2
    @test real(dρ[2, 2]) < 0 && real(dρ[1, 1]) > 0

    # the ensemble offset is the builder's own detuning operator
    @test offset_operator(a2) ≈ kron(ComplexF64[0 0; 0 1], I(2)) + kron(I(2), ComplexF64[0 0; 0 1])
    @test coupling_operator(a2, 1, 2) ≈ diagm(ComplexF64[0, 0, 0, 1])
end
