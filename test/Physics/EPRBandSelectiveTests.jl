# EPR band-selective / DEER dispatch: shift_system(::EPRSpinSystem) must exist AND
# apply a physically meaningful (Sz) offset so band_selective_* is offset-dependent.
using Test
using Pulsar
using LinearAlgebra

@testset "EPR shift_system + band-selective objective" begin
    sys = epr_system(1//2, Rational{Int}[1//1], 0.34, 9.5e9)
    δ   = 65e6

    # 1. Offset couples to the electron Sz (not identity).
    ss = shift_system(sys, δ)
    @test ss isa EPRSpinSystem
    @test ss.H_drift ≈ sys.H_drift .+ (2π * δ) .* sys.S_ops[3]
    # An identity offset would be wrong (global phase, no dynamical effect):
    @test !(ss.H_drift ≈ sys.H_drift .+ (2π * δ) .* Matrix{ComplexF64}(I, sys.dim, sys.dim))

    # 2. band_selective_fidelity now dispatches on EPRSpinSystem (was a MethodError).
    ctrl  = ControlSequence(0.01 .* randn(20, 2), 1e-9, 20)
    psi   = zeros(ComplexF64, sys.dim); psi[1] = 1.0
    tgt   = state_target(psi; psi_init = psi)
    bands = [BandWeight(-65e6, 0.5), BandWeight(0.0, 0.5)]
    @test isfinite(band_selective_fidelity(sys, ctrl, tgt, bands))
    @test size(band_selective_gradient(sys, ctrl, tgt, bands)) == size(ctrl.controls)

    # 3. The offset is dynamically effective: a driven inversion on a clean 2-level
    #    electron is offset-dependent (on-resonance ≠ far-off-resonance).
    e2   = epr_system(1//2, Rational{Int}[], 0.34, 9.5e9)   # dim 2, no nuclei
    T    = 20e-9; N = 40
    w    = zeros(N, 2); w[:, 1] .= π / T                     # nominal π rotation about x
    c2   = ControlSequence(w, T / N, N)
    inv  = state_target(ComplexF64[0, 1]; psi_init = ComplexF64[1, 0])
    F_on  = compute_fidelity(shift_system(e2, 0.0),   c2, inv)
    F_off = compute_fidelity(shift_system(e2, 200e6), c2, inv)
    @test abs(F_on - F_off) > 0.05     # offset genuinely changes the evolution
end
