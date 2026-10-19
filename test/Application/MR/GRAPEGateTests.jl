# test/Application/MR/GRAPEGateTests.jl
# ===========================================
# grape_gate_kernel (Application/MR/GRAPEGate.jl): the propagator-level gate
# objective of the MR layer. Pins its value against Φ^P_* of the propagator
# built step by step, and its first-order GRAPE gradient against central
# finite differences for every metric, at a step small enough that the
# O(dt·‖H‖) first-order error is below the tolerance.

using Test
using Pulsar
using LinearAlgebra
using Random
using Statistics

@testset "grape_gate_kernel" begin
    rng = MersenneTwister(7)
    Ix = ComplexF64[0 1; 1 0] / 2; Iy = ComplexF64[0 -im; im 0] / 2; Iz = ComplexF64[1 0; 0 -1] / 2
    drifts = [2π * δ .* Iz for δ in (-800.0, 0.0, 1200.0)]
    pwr = [2π * 5e3, 2π * 5.5e3]
    n_t = 200
    dts = fill(1e-7, n_t)                            # dt·‖H‖ ≈ 4e-3
    U_F = exp(-im * π / 2 * 2 .* Ix)                 # 90°x
    w = 0.7 .* randn(rng, 2, n_t)

    function U_member(d, p)
        U = Matrix{ComplexF64}(I, 2, 2)
        for n in 1:n_t
            U = exp(-im * dts[n] .* (d .+ p .* (w[1, n] .* Ix .+ w[2, n] .* Iy))) * U
        end
        return U
    end
    zs = [tr(U_F' * U_member(d, p)) / 2 for d in drifts for p in pwr]
    ref = Dict(:prop_real => mean(real.(zs)), :prop_squared => mean(abs2.(zs)),
               :prop_modulus => mean(abs.(zs)), :prop_real_quad => mean(1 .- (1 .- real.(zs)).^2))

    for m in (:prop_real, :prop_real_quad, :prop_squared, :prop_modulus)
        F, G = grape_gate_kernel(w, drifts, [Ix, Iy], pwr, dts, U_F; metric = m)
        @test F ≈ ref[m] atol = 1e-12
        h = 1e-6
        Gfd = similar(w)
        for i in eachindex(w)
            wp = copy(w); wm = copy(w); wp[i] += h; wm[i] -= h
            Gfd[i] = (grape_gate_kernel(wp, drifts, [Ix, Iy], pwr, dts, U_F; metric = m)[1] -
                      grape_gate_kernel(wm, drifts, [Ix, Iy], pwr, dts, U_F; metric = m)[1]) / 2h
        end
        @test maximum(abs.(G .- Gfd)) / maximum(abs, Gfd) < 1e-2
    end
    @test_throws ArgumentError grape_gate_kernel(w, drifts, [Ix, Iy], pwr, dts, U_F; metric = :state_real)
end
