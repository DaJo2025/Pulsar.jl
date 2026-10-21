# test/Application/MR/GRAPETrackingTests.jl
# ===========================================
# grape_tracking_kernel: value and gradient. The gradient was exactly twice
# ∂Φ/∂w — the co-state was injected as 2w(ρ_tar − ρ), the derivative of the
# squared distance, not of the reported w·Re Tr(ρ_tar ρ). Harmless alone (a
# global scale), wrong as soon as penalties or other terms share the objective.
# A deterministic smooth pulse at dt·‖H‖ ≈ 3e-3 keeps the first-order GRAPE
# error far below the tolerance on every Julia version.

using Test
using Pulsar
using LinearAlgebra

@testset "grape_tracking_kernel" begin
    Ix = ComplexF64[0 1; 1 0] / 2; Iy = ComplexF64[0 -im; im 0] / 2; Iz = ComplexF64[1 0; 0 -1] / 2
    n_t = 200; dt = 1e-7; pwr = 2π * 5e3
    tt = ((1:n_t) .- 0.5) ./ n_t
    w = vcat((0.7 .* sin.(2π .* 3 .* tt))', (0.5 .* cos.(2π .* 2 .* tt .+ 0.4))')
    ρ(v) = (I(2) .+ 2 .* (v[1] .* Ix .+ v[2] .* Iy .+ v[3] .* Iz)) ./ 2      # Bloch vector → ρ
    checkpoints = [TrackingPoint(60,  ρ((1.0, 0.0, 0.0)),  0.3),
                   TrackingPoint(110, ComplexF64[0 1; 0 0], 0.2),          # a coherence: non-Hermitian
                   TrackingPoint(150, ρ((0.0, -1.0, 0.0)), 0.3),
                   TrackingPoint(200, ρ((0.0, 0.0, -1.0)), 0.2)]
    drifts = [2π * δ .* Iz for δ in (-300.0, 400.0)]
    ctrl = MRControl(; drifts, operators = [Ix, Iy], rho_init = [ComplexF64[1, 0]],
                     rho_targ = [ComplexF64[0, 1]], pwr_levels = [pwr], pulse_dt = fill(dt, n_t),
                     tracking = checkpoints, verbose = false, backend = :cpu)
    F, G = grape_tracking_kernel(w, ctrl)

    # value: the weighted checkpoint overlaps of the propagated ρ(t)
    Fref = 0.0
    for H0 in drifts
        r = ComplexF64[1 0; 0 0]
        for n in 1:n_t
            U = exp(-im * dt .* (H0 .+ pwr .* (w[1, n] .* Ix .+ w[2, n] .* Iy)))
            r = U * r * U'
            for tp in checkpoints
                tp.step == n && (Fref += tp.weight * real(tr(tp.target_dm * r)) / length(drifts))
            end
        end
    end
    @test F ≈ Fref atol = 1e-12

    # gradient: central differences, and its scale is 1 (was 2)
    h = 1e-6
    Gfd = similar(w)
    for i in eachindex(w)
        wp = copy(w); wm = copy(w); wp[i] += h; wm[i] -= h
        Gfd[i] = (grape_tracking_kernel(wp, ctrl)[1] - grape_tracking_kernel(wm, ctrl)[1]) / 2h
    end
    @test maximum(abs.(G .- Gfd)) / maximum(abs, Gfd) < 1e-2
    @test dot(G, Gfd) / dot(Gfd, Gfd) ≈ 1 atol = 1e-2
end
