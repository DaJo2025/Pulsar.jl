# test/Algorithms/QuasiNewtonRobustnessTests.jl
# ===========================================
# lbfgsb_optimize, lbfgs_optimize and bfgs_optimize when the line search fails. Near an optimum whose supplied
# gradient is approximate (first-order GRAPE), no step along it lowers f: the
# line search shrinks the step to ~1e-14 and (s, y) is rounding noise. Accepting
# those pairs grew the L-BFGS direction without bound (‖d‖ ≈ 1e3 here, Inf on a
# CI runner, where the Liouvillian exponential then threw InexactError) and the
# loop ran on for every remaining iteration. Now a failed search drops the
# curvature model and retries along −g, and stops if that fails too.

using Test
using Pulsar
using LinearAlgebra
using Random

@testset "quasi-Newton: failed line searches" begin
    @testset "biased gradient, $(nameof(opt)): stops instead of running on (n = $n, bias = $bias)" for
            opt in (lbfgsb_optimize, lbfgs_optimize, bfgs_optimize), (n, bias) in ((8, 1e-4), (40, 1e-5))
        c = collect(range(-1, 1; length = n))
        evals = Ref(0); maxθ = Ref(0.0)
        F(θ) = sum(abs2, θ .- c) + 0.1 * sum(abs2, sin.(3 .* θ))
        f(θ) = (evals[] += 1; maxθ[] = max(maxθ[], maximum(abs, θ)); F(θ))
        ∇F(θ) = 2 .* (θ .- c) .+ 0.6 .* sin.(3 .* θ) .* cos.(3 .* θ)
        kw = opt === lbfgsb_optimize ? (; use_native = true) : (;)
        θb, fb, _ = opt(f, (g, θ) -> (g .= ∇F(θ) .+ bias; g), zeros(n);
                        max_iter = 300, tol = 1e-12, kw...)
        @test evals[] < 1_000                    # was ~12 000 (every iteration a failed search)
        @test maxθ[] < 3                         # trial points stay near the iterates
        _, fe, _ = opt(F, (g, θ) -> (g .= ∇F(θ); g), zeros(n);
                       max_iter = 300, tol = 1e-10, kw...)
        @test fb ≤ fe + 1e-6                     # as good as the exact-gradient run
    end

    @testset "Lindblad QC problem that diverged on CI" begin
        Random.seed!(2025)
        Ω  = 2π * 1.0e6
        sys = QuantumSystem(ComplexF64[0 0; 0 0], [Ω .* ComplexF64[0 1; 1 0], Ω .* ComplexF64[0 -im; im 0]],
                            2, 2, Dict{String,Any}())
        n_t = 40; dt = 5.0e-9
        guess = 0.30 .+ 0.10 .* randn(2, n_t)
        ctx = LindbladQCControl(sys, state_target(ComplexF64[0.0, 1.0]),
                                ControlSequence(copy(guess), dt, dt * n_t, n_t);
                                jump_ops = [ComplexF64[0 1; 0 0]], decay_rates = [1 / 50e-6],
                                method = :lbfgsb, max_iter = 60, verbose = false)
        f0, g! = Pulsar._qc_make_fg_closures(ctx, 2, n_t)
        evals = Ref(0); maxθ = Ref(0.0)
        f(θ) = (evals[] += 1; maxθ[] = max(maxθ[], maximum(abs, θ)); f0(θ))
        _, fb, _ = lbfgsb_optimize(f, g!, vec(copy(ctx.ctrl.controls)); max_iter = 60, use_native = true)
        @test -fb > 0.99
        @test maxθ[] < 10 * maximum(abs, guess)  # was ≈ 1130 (2400× the guess)
        @test evals[] < 500                      # was ≈ 2300
    end
end
