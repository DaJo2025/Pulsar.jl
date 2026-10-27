# test/Application/MR/StateKernelExactGradientTests.jl
# =====================================================
# grape_state_kernel: the gradient is the exact derivative of the reported
# fidelity on every path — generic CPU (any dim), the spin-½ Bloch fast path,
# grape_state_kernel_single and the batched / GPU kernel. Until 2026-10 every
# path used the first-order overlap ⟨λ[n+1]|Op|ψ[n]⟩, wrong by O(dt·‖H‖).
# The problems below run at dt·‖H‖ ≈ 0.2–0.4, where first order is off by
# ~10%, so a 1e-6 agreement with central differences can only come from the
# exact step derivative.

using Test
using Pulsar
using LinearAlgebra

function _fd_state_grad(w, f; h = 1e-6)
    G = similar(w)
    for i in eachindex(w)
        wp = copy(w); wm = copy(w); wp[i] += h; wm[i] -= h
        G[i] = (f(wp) - f(wm)) / 2h
    end
    return G
end

_relerr(G, Gref) = maximum(abs.(G .- Gref)) / maximum(abs, Gref)

@testset "grape_state_kernel exact gradient" begin
    Ix = ComplexF64[0 1; 1 0] / 2; Iy = ComplexF64[0 -im; im 0] / 2; Iz = ComplexF64[1 0; 0 -1] / 2
    n_t = 40; dt = 1e-5; pwr = 2π * 5e3                      # pwr·dt ≈ 0.31
    tt = ((1:n_t) .- 0.5) ./ n_t
    w2 = vcat((0.7 .* sin.(2π .* 2 .* tt) .+ 0.2)', (0.6 .* cos.(2π .* 3 .* tt .+ 0.4))')

    @testset "spin-½, generic path, $fid" for fid in (:state_real, :state_modulus, :state_squared)
        ctrl = MRControl(; drifts = [2π * δ .* Iz for δ in (-2e3, 1.5e3)], operators = [Ix, Iy],
                         rho_init = [ComplexF64[1, 0], ComplexF64[1, 1] / √2],
                         rho_targ = [ComplexF64[1, -im] / √2, ComplexF64[1, 0]],
                         pwr_levels = [0.9pwr, pwr], pulse_dt = fill(dt, n_t), fidelity = fid,
                         fast_path = false, verbose = false, backend = :cpu)
        F, G = grape_state_kernel(w2, ctrl)
        @test maximum(abs, G) > 1e-2                              # a nontrivial gradient
        @test _relerr(G, _fd_state_grad(w2, x -> grape_state_kernel(x, ctrl)[1])) < 1e-6
    end

    @testset "spin-½ Bloch fast path" begin
        ctrl = MRControl(; drifts = [2π * δ .* Iz for δ in (-2e3, 0.0, 1.5e3)], operators = [Ix, Iy],
                         rho_init = [ComplexF64[1, 0]], rho_targ = [ComplexF64[0, 1]],
                         pwr_levels = [pwr], pulse_dt = fill(dt, n_t), fidelity = :state_squared,
                         fast_path = true, verbose = false, backend = :cpu)
        F, G = grape_state_kernel(w2, ctrl)
        @test _relerr(G, _fd_state_grad(w2, x -> grape_state_kernel(x, ctrl)[1])) < 1e-6
        # the fast path and the generic path compute the same derivative
        ctrl_g = MRControl(; drifts = ctrl.drifts, operators = [Ix, Iy],
                           rho_init = ctrl.rho_init, rho_targ = ctrl.rho_targ,
                           pwr_levels = [pwr], pulse_dt = fill(dt, n_t), fidelity = :state_squared,
                           fast_path = false, verbose = false, backend = :cpu)
        Fg, Gg = grape_state_kernel(w2, ctrl_g)
        @test F ≈ Fg atol = 1e-12
        @test _relerr(G, Gg) < 1e-11
    end

    @testset "spin-1 (dim 3)" begin
        Sx = ComplexF64[0 1 0; 1 0 1; 0 1 0] / √2
        Sy = ComplexF64[0 -im 0; im 0 -im; 0 im 0] / √2
        Sz = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
        ctrl = MRControl(; drifts = [2π * 1e3 .* Sz .+ 2π * 4e3 .* (Sz^2 - 2I / 3)], operators = [Sx, Sy],
                         rho_init = [ComplexF64[1, 0, 0]], rho_targ = [ComplexF64[0, 0, 1]],
                         pwr_levels = [pwr], pulse_dt = fill(dt, n_t), fidelity = :state_squared,
                         verbose = false, backend = :cpu)
        F, G = grape_state_kernel(w2, ctrl)
        @test _relerr(G, _fd_state_grad(w2, x -> grape_state_kernel(x, ctrl)[1])) < 1e-6
    end

    # two J-coupled spins, three controls: dim 4, LAPACK eigen path
    I2 = Matrix{ComplexF64}(I, 2, 2)
    a(op) = kron(op, I2); b(op) = kron(I2, op)
    H0 = 2π * 800 .* a(Iz) .- 2π * 1.2e3 .* b(Iz) .+ 2π * 3e3 .* (a(Ix) * b(Ix) + a(Iy) * b(Iy) + a(Iz) * b(Iz))
    ops4 = [a(Ix) + b(Ix), a(Iy) + b(Iy), a(Iz)]
    w3 = vcat(w2, (0.3 .* sin.(2π .* tt .+ 1.0))')
    ψ0 = kron(ComplexF64[1, 0], ComplexF64[0, 1])
    ψT = kron(ComplexF64[0, 1], ComplexF64[1, 0])

    @testset "two coupled spins (dim 4), $fid" for fid in (:state_real, :state_squared)
        ctrl = MRControl(; drifts = [H0, H0 .+ 2π * 500 .* (a(Iz) + b(Iz))], operators = ops4,
                         rho_init = [ψ0], rho_targ = [ψT], pwr_levels = [pwr],
                         pulse_dt = fill(dt, n_t), fidelity = fid, verbose = false, backend = :cpu)
        F, G = grape_state_kernel(w3, ctrl)
        @test _relerr(G, _fd_state_grad(w3, x -> grape_state_kernel(x, ctrl)[1])) < 1e-6

        # grape_state_kernel_single: same member arithmetic, no averaging
        Fs, Gs = Pulsar.grape_state_kernel_single(w3, H0, pwr, ψ0, ψT, ctrl)
        @test _relerr(Gs, _fd_state_grad(w3,
                      x -> Pulsar.grape_state_kernel_single(x, H0, pwr, ψ0, ψT, ctrl)[1])) < 1e-6
    end

    @testset "degenerate spectrum (H = 0 on some steps)" begin
        wz = copy(w3); wz[:, 10:20] .= 0
        ctrl = MRControl(; drifts = [zeros(ComplexF64, 4, 4)], operators = ops4,
                         rho_init = [ψ0], rho_targ = [ψT], pwr_levels = [pwr],
                         pulse_dt = fill(dt, n_t), fidelity = :state_real, verbose = false, backend = :cpu)
        F, G = grape_state_kernel(wz, ctrl)
        @test _relerr(G, _fd_state_grad(wz, x -> grape_state_kernel(x, ctrl)[1])) < 1e-6
    end

    # The batched / GPU kernel integrates the same derivative by Gauss–Legendre
    # quadrature over the step; on the CPU backend it must agree with the
    # closed form to quadrature accuracy, including a long step that needs
    # several sub-intervals.
    @testset "batched kernel = closed form (dt·‖H‖ = $(round(s * pwr * dt, digits = 2)))" for s in (1.0, 30.0)
        ctrl = MRControl(; drifts = [H0, H0 .+ 2π * 500 .* (a(Iz) + b(Iz))], operators = ops4,
                         rho_init = [ψ0, b(Ix) * ψ0 .* 2], rho_targ = [ψT, ψ0],
                         pwr_levels = [0.95pwr, pwr], pulse_dt = fill(s * dt, n_t),
                         fidelity = :state_squared, verbose = false, fast_path = false)
        Fc, Gc = Pulsar._grape_cpu(w3, ctrl)
        Fb, Gb = Pulsar._grape_batched(w3, ctrl, Pulsar.cpu_backend())
        @test Fb ≈ Fc atol = 1e-12
        @test _relerr(Gb, Gc) < 1e-11
    end

    @testset "step quadrature and eigen helpers" begin
        # Gauss–Legendre nodes integrate e^{iωs} to the requested tolerance
        for ω in (0.0, 0.3, 2.0, 7.0, 40.0)
            x, wq = Pulsar._step_quadrature(ω)
            exact = ω == 0 ? 1.0 + 0im : (cis(ω) - 1) / (im * ω)
            @test abs(sum(wq .* cis.(ω .* x)) - exact) < 1e-14
            @test maximum(abs.(x .- reverse(1 .- x))) < 1e-14   # symmetric under s ↔ 1 − s
        end
        # closed-form 2×2 eigendecomposition, both branches and the diagonal case
        for H in (ComplexF64[1 0.3+0.2im; 0.3-0.2im -2], ComplexF64[-1 0.5im; -0.5im 2],
                  ComplexF64[0.4 0; 0 0.4], ComplexF64[3 0; 0 -3])
            V = Matrix{ComplexF64}(undef, 2, 2); E = Vector{Float64}(undef, 2)
            Pulsar._herm_eigen_into!(V, E, H)
            @test V * Diagonal(E) * V' ≈ H atol = 1e-14
            @test V' * V ≈ I atol = 1e-14
        end
    end
end
