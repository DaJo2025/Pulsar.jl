# test/Physics/MetricKernelTests.jl
# ===========================================
# One fidelity-metric implementation for every kernel (Computation/
# FidelityReductions.jl). Pins:
#   (a) the primitive: dF = Re(w · dz) for every reduction, incl. the pure-state
#       lift of the DM row;
#   (b) every kernel × every metric it accepts against central finite
#       differences (exact-adjoint kernels tightly, first-order GRAPE kernels at
#       a dt small enough that their O(dt·‖H‖) error is below the tolerance);
#   (c) cross-kernel agreement on the same pulse (SU(2) vs the general
#       propagator path, spin-½ vs SU(2));
#   (d) the bugs fixed on the way: spin-I and Trotter state-transfer gradients
#       (co-state conjugated the wrong way, factor 2 missing), the MAS/EPR powder
#       gradient ignoring the target metric, DM-Real clamped to [0, 1].

using Test
using Pulsar
using LinearAlgebra
using Random

@testset "Fidelity metrics in every kernel" begin

    σx = ComplexF64[0 1; 1 0]; σy = ComplexF64[0 -im; im 0]; σz = ComplexF64[1 0; 0 -1]
    Ix, Iy, Iz = σx / 2, σy / 2, σz / 2

    function fd(f, x; h = 1e-6)
        g = similar(x, Float64)
        for i in eachindex(x)
            xp = copy(x); xm = copy(x); xp[i] += h; xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2h)
        end
        return g
    end
    relerr(G, Gref) = maximum(abs.(G .- Gref)) / max(maximum(abs, Gref), 1e-14)

    @testset "primitive: dF = Re(w · dz)" begin
        rng = MersenneTwister(1)
        h = 1e-7
        for m in (:prop_real, :prop_modulus, :prop_squared, :state_squared_haar, :prop_real_quad,
                  :state_real, :state_modulus, :state_squared,
                  :dm_real, :dm_modulus, :dm_squared), _ in 1:20
            z = 0.8 * randn(rng, ComplexF64); dz = randn(rng, ComplexF64)
            dF = (overlap_fidelity(m, z + h * dz; dim = 3) -
                  overlap_fidelity(m, z - h * dz; dim = 3)) / (2h)
            @test isapprox(dF, real(overlap_weight(m, z; dim = 3) * dz); atol = 1e-6)
        end
        # DM row evaluated from a pure-state overlap: z_D = |z_S|²
        for m in (:dm_real, :dm_modulus, :dm_squared), _ in 1:20
            z = 0.8 * randn(rng, ComplexF64); dz = randn(rng, ComplexF64)
            dF = (Pulsar.pure_state_fidelity(m, z + h * dz) -
                  Pulsar.pure_state_fidelity(m, z - h * dz)) / (2h)
            @test isapprox(dF, real(Pulsar.pure_state_weight(m, z) * dz); atol = 1e-6)
        end
        @test overlap_fidelity(:state_squared_haar, 0.0; dim = 2) ≈ 1 / 3
        @test overlap_weight(:state_modulus, 0.0) == 0
        @test Pulsar.has_constant_weight(:state_real) && !Pulsar.has_constant_weight(:prop_squared)
        @test_throws ArgumentError overlap_fidelity(:dm_uhlmann, 0.5)
        @test_throws ArgumentError overlap_fidelity(:state_squared_haar, 0.5)   # needs dim
    end

    @testset "metric arithmetic does not allocate (it runs per step × control)" begin
        # 10⁵ calls in a loop must not allocate per call (a single boxed result
        # of the measurement itself is all that may show up).
        function loop_alloc(f, n)
            z = 0.3 + 0.4im; x = 0.7 - 0.2im
            acc = 0.0
            for i in 1:n
                acc += real(f(z * (1 + 1e-9i), x))
            end
            return acc
        end
        for f in ((z, x) -> Pulsar.fidelity_grad_prefactor(z, x, 0.5, STATE_MODULUS),
                  (z, x) -> Pulsar.fidelity_grad_prefactor(z, x, 0.5, DM_SQUARED),
                  (z, x) -> Pulsar.fidelity_grad_prefactor(z, x, 0.5; type = :state_squared),
                  (z, x) -> lindblad_grad_prefactor(z, x, 0.5; type = :dm_modulus),
                  (z, x) -> overlap_weight(:state_squared_haar, z; dim = 4) +
                            Pulsar.pure_state_weight(:dm_real, z) * x)
            loop_alloc(f, 10)
            @test (@allocated loop_alloc(f, 100_000)) < 1_000
        end
    end

    @testset "DM-Real is the raw overlap (no [0, 1] clamp)" begin
        A = Iz / norm(Iz)
        @test state_fidelity(A, -A, DM_REAL) ≈ -1.0
        @test state_fidelity(A, -A, DM_MODULUS) ≈ 1.0
    end

    # ── SU(2): every propagator- and state-row metric ────────────────────────
    offs, RF, dt = [-3000.0, 0.0, 2500.0], 10_000.0, 2e-6
    rng = MersenneTwister(20261019)
    φ = 2π .* rand(rng, 20)
    function su2_reference(m, angle, φs)
        UT = exp(-im * σx * (angle / 2))                     # axis x̂ (Θ = π/2, Φ = 0)
        vals = map(offs) do off
            U = Matrix{ComplexF64}(I, 2, 2)
            for p in φs
                U = exp(-im * dt * (2π * off * Iz + 2π * RF * (cos(p) * Ix + sin(p) * Iy))) * U
            end
            V = UT' * U
            Pulsar._metric_level(m) === :prop ? overlap_fidelity(m, tr(V) / 2; dim = 2) :
                                                 overlap_fidelity(m, V[1, 1])
        end
        return sum(vals) / length(vals)
    end
    @testset "SU(2) kernel :$m" for m in Pulsar.SU2_METRICS
        k = SU2Kernel(offs, RF, dt, π / 2, π / 2, 0.0; metric = m)
        F, GJ = su2_fidelity_and_grad(k, φ)
        GJ = copy(GJ)
        @test F ≈ k(φ) atol = 1e-12
        @test F ≈ su2_reference(m, π / 2, φ) atol = 1e-10
        @test relerr(GJ, fd(p -> 1 - k(p), φ)) < 1e-5
    end

    # ── spin-I: state row (fixed gradient) and propagator row ────────────────
    @testset "spin-I kernel I=$Ival" for Ival in (0.5, 1.0, 1.5)
        d = Int(2Ival + 1)
        ψi = zeros(ComplexF64, d); ψi[1] = 1
        ψt = normalize(ComplexF64.(collect(1:d)) .+ im .* ComplexF64.(collect(d:-1:1)))
        Ut = ComplexF64.(exp(-im * (π / 2) * Matrix(Pulsar.spin_Sx(Ival))))
        for m in (:state_squared, :state_real, :state_modulus)
            k = SpinIKernel(offs, RF, dt, Ival, (; psi_init = ψi, psi_targ = ψt); metric = m)
            F, GJ = spinI_fidelity_and_grad(k, φ)
            @test F ≈ k(φ) atol = 1e-12
            @test relerr(GJ, fd(p -> 1 - k(p), φ)) < 1e-5
        end
        for m in (:prop_real, :prop_modulus, :prop_squared, :state_squared_haar, :prop_real_quad)
            k = SpinIKernel(offs, RF, dt, Ival, Ut; metric = m)
            F, GJ = spinI_fidelity_and_grad(k, φ)
            @test F ≈ k(φ) atol = 1e-12
            @test relerr(GJ, fd(p -> 1 - k(p), φ)) < 1e-5
            if Ival == 0.5      # spin-½ is the SU(2) kernel (same RF-phase convention)
                @test F ≈ SU2Kernel(offs, RF, dt, π / 2, π / 2, 0.0; metric = m)(φ) atol = 1e-10
            end
        end
    end

    # ── Trotter: state row (fixed gradient) ──────────────────────────────────
    @testset "Trotter kernel :$m" for m in Pulsar.TROTTER_METRICS
        I2 = Matrix{ComplexF64}(I, 2, 2)
        drifts = [2π * (δ * kron(Iz, I2) + 0.7δ * kron(I2, Iz) + 480.0 * kron(Iz, Iz))
                  for δ in (-800.0, 0.0, 900.0)]
        ψt = normalize(ComplexF64[0.2, 0.1im, 0.3, 1.0])
        k = TrotterKernel(drifts, [2π * 8e3, 2π * 8e3], 2e-6, 2, ComplexF64[1, 0, 0, 0], ψt;
                          metric = m)
        φ2 = 2π .* rand(MersenneTwister(5), 30)
        F, GJ = trotter_fidelity_and_grad(k, φ2)
        @test F ≈ k(φ2) atol = 1e-12
        @test relerr(GJ, fd(p -> 1 - k(p), φ2)) < 1e-5
    end

    # ── Lindblad (Liouville-space GRAPE): the whole DM row ────────────────────
    # Operator targets (traceless, HS-normalized) make the overlap signed, so
    # DM-Real, -Modulus and -Squared are genuinely different functions.
    @testset "Lindblad kernel :$m, relaxation=$jumps" for m in DM_METRICS, jumps in (false, true)
        jo, rates = jumps ? ([ComplexF64[0 1; 0 0], Iz], [30.0, 50.0]) :
                            (Matrix{ComplexF64}[], Float64[])
        ctrl = LindbladMRControl(; drifts = [2π * δ * Iz for δ in (-400.0, 300.0)],
            operators = [Ix, Iy], jump_ops = jo, decay_rates = rates,
            rho_init = [vec(Iz) / norm(Iz)], rho_targ = [vec(Ix .+ 0.3 .* Iy) / norm(Ix .+ 0.3 .* Iy)],
            pwr_levels = [2π * 2e3], pulse_dt = fill(2e-6, 14), fidelity = m,
            verbose = false, backend = :cpu)
        w = 0.6 .* randn(MersenneTwister(9), 2, 14)
        F, G = grape_lindblad_kernel(w, ctrl)
        Gfd = reshape(fd(x -> first(grape_lindblad_kernel(reshape(x, 2, 14), ctrl)), vec(w)), 2, 14)
        @test relerr(G, Gfd) < 2e-3          # first-order GRAPE, dt·‖H‖ ≈ 0.03
    end

    # ── MAS / EPR powder gradient helper: honors the target metric ──────────
    @testset "powder gradient helper, $(t.type) :$m" for (t, m) in vcat(
            # An SU(2) target: for σx (det −1) Tr(σx†U) is purely imaginary for
            # every SU(2) propagator, so Φ^P_R ≡ 0 and its gradient is 0.
            [(unitary_target(exp(-im * (π / 4) * σx); metric = m), m)
             for m in (:prop_real, :prop_modulus, :prop_squared, :state_squared_haar)],
            [(state_target(ComplexF64[0.6, 0.8im]; psi_init = ComplexF64[1, 0], metric = m), m)
             for m in (:state_real, :state_modulus, :state_squared, :dm_real, :dm_squared)])
        H0 = 2π * 300.0 * Iz; Hc = [Ix, Iy]; dtp = 1e-6; nt = 16
        u0 = 2π * 4e3 .* randn(MersenneTwister(3), 2, nt)
        steps(u) = (Us = zeros(ComplexF64, nt, 2, 2);
                    for k in 1:nt; Us[k, :, :] = exp(-im * dtp * (H0 + u[1, k] * Ix + u[2, k] * Iy)); end; Us)
        Ftot(u) = (Us = steps(u); U = Matrix{ComplexF64}(I, 2, 2);
                   for k in 1:nt; U = Us[k, :, :] * U; end; compute_fidelity(U, t))
        G = Pulsar._grape_gradient_from_propagators(steps(u0), Hc, t, dtp)
        Gfd = reshape(fd(x -> Ftot(reshape(x, 2, nt)), vec(u0); h = 1e-2), 2, nt)
        @test relerr(G, Gfd) < 2e-2          # first-order GRAPE, dt·‖H‖ ≈ 0.03
    end
end
