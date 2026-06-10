# ============================================================
# Pulsar.jl — Batched GPU primitive tests
#
# Validates the device-dispatched batched primitive layer
# (Backend/BatchedPrimitives.jl) and its wiring into the GRAPE kernels.
#
# CPU correctness runs unconditionally.  GPU correctness runs ONLY when a
# functional CUDA / Metal device is present (the extension has set the loaded
# flag); otherwise those testsets are skipped so the suite passes on CPU-only
# machines and in CI.
# ============================================================

using Test, LinearAlgebra, Random

@testset "Batched GPU primitives" begin

    rand_herm(d)  = (A = randn(ComplexF64, d, d); (A + A') / 2)
    rand_state(d) = (v = randn(ComplexF64, d); v ./ norm(v))

    cpu = Pulsar.cpu_backend()

    # ── CPU primitive correctness vs LAPACK references ────────────────────────
    @testset "CPU batched_herm_propagators == eigen reference" begin
        Random.seed!(11)
        for D in (2, 3, 4, 8)
            N = 64
            H = Array{ComplexF64,3}(undef, D, D, N)
            for n in 1:N
                H[:, :, n] = rand_herm(D)
            end
            dt = 0.137
            U = Pulsar.batched_herm_propagators(cpu, H, dt)
            err = 0.0
            for n in 1:N
                F = eigen(Hermitian(H[:, :, n]))
                Uref = F.vectors * Diagonal(exp.(-im .* F.values .* dt)) * F.vectors'
                err = max(err, maximum(abs.(U[:, :, n] .- Uref)))
            end
            @test err < 1e-12
        end
    end

    @testset "CPU batched_herm_propagators per-step dt vector" begin
        Random.seed!(12)
        D, N = 4, 50
        H = Array{ComplexF64,3}(undef, D, D, N)
        for n in 1:N
            H[:, :, n] = rand_herm(D)
        end
        dtv = collect(range(0.01, 0.3; length = N))
        U = Pulsar.batched_herm_propagators(cpu, H, dtv)
        F = eigen(Hermitian(H[:, :, 9]))
        Uref = F.vectors * Diagonal(exp.(-im .* F.values .* dtv[9])) * F.vectors'
        @test maximum(abs.(U[:, :, 9] .- Uref)) < 1e-12
    end

    @testset "CPU batched_nonherm_propagators == exp reference" begin
        Random.seed!(13)
        M, N = 4, 16
        A = Array{ComplexF64,3}(undef, M, M, N)
        for n in 1:N
            A[:, :, n] = randn(ComplexF64, M, M)
        end
        dt = 0.05
        U = Pulsar.batched_nonherm_propagators(cpu, A, dt)
        err = maximum(abs.(U[:, :, 3] .- exp(A[:, :, 3] .* dt)))
        @test err < 1e-12
    end

    @testset "CPU batched_build_hamiltonian == manual sum" begin
        ops = [Matrix{ComplexF64}([0 1; 1 0]), Matrix{ComplexF64}([0 -im; im 0])]
        drift = Matrix{ComplexF64}([1 0; 0 -1])
        coeff = [0.3 0.5 0.7; 0.1 0.2 0.4]
        H = Pulsar.batched_build_hamiltonian(cpu, drift, ops, coeff)
        Href = drift + coeff[1, 2] * ops[1] + coeff[2, 2] * ops[2]
        @test maximum(abs.(H[:, :, 2] .- Href)) < 1e-14
    end

    @testset "CPU batched_matvec (+adjoint)" begin
        Random.seed!(14)
        D, K, N = 4, 3, 40
        U = randn(ComplexF64, D, D, N)
        X = randn(ComplexF64, D, K, N)
        Y  = Pulsar.batched_matvec(cpu, U, X)
        Ya = Pulsar.batched_matvec(cpu, U, X; adjoint = true)
        @test maximum(abs.(Y[:, :, 7]  .- U[:, :, 7]  * X[:, :, 7])) < 1e-12
        @test maximum(abs.(Ya[:, :, 7] .- U[:, :, 7]' * X[:, :, 7])) < 1e-12
    end

    # ── Memory-budget chunking ────────────────────────────────────────────────
    @testset "batched_chunk_size honours the byte budget" begin
        @test Pulsar.batched_chunk_size(4, 10^6; mem_budget_bytes = Inf) == 10^6
        # 16·4²·4 = 1024 bytes/matrix; 1 MiB budget → ~1024 matrices
        c = Pulsar.batched_chunk_size(4, 10^6; mem_budget_bytes = 1024^2)
        @test 1 <= c <= 10^6
        @test c * 16 * 4 * 4 * 4 <= 1024^2 * 1.001
    end

    # ── Batch-volume-aware scheduling crossover ───────────────────────────────
    @testset "plan_hybrid_execution batch-volume crossover" begin
        p = Pulsar.HybridExecutionPlanner()
        # Many small matrices → GPU even at dim=4 (the pathological case).
        @test Pulsar.plan_hybrid_execution(4, 1818, true, p;
                  op = "propagator", batch_count = 763560) === :gpu
        # Small dim AND small batch → CPU.
        @test Pulsar.plan_hybrid_execution(4, 50, true, p; batch_count = 50) === :cpu
        # Large dim → GPU.
        @test Pulsar.plan_hybrid_execution(128, 200, true, p; batch_count = 200) === :gpu
        # No GPU → always CPU.
        @test Pulsar.plan_hybrid_execution(4, 1818, false, p; batch_count = 763560) === :cpu
    end

    @testset "resolve_backend falls back to CPU without a GPU" begin
        # On a CPU-only machine, :cuda/:metal resolve to CPUBackend.
        if !Pulsar.is_cuda_available()
            @test Pulsar.resolve_backend(:cuda) isa Pulsar.CPUBackend
        end
        if !Pulsar.is_metal_available()
            @test Pulsar.resolve_backend(:metal) isa Pulsar.CPUBackend
        end
        @test Pulsar.resolve_backend(:cpu) isa Pulsar.CPUBackend
    end

    # ── Production parity: _grape_batched(CPUBackend) == _grape_cpu ────────────
    @testset "_grape_batched matches _grape_cpu (bit-for-bit on CPU)" begin
        Random.seed!(21)
        dim, n_ctrl, n_t = 4, 2, 30
        drifts = [rand_herm(dim) for _ in 1:3]
        ops    = [rand_herm(dim) for _ in 1:n_ctrl]
        pwr    = [2π * 5000.0, 2π * 7000.0]
        rho_i  = [rand_state(dim) for _ in 1:2]
        rho_t  = [rand_state(dim) for _ in 1:2]
        dt     = fill(2e-6, n_t)
        for fid in (:real, :square)
            ctrl = Pulsar.MRControl(; drifts = drifts, operators = ops,
                rho_init = rho_i, rho_targ = rho_t, pwr_levels = pwr,
                pulse_dt = dt, fidelity = fid, verbose = false, fast_path = false)
            w = 0.3 .* randn(n_ctrl, n_t)
            Fc, Gc = Pulsar._grape_cpu(w, ctrl)
            Fb, Gb = Pulsar._grape_batched(w, ctrl, Pulsar.cpu_backend())
            @test abs(Fc - Fb) < 1e-10
            @test maximum(abs.(Gc .- Gb)) < 1e-10
        end
    end

    @testset "_maybe_batched_propagators == compute_propagators on CPU" begin
        Random.seed!(22)
        n_t, dim = 25, 4
        H = Array{ComplexF64,3}(undef, n_t, dim, dim)
        for k in 1:n_t
            H[k, :, :] = rand_herm(dim)
        end
        U1 = Pulsar._maybe_batched_propagators(H, 1e-6)
        U2 = Pulsar.compute_propagators(H, 1e-6)
        @test maximum(abs.(U1 .- U2)) < 1e-13
    end

    # ── GPU correctness (only when a functional device is present) ────────────
    gpu_dev = Pulsar.is_cuda_available() ? :cuda :
              (Pulsar.is_metal_available() ? :metal : :none)

    if gpu_dev === :none
        @info "GPUBatchedTests: no functional CUDA/Metal device — GPU testsets skipped."
    else
        @testset "GPU batched_herm_propagators == CPU ($(gpu_dev))" begin
            be = Pulsar.resolve_backend(gpu_dev)
            @test !(be isa Pulsar.CPUBackend)          # sentinel: device path active
            Random.seed!(31)
            for D in (2, 4, 8)
                N = 4096
                H = Array{ComplexF64,3}(undef, D, D, N)
                for n in 1:N
                    H[:, :, n] = rand_herm(D)
                end
                dt = 0.137
                Ug = Array(Pulsar.batched_herm_propagators(be, H, dt))
                Uc = Pulsar.batched_herm_propagators(Pulsar.cpu_backend(), H, dt)
                # Metal runs FP32 → looser tolerance; CUDA is FP64.
                tol = gpu_dev === :metal ? 1e-4 : 1e-12
                @test maximum(abs.(Ug .- Uc)) < tol
            end
        end

        if gpu_dev === :metal
            @testset "Metal FP32 expm kernel (use_fp32) vs CPU" begin
                be32 = Pulsar.metal_backend(use_fp32 = true, use_fp64_fallback = false)
                @test be32 isa Pulsar.MetalBackend            # sentinel
                Random.seed!(41)
                D, N = 4, 20_000                              # ≥ min-batch → kernel path
                H = Array{ComplexF64,3}(undef, D, D, N)
                for n in 1:N
                    H[:, :, n] = rand_herm(D) .* 2.0
                end
                dt = 0.05
                Uk = Pulsar.batched_herm_propagators(be32, H, dt)
                Uc = Pulsar.batched_herm_propagators(Pulsar.cpu_backend(), H, dt)
                @test maximum(abs.(Uk .- Uc)) < 1e-4          # FP32 accuracy
                # Default Metal backend must stay exact (FP64 CPU path).
                Ud = Pulsar.batched_herm_propagators(Pulsar.metal_backend(), H, dt)
                @test maximum(abs.(Ud .- Uc)) < 1e-12
            end
        end

        @testset "GPU _grape_batched == _grape_cpu ($(gpu_dev))" begin
            Random.seed!(32)
            dim, n_ctrl, n_t = 4, 2, 40
            drifts = [rand_herm(dim) for _ in 1:8]
            ops    = [rand_herm(dim) for _ in 1:n_ctrl]
            pwr    = [2π * 5000.0, 2π * 9000.0]
            rho_i  = [rand_state(dim) for _ in 1:2]
            rho_t  = [rand_state(dim) for _ in 1:2]
            dt     = fill(2e-6, n_t)
            ctrl = Pulsar.MRControl(; drifts = drifts, operators = ops,
                rho_init = rho_i, rho_targ = rho_t, pwr_levels = pwr,
                pulse_dt = dt, fidelity = :square, verbose = false, fast_path = false)
            w = 0.3 .* randn(n_ctrl, n_t)
            Fc, Gc = Pulsar._grape_cpu(w, ctrl)
            Fg, Gg = Pulsar._grape_batched(w, ctrl, Pulsar.resolve_backend(gpu_dev))
            tol = gpu_dev === :metal ? 1e-3 : 1e-8
            @test abs(Fc - Fg) < tol
            @test maximum(abs.(Gc .- Gg)) < tol
        end
    end
end
