# ============================================================
# Pulsar.jl — GPU auto-setup tests (Runtime/GPUSetup.jl)
#
# Pure detection logic is tested unconditionally (no GPU needed).  The
# preference round-trip snapshots and restores LocalPreferences.toml so the
# suite leaves no side effects.
# ============================================================

using Test

@testset "GPU auto-setup" begin

    @testset "_decide_backend truth table" begin
        @test Pulsar._decide_backend(true,  false) === :metal
        @test Pulsar._decide_backend(false, true)  === :cuda
        @test Pulsar._decide_backend(false, false) === :cpu
        @test Pulsar._decide_backend(true,  true)  === :metal   # Metal wins on Apple Si
    end

    @testset "detect_gpu_hardware returns a valid backend" begin
        @test detect_gpu_hardware() in (:cpu, :cuda, :metal)
        # On Apple Silicon it must be :metal.
        if Sys.isapple() && Sys.ARCH === :aarch64
            @test detect_gpu_hardware() === :metal
        end
    end

    @testset "gpu_setup_status fields" begin
        st = gpu_setup_status()
        @test st.detected in (:cpu, :cuda, :metal)
        @test st.preference in (:cpu, :cuda, :metal)
        @test st.auto_load isa Bool
        @test st.cuda_installed isa Bool
        @test st.metal_installed isa Bool
        @test st.active_device in (:cpu, :cuda, :metal)
    end

    @testset "_gpu_pkg_loadable" begin
        @test Pulsar._gpu_pkg_loadable("Pkg")              # stdlib, always loadable
        @test !Pulsar._gpu_pkg_loadable("NoSuchPkg_xyz123")
    end

    @testset "_install_blocked respects guards" begin
        withenv("PULSAR_NO_GPU_INSTALL" => "1") do
            blocked, _ = Pulsar._install_blocked()
            @test blocked
        end
        withenv("CI" => "true", "PULSAR_NO_GPU_INSTALL" => nothing) do
            blocked, _ = Pulsar._install_blocked()
            @test blocked
        end
    end

    @testset "setup_gpu! preference round-trip (no install, no side effects)" begin
        lp = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
        had  = isfile(lp)
        snap = had ? read(lp, String) : nothing
        try
            # Configure CPU explicitly — benign, writes gpu_backend=cpu/auto=false.
            dev = setup_gpu!(device = :cpu, install = false, set_default = false)
            @test dev === :cpu
            @test Pulsar._preferred_device() === :cpu
            @test Pulsar._auto_load_pref() === false
            # Idempotent.
            @test setup_gpu!(device = :cpu, install = false, set_default = false) === :cpu
        finally
            if had
                write(lp, snap)
            elseif isfile(lp)
                rm(lp; force = true)
            end
        end
    end
end
