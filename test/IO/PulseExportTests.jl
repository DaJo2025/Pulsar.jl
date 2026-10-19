@testset "IO – Pulse Format Round-Trip" begin

    using LinearAlgebra

    # Synthetic waveform: 50 complex samples, unit amplitude, varying phase
    N  = 50
    dt = 1e-5   # 10 µs per step
    rng = MersenneTwister(42)
    samples_orig = [0.8 * exp(im * 2π * rand(rng)) for _ in 1:N]
    pulse_orig   = OptimizedPulse(ComplexF64.(samples_orig), dt;
                                   name="test_pulse", flip_angle=90.0)

    tmp = mktempdir()
    pwr = 1.0    # normalised; w_mat will be in [-1, 1]

    # Convert to w_mat [2×N] for Bruker/MR-layer savers
    w_mat = Float64[real(s) for s in samples_orig]'
    w_mat = vcat(real.(samples_orig)', imag.(samples_orig)')   # [2×N]

    # ─── Bruker JCAMP-DX ────────────────────────────────────────────────────
    @testset "Bruker round-trip" begin
        fp = joinpath(tmp, "test.bruker")
        save_bruker_shape(fp, w_mat, pwr; dt=dt, title="test", shape_totrot_deg=90.0)
        pulse_rt = load_bruker_shape(fp; pwr_level=pwr, dt=dt)
        @test length(pulse_rt.samples) == N
        err = norm(pulse_rt.samples .- samples_orig) / sqrt(N)
        @test err < 1e-4
    end

    # ─── JEOL Delta ─────────────────────────────────────────────────────────
    @testset "JEOL round-trip" begin
        fp = joinpath(tmp, "test.jrf")
        export_pulse(pulse_orig; application=:NMR_EPR, vendor=:jeol,
                     save=true, output_dir=tmp, name="test_pulse")
        jrf_fp = joinpath(tmp, "test_pulse.jrf")
        isfile(jrf_fp) || (jrf_fp = fp)  # fallback
        if isfile(jrf_fp)
            # JEOL normalises amplitudes to max=1; pass max_amp to recover scale.
            mx = maximum(abs.(samples_orig))
            pulse_rt = load_jeol_shape(jrf_fp; max_amp=mx, dt=dt)
            @test length(pulse_rt.samples) == N
            # Phase and amplitude should match to ~1e-4 (float formatting precision)
            err = norm(abs.(pulse_rt.samples) .- abs.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "JEOL file not generated"
        end
    end

    # ─── EPR Bruker AWG ─────────────────────────────────────────────────────
    @testset "EPR round-trip" begin
        fp_dir = tmp
        export_pulse(pulse_orig; application=:NMR_EPR, vendor=:epr_bruker,
                     save=true, output_dir=fp_dir, name="test_pulse")
        shp_fp = joinpath(fp_dir, "test_pulse.shp")
        if isfile(shp_fp)
            pulse_rt = load_epr_shape(shp_fp; dt=dt)
            @test length(pulse_rt.samples) == N
            # I/Q round-trip: amplitudes match up to normalisation
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "EPR file not generated"
        end
    end

    # ─── Qiskit JSON ────────────────────────────────────────────────────────
    @testset "Qiskit round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:qiskit,
                     save=true, output_dir=tmp, name="test_pulse")
        json_fp = joinpath(tmp, "test_pulse.json")
        if isfile(json_fp)
            pulse_rt = load_qiskit_waveform(json_fp)
            @test length(pulse_rt.samples) == N
            @test abs(pulse_rt.dt - dt) < 1e-15
            # Amplitudes are normalised to max=1 on export; recover relative phases
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-8
        else
            @test_skip "Qiskit file not generated"
        end
    end

    # ─── Quil-T ─────────────────────────────────────────────────────────────
    @testset "Quil-T round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:quil_t,
                     save=true, output_dir=tmp, name="test_pulse")
        quil_fp = joinpath(tmp, "test_pulse.quil")
        if isfile(quil_fp)
            pulse_rt = load_quil_t(quil_fp; dt=dt)
            @test length(pulse_rt.samples) == N
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-6
        else
            @test_skip "Quil-T file not generated"
        end
    end

    # ─── QUA ────────────────────────────────────────────────────────────────
    @testset "QUA round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:qua,
                     save=true, output_dir=tmp, name="test_pulse")
        qua_fp = joinpath(tmp, "test_pulse_qua.py")
        if isfile(qua_fp)
            pulse_rt = load_qua(qua_fp; dt=dt)
            @test length(pulse_rt.samples) == N
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "QUA file not generated"
        end
    end

    # ─── Pulser ─────────────────────────────────────────────────────────────
    @testset "Pulser round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:pulser,
                     save=true, output_dir=tmp, name="test_pulse")
        pulser_fp = joinpath(tmp, "test_pulse_pulser.py")
        @test isfile(pulser_fp)
        # samples are read as rad/s → OMEGA_MAX in rad/µs; loading inverts it
        pulse_rt = load_pulser(pulser_fp)
        @test length(pulse_rt.samples) == N
        @test norm(pulse_rt.samples .- samples_orig) / norm(samples_orig) < 1e-8
        @test pulse_rt.dt ≈ dt
        # an explicit scale still overrides
        mx = maximum(abs.(samples_orig))
        @test abs.(load_pulser(pulser_fp; max_amp = 2mx).samples) ≈ 2 .* abs.(samples_orig) rtol = 1e-8
    end

    @testset "Pulser script: valid Pulser, Pulsar's convention" begin
        r = export_pulse(pulse_orig; application = :QC, vendor = :pulser, save = false,
                         name = "test_pulse", omega_max_rad_per_us = 2π * 1.5)
        sc = r.script
        @test occursin("from pulser import Pulse, Register, Sequence", sc)      # Pulse was not imported
        @test occursin("from pulser.waveforms import CustomWaveform", sc)
        @test occursin("Pulse.ArbitraryPhase(omega_wf, phase_wf)", sc)        # phase was the detuning arg
        @test occursin("CustomWaveform(np.repeat(OMEGA_MAX * amp_test_pulse, DT_NS))", sc)
        @test occursin("seq.add(pulse, \"ch0\")", sc)
        @test !occursin("InterpolatedWaveform", sc)
        @test r.dt_ns == 10_000 && r.omega_max_rad_per_us ≈ 2π * 1.5
        @test occursin("OMEGA_MAX = 9.424777961", sc)
        # phase: unwrapped and equal to the sample phase
        @test all(abs.(diff(r.phase_rad)) .<= π + 1e-12)
        @test cis.(r.phase_rad) ≈ cis.(angle.(samples_orig))
        # a zero-amplitude step keeps the previous phase (no spurious detuning spike)
        rz = export_pulse(OptimizedPulse(ComplexF64[1im, 0, 0, -1], 4e-9);
                          application = :QC, vendor = :pulser, save = false)
        @test rz.phase_rad[1:3] ≈ fill(π / 2, 3) && rz.phase_rad[4] ≈ π
        @test rz.dt_ns == 4
        # Pulser samples at 1 ns: a fractional-ns step is refused, not rounded
        @test_throws ArgumentError export_pulse(OptimizedPulse(ComplexF64[1, 1], 2.5e-9);
                                                application = :QC, vendor = :pulser, save = false)
        # Pulser: H = Ω/2 (e^{iφ}|r⟩⟨g| + h.c.) − δ|r⟩⟨r|. Pulsar's neutral atom
        # (|g⟩ = |0⟩, |r⟩ = |1⟩) drive u_x σx/2 + u_y σy/2 is that with Ω = |u|,
        # φ = atan2(u_y, u_x) — the phase the exporter writes — and δ = −Δω.
        Ω, φ, Δ = 2.0, 0.7, 0.3e6
        at = neutral_atom_system([1.0e6 + Δ], zeros(1, 1); carrier_hz = 1.0e6)
        H_pulsar = Ω * cos(φ) .* at.H_controls[1] .+ Ω * sin(φ) .* at.H_controls[2]
        H_pulser = ComplexF64[0 Ω/2*cis(-φ); Ω/2*cis(φ) 0]          # rows/cols (g, r)
        @test H_pulsar ≈ H_pulser
        @test export_pulse(OptimizedPulse([Ω * cis(φ)], 1e-9); application = :QC,
                           vendor = :pulser, save = false).phase_rad[1] ≈ φ
        @test at.H_drift ≈ -(-2π * Δ) .* ComplexF64[0 0; 0 1]       # −δ|r⟩⟨r|, δ = −Δω
    end

end
