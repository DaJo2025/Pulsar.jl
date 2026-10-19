# test/Utilities/PulseAnalysisTests.jl
# ====================================
# Theme 12 — pulse analysis utilities.
#
# Cover spectrum, bandwidth, summary, Bloch sweep, and the parameter Jacobian
# helper.  Each test pins to a hand-computable reference value.

using Test
using Pulsar
using LinearAlgebra
using Statistics

const _T12σX = ComplexF64[0 1; 1 0]
const _T12σY = ComplexF64[0 -im; im 0]
const _T12σZ = ComplexF64[1 0; 0 -1]

@testset "Theme 12 — pulse analysis utilities" begin

    @testset "pulse_spectrum — single-tone cosine peaks at f₀" begin
        # 100-step waveform sampling cos(2π f₀ t) at dt; expect a peak at ±f₀.
        n_t  = 256
        dt   = 1e-6
        f0   = 50_000.0                       # 50 kHz, well below Nyquist
        t    = (0:n_t-1) .* dt
        wave = cos.(2π .* f0 .* t)
        w    = reshape(wave, 1, n_t)          # [1 × n_t] (n_ctrl = 1)
        freqs, mag = pulse_spectrum(w, dt; sided = :two)
        peak_idx = argmax(mag[:, 1])
        @test abs(abs(freqs[peak_idx]) - f0) < 1.5 * (1 / (n_t * dt))

        # One-sided spectrum should peak at f₀ (positive freq only).
        freqs_one, mag_one = pulse_spectrum(w, dt; sided = :one)
        @test all(freqs_one .>= 0)
        peak_one = argmax(mag_one[:, 1])
        @test abs(freqs_one[peak_one] - f0) < 1.5 * (1 / (n_t * dt))
    end

    @testset "pulse_spectrum — Parseval (rough energy preservation)" begin
        # Σ |w_n|² ≈ N · Σ |W_k|² for unit-normalised DFT (DFT/N).
        n_t = 64
        dt  = 1e-6
        wave = randn(n_t)
        w    = reshape(wave, 1, n_t)
        freqs, mag = pulse_spectrum(w, dt; sided = :two)
        time_energy = sum(wave.^2)
        # Per fft normalisation: F = Σ x[n] e^{-iωn} / N, so Σ|F|² = Σ|x|² / N
        freq_energy = sum(mag[:, 1].^2) * n_t
        @test isapprox(time_energy, freq_energy; rtol = 1e-10)
    end

    @testset "pulse_bandwidth — narrow tone vs broadband" begin
        n_t = 256; dt = 1e-6
        # Narrow tone at 30 kHz.
        f0  = 30_000.0
        wn  = reshape(cos.(2π .* f0 .* (0:n_t-1) .* dt), 1, n_t)
        bw_n = pulse_bandwidth(wn, dt; thresh = 0.5)
        # 2 · f0 ≈ 60 kHz, with some FFT-bin slack.
        @test abs(bw_n[1] - 2.0 * f0) < 4.0 * (1 / (n_t * dt))

        # Broadband (white noise) bandwidth should hit the Nyquist range.
        wb = reshape(randn(n_t), 1, n_t)
        bw_b = pulse_bandwidth(wb, dt; thresh = 0.05)
        @test bw_b[1] > 0.5 * (1 / dt)
    end

    @testset "pulse_summary — fields and order of magnitude" begin
        n_t = 50; dt = 2e-6
        w   = 0.5 .* ones(2, n_t)              # constant waveform, peak = 0.5
        s   = pulse_summary(w, dt)
        @test s.peak_amp ≈ 0.5 atol = 1e-12
        @test all(s.rms_amp .≈ 0.5)
        @test s.total_energy ≈ 2 * n_t * 0.25 * dt atol = 1e-14
        @test length(s.bandwidth_hz) == 2
        @test s.max_slew == 0.0
    end

    @testset "pulse_summary — slew-rate detection" begin
        # Triangle ramp on one channel, zero on the other.
        n_t = 10; dt = 1e-6
        w   = zeros(2, n_t); w[1, :] .= range(0.0, 0.9; length = n_t)
        s   = pulse_summary(w, dt)
        # diff(W; dims=1) on [n_t × n_ctrl] picks up the ramp's per-step
        # increment 0.1, divided by dt = 1e-6 → 1e5.
        @test isapprox(s.max_slew, 0.1 / dt; rtol = 1e-3)
    end

    @testset "bloch_sweep_fidelity — identity pulse on resonance" begin
        # Zero waveform with zero offset → ψ stays at ψ_init; fidelity to ψ_init = 1.
        n_t = 20; dt = 1e-6
        w   = zeros(2, n_t)
        F   = bloch_sweep_fidelity(w, dt,
                  ComplexF64[1, 0], ComplexF64[1, 0];
                  offsets_hz = [0.0],
                  b1_factors = [1.0],
                  operators  = [_T12σX/2, _T12σY/2],
                  drift_op   = _T12σZ/2,
                  fidelity   = :state_squared)
        @test size(F) == (1, 1)
        @test F[1, 1] ≈ 1.0 atol = 1e-12
    end

    @testset "bloch_sweep_fidelity — π_x reaches |1⟩, sweeps drop off" begin
        # On-resonance π pulse (constant w_x = π / T) drives |0⟩ → |1⟩.
        n_t = 200; dt = 1e-7
        T   = n_t * dt
        Ω   = π / T
        w   = zeros(2, n_t); w[1, :] .= Ω
        F   = bloch_sweep_fidelity(w, dt,
                  ComplexF64[1, 0], ComplexF64[0, 1];
                  offsets_hz = [-2e6, 0.0, 2e6],
                  b1_factors = [0.5, 1.0],
                  operators  = [_T12σX/2, _T12σY/2],
                  drift_op   = _T12σZ/2,
                  fidelity   = :state_squared)
        # On-resonance, B1 = 1.0 ⇒ near-perfect inversion.
        @test F[2, 2] > 0.999
        # B1 = 0.5 ⇒ half flip ⇒ fidelity to |1⟩ much lower than at B1 = 1.
        @test F[2, 1] < F[2, 2]
        # Off-resonance ±2 MHz ⇒ fidelity drops vs on-resonance.
        @test F[1, 2] < F[2, 2]
        @test F[3, 2] < F[2, 2]
    end

    @testset "parameter_jacobian — analytic 1-parameter derivative" begin
        # F(w; p) = -½ ‖w - p·1‖²  ⟹  ∇_w F = -(w - p·1).
        # ⟹ ∂(∇_w F)/∂p = +1 (per element).
        # So J = vec(ones) for any w_opt, p.
        grad_fn = (w, p) -> -(w .- p[1])
        w_opt   = randn(2, 3)
        p       = [0.7]
        J       = parameter_jacobian(grad_fn, w_opt, p; h = 1e-4)
        @test size(J) == (length(w_opt), length(p))
        @test all(isapprox.(J, 1.0; atol = 1e-6))
    end

    @testset "parameter_jacobian — multi-parameter" begin
        # F(w; p) = ½ Σ (p[1] · w[k,1]² + p[2] · w[k,2]²)
        # ⟹ ∇_w F has entries (p[1] · w[k,1], p[2] · w[k,2]).
        # ⟹ ∂(∇_w F)/∂p[1] non-zero only on column 1 of w; similarly p[2].
        grad_fn = (w, p) -> begin
            g = similar(w)
            g[:, 1] .= p[1] .* w[:, 1]
            g[:, 2] .= p[2] .* w[:, 2]
            g
        end
        w_opt = randn(3, 2)
        p     = [1.5, 2.5]
        J     = parameter_jacobian(grad_fn, w_opt, p; h = 1e-5)
        # J[:, 1] should hold ∂(∇F)/∂p[1] = w[:, 1] in column-1 entries, 0 elsewhere.
        # Reshape into [3 × 2] view of the gradient layout.
        col1 = reshape(J[:, 1], 3, 2)
        col2 = reshape(J[:, 2], 3, 2)
        @test col1[:, 1] ≈ w_opt[:, 1] atol = 1e-6
        @test all(isapprox.(col1[:, 2], 0.0; atol = 1e-9))
        @test all(isapprox.(col2[:, 1], 0.0; atol = 1e-9))
        @test col2[:, 2] ≈ w_opt[:, 2] atol = 1e-6
    end

    @testset "Validation — argument errors" begin
        @test_throws ArgumentError pulse_spectrum(zeros(1, 8), -1e-6)
        @test_throws ArgumentError pulse_spectrum(zeros(1, 8), 1e-6; sided = :foo)
        @test_throws ArgumentError pulse_spectrum(zeros(1, 1), 1e-6)
        @test_throws ArgumentError pulse_bandwidth(zeros(1, 8), 1e-6; thresh = 0.0)
        @test_throws ArgumentError parameter_jacobian((w, p) -> w, randn(1, 1), [1.0]; h = 0.0)
    end
end
