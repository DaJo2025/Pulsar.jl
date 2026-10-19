# ============================================================================
# Utilities/PulseAnalysis.jl  —  Theme 12: post-optimisation analysis tools
#
# Helpers for diagnosing optimised pulses **after** the optimiser has finished:
# spectral content, effective bandwidth, peak amplitude / energy, and a
# Bloch-trajectory stress test that sweeps a single-qubit fidelity surface
# over offset and drive-amplitude perturbations.
#
# Pure post-processing: no dependency on the optimiser internals beyond what
# is already exported by Pulsar.
# ============================================================================

using LinearAlgebra
using Statistics

# ---------------------------------------------------------------------------
# pulse_spectrum
# ---------------------------------------------------------------------------

"""
    pulse_spectrum(w, dt; sided=:two) → (freqs, mag)

Per-channel FFT magnitude spectrum of a control waveform.

# Arguments
- `w  :: AbstractMatrix{<:Real}` — `[n_ctrl × n_t]` (Pulsar convention) or
                                   `[n_t × n_ctrl]`.  The longer axis is
                                   treated as time.
- `dt :: Real`                   — time-step duration (s).

# Keyword arguments
- `sided :: Symbol` — `:two` (default) returns the full two-sided FFT range
  with `freqs ∈ [−f_Nyq, f_Nyq)`; `:one` returns the non-negative half
  `freqs ∈ [0, f_Nyq]` with magnitudes doubled (except DC and Nyquist).

# Returns
- `freqs :: Vector{Float64}` — frequency axis in Hz.
- `mag   :: Matrix{Float64}` — `[n_freqs × n_ctrl]` magnitude spectrum
  (`abs.(rfft(w_k))` per channel, normalised by `n_t`).

The returned magnitudes are **un-windowed** (rectangular window).  For
qualitative bandwidth diagnostics that is sufficient; for precise
out-of-band suppression measurements the caller should multiply `w` by a
Hann/Blackman window before calling this function.
"""
function pulse_spectrum(w::AbstractMatrix{<:Real}, dt::Real;
                        sided::Symbol = :two)
    sided ∈ (:one, :two) ||
        throw(ArgumentError("sided must be :one or :two (got :$sided)"))
    dt > 0 || throw(ArgumentError("dt must be positive (got $dt)"))

    # Orient as [n_t × n_ctrl] — the time axis is the longer one.
    nrow, ncol = size(w)
    W = nrow >= ncol ? Matrix{Float64}(w) : Matrix{Float64}(transpose(w))
    n_t    = size(W, 1)
    n_ctrl = size(W, 2)
    n_t ≥ 2 || throw(ArgumentError("waveform has $n_t time steps, need ≥ 2"))

    # Two-sided FFT shifted so DC sits in the middle of the array.
    F   = fft_columns(W) ./ n_t
    f0  = 1.0 / (n_t * dt)
    if sided === :two
        freqs = Float64[(k - 1 - n_t ÷ 2) * f0 for k in 1:n_t]
        mag   = abs.(fftshift_columns(F))
        return freqs, mag
    else
        n_pos    = n_t ÷ 2 + 1
        freqs    = Float64[(k - 1) * f0 for k in 1:n_pos]
        mag_full = abs.(F)
        mag      = mag_full[1:n_pos, :]
        # Double the bins between DC and Nyquist to recover one-sided power.
        last_idx = iseven(n_t) ? n_pos - 1 : n_pos
        mag[2:last_idx, :] .*= 2.0
        return freqs, mag
    end
end

# Lightweight column-wise FFT.  We avoid pulling FFTW.jl in to keep
# Utilities free of optional deps; the naive O(n²) DFT is fine for
# diagnostic-grade pulse lengths (≤ 2048 steps in practice).  If a user
# pushes past that they can multiply by a window and call FFTW directly.
function fft_columns(W::AbstractMatrix{<:Real})
    n_t, n_ctrl = size(W)
    F = Matrix{ComplexF64}(undef, n_t, n_ctrl)
    twoπ_n = 2π / n_t
    @inbounds for k in 0:(n_t - 1)
        for j in 1:n_ctrl
            acc = ComplexF64(0)
            @simd for n in 0:(n_t - 1)
                acc += W[n + 1, j] * cis(-twoπ_n * k * n)
            end
            F[k + 1, j] = acc
        end
    end
    return F
end

# fftshift along axis 1: rotate rows so that the DC bin (index 1) moves to
# the centre.  Equivalent to `circshift(F, (n_t ÷ 2, 0))`.
function fftshift_columns(F::AbstractMatrix)
    n_t = size(F, 1)
    s   = -(n_t ÷ 2)
    return circshift(F, (s, 0))
end

# ---------------------------------------------------------------------------
# pulse_bandwidth
# ---------------------------------------------------------------------------

"""
    pulse_bandwidth(w, dt; thresh=0.1) → bw_hz

Effective two-sided bandwidth of `w` per channel.  For each channel, the
returned bandwidth is `2 · max(|f|)` over frequencies `f` where the
normalised magnitude exceeds `thresh × max(magnitude)`.

`thresh` defaults to 0.1 (10 % of peak), which roughly matches the −20 dB
cutoff used in NMR pulse-design literature.

Returns `Vector{Float64}` of length `n_ctrl`.
"""
function pulse_bandwidth(w::AbstractMatrix{<:Real}, dt::Real;
                         thresh::Real = 0.1)
    0 < thresh ≤ 1 ||
        throw(ArgumentError("thresh must be in (0, 1] (got $thresh)"))
    freqs, mag = pulse_spectrum(w, dt; sided = :two)
    n_ctrl = size(mag, 2)
    out    = zeros(Float64, n_ctrl)
    for k in 1:n_ctrl
        peak = maximum(@view mag[:, k])
        peak == 0 && continue
        cut  = thresh * peak
        mask = mag[:, k] .> cut
        any(mask) || continue
        out[k] = 2 * maximum(abs.(freqs[mask]))
    end
    return out
end

# ---------------------------------------------------------------------------
# pulse_summary
# ---------------------------------------------------------------------------

"""
    pulse_summary(w, dt) → NamedTuple

Quick diagnostic summary of a control waveform.  Returns a `NamedTuple` with:

- `peak_amp        :: Float64`       — `maximum(abs.(w))` (per-channel max collapsed).
- `rms_amp         :: Vector{Float64}` — per-channel root-mean-square.
- `total_energy    :: Float64`       — `Σ_n,k w[k,n]² · dt`.
- `bandwidth_hz    :: Vector{Float64}` — per-channel `pulse_bandwidth` at 10 %.
- `max_slew        :: Float64`       — `maximum(abs.(diff(w; dims=2))) / dt`.
"""
function pulse_summary(w::AbstractMatrix{<:Real}, dt::Real)
    nrow, ncol = size(w)
    W = nrow >= ncol ? Matrix{Float64}(w) : Matrix{Float64}(transpose(w))
    # W is [n_t × n_ctrl]; expose per-channel statistics in n_ctrl-shaped vectors.
    peak_amp     = maximum(abs.(W))
    rms_amp      = vec(sqrt.(mean(W.^2; dims = 1)))
    total_energy = sum(W.^2) * dt
    bw_hz        = pulse_bandwidth(transpose(W), dt; thresh = 0.1)
    max_slew     = if size(W, 1) > 1
        maximum(abs.(diff(W; dims = 1))) / dt
    else
        0.0
    end
    return (peak_amp = peak_amp, rms_amp = rms_amp,
            total_energy = total_energy, bandwidth_hz = bw_hz,
            max_slew = max_slew)
end

# ---------------------------------------------------------------------------
# bloch_sweep_fidelity
# ---------------------------------------------------------------------------

"""
    bloch_sweep_fidelity(w, dt, ψ_init, ψ_target;
                          offsets_hz, b1_factors, operators, drift_op,
                          fidelity = :state_squared) → Matrix{Float64}

Single-qubit Bloch-trajectory stress test.  Re-simulates the waveform on a
2-D grid `(detuning Δf, drive-amplitude factor B₁)` and returns the
fidelity surface `[length(offsets_hz) × length(b1_factors)]`.

Each cell evaluates `F(Δf, B₁)` for the perturbed Hamiltonian

    H(t; Δf, B₁) = 2π·Δf · drift_op + B₁ · Σ_k w[k,n] · operators[k]

with the optimised waveform `w` reapplied unchanged.  Useful for diagnosing
how robust an optimised pulse is to off-resonance / B₁ inhomogeneity.

# Arguments
- `w :: AbstractMatrix{<:Real}` — `[n_ctrl × n_t]` waveform.
- `dt :: Real`                  — time step (s).
- `ψ_init, ψ_target`            — 2-element complex vectors.
- `offsets_hz`                  — vector of detuning samples (Hz).
- `b1_factors`                  — vector of dimensionless drive scales (1.0 = nominal).
- `operators`                   — `Vector{Matrix{ComplexF64}}` (length `n_ctrl`).
- `drift_op`                    — single 2×2 Hermitian operator scaled by `2π·Δf`.
- `fidelity`                    — `:state_squared` (Φ^S_Q, default) for
                                  `|⟨ψ_t|ψ⟩|²`, `:state_real` (Φ^S_R) for
                                  `Re⟨ψ_t|ψ⟩`.

# Returns
`Matrix{Float64}` — `[length(offsets_hz) × length(b1_factors)]` fidelity grid.
"""
function bloch_sweep_fidelity(
    w           :: AbstractMatrix{<:Real},
    dt          :: Real,
    ψ_init      :: AbstractVector,
    ψ_target    :: AbstractVector;
    offsets_hz  :: AbstractVector{<:Real},
    b1_factors  :: AbstractVector{<:Real},
    operators   :: AbstractVector{<:AbstractMatrix},
    drift_op    :: AbstractMatrix,
    fidelity    :: Symbol = :state_squared,
)
    fidelity = canonical_metric(fidelity, (:state_squared, :state_real);
                                context = "bloch_sweep_fidelity")
    n_ctrl, n_t = size(w)
    length(operators) == n_ctrl ||
        throw(ArgumentError("operators length $(length(operators)) ≠ n_ctrl $n_ctrl"))

    dim = length(ψ_init)
    dim == length(ψ_target) ||
        throw(DimensionMismatch("ψ_init and ψ_target must share length"))
    size(drift_op) == (dim, dim) ||
        throw(DimensionMismatch("drift_op must be $(dim)×$(dim)"))

    Ops = [Matrix{ComplexF64}(O) for O in operators]
    H0  = Matrix{ComplexF64}(drift_op)
    ψi  = Vector{ComplexF64}(ψ_init)
    ψt  = Vector{ComplexF64}(ψ_target)

    F = zeros(Float64, length(offsets_hz), length(b1_factors))
    H_buf = Matrix{ComplexF64}(undef, dim, dim)
    ψ_cur = Vector{ComplexF64}(undef, dim)
    ψ_nxt = Vector{ComplexF64}(undef, dim)

    @inbounds for (i, Δf) in enumerate(offsets_hz)
        for (j, B1) in enumerate(b1_factors)
            ψ_cur .= ψi
            for n in 1:n_t
                copyto!(H_buf, H0)
                H_buf .*= (2π * Float64(Δf))
                for k in 1:n_ctrl
                    α = Float64(B1) * Float64(w[k, n])
                    LinearAlgebra.axpy!(α, Ops[k], H_buf)
                end
                Uk = compute_propagator(H_buf, Float64(dt))
                mul!(ψ_nxt, Uk, ψ_cur)
                ψ_cur, ψ_nxt = ψ_nxt, ψ_cur
            end
            z = dot(ψt, ψ_cur)
            F[i, j] = fidelity === :state_squared ? abs2(z) : real(z)
        end
    end
    return F
end

# ---------------------------------------------------------------------------
# parameter_jacobian — Theme 12: ∂(∇F) / ∂p around the optimum
# ---------------------------------------------------------------------------

"""
    parameter_jacobian(grad_fn, w_opt, params; h = 1e-5)
        → J::Matrix{Float64}

Finite-difference Jacobian of the gradient w.r.t. system parameters,
evaluated at the optimum.

Useful for the implicit-function theorem step:

    ∇_w F(w_opt(p), p) = 0  ⟹  ∂w_opt/∂p ≈ −H_ww⁻¹ · J(p)

The full sensitivity Jacobian `∂w_opt/∂p` is obtained by inverting the
Hessian `H_ww` against `J`; this function returns only the cross-term
`J = ∂(∇_w F)/∂p`, leaving the Hessian solve to the caller (it is typically
supplied by the optimiser's L-BFGS approximation rather than recomputed).

# Arguments
- `grad_fn :: Function` — `(w, p) -> ∇_w F(w, p) :: Matrix{Float64}` matching
  the shape of `w_opt`.
- `w_opt   :: Matrix{Float64}` — optimised waveform.
- `params  :: AbstractVector{<:Real}` — current parameter vector.

# Keyword arguments
- `h       :: Real` — central-difference step (default `1e-5`).

# Returns
- `J :: Matrix{Float64}` of size `(length(w_opt), length(params))`, with each
  column holding `vec(∂(∇_w F)/∂p_i)`.
"""
function parameter_jacobian(grad_fn, w_opt::AbstractMatrix{<:Real},
                            params::AbstractVector{<:Real}; h::Real = 1e-5)
    h > 0 || throw(ArgumentError("h must be positive (got $h)"))
    n_w  = length(w_opt)
    n_p  = length(params)
    J    = Matrix{Float64}(undef, n_w, n_p)
    p    = Vector{Float64}(params)
    for i in 1:n_p
        p_plus = copy(p);  p_plus[i]  += h
        p_minus = copy(p); p_minus[i] -= h
        g_plus  = grad_fn(w_opt, p_plus)
        g_minus = grad_fn(w_opt, p_minus)
        size(g_plus) == size(w_opt) ||
            throw(DimensionMismatch(
                "grad_fn returned $(size(g_plus)) ≠ size(w_opt)=$(size(w_opt))"))
        J[:, i] = vec((g_plus .- g_minus) ./ (2h))
    end
    return J
end
