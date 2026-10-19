# Types/TrappedIonSystem.jl
# TrappedIonSystem: trapped-ion qubit type.
#
# Provides:
#   TrappedIonSystem            — struct: laser-driven ions with motional modes
#   trapped_ion_system(...)     — constructor

using LinearAlgebra

# ============================================================================
# TrappedIonSystem
# ============================================================================

"""
    TrappedIonSystem <: AbstractQuantumSystem

Physical model for a chain of laser-driven trapped-ion qubits.

Each ion is a two-level system (qubit) driven through its coupling to shared
motional (phonon) modes.  The Lamb-Dicke parameters `eta[i,k]` quantify the
coupling strength of ion `i` to mode `k`.

# Fields
- `H_drift`      — dim×dim drift Hamiltonian (rad/s, rotating frame)
- `H_controls`   — 2·n_ions control operators (σx and σy per ion)
- `dim`          — Hilbert space dimension (2^n_ions)
- `n_controls`   — number of control channels (2·n_ions)
- `n_ions`       — number of ions
- `freq_hz`      — Vector{Float64} qubit (electronic) transition frequencies / 2π (Hz)
- `eta`          — Matrix{Float64} Lamb-Dicke parameters (n_ions × n_modes)
- `mode_freq_hz` — Vector{Float64} motional mode frequencies / 2π (Hz)
- `Omega_hz`     — Float64 peak Rabi frequency / 2π (Hz)
- `T1_s`         — Float64 spontaneous emission lifetime (s)
- `collapse_ops` — Vector{Matrix{ComplexF64}} Lindblad operators
- `metadata`     — Dict{String,Any}
"""
struct TrappedIonSystem <: AbstractQuantumSystem
    H_drift      :: Matrix{ComplexF64}
    H_controls   :: Vector{Matrix{ComplexF64}}
    dim          :: Int
    n_controls   :: Int
    n_ions       :: Int
    freq_hz      :: Vector{Float64}
    eta          :: Matrix{Float64}
    mode_freq_hz :: Vector{Float64}
    Omega_hz     :: Float64
    T1_s         :: Float64
    collapse_ops :: Vector{Matrix{ComplexF64}}
    metadata     :: Dict{String,Any}
end

"""
    trapped_ion_system(freq_hz, eta, mode_freq_hz;
                       Omega_hz=10e3, T1_s=Inf,
                       carrier_hz=nothing, metadata=Dict()) -> TrappedIonSystem

Construct a [`TrappedIonSystem`](@ref).

The rotating-frame drift Hamiltonian is:

    H = Σ_i Δω_i/2 σz_i

where `Δω_i = ω_i − ω_c` is the detuning of ion `i` from the carrier.

Control operators (per ion, in the global rotating frame):
    H_x_i = σx_i / 2
    H_y_i = σy_i / 2

# Arguments
- `freq_hz`      — qubit transition frequencies / 2π (Hz)
- `eta`          — Lamb-Dicke parameters matrix (n_ions × n_modes)
- `mode_freq_hz` — motional mode frequencies / 2π (Hz)
- `Omega_hz`     — peak Rabi frequency / 2π (Hz; default 10 kHz)
- `T1_s`         — spontaneous emission T₁ (s)
- `carrier_hz`   — carrier frequency / 2π (Hz); defaults to mean(freq_hz)
- `extra_drift_terms` — Hermitian matrices (rad/s) appended to `H_drift`; use this to
                        add motional-mode couplings, red/blue sideband drives, or any
                        custom ion-ion interaction not covered by the default model.
- `custom_controls`   — if provided, **replaces** the default [σx/2, σy/2] per ion.
- `extra_jump_ops`, `extra_decay_rates` — append √γ·L Lindblad operators.
- `metadata`     — Dict{String,Any}

# Limitations
The default drift Hamiltonian is electronic-only (diagonal σz detuning). Motional-mode
coupling (via `eta`/`mode_freq_hz`) is **not** built in by default — supply it as an
`extra_drift_terms` entry (or a `custom_controls` operator) for MS, sideband, or
carrier-mediated gates.

# Example
```julia
sys = trapped_ion_system([1.0e6, 1.1e6], ones(2,1), [2.0e6])
@assert sys.n_ions == 2
```
"""
function trapped_ion_system(freq_hz      :: Vector{Float64},
                             eta          :: Matrix{Float64},
                             mode_freq_hz :: Vector{Float64};
                             Omega_hz     :: Float64              = 10e3,
                             T1_s         :: Float64              = Inf,
                             carrier_hz   :: Union{Float64, Nothing} = nothing,
                             extra_drift_terms :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                             custom_controls   :: Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
                             extra_jump_ops    :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                             extra_decay_rates :: Vector{Float64} = Float64[],
                             metadata     :: Dict{String,Any}     = Dict{String,Any}())::TrappedIonSystem

    n = length(freq_hz)
    dim = 2^n
    ω_c = isnothing(carrier_hz) ? 2π * mean(freq_hz) : 2π * Float64(carrier_hz)

    # Single-qubit Pauli matrices
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    σm = ComplexF64[0 0; 1 0]

    # Embed 2×2 operator on qubit i in n-qubit space
    function embed2(op, i, n)
        mats = [k == i ? op : Matrix{ComplexF64}(I, 2, 2) for k in 1:n]
        foldl(kron, mats)
    end

    # ── Drift Hamiltonian ────────────────────────────────────────────────────
    H_drift = zeros(ComplexF64, dim, dim)
    for i in 1:n
        Δω = 2π * freq_hz[i] - ω_c
        H_drift .+= (Δω / 2.0) .* embed2(σz, i, n)
    end
    for Hx in extra_drift_terms
        size(Hx) == (dim, dim) ||
            throw(ArgumentError("extra_drift_terms entry size $(size(Hx)) ≠ ($dim,$dim)"))
        H_drift .+= Hx
    end

    # ── Control operators ────────────────────────────────────────────────────
    H_controls = if custom_controls === nothing
        ops = Matrix{ComplexF64}[]
        for i in 1:n
            push!(ops, embed2(σx, i, n) ./ 2.0)
            push!(ops, embed2(σy, i, n) ./ 2.0)
        end
        ops
    else
        for H in custom_controls
            size(H) == (dim, dim) ||
                throw(ArgumentError("custom_controls entry size $(size(H)) ≠ ($dim,$dim)"))
        end
        copy(custom_controls)
    end

    # ── Lindblad collapse operators ──────────────────────────────────────────
    collapse_ops = Matrix{ComplexF64}[]
    if isfinite(T1_s) && T1_s > 0.0
        for i in 1:n
            push!(collapse_ops, sqrt(1.0 / T1_s) .* embed2(σm, i, n))
        end
    end
    length(extra_jump_ops) == length(extra_decay_rates) ||
        throw(ArgumentError("extra_jump_ops and extra_decay_rates must have equal length"))
    for (L, γ) in zip(extra_jump_ops, extra_decay_rates)
        size(L) == (dim, dim) ||
            throw(ArgumentError("extra_jump_ops entry size $(size(L)) ≠ ($dim,$dim)"))
        γ >= 0.0 || throw(ArgumentError("extra_decay_rates must be non-negative"))
        γ > 0.0 && push!(collapse_ops, sqrt(γ) .* L)
    end

    TrappedIonSystem(
        H_drift, H_controls, dim, length(H_controls),
        n, freq_hz, eta, mode_freq_hz, Omega_hz, T1_s,
        collapse_ops, metadata,
    )
end

# ─── Operators for ensemble axes (see Types/EnsembleAveraging.jl) ─────────────

"""
    offset_operator(sys::TrappedIonSystem; qubits = :all) -> Matrix{ComplexF64}

Σ σz_i/2 over `qubits` — what a qubit-frequency shift δ (Hz, at a fixed carrier)
adds in `trapped_ion_system`, so `OffsetAxis(δs; operator = offset_operator(sys))`
equals `freq_hz .+ δ`. The same operator as the MR offset Σ Iz_i, and as the
neutral-atom and spin-qubit builders; the transmon and NV builders use the
number / projector form instead (opposite sign on the qubit subspace).
"""
function offset_operator(sys::TrappedIonSystem; qubits = :all)
    idx = qubits === :all ? (1:sys.n_ions) : qubits
    return sum(_embed_site(ComplexF64[0.5 0; 0 -0.5], i, sys.n_ions, 2) for i in idx)
end
