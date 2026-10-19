# Types/NeutralAtomSystem.jl
# NeutralAtomSystem: neutral-atom (Rydberg) qubit type.
#
# Provides:
#   NeutralAtomSystem           — struct: atoms with Rydberg blockade interaction
#   neutral_atom_system(...)    — constructor

using LinearAlgebra

# ============================================================================
# NeutralAtomSystem
# ============================================================================

"""
    NeutralAtomSystem <: AbstractQuantumSystem

Physical model for an array of neutral atoms driven to Rydberg states.

Each atom is a two-level system (|g⟩ and |r⟩).  When `blockade_regime=true`
the Rydberg blockade interaction `V |rr⟩⟨rr|` is added to the drift Hamiltonian
for every pair of atoms, preventing simultaneous excitation.

# Fields
- `H_drift`        — dim×dim drift Hamiltonian (rad/s, rotating frame)
- `H_controls`     — 2·n_atoms control operators (σx and σy per atom)
- `dim`            — Hilbert space dimension (2^n_atoms)
- `n_controls`     — number of control channels (2·n_atoms)
- `n_atoms`        — number of atoms
- `freq_hz`        — Vector{Float64} qubit transition frequencies / 2π (Hz)
- `V_rydberg_hz`   — Matrix{Float64} pairwise Rydberg interaction strengths / 2π (Hz)
- `blockade_regime`— Bool; when true the strong-blockade projector is included
- `T1_s`           — Float64 spontaneous decay lifetime (s)
- `collapse_ops`   — Vector{Matrix{ComplexF64}} Lindblad operators
- `metadata`       — Dict{String,Any}
"""
struct NeutralAtomSystem <: AbstractQuantumSystem
    H_drift         :: Matrix{ComplexF64}
    H_controls      :: Vector{Matrix{ComplexF64}}
    dim             :: Int
    n_controls      :: Int
    n_atoms         :: Int
    freq_hz         :: Vector{Float64}
    V_rydberg_hz    :: Matrix{Float64}
    blockade_regime :: Bool
    T1_s            :: Float64
    collapse_ops    :: Vector{Matrix{ComplexF64}}
    metadata        :: Dict{String,Any}
end

"""
    neutral_atom_system(freq_hz, V_rydberg_hz;
                        blockade_regime=true, T1_s=Inf,
                        carrier_hz=nothing, metadata=Dict()) -> NeutralAtomSystem

Construct a [`NeutralAtomSystem`](@ref).

The rotating-frame drift Hamiltonian is:

    H = Σ_i Δω_i/2 σz_i  +  blockade_regime ? Σ_{i<j} V_{ij} |rr⟩⟨rr|_{ij} : 0

where `Δω_i = ω_i − ω_c` and `|r⟩` corresponds to the excited (Rydberg) state.

Control operators (per atom):
    H_x_i = σx_i / 2
    H_y_i = σy_i / 2

# Arguments
- `freq_hz`         — qubit transition frequencies / 2π (Hz)
- `V_rydberg_hz`    — pairwise interaction strength matrix / 2π (Hz)
- `blockade_regime` — include strong-blockade projector (default true)
- `T1_s`            — spontaneous decay lifetime (s)
- `carrier_hz`      — carrier frequency / 2π (Hz); defaults to mean(freq_hz)
- `extra_drift_terms` — Hermitian matrices (rad/s) appended to `H_drift`. Use this
                        to add finite-range vdW (C₆/r⁶) tails, dipole-dipole (C₃/r³)
                        terms, or anisotropic interactions beyond the blockade projector.
- `custom_controls`   — if provided, **replaces** the default [σx/2, σy/2] per atom.
- `extra_jump_ops`, `extra_decay_rates` — append √γ·L Lindblad operators.
- `metadata`        — Dict{String,Any}

# Example
```julia
sys = neutral_atom_system([4.0e6, 4.1e6], zeros(2,2))
@assert sys.n_atoms == 2
```
"""
function neutral_atom_system(freq_hz         :: Vector{Float64},
                              V_rydberg_hz    :: Matrix{Float64};
                              blockade_regime :: Bool                  = true,
                              T1_s            :: Float64               = Inf,
                              carrier_hz      :: Union{Float64, Nothing} = nothing,
                              extra_drift_terms :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                              custom_controls   :: Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
                              extra_jump_ops    :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                              extra_decay_rates :: Vector{Float64} = Float64[],
                              metadata        :: Dict{String,Any}      = Dict{String,Any}())::NeutralAtomSystem

    n   = length(freq_hz)
    dim = 2^n
    ω_c = isnothing(carrier_hz) ? 2π * mean(freq_hz) : 2π * Float64(carrier_hz)

    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    σm = ComplexF64[0 0; 1 0]
    # |r⟩⟨r| projector (excited state = |1⟩ in {|g⟩,|r⟩} basis)
    P_r = ComplexF64[0 0; 0 1]

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

    if blockade_regime
        for i in 1:n, j in i+1:n
            V = 2π * V_rydberg_hz[i, j]
            iszero(V) && continue
            # |rr⟩⟨rr| on pair (i,j)
            Prr = begin
                mats = [k == i || k == j ? P_r : Matrix{ComplexF64}(I, 2, 2) for k in 1:n]
                foldl(kron, mats)
            end
            H_drift .+= V .* Prr
        end
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

    NeutralAtomSystem(
        H_drift, H_controls, dim, length(H_controls),
        n, freq_hz, V_rydberg_hz, blockade_regime, T1_s,
        collapse_ops, metadata,
    )
end

# ─── Operators for ensemble axes (see Types/EnsembleAveraging.jl) ─────────────

"""
    offset_operator(sys::NeutralAtomSystem; qubits = :all) -> Matrix{ComplexF64}

Σ σz_i/2 over `qubits` — what a frequency shift δ (Hz, at a fixed carrier) adds
in `neutral_atom_system` (see `offset_operator(::TrappedIonSystem)`).
"""
function offset_operator(sys::NeutralAtomSystem; qubits = :all)
    idx = qubits === :all ? (1:sys.n_atoms) : qubits
    return sum(_embed_site(ComplexF64[0.5 0; 0 -0.5], i, sys.n_atoms, 2) for i in idx)
end

"""
    coupling_operator(sys::NeutralAtomSystem, i, j) -> Matrix{ComplexF64}

The blockade projector |rr⟩⟨rr| on atoms `i`, `j` that `V_rydberg_hz[i, j]`
multiplies (|r⟩ = |1⟩, as in the builder).
"""
function coupling_operator(sys::NeutralAtomSystem, i::Int, j::Int)
    P_r = ComplexF64[0 0; 0 1]
    return _embed_site(P_r, i, sys.n_atoms, 2) * _embed_site(P_r, j, sys.n_atoms, 2)
end
