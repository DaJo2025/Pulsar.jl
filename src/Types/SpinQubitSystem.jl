# Types/SpinQubitSystem.jl
# SpinQubitSystem: semiconductor spin qubit type.
#
# Provides:
#   SpinQubitSystem             — struct: exchange-coupled electron spins
#   spin_qubit_system(...)      — constructor

using LinearAlgebra

# ============================================================================
# SpinQubitSystem
# ============================================================================

"""
    SpinQubitSystem <: AbstractQuantumSystem

Physical model for semiconductor electron spin qubits coupled by exchange interaction.

Each qubit is an electron spin-1/2.  Nearest-neighbour pairs are coupled by the
isotropic Heisenberg exchange `J/4 (σx⊗σx + σy⊗σy + σz⊗σz)`.  A global
g-factor accounts for Zeeman splitting in an external field.

# Fields
- `H_drift`      — dim×dim drift Hamiltonian (rad/s, rotating frame)
- `H_controls`   — 2·n_spins control operators (σx and σy per spin)
- `dim`          — Hilbert space dimension (2^n_spins)
- `n_controls`   — number of control channels (2·n_spins)
- `n_spins`      — number of spin qubits
- `freq_hz`      — Vector{Float64} qubit Larmor frequencies / 2π (Hz)
- `J_hz`         — Matrix{Float64} exchange coupling strengths / 2π (Hz)
- `g_factor`     — Float64 effective g-factor (default 2.0)
- `T2_s`         — Float64 pure dephasing time (s)
- `collapse_ops` — Vector{Matrix{ComplexF64}} Lindblad operators
- `metadata`     — Dict{String,Any}
"""
struct SpinQubitSystem <: AbstractQuantumSystem
    H_drift      :: Matrix{ComplexF64}
    H_controls   :: Vector{Matrix{ComplexF64}}
    dim          :: Int
    n_controls   :: Int
    n_spins      :: Int
    freq_hz      :: Vector{Float64}
    J_hz         :: Matrix{Float64}
    g_factor     :: Float64
    T2_s         :: Float64
    collapse_ops :: Vector{Matrix{ComplexF64}}
    metadata     :: Dict{String,Any}
end

"""
    spin_qubit_system(freq_hz, J_hz;
                      g_factor=2.0, T2_s=Inf,
                      carrier_hz=nothing, metadata=Dict()) -> SpinQubitSystem

Construct a [`SpinQubitSystem`](@ref).

The rotating-frame drift Hamiltonian is:

    H = Σ_i Δω_i/2 σz_i  +  Σ_{i<j} J_{ij}/4 (σx_i σx_j + σy_i σy_j + σz_i σz_j)

where `Δω_i = ω_i − ω_c`.

Control operators (per spin):
    H_x_i = σx_i / 2
    H_y_i = σy_i / 2

# Arguments
- `freq_hz`    — qubit Larmor frequencies / 2π (Hz)
- `J_hz`       — exchange coupling matrix / 2π (Hz); only upper triangle used
- `g_factor`   — effective g-factor (default 2.0)
- `T2_s`       — pure dephasing time (s)
- `carrier_hz` — rotating-frame carrier / 2π (Hz); defaults to mean(freq_hz)
- `extra_drift_terms` — Hermitian matrices (rad/s) appended to `H_drift`. Use this for
                        longer-range exchange, spin-orbit, Dzyaloshinskii-Moriya, or
                        hyperfine coupling to nuclear bath not covered by nearest-
                        neighbour Heisenberg.
- `custom_controls`   — if provided, **replaces** the default [σx/2, σy/2] per spin.
                        Use to add a direct σz (detuning) control or a baseband J
                        control that modulates the exchange coupling.
- `extra_jump_ops`, `extra_decay_rates` — append √γ·L Lindblad operators (charge noise,
                        spectral diffusion, 1/f etc.).
- `metadata`   — Dict{String,Any}

# Limitations
The default drift is nearest-neighbour isotropic Heisenberg (dominant term). For
longer-range couplings or anisotropic XXZ / DM terms, inject via `extra_drift_terms`.

# Example
```julia
sys = spin_qubit_system([1.0e9, 1.1e9], zeros(2,2))
@assert sys.n_spins == 2
```
"""
function spin_qubit_system(freq_hz    :: Vector{Float64},
                            J_hz      :: Matrix{Float64};
                            g_factor  :: Float64               = 2.0,
                            T2_s      :: Float64               = Inf,
                            carrier_hz:: Union{Float64, Nothing} = nothing,
                            extra_drift_terms :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                            custom_controls   :: Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
                            extra_jump_ops    :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                            extra_decay_rates :: Vector{Float64} = Float64[],
                            metadata  :: Dict{String,Any}      = Dict{String,Any}())::SpinQubitSystem

    n   = length(freq_hz)
    dim = 2^n
    ω_c = isnothing(carrier_hz) ? 2π * mean(freq_hz) : 2π * Float64(carrier_hz)

    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]

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

    for i in 1:n, j in i+1:n
        J = 2π * J_hz[i, j]
        iszero(J) && continue
        H_drift .+= (J / 4.0) .* (embed2(σx, i, n) * embed2(σx, j, n) .+
                                   embed2(σy, i, n) * embed2(σy, j, n) .+
                                   embed2(σz, i, n) * embed2(σz, j, n))
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

    # ── Lindblad collapse operators (pure dephasing) ─────────────────────────
    collapse_ops = Matrix{ComplexF64}[]
    if isfinite(T2_s) && T2_s > 0.0
        for i in 1:n
            push!(collapse_ops, sqrt(1.0 / T2_s) .* embed2(σz, i, n) ./ 2.0)
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

    SpinQubitSystem(
        H_drift, H_controls, dim, length(H_controls),
        n, freq_hz, J_hz, g_factor, T2_s,
        collapse_ops, metadata,
    )
end

# ─── Operators for ensemble axes (see Types/EnsembleAveraging.jl) ─────────────

"""
    offset_operator(sys::SpinQubitSystem; qubits = :all) -> Matrix{ComplexF64}

Σ σz_i/2 over `qubits` — what a Zeeman-frequency shift δ (Hz, at a fixed carrier)
adds in `spin_qubit_system` (see `offset_operator(::TrappedIonSystem)`).
"""
function offset_operator(sys::SpinQubitSystem; qubits = :all)
    idx = qubits === :all ? (1:sys.n_spins) : qubits
    return sum(_embed_site(ComplexF64[0.5 0; 0 -0.5], i, sys.n_spins, 2) for i in idx)
end

"""
    coupling_operator(sys::SpinQubitSystem, i, j; kind = :isotropic) -> Matrix{ComplexF64}

The exchange operator `J_hz[i, j]` multiplies, (σx σx + σy σy + σz σz)/4 = S_i·S_j
(`:isotropic`, the builder's form and the MR J-coupling form), or σz σz/4 (`:zz`).
"""
function coupling_operator(sys::SpinQubitSystem, i::Int, j::Int; kind::Symbol = :isotropic)
    n = sys.n_spins
    σs = (ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1])
    pair(σ) = _embed_site(σ, i, n, 2) * _embed_site(σ, j, n, 2) ./ 4
    kind === :zz && return pair(σs[3])
    kind === :isotropic || throw(ArgumentError("kind must be :isotropic or :zz"))
    return sum(pair(σ) for σ in σs)
end
