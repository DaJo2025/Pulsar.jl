# Types/TransmonSystem.jl
# TransmonSystem: multi-level superconducting transmon qubit type.
#
# Provides:
#   TransmonSystem          — struct: multi-level transmon with anharmonicity
#   transmon_system(...)    — constructor

using LinearAlgebra

# ============================================================================
# TransmonSystem
# ============================================================================

"""
    TransmonSystem <: AbstractQuantumSystem

Physical model for a superconducting transmon qubit system.

Each qubit is modelled as a weakly anharmonic oscillator truncated at `n_levels`
energy levels.  The drift Hamiltonian (rotating frame) includes the anharmonicity
of each qubit and capacitive exchange coupling between neighbouring pairs.

# Fields
- `H_drift`         — dim×dim drift Hamiltonian (rad/s, rotating frame)
- `H_controls`      — 2·n_qubits control operators (x and y per qubit)
- `dim`             — total Hilbert space dimension (n_levels^n_qubits)
- `n_controls`      — number of control channels (2·n_qubits)
- `n_qubits`        — number of qubits
- `n_levels`        — number of levels per qubit (default 3)
- `n_comp`          — computational subspace dimension (2^n_qubits)
- `freq_hz`         — Vector{Float64} qubit frequencies / 2π (Hz)
- `anharm_hz`       — Vector{Float64} anharmonicities / 2π (Hz)
- `T1_s`            — Vector{Float64} T₁ relaxation times (s)
- `T2_s`            — Vector{Float64} T₂ coherence times (s)
- `coupling_hz`     — Matrix{Float64} exchange coupling strengths / 2π (Hz)
- `collapse_ops`    — Vector{Matrix{ComplexF64}} Lindblad operators
- `leakage_indices` — Vector{Int} indices outside the computational subspace
- `metadata`        — Dict{String,Any}
"""
struct TransmonSystem <: AbstractQuantumSystem
    H_drift        :: Matrix{ComplexF64}
    H_controls     :: Vector{Matrix{ComplexF64}}
    dim            :: Int
    n_controls     :: Int
    n_qubits       :: Int
    n_levels       :: Int
    n_comp         :: Int
    freq_hz        :: Vector{Float64}
    anharm_hz      :: Vector{Float64}
    T1_s           :: Vector{Float64}
    T2_s           :: Vector{Float64}
    coupling_hz    :: Matrix{Float64}
    collapse_ops   :: Vector{Matrix{ComplexF64}}
    leakage_indices:: Vector{Int}
    metadata       :: Dict{String,Any}
end

"""
    transmon_system(freq_hz, anharm_hz;
                    n_levels=3, coupling_hz=zeros(nq,nq),
                    T1_s=fill(Inf,nq), T2_s=fill(Inf,nq),
                    carrier_hz=nothing, metadata=Dict()) -> TransmonSystem

Construct a [`TransmonSystem`](@ref) from physical parameters.

The drift Hamiltonian (rotating frame) is built as:

    H = Σ_i [Δω_i n̂_i + α_i/2 n̂_i(n̂_i − 1)] + Σ_{i<j} g_{ij}(a†_i a_j + a_i a†_j)

where `Δω_i = ω_i − ω_c` (detuning from carrier), `α_i` is the anharmonicity,
and `g_{ij}` is the exchange coupling.

Control operators for qubit `i` are:
    H_x_i = (a_i + a†_i) / 2
    H_y_i = i(a†_i − a_i) / 2

# Arguments
- `freq_hz`     — qubit transition frequencies / 2π (Hz)
- `anharm_hz`   — anharmonicities / 2π (Hz; negative for transmon)
- `n_levels`    — energy levels per qubit (default 3)
- `coupling_hz` — exchange coupling matrix / 2π (Hz); only upper triangle used
- `T1_s`        — T₁ times (s)
- `T2_s`        — T₂ total coherence times (s). Convention: 1/T₂ = 1/(2T₁) + 1/T_φ,
                  so T₂ ≤ 2T₁. Pass the coherence-decay T₂ (Hahn-echo or FID envelope),
                  **not** the free-induction T₂*.
- `carrier_hz`  — rotating-frame carrier (Hz); defaults to mean(freq_hz)
- `extra_drift_terms` — list of extra Hermitian matrices (rad/s) appended to H_drift
- `custom_controls`   — if provided, **replaces** the default [(a+a†)/2, i(a†−a)/2] per qubit
- `extra_jump_ops`    — user-supplied Lindblad jump operators L (dim×dim)
- `extra_decay_rates` — decay rates γ (s⁻¹) associated with each `extra_jump_ops` entry;
                        the appended collapse operator is √γ · L
- `metadata`    — Dict{String,Any}

# Example
```julia
sys = transmon_system([5.0e9, 5.2e9], [-0.2e9, -0.2e9])
@assert sys.n_qubits == 2
```
"""
function transmon_system(freq_hz     :: Vector{Float64},
                         anharm_hz   :: Vector{Float64};
                         n_levels    :: Int                  = 3,
                         coupling_hz :: Matrix{Float64}      = zeros(length(freq_hz), length(freq_hz)),
                         T1_s        :: Vector{Float64}      = fill(Inf, length(freq_hz)),
                         T2_s        :: Vector{Float64}      = fill(Inf, length(freq_hz)),
                         carrier_hz  :: Union{Float64, Nothing} = nothing,
                         extra_drift_terms :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                         custom_controls   :: Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
                         extra_jump_ops    :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                         extra_decay_rates :: Vector{Float64} = Float64[],
                         metadata    :: Dict{String,Any}     = Dict{String,Any}())::TransmonSystem

    nq  = length(freq_hz)
    nl  = n_levels
    dim = nl^nq

    @assert length(anharm_hz) == nq "anharm_hz must have length n_qubits"
    @assert size(coupling_hz) == (nq, nq) "coupling_hz must be nq×nq"
    @assert length(T1_s) == nq && length(T2_s) == nq

    ω_c = isnothing(carrier_hz) ? 2π * mean(freq_hz) : 2π * Float64(carrier_hz)

    # ── Single-mode operators ────────────────────────────────────────────────
    function _a_op(nl)
        a = zeros(ComplexF64, nl, nl)
        for k in 1:nl-1; a[k, k+1] = sqrt(Float64(k)); end
        a
    end
    function _n_op(nl)
        n = zeros(ComplexF64, nl, nl)
        for k in 1:nl; n[k, k] = Float64(k - 1); end
        n
    end
    Id(n) = Matrix{ComplexF64}(I, n, n)

    function embed_op(op, qubit_idx, nl, nq)
        mats = [k == qubit_idx ? op : Id(nl) for k in 1:nq]
        foldl(kron, mats)
    end

    # ── Drift Hamiltonian ────────────────────────────────────────────────────
    H_drift = zeros(ComplexF64, dim, dim)
    a_ops   = [embed_op(_a_op(nl), i, nl, nq) for i in 1:nq]
    n_ops   = [embed_op(_n_op(nl), i, nl, nq) for i in 1:nq]

    for i in 1:nq
        Δω = 2π * freq_hz[i] - ω_c
        α  = 2π * anharm_hz[i]
        n̂  = n_ops[i]
        H_drift .+= Δω .* n̂ .+ (α / 2.0) .* (n̂ * n̂ .- n̂)
    end

    for i in 1:nq, j in i+1:nq
        g = 2π * coupling_hz[i, j]
        iszero(g) && continue
        ai, aj = a_ops[i], a_ops[j]
        H_drift .+= g .* (ai' * aj .+ ai * aj')
    end

    # ── Extra drift terms (rad/s) ────────────────────────────────────────────
    for Hx in extra_drift_terms
        size(Hx) == (dim, dim) ||
            throw(ArgumentError("extra_drift_terms entry size $(size(Hx)) ≠ ($dim,$dim)"))
        H_drift .+= Hx
    end

    # ── Control operators ────────────────────────────────────────────────────
    H_controls = if custom_controls === nothing
        ops = Matrix{ComplexF64}[]
        for i in 1:nq
            a = a_ops[i]
            push!(ops, (a .+ a') ./ 2.0)          # x
            push!(ops, (1im .* (a' .- a)) ./ 2.0) # y
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
    for i in 1:nq
        a = a_ops[i]
        if isfinite(T1_s[i]) && T1_s[i] > 0.0
            push!(collapse_ops, sqrt(1.0 / T1_s[i]) .* a)
        end
        γ2 = isfinite(T2_s[i]) && T2_s[i] > 0.0 ? 1.0 / T2_s[i] : 0.0
        γ1 = isfinite(T1_s[i]) && T1_s[i] > 0.0 ? 1.0 / T1_s[i] : 0.0
        γφ = max(γ2 - γ1 / 2.0, 0.0)
        γφ > 0.0 && push!(collapse_ops, sqrt(γφ) .* n_ops[i])
    end

    # User-supplied jump operators
    length(extra_jump_ops) == length(extra_decay_rates) ||
        throw(ArgumentError("extra_jump_ops and extra_decay_rates must have equal length"))
    for (L, γ) in zip(extra_jump_ops, extra_decay_rates)
        size(L) == (dim, dim) ||
            throw(ArgumentError("extra_jump_ops entry size $(size(L)) ≠ ($dim,$dim)"))
        γ >= 0.0 || throw(ArgumentError("extra_decay_rates must be non-negative"))
        γ > 0.0 && push!(collapse_ops, sqrt(γ) .* L)
    end

    # ── Leakage indices (outside computational subspace) ────────────────────
    n_comp = 2^nq
    leakage_indices = Int[]
    for idx in 1:dim
        # (named `levels`: assigning to `digits` here made it a local, so the
        # call on the right threw UndefVarError and transmon_system never ran)
        levels = digits(idx - 1; base=nl, pad=nq)
        any(d >= 2 for d in levels) && push!(leakage_indices, idx)
    end

    TransmonSystem(
        H_drift, H_controls, dim, length(H_controls),
        nq, nl, n_comp,
        freq_hz, anharm_hz,
        T1_s, T2_s, coupling_hz,
        collapse_ops, leakage_indices, metadata,
    )
end
