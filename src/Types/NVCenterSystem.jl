# Types/NVCenterSystem.jl
# NVCenterSystem: nitrogen-vacancy (NV) center spin type.
#
# Provides:
#   NVCenterSystem              — struct: NV spin-1 with optional 13C nuclear spins
#   nv_center_system(...)       — constructor

using LinearAlgebra

# ============================================================================
# NVCenterSystem
# ============================================================================

"""
    NVCenterSystem <: AbstractQuantumSystem

Physical model for a nitrogen-vacancy (NV) center spin system in diamond.

The NV electronic spin is S=1 with a zero-field splitting D ≈ 2.87 GHz.
In the rotating frame the ground-state subspace has three levels:
  |ms=0⟩, |ms=+1⟩, |ms=−1⟩

The qubit is typically encoded in {|0⟩, |−1⟩} or {|0⟩, |+1⟩}.
Optional 13C nuclear spins (I=1/2) are coupled via hyperfine.

# Fields
- `H_drift`         — dim×dim drift Hamiltonian (rad/s, rotating frame)
- `H_controls`      — control operators (MW x, y and optional RF x, y per nucleus)
- `dim`             — Hilbert space dimension
- `n_controls`      — number of control channels
- `D_hz`            — Float64 zero-field splitting D/2π (Hz; default 2.87 GHz)
- `E_hz`            — Float64 strain/non-axial splitting E/2π (Hz; default 0)
- `gamma_e_hz_T`    — Float64 electron gyromagnetic ratio / 2π (Hz/T; default 28e9)
- `B0_tesla`        — Float64 static B0 field (T) along NV axis
- `hyperfine_hz`    — Vector{Float64} isotropic hyperfine A_iso/2π (Hz) per 13C spin
- `n_nuclei`        — Int number of 13C nuclear spins coupled to NV
- `T1_s`            — Float64 electron spin T₁ (s)
- `T2_s`            — Float64 electron spin T₂ (s)
- `collapse_ops`    — Vector{Matrix{ComplexF64}} Lindblad operators
- `metadata`        — Dict{String,Any}
"""
struct NVCenterSystem <: AbstractQuantumSystem
    H_drift        :: Matrix{ComplexF64}
    H_controls     :: Vector{Matrix{ComplexF64}}
    dim            :: Int
    n_controls     :: Int
    D_hz           :: Float64
    E_hz           :: Float64
    gamma_e_hz_T   :: Float64
    B0_tesla       :: Float64
    hyperfine_hz   :: Vector{Float64}
    n_nuclei       :: Int
    T1_s           :: Float64
    T2_s           :: Float64
    collapse_ops   :: Vector{Matrix{ComplexF64}}
    metadata       :: Dict{String,Any}
end

"""
    nv_center_system(B0_tesla;
                     D_hz=2.87e9, E_hz=0.0,
                     gamma_e_hz_T=28.025e9,
                     hyperfine_hz=Float64[], n_nuclei=0,
                     carrier_hz=nothing,
                     T1_s=Inf, T2_s=Inf,
                     subspace=:full,
                     metadata=Dict()) -> NVCenterSystem

Construct an [`NVCenterSystem`](@ref).

The drift Hamiltonian (rotating frame, electron spin S=1 basis |0⟩,|+1⟩,|−1⟩) is:

    H_e = D Sz² + E(Sx²−Sy²) + γ_e B₀ Sz

where Sz, Sx, Sy are spin-1 operators.

For each 13C nuclear spin (I=1/2) the hyperfine term A_iso Iz·Sz is added.

The MW control operators are:
    H_x = Sx   (drives |0⟩ ↔ |±1⟩)
    H_y = Sy

# Rotating frame convention
The rotating-frame transform R = exp(−i ω_c t · |−1⟩⟨−1|) subtracts the carrier
from the `|−1⟩` level only — it is **not** an overall `ω_c · I` shift. This frame
is valid when driving the `|0⟩ ↔ |−1⟩` transition. Simultaneous coherent driving
of `|0⟩ ↔ |+1⟩` is **not** correctly captured because the `|+1⟩` manifold remains
in the lab frame. For dual-transition driving, rebuild `H_rot` with a two-level
carrier subtraction.

# Arguments
- `B0_tesla`         — static field (T) along NV symmetry axis
- `D_hz`             — zero-field splitting (Hz; default 2.87 GHz)
- `E_hz`             — transverse strain splitting (Hz; default 0)
- `gamma_e_hz_T`     — electron gyromagnetic ratio / 2π (Hz/T; default 28.025 GHz/T)
- `hyperfine_hz`     — Vector{Float64} isotropic hyperfine A_iso/2π per 13C (Hz)
- `n_nuclei`         — number of coupled 13C nuclei (must match hyperfine_hz length)
- `carrier_hz`       — MW carrier frequency (Hz); defaults to D + γ_e B₀
- `T1_s`, `T2_s`     — electron spin relaxation and coherence times (s)
- `subspace`         — `:full` for all three spin-1 levels, `:qubit` for {|0⟩,|−1⟩}
- `extra_drift_terms` — Hermitian matrices (rad/s) appended to `H_drift`. Use this to
                        add nuclear quadrupole (¹⁴N), anisotropic hyperfine, NV-to-NV
                        dipolar coupling, or a second rotating-frame carrier for
                        dual-transition driving.
- `custom_controls`   — if provided, **replaces** the default [Sx, Sy] (+ nuclear RF
                        channels if present).
- `extra_jump_ops`, `extra_decay_rates` — append √γ·L Lindblad operators (photon shot
                        noise, ionization, spin diffusion, etc.).
- `metadata`         — Dict{String,Any}

# Example
```julia
sys = nv_center_system(0.01; T2_s=2e-6)
@assert sys.B0_tesla == 0.01
```
"""
function nv_center_system(B0_tesla        :: Float64;
                           D_hz            :: Float64 = 2.87e9,
                           E_hz            :: Float64 = 0.0,
                           gamma_e_hz_T    :: Float64 = 28.025e9,
                           hyperfine_hz    :: Vector{Float64} = Float64[],
                           n_nuclei        :: Int     = length(hyperfine_hz),
                           carrier_hz      :: Union{Float64, Nothing} = nothing,
                           T1_s            :: Float64 = Inf,
                           T2_s            :: Float64 = Inf,
                           subspace        :: Symbol  = :full,
                           extra_drift_terms :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                           custom_controls   :: Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
                           extra_jump_ops    :: Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                           extra_decay_rates :: Vector{Float64} = Float64[],
                           metadata        :: Dict{String,Any} = Dict{String,Any}())::NVCenterSystem

    @assert length(hyperfine_hz) == n_nuclei "hyperfine_hz must have n_nuclei entries"
    @assert subspace ∈ (:full, :qubit) "subspace must be :full or :qubit"

    # ── Spin-1 operators (3×3) ────────────────────────────────────────────────
    # Basis: |ms=0⟩=|1⟩, |ms=+1⟩=|2⟩, |ms=−1⟩=|3⟩  (Julia 1-based)
    Sz3 = ComplexF64[0 0 0; 0 1 0; 0 0 -1]

    function _S1_x()
        v = sqrt(2.0)/2.0
        ComplexF64[0 v 0; v 0 v; 0 v 0]
    end
    function _S1_y()
        v = sqrt(2.0)/2.0
        ComplexF64[0 -im*v 0; im*v 0 -im*v; 0 im*v 0]
    end
    Sx3 = _S1_x()
    Sy3 = _S1_y()

    ω_D     = 2π * D_hz
    ω_E     = 2π * E_hz
    γ_e     = 2π * gamma_e_hz_T
    ω_Larm  = γ_e * B0_tesla

    H_e = ω_D .* (Sz3 * Sz3) .+ ω_E .* (Sx3 * Sx3 .- Sy3 * Sy3) .+ ω_Larm .* Sz3

    # Carrier: default to the |0⟩→|−1⟩ transition frequency
    ω_01 = D_hz - gamma_e_hz_T * B0_tesla   # Hz, |0⟩→|−1⟩
    ω_c  = isnothing(carrier_hz) ? 2π * ω_01 : 2π * Float64(carrier_hz)

    # Subtract carrier from |−1⟩ level
    phase_mat = zeros(ComplexF64, 3, 3)
    phase_mat[3, 3] = ω_c
    H_rot = H_e .- phase_mat

    # ── Nuclear spins (I=1/2) ─────────────────────────────────────────────────
    Id2 = ComplexF64[1 0; 0 1]
    σz2 = ComplexF64[1 0; 0 -1]
    σx2 = ComplexF64[0 1; 1 0]
    σy2 = ComplexF64[0 -im; im 0]

    # Helper: apply user-supplied extras to a given H and collapse_ops list
    function _apply_extras!(H::Matrix{ComplexF64}, d::Int,
                            ctrls::Vector{Matrix{ComplexF64}},
                            cops::Vector{Matrix{ComplexF64}})
        for Hx in extra_drift_terms
            size(Hx) == (d, d) ||
                throw(ArgumentError("extra_drift_terms entry size $(size(Hx)) ≠ ($d,$d)"))
            H .+= Hx
        end
        final_ctrls = if custom_controls === nothing
            ctrls
        else
            for Hc in custom_controls
                size(Hc) == (d, d) ||
                    throw(ArgumentError("custom_controls entry size $(size(Hc)) ≠ ($d,$d)"))
            end
            copy(custom_controls)
        end
        length(extra_jump_ops) == length(extra_decay_rates) ||
            throw(ArgumentError("extra_jump_ops and extra_decay_rates must have equal length"))
        for (L, γ) in zip(extra_jump_ops, extra_decay_rates)
            size(L) == (d, d) ||
                throw(ArgumentError("extra_jump_ops entry size $(size(L)) ≠ ($d,$d)"))
            γ >= 0.0 || throw(ArgumentError("extra_decay_rates must be non-negative"))
            γ > 0.0 && push!(cops, sqrt(γ) .* L)
        end
        return final_ctrls
    end

    if n_nuclei == 0
        if subspace == :qubit
            sel   = [1, 3]
            H_sub = H_rot[sel, sel]
            dim   = 2
            Hx    = Sx3[sel, sel]
            Hy    = Sy3[sel, sel]
        else
            dim   = 3
            H_sub = H_rot
            Hx    = Sx3
            Hy    = Sy3
        end

        collapse_ops = Matrix{ComplexF64}[]
        if subspace == :qubit
            # |0⟩ is the ground state (D·Sz²): T₁ lowers |−1⟩ → |0⟩, i.e. |0⟩⟨−1|
            σm2 = ComplexF64[0 1; 0 0]
            isfinite(T1_s) && T1_s > 0.0 && push!(collapse_ops,
                sqrt(1.0/T1_s) .* σm2)
            γ2 = isfinite(T2_s) && T2_s > 0.0 ? 1.0/T2_s : 0.0
            γ1 = isfinite(T1_s) && T1_s > 0.0 ? 1.0/T1_s : 0.0
            γφ = max(γ2 - γ1/2.0, 0.0)
            γφ > 0.0 && push!(collapse_ops, sqrt(γφ) .* (σz2 ./ 2.0))
        else
            Sm_p1 = zeros(ComplexF64, 3, 3); Sm_p1[1, 2] = 1.0
            Sm_m1 = zeros(ComplexF64, 3, 3); Sm_m1[1, 3] = 1.0
            if isfinite(T1_s) && T1_s > 0.0
                push!(collapse_ops, sqrt(1.0/T1_s) .* Sm_p1)
                push!(collapse_ops, sqrt(1.0/T1_s) .* Sm_m1)
            end
            if isfinite(T2_s) && T2_s > 0.0
                γφ = max(1.0/T2_s - 0.5/max(T1_s, Inf), 0.0)
                γφ > 0.0 && push!(collapse_ops, sqrt(γφ) .* Sz3)
            end
        end

        final_ctrls = _apply_extras!(H_sub, dim,
                                      Matrix{ComplexF64}[Hx, Hy], collapse_ops)

        return NVCenterSystem(
            H_sub,
            final_ctrls,
            dim, length(final_ctrls),
            D_hz, E_hz, gamma_e_hz_T, B0_tesla,
            hyperfine_hz, 0,
            T1_s, T2_s,
            collapse_ops, metadata,
        )
    end

    # ── With nuclear spins: full tensor product ────────────────────────────────
    dim_e = 3
    Id3   = Matrix{ComplexF64}(I, dim_e, dim_e)
    Id_n_total = Matrix{ComplexF64}(I, 2^n_nuclei, 2^n_nuclei)
    dim   = dim_e * 2^n_nuclei

    H_full = kron(H_rot, Id_n_total)

    for idx in 1:n_nuclei
        A = 2π * hyperfine_hz[idx]
        iszero(A) && continue
        Iz_n = begin
            mats = [k == idx ? (σz2 ./ 2.0) : Id2 for k in 1:n_nuclei]
            foldl(kron, mats)
        end
        H_full .+= A .* kron(Sz3, Iz_n)
    end

    H_controls = [kron(Sx3, Id_n_total), kron(Sy3, Id_n_total)]

    collapse_ops = Matrix{ComplexF64}[]
    Sm_p1_f = zeros(ComplexF64, 3, 3); Sm_p1_f[1, 2] = 1.0
    Sm_m1_f = zeros(ComplexF64, 3, 3); Sm_m1_f[1, 3] = 1.0
    if isfinite(T1_s) && T1_s > 0.0
        push!(collapse_ops, sqrt(1.0/T1_s) .* kron(Sm_p1_f, Id_n_total))
        push!(collapse_ops, sqrt(1.0/T1_s) .* kron(Sm_m1_f, Id_n_total))
    end
    if isfinite(T2_s) && T2_s > 0.0
        γφ = max(1.0/T2_s - 0.5/max(T1_s, Inf), 0.0)
        γφ > 0.0 && push!(collapse_ops, sqrt(γφ) .* kron(Sz3, Id_n_total))
    end

    final_ctrls = _apply_extras!(H_full, dim, H_controls, collapse_ops)

    return NVCenterSystem(
        H_full, final_ctrls, dim, length(final_ctrls),
        D_hz, E_hz, gamma_e_hz_T, B0_tesla,
        hyperfine_hz, n_nuclei,
        T1_s, T2_s,
        collapse_ops, metadata,
    )
end

# ─── Operators for ensemble axes (see Types/EnsembleAveraging.jl) ─────────────

"""
    offset_operator(sys::NVCenterSystem) -> Matrix{ComplexF64}

|−1⟩⟨−1| (⊗ 1 on the nuclei) — what a detuning δ of the driven |0⟩↔|−1⟩
transition from the carrier adds in `nv_center_system` (the carrier is
subtracted from the |−1⟩ level only), so an `OffsetAxis` value δ equals
`carrier_hz − δ`. For a B0 or zero-field-splitting fluctuation use a `TermAxis`
on Sz or Sz² instead — those also move |+1⟩.
"""
function offset_operator(sys::NVCenterSystem)
    dim_e = sys.dim == 2 ? 2 : 3                      # :qubit subspace {|0⟩, |−1⟩} or full spin-1
    P = zeros(ComplexF64, dim_e, dim_e); P[dim_e, dim_e] = 1
    return sys.n_nuclei == 0 ? P : kron(P, Matrix{ComplexF64}(I, 2^sys.n_nuclei, 2^sys.n_nuclei))
end
