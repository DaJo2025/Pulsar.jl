# Computation/SpatialTensors.jl
# ============================================================================
# General irreducible spherical-tensor engine for anisotropic spin interactions.
#
# ONE implementation shared by MAS solid-state NMR, EPR, DNP, and any future
# solid-state domain.  Every anisotropic interaction (CSA, dipolar, g-tensor,
# hyperfine, quadrupolar, …) is a rank-0 (isotropic) + rank-2 (anisotropic)
# spatial tensor defined in its principal-axis system (PAS).  The engine rotates
# it through the frame chain
#
#     PAS --euler_PM--> molecular/crystal --crystallite Ω--> rotor --tilt θ--> lab
#
# and expands the result into rotor-modulated Fourier components H_fourier[n]
# (n = -2..2) with  H(t) = Σ_n H_fourier[n] exp(i n ω_r t), the exact form that
# compute_propagators(::MASSpinSystem / ::DNPSpinSystem, …) already consumes.
#
# Physical conventions (Haeberlen / Mehring; cf. Levitt "Spin Dynamics",
# Bak-Nielsen SIMPSON):
#   • rank-2 PAS components  ρ₂₀ = √(3/2)·δ,  ρ₂,±₂ = −δη/2,  ρ₂,±₁ = 0,
#     with the spin-tensor normalisation √(2/3) folded in so that a STATIC axial
#     shift (η=0, tilt θ=0, PAS at polar angle β) yields the textbook pattern
#         H = [δ_iso + δ·(3cos²β−1)/2] · O.
#   • the rotor→lab tilt θ is the magic angle  θ_m = atan√2 ≈ 54.7356°  for MAS,
#     and θ = 0 for a STATIC powder (EPR, non-spinning DNP): d²_{m,0}(0)=δ_{m,0}
#     collapses the five harmonics to a single time-independent H_fourier[0].
#
# Depends only on WignerRotations.jl (Layer 1b) — no upward dependencies.
# ============================================================================
using LinearAlgebra

"""
    MAGIC_ANGLE

Magic angle θ_m = atan(√2) = acos(1/√3) ≈ 0.9553166 rad (54.7356°).  At this
rotor tilt the rank-2 secular projection d²_{0,0}(θ_m) = (3cos²θ_m − 1)/2 = 0, so
the time-independent anisotropy averages to zero over a rotor period — the
defining property of magic-angle spinning.
"""
const MAGIC_ANGLE = atan(sqrt(2.0))

"""
    SphericalInteraction

A single anisotropic spin interaction in irreducible spherical-tensor form.

# Fields
- `iso`      — rank-0 isotropic coefficient (rad/s); e.g. isotropic chemical
  shift, isotropic hyperfine, electron Zeeman offset.  Multiplies `iso_op`.
- `aniso`    — rank-2 anisotropy δ in the Haeberlen convention (rad/s).
- `eta`      — asymmetry parameter η ∈ [0,1].
- `euler_PM` — (α,β,γ) Euler angles rotating the interaction PAS into the
  molecular/crystal frame (rad).
- `lab_ops`  — map `lab spatial index m ⇒ spin operator O_m` (Hermitian pairs:
  `lab_ops[-m] == lab_ops[m]'`).  The high-field secular truncation of the
  interaction is encoded here: a secular shift/dipolar keeps only `m=0`; a
  pseudosecular hyperfine additionally keeps `m=±1` with `Sz·I∓` operators.
- `iso_op`   — spin operator carrying the isotropic part (`nothing` if the
  interaction is purely anisotropic, e.g. dipolar).

The crystallite orientation and the rotor tilt are supplied at build time
(`interaction_fourier!` / `build_orientation_fourier`), NOT stored here, so one
`SphericalInteraction` is reused across every powder orientation.
"""
struct SphericalInteraction
    iso      :: Float64
    aniso    :: Float64
    eta      :: Float64
    euler_PM :: NTuple{3,Float64}
    lab_ops  :: Dict{Int,Matrix{ComplexF64}}
    iso_op   :: Union{Nothing,Matrix{ComplexF64}}
end

"""
    _pas_rank2(δ, η) -> NTuple{5,ComplexF64}

Rank-2 PAS spherical components (q = −2..2 → index 1..5), with the √(2/3)
spin-tensor factor folded in (see module header):
`ρ̃₂₀ = δ`, `ρ̃₂,±₂ = −δη/√6`, `ρ̃₂,±₁ = 0`.
"""
@inline function _pas_rank2(δ::Float64, η::Float64)::NTuple{5,ComplexF64}
    a2 = ComplexF64(-δ * η / sqrt(6.0))
    return (a2, 0.0 + 0im, ComplexF64(δ), 0.0 + 0im, a2)   # q = -2,-1,0,1,2
end

"""
    _rotor_components(δ, η, euler_PM, cryst) -> Vector{ComplexF64}

Rank-2 spatial components in the ROTOR frame, `A^rotor_{2,m'}` for m' = −2..2
(returned length-5, index m'+3).  Rotates the PAS components through
PAS→molecular (`euler_PM`) then molecular→rotor (`cryst = (α,β,γ)`) via proper
Wigner-D² matrix composition `D²(euler_PM)·D²(cryst)` — the general/exact
rotation (not the shared-axis angle-addition shortcut).
"""
function _rotor_components(δ::Float64, η::Float64,
                           euler_PM::NTuple{3,Float64},
                           cryst::NTuple{3,Float64})::Vector{ComplexF64}
    ρ     = _pas_rank2(δ, η)
    D_pm  = wigner_D2(euler_PM[1], euler_PM[2], euler_PM[3])   # [q+3, m'+3]
    D_mr  = wigner_D2(cryst[1],    cryst[2],    cryst[3])
    Dcomb = D_pm * D_mr                                        # [q+3, m'+3]
    A = zeros(ComplexF64, 5)
    @inbounds for mr in -2:2
        s = 0.0 + 0im
        for q in -2:2
            s += ρ[q+3] * Dcomb[q+3, mr+3]
        end
        A[mr+3] = s
    end
    return A
end

"""
    interaction_fourier!(Hf, ia, cryst; theta_tilt=MAGIC_ANGLE) -> Hf

Accumulate the rotor-modulated Fourier components of one `SphericalInteraction`
into `Hf` (a `Dict{Int,Matrix{ComplexF64}}` with keys −2..2), for crystallite
orientation `cryst=(α,β,γ)`.

Derivation.  The lab-frame rank-2 component is
`A^L_{2,m}(t) = Σ_{m'} A^rotor_{2,m'} e^{−i m' ω_r t} d²_{m',m}(θ)`.  With
`H(t) = Σ_m∈lab_ops O_m · A^L_{2,m}(t) = Σ_n H_fourier[n] e^{+i n ω_r t}` the
harmonic index is `n = −m'`, giving
`H_fourier[n] += Σ_{m∈lab_ops} A^rotor_{2,−n} · d²_{−n,m}(θ) · O_m`,
plus the isotropic `iso·iso_op` in `H_fourier[0]`.

`theta_tilt = MAGIC_ANGLE` gives true MAS; `theta_tilt = 0` gives a static
powder (only `n=0` survives, `d²_{m',0}(0)=δ_{m',0}`).
"""
function interaction_fourier!(Hf::Dict{Int,Matrix{ComplexF64}},
                              ia::SphericalInteraction,
                              cryst::NTuple{3,Float64};
                              theta_tilt::Float64 = MAGIC_ANGLE)
    A  = _rotor_components(ia.aniso, ia.eta, ia.euler_PM, cryst)   # A[m'+3]
    d2 = wigner_d2(theta_tilt)                                     # d2[m'+3, m+3]
    for (m, O) in ia.lab_ops
        @inbounds for n in -2:2
            mp    = -n
            coeff = A[mp+3] * d2[mp+3, m+3]
            iszero(coeff) && continue
            LinearAlgebra.axpy!(coeff, O, Hf[n])
        end
    end
    if ia.iso_op !== nothing && ia.iso != 0.0
        LinearAlgebra.axpy!(ComplexF64(ia.iso), ia.iso_op, Hf[0])
    end
    return Hf
end

"""
    build_orientation_fourier(interactions, cryst, dim;
                              theta_tilt=MAGIC_ANGLE, H0=nothing) -> Dict{Int,Matrix}

Build the full rotor-modulated Fourier component dictionary `H_fourier[n]`,
n = −2..2, for a crystallite orientation `cryst` by summing every
`SphericalInteraction`.  `H0` (optional) is an orientation-independent drift
(e.g. bare Zeeman / offset terms) added into `H_fourier[0]`.
"""
function build_orientation_fourier(interactions::AbstractVector{SphericalInteraction},
                                   cryst::NTuple{3,Float64},
                                   dim::Int;
                                   theta_tilt::Float64 = MAGIC_ANGLE,
                                   H0::Union{Nothing,AbstractMatrix} = nothing)::Dict{Int,Matrix{ComplexF64}}
    Hf = Dict{Int,Matrix{ComplexF64}}(n => zeros(ComplexF64, dim, dim) for n in -2:2)
    if H0 !== nothing
        Hf[0] .+= H0
    end
    for ia in interactions
        interaction_fourier!(Hf, ia, cryst; theta_tilt = theta_tilt)
    end
    return Hf
end

# ============================================================================
# Convenience constructors for the common interaction "kinds".  Each encodes the
# high-field secular truncation as the lab_ops map.
# ============================================================================

"""
    principal_to_delta_eta(a::NTuple{3,Float64}) -> (δ, η)

Convert Cartesian principal values `(a_xx, a_yy, a_zz)` of a symmetric rank-2
interaction into the folded anisotropy/asymmetry `(δ, η)` consumed by
`SphericalInteraction`:  `δ = a_zz − a_iso`,  `η = (a_yy − a_xx)/δ`  (with
`η = 0` when `δ ≈ 0`).  Works for any axis assignment consistent with the PAS
Euler angles — no Haeberlen reordering required — because the engine's rank-2
components satisfy `ρ̃₂,±₂ = (a_xx − a_yy)/√6` identically.
"""
function principal_to_delta_eta(a::NTuple{3,Float64})::Tuple{Float64,Float64}
    a_iso = (a[1] + a[2] + a[3]) / 3
    δ = a[3] - a_iso
    η = abs(δ) < 1e-30 ? 0.0 : (a[2] - a[1]) / δ
    return δ, η
end

"""
    shift_interaction(iso, δ, η, euler_PM, Oz) -> SphericalInteraction

Secular shift-like interaction (CSA, electron/nuclear g-anisotropy): keeps only
the lab `m=0` component multiplying the Zeeman operator `Oz` (Iz or Sz).
"""
shift_interaction(iso::Float64, δ::Float64, η::Float64,
                  euler_PM::NTuple{3,Float64}, Oz::Matrix{ComplexF64}) =
    SphericalInteraction(iso, δ, η, euler_PM, Dict(0 => Oz), Oz)

"""
    dipolar_hetero_interaction(b, euler_PM, Iz, Sz) -> SphericalInteraction

Secular heteronuclear dipolar coupling `b·(3cos²β−1)/2 · 2·Iz·Sz` (no isotropic
part).  `euler_PM` orients the internuclear vector (unique axis β) in the MF.
"""
dipolar_hetero_interaction(b::Float64, euler_PM::NTuple{3,Float64},
                           Iz::Matrix{ComplexF64}, Sz::Matrix{ComplexF64}) =
    SphericalInteraction(0.0, b, 0.0, euler_PM, Dict(0 => 2 .* (Iz * Sz)), nothing)

"""
    dipolar_homo_interaction(b, euler_PM, Ix,Iy,Iz, Jx,Jy,Jz) -> SphericalInteraction

Secular homonuclear dipolar coupling `b·(3cos²β−1)/2 · (2IzJz − IxJx − IyJy)`
(the flip-flop-containing rank-2 spin operator).  No isotropic part.
"""
function dipolar_homo_interaction(b::Float64, euler_PM::NTuple{3,Float64},
                                  Ix,Iy,Iz, Jx,Jy,Jz)::SphericalInteraction
    O = 2 .* (Iz * Jz) .- (Ix * Jx) .- (Iy * Jy)
    return SphericalInteraction(0.0, b, 0.0, euler_PM, Dict(0 => Matrix{ComplexF64}(O)), nothing)
end

"""
    hyperfine_interaction(A_iso, A_aniso, η, euler_PM, Sz, Ix,Iy,Iz) -> SphericalInteraction

Electron–nuclear hyperfine coupling after electron-secular truncation, keeping
both the SECULAR term `A_zz·Sz·Iz` (lab m=0) and the PSEUDOSECULAR terms
`Sz·(A_zx Ix + A_zy Iy)` (lab m=±1).  The pseudosecular branch is what enables
the DNP solid effect.  Operators for m=±1 use the Hermitian raising/lowering
pair `Sz·I∓`, so `lab_ops[-1] == lab_ops[1]'`.
"""
function hyperfine_interaction(A_iso::Float64, A_aniso::Float64, η::Float64,
                               euler_PM::NTuple{3,Float64},
                               Sz, Ix, Iy, Iz)::SphericalInteraction
    Ip = Ix + im .* Iy                    # I⁺
    Im = Ix - im .* Iy                    # I⁻
    SzIz = Matrix{ComplexF64}(Sz * Iz)
    # Pseudosecular: lab m=±1 rank-2 components pair with Sz·I∓ (see module header).
    # Normalisation −1/√2 is the standard rank-2 spin-tensor coefficient for the
    # m=±1 spherical components; validated against the analytic secular limit.
    SzIm = Matrix{ComplexF64}(-(1/sqrt(2)) .* (Sz * Im))   # lab m=+1  ⇒ Sz I⁻
    SzIp = Matrix{ComplexF64}( (1/sqrt(2)) .* (Sz * Ip))   # lab m=-1  ⇒ Sz I⁺
    return SphericalInteraction(A_iso, A_aniso, η, euler_PM,
                                Dict(0 => SzIz, 1 => SzIm, -1 => SzIp), SzIz)
end
