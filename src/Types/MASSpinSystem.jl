# Types/MASSpinSystem.jl
# Magic-Angle Spinning (MAS) solid-state NMR spin system.
using LinearAlgebra

"""
    CSATensor

Chemical shift anisotropy tensor in its principal axis system (PAS).

# Fields
- `delta_iso`       — isotropic shift (rad/s)
- `delta_aniso`     — anisotropy (rad/s), Haeberlen convention
- `eta`             — asymmetry parameter ∈ [0,1]
- `euler_PAS_to_MF` — (α,β,γ) Euler angles rotating PAS to molecular frame (rad)
"""
struct CSATensor
    delta_iso    :: Float64
    delta_aniso  :: Float64
    eta          :: Float64
    euler_PAS_to_MF :: NTuple{3,Float64}
end

"""
    DipolarCoupling

Dipole–dipole coupling between two spins.

# Fields
- `spin_i`, `spin_j`  — 1-based spin indices
- `b_ij`              — dipolar coupling constant (rad/s)
- `euler_DD_to_MF`    — (α,β,γ) Euler angles rotating DD-PAS to molecular frame (rad)
"""
struct DipolarCoupling
    spin_i :: Int
    spin_j :: Int
    b_ij   :: Float64
    euler_DD_to_MF :: NTuple{3,Float64}
end

"""
    MASSpinSystem <: AbstractQuantumSystem

Solid-state NMR spin system under Magic-Angle Spinning.

Under MAS the rank-2 interaction tensors are modulated as:

    H_int(t) = Σ_{m=-2}^{2} H_m exp(im ωr t)

where the H_m Fourier components are precomputed and stored in `H_fourier`.

# Fields
- `base_system`   — underlying liquid-state MRSpinSystem
- `csa`           — CSA tensor per spin
- `dipolar`       — dipolar couplings between spin pairs
- `omega_r`       — MAS rotor frequency (rad/s)
- `rotor_period_s`— 2π/omega_r (s)
- `H_fourier`     — Dict m => H_m matrix for m ∈ {-2,-1,0,1,2}
- `H_drift`       — zero matrix (time-dependent drift via H_fourier)
- `H_controls`        — control operators from base_system [Ix_1, Iy_1, ...]
- `dim`, `n_controls`
"""
struct MASSpinSystem <: AbstractQuantumSystem
    base_system    :: MRSpinSystem
    csa            :: Vector{CSATensor}
    dipolar        :: Vector{DipolarCoupling}
    omega_r        :: Float64
    rotor_period_s :: Float64
    H_fourier      :: Dict{Int,Matrix{ComplexF64}}
    H_drift        :: Matrix{ComplexF64}
    H_controls         :: Vector{Matrix{ComplexF64}}
    dim            :: Int
    n_controls     :: Int
end

"""
    mas_spin_system(base_system, omega_r_hz; csa, dipolar) -> MASSpinSystem

Construct a MASSpinSystem. omega_r_hz is the MAS rate in Hz.
"""
function mas_spin_system(base_system::MRSpinSystem,
                         omega_r_hz::Float64;
                         csa::Vector{CSATensor}        = CSATensor[],
                         dipolar::Vector{DipolarCoupling} = DipolarCoupling[])::MASSpinSystem
    omega_r = 2π * omega_r_hz
    rotor_period_s = 2π / omega_r

    dim = base_system.dim
    H_drift_zero = zeros(ComplexF64, dim, dim)   # time dependence lives in H_fourier

    # Rotor-modulated Fourier components via the shared spherical-tensor engine,
    # WITH the magic-angle rotor projection (the hand-coded builders this replaces
    # omitted it, so the static m=0 anisotropy did not average out under spinning).
    interactions = _mas_interactions(base_system, csa, dipolar)
    H_fourier = build_orientation_fourier(interactions, (0.0, 0.0, 0.0), dim;
                                          theta_tilt = MAGIC_ANGLE)

    # Build control operators: [Ix_1, Iy_1, Ix_2, Iy_2, ...]
    H_controls = Matrix{ComplexF64}[]
    for k in 1:base_system.n_spins
        push!(H_controls, base_system.Ix[k])
        push!(H_controls, base_system.Iy[k])
    end

    return MASSpinSystem(base_system, csa, dipolar, omega_r, rotor_period_s,
                         H_fourier, H_drift_zero, H_controls,
                         dim, length(H_controls))
end

"""
    _mas_interactions(sys::MRSpinSystem, csa, dipolar) -> Vector{SphericalInteraction}

Translate the MAS CSA and dipolar tensors into engine `SphericalInteraction`s:
each CSA is a secular shift on `Iz_i` (now honouring its asymmetry η, previously
dropped); each dipolar coupling is heteronuclear-secular `b·2·Iz_i·Iz_j`.
"""
function _mas_interactions(sys::MRSpinSystem,
                           csa::Vector{CSATensor},
                           dipolar::Vector{DipolarCoupling})::Vector{SphericalInteraction}
    n = sys.n_spins
    ints = SphericalInteraction[]
    for (i, c) in enumerate(csa)
        i <= n || break
        push!(ints, shift_interaction(c.delta_iso, c.delta_aniso, c.eta,
                                      c.euler_PAS_to_MF, Matrix{ComplexF64}(sys.Iz[i])))
    end
    for dd in dipolar
        (dd.spin_i <= n && dd.spin_j <= n) || continue
        push!(ints, dipolar_hetero_interaction(dd.b_ij, dd.euler_DD_to_MF,
                                               Matrix{ComplexF64}(sys.Iz[dd.spin_i]),
                                               Matrix{ComplexF64}(sys.Iz[dd.spin_j])))
    end
    return ints
end

"""
    build_mas_hamiltonian(sys::MASSpinSystem, t::Float64) -> Matrix{ComplexF64}

Evaluate the time-dependent MAS drift Hamiltonian at time t:
    H_drift(t) = Σ_{m=-2}^{2} H_m exp(im ωr t)
"""
function build_mas_hamiltonian(sys::MASSpinSystem, t::Float64)::Matrix{ComplexF64}
    H = zeros(ComplexF64, sys.dim, sys.dim)
    @inbounds for m in -2:2
        phase = exp(im * m * sys.omega_r * t)
        LinearAlgebra.axpy!(phase, sys.H_fourier[m], H)
    end
    return H
end

"""
    rotate_spin_system(sys::MASSpinSystem, α, β, γ) -> MASSpinSystem

Return a copy of `sys` with Fourier components recomputed for crystallite
orientation (α,β,γ), used for powder averaging.  The crystallite is applied as
the common molecular→rotor rotation for every interaction via the shared engine
(`build_orientation_fourier`) — the physically correct composition, rather than
shifting each tensor's PAS Euler angles independently.  Crucially this preserves
the caller's `sys.H_controls` (a rebuild via `mas_spin_system` would regenerate
DEFAULT unscaled controls), so externally-scaled control operators survive.
"""
function rotate_spin_system(sys::MASSpinSystem, α::Float64, β::Float64, γ::Float64)::MASSpinSystem
    interactions = _mas_interactions(sys.base_system, sys.csa, sys.dipolar)
    H_fourier = build_orientation_fourier(interactions, (α, β, γ), sys.dim;
                                          theta_tilt = MAGIC_ANGLE)
    return MASSpinSystem(sys.base_system, sys.csa, sys.dipolar, sys.omega_r,
                         sys.rotor_period_s, H_fourier, sys.H_drift, sys.H_controls,
                         sys.dim, sys.n_controls)
end
