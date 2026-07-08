# Types/EPRSpinSystem.jl
# EPR spin system type for pulsed EPR optimal control.
using LinearAlgebra

# Physical constants (if not already defined elsewhere)
const _EPR_μ_B_SI  = 9.2740100783e-24   # J/T
const _EPR_hbar_SI = 1.0545718e-34      # J·s
const _MHZ_TO_RADS = 2π * 1e6

"""
    EPRSpinSystem <: AbstractQuantumSystem

Electron paramagnetic resonance spin system.

Hamiltonian (secular, high-field approximation):
    H(Ω) = (ωS(Ω) − ωmw) Sz + Σ_k Azz_k(Ω) Sz Iz_k + D(Sz²-S(S+1)/3) + E(Sx²-Sy²)

# Fields
- `S_electron`   — electron spin quantum number (usually 1//2)
- `I_nuclei`     — nuclear spin quantum numbers
- `dim`          — Hilbert space dimension
- `B0_tesla`     — static field strength (T)
- `mw_freq_hz`   — microwave frequency (Hz)
- `g_vals`       — principal g-tensor values [gxx,gyy,gzz]
- `g_euler`      — Euler angles (α,β,γ) rotating PAS→MF (rad)
- `A_vals`       — hyperfine tensor principal values per nucleus (MHz)
- `A_euler`      — Euler angles per nucleus (rad)
- `D_mhz`        — axial ZFS parameter (MHz), 0 for S=1/2
- `E_mhz`        — rhombic ZFS parameter (MHz)
- `S_ops`        — electron (Sx,Sy,Sz) in full space
- `I_ops`        — nuclear (Ix,Iy,Iz) per nucleus in full space
- `H_drift`      — nominal drift (orientation-averaged or single-crystal)
- `H_ctrl`       — control Hamiltonians [Sx,Sy] (microwave)
- `n_controls`   — 2
"""
struct EPRSpinSystem <: AbstractQuantumSystem
    S_electron  :: Rational{Int}
    I_nuclei    :: Vector{Rational{Int}}
    dim         :: Int
    B0_tesla    :: Float64
    mw_freq_hz  :: Float64
    g_vals      :: NTuple{3,Float64}
    g_euler     :: NTuple{3,Float64}
    A_vals      :: Vector{NTuple{3,Float64}}
    A_euler     :: Vector{NTuple{3,Float64}}
    D_mhz       :: Float64
    E_mhz       :: Float64
    S_ops       :: NTuple{3,Matrix{ComplexF64}}
    I_ops       :: Vector{NTuple{3,Matrix{ComplexF64}}}
    H_drift     :: Matrix{ComplexF64}
    H_controls  :: Vector{Matrix{ComplexF64}}
    n_controls  :: Int
end

"""
    epr_system(S, I_nuclei, B0_tesla, mw_freq_hz; g_vals, g_euler, A_vals, A_euler, D_mhz, E_mhz)

Construct an EPRSpinSystem. g_vals defaults to [2.0023,2.0023,2.0023], A_vals to zeros.
"""
function epr_system(S::Rational{Int},
                    I_nuclei::Vector{Rational{Int}},
                    B0_tesla::Float64,
                    mw_freq_hz::Float64;
                    g_vals::NTuple{3,Float64}           = (2.0023, 2.0023, 2.0023),
                    g_euler::NTuple{3,Float64}          = (0.0, 0.0, 0.0),
                    A_vals::Vector{NTuple{3,Float64}}   = [(0.0, 0.0, 0.0) for _ in I_nuclei],
                    A_euler::Vector{NTuple{3,Float64}}  = [(0.0, 0.0, 0.0) for _ in I_nuclei],
                    D_mhz::Float64 = 0.0,
                    E_mhz::Float64 = 0.0)::EPRSpinSystem

    dS  = Int(2*S + 1)
    dIs = [Int(2*I + 1) for I in I_nuclei]
    dim = dS * prod(dIs; init=1)

    # Electron spin operators embedded in full space
    Sx_loc = spin_Sx(Float64(S))
    Sy_loc = spin_Sy(Float64(S))
    Sz_loc = spin_Sz(Float64(S))

    I_tail = prod(dIs; init=1)
    Itail_mat = Matrix{ComplexF64}(I, I_tail, I_tail)
    Sx_full = kron(Sx_loc, Itail_mat)
    Sy_full = kron(Sy_loc, Itail_mat)
    Sz_full = kron(Sz_loc, Itail_mat)
    S_ops = (Sx_full, Sy_full, Sz_full)

    # Nuclear spin operators
    I_ops_list = NTuple{3,Matrix{ComplexF64}}[]
    for (k, Ik) in enumerate(I_nuclei)
        d_before = dS * prod(dIs[1:k-1]; init=1)
        d_after  = prod(dIs[k+1:end]; init=1)
        Ib = Matrix{ComplexF64}(I, d_before, d_before)
        Ia = Matrix{ComplexF64}(I, d_after,  d_after)
        push!(I_ops_list, (
            kron(Ib, kron(spin_Sx(Float64(Ik)), Ia)),
            kron(Ib, kron(spin_Sy(Float64(Ik)), Ia)),
            kron(Ib, kron(spin_Sz(Float64(Ik)), Ia))
        ))
    end

    # Build nominal drift at isotropic g = mean(g_vals)
    g_iso = sum(g_vals) / 3
    ω_S   = _EPR_μ_B_SI * B0_tesla * g_iso / _EPR_hbar_SI  # rad/s
    H_drift = (ω_S - 2π*mw_freq_hz) .* Sz_full

    # Add isotropic hyperfine (secular: Azz Sz Iz)
    for (k, Ak) in enumerate(A_vals)
        Azz_iso = sum(Ak)/3 * _MHZ_TO_RADS
        Iz_k    = I_ops_list[k][3]
        H_drift .+= Azz_iso .* (Sz_full * Iz_k)
    end

    # ZFS (only for S > 1/2)
    if D_mhz != 0.0 || E_mhz != 0.0
        SS1 = Float64(S*(S+1))
        H_drift .+= (D_mhz * _MHZ_TO_RADS) .* (Sz_full*Sz_full .- (SS1/3) .* Matrix{ComplexF64}(I,dim,dim))
        H_drift .+= (E_mhz * _MHZ_TO_RADS) .* (Sx_full*Sx_full .- Sy_full*Sy_full)
    end

    H_controls_vec = Matrix{ComplexF64}[Sx_full, Sy_full]

    return EPRSpinSystem(S, I_nuclei, dim, B0_tesla, mw_freq_hz,
                         g_vals, g_euler, A_vals, A_euler, D_mhz, E_mhz,
                         S_ops, I_ops_list, H_drift, H_controls_vec, 2)
end

"""
    shift_system(sys::EPRSpinSystem, offset_hz::Float64) -> EPRSpinSystem

Return a copy of `sys` with the electron resonance shifted by `2π·offset_hz`.
Used by `band_selective_fidelity` / `band_selective_gradient` so DEER-style
pump/probe (multi-offset) optimisation dispatches on `EPRSpinSystem`.

The offset couples to the electron `Sz` (`sys.S_ops[3]`), NOT the identity: an
identity term is a global phase and cancels in every `|⟨ψ|U|ψ⟩|²` / `|Tr(U†U)|²`
fidelity, which would make `band_selective_*` return the SAME value for every
band — defeating band selectivity.  (The `QuantumSystem`/`SpinSystem` siblings in
`Physics/Objectives.jl` currently use the identity form and share that no-op.)

Defined here rather than beside the siblings because `EPRSpinSystem` is not yet
in scope when `Objectives.jl` loads; adding a method to the existing
`shift_system` generic from this later-loaded file is fine.
"""
function shift_system(sys::EPRSpinSystem, offset_hz::Float64)::EPRSpinSystem
    H_shifted = sys.H_drift .+ (2π * offset_hz) .* sys.S_ops[3]
    return EPRSpinSystem(
        sys.S_electron, sys.I_nuclei, sys.dim,
        sys.B0_tesla, sys.mw_freq_hz,
        sys.g_vals, sys.g_euler,
        sys.A_vals, sys.A_euler,
        sys.D_mhz, sys.E_mhz,
        sys.S_ops, sys.I_ops,
        H_shifted, sys.H_controls, sys.n_controls,
    )
end
