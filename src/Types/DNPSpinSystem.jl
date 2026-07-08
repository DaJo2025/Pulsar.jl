# Types/DNPSpinSystem.jl
# Dynamic Nuclear Polarization (DNP) spin system.
# Requires MASSpinSystem.jl (CSATensor, DipolarCoupling) and
# EPRSpinSystem.jl (_EPR_μ_B_SI, _EPR_hbar_SI) to be loaded first.
using LinearAlgebra

"""
    DNPSpinSystem <: AbstractQuantumSystem

Electron–nuclear spin system for DNP optimal control.
Combines MAS solid-state NMR dynamics (time-dependent H_fourier) with
electron spin operators for microwave control.

# Fields
- `S_electron`    — electron spin (usually 1//2)
- `I_nuclei`      — nuclear spin quantum numbers
- `g_vals`        — g-tensor principal values
- `g_euler`       — g-tensor Euler angles (PAS→MF)
- `A_vals`        — hyperfine tensor principal values per nucleus (Hz; internally multiplied by 2π)
- `A_euler`       — hyperfine Euler angles per nucleus
- `csa_electron`  — electron CSA tensors
- `csa_nuclei`    — nuclear CSA tensors
- `dipolar`       — electron–nuclear dipolar couplings
- `omega_r`       — MAS rotor frequency (rad/s)
- `rotor_period_s`
- `H_fourier`     — MAS Fourier components m=-2..2
- `B0_tesla`      — static field
- `mw_freq_hz`    — microwave carrier frequency (Hz)
- `rf_freq_hz`    — nuclear RF carrier frequency (Hz); 0 if no RF
- `dim`           — total Hilbert space dimension
- `S_ops`         — electron (Sx,Sy,Sz) in full space
- `I_ops`         — nuclear ops per nucleus in full space
- `H_ctrl_mw`     — [Sx,Sy] microwave control operators
- `H_ctrl_rf`     — [Ix,Iy] nuclear RF control operators (empty if rf_freq_hz==0)
- `H_drift`       — nominal zero-order drift (m=0 Fourier term)
- `H_ctrl`        — vcat(H_ctrl_mw, H_ctrl_rf) for interface compatibility
- `n_controls`
- `interactions`  — anisotropic `SphericalInteraction`s (g-anisotropy, hyperfine,
  dipolar, CSA) driving the orientation dependence, in molecular-frame PAS Euler
  angles.  Powder averaging rotates these through the crystallite orientation via
  the shared spherical-tensor engine (`build_orientation_fourier`).
- `H0`            — orientation-INDEPENDENT drift (nuclear Zeeman + any
  `extra_drift_terms`); added into `H_fourier[0]` for every crystallite.
"""
struct DNPSpinSystem <: AbstractQuantumSystem
    S_electron    :: Rational{Int}
    I_nuclei      :: Vector{Rational{Int}}
    g_vals        :: NTuple{3,Float64}
    g_euler       :: NTuple{3,Float64}
    A_vals        :: Vector{NTuple{3,Float64}}
    A_euler       :: Vector{NTuple{3,Float64}}
    csa_electron  :: Vector{CSATensor}
    csa_nuclei    :: Vector{CSATensor}
    dipolar       :: Vector{DipolarCoupling}
    omega_r       :: Float64
    rotor_period_s:: Float64
    H_fourier     :: Dict{Int,Matrix{ComplexF64}}
    B0_tesla      :: Float64
    mw_freq_hz    :: Float64
    rf_freq_hz    :: Float64
    dim           :: Int
    S_ops         :: NTuple{3,Matrix{ComplexF64}}
    I_ops         :: Vector{NTuple{3,Matrix{ComplexF64}}}
    H_ctrl_mw     :: Vector{Matrix{ComplexF64}}
    H_ctrl_rf     :: Vector{Matrix{ComplexF64}}
    H_drift       :: Matrix{ComplexF64}
    H_controls    :: Vector{Matrix{ComplexF64}}
    n_controls    :: Int
    interactions  :: Vector{SphericalInteraction}
    H0            :: Matrix{ComplexF64}
end

"""
    electron_polarized_state(sys::DNPSpinSystem) -> Matrix{ComplexF64}

Return the initial density matrix ρ₀ ∝ Sz (electron polarized, nuclei unpolarized).
"""
function electron_polarized_state(sys::DNPSpinSystem)::Matrix{ComplexF64}
    Sz = sys.S_ops[3]
    denom = real(tr(Sz * Sz))
    denom < 1e-15 && return Sz
    return Sz / denom
end

"""
    nuclear_polarization_operator(sys::DNPSpinSystem) -> Matrix{ComplexF64}

Return the nuclear polarization observable O_I = Σ_k I_{z,k}.
"""
function nuclear_polarization_operator(sys::DNPSpinSystem)::Matrix{ComplexF64}
    return sum(ops[3] for ops in sys.I_ops)
end

"""
    dnp_system(S_electron, I_nuclei, B0_tesla, mw_freq_hz, omega_r_hz;
               g_vals, g_euler, A_vals, A_euler,
               csa_electron, csa_nuclei, dipolar, rf_freq_hz,
               nuclear_isotope, gamma_n_hz_T,
               extra_drift_terms, custom_controls,
               extra_jump_ops, extra_decay_rates) -> DNPSpinSystem

Construct a DNPSpinSystem.

# Key options
- `A_vals`           — hyperfine principal values per nucleus in **Hz** (internally multiplied by 2π).
- `nuclear_isotope`  — `String` (single isotope, broadcast to all nuclei) or
                        `Vector{String}` (per-nucleus), keys from `GYRO_MHZ_PER_T`.
                        Default `"1H"`. Sets γ_n/2π for the nuclear Zeeman term.
- `gamma_n_hz_T`     — explicit override (Float64 or Vector{Float64}, Hz/T); if
                        provided, takes precedence over `nuclear_isotope`.
- `extra_drift_terms` — list of extra matrices appended to H_drift (rad/s).
- `custom_controls`   — if provided, **replaces** the default [Sx, Sy, (Ix, Iy)] controls.
- `extra_jump_ops`, `extra_decay_rates` — user-supplied Lindblad jump operators
  and per-operator decay rates (appended via `lindblad_system_from_jump_ops`
  downstream; stored in `metadata` for propagation-time use).
"""
function dnp_system(S_electron::Rational{Int},
                    I_nuclei::Vector{Rational{Int}},
                    B0_tesla::Float64,
                    mw_freq_hz::Float64,
                    omega_r_hz::Float64;
                    g_vals::NTuple{3,Float64}           = (2.0023, 2.0023, 2.0023),
                    g_euler::NTuple{3,Float64}          = (0.0, 0.0, 0.0),
                    A_vals::Vector{NTuple{3,Float64}}   = [(0.0,0.0,0.0) for _ in I_nuclei],
                    A_euler::Vector{NTuple{3,Float64}}  = [(0.0,0.0,0.0) for _ in I_nuclei],
                    csa_electron::Vector{CSATensor}     = CSATensor[],
                    csa_nuclei::Vector{CSATensor}       = CSATensor[],
                    dipolar::Vector{DipolarCoupling}    = DipolarCoupling[],
                    rf_freq_hz::Float64                 = 0.0,
                    nuclear_isotope::Union{String,Vector{String}} = "1H",
                    gamma_n_hz_T::Union{Nothing,Float64,Vector{Float64}} = nothing,
                    extra_drift_terms::Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                    custom_controls::Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
                    )::DNPSpinSystem

    omega_r = 2π * omega_r_hz
    dS = Int(2*S_electron + 1)
    dIs = [Int(2*I+1) for I in I_nuclei]
    dim = dS * prod(dIs; init=1)

    # ── Resolve per-nucleus γ_n/2π (Hz/T) ────────────────────────────────────
    n_nuc = length(I_nuclei)
    iso_vec = nuclear_isotope isa String ? fill(nuclear_isotope, n_nuc) :
                                            copy(nuclear_isotope)
    length(iso_vec) == n_nuc ||
        throw(ArgumentError("nuclear_isotope length $(length(iso_vec)) ≠ n_nuclei $n_nuc"))
    γn_vec = if gamma_n_hz_T === nothing
        [GYRO_MHZ_PER_T[iso] * 1e6 for iso in iso_vec]
    elseif gamma_n_hz_T isa Float64
        fill(gamma_n_hz_T, n_nuc)
    else
        length(gamma_n_hz_T) == n_nuc ||
            throw(ArgumentError("gamma_n_hz_T length $(length(gamma_n_hz_T)) ≠ n_nuclei $n_nuc"))
        copy(gamma_n_hz_T)
    end

    # Electron spin operators
    Sx_loc = spin_Sx(Float64(S_electron))
    Sy_loc = spin_Sy(Float64(S_electron))
    Sz_loc = spin_Sz(Float64(S_electron))

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

    # ── Orientation-INDEPENDENT drift H0 (nuclear Zeeman + extra terms) ──────
    # The electron Zeeman offset and the (an)isotropic hyperfine/g/dipolar/CSA
    # interactions are built below as SphericalInteractions so they rotate with
    # the crystallite; only the purely isotropic nuclear Zeeman lives in H0.
    H0 = zeros(ComplexF64, dim, dim)
    for (k, _) in enumerate(I_nuclei)
        ω_I = 2π * γn_vec[k] * B0_tesla
        H0 .+= (ω_I - 2π*rf_freq_hz) .* I_ops_list[k][3]
    end
    for Hx in extra_drift_terms
        size(Hx) == (dim, dim) ||
            throw(ArgumentError("extra_drift_terms entry size $(size(Hx)) ≠ ($dim,$dim)"))
        H0 .+= Hx
    end

    # ── Anisotropic interactions (spherical-tensor engine) ───────────────────
    interactions = SphericalInteraction[]

    # Electron Zeeman with g-anisotropy: iso offset (ω_S,iso − 2π·mw) on Sz plus
    # the rank-2 g-tensor anisotropy.  Principal Zeeman frequencies ω_i = μ_B B0 g_i/ħ.
    ωg = ntuple(i -> _EPR_μ_B_SI * B0_tesla * g_vals[i] / _EPR_hbar_SI, 3)
    δg, ηg = principal_to_delta_eta(ωg)
    ω_S_iso = sum(ωg)/3
    push!(interactions,
          shift_interaction(ω_S_iso - 2π*mw_freq_hz, δg, ηg, g_euler, Sz_full))

    # Hyperfine per nucleus: isotropic + rank-2 anisotropy, secular Sz·Iz plus the
    # PSEUDOSECULAR Sz·I± that enables the solid effect.  A_vals in Hz → rad/s.
    for (k, Ak) in enumerate(A_vals)
        a = ntuple(i -> 2π * Ak[i], 3)
        A_iso = sum(a)/3
        δA, ηA = principal_to_delta_eta(a)
        Ixk, Iyk, Izk = I_ops_list[k]
        push!(interactions,
              hyperfine_interaction(A_iso, δA, ηA, A_euler[k], Sz_full, Ixk, Iyk, Izk))
    end

    # Electron CSA (rank-2 shift on Sz) and nuclear CSA (rank-2 shift on Iz_k).
    for c in csa_electron
        push!(interactions,
              shift_interaction(c.delta_iso, c.delta_aniso, c.eta, c.euler_PAS_to_MF, Sz_full))
    end
    for (k, c) in enumerate(csa_nuclei)
        k <= length(I_ops_list) || break
        push!(interactions,
              shift_interaction(c.delta_iso, c.delta_aniso, c.eta, c.euler_PAS_to_MF,
                                I_ops_list[k][3]))
    end

    # Dipolar couplings.  Index convention: 1 = electron, k+1 = nucleus k.
    # Electron–nucleus couplings are heteronuclear-secular (2·Sz·Iz); nucleus–
    # nucleus couplings are homonuclear (flip-flop-containing rank-2 operator).
    _spin_xyz(idx) = idx == 1 ? (Sx_full, Sy_full, Sz_full) : I_ops_list[idx-1]
    for dd in dipolar
        oxi, oyi, ozi = _spin_xyz(dd.spin_i)
        oxj, oyj, ozj = _spin_xyz(dd.spin_j)
        if dd.spin_i == 1 || dd.spin_j == 1
            push!(interactions,
                  dipolar_hetero_interaction(dd.b_ij, dd.euler_DD_to_MF, ozi, ozj))
        else
            push!(interactions,
                  dipolar_homo_interaction(dd.b_ij, dd.euler_DD_to_MF,
                                           oxi, oyi, ozi, oxj, oyj, ozj))
        end
    end

    # ── Build reference-orientation Fourier components via the shared engine ──
    # MAS (ω_r > 0) ⇒ magic-angle rotor projection; ω_r = 0 ⇒ static powder.
    tilt = omega_r > 0 ? MAGIC_ANGLE : 0.0
    H_fourier = build_orientation_fourier(interactions, (0.0, 0.0, 0.0), dim;
                                          theta_tilt = tilt, H0 = H0)
    H_drift = copy(H_fourier[0])

    H_ctrl_mw = Matrix{ComplexF64}[Sx_full, Sy_full]
    H_ctrl_rf = if isempty(I_nuclei)
        Matrix{ComplexF64}[]
    else
        Matrix{ComplexF64}[
            sum(ops[1] for ops in I_ops_list),
            sum(ops[2] for ops in I_ops_list)
        ]
    end
    H_controls_all = custom_controls === nothing ?
        vcat(H_ctrl_mw, H_ctrl_rf) : copy(custom_controls)

    return DNPSpinSystem(S_electron, I_nuclei, g_vals, g_euler,
                         A_vals, A_euler, csa_electron, csa_nuclei, dipolar,
                         omega_r, omega_r > 0 ? 2π/omega_r : Inf, H_fourier,
                         B0_tesla, mw_freq_hz, rf_freq_hz,
                         dim, S_ops, I_ops_list,
                         H_ctrl_mw, H_ctrl_rf, H_drift, H_controls_all,
                         length(H_controls_all), interactions, H0)
end

"""
    dnp_oriented_fourier(sys::DNPSpinSystem, α, β, γ) -> Dict{Int,Matrix{ComplexF64}}

Rotor-modulated Fourier components of `sys` for crystallite orientation (α,β,γ),
rebuilt from the stored anisotropic interactions via the shared spherical-tensor
engine.  This is what makes DNP powder averaging real: each crystallite sees a
genuinely different (an)isotropic Hamiltonian.
"""
function dnp_oriented_fourier(sys::DNPSpinSystem,
                              α::Float64, β::Float64, γ::Float64)::Dict{Int,Matrix{ComplexF64}}
    tilt = sys.omega_r > 0 ? MAGIC_ANGLE : 0.0
    return build_orientation_fourier(sys.interactions, (α, β, γ), sys.dim;
                                     theta_tilt = tilt, H0 = sys.H0)
end
