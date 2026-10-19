# Types/NMRSpinSystem.jl
#
# Spin system types for solution NMR — both homonuclear and heteronuclear.
#
# Two complementary types are defined here:
#
#   MRSpinSystem — a single rotating frame; accepts any mix of isotopes from
#     GYRO_MHZ_PER_T.  Works for homonuclear (mr_system(["1H","1H"])) and
#     single-carrier heteronuclear (mr_system(["1H","13C"])) experiments.
#     Builds full tensor-product spin operators and the rotating-frame drift
#     Hamiltonian (chemical shifts + J-couplings).
#
#   HeteronuclearSystem — multiple rotating frames, one per nucleus species.
#     Constructed from a Vector{MRSpinSystem}, one per species, each with its
#     own carrier frequency.  Required when 1H (600 MHz) and 13C (150 MHz)
#     need separate frames for accurate selective-pulse simulation.
#
# Any nucleus present in GYRO_MHZ_PER_T / SPIN_QUANTUM_NUMBER is supported
# by both types.

using LinearAlgebra

# ─── Physical constants ───────────────────────────────────────────────────────

"""
Gyromagnetic ratio γ/(2π) in MHz/T for common NMR/EPR isotopes.
Positive = precesses in the same sense as ¹H; negative = opposite sense.
Values from IUPAC 2001 recommendations.
"""
const GYRO_MHZ_PER_T = Dict{String,Float64}(
    "1H"      =>  42.5774,
    "2H"      =>   6.5360,
    "13C"     =>  10.7084,
    "14N"     =>   3.0766,
    "15N"     =>  -4.3160,
    "17O"     =>  -5.7742,
    "19F"     =>  40.0776,
    "23Na"    =>  11.2686,
    "29Si"    =>  -8.4653,
    "31P"     =>  17.2351,
    "129Xe"   => -11.7778,
    "electron" => -28024.9522,
)

"""
Spin quantum number I for common isotopes.
"""
const SPIN_QUANTUM_NUMBER = Dict{String,Float64}(
    "1H"      => 0.5,
    "2H"      => 1.0,
    "13C"     => 0.5,
    "14N"     => 1.0,
    "15N"     => 0.5,
    "17O"     => 2.5,
    "19F"     => 0.5,
    "23Na"    => 1.5,
    "29Si"    => 0.5,
    "31P"     => 0.5,
    "129Xe"   => 0.5,
    "electron" => 0.5,
)

# ─── Single-spin operator primitives ─────────────────────────────────────────

function _Iz_single(I::Float64)::Matrix{ComplexF64}
    dim = round(Int, 2I + 1)
    m   = collect(I:-1.0:-I)
    return ComplexF64.(diagm(m))
end

function _Ip_single(I::Float64)::Matrix{ComplexF64}
    dim = round(Int, 2I + 1)
    Ip  = zeros(ComplexF64, dim, dim)
    ms  = collect(I:-1.0:-I)
    for row in 1:(dim - 1)
        m = ms[row + 1]
        Ip[row, row + 1] = sqrt(I * (I + 1) - m * (m + 1))
    end
    return Ip
end

function _single_spin_ops(I::Float64)
    Iz = _Iz_single(I)
    Ip = _Ip_single(I)
    Im = Matrix(Ip')
    Ix = (Ip + Im) / 2
    Iy = (Ip - Im) / (2im)
    return (Ix=Ix, Iy=Iy, Iz=Iz, Ip=Ip, Im=Im)
end

# ─── MRSpinSystem ─────────────────────────────────────────────────────────────

"""
    MRSpinSystem

Physical description of a solution NMR spin system. Stores isotope identities
and precomputed tensor-product spin operators for the composite Hilbert space.

Supports any nucleus in `GYRO_MHZ_PER_T`. Use `mr_system(["1H","1H"])` for
homonuclear or `mr_system(["1H","13C"])` for single-frame heteronuclear systems.
For multi-carrier heteronuclear experiments see [`HeteronuclearSystem`](@ref).

All operators follow the ℏ = 1 convention:
    Ix = 0.5σx,  Iy = 0.5σy,  Iz = 0.5σz  (for spin-1/2)

Construct via [`mr_system`](@ref).
"""
struct MRSpinSystem
    n_spins  :: Int
    isotopes :: Vector{String}
    spin_I   :: Vector{Float64}
    dim      :: Int
    Ix :: Vector{Matrix{ComplexF64}}
    Iy :: Vector{Matrix{ComplexF64}}
    Iz :: Vector{Matrix{ComplexF64}}
    Ip :: Vector{Matrix{ComplexF64}}
    Im :: Vector{Matrix{ComplexF64}}
end

"""
    mr_system(isotopes::Vector{String}) → MRSpinSystem
    mr_system(isotope::String)          → MRSpinSystem

Create an `MRSpinSystem` from a list of isotope labels, e.g. `["1H", "13C"]`.

Supported isotopes: $(join(sort(collect(keys(SPIN_QUANTUM_NUMBER))), ", ")).

Builds tensor-product spin operators for the composite Hilbert space.
For a single spin-1/2 (dim = 2): Ix = [0 0.5; 0.5 0], Iy = [0 -0.5i; 0.5i 0],
Iz = [0.5 0; 0 -0.5].

# Example
```julia
sys_H    = mr_system("1H")             # single proton (dim = 2)
sys_HC   = mr_system(["1H", "13C"])   # 1H–13C pair  (dim = 4)
sys_2H   = mr_system(["1H", "1H"])    # two protons   (dim = 4)
```
"""
function mr_system(isotopes::Vector{String})::MRSpinSystem
    for iso in isotopes
        iso == "electron" &&
            throw(ArgumentError(
                "mr_system is for nuclear spins only. For electron spins, " *
                "use epr_system(S, I_nuclei, B0_tesla, mw_freq_hz; ...)."))
        haskey(SPIN_QUANTUM_NUMBER, iso) ||
            throw(ArgumentError(
                "Unknown isotope '$iso'. Supported: " *
                join(sort(collect(filter(k -> k != "electron", keys(SPIN_QUANTUM_NUMBER)))), ", ")))
    end

    n    = length(isotopes)
    Is   = [SPIN_QUANTUM_NUMBER[iso] for iso in isotopes]
    dims = [round(Int, 2I + 1) for I in Is]
    D    = prod(dims)

    singles = [_single_spin_ops(I) for I in Is]

    function _embed(k::Int, op::Matrix{ComplexF64})::Matrix{ComplexF64}
        n == 1 && return op
        mats = [i == k ? op : Matrix{ComplexF64}(I, dims[i], dims[i])
                for i in 1:n]
        return kron(mats...)
    end

    Ix = [_embed(k, singles[k].Ix) for k in 1:n]
    Iy = [_embed(k, singles[k].Iy) for k in 1:n]
    Iz = [_embed(k, singles[k].Iz) for k in 1:n]
    Ip = [_embed(k, singles[k].Ip) for k in 1:n]
    Im = [_embed(k, singles[k].Im) for k in 1:n]

    return MRSpinSystem(n, isotopes, Is, D, Ix, Iy, Iz, Ip, Im)
end

mr_system(isotope::String) = mr_system([isotope])

# ─── Operator access ──────────────────────────────────────────────────────────

"""
    spin_op(sys, name::Symbol)                   → Matrix{ComplexF64}
    spin_op(sys, name::Symbol, k::Int)            → Matrix{ComplexF64}
    spin_op(sys, name::Symbol, isotope::String)   → Matrix{ComplexF64}

Return a spin operator in the full Hilbert space.

`name` must be one of: `:Ix`, `:Iy`, `:Iz`, `:Ip`, `:Im`.

- No third argument: requires a single-spin system; returns that spin's operator.
- `k::Int`: returns the operator for the kth spin.
- `isotope::String`: returns the **sum** over all spins of that isotope type.

# Example
```julia
sys = mr_system(["1H", "13C"])
Lx_H  = spin_op(sys, :Ix, "1H")    # Ix on proton (spin 1)
Lz_C  = spin_op(sys, :Iz, "13C")   # Iz on carbon (spin 2)
Lx_1  = spin_op(sys, :Ix, 1)       # same as Lx_H by index
```
"""
function spin_op(sys::MRSpinSystem, name::Symbol)::Matrix{ComplexF64}
    sys.n_spins == 1 ||
        throw(ArgumentError(
            "Omit isotope/index only for single-spin systems. " *
            "Use spin_op(sys, :$name, isotope) or spin_op(sys, :$name, k)."))
    return _ops_field(sys, name)[1]
end

function spin_op(sys::MRSpinSystem, name::Symbol, k::Int)::Matrix{ComplexF64}
    1 ≤ k ≤ sys.n_spins ||
        throw(ArgumentError("Spin index $k out of range [1, $(sys.n_spins)]"))
    return _ops_field(sys, name)[k]
end

function spin_op(sys::MRSpinSystem, name::Symbol, isotope::String)::Matrix{ComplexF64}
    indices = findall(==(isotope), sys.isotopes)
    isempty(indices) &&
        throw(ArgumentError("No spin of type '$isotope' in system " *
                            "(isotopes: $(sys.isotopes))"))
    ops = _ops_field(sys, name)
    return sum(ops[k] for k in indices)
end

function _ops_field(sys::MRSpinSystem, name::Symbol)
    name == :Ix && return sys.Ix
    name == :Iy && return sys.Iy
    name == :Iz && return sys.Iz
    name == :Ip && return sys.Ip
    name == :Im && return sys.Im
    throw(ArgumentError("Unknown operator name ':$name'. Use :Ix, :Iy, :Iz, :Ip, :Im"))
end

# ─── State constructors ───────────────────────────────────────────────────────

"""
    spin_state(sys::MRSpinSystem, name::Symbol) → Vector{ComplexF64}

Return a normalised pure state vector for a single spin-1/2 system (`dim = 2`).

Available states (using NMR/Spinach notation):

| Symbol | State        | Eigenvalue | Description             |
|--------|-------------|------------|-------------------------|
| `:Iz`  | `[1, 0]`    | ⟨Iz⟩ = +½ | thermal equilibrium (+z)|
| `:mIz` | `[0, 1]`    | ⟨Iz⟩ = −½ | inverted (−z)           |
| `:Ix`  | `[1, 1]/√2` | ⟨Ix⟩ = +½ | +x eigenstate           |
| `:mIx` | `[1,−1]/√2` | ⟨Ix⟩ = −½ | −x eigenstate           |
| `:Iy`  | `[1, i]/√2` | ⟨Iy⟩ = +½ | +y eigenstate           |
| `:mIy` | `[1,−i]/√2` | ⟨Iy⟩ = −½ | −y eigenstate           |

# Example
```julia
sys      = mr_system("1H")
rho_init = spin_state(sys, :Iz)    # thermal equilibrium
rho_targ = spin_state(sys, :mIy)   # target after 90°x: Iz → −Iy
```
"""
function spin_state(sys::MRSpinSystem, name::Symbol)::Vector{ComplexF64}
    sys.dim == 2 ||
        throw(ArgumentError(
            "spin_state by symbol name is only defined for single spin-1/2 " *
            "(dim = 2). Got dim = $(sys.dim)."))
    if     name == :Iz  || name == :alpha;  return ComplexF64[1, 0]
    elseif name == :mIz || name == :beta;   return ComplexF64[0, 1]
    elseif name == :Ix;  return ComplexF64[1,  1] / √2
    elseif name == :mIx; return ComplexF64[1, -1] / √2
    elseif name == :Iy;  return ComplexF64[1,  im] / √2
    elseif name == :mIy; return ComplexF64[1, -im] / √2
    elseif name == :Ip;  return ComplexF64[1, 0]   # |α⟩ — Hilbert-space rep. of I+ coherence
    elseif name == :Im;  return ComplexF64[0, 1]   # |β⟩ — Hilbert-space rep. of I− coherence
    else
        throw(ArgumentError(
            "Unknown state ':$name'. Available: :Iz, :mIz, :Ix, :mIx, :Iy, :mIy, :Ip, :Im"))
    end
end

# ─── Hamiltonian builder (solution NMR) ──────────────────────────────────────

"""
    hamiltonian(sys::MRSpinSystem; kwargs...) → Matrix{ComplexF64}

Build the rotating-frame drift Hamiltonian in rad/s.

# Keyword arguments

| Keyword          | Type                    | Default           | Description                          |
|------------------|-------------------------|-------------------|--------------------------------------|
| `shifts_ppm`     | `Vector{Float64}`       | `zeros(n_spins)`  | chemical shifts in ppm               |
| `B0_tesla`       | `Float64`               | `14.1`            | static field in Tesla (600 MHz ¹H)   |
| `offset_hz`      | `Float64`               | `0.0`             | global offset applied to all spins   |
| `offsets_hz`     | `Vector{Float64}`       | `fill(offset_hz)` | per-spin offset (overrides above)    |
| `couplings_hz`   | `Matrix{Float64}`       | `zeros(n,n)`      | symmetric J-coupling matrix in Hz    |

# Returns
Drift Hamiltonian matrix in rad/s (angular frequency), compatible with
`exp(-i H dt)` time evolution. Pass this into `MRControl.drifts`.

# Example
```julia
sys = mr_system(["1H", "13C"])

# Single on-resonance Hamiltonian
H0 = hamiltonian(sys)

# Frequency-offset ensemble for broadband pulse design (±6 kHz)
drifts = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-6000, 6000, length=25)]

# With chemical shifts and J-coupling
J = zeros(2, 2); J[1,2] = J[2,1] = 150.0  # 150 Hz 1H–13C coupling
H = hamiltonian(sys; shifts_ppm=[1.5, 30.0], B0_tesla=14.1, couplings_hz=J)
```
"""
function hamiltonian(sys::MRSpinSystem;
                     shifts_ppm    :: Vector{Float64}  = zeros(sys.n_spins),
                     B0_tesla      :: Float64           = 14.1,
                     offset_hz     :: Float64           = 0.0,
                     offsets_hz    :: Vector{Float64}   = fill(offset_hz, sys.n_spins),
                     couplings_hz  :: Matrix{Float64}   = zeros(sys.n_spins, sys.n_spins)
                    )::Matrix{ComplexF64}
    H = zeros(ComplexF64, sys.dim, sys.dim)

    for k in 1:sys.n_spins
        γ_hz_per_T = GYRO_MHZ_PER_T[sys.isotopes[k]] * 1e6
        ν0_hz      = abs(γ_hz_per_T) * B0_tesla
        shift_hz   = shifts_ppm[k] * ν0_hz * 1e-6
        Δν         = shift_hz + offsets_hz[k]
        H         .+= (2π * Δν) .* sys.Iz[k]
    end

    for i in 1:sys.n_spins, j in (i + 1):sys.n_spins
        if abs(couplings_hz[i, j]) > 1e-12
            H .+= (2π * couplings_hz[i, j]) .* (sys.Iz[i] * sys.Iz[j])
        end
    end

    return H
end

# ─── HeteronuclearSystem ──────────────────────────────────────────────────────

"""
    HeteronuclearSystem <: AbstractQuantumSystem

Multi-species spin system for heteronuclear NMR with separate carrier frequencies.
Each species has its own rotating frame; the drift Hamiltonian is built by
[`hamiltonian(sys::HeteronuclearSystem; ...)`](@ref).

In the rotating frame of each carrier the drift Hamiltonian is:

    H_drift = Σ_i 2π Ω_i Iz_i  +  2π Σ_{i<j} J_ij Iz_i Iz_j

where Ω_i = offset of spin i from its species carrier (Hz).

For single-frame heteronuclear systems (e.g. simple INEPT) use
[`MRSpinSystem`](@ref) with `mr_system(["1H","13C"])` instead.

# Fields
- `subsystems`           — one `MRSpinSystem` per nucleus type
- `carriers_hz`          — carrier frequency per subsystem (Hz)
- `spins_per_subsystem`  — number of spins in each subsystem
- `Iz_ops`               — global Iz operators for each spin (tensor-product embedded)
- `H_controls`           — control operators [Ix_1, Iy_1, Ix_2, Iy_2, ...]
- `H_drift`              — zero matrix (real drift built by `hamiltonian()`)
- `dim`                  — total Hilbert space dimension
- `n_controls`           — length(H_controls)
"""
struct HeteronuclearSystem <: AbstractQuantumSystem
    subsystems          :: Vector{MRSpinSystem}
    carriers_hz         :: Vector{Float64}
    spins_per_subsystem :: Vector{Int}
    Iz_ops              :: Vector{Matrix{ComplexF64}}
    H_controls          :: Vector{Matrix{ComplexF64}}
    H_drift             :: Matrix{ComplexF64}
    dim                 :: Int
    n_controls          :: Int
end

"""
    heteronuclear_system(subsystems, carriers_hz) → HeteronuclearSystem

Build a `HeteronuclearSystem` from a list of `MRSpinSystem` objects (one per
nucleus type) and their respective carrier frequencies in Hz.

# Example
```julia
sys_H  = mr_system("1H")
sys_C  = mr_system("13C")
hsys   = heteronuclear_system([sys_H, sys_C], [600e6, 150e6])
H      = hamiltonian(hsys; offsets_hz=[200.0, 50.0],
                     J_couplings=Dict((1,2) => 145.0))
```
"""
function heteronuclear_system(subsystems::Vector{MRSpinSystem},
                               carriers_hz::Vector{Float64})::HeteronuclearSystem
    length(subsystems) == length(carriers_hz) ||
        throw(ArgumentError("subsystems and carriers_hz must have the same length"))

    subsystem_dims      = [s.dim for s in subsystems]
    dim                 = prod(subsystem_dims)
    spins_per_subsystem = [s.n_spins for s in subsystems]

    Iz_ops     = Matrix{ComplexF64}[]
    H_controls = Matrix{ComplexF64}[]

    for (α, sys) in enumerate(subsystems)
        d_before = prod(subsystem_dims[1:α-1]; init=1)
        d_after  = prod(subsystem_dims[α+1:end]; init=1)
        I_before = Matrix{ComplexF64}(I, d_before, d_before)
        I_after  = Matrix{ComplexF64}(I, d_after,  d_after)

        for k in 1:sys.n_spins
            push!(Iz_ops, kron(I_before, kron(sys.Iz[k], I_after)))
        end

        for k in 1:sys.n_spins
            push!(H_controls, kron(I_before, kron(sys.Ix[k], I_after)))
            push!(H_controls, kron(I_before, kron(sys.Iy[k], I_after)))
        end
    end

    H_drift_zero = zeros(ComplexF64, dim, dim)
    return HeteronuclearSystem(subsystems, carriers_hz, spins_per_subsystem,
                               Iz_ops, H_controls, H_drift_zero, dim, length(H_controls))
end

"""
    hamiltonian(sys::HeteronuclearSystem; offsets_hz, J_couplings) → Matrix{ComplexF64}

Build the rotating-frame drift Hamiltonian for a heteronuclear system.

# Arguments
- `offsets_hz`  — chemical shift offset from carrier for each spin (length = total n_spins)
- `J_couplings` — `Dict` mapping `(i,j)` spin pairs to scalar J coupling in Hz
"""
function hamiltonian(sys::HeteronuclearSystem;
                     offsets_hz  :: Vector{Float64}                    = zeros(sum(sys.spins_per_subsystem)),
                     J_couplings :: Dict{Tuple{Int,Int},Float64}       = Dict{Tuple{Int,Int},Float64}()
                    )::Matrix{ComplexF64}
    n_spins = sum(sys.spins_per_subsystem)
    length(offsets_hz) == n_spins ||
        throw(ArgumentError("offsets_hz length $(length(offsets_hz)) ≠ n_spins $n_spins"))
    H = zeros(ComplexF64, sys.dim, sys.dim)
    for i in 1:n_spins
        H .+= (2π * offsets_hz[i]) .* sys.Iz_ops[i]
    end
    for ((i,j), Jij) in J_couplings
        H .+= (2π * Jij) .* (sys.Iz_ops[i] * sys.Iz_ops[j])
    end
    return H
end

# ─── Operators for ensemble axes (see Types/EnsembleAveraging.jl) ─────────────

function offset_operator(sys::MRSpinSystem; spins = :all)
    idx = spins === :all ? (1:sys.n_spins) : spins
    return sum(spin_op(sys, :Iz, i) for i in idx)
end

function coupling_operator(sys::MRSpinSystem, i::Int, j::Int; kind::Symbol = :isotropic)
    kind === :zz && return spin_op(sys, :Iz, i) * spin_op(sys, :Iz, j)
    kind === :isotropic || throw(ArgumentError("kind must be :isotropic or :zz"))
    return sum(spin_op(sys, a, i) * spin_op(sys, a, j) for a in (:Ix, :Iy, :Iz))
end
