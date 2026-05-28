# SU2Objectives.jl — High-level SU(2) kernel struct for phase-only pulse optimisation
#
# Depends on Computation/SU2Propagator.jl (Layer 1b).
# Provides SU2Kernel: a self-contained, callable problem specification that
# encapsulates the precomputed scalar SU(2) ensemble parameters and the target
# rotation.  Designed to compose with all Pulsar optimisers via plain closures.

using LinearAlgebra: norm

# ─────────────────────────────────────────────────────────────────────────────
# Pauli matrices (local; not re-exported to avoid shadowing QuantumSystem.jl)
# ─────────────────────────────────────────────────────────────────────────────

const _SU2_SX = ComplexF64[0 1; 1 0]
const _SU2_SY = ComplexF64[0 -im; im 0]
const _SU2_SZ = ComplexF64[1 0; 0 -1]

# ─────────────────────────────────────────────────────────────────────────────
# SU2Kernel
# ─────────────────────────────────────────────────────────────────────────────

"""
    SU2Kernel

Self-contained problem specification for phase-only SU(2) pulse optimisation.
Holds the precomputed scalar ensemble parameters and the SU(2) target.

## Fields
- `alpha`  : precomputed time-independent SU(2) A-factors, length `N_ens`
- `betas`  : precomputed time-independent SU(2) scalar factors, length `N_ens`
- `AT`     : target SU(2) A component, |AT|² + |BT|² = 1
- `BT`     : target SU(2) B component
- `metric` : `:real_trace` or `:squared`
- `N_ens`  : ensemble size (N_off, or N_off × N_amp for robustness)

## Callable interface
`kernel(phi)` evaluates the fidelity forward pass (no gradient).
Use `su2_fidelity_and_grad(kernel, phi)` to get `(F, GJ)`.

## Constructor from physical parameters
```julia
SU2Kernel(offsets_hz, RF_max_hz, dt, rotation_angle, Theta, Phi_ax;
          amp_factors=nothing, metric=:real_trace)
```
- `offsets_hz`     : resonance offsets in Hz (AbstractVector)
- `RF_max_hz`      : constant peak RF amplitude in Hz
- `dt`             : time step in seconds
- `rotation_angle` : target rotation angle in radians (e.g. π/2 for 90°)
- `Theta`          : polar angle of rotation axis  (radians)
- `Phi_ax`         : azimuthal angle of rotation axis (radians)
- `amp_factors`    : optional Vector of B1 scale factors for robustness
                     (e.g. `[0.9,1.0,1.1]`); if `nothing`, pure offset ensemble
- `metric`         : `:real_trace` (default) or `:squared`
"""
struct SU2Kernel
    alpha  :: Vector{ComplexF64}
    betas  :: Vector{ComplexF64}
    AT     :: ComplexF64
    BT     :: ComplexF64
    metric :: Symbol
    N_ens  :: Int
end

function SU2Kernel(offsets_hz        :: AbstractVector{<:Real},
                   RF_max_hz         :: Real,
                   dt                :: Real,
                   rotation_angle    :: Real,
                   Theta             :: Real,
                   Phi_ax            :: Real;
                   amp_factors       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
                   metric            :: Symbol = :real_trace)

    offsets_rad = 2π .* offsets_hz
    RF_max_rad  = 2π * RF_max_hz

    # Precompute SU(2) scalar ensemble parameters
    if amp_factors === nothing
        params = precompute_su2_params(offsets_rad, RF_max_rad, dt)
    else
        params = precompute_su2_params_ensemble(offsets_rad, amp_factors, RF_max_rad, dt)
    end

    # Validate unitarity
    validate_su2_params(params.alpha, params.betas)

    # Target SU(2) A, B from rotation axis (Theta, Phi_ax) and angle
    nT  = sin(Theta)*cos(Phi_ax) .* _SU2_SX .+
          sin(Theta)*sin(Phi_ax) .* _SU2_SY .+
          cos(Theta)              .* _SU2_SZ
    UT  = exp(-im * nT * (rotation_angle / 2))   # 2×2 matrix exp via LinearAlgebra
    AT  = ComplexF64(UT[1, 1])
    BT  = ComplexF64(UT[2, 1])

    @assert isapprox(abs2(AT) + abs2(BT), 1.0; atol=1e-10) "Target SU(2) norm != 1"

    N_ens = length(params.alpha)
    return SU2Kernel(params.alpha, params.betas, AT, BT, metric, N_ens)
end

# Callable: forward fidelity only
(k::SU2Kernel)(phi::AbstractVector{Float64}) =
    su2_fidelity_forward(phi, k.alpha, k.betas, k.AT, k.BT; metric=k.metric)

"""
    su2_fidelity_and_grad(kernel::SU2Kernel, phi) → (F, GJ)

Evaluate fidelity `F` and the gradient `GJ` of the cost `J = 1 − F` w.r.t.
the phase vector `phi`.  `GJ` can be passed directly as the gradient to L-BFGS-B
(it is the gradient of the objective to minimise, not the negated fidelity gradient).
"""
function su2_fidelity_and_grad(kernel::SU2Kernel,
                               phi::AbstractVector{Float64})
    return su2_fidelity_gradient(phi, kernel.alpha, kernel.betas,
                                 kernel.AT, kernel.BT; metric=kernel.metric)
end
