# SpinIObjectives.jl — High-level kernel structs for spin-I and multi-spin Trotter GRAPE
#
# Mirrors SU2Objectives.jl for:
#   SpinIKernel  — arbitrary single spin-I (dim = 2I+1), phase-only
#   TrotterKernel — N coupled spin-1/2 via Strang-split Trotter, phase-only
#
# Depends on:
#   Computation/SpinIPropagator.jl  (precompute_spinI_params, spinI_fidelity_gradient, …)
#   Computation/TrotterPropagator.jl (precompute_trotter_params, trotter_fidelity_gradient, …)
#   Types/QuantumSystem.jl           (spin_Sx, spin_Sz)
#   Computation/Propagators.jl       (compute_propagator)

# ─────────────────────────────────────────────────────────────────────────────
# SpinIKernel
# ─────────────────────────────────────────────────────────────────────────────

"""
    SpinIKernel

Self-contained problem specification for phase-only pulse optimisation of a single
spin-I (arbitrary half-integer or integer) system.  Holds the precomputed per-step
propagators D_k and the target specification; everything needed by the optimiser.

## Fields
- `Dk_matrices` : `Array{ComplexF64,3}` of shape `[N_ens, dim, dim]`
- `m_values`    : `Vector{Float64}` of length `dim = 2I+1` — Iz eigenvalues
- `target`      : for `:state` — `(; psi_init, psi_targ)` NamedTuple;
                  for `:real_trace` — `Matrix{ComplexF64}` (dim × dim)
- `metric`      : `:state` or `:real_trace`
- `dim`         : `2I+1`
- `N_ens`       : ensemble size

## Callable interface
`kernel(phi)` evaluates fidelity (no gradient).
Use `spinI_fidelity_and_grad(kernel, phi)` for `(F, GJ)`.

## Constructor
```julia
SpinIKernel(offsets_hz, RF_max_hz, dt, spin_I, target;
            amp_factors=nothing, metric=:state)
```
- `offsets_hz`  : resonance offsets in Hz
- `RF_max_hz`   : constant peak RF amplitude in Hz
- `dt`          : time step in seconds
- `spin_I`      : spin quantum number (0.5, 1.0, 1.5, …)
- `target`      : for `:state` — `(psi_init=..., psi_targ=...)`;
                  for `:real_trace` — `Matrix{ComplexF64}` unitary target
- `amp_factors` : optional B1 scale factors for robustness
- `metric`      : `:state` (default) or `:real_trace`
"""
struct SpinIKernel
    Dk_matrices :: Array{ComplexF64,3}
    m_values    :: Vector{Float64}
    target      :: Any                   # NamedTuple or Matrix{ComplexF64}
    metric      :: Symbol
    dim         :: Int
    N_ens       :: Int
end

function SpinIKernel(offsets_hz  :: AbstractVector{<:Real},
                     RF_max_hz   :: Real,
                     dt          :: Real,
                     spin_I      :: Real,
                     target;
                     amp_factors :: Union{Nothing,AbstractVector{<:Real}} = nothing,
                     metric      :: Symbol = :state)

    offsets_rad = 2π .* offsets_hz
    RF_max_rad  = 2π * RF_max_hz

    if amp_factors === nothing
        Dk, m_vals = precompute_spinI_params(offsets_rad, RF_max_rad, dt, spin_I)
    else
        Dk, m_vals = precompute_spinI_params_ensemble(offsets_rad, amp_factors,
                                                       RF_max_rad, dt, spin_I)
    end

    N_ens = size(Dk, 1)
    dim   = size(Dk, 2)

    return SpinIKernel(Dk, m_vals, target, metric, dim, N_ens)
end

# Callable: forward fidelity only
(k::SpinIKernel)(phi::AbstractVector{Float64}) =
    spinI_fidelity_forward(phi, k.Dk_matrices, k.target, k.m_values; metric=k.metric)

"""
    spinI_fidelity_and_grad(kernel::SpinIKernel, phi) → (F, GJ)

Fidelity and gradient of J = 1 − F w.r.t. phi for a SpinIKernel.
GJ can be passed directly as the `grad!` output to L-BFGS-B.
"""
function spinI_fidelity_and_grad(kernel::SpinIKernel, phi::AbstractVector{Float64})
    return spinI_fidelity_gradient(phi, kernel.Dk_matrices, kernel.target,
                                   kernel.m_values; metric=kernel.metric)
end

# ─────────────────────────────────────────────────────────────────────────────
# TrotterKernel
# ─────────────────────────────────────────────────────────────────────────────

"""
    TrotterKernel

Self-contained problem specification for phase-only pulse optimisation of N coupled
spin-1/2 particles using the Strang-split Trotter approximation.

## Fields
- `tp`       : NamedTuple from `precompute_trotter_params`
- `psi_init` : `Vector{ComplexF64}` — initial state in 2^n_spins dim space
- `psi_targ` : `Vector{ComplexF64}` — target state
- `metric`   : `:state` (ensemble state-transfer fidelity; only option currently)
- `n_spins`  : number of spin-1/2 particles
- `dim`      : `2^n_spins`
- `N_ens`    : ensemble size (number of drift Hamiltonians)

## Callable interface
`kernel(phi)` evaluates fidelity (no gradient).
Use `trotter_fidelity_and_grad(kernel, phi)` for `(F, GJ)`.

## Constructor
```julia
TrotterKernel(drifts, Omega_RF_per_spin, dt, n_spins, psi_init, psi_targ;
              metric=:state)
```
- `drifts`            : `Vector{Matrix}` of length N_ens — drift Hamiltonians (rad/s)
- `Omega_RF_per_spin` : `Vector{Float64}` of length n_spins — peak Rabi per spin (rad/s)
- `dt`                : time step in seconds
- `n_spins`           : number of coupled spin-1/2
- `psi_init`          : initial state, length `2^n_spins`
- `psi_targ`          : target  state, length `2^n_spins`

## Example (2-spin homonuclear)
```julia
# Build drifts from NMRSpinSystem for each ensemble member
sys     = mr_system(["1H","1H"]; shifts_ppm=[δ1, δ2], J_hz=[0 J; J 0],
                    field_mhz=600.0)
H_drift = hamiltonian(sys)   # single drift for on-resonance system

kernel = TrotterKernel([H_drift], [2π*10e3, 2π*10e3], 1e-6, 2,
                        spin_state(sys, :alpha_alpha),
                        spin_state(sys, :beta_beta))
```
"""
struct TrotterKernel
    tp       :: NamedTuple
    psi_init :: Vector{ComplexF64}
    psi_targ :: Vector{ComplexF64}
    metric   :: Symbol
    n_spins  :: Int
    dim      :: Int
    N_ens    :: Int
end

function TrotterKernel(drifts            :: AbstractVector,
                        Omega_RF_per_spin :: AbstractVector{<:Real},
                        dt                :: Real,
                        n_spins           :: Int,
                        psi_init          :: AbstractVector{<:Number},
                        psi_targ          :: AbstractVector{<:Number};
                        metric            :: Symbol = :state)

    tp    = precompute_trotter_params(drifts, Omega_RF_per_spin, dt, n_spins)
    N_ens = length(drifts)
    dim   = 2^n_spins

    @assert length(psi_init) == dim "psi_init length $(length(psi_init)) ≠ dim=$dim"
    @assert length(psi_targ) == dim "psi_targ length $(length(psi_targ)) ≠ dim=$dim"

    return TrotterKernel(tp,
                         ComplexF64.(psi_init),
                         ComplexF64.(psi_targ),
                         metric, n_spins, dim, N_ens)
end

# Callable: forward fidelity only
(k::TrotterKernel)(phi::AbstractVector{Float64}) =
    trotter_fidelity_forward(phi, k.tp, k.psi_init, k.psi_targ; metric=k.metric)

"""
    trotter_fidelity_and_grad(kernel::TrotterKernel, phi) → (F, GJ)

Fidelity and gradient of J = 1 − F w.r.t. phi for a TrotterKernel.
"""
function trotter_fidelity_and_grad(kernel::TrotterKernel, phi::AbstractVector{Float64})
    return trotter_fidelity_gradient(phi, kernel.tp, kernel.psi_init, kernel.psi_targ;
                                     metric=kernel.metric)
end
