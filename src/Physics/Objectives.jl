# Physics/Objectives.jl
# Fidelity objective functions for quantum control optimization.
# Extracted from Core/Fidelity.jl.

using LinearAlgebra
using Statistics: mean

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — Fidelity metric type hierarchy
#
# Type-dispatched fidelity variants eliminate runtime Symbol comparisons and
# allow the compiler to specialise on the metric without type-instability.
# The Symbol-based API (`type = :state_real`, etc.) is a thin wrapper that
# resolves the symbol once and delegates to the typed methods.
#
# Names follow the Φ^X_Y grid (see Types/Targets.jl): level X ∈ {Prop, State,
# DM} × reduction Y ∈ {Real, Modulus, Squared} of the complex overlap z.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AbstractFidelityMetric

Abstract supertype for all fidelity metric singletons.

Concrete subtypes are zero-field structs used as dispatch tags.  Each is a
choice of what is compared (propagator, state or density matrix) and how the
complex overlap `z` is reduced to a real number:

| Symbol           | Type           | Paper     | Formula                       |
|:---------------- |:-------------- |:--------- |:----------------------------- |
| `:prop_real`     | `PropReal`     | Φ^P_R     | Re Tr(U_t† U)/dim             |
| `:prop_modulus`  | `PropModulus`  | Φ^P_M     | \\|Tr(U_t† U)\\|/dim          |
| `:prop_squared`  | `PropSquared`  | Φ^P_Q     | \\|Tr(U_t† U)\\|²/dim²        |
| `:state_real`    | `StateReal`    | Φ^S_R     | Re⟨ψ_t\\|ψ⟩                   |
| `:state_modulus` | `StateModulus` | Φ^S_M     | \\|⟨ψ_t\\|ψ⟩\\|               |
| `:state_squared` | `StateSquared` | Φ^S_Q     | \\|⟨ψ_t\\|ψ⟩\\|²              |
| `:dm_real`       | `DMReal`       | Φ^D_R     | Re Tr(ρ_t† ρ)                 |
| `:dm_modulus`    | `DMModulus`    | Φ^D_M     | \\|Tr(ρ_t† ρ)\\|              |
| `:dm_squared`    | `DMSquared`    | Φ^D_Q     | \\|Tr(ρ_t† ρ)\\|²             |
| `:prop_average`  | `PropAverage`  | Φ^P_avg   | (dim·Φ^P_Q + 1)/(dim + 1)     |
| `:dm_uhlmann`    | `DMUhlmann`    | Φ^D_Uhl   | (Tr√(√ρ σ √ρ))²               |

The pre-Φ^X_Y type names (`RealOverlap`, `NormalizedGate`, …) remain as
deprecated aliases.
"""
abstract type AbstractFidelityMetric end

# ── Propagator level (P) ─────────────────────────────────────────────────────
"""
Φ^P_R (Prop-Real): `Re[Tr(U_t† U)] / dim`.

This is a **signed** phase-sensitive overlap, **not** a fidelity — it can
take values in `[-1, 1]`.  It is suitable as an objective for gradient ascent
(because the gradient is linear in the overlap, not in its modulus) and for
tracking phase alignment during optimisation.

Use `PropSquared` if you want a conventional fidelity in `[0, 1]`.
"""
struct PropReal     <: AbstractFidelityMetric end

"Φ^P_M (Prop-Modulus): |Tr(U_t† U)|/dim  (modulus of the normalised gate overlap)."
struct PropModulus  <: AbstractFidelityMetric end

"Φ^P_Q (Prop-Squared): |Tr(U_t† U)|²/dim²  (standard process fidelity)."
struct PropSquared  <: AbstractFidelityMetric end

"Φ^P_avg (Prop-Average): (dim·Φ^P_Q + 1)/(dim + 1)  (Haar-average gate fidelity)."
struct PropAverage  <: AbstractFidelityMetric end

# ── State level (S) ──────────────────────────────────────────────────────────
"Φ^S_R (State-Real): Re(⟨ψ_target|ψ_final⟩).  Phase-sensitive, range [−1, 1]."
struct StateReal    <: AbstractFidelityMetric end

"Φ^S_M (State-Modulus): |⟨ψ_target|ψ_final⟩|.  Non-smooth at zero, range [0, 1]."
struct StateModulus <: AbstractFidelityMetric end

"Φ^S_Q (State-Squared): |⟨ψ_target|ψ_final⟩|².  Phase-insensitive, range [0, 1]."
struct StateSquared <: AbstractFidelityMetric end

# ── Density-matrix level (D) ─────────────────────────────────────────────────
"Φ^D_R (DM-Real): Re Tr(ρ† σ).  Fast linear approximation for mixed states."
struct DMReal       <: AbstractFidelityMetric end

"Φ^D_M (DM-Modulus): |Tr(ρ† σ)|  (modulus of the Hilbert–Schmidt overlap)."
struct DMModulus    <: AbstractFidelityMetric end

"Φ^D_Q (DM-Squared): |Tr(ρ† σ)|²  (squared Hilbert–Schmidt overlap)."
struct DMSquared    <: AbstractFidelityMetric end

"Φ^D_Uhl (DM-Uhlmann): Uhlmann–Jozsa (Tr√(√ρ σ √ρ))².  For mixed states."
struct DMUhlmann    <: AbstractFidelityMetric end

# ── Theme 4 — Subspace / cooperative / process-tomography metrics ──────────

"""
    EssentialSubspaceGate(indices)

Gate fidelity restricted to an *essential* subspace defined by row/column
indices (e.g. the qubit subspace `[1, 2]` inside a 3- or 4-level transmon
Hilbert space):

    F = |Tr(Π_e U_target† U Π_e)|² / dim_e²

If `U_target` is supplied at the *full* dimension, it is restricted to the
same indices automatically. If it is supplied at the *essential* dimension,
it is used as-is. Source: Quandary `optim_target` with guard levels.
"""
struct EssentialSubspaceGate <: AbstractFidelityMetric
    indices :: Vector{Int}
end

"""
    CooperativeTargetFidelity(gate_metric, state_metric; α=0.5, β=0.5)

Single-scalar weighted sum of a gate fidelity and a state-transfer fidelity:

    F = α · gate_fidelity(U, U_target, gate_metric)
      + β · state_fidelity(ψ_target, U·ψ_init, state_metric)

Source: Quandary `optim_target = "gate, file"`. Use this when both a gate
shape and a specific state preparation must be enforced inside one cost
function rather than via the multi-objective optimiser.
"""
struct CooperativeTargetFidelity{G<:AbstractFidelityMetric,S<:AbstractFidelityMetric} <: AbstractFidelityMetric
    gate_metric  :: G
    state_metric :: S
    alpha        :: Float64
    beta         :: Float64
end

CooperativeTargetFidelity(g::AbstractFidelityMetric, s::AbstractFidelityMetric;
                          α::Real = 0.5, β::Real = 0.5) =
    CooperativeTargetFidelity(g, s, Float64(α), Float64(β))

"""
    ProcessTomographyFidelity(dim)

Process fidelity computed via the basis-state decomposition:

    F_proc = (1/d²) Σ_k |⟨e_k| U_target† U |e_k⟩|²

where `{e_k}` is the computational basis.  Equal to `|Tr(U_target† U)|²/d²`
for unitary `U`, but is the *correct* expression when `U` is a quantum
channel reconstructed from basis-state transfers (Quandary
`initialcondition = "basis"`). Lifts the basis-state pattern from
[comparisons/Translator/TransmonAnnotation.jl] into a first-class metric.
"""
struct ProcessTomographyFidelity <: AbstractFidelityMetric
    dim :: Int
end

# ── Exported singletons (use as values, e.g. state_fidelity(ψ, φ, STATE_REAL)) ─
const PROP_REAL     = PropReal()
const PROP_MODULUS  = PropModulus()
const PROP_SQUARED  = PropSquared()
const PROP_AVERAGE  = PropAverage()
const STATE_REAL    = StateReal()
const STATE_MODULUS = StateModulus()
const STATE_SQUARED = StateSquared()
const DM_REAL       = DMReal()
const DM_MODULUS    = DMModulus()
const DM_SQUARED    = DMSquared()
const DM_UHLMANN    = DMUhlmann()

# ── Φ^X_Y name of each overlap metric type ──────────────────────────────────
metric_symbol(::PropReal)     = :prop_real
metric_symbol(::PropModulus)  = :prop_modulus
metric_symbol(::PropSquared)  = :prop_squared
metric_symbol(::PropAverage)  = :prop_average
metric_symbol(::StateReal)    = :state_real
metric_symbol(::StateModulus) = :state_modulus
metric_symbol(::StateSquared) = :state_squared
metric_symbol(::DMReal)       = :dm_real
metric_symbol(::DMModulus)    = :dm_modulus
metric_symbol(::DMSquared)    = :dm_squared
metric_symbol(::DMUhlmann)    = :dm_uhlmann

const _PropLevelMetric  = Union{PropReal, PropModulus, PropSquared}
const _StateLevelMetric = Union{StateReal, StateModulus, StateSquared}
const _DMLevelMetric    = Union{DMReal, DMModulus, DMSquared}

# ── Pre-Φ^X_Y names, kept as deprecated aliases ─────────────────────────────
Base.@deprecate_binding RealOverlap       StateReal
Base.@deprecate_binding SquaredOverlap    StateSquared
Base.@deprecate_binding ModulusOverlap    StateModulus
Base.@deprecate_binding UhlmannFidelity   DMUhlmann
Base.@deprecate_binding LinearDMFidelity  DMReal
Base.@deprecate_binding SquaredDMFidelity DMSquared
Base.@deprecate_binding ModulusDMFidelity DMModulus
Base.@deprecate_binding NormalizedGate    PropSquared
Base.@deprecate_binding RealGate          PropReal
Base.@deprecate_binding AverageGate       PropAverage
Base.@deprecate_binding ModulusGate       PropModulus
Base.@deprecate_binding REAL_OVERLAP      STATE_REAL
Base.@deprecate_binding SQUARED_OVERLAP   STATE_SQUARED
Base.@deprecate_binding MODULUS_OVERLAP   STATE_MODULUS
Base.@deprecate_binding UHLMANN_FIDELITY  DM_UHLMANN
Base.@deprecate_binding LINEAR_DM         DM_REAL
Base.@deprecate_binding SQUARED_DM        DM_SQUARED
Base.@deprecate_binding MODULUS_DM        DM_MODULUS
Base.@deprecate_binding NORMALIZED_GATE   PROP_SQUARED
Base.@deprecate_binding REAL_GATE         PROP_REAL
Base.@deprecate_binding AVERAGE_GATE      PROP_AVERAGE
Base.@deprecate_binding MODULUS_GATE      PROP_MODULUS

# ── Symbol → metric conversion (used by the Symbol-keyword wrappers) ────────
# Canonical names are matched first; anything else (including a pre-Φ^X_Y
# name) goes through `canonical_metric`, which translates or throws.
const _STATE_FN_METRICS = (STATE_METRICS..., DM_METRICS..., :dm_uhlmann)
const _GATE_FN_METRICS  = (PROP_METRICS..., :prop_average)

function _state_symbol_to_metric(sym::Symbol)::AbstractFidelityMetric
    sym === :state_real    && return STATE_REAL
    sym === :state_squared && return STATE_SQUARED
    sym === :state_modulus && return STATE_MODULUS
    sym === :dm_real       && return DM_REAL
    sym === :dm_squared    && return DM_SQUARED
    sym === :dm_modulus    && return DM_MODULUS
    sym === :dm_uhlmann    && return DM_UHLMANN
    return _state_symbol_to_metric(
        canonical_metric(sym, _STATE_FN_METRICS; context = "state_fidelity"))
end

function _gate_symbol_to_metric(sym::Symbol)::AbstractFidelityMetric
    sym === :prop_squared && return PROP_SQUARED
    sym === :prop_real    && return PROP_REAL
    sym === :prop_modulus && return PROP_MODULUS
    sym === :prop_average && return PROP_AVERAGE
    return _gate_symbol_to_metric(
        canonical_metric(sym, _GATE_FN_METRICS; context = "gate_fidelity"))
end

# ── Krotov χ-boundary derivation ─────────────────────────────────────────────
#
# Krotov's method requires a boundary condition χ(T) = -∂J_T/∂⟨ψ(T)| (state
# target) or the analogous derivative in U-space.  For the linear real-overlap
# functional this is just the target itself; for nonlinear functionals like
# absolute-square the boundary depends on the current end-point trajectory.
#
# Mirrors Krotov.jl's `make_chi` (workspace.jl:171-176) — a metric-dispatched
# auto-generator so each new fidelity metric gets a working Krotov boundary
# without the user hand-deriving it.
#
# Convention: target is a `QuantumTarget`; ψ_T is the trajectory end-point
# (state vector for state targets, propagator for unitary targets).  The
# returned object has the same shape as the target data.
"""
    make_chi(metric::AbstractFidelityMetric, target, ψ_T) -> Vector{ComplexF64}
                                                            or Matrix{ComplexF64}

Return the Krotov co-state boundary χ(T) consistent with the given fidelity
metric.  Used by `krotov_optimize` when no `chi_constructor` is supplied;
mirrors the auto-derivation pattern in Krotov.jl.
"""
make_chi(::StateReal,    target, _ψ_T) = target.target_state
make_chi(::StateModulus, target, _ψ_T) = target.target_state   # equivalent up to a phase factor
make_chi(::PropReal,       target, _U_T) = target.target_unitary

# Nonlinear functionals: boundary depends on the current trajectory.
function make_chi(::StateSquared, target, ψ_T)
    ψ_T === nothing && return target.target_state
    s = dot(target.target_state, ψ_T)
    return s .* target.target_state
end

function make_chi(::PropSquared, target, U_T)
    U_T === nothing && return target.target_unitary
    d  = target.dim
    α  = tr(target.target_unitary' * U_T) / (d * d)
    return α .* target.target_unitary
end

# ── Type-dispatched state fidelity ───────────────────────────────────────────

"""
    state_fidelity(ψ_targ, ψ_final, metric::AbstractFidelityMetric) -> Float64

Type-dispatched state fidelity.  Prefer this form in new code: the metric is
resolved at compile time, making the call branch-free and type-stable.

# Example
```julia
F = state_fidelity(ψ_target, ψ_final, STATE_REAL)
F = state_fidelity(ρ_target, ρ_final, DM_UHLMANN)
```
"""
state_fidelity(ψ_targ, ψ_final, m::_StateLevelMetric) =
    overlap_fidelity(metric_symbol(m), dot(ψ_targ, ψ_final))
state_fidelity(ψ_targ, ψ_final, ::DMUhlmann) = _sf_uhlmann(ψ_targ, ψ_final)

# ── DM Hilbert–Schmidt inner product Tr(ρ† σ) ────────────────────────────────
# Accepts either density matrices (as AbstractMatrix) or vectorised forms
# (as AbstractVector, which is the natural Liouville-space representation).
@inline _dm_inner(rho::AbstractMatrix, sigma::AbstractMatrix)::ComplexF64 =
    tr(rho' * sigma)
@inline _dm_inner(rho::AbstractVector, sigma::AbstractVector)::ComplexF64 =
    dot(rho, sigma)

# Matrix form: ρ, σ are density matrices (or operators), z_D = Tr(ρ† σ).
# DM-Real is the raw Re z_D — NOT clamped to [0, 1]: for traceless operators
# (operator transfer) the overlap is legitimately negative, and a perfect
# inversion must score −1, not 0.
state_fidelity(rho::AbstractMatrix, sigma::AbstractMatrix, m::_DMLevelMetric)::Float64 =
    overlap_fidelity(metric_symbol(m), _dm_inner(rho, sigma))

# Pure-state shortcut: with ρ_T = |ψ_T⟩⟨ψ_T| and ρ_f = |ψ_f⟩⟨ψ_f|,
#   z_D = Tr(ρ_T† ρ_f) = |⟨ψ_T|ψ_f⟩|² = |z_S|² (real, ≥ 0),
# so DM-Real = DM-Modulus = |z_S|² and DM-Squared = |z_S|⁴.
state_fidelity(ψ_targ::AbstractVector, ψ_final::AbstractVector, m::_DMLevelMetric)::Float64 =
    pure_state_fidelity(metric_symbol(m), dot(ψ_targ, ψ_final))

# ── Type-dispatched gate fidelity ────────────────────────────────────────────

"""
    gate_fidelity(U, U_target, metric::AbstractFidelityMetric) -> Float64

Type-dispatched gate fidelity.  Prefer this form in new code.

# Example
```julia
F = gate_fidelity(U_opt, U_target, PROP_SQUARED)
F = gate_fidelity(U_opt, U_target, PROP_AVERAGE)
```
"""
function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64},
                       m::Union{_PropLevelMetric, PropAverage})::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    size(U) == size(U_target) || throw(DimensionMismatch(
        "U $(size(U)) ≠ U_target $(size(U_target))"))
    dim = size(U, 1)
    return overlap_fidelity(metric_symbol(m), tr(U_target' * U) / dim; dim)
end

# ── Theme 4 — EssentialSubspaceGate dispatch ───────────────────────────────

function gate_fidelity(U::AbstractMatrix{ComplexF64},
                       U_target::AbstractMatrix{ComplexF64},
                       m::EssentialSubspaceGate)::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    idx  = m.indices
    d_e  = length(idx)
    d_e > 0 || throw(ArgumentError("EssentialSubspaceGate: indices must be non-empty"))
    d_full = size(U, 1)
    all(1 .≤ idx .≤ d_full) || throw(ArgumentError(
        "EssentialSubspaceGate: indices $(idx) out of range for U of size $d_full"))

    U_ess  = U[idx, idx]
    Ut_ess = if size(U_target, 1) == d_e
        U_target
    else
        size(U_target, 1) == d_full || throw(DimensionMismatch(
            "U_target dim $(size(U_target,1)) ≠ d_full $d_full or d_e $d_e"))
        U_target[idx, idx]
    end
    return abs2(tr(Ut_ess' * U_ess) / d_e)
end

# ── Theme 4 — ProcessTomographyFidelity dispatch ───────────────────────────
"""
    gate_fidelity(U, U_target, m::ProcessTomographyFidelity) -> Float64

Process fidelity reconstructed from basis-state transfers:
    F = (1/d²) · |Σ_k ⟨e_k| U_target† U |e_k⟩|²

The complex overlaps are summed *coherently* (this is the trace), then the
modulus is squared. For unitary `U`, the formula reduces to the standard
normalised gate fidelity `|Tr(U_target† U)|²/d²` — but written as a basis
sum it is what Quandary's `initialcondition="basis"` reconstructs when only
column-by-column propagated states are available.
"""
function gate_fidelity(U::AbstractMatrix{ComplexF64},
                       U_target::AbstractMatrix{ComplexF64},
                       m::ProcessTomographyFidelity)::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    d = m.dim
    size(U, 1) == d || throw(DimensionMismatch(
        "U dim $(size(U,1)) ≠ ProcessTomographyFidelity.dim $d"))
    size(U_target) == size(U) || throw(DimensionMismatch(
        "U_target $(size(U_target)) ≠ U $(size(U))"))
    z = ComplexF64(0)
    @inbounds for k in 1:d
        ϕ_k = @view U[:, k]
        ψ_k = @view U_target[:, k]
        z  += dot(ψ_k, ϕ_k)
    end
    return abs2(z) / (d * d)
end

# ── Theme 4 — CooperativeTargetFidelity helper ─────────────────────────────
"""
    cooperative_fidelity(U, U_target, ψ_init, ψ_target, m::CooperativeTargetFidelity) -> Float64

Evaluate `α · gate_fidelity(U, U_target, m.gate_metric)
        + β · state_fidelity(ψ_target, U·ψ_init, m.state_metric)`.

`gate_fidelity` is dispatched on `m.gate_metric`; `state_fidelity` on
`m.state_metric`. Both metrics may be any `AbstractFidelityMetric` for
which the corresponding dispatch exists.
"""
function cooperative_fidelity(U::AbstractMatrix{ComplexF64},
                              U_target::AbstractMatrix{ComplexF64},
                              ψ_init::AbstractVector{ComplexF64},
                              ψ_target::AbstractVector{ComplexF64},
                              m::CooperativeTargetFidelity)::Float64
    F_g  = gate_fidelity(Matrix(U), Matrix(U_target), m.gate_metric)
    ψ_f  = U * ψ_init
    F_s  = state_fidelity(ψ_target, ψ_f, m.state_metric)
    return m.alpha * F_g + m.beta * F_s
end

# ── Type-dispatched GRAPE prefactor ─────────────────────────────────────────

"""
    fidelity_grad_prefactor(z, inner, dt_pwr, metric::AbstractFidelityMetric) -> Float64

Type-dispatched version of `fidelity_grad_prefactor`.  Branch-free and
type-stable — the compiler specialises on `metric` at call time.
"""
# All overlap metrics share one rule (Computation/FidelityReductions.jl):
# dF = Re(w · dz) with the chain weight w(z), and here ∂z = −i·dt_pwr·inner, so
# ∂F/∂u = dt_pwr · Im(w · inner).  w = 1 (real), 2z̄ (squared), z̄/|z| (modulus;
# the z̄ phase factor is essential — without it the gradient is wrong whenever
# z is not real-positive).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 m::Union{_StateLevelMetric, _PropLevelMetric}) =
    dt_pwr * imag(overlap_weight(metric_symbol(m), z) * inner)

# DM metrics on a closed-system pure state: z_D = |z_S|², so the weight on the
# state overlap z_S is w_D(z_D)·2z̄_S (2z̄_S for DM-Real/-Modulus, 4|z_S|²z̄_S for
# DM-Squared).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 m::_DMLevelMetric) =
    dt_pwr * imag(pure_state_weight(metric_symbol(m), z) * inner)

# PropAverage: F_avg = (d·Φ^P_Q + 1)/(d+1) is affine in Φ^P_Q, so ∂F_avg =
# d/(d+1)·∂Φ^P_Q.  The chain factor here is the PropSquared one; the
# dimension-dependent d/(d+1) scale is applied by `compute_gradient_gate`
# (which knows `dim`).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::PropAverage) =
    fidelity_grad_prefactor(z, inner, dt_pwr, PROP_SQUARED)

# Catch-all: metrics with no closed-form first-order GRAPE gradient (e.g. the
# Uhlmann fidelity, which involves a matrix square root).  These are still fully
# usable as a *forward* figure of merit — for derivative-free optimizers
# (CMA-ES, PSO, DE, Nelder–Mead) and for evaluation — but cannot drive a
# gradient-based optimizer.  Fail with a clear message instead of a MethodError.
function fidelity_grad_prefactor(::ComplexF64, ::ComplexF64, ::Float64,
                                 metric::AbstractFidelityMetric)
    throw(ArgumentError(
        "$(nameof(typeof(metric))) has no closed-form GRAPE gradient. " *
        "Use it as a forward figure of merit with a derivative-free optimizer " *
        "(:cmaes, :pso, :de, :nelder_mead), or choose a differentiable metric: " *
        "gates → :prop_real, :prop_modulus, :prop_squared, :prop_average; " *
        "states → :state_real, :state_modulus, :state_squared, " *
        ":dm_real, :dm_modulus, :dm_squared."))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — State Fidelities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    STATE_FIDELITY_TYPES :: Tuple

All valid `type` symbols accepted by [`state_fidelity`](@ref).

| Symbol           | Paper    | Formula                       | Range    | Notes                     |
|:---------------- |:-------- |:----------------------------- |:-------- |:------------------------- |
| `:state_real`    | Φ^S_R    | Re⟨ψ_t\\|ψ(T)⟩                | [−1, 1]  | Default. Phase-coherent.  |
| `:state_modulus` | Φ^S_M    | \\|⟨ψ_t\\|ψ(T)⟩\\|             | [0, 1]   | Non-smooth at zero.       |
| `:state_squared` | Φ^S_Q    | \\|⟨ψ_t\\|ψ(T)⟩\\|²            | [0, 1]   | Phase-insensitive.        |
| `:dm_real`       | Φ^D_R    | Re Tr(ρ† σ)                   | [0, 1]   | Fast; pure-state approx.  |
| `:dm_modulus`    | Φ^D_M    | \\|Tr(ρ† σ)\\|                  | [0, 1]   |                           |
| `:dm_squared`    | Φ^D_Q    | \\|Tr(ρ† σ)\\|²                 | [0, 1]   |                           |
| `:dm_uhlmann`    | Φ^D_Uhl  | (Tr√(√ρ σ √ρ))²               | [0, 1]   | Mixed states / Lindblad.  |
"""
const STATE_FIDELITY_TYPES = _STATE_FN_METRICS

"""
    state_fidelity(ψ_targ, ψ_final; type::Symbol = :state_real) -> Float64

Compute a scalar fidelity between target state `ψ_targ` and achieved state `ψ_final`.

# Arguments
- `ψ_targ`  — target state vector (or density matrix for `:dm_*` types)
- `ψ_final` — achieved final state vector (or density matrix)
- `type`    — fidelity type; see [`STATE_FIDELITY_TYPES`](@ref)

# Examples
```julia
F = state_fidelity(ρtg, ψ)                       # :state_real (default)
F = state_fidelity(ρtg, ψ; type = :state_squared)
F = state_fidelity(ρ_mat, σ_mat; type = :dm_uhlmann)
```
"""
function state_fidelity(ψ_targ, ψ_final; type::Symbol = :state_real)::Float64
    return state_fidelity(ψ_targ, ψ_final, _state_symbol_to_metric(type))
end

# ── :dm_uhlmann — (Tr√(√ρ σ √ρ))² ────────────────────────────────────────
function _sf_uhlmann(rho::AbstractMatrix, sigma::AbstractMatrix)::Float64
    _check_square(rho, "rho"); _check_square(sigma, "sigma")
    size(rho) == size(sigma) || throw(DimensionMismatch(
        "rho $(size(rho)) ≠ sigma $(size(sigma))"))
    F_eig    = eigen(Hermitian(rho))
    λ        = max.(real.(F_eig.values), 0.0)
    V        = F_eig.vectors
    sqrt_rho = V * Diagonal(sqrt.(λ)) * V'
    M        = sqrt_rho * sigma * sqrt_rho
    μ        = real.(eigen(Hermitian(M)).values)
    return clamp((sum(sqrt.(max.(μ, 0.0))))^2, 0.0, 1.0)
end

"""
    state_overlap(ψ_targ, ψ_final) -> ComplexF64

Raw complex overlap ⟨ψ_targ|ψ_final⟩. Used by GRAPE gradient routines before
applying the fidelity chain-rule factor via [`fidelity_grad_prefactor`](@ref).
"""
@inline state_overlap(ψ_targ, ψ_final)::ComplexF64 = dot(ψ_targ, ψ_final)

# ─── Convenience alias ────────────────────────────────────────────────────
"""
    dm_fidelity(rho, sigma) -> Float64

Uhlmann–Jozsa fidelity (Tr√(√ρ σ √ρ))². Alias for
`state_fidelity(rho, sigma; type=:dm_uhlmann)`.
"""
dm_fidelity(rho::Matrix{ComplexF64}, sigma::Matrix{ComplexF64})::Float64 =
    _sf_uhlmann(rho, sigma)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1b — Gate Fidelities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GATE_FIDELITY_TYPES :: Tuple

All valid `type` symbols accepted by [`gate_fidelity`](@ref).

| Symbol          | Paper    | Formula                                    |
|:--------------- |:-------- |:------------------------------------------ |
| `:prop_real`    | Φ^P_R    | Re[Tr(U†_t U)]/dim                         |
| `:prop_modulus` | Φ^P_M    | \\|Tr(U†_t U)\\|/dim                       |
| `:prop_squared` | Φ^P_Q    | \\|Tr(U†_t U)\\|²/dim²  (process fidelity) |
| `:prop_average` | Φ^P_avg  | (dim·Φ^P_Q + 1)/(dim+1)  (Haar average)    |
"""
const GATE_FIDELITY_TYPES = _GATE_FN_METRICS

"""
    gate_fidelity(U, U_target; type::Symbol = :prop_squared) -> Float64

Gate fidelity between achieved propagator `U` and target `U_target`.

# Types
- `:prop_squared` — Φ^P_Q, |Tr(U†_t U)|²/dim² (default)
- `:prop_real`    — Φ^P_R, Re[Tr(U†_t U)]/dim
- `:prop_modulus` — Φ^P_M, |Tr(U†_t U)|/dim
- `:prop_average` — Φ^P_avg, (dim·Φ^P_Q + 1)/(dim+1): average over Haar-random inputs
"""
function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64};
                       type::Symbol = :prop_squared)::Float64
    return gate_fidelity(U, U_target, _gate_symbol_to_metric(type))
end

"""
    gate_fidelity_unnormalized(U, U_target) -> ComplexF64

Complex overlap Tr(U†_t U)/dim. Used by gradient routines before squaring.
"""
gate_fidelity_unnormalized(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64})::ComplexF64 =
    tr(U_target' * U) / size(U, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — GRAPE Gradient Pre-factor
# ═══════════════════════════════════════════════════════════════════════════════

"""
    fidelity_grad_prefactor(z, inner, dt_pwr; type::Symbol) -> Float64

Chain-rule coefficient ∂F/∂w[k,n] for the GRAPE adjoint method.

# Arguments
- `z`      — complex overlap ⟨ψ_targ|ψ(T)⟩
- `inner`  — inner product ⟨λ[n+1]|Op[k]|ψ[n]⟩
- `dt_pwr` — scalar = dt[n] × pwr_level (rad/s × s = dimensionless)
- `type`   — fidelity metric (must match the metric used to compute F)

| `type`                            | Gradient formula                            |
|:--------------------------------- |:------------------------------------------- |
| `:state_real`, `:prop_real`       | dt_pwr · Im(inner)                          |
| `:state_squared`, `:prop_squared` | 2 · dt_pwr · Im(z̄ · inner)                 |
| `:state_modulus`, `:prop_modulus` | dt_pwr · Im(z̄ · inner)/\\|z\\|  (0 if \\|z\\| < ε) |
| `:dm_real`, `:dm_modulus`         | 2 · dt_pwr · Im(z̄ · inner)  (pure ρ_T)     |
| `:dm_squared`                     | 4\\|z\\|² · dt_pwr · Im(z̄ · inner)  (pure ρ_T) |

Other metrics raise `ArgumentError` (mixed-state density-matrix fidelities
require Liouville-space adjoint gradients — see `lindblad_grad_prefactor`).
"""
@inline function fidelity_grad_prefactor(z::ComplexF64,
                                          inner::ComplexF64,
                                          dt_pwr::Float64;
                                          type::Symbol)::Float64
    if type === :state_real
        return fidelity_grad_prefactor(z, inner, dt_pwr, STATE_REAL)
    elseif type === :state_squared
        return fidelity_grad_prefactor(z, inner, dt_pwr, STATE_SQUARED)
    elseif type === :state_modulus
        return fidelity_grad_prefactor(z, inner, dt_pwr, STATE_MODULUS)
    elseif type === :prop_real
        return fidelity_grad_prefactor(z, inner, dt_pwr, PROP_REAL)
    elseif type === :prop_squared
        return fidelity_grad_prefactor(z, inner, dt_pwr, PROP_SQUARED)
    elseif type === :prop_modulus
        return fidelity_grad_prefactor(z, inner, dt_pwr, PROP_MODULUS)
    elseif type === :dm_real
        return fidelity_grad_prefactor(z, inner, dt_pwr, DM_REAL)
    elseif type === :dm_squared
        return fidelity_grad_prefactor(z, inner, dt_pwr, DM_SQUARED)
    elseif type === :dm_modulus
        return fidelity_grad_prefactor(z, inner, dt_pwr, DM_MODULUS)
    else
        return fidelity_grad_prefactor(z, inner, dt_pwr;
            type = canonical_metric(type, _GRAD_PREFACTOR_METRICS;
                                    context = "fidelity_grad_prefactor"))
    end
end

const _GRAD_PREFACTOR_METRICS = (STATE_METRICS..., DM_METRICS..., PROP_METRICS...)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Ensemble Fidelities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ensemble_fidelity(ψ_finals, ψ_targs;
                      weights  = nothing,
                      mode     :: Symbol = :mean,
                      fid_type :: Symbol = :state_real) -> Float64

Aggregate state fidelity over an ensemble of (final state, target) pairs.

# Arguments
- `ψ_finals` — vector of achieved final states (length N)
- `ψ_targs`  — vector of target states (same length N)
- `weights`  — optional weight vector (must sum to 1 for `:weighted`)
- `mode`     — aggregation rule:
  - `:mean`     — (1/N) Σ Fᵢ   or Σ wᵢ Fᵢ if `weights` given
  - `:weighted` — Σ wᵢ Fᵢ      (requires `weights`)
  - `:minimax`  — minᵢ Fᵢ      (worst-case / robust guarantee)
- `fid_type` — per-member fidelity type; see [`STATE_FIDELITY_TYPES`](@ref)

# Example
```julia
F = ensemble_fidelity(ψ_list, fill(ρtg, N); mode=:mean, fid_type=:state_real)
F = ensemble_fidelity(ψ_list, ρtg_list;    mode=:minimax)
```
"""
function ensemble_fidelity(ψ_finals::AbstractVector,
                            ψ_targs::AbstractVector;
                            weights  = nothing,
                            mode     :: Symbol = :mean,
                            fid_type :: Symbol = :state_real)::Float64
    N = length(ψ_finals)
    N == length(ψ_targs) || throw(ArgumentError(
        "ψ_finals has $N elements but ψ_targs has $(length(ψ_targs))"))
    N > 0 || throw(ArgumentError("ensemble is empty"))

    Fs = [state_fidelity(ψ_targs[i], ψ_finals[i]; type = fid_type) for i in 1:N]

    if mode == :mean
        return isnothing(weights) ? mean(Fs) : dot(weights, Fs)
    elseif mode == :weighted
        isnothing(weights) && throw(ArgumentError(
            "mode=:weighted requires a weights vector"))
        length(weights) == N || throw(ArgumentError(
            "weights length $(length(weights)) ≠ ensemble size $N"))
        return dot(weights, Fs)
    elseif mode == :minimax
        return minimum(Fs)
    else
        throw(ArgumentError(
            "Unknown ensemble mode ':$mode'. Valid: :mean, :weighted, :minimax"))
    end
end

# ─── Legacy 2-argument ensemble (backward compatibility) ──────────────────

"""
    ensemble_fidelity(propagators, targets) -> Float64

Backward-compatible form: average `compute_fidelity` over (propagator, target) pairs.
"""
function ensemble_fidelity(propagators::Vector{Matrix{ComplexF64}},
                            targets::Vector{QuantumTarget})::Float64
    N = length(propagators)
    N == length(targets) || throw(ArgumentError(
        "propagators has $N elements but targets has $(length(targets))"))
    N > 0 || throw(ArgumentError("propagators and targets are empty"))
    return sum(compute_fidelity(propagators[i], targets[i]) for i in 1:N) / N
end

"""
    infidelity(U_total, target) -> Float64

Return `1 − compute_fidelity(U_total, target)`. Minimisation-friendly.
"""
infidelity(U_total::Matrix{ComplexF64}, target::QuantumTarget)::Float64 =
    1.0 - compute_fidelity(U_total, target)

# ═══════════════════════════════════════════════════════════════════════════════
# Legacy API — backward compatibility with existing Pulsar Core code
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_fidelity(U_total, target) -> Float64

Legacy dispatcher. Calls `gate_fidelity` or `state_fidelity` based on `target.type`.
"""
function compute_fidelity(U_total::Matrix{ComplexF64}, target::QuantumTarget)::Float64
    dim = target.dim
    size(U_total) == (dim, dim) || throw(DimensionMismatch(
        "U_total size $(size(U_total)) ≠ target.dim = $dim"))
    if target.type == "unitary"
        target.target_unitary === nothing && throw(ArgumentError(
            "target.type is \"unitary\" but target.target_unitary is nothing"))
        sym = target.metric === :auto ? :prop_squared : target.metric
        return gate_fidelity(U_total, target.target_unitary; type = sym)
    elseif target.type == "state"
        target.target_state === nothing && throw(ArgumentError(
            "target.type is \"state\" but target.target_state is nothing"))
        sym = target.metric === :auto ? :state_squared : target.metric
        psi_init  = target.initial_state === nothing ?
                        target.target_state : target.initial_state
        psi_final = U_total * psi_init
        return state_fidelity(target.target_state, psi_final; type = sym)
    elseif target.type == "subspace"
        throw(ArgumentError("Subspace fidelity is not yet implemented"))
    else
        throw(ArgumentError(
            "Unknown target type \"$(target.type)\"; expected \"unitary\" or \"state\""))
    end
end

function compute_fidelity(system::AbstractQuantumSystem,
                           controls::ControlSequence,
                           target::QuantumTarget)::Float64
    H_total = build_total_hamiltonian(system, controls)
    props   = compute_propagators(H_total, controls.dt)
    U_total = compute_total_propagator(props)
    return compute_fidelity(U_total, target)
end

"""
    state_transfer_fidelity(U, psi_init, psi_target) -> Float64

Return |⟨ψ_target|U|ψ_init⟩|².
"""
function state_transfer_fidelity(U::Matrix{ComplexF64},
                                  psi_init::Vector{ComplexF64},
                                  psi_target::Vector{ComplexF64})::Float64
    size(U, 1) == size(U, 2) || throw(ArgumentError("U must be square"))
    nrm_i = norm(psi_init);  nrm_t = norm(psi_target)
    nrm_i < eps(Float64) && throw(ArgumentError("psi_init has zero norm"))
    nrm_t < eps(Float64) && throw(ArgumentError("psi_target has zero norm"))
    return abs2(dot(psi_target / nrm_t, U * (psi_init / nrm_i)))
end

"""
    state_transfer_fidelity_unnormalized(U, psi_init, psi_target) -> ComplexF64

Return complex overlap ⟨ψ_target|U|ψ_init⟩ without squaring. Used by gradient routines.
"""
function state_transfer_fidelity_unnormalized(U::Matrix{ComplexF64},
                                               psi_init::Vector{ComplexF64},
                                               psi_target::Vector{ComplexF64})::ComplexF64
    return dot(psi_target / norm(psi_target), U * (psi_init / norm(psi_init)))
end

# ─── Internal helpers ─────────────────────────────────────────────────────

function _check_square(A::AbstractMatrix, name::String)
    m, n = size(A)
    m == n || throw(ArgumentError("$name must be square, got $m × $n"))
end

# ============================================================================
# Band-selective objectives
# ============================================================================

"""
    BandWeight

Weight descriptor for band-selective pulse design.

# Fields
- `offset_hz` — frequency offset from carrier (Hz)
- `weight`    — positive = pass band (maximize F), negative = stop band (minimize F)
"""
struct BandWeight
    offset_hz :: Float64
    weight    :: Float64
end

"""
    shift_system(sys::QuantumSystem, offset_hz::Float64) -> QuantumSystem

Return a copy of `sys` with drift Hamiltonian shifted by 2π*offset_hz (uniform Zeeman offset).
"""
function shift_system(sys::QuantumSystem, offset_hz::Float64)::QuantumSystem
    dim = sys.dim
    H_shifted = sys.H_drift .+ (2π * offset_hz) .* Matrix{ComplexF64}(I, dim, dim)
    return QuantumSystem(H_shifted, sys.H_controls, dim, sys.n_controls, sys.metadata)
end

"""
    shift_system(sys::SpinSystem, offset_hz::Float64) -> SpinSystem

Return a copy of `sys` with uniform frequency offset added to all spins.
"""
function shift_system(sys::SpinSystem, offset_hz::Float64)::SpinSystem
    dim = sys.dim
    H_shifted = sys.H_drift .+ (2π * offset_hz) .* Matrix{ComplexF64}(I, dim, dim)
    return SpinSystem(sys.spins, sys.couplings, sys.chemical_shifts .+ offset_hz,
                      H_shifted, sys.H_controls, dim, sys.n_controls)
end

"""
    band_selective_fidelity(sys, ctrl, target, band_weights) -> Float64

Weighted fidelity over a frequency band:
    F_band = Σ_k w_k * F(sys + Δk, ctrl, target)

Positive weights = pass band; negative weights = stop band.
"""
function band_selective_fidelity(sys::AbstractQuantumSystem,
                                  ctrl::ControlSequence,
                                  target::QuantumTarget,
                                  band_weights::Vector{BandWeight})::Float64
    F = 0.0
    for bw in band_weights
        sys_δ = shift_system(sys, bw.offset_hz)
        F += bw.weight * compute_fidelity(sys_δ, ctrl, target)
    end
    return F
end

# Bloch/DNP fidelity functions are in Physics/MRPhysics.jl
# (loaded after Types/BlochSystem.jl and Types/DNPSpinSystem.jl)
