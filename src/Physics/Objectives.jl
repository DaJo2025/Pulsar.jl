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
# The legacy Symbol-based API (`type=:real`, etc.) is preserved as a thin
# wrapper that resolves the symbol once and delegates to the typed methods.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AbstractFidelityMetric

Abstract supertype for all fidelity metric singletons.

Concrete subtypes are zero-field structs used as dispatch tags:

| Type                   | Equivalent Symbol  | Formula                              |
|:---------------------- |:------------------ |:------------------------------------ |
| `RealOverlap`          | `:real`            | Re(⟨ψ_t\\|ψ⟩)                       |
| `SquaredOverlap`       | `:square`          | \\|⟨ψ_t\\|ψ⟩\\|²                    |
| `ModulusOverlap`       | `:modulus`         | \\|⟨ψ_t\\|ψ⟩\\|                     |
| `UhlmannFidelity`      | `:dm_uhlmann`      | (Tr√(√ρ σ √ρ))²                     |
| `LinearDMFidelity`     | `:dm_linear`       | Re Tr(ρ† σ)                         |
| `NormalizedGate`       | `:normalized`      | \\|Tr(U†_t U)\\|²/dim²              |
| `RealGate`             | `:real` (gate)     | Re[Tr(U†_t U)]/dim                  |
| `AverageGate`          | `:average`         | (dim·F_norm + 1)/(dim+1)            |
"""
abstract type AbstractFidelityMetric end

# ── State metrics ────────────────────────────────────────────────────────────
"Fidelity metric: Re(⟨ψ_target|ψ_final⟩).  Phase-sensitive, range [−1, 1]."
struct RealOverlap      <: AbstractFidelityMetric end

"Fidelity metric: |⟨ψ_target|ψ_final⟩|².  Phase-insensitive, range [0, 1]."
struct SquaredOverlap   <: AbstractFidelityMetric end

"Fidelity metric: |⟨ψ_target|ψ_final⟩|.  Non-smooth at zero, range [0, 1]."
struct ModulusOverlap   <: AbstractFidelityMetric end

"Fidelity metric: Uhlmann–Jozsa (Tr√(√ρ σ √ρ))².  For mixed states."
struct UhlmannFidelity  <: AbstractFidelityMetric end

"Fidelity metric: Re Tr(ρ† σ).  Fast linear approximation for mixed states."
struct LinearDMFidelity <: AbstractFidelityMetric end

# ── Gate metrics ─────────────────────────────────────────────────────────────
"Gate fidelity metric: |Tr(U†_t U)|²/dim²  (standard process fidelity)."
struct NormalizedGate   <: AbstractFidelityMetric end

"""
Gate overlap metric: `Re[Tr(U†_t U)] / dim`.

This is a **signed** phase-sensitive overlap, **not** a fidelity — it can
take values in `[-1, 1]`.  It is suitable as an objective for gradient ascent
(because the gradient is linear in the overlap, not in its modulus) and for
tracking phase alignment during optimisation.

Use `NormalizedGate` if you want a conventional fidelity in `[0, 1]`.
"""
struct RealGate         <: AbstractFidelityMetric end

"Gate fidelity metric: (dim·F_norm + 1)/(dim+1)  (Haar-average gate fidelity)."
struct AverageGate      <: AbstractFidelityMetric end

"Gate overlap metric: |Tr(U†_t U)|/dim  (modulus of normalised gate overlap)."
struct ModulusGate      <: AbstractFidelityMetric end

"DM fidelity metric: |Tr(ρ† σ)|²  (squared Hilbert–Schmidt overlap)."
struct SquaredDMFidelity <: AbstractFidelityMetric end

"DM fidelity metric: |Tr(ρ† σ)|  (modulus of Hilbert–Schmidt overlap)."
struct ModulusDMFidelity <: AbstractFidelityMetric end

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

# ── Exported singletons (use as values, e.g. state_fidelity(ψ, φ, REAL_OVERLAP)) ─
const REAL_OVERLAP     = RealOverlap()
const SQUARED_OVERLAP  = SquaredOverlap()
const MODULUS_OVERLAP  = ModulusOverlap()
const UHLMANN_FIDELITY = UhlmannFidelity()
const LINEAR_DM        = LinearDMFidelity()
const NORMALIZED_GATE  = NormalizedGate()
const REAL_GATE        = RealGate()
const AVERAGE_GATE     = AverageGate()
const MODULUS_GATE     = ModulusGate()
const SQUARED_DM       = SquaredDMFidelity()
const MODULUS_DM       = ModulusDMFidelity()

# ── Symbol → metric conversion (used by legacy wrappers) ────────────────────
function _state_symbol_to_metric(sym::Symbol)::AbstractFidelityMetric
    if sym === :real       return REAL_OVERLAP
    elseif sym === :square  return SQUARED_OVERLAP
    elseif sym === :modulus return MODULUS_OVERLAP
    elseif sym === :dm_uhlmann return UHLMANN_FIDELITY
    elseif sym === :dm_linear  return LINEAR_DM
    elseif sym === :dm_real    return LINEAR_DM
    elseif sym === :dm_square  return SQUARED_DM
    elseif sym === :dm_modulus return MODULUS_DM
    else
        throw(ArgumentError(
            "Unknown state fidelity type ':$sym'. " *
            "Valid: :real, :square, :modulus, :dm_uhlmann, :dm_linear, " *
            ":dm_real, :dm_square, :dm_modulus"))
    end
end

function _gate_symbol_to_metric(sym::Symbol)::AbstractFidelityMetric
    if sym === :normalized return NORMALIZED_GATE
    elseif sym === :real   return REAL_GATE
    elseif sym === :average return AVERAGE_GATE
    elseif sym === :modulus return MODULUS_GATE
    else
        throw(ArgumentError(
            "Unknown gate fidelity type ':$sym'. " *
            "Valid: :normalized, :real, :average, :modulus"))
    end
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
make_chi(::RealOverlap,    target, _ψ_T) = target.target_state
make_chi(::ModulusOverlap, target, _ψ_T) = target.target_state   # equivalent up to a phase factor
make_chi(::RealGate,       target, _U_T) = target.target_unitary

# Nonlinear functionals: boundary depends on the current trajectory.
function make_chi(::SquaredOverlap, target, ψ_T)
    ψ_T === nothing && return target.target_state
    s = dot(target.target_state, ψ_T)
    return s .* target.target_state
end

function make_chi(::NormalizedGate, target, U_T)
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
F = state_fidelity(ψ_target, ψ_final, REAL_OVERLAP)
F = state_fidelity(ρ_target, ρ_final, UHLMANN_FIDELITY)
```
"""
state_fidelity(ψ_targ, ψ_final, ::RealOverlap)     = real(dot(ψ_targ, ψ_final))
state_fidelity(ψ_targ, ψ_final, ::SquaredOverlap)  = abs2(dot(ψ_targ, ψ_final))
state_fidelity(ψ_targ, ψ_final, ::ModulusOverlap)  = abs(dot(ψ_targ, ψ_final))
state_fidelity(ψ_targ, ψ_final, ::UhlmannFidelity) = _sf_uhlmann(ψ_targ, ψ_final)
state_fidelity(ψ_targ::AbstractMatrix, ψ_final::AbstractMatrix, ::LinearDMFidelity) =
    _sf_linear(ψ_targ, ψ_final)
# Pure-state shortcut: with ρ_t = |ψ_t⟩⟨ψ_t| and ρ_f = |ψ_f⟩⟨ψ_f|,
#   Re Tr(ρ_t† ρ_f) = |⟨ψ_t|ψ_f⟩|² (real, ≥ 0).
state_fidelity(ψ_targ::AbstractVector, ψ_final::AbstractVector, ::LinearDMFidelity) =
    abs2(dot(ψ_targ, ψ_final))

# ── DM Hilbert–Schmidt inner product Tr(ρ† σ) ────────────────────────────────
# Accepts either density matrices (as AbstractMatrix) or vectorised forms
# (as AbstractVector, which is the natural Liouville-space representation).
@inline _dm_inner(rho::AbstractMatrix, sigma::AbstractMatrix)::ComplexF64 =
    tr(rho' * sigma)
@inline _dm_inner(rho::AbstractVector, sigma::AbstractVector)::ComplexF64 =
    dot(rho, sigma)

# Matrix form: ρ, σ are density matrices, _dm_inner = Tr(ρ† σ) = z_DM.
state_fidelity(rho::AbstractMatrix, sigma::AbstractMatrix, ::SquaredDMFidelity)::Float64 =
    abs2(_dm_inner(rho, sigma))
state_fidelity(rho::AbstractMatrix, sigma::AbstractMatrix, ::ModulusDMFidelity)::Float64 =
    abs(_dm_inner(rho, sigma))

# Pure-state shortcut: with ρ_T = |ψ_T⟩⟨ψ_T| and ρ_f = |ψ_f⟩⟨ψ_f|,
#   z_DM = Tr(ρ_T† ρ_f) = |⟨ψ_T|ψ_f⟩|² = |z_S|² (real, ≥ 0).
# Therefore |z_DM|  = |z_S|²   and  |z_DM|² = |z_S|⁴.
state_fidelity(ψ_targ::AbstractVector, ψ_final::AbstractVector, ::ModulusDMFidelity)::Float64 =
    abs2(dot(ψ_targ, ψ_final))               # |z_S|²  ≡  |z_DM|
state_fidelity(ψ_targ::AbstractVector, ψ_final::AbstractVector, ::SquaredDMFidelity)::Float64 =
    abs2(dot(ψ_targ, ψ_final))^2             # |z_S|⁴  ≡  |z_DM|²

# ── Type-dispatched gate fidelity ────────────────────────────────────────────

"""
    gate_fidelity(U, U_target, metric::AbstractFidelityMetric) -> Float64

Type-dispatched gate fidelity.  Prefer this form in new code.

# Example
```julia
F = gate_fidelity(U_opt, U_target, NORMALIZED_GATE)
F = gate_fidelity(U_opt, U_target, AVERAGE_GATE)
```
"""
function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64},
                       ::NormalizedGate)::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    size(U) == size(U_target) || throw(DimensionMismatch(
        "U $(size(U)) ≠ U_target $(size(U_target))"))
    dim = size(U, 1)
    return abs2(tr(U_target' * U) / dim)
end

function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64},
                       ::RealGate)::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    size(U) == size(U_target) || throw(DimensionMismatch(
        "U $(size(U)) ≠ U_target $(size(U_target))"))
    dim = size(U, 1)
    return real(tr(U_target' * U) / dim)
end

function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64},
                       ::AverageGate)::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    size(U) == size(U_target) || throw(DimensionMismatch(
        "U $(size(U)) ≠ U_target $(size(U_target))"))
    dim = size(U, 1)
    F_norm = abs2(tr(U_target' * U) / dim)
    return (dim * F_norm + 1) / (dim + 1)
end

function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64},
                       ::ModulusGate)::Float64
    _check_square(U, "U"); _check_square(U_target, "U_target")
    size(U) == size(U_target) || throw(DimensionMismatch(
        "U $(size(U)) ≠ U_target $(size(U_target))"))
    dim = size(U, 1)
    return abs(tr(U_target' * U) / dim)
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
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::RealOverlap)    = dt_pwr * imag(inner)
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::SquaredOverlap) = 2 * dt_pwr * imag(conj(z) * inner)
@inline function fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                          ::ModulusOverlap)
    absz = abs(z)
    absz < 1e-14 && return 0.0
    # ∂|z|/∂u = Re(z̄ ∂z)/|z| = dt_pwr · Im(z̄ · inner)/|z|  (∂z = −i·dt_pwr·inner).
    # The phase factor z̄ is essential — without it the gradient is wrong whenever
    # the overlap z is not real-positive (e.g. a purely imaginary trace).
    return dt_pwr * imag(conj(z) * inner) / absz
end

# ── Gate metrics whose chain factor matches the corresponding state metric ──
# NormalizedGate: F = |z|² (same chain factor as SquaredOverlap).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::NormalizedGate) = 2 * dt_pwr * imag(conj(z) * inner)
# RealGate: F = Re(z) (same chain factor as RealOverlap).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::RealGate)      = dt_pwr * imag(inner)

# ── New metric prefactors ────────────────────────────────────────────────────
# Modulus-of-gate-overlap: F = |z|, ∂F/∂u = dt_pwr · Im(inner / |z|).
@inline function fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64,
                                          dt_pwr::Float64, ::ModulusGate)
    absz = abs(z)
    absz < 1e-14 && return 0.0
    # ∂|z|/∂u = Re(z̄ ∂z)/|z| = dt_pwr · Im(z̄ · inner)/|z|  (∂z = −i·dt_pwr·inner).
    # The phase factor z̄ is essential — without it the gradient is wrong whenever
    # the overlap z is not real-positive (e.g. a purely imaginary trace).
    return dt_pwr * imag(conj(z) * inner) / absz
end

# AverageGate: F_avg = (d·F_norm + 1)/(d+1) is an affine function of the process
# fidelity F_norm = |z|², so ∂F_avg = d/(d+1) · ∂F_norm.  The chain factor here is
# the NormalizedGate one; the dimension-dependent d/(d+1) scale is applied by
# `compute_gradient_gate` (which knows `dim`).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::AverageGate) = 2 * dt_pwr * imag(conj(z) * inner)

# Linear DM fidelity (Re Tr(ρ† σ)) for closed-system pure ρ_T:
# F = Re|⟨ψ_T|ψ⟩|² = |⟨ψ_T|ψ⟩|² is real and ≥ 0, so the chain factor
# matches SquaredOverlap: ∂F/∂u = 2 · dt_pwr · Im(z̄ · inner).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::LinearDMFidelity)  = 2 * dt_pwr * imag(conj(z) * inner)

# Modulus DM fidelity: F = |z_DM| with z_DM = |z_S|² ≥ 0 ⇒ identical to
# SquaredOverlap when ρ_T is pure (closed system).
@inline fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64, dt_pwr::Float64,
                                 ::ModulusDMFidelity) = 2 * dt_pwr * imag(conj(z) * inner)

# Squared DM fidelity: F = |z_DM|² = |z_S|⁴, ∂F/∂u = 4|z_S|² · dt_pwr · Im(z̄·inner).
@inline function fidelity_grad_prefactor(z::ComplexF64, inner::ComplexF64,
                                          dt_pwr::Float64, ::SquaredDMFidelity)
    return 4 * abs2(z) * dt_pwr * imag(conj(z) * inner)
end

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
        "gates → :normalized, :real, :average, :modulus; " *
        "states → :real, :square, :modulus, :dm_linear, :dm_real, :dm_square, :dm_modulus."))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — State Fidelities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    STATE_FIDELITY_TYPES :: Tuple

All valid `type` symbols accepted by [`state_fidelity`](@ref).

| Symbol        | Formula                                     | Range    | Notes                        |
|:------------- |:------------------------------------------- |:-------- |:---------------------------- |
| `:real`       | Re(⟨ψ_t\\|ψ(T)⟩)                           | [−1, 1]  | Default. Phase-coherent.     |
| `:square`     | \\|⟨ψ_t\\|ψ(T)⟩\\|²                        | [0, 1]   | Phase-insensitive.           |
| `:modulus`    | \\|⟨ψ_t\\|ψ(T)⟩\\|                         | [0, 1]   | Non-smooth at zero.          |
| `:dm_uhlmann` | (Tr√(√ρ σ √ρ))²                             | [0, 1]   | Mixed states / Lindblad.     |
| `:dm_linear`  | Re Tr(ρ† σ)                                  | [0, 1]   | Fast; pure-state approx.     |
"""
const STATE_FIDELITY_TYPES = (:real, :square, :modulus, :dm_uhlmann, :dm_linear,
                              :dm_real, :dm_square, :dm_modulus)

"""
    state_fidelity(ψ_targ, ψ_final; type::Symbol = :real) -> Float64

Compute a scalar fidelity between target state `ψ_targ` and achieved state `ψ_final`.

# Arguments
- `ψ_targ`  — target state vector (or density matrix for `:dm_*` types)
- `ψ_final` — achieved final state vector (or density matrix)
- `type`    — fidelity type; see [`STATE_FIDELITY_TYPES`](@ref)

# Examples
```julia
F = state_fidelity(ρtg, ψ)                       # :real (default)
F = state_fidelity(ρtg, ψ; type = :square)
F = state_fidelity(ρ_mat, σ_mat; type = :dm_uhlmann)
```
"""
function state_fidelity(ψ_targ, ψ_final; type::Symbol = :real)::Float64
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

# ── :dm_linear — Re Tr(ρ† σ) ──────────────────────────────────────────────
function _sf_linear(rho::AbstractMatrix, sigma::AbstractMatrix)::Float64
    _check_square(rho, "rho"); _check_square(sigma, "sigma")
    return clamp(real(tr(rho' * sigma)), 0.0, 1.0)
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

| Symbol        | Formula                                         |
|:------------- |:----------------------------------------------- |
| `:normalized` | \\|Tr(U†_t U)\\|²/dim²  (process fidelity)     |
| `:real`       | Re[Tr(U†_t U)]/dim                              |
| `:average`    | (dim·F_normalized + 1)/(dim+1)  (Haar average) |
"""
const GATE_FIDELITY_TYPES = (:normalized, :real, :average, :modulus)

"""
    gate_fidelity(U, U_target; type::Symbol = :normalized) -> Float64

Gate fidelity between achieved propagator `U` and target `U_target`.

# Types
- `:normalized` — |Tr(U†_t U)|²/dim²
- `:real`       — Re[Tr(U†_t U)]/dim
- `:average`    — (dim·F_norm + 1)/(dim+1): average over Haar-random inputs
"""
function gate_fidelity(U::Matrix{ComplexF64}, U_target::Matrix{ComplexF64};
                       type::Symbol = :normalized)::Float64
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
- `type`   — fidelity type (must match the type used to compute F)

| `type`     | Gradient formula                             |
|:---------- |:-------------------------------------------- |
| `:real`    | dt_pwr · Im(inner)                           |
| `:square`  | 2 · dt_pwr · Im(z̄ · inner)                 |
| `:modulus` | dt_pwr · Im(inner / |z|)   (0 if |z| < ε)  |

Other types raise `ArgumentError` (density-matrix fidelities require
Liouville-space adjoint gradients, not implemented here).
"""
@inline function fidelity_grad_prefactor(z::ComplexF64,
                                          inner::ComplexF64,
                                          dt_pwr::Float64;
                                          type::Symbol)::Float64
    if type === :real
        return fidelity_grad_prefactor(z, inner, dt_pwr, REAL_OVERLAP)
    elseif type === :square
        return fidelity_grad_prefactor(z, inner, dt_pwr, SQUARED_OVERLAP)
    elseif type === :modulus
        return fidelity_grad_prefactor(z, inner, dt_pwr, MODULUS_OVERLAP)
    elseif type === :gate_modulus
        return fidelity_grad_prefactor(z, inner, dt_pwr, MODULUS_GATE)
    elseif type === :dm_real || type === :dm_linear
        return fidelity_grad_prefactor(z, inner, dt_pwr, LINEAR_DM)
    elseif type === :dm_square
        return fidelity_grad_prefactor(z, inner, dt_pwr, SQUARED_DM)
    elseif type === :dm_modulus
        return fidelity_grad_prefactor(z, inner, dt_pwr, MODULUS_DM)
    else
        throw(ArgumentError(
            "fidelity_grad_prefactor: ':$type' has no Hilbert-space adjoint " *
            "GRAPE formula. Supported: :real, :square, :modulus, :gate_modulus, " *
            ":dm_real, :dm_linear, :dm_square, :dm_modulus"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Ensemble Fidelities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ensemble_fidelity(ψ_finals, ψ_targs;
                      weights  = nothing,
                      mode     :: Symbol = :mean,
                      fid_type :: Symbol = :real) -> Float64

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
F = ensemble_fidelity(ψ_list, fill(ρtg, N); mode=:mean, fid_type=:real)
F = ensemble_fidelity(ψ_list, ρtg_list;    mode=:minimax)
```
"""
function ensemble_fidelity(ψ_finals::AbstractVector,
                            ψ_targs::AbstractVector;
                            weights  = nothing,
                            mode     :: Symbol = :mean,
                            fid_type :: Symbol = :real)::Float64
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
        sym = target.metric === :auto ? :normalized : target.metric
        return gate_fidelity(U_total, target.target_unitary; type = sym)
    elseif target.type == "state"
        target.target_state === nothing && throw(ArgumentError(
            "target.type is \"state\" but target.target_state is nothing"))
        sym = target.metric === :auto ? :square : target.metric
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
