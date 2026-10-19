# Computation/FidelityReductions.jl
#
# One implementation of the fidelity metrics, shared by every kernel at every
# layer (the SU(2) / Spin-I / Trotter kernels live in this layer and cannot call
# the Physics-layer metric types, so the arithmetic lives here and Physics
# delegates to it).
#
# Every metric reduces a complex overlap z at its own level to a real number:
#
#     z_P = Tr(U_t† U)/dim     z_S = ⟨ψ_t|ψ⟩     z_D = Tr(ρ_t† ρ)
#
#     reduction   F = f(z)                  chain weight w(z)
#     real        Re z                      1
#     modulus     |z|                       z̄/|z|          (0 at z = 0)
#     squared     |z|²                      2z̄
#     average     (dim·|z|² + 1)/(dim + 1)  2·dim·z̄/(dim + 1)   (:prop_average)
#     real_quad   1 − (1 − Re z)²           2(1 − Re z)         (:prop_real_quad)
#
# The chain weight is defined by dF = Re(w · dz), so a kernel only needs dz
# from its own adjoint: dF/du = Re(w · ∂z/∂u). For an overlap ⟨target|·⟩ the
# co-state is seeded with conj(w) · target. The weights are the ones tabulated
# in the fidelity-comparisons paper (classes A, B, B′, B″).

"""
    overlap_fidelity(metric, z; dim = 0) -> Float64

Value of fidelity `metric` (a Φ^X_Y name, see [`FIDELITY_METRICS`](@ref)) for a
complex overlap `z` at the metric's own level. `dim` is needed only for
`:prop_average`. See also [`overlap_weight`](@ref).
"""
function overlap_fidelity(metric::Symbol, z::Number; dim::Integer = 0)::Float64
    r = _metric_reduction(metric)
    r === :real      && return real(z)
    r === :squared   && return abs2(z)
    r === :modulus   && return abs(z)
    r === :real_quad && return 1.0 - (1.0 - real(z))^2
    r === :average   && return (_average_dim(metric, dim) * abs2(z) + 1) / (dim + 1)
    throw(ArgumentError("overlap_fidelity: no overlap form for :$metric"))
end

"""
    overlap_weight(metric, z; dim = 0) -> ComplexF64

Chain weight `w` of fidelity `metric` at overlap `z`, defined by
`dF = Re(w · dz)`: `1` for the real reductions, `2z̄` for squared, `z̄/|z|`
for modulus (zero at `z = 0`). A kernel turns its overlap derivative into a
fidelity gradient as `Re(w · ∂z/∂u)`; an adjoint for an overlap `⟨target|·⟩`
seeds its co-state with `conj(w) · target`.
"""
function overlap_weight(metric::Symbol, z::Number; dim::Integer = 0)::ComplexF64
    r = _metric_reduction(metric)
    r === :real      && return one(ComplexF64)
    r === :squared   && return 2 * conj(ComplexF64(z))
    if r === :modulus
        a = abs(z)
        return a < 1e-14 ? zero(ComplexF64) : conj(ComplexF64(z)) / a
    end
    r === :real_quad && return ComplexF64(2 * (1 - real(z)))
    r === :average   && return 2 * _average_dim(metric, dim) * conj(ComplexF64(z)) / (dim + 1)
    throw(ArgumentError("overlap_weight: no overlap form for :$metric"))
end

"""
    has_constant_weight(metric) -> Bool

`true` for the real-part metrics, whose chain weight is the constant 1 — their
gradient needs no per-member overlap, so kernels may precompute the co-state.
"""
has_constant_weight(metric::Symbol)::Bool = _metric_reduction(metric) === :real

# Density-matrix metrics evaluated from a PURE-state overlap z_S: then
# z_D = Tr(ρ_t ρ) = |z_S|² (real, ≥ 0) and dz_D = Re(2·z̄_S · dz_S), so the
# weight on z_S is w_D(z_D) · 2z̄_S (w_D is real for real z_D ≥ 0). State
# metrics pass through unchanged. Used by kernels that propagate kets but are
# asked for a DM-level figure of merit.
"""
    pure_state_fidelity(metric, z_S; dim = 0) -> Float64

Like [`overlap_fidelity`](@ref) for a state-vector kernel: `:state_*` metrics use
`z_S` directly, `:dm_*` metrics use the pure-state density-matrix overlap
`z_D = |z_S|²`.
"""
function pure_state_fidelity(metric::Symbol, zS::Number; dim::Integer = 0)::Float64
    _metric_level(metric) === :dm || return overlap_fidelity(metric, zS; dim)
    return overlap_fidelity(metric, abs2(zS); dim)
end

"""
    pure_state_weight(metric, z_S; dim = 0) -> ComplexF64

Chain weight with respect to the state overlap `z_S` (see
[`pure_state_fidelity`](@ref)): `dF = Re(w · dz_S)`.
"""
function pure_state_weight(metric::Symbol, zS::Number; dim::Integer = 0)::ComplexF64
    _metric_level(metric) === :dm || return overlap_weight(metric, zS; dim)
    zD = abs2(zS)
    return real(overlap_weight(metric, zD; dim)) * 2 * conj(ComplexF64(zS))
end

# Symbol comparisons only — this runs once per (step, control) inside gradient
# loops, so it must not allocate (building a String here cost the low-memory
# adjoint its memory advantage).
function _metric_reduction(metric::Symbol)::Symbol
    (metric === :prop_real    || metric === :state_real    || metric === :dm_real)    && return :real
    (metric === :prop_squared || metric === :state_squared || metric === :dm_squared) && return :squared
    (metric === :prop_modulus || metric === :state_modulus || metric === :dm_modulus) && return :modulus
    metric === :prop_average   && return :average
    metric === :prop_real_quad && return :real_quad
    throw(ArgumentError("Fidelity metric :$metric has no overlap reduction " *
                        "(supported: $(_format_metrics(FIDELITY_METRICS)) except :dm_uhlmann)."))
end

function _average_dim(metric::Symbol, dim::Integer)
    dim > 0 || throw(ArgumentError(":$metric needs the Hilbert-space dimension (dim > 0)."))
    return dim
end
