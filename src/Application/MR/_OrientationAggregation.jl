# ============================================================================
# Application/MR/_OrientationAggregation.jl
# Backward-compatible shims for MR domain optimizers.  The canonical powder
# reduction now lives in Physics/PowderAveraging.jl (`powder_average_scalar`,
# `powder_average_matrix`, `orient_fidelity`, `orient_gradient`); the names
# below are preserved so existing EPR/DNP call sites keep working while
# delegating to the single shared implementation.
#
# Used by:
#   - EPROptControl.jl  (g-tensor orientation rotation, band-selective or raw)
#   - DNPOptControl.jl  (electron-nuclear orientation rotation)
# ============================================================================

# Weighted per-orientation gradient sum — delegates to the canonical
# `powder_average_matrix` (Physics/PowderAveraging.jl).
_aggregate_orientation_gradient(ctrl_shape::Tuple{Int,Int}, orientations,
                                rotate_fn::Function, gradient_fn::Function) =
    powder_average_matrix((α, β, γ) -> gradient_fn(rotate_fn(α, β, γ)),
                          orientations, ctrl_shape)

# Weighted per-orientation fidelity mean — delegates to `powder_average_scalar`.
_aggregate_orientation_fidelity(orientations, rotate_fn::Function,
                                fidelity_fn::Function) =
    powder_average_scalar((α, β, γ) -> fidelity_fn(rotate_fn(α, β, γ)), orientations)

# `(s,c,t)`-closure wrappers — thin aliases of the canonical `orient_*` helpers.
_wrap_orient_gradient(base_grad::Function,
                      orientations::Union{Nothing,Vector{NTuple{4,Float64}}},
                      rotate_fn::Function) =
    orient_gradient(base_grad, orientations, rotate_fn)

_wrap_orient_fidelity(base_fid::Function,
                      orientations::Union{Nothing,Vector{NTuple{4,Float64}}},
                      rotate_fn::Function) =
    orient_fidelity(base_fid, orientations, rotate_fn)
