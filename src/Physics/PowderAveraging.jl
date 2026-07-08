# Physics/PowderAveraging.jl
# ============================================================================
# Unified crystallite/powder averaging — ONE reduction used by every solid-state
# domain (MAS SSNMR, EPR, DNP, …).  Previously each domain hand-rolled its own
# orientation loop (MAS in compute_grape_gradient_powder, EPR/DNP in
# _OrientationAggregation.jl); this module is the single home for the weighted
# sum over a `powder_grid` and the thin `orient_*` wrappers that turn a
# single-orientation kernel into a powder-averaged one.
#
# An orientation grid is a `Vector{NTuple{4,Float64}}` of `(α, β, γ, weight)`;
# `powder_grid(...)` returns weights summing to 1.
# ============================================================================
using LinearAlgebra

"""
    powder_average_scalar(kernel, orientations) -> Float64

Weighted powder mean `Σ_Ω w_Ω · kernel(α,β,γ) / Σ_Ω w_Ω` of a scalar observable
(e.g. fidelity).  `kernel(α,β,γ)::Real` is evaluated once per crystallite; the
loop is threaded.  Normalised by the weight sum so the result is a true mean
even if a grid's weights do not sum to exactly 1.
"""
function powder_average_scalar(kernel, orientations::Vector{NTuple{4,Float64}})::Float64
    n = length(orientations)
    F = Vector{Float64}(undef, n)
    W = Vector{Float64}(undef, n)
    Threads.@threads :static for o in 1:n
        α, β, γ, w = orientations[o]
        F[o] = kernel(α, β, γ)
        W[o] = w
    end
    wsum = sum(W)
    return wsum < 1e-15 ? 0.0 : sum(W .* F) / wsum
end

"""
    powder_average_matrix(kernel, orientations, shape) -> Matrix{Float64}

Weighted powder sum `Σ_Ω w_Ω · kernel(α,β,γ)` of a matrix observable (e.g. a
GRAPE gradient of `shape = (n_controls, n_timesteps)`).  Threaded with a lock on
the accumulator.  NOT weight-normalised: it pairs with the objective's own
convention (grids sum to 1, so the gradient of `powder_average_scalar` is exactly
this sum).  `kernel(α,β,γ)::Matrix` must return `shape`.
"""
function powder_average_matrix(kernel, orientations::Vector{NTuple{4,Float64}},
                               shape::Tuple{Int,Int})::Matrix{Float64}
    G  = zeros(Float64, shape)
    lk = ReentrantLock()
    Threads.@threads :static for o in 1:length(orientations)
        α, β, γ, w = orientations[o]
        Go = kernel(α, β, γ)
        lock(lk) do
            G .+= w .* Go
        end
    end
    return G
end

"""
    orient_fidelity(base_fid, orientations, rotate_fn) -> Function

Wrap a single-orientation fidelity closure `base_fid(sys_Ω, ctrl, target)` into a
powder-averaged `(s, c, t) -> Float64`.  Identity passthrough when
`orientations === nothing`.  `rotate_fn(α,β,γ)` returns the oriented system.
"""
function orient_fidelity(base_fid::Function,
                         orientations::Union{Nothing,Vector{NTuple{4,Float64}}},
                         rotate_fn::Function)::Function
    orientations === nothing && return base_fid
    return (s, c, t) -> powder_average_scalar(
        (α, β, γ) -> base_fid(rotate_fn(α, β, γ), c, t), orientations)
end

"""
    orient_gradient(base_grad, orientations, rotate_fn) -> Function

Wrap a single-orientation gradient closure `base_grad(sys_Ω, ctrl, target)` into a
powder-averaged `(s, c, t) -> Matrix`.  Identity passthrough when
`orientations === nothing`.
"""
function orient_gradient(base_grad::Function,
                         orientations::Union{Nothing,Vector{NTuple{4,Float64}}},
                         rotate_fn::Function)::Function
    orientations === nothing && return base_grad
    return (s, c, t) -> powder_average_matrix(
        (α, β, γ) -> base_grad(rotate_fn(α, β, γ), c, t),
        orientations, size(c.controls))
end
