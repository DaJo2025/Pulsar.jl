# Computation/ExactStepGradient.jl
#
# Exact derivative of one GRAPE step U = exp(−i H dt) along a control operator.
#
# First-order GRAPE replaces ∂U/∂u by −i·dt·Op·U (or U·Op), which is accurate
# only to O(dt·‖H‖): a few percent at typical NMR steps, enough to stall an
# optimizer short of the optimum. The exact derivative is the Duhamel integral
#
#     ∂U = −i·dt·pwr ∫₀¹ e^{−i(1−s)H dt} Op e^{−isH dt} ds,
#
# so the overlap a state kernel needs, inner = ⟨λ[n+1]| ∂U |ψ[n]⟩ / (−i·dt·pwr),
# is ∫₀¹ ⟨χ(s)|Op|ψ(s)⟩ ds with ψ(s) = e^{−isH dt}ψ[n], χ(s) = e^{+i(1−s)H dt}λ[n+1].
# It drops into fidelity_grad_prefactor unchanged (∂z = −i·dt·pwr·inner).
#
# Two evaluations of the same quantity:
#   • _exact_step_overlaps! — closed form (Daleckii–Krein) from the eigen-
#     decomposition H = V diag(E) V†:  inner = ⟨λ|V (Φ ∘ (V†·Op·V)) V†|ψ⟩,
#     Φ_ab = (e^{μ_a} − e^{μ_b})/(μ_a − μ_b), μ = −i·E·dt.  Used by the CPU kernels,
#     which already diagonalise every step.
#   • _step_quadrature — Gauss–Legendre nodes in s, sized from ‖H‖·dt so the rule
#     is exact to ~1e-15.  Used by the batched / GPU kernels, which then need only
#     propagators at the nodes (the existing device primitives), no eigenvectors.

using LinearAlgebra

# ─── Eigendecomposition of a step Hamiltonian ────────────────────────────────

"""
    _herm_eigen_into!(V, E, H) -> (V, E)

Eigendecomposition `H = V·diag(E)·V†` of a Hermitian `H`, written in place. 2×2
in closed form (no LAPACK call — the spin-½ kernels run this every step);
otherwise LAPACK, exactly as `_expm_neg_i_into!` does, so a propagator rebuilt
from (V, E) is bit-identical to it.
"""
function _herm_eigen_into!(V::Matrix{ComplexF64}, E::Vector{Float64}, H::AbstractMatrix)
    size(H, 1) == 2 && return _herm_eigen_2x2!(V, E, H)
    F = eigen(Hermitian(Matrix{ComplexF64}(H)))
    copyto!(V, F.vectors)
    copyto!(E, F.values)
    return V, E
end

function _herm_eigen_2x2!(V::Matrix{ComplexF64}, E::Vector{Float64}, H::AbstractMatrix)
    @inbounds begin
        h11 = real(H[1, 1]); h22 = real(H[2, 2]); off = ComplexF64(H[1, 2])
        a  = 0.5 * (h11 + h22)
        dz = 0.5 * (h11 - h22)
        m  = sqrt(dz^2 + abs2(off))
        E[1] = a - m; E[2] = a + m
        if m < 1e-300
            V[1, 1] = 1; V[2, 1] = 0; V[1, 2] = 0; V[2, 2] = 1
            return V, E
        end
        # eigenvector of a + m, from whichever row of (H − (a+m)I) is better conditioned
        if dz >= 0
            x = ComplexF64(m + dz); y = conj(off); nrm = sqrt(2m * (m + dz))
        else
            x = off; y = ComplexF64(m - dz); nrm = sqrt(2m * (m - dz))
        end
        x /= nrm; y /= nrm
        V[1, 2] = x;           V[2, 2] = y             # E[2] = a + m
        V[1, 1] = -conj(y);    V[2, 1] = conj(x)       # E[1] = a − m (orthogonal)
    end
    return V, E
end

"""
    _propagator_from_eigen!(P, V, E, dt, tmp) -> P

`P = V·diag(e^{−i E dt})·V†` — the same arithmetic as `_expm_neg_i_into!` for
D > 2, so the result is bit-identical to it.
"""
function _propagator_from_eigen!(P::Matrix{ComplexF64}, V::Matrix{ComplexF64},
                                 E::Vector{Float64}, dt::Float64, tmp::Matrix{ComplexF64})
    @inbounds for j in eachindex(E)
        c = cis(-E[j] * dt)
        @simd for i in axes(V, 1)
            tmp[i, j] = V[i, j] * c
        end
    end
    mul!(P, tmp, V')
    return P
end

# ─── Exact step overlaps (Daleckii–Krein) ────────────────────────────────────

# Φ_ab = (e^{μ_a} − e^{μ_b})/(μ_a − μ_b) with μ = −i E dt, written as
# e^{μ_a}·(e^{iθ} − 1)/(iθ), θ = −(E_b − E_a)·dt, and (e^{iθ} − 1)/(iθ) =
# sin θ/θ + i·2 sin²(θ/2)/θ — no cancellation at small θ (→ 1 as θ → 0).
@inline function _dk_weight(Ea::Float64, Eb::Float64, dt::Float64)::ComplexF64
    θ = -(Eb - Ea) * dt
    g = θ == 0 ? one(ComplexF64) :
        complex(sin(θ) / θ, 2 * sin(θ / 2)^2 / θ)
    return cis(-Ea * dt) * g
end

"""
    StepGradWorkspace(D)

Scratch for [`_exact_step_overlaps!`]: one per thread.
"""
struct StepGradWorkspace
    λt :: Vector{ComplexF64}
    ψt :: Vector{ComplexF64}
    C  :: Matrix{ComplexF64}
    Vc :: Matrix{ComplexF64}
    T  :: Matrix{ComplexF64}
    M  :: Matrix{ComplexF64}
end
StepGradWorkspace(D::Int) = StepGradWorkspace(
    Vector{ComplexF64}(undef, D), Vector{ComplexF64}(undef, D),
    Matrix{ComplexF64}(undef, D, D), Matrix{ComplexF64}(undef, D, D),
    Matrix{ComplexF64}(undef, D, D), Matrix{ComplexF64}(undef, D, D))

"""
    _exact_step_overlaps!(inner, V, E, dt, λ, ψ, ops, ws) -> inner

`inner[k] = ⟨λ| V (Φ ∘ (V†·ops[k]·V)) V† |ψ⟩` for the step `exp(−i H dt)`,
`H = V diag(E) V†` — the exact replacement of the first-order `⟨λ|ops[k]|ψ⟩`
(see the file header). `λ` is the co-state after the step, `ψ` the state
before it. Costs two D³ products per call, independent of the number of
controls: Σ_ab C_ab (V†·Op·V)_ab = Σ_ij Op_ij (conj(V)·C·Vᵀ)_ij with
C_ab = conj(λ̃_a)·Φ_ab·ψ̃_b.
"""
function _exact_step_overlaps!(inner::AbstractVector{ComplexF64}, V::Matrix{ComplexF64},
                               E::Vector{Float64}, dt::Float64,
                               λ::AbstractVector, ψ::AbstractVector,
                               ops::AbstractVector{<:AbstractMatrix}, ws::StepGradWorkspace)
    D = length(E)
    mul!(ws.λt, V', λ)
    mul!(ws.ψt, V', ψ)
    @inbounds for b in 1:D, a in 1:D
        ws.C[a, b] = conj(ws.λt[a]) * _dk_weight(E[a], E[b], dt) * ws.ψt[b]
    end
    @inbounds for j in 1:D, i in 1:D
        ws.Vc[i, j] = conj(V[i, j])
    end
    mul!(ws.T, ws.Vc, ws.C)
    mul!(ws.M, ws.T, transpose(V))
    @inbounds for k in eachindex(ops)
        Op = ops[k]
        s = zero(ComplexF64)
        for j in 1:D, i in 1:D
            s += Op[i, j] * ws.M[i, j]
        end
        inner[k] = s
    end
    return inner
end

# ─── Gauss–Legendre rule for the step integral (batched / GPU kernels) ────────

const _GL01_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
const _GL01_LOCK  = ReentrantLock()

# K-point Gauss–Legendre nodes and weights on [0, 1] (Golub–Welsch).
function _gauss_legendre01(K::Int)
    lock(_GL01_LOCK) do
        get!(_GL01_CACHE, K) do
            K == 1 && return ([0.5], [1.0])
            β = [k / sqrt(4k^2 - 1) for k in 1:K-1]
            F = eigen(SymTridiagonal(zeros(K), β))
            x = F.values
            w = 2 .* F.vectors[1, :] .^ 2
            p = sortperm(x)
            ((x[p] .+ 1) ./ 2, w[p] ./ 2)
        end
    end
end

# Error constant of K-point Gauss–Legendre on [0, 1] for e^{iωs}:
# (K!)⁴ / ((2K + 1)·((2K)!)³) · ω^{2K}  (log form, no overflow).
_gl_log_error(K::Int, ω::Float64) =
    4 * sum(log, 1:K; init = 0.0) - log(2K + 1) - 3 * sum(log, 1:2K; init = 0.0) +
    2K * log(max(ω, 1e-300))

"""
    _step_quadrature(ω; tol = 1e-15) -> (nodes, weights)

Nodes `s ∈ (0, 1)` and weights for ∫₀¹ f(s) ds where `f` is a sum of
oscillations e^{iωₐᵦ s} with |ωₐᵦ| ≤ `ω` (the spectral spread of `H` times
`dt`): the smallest Gauss–Legendre rule whose error bound is below `tol`,
split into equal sub-intervals once ω is large. The node set is symmetric
under s ↔ 1 − s, which the batched kernels use to pair propagators.
"""
function _step_quadrature(ω::Float64; tol::Float64 = 1e-15)
    # J equal sub-intervals each see frequency ω/J and carry weight 1/J, so the
    # composite error is bounded by the single-interval bound at ω/J.
    J = max(1, ceil(Int, ω / 6))
    ωJ = ω / J
    K = 1
    while K < 16 && _gl_log_error(K, ωJ) > log(tol)
        K += 1
    end
    x, w = _gauss_legendre01(K)
    nodes   = Float64[(i - 1 + xj) / J for i in 1:J for xj in x]
    weights = Float64[wj / J        for _ in 1:J for wj in w]
    return nodes, weights
end

"""
    _max_step_spread(drift_m, ops, coeff, dt_vec, n_member) -> ω

Upper bound on (spectral spread of H)·dt over a batch laid out like the
batched kernels (`col = (n−1)·n_member + m`): spread ≤ 2‖H‖₂ and
‖H‖₂ ≤ ‖drift_m‖₂ + Σ_k |coeff[k, col]|·‖ops[k]‖₂. Sizes [`_step_quadrature`].
"""
function _max_step_spread(drift_m, ops, coeff::AbstractMatrix, dt_vec::AbstractVector,
                          n_member::Int)
    dn = [opnorm(Matrix{ComplexF64}(drift_m[:, :, m])) for m in 1:size(drift_m, 3)]
    on = [opnorm(Matrix{ComplexF64}(o)) for o in ops]
    ω = 0.0
    @inbounds for col in axes(coeff, 2)
        h = dn[(col - 1) % n_member + 1]
        for k in eachindex(on)
            h += abs(coeff[k, col]) * on[k]
        end
        ω = max(ω, 2h * dt_vec[col])
    end
    return ω
end
