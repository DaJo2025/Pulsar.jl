# ============================================================================
# Gradient/Generic/QuasiNewton.jl
# BFGS, L-BFGS, and L-BFGS-B (box-constrained) optimizers
#
# Interface: f(θ) -> Real,  grad!(g, θ)  [in-place gradient]
# Returns:   (θ_opt, f_opt, stats)
# ============================================================================

using LinearAlgebra
# `_converged_mode` defined in Generic/FirstOrder.jl (loaded earlier).

# Strong-Wolfe line search lives in Gradient/_LineSearch.jl.
# QN callers use the Nocedal two-point bracket (α_max=10.0, max_iter=60,
# zoom_iter=40, zoom_eps=1e-14) — see `_qn_ls!` below.

# ---------------------------------------------------------------------------
# LBFGSB.jl extension hook
# ---------------------------------------------------------------------------
#
# When the optional `LBFGSB` package is loaded, the extension
# `ext/PulsarLBFGSBExt.jl` flips `_LBFGSB_LOADED[]` and provides a method for
# `_ext_lbfgsb_optimize`. The default body below is unreachable but kept as a
# defensive guard.

const _LBFGSB_LOADED = Ref(false)

function _ext_lbfgsb_optimize(::Any, ::Any, ::AbstractVector{<:Real}; kwargs...)
    error("PulsarLBFGSBExt is not loaded — `import LBFGSB` first or call " *
          "`lbfgsb_optimize(...; use_native=true)`.")
end

@inline function _qn_ls!(θ_t, g_buf, f, grad!, θ, d, g0, f0;
                          c1::Float64=1e-4, c2::Float64=0.9)
    wolfe_line_search!(θ_t, g_buf, f, grad!, θ, d, g0, f0;
                       c1=c1, c2=c2,
                       α_max=10.0, max_iter=60,
                       zoom_iter=40, zoom_eps=1e-14,
                       two_point_bracket=true)
end

# ---------------------------------------------------------------------------
# BFGS
# ---------------------------------------------------------------------------

"""
    bfgs_optimize(f, grad!, θ0; max_iter, tol, lower, upper, verbose)
        → (θ_opt, f_opt, stats)

BFGS with inverse Hessian approximation and strong Wolfe line search.
Uses rank-2 update: H ← (I-ρsy')H(I-ρys') + ρss'.
Suitable for problems with n ≲ 500.
"""
function bfgs_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    check_invariants :: Bool = false,
    convergence_mode :: Symbol = :gradient_norm,
    f_tol   :: Float64 = 1e-8,
    wolfe_c1 :: Float64 = 1e-4,
    wolfe_c2 :: Float64 = 0.9,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    g    = zeros(n)
    g_new= zeros(n)
    H    = Matrix{Float64}(I, n, n)   # inverse Hessian approximation
    θ_t  = similar(θ)                 # wolfe trial buffer (hoisted)
    g_ls = zeros(n)                   # wolfe gradient scratch (hoisted)

    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    f_prev  = f_cur
    θ_best  = copy(θ)
    f_best  = f_cur
    converged = false

    for iter in 1:max_iter
        gnorm = norm(g)
        g_ok  = gnorm < tol
        f_ok  = iter > 1 && abs(f_cur - f_prev) < f_tol
        _converged_mode(convergence_mode, g_ok, f_ok) && (converged = true; break)

        d = -(H * g)         # descent direction
        bounded && begin
            # Project direction: don't move away from active bound
            for i in 1:n
                if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
                   (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                    d[i] = 0.0
                end
            end
        end

        norm(d) < 1e-14 && break

        α, f_new = _qn_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur;
                           c1=wolfe_c1, c2=wolfe_c2)
        n_evals += 2

        s = α .* d
        θ .+= s
        bounded && @. θ = clamp(θ, lb, ub)
        f_prev = f_cur
        f_cur  = f_new

        grad!(g_new, θ);  n_evals += 1
        y = g_new .- g
        sy = dot(s, y)

        if sy > 1e-14 * dot(s, s)
            if check_invariants
                ok, msg = check_bfgs_curvature(s, y)
                _assert_invariant(ok, msg, :bfgs_curvature,
                                  (; iter=iter, sy=sy, ss=dot(s, s)))
            end
            ρ = 1.0 / sy
            # BFGS rank-2 update (matrix form for small n)
            A = I - ρ * (s * y')
            H = A * H * A' + ρ * (s * s')
        else
            # Reset on curvature failure
            H = Matrix{Float64}(I, n, n)
        end

        @. g = g_new
        if f_cur < f_best
            f_best = f_cur
            θ_best .= θ
        end

        verbose && iter % 50 == 0 &&
            @printf("  bfgs iter %4d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# L-BFGS two-loop recursion
# ---------------------------------------------------------------------------

function _lbfgs_direction!(d, g, S, Y, ρ_list, m_used)
    # d ← -H * g using two-loop recursion (Nocedal & Wright Alg. 7.4)
    n     = length(g)
    q     = copy(g)
    α_arr = zeros(m_used)

    for i in m_used:-1:1
        α_arr[i] = ρ_list[i] * dot(S[i], q)
        @. q -= α_arr[i] * Y[i]
    end

    # Initial Hessian scaling H₀ = (s_{k-1}'y_{k-1})/(y_{k-1}'y_{k-1}) * I
    γ = m_used > 0 ? (dot(S[m_used], Y[m_used]) /
                      max(dot(Y[m_used], Y[m_used]), 1e-30)) : 1.0
    @. d = γ * q

    for i in 1:m_used
        β = ρ_list[i] * dot(Y[i], d)
        @. d += (α_arr[i] - β) * S[i]
    end
    @. d = -d   # ascent → descent
end

# ---------------------------------------------------------------------------
# L-BFGS
# ---------------------------------------------------------------------------

"""
    lbfgs_optimize(f, grad!, θ0; memory, max_iter, tol, lower, upper, verbose)
        → (θ_opt, f_opt, stats)

Limited-memory BFGS using Nocedal's two-loop recursion.
`memory` is the number of (s,y) pairs retained (default 10).
"""
function lbfgs_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    memory  :: Int     = 10,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    check_invariants :: Bool = false,
    convergence_mode :: Symbol = :gradient_norm,
    f_tol   :: Float64 = 1e-8,
    wolfe_c1 :: Float64 = 1e-4,
    wolfe_c2 :: Float64 = 0.9,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    g      = zeros(n)
    g_new  = zeros(n)
    d      = zeros(n)
    θ_t    = similar(θ)                   # wolfe trial buffer (hoisted)
    g_ls   = zeros(n)                     # wolfe gradient scratch (hoisted)
    S      = Vector{Vector{Float64}}()   # stored s vectors
    Y      = Vector{Vector{Float64}}()   # stored y vectors
    ρ_list = Float64[]

    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    f_prev  = f_cur
    θ_best  = copy(θ)
    f_best  = f_cur
    converged = false

    for iter in 1:max_iter
        gnorm = norm(g)
        g_ok  = gnorm < tol
        f_ok  = iter > 1 && abs(f_cur - f_prev) < f_tol
        _converged_mode(convergence_mode, g_ok, f_ok) && (converged = true; break)

        m_used = length(S)
        _lbfgs_direction!(d, g, S, Y, ρ_list, m_used)

        bounded && begin
            for i in 1:n
                if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
                   (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                    d[i] = 0.0
                end
            end
        end
        norm(d) < 1e-14 && break

        α, f_new = _qn_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur;
                           c1=wolfe_c1, c2=wolfe_c2)
        n_evals += 2

        s = α .* d
        θ .+= s
        bounded && @. θ = clamp(θ, lb, ub)
        f_prev = f_cur
        f_cur  = f_new

        grad!(g_new, θ);  n_evals += 1
        y  = g_new .- g
        sy = dot(s, y)

        if sy > 1e-14 * dot(s, s)
            ρ_k = 1.0 / sy
            if check_invariants
                ok1, msg1 = check_bfgs_curvature(s, y)
                _assert_invariant(ok1, msg1, :bfgs_curvature,
                                  (; iter=iter, sy=sy, ss=dot(s, s)))
                ok2, msg2 = check_lbfgs_pair_positive(ρ_k; k=length(S)+1)
                _assert_invariant(ok2, msg2, :lbfgs_pair_positive,
                                  (; iter=iter, k=length(S)+1, ρ=ρ_k))
            end
            push!(S, copy(s))
            push!(Y, copy(y))
            push!(ρ_list, ρ_k)
            if length(S) > memory
                popfirst!(S); popfirst!(Y); popfirst!(ρ_list)
            end
        end

        @. g = g_new
        if f_cur < f_best
            f_best = f_cur
            θ_best .= θ
        end

        verbose && iter % 100 == 0 &&
            @printf("  lbfgs iter %4d  f=%.6e  |g|=%.3e  m=%d\n",
                    iter, f_cur, gnorm, length(S))
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# L-BFGS-B (box-constrained)
# ---------------------------------------------------------------------------

"""
    lbfgsb_optimize(f, grad!, θ0; lower, upper, memory, max_iter, tol, verbose)
        → (θ_opt, f_opt, stats)

L-BFGS-B: box-constrained L-BFGS using the projected-gradient active-set
strategy (Zhu et al. 1997).  Active bound constraints are identified by the
Cauchy point; a subspace L-BFGS direction is computed for free variables.
For simplicity this implementation uses projected gradient clipping on the
L-BFGS direction, which is equivalent to the L-BFGS-B active-set strategy
for moderate-dimension problems.
"""
function lbfgsb_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    memory  :: Int     = 10,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = false,
    check_invariants :: Bool = false,
    convergence_mode :: Symbol = :gradient_norm,
    f_tol   :: Float64 = 1e-8,
    wolfe_c1 :: Float64 = 1e-4,
    wolfe_c2 :: Float64 = 0.9,
    callback = nothing,
    use_native :: Bool = false,
)
    # If the LBFGSB.jl extension is loaded, delegate to the reference Fortran
    # kernel by default. The native Julia path remains available via
    # `use_native=true` (e.g. for offline installs or fallback testing).
    if !use_native && _LBFGSB_LOADED[]
        return _ext_lbfgsb_optimize(f, grad!, θ0;
            lower=lower, upper=upper, memory=memory, max_iter=max_iter,
            tol=tol, f_tol=f_tol, convergence_mode=convergence_mode,
            verbose=verbose, callback=callback)
    end
    n  = length(θ0)
    lb = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub = upper === nothing ? fill( Inf, n) : Float64.(upper)

    θ      = clamp.(float.(θ0), lb, ub)
    g      = zeros(n)
    g_new  = zeros(n)
    g_proj = zeros(n)
    d      = zeros(n)
    θ_t    = similar(θ)                  # wolfe trial buffer (hoisted)
    g_ls   = zeros(n)                    # wolfe gradient scratch (hoisted)
    S      = Vector{Vector{Float64}}()
    Y      = Vector{Vector{Float64}}()
    ρ_list = Float64[]

    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    f_prev  = f_cur
    θ_best  = copy(θ)
    f_best  = f_cur
    converged = false

    for iter in 1:max_iter
        # Projected gradient for convergence check
        @. g_proj = θ - clamp(θ - g, lb, ub)   # projected grad ≈ θ - P[θ-g]
        pgnorm = norm(g_proj)
        g_ok   = pgnorm < tol
        f_ok   = iter > 1 && abs(f_cur - f_prev) < f_tol
        _converged_mode(convergence_mode, g_ok, f_ok) && (converged = true; break)

        # L-BFGS direction
        m_used = length(S)
        _lbfgs_direction!(d, g, S, Y, ρ_list, m_used)
        if !all(isfinite, d)                 # degenerate curvature model
            empty!(S); empty!(Y); empty!(ρ_list)
            @. d = -g
        end

        # Project direction: free only on inactive constraints
        for i in 1:n
            if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
               (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                d[i] = 0.0
            end
        end
        norm(d) < 1e-14 && break

        # After projection the L-BFGS direction may no longer be a descent
        # direction for the objective (dot(d, g) ≥ 0).  Reset to the projected
        # steepest-descent direction and clear the L-BFGS history.
        if dot(d, g) >= -1e-14 * max(dot(g, g), 1e-30)
            empty!(S); empty!(Y); empty!(ρ_list)
            @. d = -g
            for i in 1:n
                if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
                   (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                    d[i] = 0.0
                end
            end
            norm(d) < 1e-14 && break
        end

        α, f_new = _qn_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur;
                           c1=wolfe_c1, c2=wolfe_c2)
        n_evals += 2

        # A line search that cannot lower f, or whose step is at the rounding
        # level of θ, carries no curvature information. Near an optimum where the
        # supplied gradient is approximate (first-order GRAPE) the step shrinks to
        # ~1e-14 and (s, y) is rounding noise; accepting such pairs used to grow
        # the L-BFGS direction without bound (‖d‖ ~ 1e3 … Inf, which overflowed a
        # propagator) while the loop ran on for every remaining iteration. Drop
        # the curvature model and retry along −g; if that fails too, stop.
        if !(f_new < f_cur) || α * norm(d, Inf) <= 1e-12 * max(1.0, norm(θ, Inf))
            isempty(S) && break
            empty!(S); empty!(Y); empty!(ρ_list)
            continue
        end

        s = α .* d
        θ .+= s
        @. θ = clamp(θ, lb, ub)
        f_prev = f_cur
        f_cur  = f_new

        grad!(g_new, θ);  n_evals += 1
        y  = g_new .- g
        sy = dot(s, y)

        if sy > 1e-14 * dot(s, s)
            ρ_k = 1.0 / sy
            if check_invariants
                ok1, msg1 = check_bfgs_curvature(s, y)
                _assert_invariant(ok1, msg1, :bfgs_curvature,
                                  (; iter=iter, sy=sy, ss=dot(s, s)))
                ok2, msg2 = check_lbfgs_pair_positive(ρ_k; k=length(S)+1)
                _assert_invariant(ok2, msg2, :lbfgs_pair_positive,
                                  (; iter=iter, k=length(S)+1, ρ=ρ_k))
            end
            push!(S, copy(s))
            push!(Y, copy(y))
            push!(ρ_list, ρ_k)
            if length(S) > memory
                popfirst!(S); popfirst!(Y); popfirst!(ρ_list)
            end
        end

        @. g = g_new
        if f_cur < f_best
            f_best = f_cur
            θ_best .= θ
        end

        verbose && iter % 100 == 0 &&
            @printf("  lbfgsb iter %4d  f=%.6e  |∇P|=%.3e  m=%d\n",
                    iter, f_cur, pgnorm, length(S))
        isnothing(callback) || callback(iter, f_cur; grad=pgnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end
