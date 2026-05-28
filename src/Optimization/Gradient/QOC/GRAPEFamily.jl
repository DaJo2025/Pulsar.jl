# ============================================================================
# Gradient/QOC/GRAPEFamily.jl
# GRAPE-family optimizers for quantum optimal control
#
# Generic interface (dispatches on first arg type Function):
#   grape_optimize(f, grad!, θ0; ...)
#   grape_cg_optimize(f, grad!, θ0; ...)
#   grape_lbfgsb_optimize(f, grad!, θ0; ...)
#
# Intended use with grape_state_kernel from MR/GRAPEState.jl:
#   fid, g = grape_state_kernel(reshape(θ, n_ctrl, N_ts), ctrl)
#   f(θ)   = -fid
#   grad!(g_out, θ) = begin
#       fid, grad_mat = grape_state_kernel(reshape(θ, n_ctrl, N_ts), ctrl)
#       g_out .= -vec(grad_mat)
#   end
#
# NOTE: the existing grape_optimize(system, target; ...) in GRAPE.jl operates
#       on AbstractQuantumSystem — different first argument type, no dispatch conflict.
# ============================================================================

using LinearAlgebra

# Strong-Wolfe line search lives in Gradient/_LineSearch.jl.
# GRAPE-family callers use the simple bracket (α_max=5.0, max_iter=40,
# zoom_iter=30, zoom_eps=1e-13).

@inline function _gf_ls!(θ_t, g_buf, f, grad!, θ, d, g0, f0)
    wolfe_line_search!(θ_t, g_buf, f, grad!, θ, d, g0, f0;
                       α_max=5.0, max_iter=40,
                       zoom_iter=30, zoom_eps=1e-13,
                       two_point_bracket=false)
end

# ---------------------------------------------------------------------------
# L-BFGS two-loop (same as QuasiNewton, self-contained copy)
# ---------------------------------------------------------------------------

# Delegate to the canonical L-BFGS two-loop in Generic/QuasiNewton.jl.
@inline _gf_lbfgs_dir!(d, g, S, Y, ρ_list) =
    _lbfgs_direction!(d, g, S, Y, ρ_list, length(S))

# ---------------------------------------------------------------------------
# GRAPE (gradient ascent — standard form)
# ---------------------------------------------------------------------------

"""
    grape_optimize(f, grad!, θ0; lower, upper, step, max_iter, tol,
                   adaptive, verbose) → (θ_opt, f_opt, stats)

Standard GRAPE gradient ascent with adaptive step size.
`f` is the (negated) fidelity to minimise; `grad!` fills ∇f in-place.
Step size is scaled by gradient norm to give unit-norm steps, then adapted
by tracking improvement (halved on no progress, grown on consistent progress).

Dispatch note: this method matches `grape_optimize(f::Function, ...)` and does
NOT conflict with the existing `grape_optimize(system::AbstractQuantumSystem, ...)`
defined in `Optimization/GRAPE.jl`.
"""
function grape_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step    :: Float64 = 0.1,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    adaptive:: Bool    = true,
    verbose :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    t_start = time()
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)

    grad!(g, θ);  n_fid = 0;  n_grd = 1
    f_cur   = f(θ);  n_fid += 1
    θ_best  = copy(θ);  f_best = f_cur
    α       = step
    no_prog = 0
    converged = false
    n_iter  = 0
    fid_hist  = Float64[-f_cur]
    grad_hist = Float64[norm(g)]

    for iter in 1:max_iter
        n_iter = iter
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        # Normalised gradient direction
        d    = -g ./ max(gnorm, 1e-30)
        θ_new = clamp.(θ .+ α .* d, lb, ub)
        f_new = f(θ_new);  n_fid += 1
        grad_new = zeros(n)
        grad!(grad_new, θ_new);  n_grd += 1

        if f_new < f_cur
            θ  .= θ_new
            g  .= grad_new
            f_cur = f_new
            no_prog = 0
            if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end
            adaptive && (α = min(α * 1.05, step * 10.0))
        else
            no_prog += 1
            adaptive && (α = max(α * 0.5, step * 1e-4))
            no_prog > 20 && break
        end

        push!(fid_hist, -f_cur)
        push!(grad_hist, gnorm)
        verbose && iter % print_interval == 0 &&
            @printf("  grape iter %4d  F=%.6f  α=%.3e  |g|=%.3e\n",
                    iter, -f_cur, α, gnorm)
        isnothing(callback) || callback(iter, -f_cur; grad=gnorm, evals=n_fid+n_grd)
    end

    verbose &&
        @printf("  grape done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_fid + n_grd, converged)

    reason = converged ? "gradient norm < tol ($tol)" : "maximum iterations reached"
    return OptimizationResult(
        reshape(copy(θ_best), 1, n),
        -f_best,
        fid_hist,
        grad_hist,
        n_iter,
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grd,
        Dict{String,Any}("algorithm" => "GRAPE"),
    )
end

# ---------------------------------------------------------------------------
# GRAPE-CG (nonlinear CG direction instead of raw gradient)
# ---------------------------------------------------------------------------

"""
    grape_cg_optimize(f, grad!, θ0; lower, upper, max_iter, tol, cg_method,
                      verbose) → (θ_opt, f_opt, stats)

GRAPE with nonlinear conjugate gradient update (PR+ by default).
Uses strong Wolfe line search.  Better convergence than basic GRAPE on
smooth fidelity landscapes.
"""
function grape_cg_optimize(
    f         :: Function,
    grad!     :: Function,
    θ0        :: AbstractVector{<:Real};
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    max_iter  :: Int     = 500,
    tol       :: Float64 = 1e-6,
    cg_method :: Symbol  = :PR,
    verbose   :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    t_start = time()
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)
    g_old   = zeros(n)
    d       = zeros(n)
    θ_t     = similar(θ)                  # wolfe trial buffer (hoisted)
    g_ls    = zeros(n)                    # wolfe gradient scratch (hoisted)

    grad!(g, θ);  n_fid = 0;  n_grd = 1
    f_cur   = f(θ);  n_fid += 1
    @. d    = -g
    @. g_old= g
    θ_best  = copy(θ);  f_best = f_cur
    converged = false
    n_iter  = 0
    fid_hist  = Float64[-f_cur]
    grad_hist = Float64[norm(g)]

    for iter in 1:max_iter
        n_iter = iter
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        # Ensure descent
        dot(d, g) >= 0.0 && (@. d = -g)
        bounded && begin
            for i in 1:n
                if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
                   (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                    d[i] = 0.0
                end
            end
            norm(d) < 1e-14 && (@. d = -g)
        end

        α, f_new = _gf_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur)
        n_fid += 2

        θ .+= α .* d
        bounded && @. θ = clamp(θ, lb, ub)
        f_cur = f_new

        grad!(g, θ);  n_grd += 1
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        # CG β coefficient (auto-restart every n iters)
        restart = iter % n == 0 || dot(g_old, g_old) < 1e-30
        β       = restart ? 0.0 : _cg_beta(g, g_old, d, cg_method)
        @. d    = -g + β * d
        @. g_old = g

        push!(fid_hist, -f_cur)
        push!(grad_hist, gnorm)
        verbose && iter % print_interval == 0 &&
            @printf("  grape_cg(%s) iter %4d  F=%.6f  |g|=%.3e\n",
                    cg_method, iter, -f_cur, gnorm)
        isnothing(callback) || callback(iter, -f_cur; grad=gnorm, evals=n_fid+n_grd)
    end

    verbose &&
        @printf("  grape_cg done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_fid + n_grd, converged)

    reason = converged ? "gradient norm < tol ($tol)" : "maximum iterations reached"
    return OptimizationResult(
        reshape(copy(θ_best), 1, n),
        -f_best,
        fid_hist,
        grad_hist,
        n_iter,
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grd,
        Dict{String,Any}("algorithm" => "GRAPE-CG ($cg_method)"),
    )
end

# ---------------------------------------------------------------------------
# GRAPE-L-BFGS-B (box-constrained L-BFGS update)
# ---------------------------------------------------------------------------

"""
    grape_lbfgsb_optimize(f, grad!, θ0; lower, upper, memory, max_iter, tol,
                          verbose) → (θ_opt, f_opt, stats)

GRAPE with L-BFGS-B quasi-Newton update.  Provides superlinear convergence
for smooth quantum control problems.  Box constraints enforced by projected
direction and gradient clipping.

This is the recommended method for high-fidelity NMR pulse optimisation.
"""
function grape_lbfgsb_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    memory  :: Int     = 10,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    t_start = time()
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)

    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)
    g_new   = zeros(n)
    d       = zeros(n)
    θ_t     = similar(θ)                   # wolfe trial buffer (hoisted)
    g_ls    = zeros(n)                     # wolfe gradient scratch (hoisted)
    S       = Vector{Vector{Float64}}()
    Y       = Vector{Vector{Float64}}()
    ρ_list  = Float64[]

    grad!(g, θ);  n_fid = 0;  n_grd = 1
    f_cur   = f(θ);  n_fid += 1
    θ_best  = copy(θ);  f_best = f_cur
    converged = false
    n_iter  = 0
    fid_hist  = Float64[-f_cur]
    grad_hist = Float64[norm(g)]

    for iter in 1:max_iter
        n_iter = iter
        # Projected gradient norm for convergence
        pg_norm = norm(θ .- clamp.(θ .- g, lb, ub))
        pg_norm < tol && (converged = true; break)

        # L-BFGS direction
        _gf_lbfgs_dir!(d, g, S, Y, ρ_list)

        # Project direction onto feasible cone
        for i in 1:n
            if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
               (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                d[i] = 0.0
            end
        end
        norm(d) < 1e-14 && break

        α, f_new = _gf_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur)
        n_fid += 2

        s = α .* d              # freshly allocated; moved directly into S
        θ .+= s
        @. θ = clamp(θ, lb, ub)
        f_cur = f_new

        grad!(g_new, θ);  n_grd += 1
        y  = g_new .- g         # fresh; moved directly into Y below
        sy = dot(s, y)
        if sy > 1e-14 * dot(s, s)
            push!(S, s); push!(Y, y); push!(ρ_list, 1.0/sy)
            length(S) > memory && (popfirst!(S); popfirst!(Y); popfirst!(ρ_list))
        end
        @. g = g_new
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        push!(fid_hist, -f_cur)
        push!(grad_hist, pg_norm)
        verbose && iter % print_interval == 0 &&
            @printf("  grape_lbfgsb iter %4d  F=%.6f  |∇P|=%.3e  m=%d\n",
                    iter, -f_cur, pg_norm, length(S))
        isnothing(callback) || callback(iter, -f_cur; grad=pg_norm, evals=n_fid+n_grd)
    end

    verbose &&
        @printf("  grape_lbfgsb done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_fid + n_grd, converged)

    reason = converged ? "projected gradient norm < tol ($tol)" : "maximum iterations reached"
    return OptimizationResult(
        reshape(copy(θ_best), 1, n),
        -f_best,
        fid_hist,
        grad_hist,
        n_iter,
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grd,
        Dict{String,Any}("algorithm" => "GRAPE-L-BFGS-B", "lbfgs_memory" => memory),
    )
end

# ---------------------------------------------------------------------------
# SU(2) scalar GRAPE (phase-only, constant Rabi)
# ---------------------------------------------------------------------------

"""
    grape_su2_optimize(kernel::SU2Kernel, phi_init;
                       memory=20, max_iter=2000, tol=1e-12,
                       verbose=false, print_interval=200,
                       callback=nothing) → OptimizationResult

Phase-only L-BFGS-B GRAPE using the precomputed SU(2) scalar kernel.

`phi_init` is a `Vector{Float64}` of length `N_TS` (one phase per time step).
Returns an `OptimizationResult` where:
- `result.controls` is `[1 × N_TS]` — the optimised phase vector.
- `result.fidelity` is the best SU(2) ensemble fidelity achieved.
- Cartesian waveform: `w = vcat(cos.(phi)', sin.(phi)')`.

The per-call caching layer ensures each `(phi)` evaluation triggers the SU(2)
kernel at most once (shared by the `f` and `grad!` closures passed to L-BFGS-B).
"""
function grape_su2_optimize(
    kernel      :: SU2Kernel,
    phi_init    :: AbstractVector{Float64};
    memory      :: Int     = 20,
    max_iter    :: Int     = 2000,
    tol         :: Float64 = 1e-12,
    verbose     :: Bool    = false,
    print_interval :: Int  = 200,
    callback            = nothing,
)
    N_TS = length(phi_init)

    # Per-call cache: avoids evaluating the kernel twice for (f, grad!)
    phi_cache  = fill(NaN, N_TS)
    F_cache    = Ref(0.0)
    GJ_cache   = zeros(Float64, N_TS)

    function _update!(φ::AbstractVector{Float64})
        φ == phi_cache && return
        F, GJ      = su2_fidelity_and_grad(kernel, φ)
        F_cache[]  = F
        GJ_cache  .= GJ        # GJ = gradient of J = 1 − F  (minimize)
        phi_cache .= φ
    end

    f(φ::Vector{Float64})           = (_update!(φ); -F_cache[])  # minimize -F
    grad!(g::Vector{Float64}, φ::Vector{Float64}) =
        (_update!(φ); g .= GJ_cache; g)     # GJ = d(-F)/dφ  (already the cost gradient)

    lb = fill(-1e6, N_TS)
    ub = fill( 1e6, N_TS)

    return grape_lbfgsb_optimize(f, grad!, copy(phi_init);
                                 lower          = lb,
                                 upper          = ub,
                                 memory         = memory,
                                 max_iter       = max_iter,
                                 tol            = tol,
                                 verbose        = verbose,
                                 print_interval = print_interval,
                                 callback       = callback)
end

"""
    grape_su2_multistart(kernel::SU2Kernel, N_TS;
                         n_restarts=20, memory=20, max_iter=2000, tol=1e-12,
                         rng=Random.GLOBAL_RNG,
                         verbose=false, print_interval=200) → OptimizationResult

Run `grape_su2_optimize` from `n_restarts` random initial phase vectors (uniform
on [0, 2π]) and return the best result.  Restarts are parallelised with
`Threads.@threads`.
"""
function grape_su2_multistart(
    kernel      :: SU2Kernel,
    N_TS        :: Int;
    n_restarts  :: Int     = 20,
    memory      :: Int     = 20,
    max_iter    :: Int     = 2000,
    tol         :: Float64 = 1e-12,
    rng                    = Random.GLOBAL_RNG,
    verbose     :: Bool    = false,
    print_interval :: Int  = 200,
)
    seeds = rand(rng, UInt64, n_restarts)

    results = Vector{OptimizationResult}(undef, n_restarts)
    Threads.@threads for run in 1:n_restarts
        local_rng = Random.MersenneTwister(seeds[run])
        phi_0     = 2π .* rand(local_rng, N_TS)
        results[run] = grape_su2_optimize(kernel, phi_0;
                                          memory         = memory,
                                          max_iter       = max_iter,
                                          tol            = tol,
                                          verbose        = false,
                                          print_interval = print_interval + 1)
    end

    best_idx = argmax(r -> r.fidelity, results)
    if verbose
        @printf("  grape_su2_multistart: best F = %.8f (run %d / %d)\n",
                results[best_idx].fidelity, best_idx, n_restarts)
    end
    return results[best_idx]
end

# ─────────────────────────────────────────────────────────────────────────────
# Spin-I GRAPE (arbitrary single spin-I, phase-only)
# ─────────────────────────────────────────────────────────────────────────────

"""
    grape_spinI_optimize(kernel::SpinIKernel, phi_init;
                         memory=20, max_iter=2000, tol=1e-12,
                         verbose=false, print_interval=200,
                         callback=nothing) → OptimizationResult

L-BFGS-B optimizer for a `SpinIKernel` (arbitrary spin-I phase-only pulse).
Uses per-call caching so the kernel is evaluated only once per (f, grad!) pair.
"""
function grape_spinI_optimize(
    kernel         :: SpinIKernel,
    phi_init       :: AbstractVector{Float64};
    memory         :: Int     = 20,
    max_iter       :: Int     = 2000,
    tol            :: Float64 = 1e-12,
    verbose        :: Bool    = false,
    print_interval :: Int     = 200,
    callback                  = nothing,
)
    N_TS = length(phi_init)

    phi_cache = fill(NaN, N_TS)
    F_cache   = Ref(0.0)
    GJ_cache  = zeros(Float64, N_TS)

    function _update!(φ::AbstractVector{Float64})
        φ == phi_cache && return
        F, GJ     = spinI_fidelity_and_grad(kernel, φ)
        F_cache[] = F
        GJ_cache .= GJ
        phi_cache .= φ
    end

    f(φ::Vector{Float64})                        = (_update!(φ); -F_cache[])
    grad!(g::Vector{Float64}, φ::Vector{Float64}) = (_update!(φ); g .= GJ_cache; g)

    return grape_lbfgsb_optimize(f, grad!, copy(phi_init);
                                 lower          = fill(-1e6, N_TS),
                                 upper          = fill( 1e6, N_TS),
                                 memory         = memory,
                                 max_iter       = max_iter,
                                 tol            = tol,
                                 verbose        = verbose,
                                 print_interval = print_interval,
                                 callback       = callback)
end

"""
    grape_spinI_multistart(kernel::SpinIKernel, N_TS;
                           n_restarts=20, memory=20, max_iter=2000, tol=1e-12,
                           rng=Random.GLOBAL_RNG,
                           verbose=false, print_interval=200) → OptimizationResult

Run `grape_spinI_optimize` from `n_restarts` random phase vectors and return the best.
Restarts are parallelised with `Threads.@threads`.
"""
function grape_spinI_multistart(
    kernel         :: SpinIKernel,
    N_TS           :: Int;
    n_restarts     :: Int     = 20,
    memory         :: Int     = 20,
    max_iter       :: Int     = 2000,
    tol            :: Float64 = 1e-12,
    rng                       = Random.GLOBAL_RNG,
    verbose        :: Bool    = false,
    print_interval :: Int     = 200,
)
    seeds   = rand(rng, UInt64, n_restarts)
    results = Vector{OptimizationResult}(undef, n_restarts)
    Threads.@threads for run in 1:n_restarts
        local_rng    = Random.MersenneTwister(seeds[run])
        phi_0        = 2π .* rand(local_rng, N_TS)
        results[run] = grape_spinI_optimize(kernel, phi_0;
                                            memory         = memory,
                                            max_iter       = max_iter,
                                            tol            = tol,
                                            verbose        = false,
                                            print_interval = print_interval + 1)
    end

    best_idx = argmax(r -> r.fidelity, results)
    if verbose
        @printf("  grape_spinI_multistart: best F = %.8f (run %d / %d)\n",
                results[best_idx].fidelity, best_idx, n_restarts)
    end
    return results[best_idx]
end

# ─────────────────────────────────────────────────────────────────────────────
# Trotter GRAPE (N coupled spin-1/2, Strang-split, phase-only)
# ─────────────────────────────────────────────────────────────────────────────

"""
    grape_trotter_optimize(kernel::TrotterKernel, phi_init;
                           memory=20, max_iter=2000, tol=1e-12,
                           verbose=false, print_interval=200,
                           callback=nothing) → OptimizationResult

L-BFGS-B optimizer for a `TrotterKernel` (N coupled spin-1/2, Strang-split).
Uses per-call caching so the kernel is evaluated only once per (f, grad!) pair.
"""
function grape_trotter_optimize(
    kernel         :: TrotterKernel,
    phi_init       :: AbstractVector{Float64};
    memory         :: Int     = 20,
    max_iter       :: Int     = 2000,
    tol            :: Float64 = 1e-12,
    verbose        :: Bool    = false,
    print_interval :: Int     = 200,
    callback                  = nothing,
)
    N_TS = length(phi_init)

    phi_cache = fill(NaN, N_TS)
    F_cache   = Ref(0.0)
    GJ_cache  = zeros(Float64, N_TS)

    function _update!(φ::AbstractVector{Float64})
        φ == phi_cache && return
        F, GJ     = trotter_fidelity_and_grad(kernel, φ)
        F_cache[] = F
        GJ_cache .= GJ
        phi_cache .= φ
    end

    f(φ::Vector{Float64})                        = (_update!(φ); -F_cache[])
    grad!(g::Vector{Float64}, φ::Vector{Float64}) = (_update!(φ); g .= GJ_cache; g)

    return grape_lbfgsb_optimize(f, grad!, copy(phi_init);
                                 lower          = fill(-1e6, N_TS),
                                 upper          = fill( 1e6, N_TS),
                                 memory         = memory,
                                 max_iter       = max_iter,
                                 tol            = tol,
                                 verbose        = verbose,
                                 print_interval = print_interval,
                                 callback       = callback)
end

"""
    grape_trotter_multistart(kernel::TrotterKernel, N_TS;
                             n_restarts=20, memory=20, max_iter=2000, tol=1e-12,
                             rng=Random.GLOBAL_RNG,
                             verbose=false, print_interval=200) → OptimizationResult

Run `grape_trotter_optimize` from `n_restarts` random phase vectors and return the best.
Restarts are parallelised with `Threads.@threads`.
"""
function grape_trotter_multistart(
    kernel         :: TrotterKernel,
    N_TS           :: Int;
    n_restarts     :: Int     = 20,
    memory         :: Int     = 20,
    max_iter       :: Int     = 2000,
    tol            :: Float64 = 1e-12,
    rng                       = Random.GLOBAL_RNG,
    verbose        :: Bool    = false,
    print_interval :: Int     = 200,
)
    seeds   = rand(rng, UInt64, n_restarts)
    results = Vector{OptimizationResult}(undef, n_restarts)
    Threads.@threads for run in 1:n_restarts
        local_rng    = Random.MersenneTwister(seeds[run])
        phi_0        = 2π .* rand(local_rng, N_TS)
        results[run] = grape_trotter_optimize(kernel, phi_0;
                                              memory         = memory,
                                              max_iter       = max_iter,
                                              tol            = tol,
                                              verbose        = false,
                                              print_interval = print_interval + 1)
    end

    best_idx = argmax(r -> r.fidelity, results)
    if verbose
        @printf("  grape_trotter_multistart: best F = %.8f (run %d / %d)\n",
                results[best_idx].fidelity, best_idx, n_restarts)
    end
    return results[best_idx]
end
