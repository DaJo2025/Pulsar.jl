# ============================================================================
# Metaheur/Swarm.jl — PSO (generic) and Differential Evolution
# ============================================================================

# ── Generic PSO ──────────────────────────────────────────────────────────────

"""
    pso_optimize(f, θ0; max_evals, max_iters, lower, upper, popsize, seed,
                 w, c1, c2, v_max, maximize) → (θ_best, f_best, stats)

Particle Swarm Optimization: inertia + cognitive + social velocity update.
Minimises `f`; set `maximize=true` to maximise.
"""
function pso_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 50_000,
    max_iters :: Int     = 1_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    popsize   :: Int     = 0,
    seed      :: Union{Nothing,Int} = nothing,
    w         :: Float64 = 0.7,
    c1        :: Float64 = 1.5,
    c2        :: Float64 = 1.5,
    v_max     :: Float64 = Inf,
    maximize  :: Bool    = false,
    callback = nothing,
)
    base_seed = seed === nothing ? rand(UInt32) : UInt32(seed)
    rng = MersenneTwister(base_seed)
    n   = length(θ0)
    lb  = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub  = upper === nothing ? fill( Inf, n) : Float64.(upper)
    np  = popsize > 0 ? popsize : max(20, 2 + n ÷ 2)
    obj = maximize ? (x -> -f(x)) : f
    nth = Threads.maxthreadid()
    rngs = [MersenneTwister(base_seed + UInt32(t)) for t in 0:nth-1]

    # Determine velocity range
    vrange = v_max < Inf ? v_max :
             (all(isfinite, lb) && all(isfinite, ub) ? maximum(ub .- lb) : 2.0)

    # Initialise particles (serial for reproducibility)
    pos = Vector{Vector{Float64}}(undef, np)
    vel = [zeros(n) for _ in 1:np]
    pos[1] = clamp.(Float64.(θ0), lb, ub)
    for i in 2:np
        if all(isfinite, lb) && all(isfinite, ub)
            pos[i] = lb .+ rand(rng, n) .* (ub .- lb)
        else
            pos[i] = clamp.(Float64.(θ0) .+ 0.1 * vrange .* randn(rng, n), lb, ub)
        end
    end

    # Parallel initial fitness evaluation
    fvals = Vector{Float64}(undef, np)
    @threadsif true for i in 1:np
        fvals[i] = obj(pos[i])
    end
    n_evals  = np
    p_best   = copy.(pos)
    p_best_f = copy(fvals)
    gi       = argmin(fvals)
    g_best   = copy(pos[gi])
    g_best_f = fvals[gi]
    history  = Float64[]

    iter = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1

        # Lesson 5: per-thread RNG → velocity/position update runs threaded.
        new_pos = Vector{Vector{Float64}}(undef, np)
        @threadsif true for i in 1:np
            rng_i = rngs[Threads.threadid()]
            r1 = rand(rng_i, n)
            r2 = rand(rng_i, n)
            vel[i] = w .* vel[i] .+
                     c1 .* r1 .* (p_best[i] .- pos[i]) .+
                     c2 .* r2 .* (g_best    .- pos[i])
            @inbounds for k in 1:n
                vel[i][k] = clamp(vel[i][k], -vrange, vrange)
            end
            new_pos[i] = clamp.(pos[i] .+ vel[i], lb, ub)
        end

        # Parallel fitness evaluation of all new positions
        new_fvals = Vector{Float64}(undef, np)
        @threadsif true for i in 1:np
            new_fvals[i] = obj(new_pos[i])
        end
        n_evals += np

        # Update personal and global bests (serial: g_best is shared)
        pos .= new_pos
        for i in 1:np
            fi = new_fvals[i]
            if fi < p_best_f[i]
                p_best[i]   = copy(pos[i])
                p_best_f[i] = fi
                if fi < g_best_f
                    g_best_f = fi
                    g_best   = copy(pos[i])
                end
            end
        end

        push!(history, g_best_f)
        isnothing(callback) || callback(iter, maximize ? -g_best_f : g_best_f; grad=nothing, evals=n_evals)
    end

    f_out = maximize ? -g_best_f : g_best_f
    stats = (evals=n_evals, iters=iter, converged=false, history=history)
    return g_best, f_out, stats
end

# ── Differential Evolution ────────────────────────────────────────────────────

"""
    de_optimize(f, θ0; max_evals, max_iters, lower, upper, popsize, seed,
                F, CR, strategy, maximize) → (θ_best, f_best, stats)

Differential Evolution DE/rand/1/bin. `strategy` is reserved for future variants.
Minimises `f`; set `maximize=true` to maximise.
"""
function de_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 50_000,
    max_iters :: Int     = 1_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    popsize   :: Int     = 0,
    seed      :: Union{Nothing,Int} = nothing,
    F         :: Float64 = 0.8,    # differential weight
    CR        :: Float64 = 0.9,    # crossover probability
    strategy  :: Symbol  = :rand1bin,
    maximize  :: Bool    = false,
    callback = nothing,
)
    base_seed = seed === nothing ? rand(UInt32) : UInt32(seed)
    rng = MersenneTwister(base_seed)
    n   = length(θ0)
    lb  = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub  = upper === nothing ? fill( Inf, n) : Float64.(upper)
    np  = popsize > 0 ? popsize : max(10 * n, 30)
    obj = maximize ? (x -> -f(x)) : f
    nth = Threads.maxthreadid()
    rngs = [MersenneTwister(base_seed + UInt32(t)) for t in 0:nth-1]

    # Initialise population (serial for reproducibility)
    pop = Vector{Vector{Float64}}(undef, np)
    pop[1] = clamp.(Float64.(θ0), lb, ub)
    for i in 2:np
        if all(isfinite, lb) && all(isfinite, ub)
            pop[i] = lb .+ rand(rng, n) .* (ub .- lb)
        else
            span = all(isfinite, lb) ? (ub .- lb) : fill(2.0, n)
            pop[i] = clamp.(Float64.(θ0) .+ (span ./ 6) .* randn(rng, n), lb, ub)
        end
    end

    # Parallel initial fitness evaluation
    fvals = Vector{Float64}(undef, np)
    @threadsif true for i in 1:np
        fvals[i] = obj(pop[i])
    end
    n_evals = np
    bi      = argmin(fvals)
    θ_best  = copy(pop[bi])
    f_best  = fvals[bi]
    history = Float64[]

    iter = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1

        # Lesson 5: per-thread RNG → trial-vector construction runs threaded.
        trials = Vector{Vector{Float64}}(undef, np)
        @threadsif true for i in 1:np
            rng_i   = rngs[Threads.threadid()]
            pool    = [k for k in 1:np if k != i]
            r1      = pool[rand(rng_i, 1:length(pool))]
            pool2   = [k for k in pool if k != r1]
            r2      = pool2[rand(rng_i, 1:length(pool2))]
            pool3   = [k for k in pool2 if k != r2]
            r3      = pool3[rand(rng_i, 1:length(pool3))]

            mutant = clamp.(pop[r1] .+ F .* (pop[r2] .- pop[r3]), lb, ub)
            j_rand = rand(rng_i, 1:n)
            trial  = copy(pop[i])
            for j in 1:n
                if j == j_rand || rand(rng_i) < CR
                    trial[j] = mutant[j]
                end
            end
            trials[i] = trial
        end

        # Parallel fitness evaluation of trial vectors
        trial_fvals = Vector{Float64}(undef, np)
        @threadsif true for i in 1:np
            trial_fvals[i] = obj(trials[i])
        end
        n_evals += np

        # Selection (serial: updates shared f_best)
        for i in 1:np
            if trial_fvals[i] < fvals[i]
                pop[i]   = trials[i]
                fvals[i] = trial_fvals[i]
                if trial_fvals[i] < f_best
                    f_best = trial_fvals[i]
                    θ_best = copy(trials[i])
                end
            end
        end

        push!(history, f_best)
        isnothing(callback) || callback(iter, maximize ? -f_best : f_best; grad=nothing, evals=n_evals)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=n_evals, iters=iter, converged=false, history=history)
    return θ_best, f_out, stats
end
