# ============================================================================
# Metaheur/GA.jl — Real-coded Genetic Algorithm
# ============================================================================

"""
    ga_optimize(f, θ0; max_evals, max_iters, lower, upper, popsize, seed,
                crossover_prob, mutation_prob, mutation_sigma, elite_frac,
                eta_c, maximize) → (θ_best, f_best, stats)

Real-coded GA: tournament selection, SBX crossover, Gaussian mutation, elitism.
Minimises `f` by default; set `maximize=true` to maximise.
"""
function ga_optimize(
    f          :: Function,
    θ0         :: AbstractVector{<:Real};
    max_evals  :: Int     = 50_000,
    max_iters  :: Int     = 1_000,
    lower      :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper      :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    popsize    :: Int     = 0,
    seed       :: Union{Nothing,Int} = nothing,
    crossover_prob :: Float64 = 0.9,
    mutation_prob  :: Float64 = 0.1,
    mutation_sigma :: Float64 = 0.1,
    elite_frac     :: Float64 = 0.1,
    eta_c          :: Float64 = 2.0,   # SBX distribution index
    maximize   :: Bool    = false,
    callback = nothing,
)
    base_seed = seed === nothing ? rand(UInt32) : UInt32(seed)
    rng  = MersenneTwister(base_seed)
    n    = length(θ0)
    lb   = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub   = upper === nothing ? fill( Inf, n) : Float64.(upper)
    np   = popsize > 0 ? popsize : max(20, 4 * n)
    obj  = maximize ? (x -> -f(x)) : f
    nth  = Threads.maxthreadid()
    # Per-thread RNGs derived from base seed for thread-safe reproducibility
    rngs = [MersenneTwister(base_seed + UInt32(t)) for t in 0:nth-1]

    # ── Initialize population ─────────────────────────────────────────────────
    pop = Vector{Vector{Float64}}(undef, np)
    pop[1] = clamp.(Float64.(θ0), lb, ub)
    for i in 2:np
        if all(isfinite, lb) && all(isfinite, ub)
            pop[i] = lb .+ rand(rng, n) .* (ub .- lb)
        else
            pop[i] = clamp.(Float64.(θ0) .+ mutation_sigma .* randn(rng, n), lb, ub)
        end
    end

    # Parallel initial fitness evaluation
    fvals   = Vector{Float64}(undef, np)
    @threadsif true for i in 1:np
        fvals[i] = obj(pop[i])
    end
    n_evals = np
    history = Float64[]
    best_i  = argmin(fvals)
    θ_best  = copy(pop[best_i])
    f_best  = fvals[best_i]

    n_elite = max(1, round(Int, elite_frac * np))

    converged = false
    iter = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1
        order = sortperm(fvals)         # ascending = best first (minimise)

        # ── Elitism: carry forward best n_elite ───────────────────────────────
        new_pop   = Vector{Vector{Float64}}(undef, np)
        new_fvals = Vector{Float64}(undef, np)
        for k in 1:n_elite
            new_pop[k]   = copy(pop[order[k]])
            new_fvals[k] = fvals[order[k]]
        end

        # Lesson 5: per-thread RNG → offspring generation runs threaded.
        # Reproducibility is preserved across runs at fixed `nthreads()`.
        n_offspring = np - n_elite
        offspring   = Vector{Vector{Float64}}(undef, n_offspring)
        @threadsif true for k in 1:n_offspring
            rng_k = rngs[Threads.threadid()]
            # Tournament (size 2)
            a, b = rand(rng_k, 1:np, 2)
            p1   = pop[fvals[a] <= fvals[b] ? a : b]
            a, b = rand(rng_k, 1:np, 2)
            p2   = pop[fvals[a] <= fvals[b] ? a : b]

            # SBX crossover
            child = copy(p1)
            if rand(rng_k) < crossover_prob
                for j in 1:n
                    if rand(rng_k) < 0.5
                        u = rand(rng_k)
                        β = u < 0.5 ? (2u)^(1/(eta_c+1)) : (1/(2*(1-u)))^(1/(eta_c+1))
                        child[j] = clamp(0.5*((1+β)*p1[j] + (1-β)*p2[j]), lb[j], ub[j])
                    end
                end
            end

            # Gaussian mutation
            for j in 1:n
                if rand(rng_k) < mutation_prob
                    scale = isfinite(ub[j]-lb[j]) ? (ub[j]-lb[j])*mutation_sigma : mutation_sigma
                    child[j] = clamp(child[j] + scale * randn(rng_k), lb[j], ub[j])
                end
            end
            offspring[k] = child
        end

        # ── Parallel fitness evaluation of offspring ──────────────────────────
        off_fvals = Vector{Float64}(undef, n_offspring)
        @threadsif true for k in 1:n_offspring
            off_fvals[k] = obj(offspring[k])
        end

        # Assemble new population
        for k in 1:n_offspring
            new_pop[n_elite + k]   = offspring[k]
            new_fvals[n_elite + k] = off_fvals[k]
        end
        n_evals += n_offspring

        pop   = new_pop
        fvals = new_fvals

        bi = argmin(fvals)
        if fvals[bi] < f_best
            f_best = fvals[bi]
            θ_best = copy(pop[bi])
        end
        push!(history, f_best)
        isnothing(callback) || callback(iter, maximize ? -f_best : f_best; grad=nothing, evals=n_evals)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=n_evals, iters=iter, converged=converged, history=history)
    return θ_best, f_out, stats
end
