# ============================================================================
# Utilities/EnsembleMap.jl
# Device-dispatched ensemble map for broadband objective functions.
#
# Replaces explicit `Threads.@threads for i in eachindex(drifts)` loops with
# a single call that automatically uses the active device set by set_device!.
#
# Usage:
#   F_parts = ensemble_map(eachindex(drifts)) do i
#       H0 = drifts[i];  ψ = copy(ρ0)
#       for k in 1:N_ts
#           ψ = compute_propagator(H0 + pwr*(w[1,k]*Lx + w[2,k]*Ly), dt) * ψ
#       end
#       state_fidelity(ρtg, ψ; type=:real)
#   end
#   return -sum(F_parts) / length(F_parts)
# ============================================================================

# ============================================================================
# ScratchBuffer pool
# ============================================================================

"""
    ScratchBufferPool(n_params::Int; pool_size::Int=Threads.nthreads())

A `Channel`-based pool of pre-allocated `Vector{Float64}` gradient buffers.

Each buffer is length `n_params`.  Workers borrow a buffer from the pool before
computing a gradient, write into it, accumulate into a shared result, then
return the buffer.  This eliminates repeated allocations in the hot gradient
loop.

# Usage
```julia
pool = ScratchBufferPool(n_params)
buf  = acquire!(pool)
# ... use buf ...
release!(pool, buf)
```
"""
struct ScratchBufferPool
    channel  :: Channel{Vector{Float64}}
    n_params :: Int

    function ScratchBufferPool(n_params::Int; pool_size::Int = Threads.nthreads())
        n_params > 0  || throw(ArgumentError("n_params must be positive"))
        pool_size > 0 || throw(ArgumentError("pool_size must be positive"))
        ch = Channel{Vector{Float64}}(pool_size)
        for _ in 1:pool_size
            put!(ch, zeros(Float64, n_params))
        end
        new(ch, n_params)
    end
end

"""
    acquire!(pool::ScratchBufferPool) -> Vector{Float64}

Borrow a zero-initialised scratch buffer from `pool`.  Blocks (briefly) if
all buffers are currently in use.  The caller must call `release!(pool, buf)`
when done.
"""
function acquire!(pool::ScratchBufferPool)::Vector{Float64}
    buf = take!(pool.channel)
    fill!(buf, 0.0)   # reset before use
    return buf
end

"""
    release!(pool::ScratchBufferPool, buf::Vector{Float64})

Return `buf` to `pool` so it can be reused by another worker.
"""
function release!(pool::ScratchBufferPool, buf::Vector{Float64})
    put!(pool.channel, buf)
    return nothing
end

# ============================================================================
# EnsembleStrategy hierarchy
# ============================================================================

"""
    AbstractEnsembleStrategy

Abstract supertype for ensemble parallelism strategies.

| Concrete type           | Description                                    |
|:----------------------- |:---------------------------------------------- |
| `ThreadedEnsemble`      | `Threads.@threads` within a single process     |
| `DistributedEnsemble`   | `Distributed.pmap` across worker processes     |
| `SequentialEnsemble`    | Serial loop (useful for debugging / profiling) |
"""
abstract type AbstractEnsembleStrategy end

"""
    ThreadedEnsemble(; n_threads=Threads.maxthreadid())

Parallelise ensemble members using Julia's built-in threading.
Default strategy; scales with `julia -t N`.
"""
struct ThreadedEnsemble <: AbstractEnsembleStrategy
    n_threads :: Int
    ThreadedEnsemble(; n_threads::Int = Threads.nthreads()) = new(n_threads)
end

"""
    SequentialEnsemble()

Evaluate ensemble members one-by-one in a serial loop.
Useful for debugging, profiling, or single-threaded environments.
"""
struct SequentialEnsemble <: AbstractEnsembleStrategy end

"""
    DistributedEnsemble(; workers=Distributed.workers())

Parallelise ensemble members across Julia worker processes using `pmap`.
Requires `using Distributed` and at least one worker process.

# Example
```julia
using Distributed
addprocs(4)
@everywhere using Pulsar

strat  = DistributedEnsemble()
result = ensemble_map_strategy(f, 1:100, strat)
```
"""
struct DistributedEnsemble <: AbstractEnsembleStrategy
    workers :: Vector{Int}
    function DistributedEnsemble(; workers = nothing)
        w = isnothing(workers) ? _get_distributed_workers() : collect(Int, workers)
        new(w)
    end
end

# Lazy lookup so Distributed is not required at load time
function _get_distributed_workers()
    if isdefined(Main, :Distributed)
        return Main.Distributed.workers()
    else
        @warn "DistributedEnsemble: Distributed.jl not loaded. " *
              "Add `using Distributed` and `addprocs(N)` to use distributed parallelism. " *
              "Falling back to single worker."
        return [1]
    end
end

"""
    default_ensemble_strategy() -> AbstractEnsembleStrategy

Return the recommended strategy for the current environment:
- `ThreadedEnsemble` when `Threads.nthreads() > 1`
- `SequentialEnsemble` otherwise
"""
function default_ensemble_strategy()::AbstractEnsembleStrategy
    return Threads.nthreads() > 1 ? ThreadedEnsemble() : SequentialEnsemble()
end

# ── Strategy-dispatched ensemble map ─────────────────────────────────────────

"""
    ensemble_map_strategy(f, iter, strategy::AbstractEnsembleStrategy) -> Vector{Float64}

Map `f(idx)` over every element of `iter` using the given `strategy`.

Returns a `Vector{Float64}` in the same order as `iter`.

# Example
```julia
strat   = ThreadedEnsemble()
results = ensemble_map_strategy(eachindex(drifts), strat) do i
    fidelity_single(θ, drifts[i])
end
```
"""
function ensemble_map_strategy(f, iter, strategy::ThreadedEnsemble)::Vector{Float64}
    indices = collect(iter)
    n   = length(indices)
    out = Vector{Float64}(undef, n)
    @threadsif true for k in 1:n
        out[k] = Float64(f(indices[k]))
    end
    return out
end

function ensemble_map_strategy(f, iter, ::SequentialEnsemble)::Vector{Float64}
    indices = collect(iter)
    return Float64[f(idx) for idx in indices]
end

function ensemble_map_strategy(f, iter, strategy::DistributedEnsemble)::Vector{Float64}
    indices = collect(iter)
    if isdefined(Main, :Distributed)
        pool  = Main.Distributed.WorkerPool(strategy.workers)
        parts = Main.Distributed.pmap(f, pool, indices)
        return Float64.(parts)
    else
        @warn "ensemble_map_strategy(DistributedEnsemble): Distributed.jl not available. " *
              "Falling back to threaded evaluation."
        return ensemble_map_strategy(f, iter, ThreadedEnsemble())
    end
end

# ── Strategy-dispatched ensemble gradient accumulation ──────────────────────

"""
    ensemble_gradient_accumulate!(g_out, θ, grad_member!, iter, strategy;
                                  weight=nothing, pool=nothing) -> g_out

Accumulate the ensemble-averaged gradient into `g_out` using the given strategy
and an optional `ScratchBufferPool`.

When `pool` is provided the per-member gradient is written into a borrowed
scratch buffer (no allocation per member). When `pool` is `nothing`, per-thread
buffers are allocated once at call time (same cost as before).

# Arguments
- `g_out`        — pre-allocated `Vector{Float64}`, overwritten on return
- `θ`            — current parameter vector
- `grad_member!` — `(g, θ, idx) → nothing`: writes per-member gradient into `g`
- `iter`         — ensemble member indices
- `strategy`     — `AbstractEnsembleStrategy`
- `weight`       — optional weight vector (uniform if `nothing`)
- `pool`         — optional `ScratchBufferPool` for allocation-free execution
"""
function ensemble_gradient_accumulate!(
    g_out        :: AbstractVector{<:Real},
    θ            :: AbstractVector{<:Real},
    grad_member! ,
    iter;
    strategy  :: AbstractEnsembleStrategy = default_ensemble_strategy(),
    weight    :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    pool      :: Union{Nothing, ScratchBufferPool}      = nothing,
)
    indices = collect(iter)
    n       = length(indices)
    p       = length(g_out)

    w = if weight === nothing
        nothing
    else
        length(weight) == n || throw(ArgumentError(
            "weight length ($(length(weight))) ≠ ensemble size ($n)"))
        Float64.(weight)
    end

    # Choose buffer source: pool (pre-allocated, reused) or fresh allocation
    _get_buf = if pool !== nothing
        () -> acquire!(pool)
    else
        () -> zeros(Float64, p)
    end
    _put_buf = if pool !== nothing
        (buf) -> release!(pool, buf)
    else
        (_) -> nothing
    end

    acc   = zeros(Float64, p)
    w_sum = Ref(0.0)
    lk    = ReentrantLock()

    if strategy isa ThreadedEnsemble || strategy isa SequentialEnsemble
        @threadsif true for k in 1:n
            buf = _get_buf()
            grad_member!(buf, θ, indices[k])
            wk = w === nothing ? 1.0 : w[k]
            lock(lk) do
                @. acc += wk * buf
                w_sum[] += wk
            end
            _put_buf(buf)
        end
    elseif strategy isa DistributedEnsemble
        # Lesson 8: distributed gradient accumulation via `pmap`.
        # The closure below is serialised to each worker, where its body
        # references `grad_member!` and `θ` from its captured scope. For this
        # to succeed across processes, callers must arrange for the relevant
        # symbols to be available on every worker, e.g.
        #
        #   using Distributed
        #   addprocs(N)
        #   @everywhere using Pulsar
        #   @everywhere include("setup.jl")   # defines per-member functions
        #
        # On a single-worker setup the call degenerates to the sequential path
        # (default Distributed has only worker 1).
        if isdefined(Main, :Distributed)
            pool = Main.Distributed.WorkerPool(strategy.workers)
            partials = Main.Distributed.pmap(pool, indices) do idx
                g_local = zeros(Float64, p)
                grad_member!(g_local, θ, idx)
                return g_local
            end
            for k in 1:n
                gk = partials[k]
                wk = w === nothing ? 1.0 : w[k]
                @. acc += wk * gk
                w_sum[] += wk
            end
        else
            @warn "ensemble_gradient_accumulate!(DistributedEnsemble): " *
                  "Distributed.jl not loaded — falling back to local accumulation."
            for k in 1:n
                buf = _get_buf()
                grad_member!(buf, θ, indices[k])
                wk = w === nothing ? 1.0 : w[k]
                @. acc += wk * buf
                w_sum[] += wk
                _put_buf(buf)
            end
        end
    end

    scale = w_sum[] > 0.0 ? 1.0 / w_sum[] : 1.0 / n
    @. g_out = acc * scale
    return g_out
end

"""
    ensemble_map(f, iter; device=Pulsar.get_device()) → Vector{Float64}

Map `f(idx)` over every element of `iter` using device-appropriate parallelism.

| Device   | Parallelism                                               |
|----------|-----------------------------------------------------------|
| `:cpu`   | `Threads.@threads` — scales with `julia -t N` (default)  |
| `:cuda`  | NVIDIA GPU dispatch if CUDA.jl loaded; else CPU threads   |
| `:metal` | Apple GPU dispatch if Metal.jl loaded; else CPU threads   |

For GRAPE-based optimizers the GPU path is fully exploited automatically
through `MRControl.backend` + `grape_state_kernel` (which runs an optimised
batched GPU kernel). For direct and metaheuristic optimizers the ensemble
function `f` is arbitrary Julia code; the GPU path currently dispatches via
multi-threaded CPU (GPU batching for generic kernels is provided by extensions).

# Arguments
- `f`      — callable, signature `f(idx) → Real`
- `iter`   — any iterable (`eachindex(drifts)`, `1:n`, `CartesianIndices(...)`)
- `device` — compute device symbol (default: `Pulsar.get_device()`)

# Returns
`Vector{Float64}` with `f(idx)` for each element of `iter`, in index order.

# Example
```julia
Pulsar.set_device!(:cpu)    # or :metal / :cuda

function nmr_obj(θ)
    w = reshape(θ, 2, N_ts)
    F_parts = ensemble_map(eachindex(drifts)) do i
        H0 = drifts[i];  ψ = copy(ρ0)
        for k in 1:N_ts
            ψ = compute_propagator(H0 + pwr*(w[1,k]*Lx + w[2,k]*Ly), dt) * ψ
        end
        state_fidelity(ρtg, ψ; type=:real)
    end
    return -sum(F_parts) / length(F_parts)
end
```
"""
function ensemble_map(f, iter; device::Symbol = get_device())
    indices = collect(iter)
    n   = length(indices)
    out = Vector{Float64}(undef, n)

    if device === :cpu
        @threadsif true for k in 1:n
            out[k] = Float64(f(indices[k]))
        end

    elseif device === :cuda
        if _CUDA_LOADED[]
            # Note: for derivative-free / metaheuristic objectives the ensemble
            # function `f` is arbitrary Julia code that cannot be dispatched to
            # the CUDA GPU directly. ensemble_map uses CPU threads here.
            # GPU acceleration for these algorithms is available only through
            # the MRControl/grape_state_kernel path (set backend=:cuda there).
            @info "ensemble_map: objective function uses CPU threads even with " *
                  ":cuda device. GPU acceleration for arbitrary closures is not " *
                  "supported. For GRAPE-based QOC, set MRControl.backend=:cuda." maxlog=1
            @threadsif true for k in 1:n
                out[k] = Float64(f(indices[k]))
            end
        else
            @warn "ensemble_map: CUDA.jl is not loaded — using CPU threads.\n" *
                  "  Add `import CUDA` before `using Pulsar` to enable GPU dispatch."
            @threadsif true for k in 1:n
                out[k] = Float64(f(indices[k]))
            end
        end

    elseif device === :metal
        if _METAL_LOADED[]
            # Note: for derivative-free / metaheuristic objectives the ensemble
            # function `f` is arbitrary Julia code that cannot be dispatched to
            # the Metal GPU directly. ensemble_map uses CPU threads here.
            # GPU acceleration for these algorithms is available only through
            # the MRControl/grape_state_kernel path (set backend=:metal there).
            @info "ensemble_map: objective function uses CPU threads even with " *
                  ":metal device. GPU acceleration for arbitrary closures is not " *
                  "supported. For GRAPE-based QOC, set MRControl.backend=:metal." maxlog=1
            @threadsif true for k in 1:n
                out[k] = Float64(f(indices[k]))
            end
        else
            @warn "ensemble_map: Metal.jl is not loaded — using CPU threads.\n" *
                  "  Add `import Metal` before `using Pulsar` to enable GPU dispatch."
            @threadsif true for k in 1:n
                out[k] = Float64(f(indices[k]))
            end
        end

    else
        throw(ArgumentError("ensemble_map: unknown device :$device. " *
                            "Choose :cpu, :cuda, or :metal."))
    end

    return out
end

# ---------------------------------------------------------------------------
# GPU dispatch stub
# ---------------------------------------------------------------------------
# Default implementation: CPU threads. Overloaded by GPU extension modules
# when CUDA.jl / Metal.jl is loaded, replacing this with a batched GPU kernel.
function _ensemble_map_gpu!(out::Vector{Float64}, f, indices::Vector, device::Symbol)
    n = length(indices)
    @threadsif true for k in 1:n
        out[k] = Float64(f(indices[k]))
    end
end

# ============================================================================
# ensemble_grad!
# ============================================================================

"""
    ensemble_grad!(g_out, θ, grad_member!, iter; weight=nothing, device=get_device())

Accumulate the ensemble-averaged gradient **in-place** into `g_out`.

Uses the identity

    ∇_θ  E[f(θ, ξ)]  =  E[∇_θ f(θ, ξ)]

so `g_out` receives the (optionally weighted) mean of per-member gradients,
which is the exact gradient of the ensemble-averaged objective built by
`ensemble_map`.

Compatible with every gradient-based optimizer in Pulsar that accepts a
`grad!(g, θ)` signature: `lbfgs_optimize`, `bfgs_optimize`, `adam_optimize`,
`cg_optimize`, `grape_cg_optimize`, `grape_lbfgsb_optimize`, `krotov_optimize`,
`newton_optimize`, and all variants in `Gradient/Generic/`.

# Arguments
- `g_out`         — pre-allocated `Vector{Float64}` of length `length(θ)`;
                    overwritten on return.
- `θ`             — current parameter vector.
- `grad_member!`  — callable with signature `grad_member!(g, θ, idx)` that
                    writes the gradient for ensemble member `idx` into `g`.
                    Must be thread-safe (each call receives its own buffer).
- `iter`          — iterable of member indices, e.g. `eachindex(drifts)`.
- `weight`        — optional `AbstractVector{<:Real}` of length `length(iter)`.
                    If `nothing` (default) all members are equally weighted.
- `device`        — `:cpu` (default), `:cuda`, or `:metal`; follows the same
                    device dispatch as `ensemble_map`.

# Example — broadband NMR with any gradient-based optimizer
```julia
drifts  = [hamiltonian(sys; offset_hz=Δf) for Δf in range(-5000, 5000, 21)]
N_ts    = 200;  dt = 2e-6;  pwr = 2π*10_000.0

function f(θ)
    F_parts = ensemble_map(eachindex(drifts)) do i
        fidelity_single(θ, drifts[i])
    end
    return -mean(F_parts)
end

function grad!(g, θ)
    ensemble_grad!(g, θ, eachindex(drifts)) do gi, ti, i
        grape_gradient_single!(gi, ti, drifts[i])   # your per-member grad
    end
end

# Plug into any optimizer — no GRAPE-specific code needed here:
θ_opt, f_opt, stats = lbfgs_optimize(f, grad!, θ0)
θ_opt, f_opt, stats = adam_optimize(f, grad!, θ0)
θ_opt, f_opt, stats = cg_optimize(f, grad!, θ0)
```
"""
function ensemble_grad!(
    g_out        :: AbstractVector{<:Real},
    θ            :: AbstractVector{<:Real},
    grad_member! ,   # (g, θ, idx) → nothing
    iter;
    weight       :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    device       :: Symbol = get_device(),
)
    indices = collect(iter)
    n       = length(indices)
    p       = length(g_out)

    # Validate weights
    w = if weight === nothing
        nothing
    else
        length(weight) == n || throw(ArgumentError(
            "ensemble_grad!: weight length ($(length(weight))) ≠ " *
            "number of ensemble members ($n)."))
        Float64.(weight)
    end

    # Per-thread gradient buffers (avoids allocations inside the loop)
    n_threads = Threads.maxthreadid()
    bufs = [zeros(Float64, p) for _ in 1:n_threads]

    # Accumulator (protected by a lock for thread safety)
    acc   = zeros(Float64, p)
    w_sum = Ref(0.0)
    lk    = ReentrantLock()

    _eg_threaded!(acc, w_sum, lk, bufs, θ, grad_member!, indices, w)

    # Normalise and write result
    scale = w_sum[] > 0.0 ? 1.0 / w_sum[] : 1.0 / n
    @. g_out = acc * scale
    return g_out
end

# Internal threaded kernel (device-agnostic for now; GPU path same as ensemble_map)
function _eg_threaded!(acc, w_sum, lk, bufs, θ, grad_member!, indices, w)
    n = length(indices)
    @threadsif true for k in 1:n
        tid = Threads.threadid()
        buf = bufs[tid]
        fill!(buf, 0.0)
        grad_member!(buf, θ, indices[k])
        wk = w === nothing ? 1.0 : w[k]
        lock(lk) do
            @. acc += wk * buf
            w_sum[] += wk
        end
    end
end

"""
    ensemble_fobj(f_member, grad_member!, iter; weight=nothing, device=get_device())
        → (f::Function, grad!::Function)

Convenience constructor: returns a matched `(f, grad!)` pair that computes the
ensemble-averaged objective and gradient, ready to pass to any Pulsar optimizer.

# Arguments
- `f_member`      — `f_member(θ, idx) → Real`
- `grad_member!`  — `grad_member!(g, θ, idx) → nothing`
- `iter`          — ensemble member indices
- `weight`        — optional weight vector (see `ensemble_grad!`)
- `device`        — compute device (see `ensemble_map`)

# Example
```julia
f, grad! = ensemble_fobj(eachindex(drifts);
                         weight = ones(length(drifts))) do θ, i
    fidelity_single(θ, drifts[i])
end  # NOTE: f_member passed as do-block

# Not quite — use keyword form:
f_mem(θ, i)       = fidelity_single(θ, drifts[i])
gm!(g, θ, i)      = grape_gradient_single!(g, θ, drifts[i])
f_ens, grad_ens!  = ensemble_fobj(f_mem, gm!, eachindex(drifts))

θ_opt, _, _ = lbfgs_optimize(f_ens, grad_ens!, θ0)
```
"""
function ensemble_fobj(
    f_member,
    grad_member!,
    iter;
    weight  :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    device  :: Symbol = get_device(),
)
    indices = collect(iter)

    f = function(θ)
        parts = ensemble_map(indices; device=device) do i
            Float64(f_member(θ, i))
        end
        w = weight
        if w === nothing
            return -mean(parts)
        else
            return -dot(w, parts) / sum(w)
        end
    end

    grad! = function(g, θ)
        ensemble_grad!(g, θ, grad_member!, indices;
                       weight=weight, device=device)
    end

    return f, grad!
end
