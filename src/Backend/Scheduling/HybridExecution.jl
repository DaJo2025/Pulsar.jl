"""
    HybridExecution.jl

Hybrid CPU/GPU execution orchestrator for Pulsar.jl.

Automatically selects the most efficient compute device (`:cpu`, `:gpu`, or
`:hybrid`) for each quantum control operation based on:

- **Problem size**: Hilbert-space dimension `d` and number of time steps `N`.
- **Available hardware**: CUDA, Metal, or CPU-only.
- **Empirical timings**: Historical performance recorded during previous runs
  is used to refine decisions (adaptive learning).
- **Operation type**: FLOP estimates differ significantly between propagator
  computation, fidelity evaluation, and gradient computation.

# Decision rules (default thresholds)
| Condition                           | Decision    |
|-------------------------------------|-------------|
| `d ≤ cpu_threshold_dim` (≤16)       | `:cpu`      |
| `d > 64`   AND gpu available        | `:gpu`      |
| `16 < d ≤ 64`                       | benchmark   |
| GPU memory would exceed threshold   | `:cpu`      |
| Historical data available           | use history |
"""

using LinearAlgebra
using Statistics: mean

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

"""
    HybridExecutionPlanner

Stateful orchestrator for hybrid CPU/GPU execution decisions.

# Fields
- `cpu_threshold_dim::Int`: Hilbert-space dimension at or below which CPU
  is preferred.  Default 16 (≤ 4 qubits).  Below this threshold GPU launch
  overhead dominates computation time.
- `gpu_threshold_dim::Int`: Dimension above which GPU is preferred when
  available.  Default 64 (> 6 qubits).
- `gpu_memory_threshold::Float64`: If the estimated GPU allocation would
  exceed this fraction of total GPU memory, fall back to CPU.  Default 0.85.
- `enable_overlap::Bool`: When `true`, the planner schedules CPU work to
  overlap with GPU data transfers (experimental; requires async backends).
- `historical_timings::Dict{String,Float64}`: Cache of `(op, device, dim, N)`
  → seconds/step.  Updated by `update_historical_performance!`.
"""
mutable struct HybridExecutionPlanner
    cpu_threshold_dim::Int
    gpu_threshold_dim::Int
    gpu_memory_threshold::Float64
    enable_overlap::Bool
    historical_timings::Dict{String,Float64}
    # Batch-volume awareness: for *many small* matrices the batch COUNT, not the
    # matrix dim, decides CPU vs GPU.  A small-dim problem still goes to the GPU
    # once the batch is large enough to amortise launch + transfer overhead.
    cpu_batch_threshold::Int        # max batch count that still prefers CPU at small dim
    gpu_volume_threshold::Float64   # batch_count·dim² element count that forces GPU

    function HybridExecutionPlanner(
        cpu_threshold_dim::Int,
        gpu_threshold_dim::Int,
        gpu_memory_threshold::Float64,
        enable_overlap::Bool,
        historical_timings::Dict{String,Float64},
        cpu_batch_threshold::Int = 512,
        gpu_volume_threshold::Float64 = 2.0e6,
    )
        cpu_threshold_dim > 0 ||
            throw(ArgumentError("cpu_threshold_dim must be positive"))
        0 < gpu_memory_threshold ≤ 1.0 ||
            throw(ArgumentError("gpu_memory_threshold must be in (0, 1]"))
        cpu_batch_threshold > 0 ||
            throw(ArgumentError("cpu_batch_threshold must be positive"))
        new(cpu_threshold_dim, gpu_threshold_dim, gpu_memory_threshold,
            enable_overlap, historical_timings,
            cpu_batch_threshold, gpu_volume_threshold)
    end
end

"""
    HybridExecutionPlanner(; cpu_threshold_dim=16, gpu_threshold_dim=64,
                              gpu_memory_threshold=0.85, enable_overlap=false,
                              cpu_batch_threshold=512, gpu_volume_threshold=2e6,
                              historical_timings=Dict{String,Float64}()
                            ) -> HybridExecutionPlanner

Keyword constructor with recommended defaults.

`cpu_batch_threshold` and `gpu_volume_threshold` add **batch-volume awareness**:
a problem with a small matrix dimension but a large batch (e.g. an ensemble ×
many timesteps of 4×4 matrices) is routed to the GPU, where the old dim-only
heuristic wrongly forced CPU.

# Examples
```julia
planner = HybridExecutionPlanner()
planner = HybridExecutionPlanner(cpu_threshold_dim=8, gpu_memory_threshold=0.7)
```
"""
function HybridExecutionPlanner(;
    cpu_threshold_dim::Int = 16,
    gpu_threshold_dim::Int = 64,
    gpu_memory_threshold::Float64 = 0.85,
    enable_overlap::Bool = false,
    cpu_batch_threshold::Int = 512,
    gpu_volume_threshold::Float64 = 2.0e6,
    historical_timings::Dict{String,Float64} = Dict{String,Float64}(),
)::HybridExecutionPlanner
    return HybridExecutionPlanner(
        cpu_threshold_dim, gpu_threshold_dim, gpu_memory_threshold,
        enable_overlap, historical_timings,
        cpu_batch_threshold, gpu_volume_threshold,
    )
end

# ---------------------------------------------------------------------------
# GPU availability helpers
# ---------------------------------------------------------------------------

"""
    _gpu_available() -> Bool

Return `true` if any GPU backend (CUDA or Metal) is available, i.e. the
corresponding package extension has set `_CUDA_LOADED[]` / `_METAL_LOADED[]`.
"""
function _gpu_available()::Bool
    return _CUDA_LOADED[] || _METAL_LOADED[]
end

# Resolve a loaded GPU package module by name (e.g. "CUDA", "Metal") without a
# bare reference — the package is only present when its extension is active.
function _loaded_gpu_module(name::AbstractString)
    id = Base.identify_package(name)
    (id !== nothing && haskey(Base.loaded_modules, id)) || return nothing
    return Base.loaded_modules[id]
end

"""
    _gpu_free_memory_gb() -> Float64

Return the amount of free GPU memory in GiB.
Returns `Inf` when no GPU is available (no memory constraint).
"""
function _gpu_free_memory_gb()::Float64
    if _CUDA_LOADED[]
        try
            CUDAmod = _loaded_gpu_module("CUDA")
            CUDAmod === nothing || return CUDAmod.memory_info()[1] / 1024^3
        catch; end
    end
    if _METAL_LOADED[]
        try
            Metalmod = _loaded_gpu_module("Metal")
            if Metalmod !== nothing
                dev = Metalmod.current_device()
                return Metalmod.recommended_working_set_size(dev) / 1024^3
            end
        catch; end
    end
    return Inf
end

"""
    _gpu_total_memory_gb() -> Float64

Return total GPU memory in GiB, or `Inf` if unavailable.
"""
function _gpu_total_memory_gb()::Float64
    if _CUDA_LOADED[]
        try
            CUDAmod = _loaded_gpu_module("CUDA")
            CUDAmod === nothing || return CUDAmod.memory_info()[2] / 1024^3
        catch; end
    end
    if _METAL_LOADED[]
        try
            Metalmod = _loaded_gpu_module("Metal")
            if Metalmod !== nothing
                dev = Metalmod.current_device()
                return Metalmod.recommended_working_set_size(dev) / 1024^3
            end
        catch; end
    end
    return Inf
end

# ---------------------------------------------------------------------------
# FLOP estimates
# ---------------------------------------------------------------------------

"""
    estimate_flops(op::String, dim::Int, n_timesteps::Int,
                   n_controls::Int=1) -> Float64

Estimate the number of floating-point operations for a given operation.

| Operation     | FLOP estimate (leading order)                          |
|---------------|--------------------------------------------------------|
| `"propagator"`| `N × d³` (eigendecomposition per step)                 |
| `"fidelity"`  | `d²`     (trace of d×d product)                        |
| `"gradient"`  | `N × n_ctrl × d²` (inner products, shared propagators) |
| `"matmul"`    | `d³`     (single matrix–matrix multiply)               |
| anything else | `N × d²` (generic estimate)                            |

# Arguments
- `op`: Operation name string.
- `dim`: Hilbert-space dimension `d`.
- `n_timesteps`: Number of time steps `N`.
- `n_controls`: Number of control channels (used for `"gradient"`).

# Returns
Estimated FLOP count as `Float64`.
"""
function estimate_flops(
    op::String,
    dim::Int,
    n_timesteps::Int,
    n_controls::Int = 1,
)::Float64
    d = Float64(dim)
    N = Float64(n_timesteps)
    J = Float64(n_controls)
    return if op == "propagator"
        N * d^3
    elseif op == "fidelity"
        d^2
    elseif op == "gradient"
        # Shared propagators O(N d³) + inner products O(N J d²)
        N * d^3 + N * J * d^2
    elseif op == "matmul"
        d^3
    else
        N * d^2
    end
end

# ---------------------------------------------------------------------------
# Operation time estimator
# ---------------------------------------------------------------------------

"""
    estimate_operation_time(op::String, dim::Int, n_timesteps::Int,
                             device::Symbol;
                             n_controls::Int=1,
                             planner::Union{HybridExecutionPlanner,Nothing}=nothing
                            ) -> Float64

Estimate the wall-clock time in seconds for an operation on the given device.

# Estimation approach
1. If `planner` provides a matching historical timing entry, use it directly.
2. Otherwise apply empirical peak-FLOP estimates:

| Device  | Peak GFLOPS (FP64) | Notes                          |
|---------|--------------------|--------------------------------|
| `:cpu`  | 200 GFLOP/s        | 8-core with AVX2               |
| `:gpu`  | 5000 GFLOP/s       | modern CUDA GPU (A100-class)   |

For GPU the estimate **adds** a data-transfer overhead:
    `transfer_time ≈ n_bytes / (25 GB/s PCIe bandwidth)`

# Arguments
- `op`: Operation name (see `estimate_flops`).
- `dim`: Hilbert-space dimension.
- `n_timesteps`: Number of time steps.
- `device`: `:cpu` or `:gpu`.
- `n_controls`: Number of control channels.
- `planner`: Optional planner whose `historical_timings` are consulted first.

# Returns
Estimated time in seconds.
"""
function estimate_operation_time(
    op::String,
    dim::Int,
    n_timesteps::Int,
    device::Symbol;
    n_controls::Int = 1,
    planner::Union{HybridExecutionPlanner,Nothing} = nothing,
)::Float64
    # Check historical cache first
    if planner !== nothing
        key = _timing_key(op, device, dim, n_timesteps)
        if haskey(planner.historical_timings, key)
            return planner.historical_timings[key]
        end
    end

    flops = estimate_flops(op, dim, n_timesteps, n_controls)

    if device == :cpu
        peak_gflops = 200.0   # Typical 8-core AVX2 FP64
        return flops / (peak_gflops * 1e9)
    elseif device == :gpu
        peak_gflops = 5000.0  # A100-class FP64 (conservative)
        compute_time = flops / (peak_gflops * 1e9)
        # Round-trip transfer: upload H_array + download results (×2)
        # Each time step: one (dim×dim) ComplexF64 matrix = 16 dim² bytes
        n_bytes = Float64(n_timesteps) * dim^2 * 16  # ComplexF64 per slice
        transfer_time = 2.0 * n_bytes / (25.0 * 1024^3)  # upload + download
        return compute_time + transfer_time
    else
        error("Unknown device :$device.  Expected :cpu or :gpu.")
    end
end

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

"""
    adaptive_backend_selection(op::String, dim::Int, n_timesteps::Int,
                                available_resources::Dict{String,Any};
                                n_controls::Int=1,
                                planner::Union{HybridExecutionPlanner,Nothing}=nothing
                               ) -> Symbol

Select the fastest available backend for a given operation.

# Decision procedure
1. If `available_resources["gpu"]` is `false`, return `:cpu` immediately.
2. Estimate CPU time and GPU time (including transfer overhead).
3. Return the symbol with the lower estimated time.

# Arguments
- `op`: Operation name.
- `dim`: Hilbert-space dimension.
- `n_timesteps`: Number of time steps.
- `available_resources`: Dict with at least key `"gpu"::Bool`.
- `n_controls`: Number of control channels.
- `planner`: Optional planner for historical data.

# Returns
`:cpu` or `:gpu`.

# Examples
```julia
resources = Dict("gpu" => cuda_available, "n_cpu_threads" => 8)
device = adaptive_backend_selection("gradient", 32, 500, resources; n_controls=4)
```
"""
function adaptive_backend_selection(
    op::String,
    dim::Int,
    n_timesteps::Int,
    available_resources::Dict{String,Any};
    n_controls::Int = 1,
    planner::Union{HybridExecutionPlanner,Nothing} = nothing,
)::Symbol
    gpu_available = get(available_resources, "gpu", false)::Bool
    !gpu_available && return :cpu

    t_cpu = estimate_operation_time(op, dim, n_timesteps, :cpu;
                                    n_controls = n_controls, planner = planner)
    t_gpu = estimate_operation_time(op, dim, n_timesteps, :gpu;
                                    n_controls = n_controls, planner = planner)

    return t_gpu < t_cpu ? :gpu : :cpu
end

# ---------------------------------------------------------------------------
# Plan hybrid execution
# ---------------------------------------------------------------------------

"""
    plan_hybrid_execution(dim::Int, n_timesteps::Int,
                           gpu_available::Bool,
                           planner::HybridExecutionPlanner;
                           op::String="propagator",
                           n_controls::Int=1) -> Symbol

High-level decision function that returns `:cpu`, `:gpu`, or `:hybrid`.

# Rules (applied in order)
1. **No GPU** → `:cpu`.
2. **Small system** (`dim ≤ planner.cpu_threshold_dim`) → `:cpu`.
   GPU launch overhead dominates for small matrices.
3. **GPU memory** overflow → `:cpu`.
   Estimated allocation > `planner.gpu_memory_threshold × total_gpu_memory`.
4. **Large system** (`dim > planner.gpu_threshold_dim`) → `:gpu`.
5. **Intermediate system**: consult historical timings or fall back to FLOP
   estimates via `adaptive_backend_selection`.  Return `:hybrid` when the
   two devices have comparable estimated times (within 20%).

# Arguments
- `dim`: Hilbert-space dimension.
- `n_timesteps`: Number of time steps.
- `gpu_available`: Whether a GPU backend is functional.
- `planner`: Planner instance (thresholds + history).
- `op`: Operation for timing estimate.  Default `"propagator"`.
- `n_controls`: Number of control channels.

# Returns
`:cpu`, `:gpu`, or `:hybrid`.
"""
function plan_hybrid_execution(
    dim::Int,
    n_timesteps::Int,
    gpu_available::Bool,
    planner::HybridExecutionPlanner;
    op::String = "propagator",
    n_controls::Int = 1,
    batch_count::Int = n_timesteps,
)::Symbol
    # `batch_count` is the total number of matrices in the batched eigensolve
    # (fused ensemble-member × timestep, etc.).  It — not `dim` alone — drives
    # the small-matrix decision: 763 560 4×4 matrices belong on the GPU even
    # though dim=4 is "small".
    N = max(batch_count, 1)

    # Rule 1: No GPU
    !gpu_available && return :cpu

    # Rule 2: Small system AND small batch → CPU (launch + transfer overhead
    # dominates).  Previously this returned CPU on small dim regardless of N.
    if dim <= planner.cpu_threshold_dim && N <= planner.cpu_batch_threshold
        return :cpu
    end

    # Rule 3: GPU memory check — sized from the full batch, not just timesteps.
    estimated_bytes = N * dim * dim * sizeof(ComplexF64)
    total_gpu_gb = _gpu_total_memory_gb()
    if isfinite(total_gpu_gb)
        # A batch larger than the budget is chunked (see batched_chunk_size), so
        # only fall back to CPU when even a single matrix is implausibly large.
        single_bytes = dim * dim * sizeof(ComplexF64)
        if single_bytes / (total_gpu_gb * 1024^3) > planner.gpu_memory_threshold
            @info "Falling back to CPU: a single $(dim)×$(dim) matrix " *
                  "exceeds the GPU memory budget."
            return :cpu
        end
    end

    # Rule 4: Large matrix dimension → GPU.
    dim > planner.gpu_threshold_dim && return :gpu

    # Rule 4b: Large batch volume (count·dim²) → GPU even at small dim.  This is
    # the path that captures the many-small-matrices regime.
    if N * dim * dim >= planner.gpu_volume_threshold
        return :gpu
    end

    # Rule 5: Intermediate – compare estimates (timed over the full batch).
    t_cpu = estimate_operation_time(op, dim, N, :cpu;
                                    n_controls = n_controls, planner = planner)
    t_gpu = estimate_operation_time(op, dim, N, :gpu;
                                    n_controls = n_controls, planner = planner)

    ratio = t_cpu / max(t_gpu, 1e-15)
    if ratio > 1.2
        return :gpu
    elseif ratio < 0.833
        return :cpu
    else
        # Times within 20% of each other → hybrid
        return :hybrid
    end
end

# ---------------------------------------------------------------------------
# Historical performance update
# ---------------------------------------------------------------------------

"""
    _timing_key(op::String, device::Symbol, dim::Int, n_timesteps::Int) -> String

Construct a canonical string key for the `historical_timings` dictionary.
"""
function _timing_key(op::String, device::Symbol, dim::Int, n_timesteps::Int)::String
    return "$(op)|$(device)|$(dim)|$(n_timesteps)"
end

"""
    update_historical_performance!(planner::HybridExecutionPlanner,
                                    op::String, device::Symbol,
                                    dim::Int, n_timesteps::Int,
                                    actual_time::Float64)

Update the timing database with an observed wall-clock measurement.

The update uses an exponential moving average with `α = 0.3` so that
recent measurements are weighted more heavily:

    new_estimate = α × actual_time + (1 - α) × old_estimate

When no prior estimate exists, `actual_time` is stored directly.

# Arguments
- `planner`: Planner whose `historical_timings` dict is updated.
- `op`: Operation name.
- `device`: `:cpu` or `:gpu`.
- `dim`: Hilbert-space dimension of the measured problem.
- `n_timesteps`: Number of time steps of the measured problem.
- `actual_time`: Measured wall-clock time in seconds.

# Thread safety
`historical_timings` is a plain `Dict` and is **not** thread-safe.
Call this function from a single-threaded context (e.g., after each
optimisation iteration) rather than from parallel regions.

# Examples
```julia
t_start = time()
result = batch_propagators_cpu(H_array, dt, backend)
t_elapsed = time() - t_start
update_historical_performance!(planner, "propagator", :cpu, 16, 200, t_elapsed)
```
"""
function update_historical_performance!(
    planner::HybridExecutionPlanner,
    op::String,
    device::Symbol,
    dim::Int,
    n_timesteps::Int,
    actual_time::Float64,
)
    actual_time >= 0 || throw(ArgumentError("actual_time must be non-negative"))
    key = _timing_key(op, device, dim, n_timesteps)
    α = 0.3  # EMA smoothing factor
    if haskey(planner.historical_timings, key)
        old = planner.historical_timings[key]
        planner.historical_timings[key] = α * actual_time + (1.0 - α) * old
    else
        planner.historical_timings[key] = actual_time
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Resource summary
# ---------------------------------------------------------------------------

"""
    available_resources() -> Dict{String, Any}

Return a dictionary of available compute resources for use with
`adaptive_backend_selection` and `plan_hybrid_execution`.

# Keys
- `"gpu"::Bool`: Whether any GPU backend is available.
- `"cuda"::Bool`: Whether CUDA is available.
- `"metal"::Bool`: Whether Metal is available.
- `"n_cpu_threads"::Int`: Number of Julia threads.
- `"cpu_memory_gb"::Float64`: Total RAM in GiB.
- `"gpu_free_memory_gb"::Float64`: Free GPU memory in GiB (or `Inf`).
- `"gpu_total_memory_gb"::Float64`: Total GPU memory in GiB (or `Inf`).

# Examples
```julia
resources = available_resources()
device = adaptive_backend_selection("gradient", 32, 500, resources)
```
"""
function available_resources()::Dict{String,Any}
    cuda  = _CUDA_LOADED[]
    metal = _METAL_LOADED[]
    gpu   = cuda || metal

    return Dict{String,Any}(
        "gpu"               => gpu,
        "cuda"              => cuda,
        "metal"             => metal,
        "n_cpu_threads"     => Threads.nthreads(),
        "cpu_memory_gb"     => Sys.total_memory() / 1024^3,
        "gpu_free_memory_gb"  => _gpu_free_memory_gb(),
        "gpu_total_memory_gb" => _gpu_total_memory_gb(),
    )
end
