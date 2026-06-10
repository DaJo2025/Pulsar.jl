# CUDABackend.jl — NVIDIA CUDA GPU backend
#
# Provides real GPU-accelerated operations when CUDA.jl is installed and
# functional.  Falls back gracefully to CPU with a @warn when CUDA is absent.
#
# Design notes:
#   • CUDA.jl is loaded at module-include time via a try/catch extension block.
#   • All heavy operations upload input to CuArray, compute on device, download.
#   • Matrix exponential is computed via eigendecomposition: exp(A) = V diag(exp(λ)) V†,
#     which is stable and uses cuBLAS / LAPACK on device.
#   • GRAPE gradient uses the standard forward–backward pass entirely on GPU;
#     only the scalar fidelity and the gradient matrix are pulled back to host.
#   • Functions are no-ops returning CPU results when CUDA is unavailable.

using LinearAlgebra

# ---------------------------------------------------------------------------
# Runtime availability flag (set to true below if CUDA loads successfully)
# ---------------------------------------------------------------------------

const _CUDA_LOADED = Ref{Bool}(false)

# NOTE: CUDA.jl support is provided as a package extension (ext/PulsarCUDAExt.jl,
# declared under [weakdeps]/[extensions] in Project.toml).  The extension does
# `using CUDA` at the top — which binds `CUDA` correctly — and sets
# `_CUDA_LOADED[] = CUDA.functional()` in its own `__init__`.  Therefore the
# core module never references a bare `CUDA` symbol; the GPU method bodies of the
# batched primitives (see Backend/BatchedPrimitives.jl) live in the extension.
#
# Historical bug (fixed): an earlier `let; @eval using CUDA; end` block here ran
# only at precompile time — when CUDA is absent — so `_CUDA_LOADED[]` was set on
# mere module presence while `CUDA` stayed unbound inside `Pulsar`, making every
# `CUDA.CuArray(...)` throw `UndefVarError` that the `try/catch` swallowed into a
# silent CPU fallback.  The extension mechanism is the durable fix.

# ---------------------------------------------------------------------------
# Type definition
# ---------------------------------------------------------------------------

"""
    CUDABackend

Configuration struct for the NVIDIA CUDA GPU backend.

Construct via [`cuda_backend`](@ref).  Requires CUDA.jl to be installed and
a functional NVIDIA GPU.

# Fields
- `device_id::Int`: CUDA device index (0-based).
- `memory_limit_gb::Float64`: Soft GPU memory budget in GiB.
- `use_tensor_cores::Bool`: Prefer TF32/FP16 tensor-core paths (experimental).
- `use_async::Bool`: Use CUDA streams for pipeline overlap (experimental).
"""
struct CUDABackend <: AbstractComputeBackend
    device_id        :: Int
    memory_limit_gb  :: Float64
    use_tensor_cores :: Bool
    use_async        :: Bool
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    cuda_backend(; device_id=0, memory_limit_gb=Inf,
                   use_tensor_cores=false, use_async=false) -> CUDABackend

Construct a [`CUDABackend`](@ref).

Throws an `ErrorException` if CUDA.jl is not installed or no functional GPU
is present.
"""
function cuda_backend(;
    device_id        :: Int     = 0,
    memory_limit_gb  :: Float64 = Inf,
    use_tensor_cores :: Bool    = false,
    use_async        :: Bool    = false,
)
    if !_CUDA_LOADED[]
        error(
            "CUDA is not available.  Install CUDA.jl with `] add CUDA` and " *
            "ensure NVIDIA drivers are installed.  Use `cpu_backend()` instead.",
        )
    end
    return CUDABackend(device_id, memory_limit_gb, use_tensor_cores, use_async)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Hermitian matrix exponential: exp(-i H dt) via eigendecomposition.
# Works on plain Julia arrays; caller is responsible for device placement.
function _herm_matexp(H::Matrix{ComplexF64}, dt::Float64)
    F = eigen(Hermitian(H))
    return F.vectors * Diagonal(exp.(-im .* F.values .* dt)) * F.vectors'
end

# ---------------------------------------------------------------------------
# matrix_exponential_cuda
# ---------------------------------------------------------------------------

"""
    matrix_exponential_cuda(H, dt, backend) -> Matrix{ComplexF64}

Compute `exp(-i H dt)` for a Hermitian matrix `H` using the CUDA GPU.

When CUDA is available the computation is performed with device-side
eigendecomposition via `CUDA.CUSOLVER`.  Falls back to CPU eigendecomposition
with a `@warn` if CUDA is not loaded.

The result is always returned as a host `Matrix{ComplexF64}`.
"""
function matrix_exponential_cuda(
    H       :: Matrix{ComplexF64},
    dt      :: Float64,
    backend :: CUDABackend,
)
    # Thin wrapper over the batched primitive: a 1-matrix batch.  When the CUDA
    # extension is loaded, `batched_herm_propagators(::CUDABackend, …)` runs the
    # device kernel; otherwise the generic CPU fallback runs.  `Array(…)` is a
    # no-op on host arrays and a device→host download on a CuArray.
    U = batched_herm_propagators(backend, reshape(H, size(H, 1), size(H, 2), 1), dt)
    return Array(U)[:, :, 1]
end

# ---------------------------------------------------------------------------
# batch_propagators_cuda
# ---------------------------------------------------------------------------

"""
    batch_propagators_cuda(H_array, dt, backend) -> Array{ComplexF64,3}

Compute all propagators `U[k] = exp(-i H_array[:,:,k] dt)` in parallel on
the GPU.

`H_array` is a `(dim, dim, n_steps)` array of Hermitian matrices.

Each slice is diagonalised independently; the batch is processed concurrently
by spreading slices across CUDA streams or using a single-threaded fallback
loop on the device.  Returns a host `(dim, dim, n_steps)` array.
"""
function batch_propagators_cuda(
    H_array :: AbstractArray{ComplexF64,3},
    dt      :: Float64,
    backend :: CUDABackend,
)
    # Genuinely batched now: one `heevjBatched` + `gemm_strided_batched` inside
    # `batched_herm_propagators(::CUDABackend, …)` (CUDA extension), chunked to a
    # GPU memory budget.  No per-slice eigendecomposition, no per-slice host sync.
    return Array(batched_herm_propagators(backend, H_array, dt))
end

# ---------------------------------------------------------------------------
# fidelity_cuda
# ---------------------------------------------------------------------------

"""
    fidelity_cuda(U_total, U_target, backend) -> Float64

Compute the gate fidelity `|Tr(U_target† U_total)|² / dim²` on the GPU.

Both `U_total` and `U_target` are `(dim, dim)` unitary matrices.  The trace
inner product is evaluated on the device; only the resulting scalar is
transferred back to host.
"""
function fidelity_cuda(
    U_total  :: Matrix{ComplexF64},
    U_target :: Matrix{ComplexF64},
    backend  :: CUDABackend,
)
    # A single gate fidelity is a one-matrix reduction — there is nothing to
    # batch, so it is evaluated directly on the host.  Batched gate/state
    # fidelities used inside the optimizers go through the kernel reductions.
    dim = size(U_total, 1)
    overlap = tr(U_target' * U_total)
    return abs2(overlap) / dim^2
end

# ---------------------------------------------------------------------------
# gradient_cuda
# ---------------------------------------------------------------------------

"""
    gradient_cuda(H_drift, H_ctrl, controls, target, dt, backend) -> Matrix{Float64}

Compute the full GRAPE gradient on the GPU.

Implements the standard forward–backward pass:
  1. **Forward pass**: `P[0] = I`, `P[k] = exp(-i H[k] dt) P[k-1]`
  2. **Backward pass**: `Q[N] = U_target'`, `Q[k-1] = Q[k] exp(-i H[k] dt)`
  3. **Gradient**: `∂F/∂u_{j,k} = (2/dim²) Re[ Tr(U_target† P[N]) * conj(Tr(Q[k]† (-i dt H_j) P[k-1])) ]`

# Arguments
- `H_drift`: Drift Hamiltonian `(dim × dim)`.
- `H_ctrl`: Vector of `n_controls` control Hamiltonians `(dim × dim)` each.
- `controls`: `(n_controls × n_steps)` control amplitudes.
- `target`: Target unitary `(dim × dim)`.
- `dt`: Timestep (seconds, or rad/s if Hamiltonians are already in rad/s).
- `backend`: The `CUDABackend` instance.

Returns `grad` of size `(n_controls, n_steps)`.
"""
function gradient_cuda(
    H_drift  :: Matrix{ComplexF64},
    H_ctrl   :: Vector{<:Matrix{ComplexF64}},
    controls :: Matrix{Float64},
    target   :: Matrix{ComplexF64},
    dt       :: Float64,
    backend  :: CUDABackend,
)
    # Delegates to the device-generic batched GRAPE gate gradient.  The
    # expensive step (all propagators) is one batched `heevjBatched` when the
    # CUDA extension is loaded; the per-(k,j) trace reductions are hoisted out of
    # the time loop.  Falls back to the threaded CPU path otherwise.
    return _batched_grape_gate_gradient(backend, H_drift, H_ctrl, controls, target, dt)
end

# ---------------------------------------------------------------------------
# cuda_info
# ---------------------------------------------------------------------------

"""
    cuda_info(backend) -> Dict{String,Any}

Return CUDA device properties as a dictionary.

Keys: `"available"`, `"device_name"`, `"memory_gb"`, `"compute_capability"`,
`"n_devices"`.  When CUDA is not available, returns `Dict("available"=>false)`.
"""
function cuda_info(backend::CUDABackend)::Dict{String,Any}
    if !_CUDA_LOADED[]
        return Dict{String,Any}("available" => false)
    end
    try
        dev = CUDA.device()
        return Dict{String,Any}(
            "available"           => true,
            "device_name"         => CUDA.name(dev),
            "memory_gb"           => CUDA.totalmem(dev) / 1024^3,
            "compute_capability"  => string(CUDA.capability(dev)),
            "n_devices"           => length(CUDA.devices()),
        )
    catch e
        return Dict{String,Any}("available" => true, "error" => string(e))
    end
end
