# MetalBackend.jl — Apple Metal GPU backend
#
# Provides real GPU-accelerated operations when Metal.jl is installed and
# running on Apple Silicon (M1/M2/M3/M4) or a macOS Metal-capable GPU.
# Falls back gracefully to CPU with a @warn when Metal is absent.
#
# Design notes:
#   • Metal.jl is loaded at module-include time via a try/catch block.
#   • Metal natively supports Float32; Float64 is NOT supported in Metal shaders.
#     For quantum mechanics (which requires complex arithmetic to ~1e-10), we:
#     (a) Use Float32 for intermediate computations when use_fp32=true, or
#     (b) Use Metal for embarrassingly-parallel real-valued work and fall back
#         to CPU for the precision-critical complex eigendecompositions.
#   • The strategy used here: upload to MtlArray, run Metal BLAS (gemm!),
#     but perform eigendecomposition on CPU (Float64) — this is still faster
#     than pure CPU for large matrix products.
#   • use_fp64_fallback=true (default) routes eigen to CPU.

using LinearAlgebra

# ---------------------------------------------------------------------------
# Runtime availability flag
# ---------------------------------------------------------------------------

const _METAL_LOADED = Ref{Bool}(false)

# Metal.jl support is a package extension (ext/PulsarMetalExt.jl).  The extension
# does `using Metal` (binding `Metal` correctly) and sets
# `_METAL_LOADED[] = Metal.functional()` in its `__init__`.  The core module
# therefore never references a bare `Metal` symbol; the GPU primitive method
# bodies live in the extension.  See the matching note in CUDABackend.jl for the
# precompile-time binding bug this replaces.

# ---------------------------------------------------------------------------
# Type definition
# ---------------------------------------------------------------------------

"""
    MetalBackend

Configuration struct for the Apple Metal GPU backend (Apple Silicon / macOS).

Construct via [`metal_backend`](@ref).  Requires Metal.jl and Apple Silicon
hardware (M1/M2/M3/M4 or an AMD GPU via macOS Metal API).

# Fields
- `device_id::Int`: Metal device index (0-based).
- `memory_limit_gb::Float64`: Soft GPU memory budget in GiB.
- `use_fp32::Bool`: Use Float32 throughout (native Metal precision).
  Faster but accumulates rounding error; suitable for ≤ 6-qubit systems.
- `use_fp64_fallback::Bool`: Route FP64-critical steps (eigendecomposition)
  through CPU.  Recommended `true` (default).
"""
struct MetalBackend <: AbstractComputeBackend
    device_id         :: Int
    memory_limit_gb   :: Float64
    use_fp32          :: Bool
    use_fp64_fallback :: Bool
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    metal_backend(; device_id=0, memory_limit_gb=Inf,
                    use_fp32=false, use_fp64_fallback=true) -> MetalBackend

Construct a [`MetalBackend`](@ref).

Throws an `ErrorException` if Metal.jl is not installed or if the system
is not running on Apple Silicon / macOS with Metal support.
"""
function metal_backend(;
    device_id         :: Int     = 0,
    memory_limit_gb   :: Float64 = Inf,
    use_fp32          :: Bool    = false,
    use_fp64_fallback :: Bool    = true,
)
    if !_METAL_LOADED[]
        error(
            "Metal is not available.  Install Metal.jl with `] add Metal` on " *
            "an Apple Silicon Mac.  Use `cpu_backend()` instead.",
        )
    end
    return MetalBackend(device_id, memory_limit_gb, use_fp32, use_fp64_fallback)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# CPU fallback: Hermitian matrix exponential via eigendecomposition.
# (Retained for use by the Metal extension's CPU-eigendecomposition step.)
function _herm_matexp_metal(H::Matrix{ComplexF64}, dt::Float64)
    F = eigen(Hermitian(H))
    return F.vectors * Diagonal(exp.(-im .* F.values .* dt)) * F.vectors'
end

# ---------------------------------------------------------------------------
# matrix_exponential_metal
# ---------------------------------------------------------------------------

"""
    matrix_exponential_metal(H, dt, backend) -> Matrix{ComplexF64}

Compute `exp(-i H dt)` for a Hermitian matrix `H`.

Strategy:
- Eigendecomposition is always on CPU (Metal lacks FP64 shader support).
- The similarity transform `V * Diagonal(exp(λ)) * V'` uses Metal BLAS
  (FP32) when `backend.use_fp32=true`, otherwise CPU FP64.

The result is always returned as a host `Matrix{ComplexF64}`.
"""
function matrix_exponential_metal(
    H       :: Matrix{ComplexF64},
    dt      :: Float64,
    backend :: MetalBackend,
)
    # Thin wrapper over the batched primitive (1-matrix batch).  The Metal
    # extension's `batched_herm_propagators(::MetalBackend, …)` builds the
    # eigendecomposition on CPU (Metal lacks an FP64 batched eigensolver) and
    # does the scale + batched matmul on device in FP32; otherwise the generic
    # CPU fallback runs.  `Array(…)` downloads if the result is an MtlArray.
    U = batched_herm_propagators(backend, reshape(H, size(H, 1), size(H, 2), 1), dt)
    return Array(U)[:, :, 1]
end

# ---------------------------------------------------------------------------
# batch_propagators_metal
# ---------------------------------------------------------------------------

"""
    batch_propagators_metal(H_array, dt, backend) -> Array{ComplexF64,3}

Compute all propagators `U[k] = exp(-i H_array[:,:,k] dt)` in parallel.

`H_array` is a `(dim, dim, n_steps)` array of Hermitian matrices.

Eigendecompositions are performed on CPU in parallel (Julia threads);
the final similarity transform `V D V'` may use Metal BLAS (FP32) when
`use_fp32=true`.  Returns a host `(dim, dim, n_steps)` array.
"""
function batch_propagators_metal(
    H_array :: AbstractArray{ComplexF64,3},
    dt      :: Float64,
    backend :: MetalBackend,
)
    return Array(batched_herm_propagators(backend, H_array, dt))
end

# ---------------------------------------------------------------------------
# fidelity_metal
# ---------------------------------------------------------------------------

"""
    fidelity_metal(U_total, U_target, backend) -> Float64

Compute the gate fidelity `|Tr(U_target† U_total)|² / dim²`.

The inner product `Tr(U_target† U_total) = sum(conj(U_target) .* U_total)`
is computed on the Metal GPU (FP32) and accumulated on CPU (FP64) when
`use_fp32=true`; otherwise computed on CPU.
"""
function fidelity_metal(
    U_total  :: Matrix{ComplexF64},
    U_target :: Matrix{ComplexF64},
    backend  :: MetalBackend,
)
    # Single gate fidelity — a one-matrix reduction; evaluated on the host.
    dim = size(U_total, 1)
    overlap = tr(U_target' * U_total)
    return abs2(overlap) / dim^2
end

# ---------------------------------------------------------------------------
# gradient_metal
# ---------------------------------------------------------------------------

"""
    gradient_metal(H_drift, H_ctrl, controls, target, dt, backend) -> Matrix{Float64}

Compute the full GRAPE gradient using the Metal GPU backend.

Implements the standard forward–backward GRAPE pass.  Matrix products for
the similarity transforms use Metal BLAS (FP32) when `use_fp32=true`;
eigendecompositions are always on CPU (FP64).

# Arguments
- `H_drift`: Drift Hamiltonian `(dim × dim)`.
- `H_ctrl`: Vector of `n_controls` control Hamiltonians.
- `controls`: `(n_controls × n_steps)` control amplitudes.
- `target`: Target unitary `(dim × dim)`.
- `dt`: Timestep.
- `backend`: The `MetalBackend` instance.

Returns `grad` of size `(n_controls, n_steps)`.
"""
function gradient_metal(
    H_drift  :: Matrix{ComplexF64},
    H_ctrl   :: Vector{<:Matrix{ComplexF64}},
    controls :: Matrix{Float64},
    target   :: Matrix{ComplexF64},
    dt       :: Float64,
    backend  :: MetalBackend,
)
    # Delegates to the device-generic batched GRAPE gate gradient (shared with
    # the CUDA path).  Batched propagators via the Metal primitive override when
    # available; threaded CPU otherwise.
    return _batched_grape_gate_gradient(backend, H_drift, H_ctrl, controls, target, dt)
end

# ---------------------------------------------------------------------------
# metal_info
# ---------------------------------------------------------------------------

"""
    metal_info(backend) -> Dict{String,Any}

Return Metal GPU device properties as a dictionary.

Keys: `"available"`, `"device_name"`, `"memory_gb"`, `"supports_fp64"`.
When Metal is not available, returns `Dict("available"=>false)`.
"""
function metal_info(backend::MetalBackend)::Dict{String,Any}
    if !_METAL_LOADED[]
        return Dict{String,Any}("available" => false)
    end
    try
        dev = Metal.device()
        return Dict{String,Any}(
            "available"    => true,
            "device_name"  => Metal.name(dev),
            "memory_gb"    => Metal.recommendedMaxWorkingSetSize(dev) / 1024^3,
            "supports_fp64" => false,   # Metal shaders are FP32-only
        )
    catch e
        return Dict{String,Any}("available" => true, "error" => string(e))
    end
end
