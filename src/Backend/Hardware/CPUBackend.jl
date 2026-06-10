"""
    CPUBackend.jl

CPU backend for Pulsar.jl with multi-threading and BLAS/LAPACK support.
Provides matrix multiplication, matrix exponentials, and batch propagator
computation using Julia's built-in threading and LinearAlgebra facilities.
"""

using LinearAlgebra
using Base.Threads

# ---------------------------------------------------------------------------
# Struct definition
# ---------------------------------------------------------------------------

"""
    CPUBackend

Configuration for the CPU computation backend.

# Fields
- `num_threads::Int`: Number of Julia threads to use for parallel operations.
  Defaults to `Threads.nthreads()`.
- `use_blas::Bool`: Whether to use BLAS routines for matrix multiplication.
  When `true`, `LinearAlgebra.mul!` dispatches to the system BLAS library
  (OpenBLAS or MKL).  Recommended for dimensions ≥ 32.
- `memory_limit_gb::Float64`: Soft memory budget in GiB.  Operations that
  would exceed this limit will raise an `OutOfMemoryError` before
  allocating.  Set to `Inf` to disable the check.
"""
struct CPUBackend <: AbstractComputeBackend
    num_threads::Int
    use_blas::Bool
    memory_limit_gb::Float64

    function CPUBackend(num_threads::Int, use_blas::Bool, memory_limit_gb::Float64)
        num_threads > 0 || throw(ArgumentError("num_threads must be positive, got $num_threads"))
        memory_limit_gb > 0 || throw(ArgumentError("memory_limit_gb must be positive"))
        new(num_threads, use_blas, memory_limit_gb)
    end
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    cpu_backend(; num_threads=Threads.nthreads(), use_blas=true,
                  memory_limit_gb=Inf) -> CPUBackend

Construct a `CPUBackend` with sensible defaults.

# Keyword Arguments
- `num_threads::Int`: Number of threads.  Defaults to the Julia process
  thread count (set via `JULIA_NUM_THREADS` or `--threads`).
- `use_blas::Bool`: Enable BLAS dispatch for `mul!`.  Default `true`.
- `memory_limit_gb::Float64`: Memory budget in GiB.  Default `Inf`
  (no limit).

# Examples
```julia
backend = cpu_backend()                        # use all available threads
backend = cpu_backend(num_threads=4, use_blas=true, memory_limit_gb=8.0)
```
"""
function cpu_backend(;
    num_threads::Int = Threads.nthreads(),
    use_blas::Bool = true,
    memory_limit_gb::Float64 = Inf,
)::CPUBackend
    return CPUBackend(num_threads, use_blas, memory_limit_gb)
end

# ---------------------------------------------------------------------------
# Memory guard helper
# ---------------------------------------------------------------------------

"""
    _check_memory(backend::CPUBackend, n_bytes::Int)

Throw `OutOfMemoryError` if `n_bytes` exceeds the backend memory budget.
Called before allocating large temporaries.
"""
function _check_memory(backend::CPUBackend, n_bytes::Int)
    limit = backend.memory_limit_gb * 1024^3
    if n_bytes > limit
        error(
            "Requested allocation ($(round(n_bytes / 1024^3, digits=3)) GiB) " *
            "exceeds CPUBackend memory limit ($(backend.memory_limit_gb) GiB).",
        )
    end
end

# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

"""
    matrix_multiply_cpu(A::AbstractMatrix, B::AbstractMatrix,
                        backend::CPUBackend) -> Matrix

Compute the matrix product `C = A * B` using the CPU backend.

When `backend.use_blas` is `true` the multiplication is delegated to
`LinearAlgebra.mul!` which dispatches to the system BLAS library.
Otherwise a plain Julia `*` is used (still multi-threaded via BLAS
internally on most installations).

# Arguments
- `A`: Left operand.  Any element type supported by `LinearAlgebra`.
- `B`: Right operand.  Must be conformable with `A`.
- `backend`: `CPUBackend` configuration.

# Returns
Dense `Matrix{T}` where `T` is promoted from the element types of `A`
and `B`.

# Throws
- `DimensionMismatch` if `size(A, 2) ≠ size(B, 1)`.
- `OutOfMemoryError` if result would exceed memory budget.
"""
function matrix_multiply_cpu(
    A::AbstractMatrix,
    B::AbstractMatrix,
    backend::CPUBackend,
)::Matrix
    m, k1 = size(A)
    k2, n = size(B)
    k1 == k2 || throw(DimensionMismatch(
        "matrix multiply: A is $(m)×$(k1) but B is $(k2)×$(n)",
    ))

    T = promote_type(eltype(A), eltype(B))
    _check_memory(backend, m * n * sizeof(T))

    if backend.use_blas
        C = Matrix{T}(undef, m, n)
        mul!(C, A, B)
        return C
    else
        return A * B
    end
end

# ---------------------------------------------------------------------------
# Matrix exponential via eigendecomposition
# ---------------------------------------------------------------------------

"""
    matrix_exponential_cpu(H::AbstractMatrix{ComplexF64}, dt::Float64,
                           backend::CPUBackend) -> Matrix{ComplexF64}

Compute the unitary propagator `U = exp(-i * H * dt)` for a Hermitian
Hamiltonian `H`.

The implementation uses LAPACK's divide-and-conquer eigensolver
(`LinearAlgebra.eigen`) which is numerically stable and accurate for
Hermitian matrices.  The decomposition is `H = V Λ V†`, giving
`exp(-i H dt) = V * diag(exp(-i λ_k dt)) * V†`.

# Arguments
- `H`: Hermitian Hamiltonian matrix of dimension `d × d`.
- `dt`: Time step in natural units.
- `backend`: `CPUBackend` configuration.

# Returns
Unitary matrix `U ∈ ℂ^{d×d}`.

# Notes
- `H` must be Hermitian.  A symmetrisation `(H + H') / 2` is applied
  silently to guard against small floating-point asymmetries.
- For very small dimensions (`d ≤ 4`) the Cayley-Hamilton or Padé path
  through `exp` may be faster, but the eigendecomposition path is used
  uniformly for numerical consistency.
"""
function matrix_exponential_cpu(
    H::AbstractMatrix{ComplexF64},
    dt::Float64,
    backend::CPUBackend,
)::Matrix{ComplexF64}
    d = size(H, 1)
    size(H, 2) == d || throw(DimensionMismatch("H must be square, got $(size(H))"))

    # Symmetrise to guard against numerical non-Hermiticity
    H_herm = Hermitian((H .+ H') ./ 2)

    # LAPACK eigen for Hermitian matrices (uses dsyevd / zheevd)
    F = eigen(H_herm)          # F.values::Vector{Float64}, F.vectors::Matrix{ComplexF64}
    λ = F.values               # real eigenvalues
    V = F.vectors              # unitary eigenvector matrix

    # Phase factors exp(-i λ_k dt)
    phases = exp.(((-1im * dt) .* λ))

    # U = V * diag(phases) * V†
    # Avoid forming full diagonal matrix; use broadcasting
    U = V * (phases .* V')
    return U
end

# ---------------------------------------------------------------------------
# Batch propagator computation
# ---------------------------------------------------------------------------

"""
    batch_propagators_cpu(H_array::AbstractArray{ComplexF64,3}, dt::Float64,
                          backend::CPUBackend) -> Array{ComplexF64,3}

Compute the unitary propagators `U[k] = exp(-i H[k] dt)` for each
time step `k` in parallel using Julia threads.

# Arguments
- `H_array`: Array of shape `(n_timesteps, d, d)` containing Hamiltonians.
  `H_array[k, :, :]` is the Hamiltonian at time step `k`.
- `dt`: Time step (scalar, same for all steps).
- `backend`: `CPUBackend` configuration.

# Returns
Array of shape `(n_timesteps, d, d)` containing the propagators.

# Notes
- Thread count is governed by `backend.num_threads`.  The implementation
  uses `Threads.@threads` with the `:static` scheduler for deterministic
  thread assignment.
- Memory estimate: `n_timesteps × d² × 16` bytes (ComplexF64).
"""
function batch_propagators_cpu(
    H_array::AbstractArray{ComplexF64,3},
    dt::Float64,
    backend::CPUBackend,
)::Array{ComplexF64,3}
    n_timesteps, d, d2 = size(H_array)
    d == d2 || throw(DimensionMismatch("H_array slices must be square: got $(d)×$(d2)"))

    _check_memory(backend, n_timesteps * d * d * sizeof(ComplexF64))

    U_array = Array{ComplexF64,3}(undef, n_timesteps, d, d)

    BLAS.set_num_threads(max(1, backend.num_threads ÷ max(1, Threads.nthreads())))

    @threads :static for k in 1:n_timesteps
        H_k = @view H_array[k, :, :]
        U_k = matrix_exponential_cpu(Matrix{ComplexF64}(H_k), dt, backend)
        U_array[k, :, :] .= U_k
    end

    return U_array
end

# ---------------------------------------------------------------------------
# CPU info
# ---------------------------------------------------------------------------

"""
    cpu_info() -> Dict{String, Any}

Return a dictionary describing the current CPU capabilities and Julia
thread configuration.

# Keys
- `"num_threads"`: Number of Julia threads (`Threads.nthreads()`).
- `"cpu_name"`: CPU model string from `Sys.CPU_NAME`.
- `"cpu_threads"`: Hardware logical core count (`Sys.CPU_THREADS`).
- `"memory_total_gb"`: Total RAM in GiB (`Sys.total_memory()`).
- `"memory_free_gb"`: Free RAM in GiB (`Sys.free_memory()`).
- `"simd_width"`: Detected SIMD register width in bits (AVX512→512,
  AVX2→256, SSE2/NEON→128, generic→64).
- `"blas_vendor"`: BLAS library name (e.g. `"openblas"`, `"mkl"`).
- `"blas_threads"`: Number of threads BLAS is currently using.
- `"julia_version"`: Julia version string.

# Examples
```julia
info = cpu_info()
println("Threads: ", info["num_threads"])
println("CPU: ", info["cpu_name"])
```
"""
function cpu_info()::Dict{String,Any}
    cpu_name = string(Sys.CPU_NAME)

    # Heuristic SIMD detection via CPU name
    simd_width = if occursin("avx512", lowercase(cpu_name))
        512
    elseif occursin("avx2", lowercase(cpu_name)) || occursin("avx", lowercase(cpu_name))
        256
    elseif occursin("neon", lowercase(cpu_name)) || occursin("apple", lowercase(cpu_name))
        128
    elseif Sys.ARCH === :x86_64
        128   # SSE2 baseline on x86_64
    else
        64
    end

    blas_vendor = try
        string(BLAS.vendor())
    catch
        "unknown"
    end

    blas_threads = try
        BLAS.get_num_threads()
    catch
        -1
    end

    return Dict{String,Any}(
        "num_threads"     => Threads.nthreads(),
        "cpu_name"        => cpu_name,
        "cpu_threads"     => Sys.CPU_THREADS,
        "memory_total_gb" => Sys.total_memory() / 1024^3,
        "memory_free_gb"  => Sys.free_memory() / 1024^3,
        "simd_width"      => simd_width,
        "blas_vendor"     => blas_vendor,
        "blas_threads"    => blas_threads,
        "julia_version"   => string(VERSION),
    )
end
