# Backend/BatchedPrimitives.jl — device-dispatched batched primitive layer
#
# This is the single seam that lets GPU batching reach *every* optimizer,
# fidelity metric, ensemble-averaging mode, and application in Pulsar.  Each
# primitive is a generic function with a CPU fallback method defined here (on
# `::AbstractComputeBackend`, so it works for *any* backend — including a
# CUDABackend/MetalBackend on a machine where the GPU extension is not loaded).
# The GPU extensions (`ext/PulsarCUDAExt.jl`, `ext/PulsarMetalExt.jl`) add
# specialised methods on `::CUDABackend` / `::MetalBackend` that run the genuinely
# batched device kernels (`heevjBatched` + `gemm_strided_batched`).
#
# Layout convention for the matrix primitives: a batch of `N` matrices is a
# dense `(D, D, N)` array; the trailing dimension is the batch axis (it fuses,
# e.g., ensemble-member × timestep).  `dt` is either a scalar (shared by all
# `N`) or a length-`N` vector (per-matrix, e.g. a non-uniform `pulse_dt`).
#
# Primitives provided here:
#   batched_herm_propagators     U = exp(-i H dt)         (Hermitian H)
#   batched_nonherm_propagators  U = exp(A dt)            (general A, e.g. Liouvillian)
#   batched_build_hamiltonian    H = drift + Σ_k coeff_k op_k   (one gemm)
#   batched_matvec               Y[:,:,n] = U[:,:,n] (Xᴴ?) X[:,:,n]
#   batched_chunk_size           memory-budgeted chunk count for the batch axis
#   resolve_backend              :cpu/:cuda/:metal symbol → backend struct
#   _batched_grape_gate_gradient device-generic GRAPE gate gradient (reference)

using LinearAlgebra

# ---------------------------------------------------------------------------
# dt normalisation
# ---------------------------------------------------------------------------

@inline _batched_dt_vector(dt::Real, N::Int) = fill(Float64(dt), N)
@inline function _batched_dt_vector(dt::AbstractVector{<:Real}, N::Int)
    length(dt) == N || throw(DimensionMismatch(
        "dt vector length $(length(dt)) ≠ batch size $N"))
    return collect(Float64, dt)
end

# ---------------------------------------------------------------------------
# 1. batched_herm_propagators — U = exp(-i H dt) for a Hermitian batch
# ---------------------------------------------------------------------------

"""
    batched_herm_propagators(backend, H, dt) -> U

Compute `U[:,:,n] = exp(-i · H[:,:,n] · dt_n)` for a batch of `N` Hermitian
matrices `H::(D, D, N)`.  `dt` is a scalar (shared) or a length-`N` vector.

The generic method here is the **CPU fallback** (threaded LAPACK
eigendecomposition via [`_expm_neg_i_into!`]).  The CUDA extension overrides this
for `::CUDABackend` with a single batched `heevjBatched` + `gemm_strided_batched`
(valid for `D ≤ 32`; larger `D` falls back per-matrix), chunked to a GPU memory
budget.  Returns a `(D, D, N)` array on the backend's native device (a host
`Array` here; a `CuArray`/`MtlArray` from the GPU methods).
"""
batched_herm_propagators(::AbstractComputeBackend,
                         H::AbstractArray{<:Complex,3}, dt) =
    _batched_herm_cpu(H, dt)

"""
    _batched_herm_cpu(H, dt) -> U

Threaded LAPACK-eigendecomposition CPU implementation of
[`batched_herm_propagators`].  Also the on-error fallback used by the GPU
extensions, so they never crash a job when a device op is unavailable.
"""
function _batched_herm_cpu(H::AbstractArray{<:Complex,3}, dt)
    D, D2, N = size(H)
    D == D2 || throw(DimensionMismatch("H must be (D,D,N), got $(size(H))"))
    dtv = _batched_dt_vector(dt, N)
    U   = Array{ComplexF64,3}(undef, D, D, N)

    nth  = Threads.maxthreadid()
    Ubuf = [Matrix{ComplexF64}(undef, D, D) for _ in 1:nth]
    Hbuf = [Matrix{ComplexF64}(undef, D, D) for _ in 1:nth]
    Tbuf = [Matrix{ComplexF64}(undef, D, D) for _ in 1:nth]

    old = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        Threads.@threads :static for n in 1:N
            tid = Threads.threadid()
            @inbounds @views Hbuf[tid] .= H[:, :, n]
            _expm_neg_i_into!(Ubuf[tid], Hbuf[tid], dtv[n], Tbuf[tid])
            @inbounds @views U[:, :, n] .= Ubuf[tid]
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(old)
    end
    return U
end

# ---------------------------------------------------------------------------
# 2. batched_nonherm_propagators — U = exp(A dt) for a general (non-Hermitian)
#    batch.  Used by the Lindblad path, where A is the Liouvillian superoperator
#    (dim D², already including the −i coherent factor) and U = exp(𝓛 dt).
# ---------------------------------------------------------------------------

"""
    batched_nonherm_propagators(backend, A, dt) -> U

Compute `U[:,:,n] = exp(A[:,:,n] · dt_n)` for a batch of `N` general (possibly
non-Hermitian) matrices `A::(M, M, N)`.  `dt` is scalar or length-`N`.

Generic method = CPU fallback (LAPACK `exp` per slice).  The GPU extensions
override with batched scaling-and-squaring via `gemm_strided_batched`, chunked
to a memory budget (`exp` of a non-Hermitian matrix cannot use the batched
Hermitian eigensolver).  Used for open-system (Lindblad) propagators.
"""
batched_nonherm_propagators(::AbstractComputeBackend,
                            A::AbstractArray{<:Number,3}, dt) =
    _batched_nonherm_cpu(A, dt)

"""
    _batched_nonherm_cpu(A, dt) -> U

Threaded LAPACK `exp` CPU implementation of [`batched_nonherm_propagators`];
also the GPU extensions' on-error fallback.
"""
function _batched_nonherm_cpu(A::AbstractArray{<:Number,3}, dt)
    M, M2, N = size(A)
    M == M2 || throw(DimensionMismatch("A must be (M,M,N), got $(size(A))"))
    dtv = _batched_dt_vector(dt, N)
    U   = Array{ComplexF64,3}(undef, M, M, N)
    Threads.@threads :static for n in 1:N
        @inbounds @views U[:, :, n] .= exp(Matrix{ComplexF64}(A[:, :, n]) .* dtv[n])
    end
    return U
end

# ---------------------------------------------------------------------------
# 3. batched_build_hamiltonian — H[:,:,n] = drift_n + Σ_k coeff[k,n] · op_k
#    Builds the *entire* batch with a single gemm:  reshape(ops, D², n_ctrl) is
#    contracted with coeff::(n_ctrl, N).  The GPU extension runs the same gemm
#    on-device so the Hamiltonian batch is assembled without a host round-trip.
# ---------------------------------------------------------------------------

"""
    batched_build_hamiltonian(backend, drift, ops, coeff) -> H

Assemble `H[:,:,n] = drift(_n) + Σ_k coeff[k,n] · ops[k]` for all `N` columns at
once.  `drift` is either a shared `(D,D)` matrix or a per-column `(D,D,N)` array;
`ops` is a length-`n_ctrl` vector of `(D,D)` matrices; `coeff::(n_ctrl, N)`.

The control part is one matrix–matrix product `reshape(ops, D², n_ctrl) · coeff`,
i.e. all `N` control Hamiltonians in a single gemm — the key to keeping the GPU
busy instead of launching `N` tiny kernels.
"""
batched_build_hamiltonian(::AbstractComputeBackend, drift,
                          ops::AbstractVector{<:AbstractMatrix},
                          coeff::AbstractMatrix) =
    _batched_build_ham_cpu(drift, ops, coeff)

"""
    _batched_build_ham_cpu(drift, ops, coeff) -> H

Single-gemm CPU implementation of [`batched_build_hamiltonian`]; also the GPU
extensions' on-error fallback.
"""
function _batched_build_ham_cpu(drift,
                                ops::AbstractVector{<:AbstractMatrix},
                                coeff::AbstractMatrix)
    n_ctrl, N = size(coeff)
    D = size(ops[1], 1)
    O = Array{ComplexF64,2}(undef, D * D, n_ctrl)
    @inbounds for k in 1:n_ctrl
        @views O[:, k] .= vec(ops[k])
    end
    Hc  = reshape(O * ComplexF64.(coeff), D, D, N)   # control part, (D,D,N)
    H   = Array{ComplexF64,3}(undef, D, D, N)
    if ndims(drift) == 2
        @inbounds for n in 1:N
            @views H[:, :, n] .= Hc[:, :, n] .+ drift
        end
    else
        @inbounds for n in 1:N
            @views H[:, :, n] .= Hc[:, :, n] .+ drift[:, :, n]
        end
    end
    return H
end

# ---------------------------------------------------------------------------
# 4. batched_matvec — Y[:,:,n] = op(U[:,:,n]) · X[:,:,n]
#    With `adjoint=true`, applies U[:,:,n]ᴴ.  Drives the forward/backward sweeps
#    (X carries K state/co-state columns propagated together).  GPU extension =
#    `gemm_strided_batched`.
# ---------------------------------------------------------------------------

"""
    batched_matvec(backend, U, X; adjoint=false) -> Y

Batched matrix product `Y[:,:,n] = U[:,:,n] · X[:,:,n]` (or `U[:,:,n]ᴴ · X[:,:,n]`
when `adjoint=true`), with `U::(D,D,N)` and `X::(D,K,N)`.  `K` columns are
propagated together (e.g. all state pairs of an ensemble member).  GPU extension
maps this onto a single `gemm_strided_batched`.
"""
batched_matvec(::AbstractComputeBackend, U::AbstractArray{<:Number,3},
               X::AbstractArray{<:Number,3}; adjoint::Bool=false) =
    _batched_matvec_cpu(U, X; adjoint=adjoint)

"""
    _batched_matvec_cpu(U, X; adjoint=false) -> Y

Threaded CPU implementation of [`batched_matvec`]; also the GPU extensions'
on-error fallback.
"""
function _batched_matvec_cpu(U::AbstractArray{<:Number,3},
                             X::AbstractArray{<:Number,3}; adjoint::Bool=false)
    D, _, N = size(U)
    K = size(X, 2)
    size(X, 3) == N || throw(DimensionMismatch("U and X batch sizes differ"))
    Y = Array{ComplexF64,3}(undef, D, K, N)
    Threads.@threads :static for n in 1:N
        @inbounds @views if adjoint
            mul!(Y[:, :, n], U[:, :, n]', X[:, :, n])
        else
            mul!(Y[:, :, n], U[:, :, n], X[:, :, n])
        end
    end
    return Y
end

# ---------------------------------------------------------------------------
# 5. batched_chunk_size — memory-budgeted chunk count along the batch axis
# ---------------------------------------------------------------------------

"""
    batched_chunk_size(D, N; eltype=ComplexF64, mem_budget_bytes=Inf,
                       scratch_factor=4.0) -> chunk

Largest batch-axis chunk `≤ N` whose working set fits `mem_budget_bytes`.  Each
`(D, D, chunk)` array is `sizeof(eltype)·D²·chunk` bytes; `scratch_factor`
accounts for the eigenvector/eigenvalue/output/workspace copies a batched
eigensolve needs simultaneously (≈3–4×).  Returns `N` when the budget is `Inf`.
"""
function batched_chunk_size(D::Int, N::Int; eltype::Type=ComplexF64,
                            mem_budget_bytes::Real=Inf, scratch_factor::Real=4.0)
    (isfinite(mem_budget_bytes) && mem_budget_bytes > 0) || return N
    bytes_per_matrix = sizeof(eltype) * D * D * scratch_factor
    chunk = max(1, Int(floor(mem_budget_bytes / bytes_per_matrix)))
    return min(chunk, N)
end

# ---------------------------------------------------------------------------
# 6. resolve_backend — map a device symbol to a backend struct (with fallback)
# ---------------------------------------------------------------------------

"""
    resolve_backend(device=get_device(); kwargs...) -> AbstractComputeBackend

Return the backend struct for `device` (`:cpu`, `:cuda`, `:metal`), falling back
to [`cpu_backend`](@ref) when the requested GPU package/extension is not loaded.
Keyword arguments are forwarded to the chosen backend constructor.
"""
function resolve_backend(device::Symbol=get_device(); kwargs...)
    if device === :cuda && _CUDA_LOADED[]
        return cuda_backend(; kwargs...)
    elseif device === :metal && _METAL_LOADED[]
        return metal_backend(; kwargs...)
    else
        return cpu_backend(; kwargs...)
    end
end

# ---------------------------------------------------------------------------
# 7. BatchedWorkspace — reusable device buffers keyed by batch shape
# ---------------------------------------------------------------------------

"""
    BatchedWorkspace

Opaque holder for preallocated device arrays (Hamiltonian batch, eigenvectors,
eigenvalues, propagators, state/co-state buffers) reused across optimizer
iterations to avoid reallocating GPU memory every gradient evaluation.  Buffers
are created lazily by the GPU extensions and keyed by `(typeof(backend), D,
N_chunk, eltype)` in the module-level [`_BATCHED_WORKSPACES`] cache; a shape
change invalidates and reallocates.  On CPU this is a no-op placeholder.
"""
mutable struct BatchedWorkspace
    key::Any
    buffers::Dict{Symbol,Any}
end
BatchedWorkspace(key) = BatchedWorkspace(key, Dict{Symbol,Any}())

const _BATCHED_WORKSPACES = Dict{Any,BatchedWorkspace}()

"""
    get_workspace(key) -> BatchedWorkspace

Fetch (or lazily create) the cached [`BatchedWorkspace`] for `key`.  Used by the
GPU extensions to recycle device buffers across calls of the same shape.
"""
get_workspace(key) = get!(() -> BatchedWorkspace(key), _BATCHED_WORKSPACES, key)

"""
    clear_workspaces!()

Drop all cached [`BatchedWorkspace`] buffers (frees the device memory they hold
once the GPU arrays are garbage-collected).  Call after a large job to release
GPU memory, or when changing problem shape repeatedly.
"""
function clear_workspaces!()
    empty!(_BATCHED_WORKSPACES)
    return nothing
end

# ---------------------------------------------------------------------------
# 8. _batched_grape_gate_gradient — device-generic GRAPE gate gradient
#
# Reference implementation expressed entirely in terms of the primitives above,
# so it runs on CPU (generic methods) or GPU (extension methods) unchanged.  The
# expensive propagator batch is built with one batched eigensolve; the
# forward/backward chains are sequential (inherent data dependence) but operate
# on whichever device the propagators live; the per-(k,j) trace reductions are
# hoisted out of the time loop into the precomputed `A_k = P_k · Q_{k+1}ᴴ`.
# ---------------------------------------------------------------------------

"""
    _batched_grape_gate_gradient(backend, H_drift, H_ctrl, controls, target, dt)
        -> grad::(n_ctrl, n_steps)

Full GRAPE gate-fidelity gradient `∂F/∂u` for `F = |Tr(U_target† U)|²/D²`,
written through the batched primitives.  `H_ctrl` is a vector of control
Hamiltonians, `controls::(n_ctrl, n_steps)`.  Used by the legacy
`gradient_cuda` / `gradient_metal` wrappers; the GPU extensions accelerate it
automatically via their primitive overrides.
"""
function _batched_grape_gate_gradient(backend::AbstractComputeBackend,
                                      H_drift::AbstractMatrix{ComplexF64},
                                      H_ctrl::AbstractVector{<:AbstractMatrix{ComplexF64}},
                                      controls::AbstractMatrix{Float64},
                                      target::AbstractMatrix{ComplexF64},
                                      dt::Float64)
    n_ctrl, n_steps = size(controls)
    D = size(H_drift, 1)

    # Build all step Hamiltonians and propagators as one batch (the GPU win).
    H = batched_build_hamiltonian(backend, Matrix{ComplexF64}(H_drift),
                                  Matrix{ComplexF64}.(H_ctrl), controls)
    U = Array(batched_herm_propagators(backend, H, dt))   # (D,D,n_steps) on host

    # Forward propagators P[k] (P[1]=I) and backward co-states Q[k] (Q[N+1]=U_t†).
    P = Array{ComplexF64,3}(undef, D, D, n_steps + 1)
    Q = Array{ComplexF64,3}(undef, D, D, n_steps + 1)
    @views P[:, :, 1] .= Matrix{ComplexF64}(I, D, D)
    @inbounds for k in 1:n_steps
        @views mul!(P[:, :, k + 1], U[:, :, k], P[:, :, k])
    end
    @views Q[:, :, n_steps + 1] .= target'
    @inbounds for k in n_steps:-1:1
        @views mul!(Q[:, :, k], Q[:, :, k + 1], U[:, :, k])
    end

    Φ    = tr(target' * @view(P[:, :, n_steps + 1])) / D
    grad = zeros(Float64, n_ctrl, n_steps)
    pref = 2.0 / D^2
    @inbounds for k in 1:n_steps
        # A_k = P[k] · Q[k+1]ᴴ  →  Tr(Q[k+1]ᴴ (−i dt H_j) P[k]) = −i dt Tr(H_j A_k)
        A_k = @views P[:, :, k] * Q[:, :, k + 1]'
        for j in 1:n_ctrl
            Hj = H_ctrl[j]
            s = zero(ComplexF64)
            for q in 1:D, p in 1:D
                s += Hj[p, q] * A_k[q, p]
            end
            trM = -im * dt * s
            grad[j, k] = pref * real(conj(Φ) * trM)
        end
    end
    return grad
end
