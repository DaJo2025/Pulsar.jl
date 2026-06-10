module PulsarCUDAExt

# NVIDIA CUDA GPU methods for Pulsar's batched primitive layer.
#
# `using CUDA` here binds `CUDA` correctly (the historical precompile-time
# `@eval using CUDA` bug is gone — see Backend/Hardware/CUDABackend.jl).  This
# module is loaded automatically when the user has `import CUDA` and `using
# Pulsar`, via the [extensions] entry in Project.toml.
#
# It overrides the generic (CPU-fallback) primitive methods from
# Backend/BatchedPrimitives.jl with genuinely batched device kernels:
#   • batched_herm_propagators  — heevjBatched + gemm_strided_batched  (D ≤ 32)
#   • batched_nonherm_propagators — batched scaling-and-squaring (Lindblad)
#   • batched_build_hamiltonian — one on-device gemm for all N control H's
#   • batched_matvec            — gemm_strided_batched forward/backward sweeps
#
# Every device method is wrapped so that an unsupported op or device error
# degrades to the threaded CPU implementation with a @warn, never a crash.

using CUDA
using CUDA.CUSOLVER
using CUDA.CUBLAS
using LinearAlgebra
using Pulsar

import Pulsar: batched_herm_propagators, batched_nonherm_propagators,
               batched_build_hamiltonian, batched_matvec,
               _batched_herm_cpu, _batched_nonherm_cpu,
               _batched_build_ham_cpu, _batched_matvec_cpu,
               batched_chunk_size, CUDABackend, _CUDA_LOADED

# Heuristic upper bound on the Jacobi batched Hermitian eigensolver dimension.
const _HEEVJ_MAX_DIM = 32

function __init__()
    # Authoritative flag: only true when a functional NVIDIA GPU is present.
    try
        _CUDA_LOADED[] = CUDA.functional()
    catch
        _CUDA_LOADED[] = false
    end
    return nothing
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# GPU memory budget (bytes) for a chunk, honouring backend.memory_limit_gb and
# the device's currently-available memory (with a safety margin).
function _cuda_mem_budget(backend::CUDABackend)
    avail = try
        Float64(CUDA.available_memory())
    catch
        Inf
    end
    lim = isfinite(backend.memory_limit_gb) ? backend.memory_limit_gb * 1024^3 : Inf
    return 0.6 * min(avail, lim)
end

_to_cu(::Type{T}, x::AbstractArray) where {T} =
    x isa CuArray ? x : CuArray(convert(Array{T}, Array(x)))

# ---------------------------------------------------------------------------
# 1. batched_herm_propagators — exp(-i H dt), Hermitian batch (the 98× kernel)
# ---------------------------------------------------------------------------

# Batched eigendecomposition of one chunk H::(D,D,n) on device, returning U.
function _cuda_herm_chunk(H::CuArray{ComplexF64,3}, dtv::Vector{Float64})
    D, _, n = size(H)
    A    = copy(H)                                   # heevjBatched! overwrites A
    W, V = CUSOLVER.heevjBatched!('V', 'U', A)       # W:(D,n) real eigvals, V eigvecs
    dt_d = CuArray(reshape(dtv, 1, n))               # (1,n)
    expλ = exp.((-im) .* dt_d .* W)                  # (D,n) ComplexF64
    M    = V .* reshape(expλ, 1, D, n)               # scale eigenvector columns
    U    = CuArray{ComplexF64}(undef, D, D, n)
    CUBLAS.gemm_strided_batched!('N', 'C', ComplexF64(1), M, V, ComplexF64(0), U)
    return U
end

function batched_herm_propagators(backend::CUDABackend,
                                  H::AbstractArray{<:Complex,3}, dt)
    D, D2, N = size(H)
    D == D2 || throw(DimensionMismatch("H must be (D,D,N), got $(size(H))"))
    # Larger than the batched Jacobi eigensolver supports → CPU per-matrix path.
    if D > _HEEVJ_MAX_DIM
        return _batched_herm_cpu(H, dt)
    end
    try
        dtv   = Pulsar._batched_dt_vector(dt, N)
        Hd    = _to_cu(ComplexF64, H)
        chunk = batched_chunk_size(D, N; mem_budget_bytes = _cuda_mem_budget(backend))
        if chunk >= N
            return _cuda_herm_chunk(Hd, dtv)
        end
        U = CuArray{ComplexF64}(undef, D, D, N)
        @views for lo in 1:chunk:N
            hi = min(lo + chunk - 1, N)
            U[:, :, lo:hi] .= _cuda_herm_chunk(Hd[:, :, lo:hi], dtv[lo:hi])
        end
        return U
    catch e
        @warn "PulsarCUDAExt.batched_herm_propagators: GPU path failed ($e); CPU fallback." maxlog = 1
        return _batched_herm_cpu(H, dt)
    end
end

# ---------------------------------------------------------------------------
# 2. batched_nonherm_propagators — exp(A dt) via batched scaling & squaring
#    (Lindblad: A is the non-Hermitian Liouvillian; heevjBatched cannot apply.)
# ---------------------------------------------------------------------------

# Degree-`order` Taylor of exp(B) for a batch B::(M,M,n) via batched gemm
# (Horner), followed by `s` batched squarings.  B = A·dt / 2^s with `s` chosen
# from the batch max 1-norm so ‖B‖₁ ≲ 0.5.
function _cuda_nonherm_chunk(A::CuArray{ComplexF64,3}, dtv::Vector{Float64};
                             order::Int = 13)
    M, _, n = size(A)
    dt_d = CuArray(reshape(dtv, 1, 1, n))
    B    = A .* dt_d                                  # (M,M,n)
    # batch max 1-norm = max over columns of sum_i |B[i,j,n]|
    nrm  = maximum(sum(abs.(B); dims = 1))
    s    = max(0, ceil(Int, log2(max(nrm, 1e-300))) + 1)
    B  ./= ComplexF64(2.0^s)

    Ibatch = CuArray{ComplexF64}(undef, M, M, n)
    Ieye   = CuMatrix{ComplexF64}(I, M, M)
    @views for k in 1:n
        Ibatch[:, :, k] .= Ieye
    end
    # Horner: E = I + B/order (I + B/(order-1) (… (I + B)))
    E   = copy(Ibatch)
    tmp = CuArray{ComplexF64}(undef, M, M, n)
    for k in order:-1:1
        # tmp = B * E ; E = I + tmp / k
        CUBLAS.gemm_strided_batched!('N', 'N', ComplexF64(1), B, E, ComplexF64(0), tmp)
        E .= Ibatch .+ tmp ./ ComplexF64(k)
    end
    # square s times
    for _ in 1:s
        CUBLAS.gemm_strided_batched!('N', 'N', ComplexF64(1), E, E, ComplexF64(0), tmp)
        E, tmp = tmp, E
    end
    return E
end

function batched_nonherm_propagators(backend::CUDABackend,
                                     A::AbstractArray{<:Number,3}, dt)
    M, M2, N = size(A)
    M == M2 || throw(DimensionMismatch("A must be (M,M,N), got $(size(A))"))
    try
        dtv   = Pulsar._batched_dt_vector(dt, N)
        Ad    = _to_cu(ComplexF64, A)
        # scaling-and-squaring needs ~3 (M,M,chunk) buffers simultaneously
        chunk = batched_chunk_size(M, N; mem_budget_bytes = _cuda_mem_budget(backend),
                                   scratch_factor = 6.0)
        if chunk >= N
            return _cuda_nonherm_chunk(Ad, dtv)
        end
        U = CuArray{ComplexF64}(undef, M, M, N)
        @views for lo in 1:chunk:N
            hi = min(lo + chunk - 1, N)
            U[:, :, lo:hi] .= _cuda_nonherm_chunk(Ad[:, :, lo:hi], dtv[lo:hi])
        end
        return U
    catch e
        @warn "PulsarCUDAExt.batched_nonherm_propagators: GPU path failed ($e); CPU fallback." maxlog = 1
        return _batched_nonherm_cpu(A, dt)
    end
end

# ---------------------------------------------------------------------------
# 3. batched_build_hamiltonian — one on-device gemm for all N control H's
# ---------------------------------------------------------------------------

function batched_build_hamiltonian(backend::CUDABackend, drift,
                                   ops::AbstractVector{<:AbstractMatrix},
                                   coeff::AbstractMatrix)
    try
        n_ctrl, N = size(coeff)
        D = size(ops[1], 1)
        O = CuArray{ComplexF64}(undef, D * D, n_ctrl)
        for k in 1:n_ctrl
            @views O[:, k] .= CuArray(vec(convert(Matrix{ComplexF64}, Matrix(ops[k]))))
        end
        C  = _to_cu(ComplexF64, ComplexF64.(coeff))           # (n_ctrl, N)
        Hc = reshape(O * C, D, D, N)                          # control part
        if ndims(drift) == 2
            Hd = _to_cu(ComplexF64, ComplexF64.(drift))       # (D,D)
            return Hc .+ reshape(Hd, D, D, 1)
        else
            Hd = _to_cu(ComplexF64, ComplexF64.(drift))       # (D,D,N)
            return Hc .+ Hd
        end
    catch e
        @warn "PulsarCUDAExt.batched_build_hamiltonian: GPU path failed ($e); CPU fallback." maxlog = 1
        return _batched_build_ham_cpu(drift, ops, coeff)
    end
end

# ---------------------------------------------------------------------------
# 4. batched_matvec — gemm_strided_batched for forward/backward sweeps
# ---------------------------------------------------------------------------

function batched_matvec(backend::CUDABackend,
                        U::AbstractArray{<:Number,3},
                        X::AbstractArray{<:Number,3}; adjoint::Bool=false)
    try
        D, _, N = size(U)
        K = size(X, 2)
        Ud = _to_cu(ComplexF64, U)
        Xd = _to_cu(ComplexF64, X)
        Y  = CuArray{ComplexF64}(undef, D, K, N)
        ta = adjoint ? 'C' : 'N'
        CUBLAS.gemm_strided_batched!(ta, 'N', ComplexF64(1), Ud, Xd, ComplexF64(0), Y)
        return Y
    catch e
        @warn "PulsarCUDAExt.batched_matvec: GPU path failed ($e); CPU fallback." maxlog = 1
        return _batched_matvec_cpu(U, X; adjoint=adjoint)
    end
end

end  # module PulsarCUDAExt
