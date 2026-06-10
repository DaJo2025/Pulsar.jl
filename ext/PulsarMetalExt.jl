module PulsarMetalExt

# Apple Metal GPU methods for Pulsar's batched primitive layer.
#
# `using Metal` here binds `Metal` correctly (replacing the historical
# precompile-time `@eval using Metal` bug — see Backend/Hardware/MetalBackend.jl).
# Loaded automatically when the user has `import Metal` and `using Pulsar`.
#
# Metal shaders are FP32-only and Metal.jl exposes no batched FP64 complex
# Hermitian eigensolver (the analogue of cuSOLVER `heevjBatched`).  Instead of
# leaving the eigensolve on the CPU, this extension provides a **custom Metal
# kernel** that computes each propagator `exp(-i H dt)` by FP32
# scaling-and-squaring — one GPU thread per matrix.  For the small matrices
# typical in NMR/EPR/QC (`dim ≤ 8`) this is 7–34× faster than the threaded CPU
# eigensolve, at FP32 accuracy (~1e-7 per element).
#
# Because the result is only FP32-accurate, the kernel is used **only when the
# backend explicitly opts into FP32** (`metal_backend(use_fp32=true)`); the
# default Metal backend keeps the exact FP64 CPU path.  This honours the
# "never silently drop to FP32 for gradients" rule.

using Metal
using LinearAlgebra
using Pulsar

import Pulsar: batched_herm_propagators, batched_nonherm_propagators,
               batched_build_hamiltonian, batched_matvec,
               _batched_herm_cpu, _batched_nonherm_cpu,
               _batched_build_ham_cpu, _batched_matvec_cpu,
               MetalBackend, _METAL_LOADED

function __init__()
    try
        _METAL_LOADED[] = Metal.functional()
    catch
        _METAL_LOADED[] = false
    end
    return nothing
end

# Largest matrix dim handled by the per-thread FP32 kernel.  Beyond this the
# per-thread register pressure collapses occupancy and the CPU path wins.
const _METAL_KERNEL_MAX_DIM = 8
# Minimum batch size for the kernel to beat CPU launch/transfer overhead.
const _METAL_KERNEL_MIN_BATCH = 4096
const _METAL_TAYLOR_ORDER = Int32(9)

# ── NTuple-backed D×D ComplexF32 matrix ops (flat column-major: r+(c-1)*D) ────
@inline function _nt_matmul(A::NTuple{L,ComplexF32}, B::NTuple{L,ComplexF32},
                            ::Val{D}) where {L,D}
    ntuple(Val(L)) do k
        r = (k - 1) % D + 1
        c = (k - 1) ÷ D + 1
        acc = ComplexF32(0, 0)
        @inbounds for q in 1:D
            acc += A[r + (q - 1) * D] * B[q + (c - 1) * D]
        end
        acc
    end
end
@inline _nt_scale(A::NTuple{L,ComplexF32}, s::ComplexF32) where {L} =
    ntuple(k -> A[k] * s, Val(L))
@inline _nt_addI(A::NTuple{L,ComplexF32}, ::Val{D}) where {L,D} =
    ntuple(Val(L)) do k
        ((k - 1) % D == (k - 1) ÷ D) ? A[k] + ComplexF32(1, 0) : A[k]
    end

# ── Kernel: U[:,:,i] = exp(-i H[:,:,i] dt[i]) via FP32 scaling-and-squaring ────
function _metal_expm_kernel!(U, H, dtarr, ::Val{D}, p::Int32, N::Int32) where {D}
    i = Metal.thread_position_in_grid_1d()
    if i <= N
        L = D * D
        @inbounds dt = dtarr[i]
        M = ntuple(Val(L)) do k
            r = (k - 1) % D + 1
            c = (k - 1) ÷ D + 1
            @inbounds h = H[r, c, i]
            ComplexF32(dt * imag(h), -dt * real(h))          # -i·dt·H
        end
        nrm = 0.0f0
        @inbounds for c in 1:D
            s = 0.0f0
            for r in 1:D
                s += abs(M[r + (c - 1) * D])
            end
            nrm = max(nrm, s)
        end
        sq = 0
        scale = 1.0f0
        t = nrm
        while t > 0.5f0 && sq < 32                            # scale so ‖B‖₁ ≲ 0.5
            t *= 0.5f0; scale *= 0.5f0; sq += 1
        end
        M = _nt_scale(M, ComplexF32(scale, 0))
        E = ntuple(k -> ((k - 1) % D == (k - 1) ÷ D) ? ComplexF32(1, 0) : ComplexF32(0, 0),
                   Val(L))
        k = p                                                 # Horner Taylor
        while k >= 1
            ME = _nt_matmul(M, E, Val(D))
            E = _nt_addI(_nt_scale(ME, ComplexF32(1.0f0 / Float32(k), 0)), Val(D))
            k -= 1
        end
        s2 = 0
        while s2 < sq                                         # undo the scaling
            E = _nt_matmul(E, E, Val(D))
            s2 += 1
        end
        @inbounds for c in 1:D, r in 1:D
            U[r, c, i] = E[r + (c - 1) * D]
        end
    end
    return
end

# Host wrapper: returns a host Array{ComplexF64,3} (the sweeps run on CPU).
function _metal_batched_herm(H::AbstractArray{<:Complex,3}, dtv::Vector{Float64})
    D, _, N = size(H)
    Hf  = Metal.MtlArray(ComplexF32.(H))
    Uf  = Metal.MtlArray{ComplexF32}(undef, D, D, N)
    dtf = Metal.MtlArray(Float32.(dtv))
    nthreads = 256
    Metal.@metal threads = nthreads groups = cld(N, nthreads) _metal_expm_kernel!(
        Uf, Hf, dtf, Val(D), _METAL_TAYLOR_ORDER, Int32(N))
    Metal.synchronize()
    return ComplexF64.(Array(Uf))
end

# ── Primitive override ────────────────────────────────────────────────────────
function batched_herm_propagators(be::MetalBackend,
                                  H::AbstractArray{<:Complex,3}, dt)
    D, _, N = size(H)
    use_kernel = be.use_fp32 && D <= _METAL_KERNEL_MAX_DIM && N >= _METAL_KERNEL_MIN_BATCH
    if use_kernel
        try
            return _metal_batched_herm(H, Pulsar._batched_dt_vector(dt, N))
        catch e
            @warn "PulsarMetalExt.batched_herm_propagators: GPU kernel failed ($e); CPU fallback." maxlog = 1
        end
    end
    # Default / large-dim / FP64-exact path: threaded CPU eigensolve.
    return _batched_herm_cpu(H, dt)
end

# Eigensolve-free / matmul-bound primitives stay on CPU for now (the eigensolve
# was the bottleneck; these are cheap and FP64-exact).  The seam is here for a
# future FP32 device version.
batched_nonherm_propagators(::MetalBackend, A::AbstractArray{<:Number,3}, dt) =
    _batched_nonherm_cpu(A, dt)

batched_build_hamiltonian(::MetalBackend, drift,
                          ops::AbstractVector{<:AbstractMatrix},
                          coeff::AbstractMatrix) =
    _batched_build_ham_cpu(drift, ops, coeff)

batched_matvec(::MetalBackend, U::AbstractArray{<:Number,3},
               X::AbstractArray{<:Number,3}; adjoint::Bool=false) =
    _batched_matvec_cpu(U, X; adjoint=adjoint)

end  # module PulsarMetalExt
