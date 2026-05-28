"""
    MR/GRAPETracking.jl

GRAPE kernel for trajectory-tracking optimal control in Hilbert space.

Standard GRAPE optimises only the terminal fidelity  F_T = |⟨ψ_T|ψ(T)⟩|².
Tracking GRAPE adds intermediate checkpoints: at user-specified time steps
{j₁, j₂, …, j_M} the state ρ(t_{jₖ}) should match a target ρ_tar(jₖ), with
a weighting w(jₖ).  The total objective is:

    Φ = Σ_{drift} Σ_{state-pair} [ Σ_k w_k Tr(ρ_tar(jₖ) ρ(jₖ)) ]   (average over ensemble)

The density-matrix co-state recursion (Pontryagin):

    Λ_{N+1} = 0
    At checkpoint j_k:  Λ_{j_k} += 2 w_k (ρ_tar(j_k) - ρ(j_k + 1))
    Backward step:      Λ_n = U_{n+1}† Λ_{n+1} U_{n+1}   (for n < j_k)

Gradient:
    ∂Φ/∂w[ctrl, n] = -Δt Re(i Tr(Λ_{n+1}† [H_ctrl, ρ(n+1)]))

where H_ctrl = pwr × operators[ctrl] (rad/s operator, not multiplied by w).

## Public API

- `TrackingPoint(step, target_dm, weight)` — one intermediate checkpoint.
- `grape_tracking_kernel(waveform, ctrl)` — fidelity + gradient for `MRControl`
  that carries a non-empty `ctrl.tracking` vector.

## Integration with `optimcon`

Add `tracking = [TrackingPoint(...), ...]` to `MRControl`.  `optimcon` detects
a non-empty `ctrl.tracking` and calls this kernel instead of `grape_state_kernel`.

## Reference

Sklarz & Tannor, "Loading a Bose-Einstein condensate onto an optical lattice:
An application of optimal control theory to the nonlinear Schrödinger equation",
Phys. Rev. A 66, 053619 (2002).

Palao & Kosloff, "Quantum computing by an optimal control algorithm for unitary
transformations", Phys. Rev. Lett. 89, 188301 (2002).
"""

using LinearAlgebra

# ─── TrackingPoint ─────────────────────────────────────────────────────────────

"""
    TrackingPoint(step::Int, target_dm::Matrix{ComplexF64}, weight::Float64)

Specifies one intermediate-time objective for tracking GRAPE.

# Fields
- `step`      — time-step index (1-based) at which the checkpoint is evaluated.
                 The density matrix **after** propagation through step `step` is
                 compared to `target_dm`.  `step ∈ 1:n_t`.
- `target_dm` — target density matrix at this checkpoint (N×N, Hermitian, Tr=1).
                 Build from a pure state ψ as `ψ * ψ'`.
- `weight`    — relative weight w_k ≥ 0.  The total fidelity is a weighted sum
                 over all checkpoints; normalise weights to sum to 1.0 for
                 compatibility with standard fidelity reporting.

# Example
```julia
# Oscillate between +Iz and −Iz every 100 µs at 2 µs steps
n_per_seg = 50   # 100 µs / 2 µs

psi_pz = [1.0+0im, 0.0+0im]   # |↑⟩
psi_mz = [0.0+0im, 1.0+0im]   # |↓⟩

checkpoints = [
    TrackingPoint(k * n_per_seg, isodd(k) ? psi_mz * psi_mz' : psi_pz * psi_pz', 1.0)
    for k in 1:10
]
# Normalise weights
w0 = 1.0 / length(checkpoints)
checkpoints = [TrackingPoint(tp.step, tp.target_dm, w0) for tp in checkpoints]
```
"""
struct TrackingPoint
    step      :: Int
    target_dm :: Matrix{ComplexF64}
    weight    :: Float64
end

# Convenience constructor accepting a real or complex matrix
TrackingPoint(step::Int, target_dm::AbstractMatrix{<:Number}, weight::Real) =
    TrackingPoint(step, Matrix{ComplexF64}(target_dm), Float64(weight))

# ─── grape_tracking_kernel ────────────────────────────────────────────────────

"""
    grape_tracking_kernel(waveform, ctrl) → (fidelity, grad)

GRAPE fidelity and gradient for trajectory-tracking problems.

Requires `ctrl.tracking` to be a non-empty `Vector{TrackingPoint}`.
If `ctrl.tracking` is empty this function raises an error; call
`grape_state_kernel` for standard terminal-only optimisation.

# Algorithm (per drift × pwr ensemble member, per initial state pair)

## Forward pass
```
ρ[1] = ρ_init * ρ_init'
for n = 1:n_t
    H_n = H_drift + pwr * Σ_k w[k,n] * Op[k]
    U[n] = exp(-i H_n Δt)
    ρ[n+1] = U[n] ρ[n] U[n]†
end
```

## Accumulate checkpoint fidelity
```
for each TrackingPoint tp:
    Φ += tp.weight * Re(Tr(tp.target_dm * ρ[tp.step + 1]))
```

## Backward pass (co-state recursion)
```
Λ = 0   (N×N complex matrix)
for n = n_t:-1:1
    if n is a checkpoint step j_k:
        ρ_at_ckpt = ρ[n + 1]   # state *after* step n
        Λ += 2 * w_k * (ρ_tar(j_k) - ρ_at_ckpt)
    # Gradient at step n using co-state *before* backward step
    for k = 1:n_ctrl:
        H_ctrl_k = pwr * Op[k]
        comm_k   = H_ctrl_k * ρ[n+1] - ρ[n+1] * H_ctrl_k
        grad[k, n] += -Δt * Re(i * Tr(Λ' * comm_k))
    Λ = U[n]† Λ U[n]
end
```

# Returns
- `fidelity::Float64` — ensemble + state-pair averaged checkpoint fidelity ∈ [0,1].
- `grad::Matrix{Float64}` — gradient ∂Φ/∂waveform, shape `[n_ctrl × n_t]`.

# Notes on normalisation
The fidelity for a pure state at a single checkpoint is `|⟨ψ_tar|ψ⟩|²`.
For density matrices: `Tr(ρ_tar ρ)` ∈ [0, 1] for pure targets and states.
Multiply weights by a factor of `1 / n_checkpoints` to keep fidelity ∈ [0, 1].
"""
function grape_tracking_kernel(waveform::Matrix{Float64}, ctrl)
    isempty(ctrl.tracking) &&
        error("grape_tracking_kernel: ctrl.tracking is empty. " *
              "Add TrackingPoint entries or call grape_state_kernel instead.")

    # ── Backend dispatch ──────────────────────────────────────────────────────
    if ctrl.backend == :metal
        return _grape_tracking_gpu(waveform, ctrl, :metal)
    elseif ctrl.backend == :cuda
        return _grape_tracking_gpu(waveform, ctrl, :cuda)
    end
    # :cpu falls through to the threaded implementation.
    return _grape_tracking_cpu_threaded(waveform, ctrl)
end

# ─── CPU implementation (multi-threaded) ─────────────────────────────────────

function _grape_tracking_cpu_threaded(waveform::Matrix{Float64}, ctrl)
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs
    dim     = size(ctrl.drifts[1], 1)

    ens_pairs = [(H, p) for H in ctrl.drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)

    # Per-outer-member result buffers
    fid_buf  = zeros(Float64, n_outer)
    grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:n_outer]

    # Build a lookup: step index → (weight, target_dm) for O(1) access
    ckpt_map = Dict{Int, Vector{Tuple{Float64, Matrix{ComplexF64}}}}()
    for tp in ctrl.tracking
        (tp.step < 1 || tp.step > n_t) &&
            error("TrackingPoint.step=$(tp.step) out of range [1, $n_t]")
        list = get!(ckpt_map, tp.step, Tuple{Float64, Matrix{ComplexF64}}[])
        push!(list, (tp.weight, tp.target_dm))
    end
    ckpt_steps = sort(collect(keys(ckpt_map)))

    n_th = Threads.maxthreadid()
    H_bufs   = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    VD_bufs  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    Ps_bufs  = [[Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_t] for _ in 1:n_th]

    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads :static for idx in 1:n_outer
        tid          = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        H_buf        = H_bufs[tid]
        VD_buf       = VD_bufs[tid]
        Ps           = Ps_bufs[tid]
        grad_local   = grad_buf[idx]

        # ── Build propagators ────────────────────────────────────────────────
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
            end
            _expm_neg_i_into!(Ps[n], H_buf, ctrl.pulse_dt[n], VD_buf)
        end

        fid_local = 0.0

        for s in 1:n_pairs
            ψ₀ = ctrl.rho_init[s]

            # ── Forward: density-matrix trajectory ──────────────────────────
            rho_traj = Vector{Matrix{ComplexF64}}(undef, n_t + 1)
            rho_traj[1] = ψ₀ * ψ₀'
            for n in 1:n_t
                rho_traj[n + 1] = Ps[n] * rho_traj[n] * Ps[n]'
            end

            # ── Checkpoint fidelity ──────────────────────────────────────────
            for n_ck in ckpt_steps
                for (w_k, ρ_tar_k) in ckpt_map[n_ck]
                    fid_local += w_k * real(tr(ρ_tar_k * rho_traj[n_ck + 1]))
                end
            end

            # ── Backward: co-state recursion ─────────────────────────────────
            Lambda = zeros(ComplexF64, dim, dim)

            for n in n_t:-1:1
                if haskey(ckpt_map, n)
                    for (w_k, ρ_tar_k) in ckpt_map[n]
                        @. Lambda += 2.0 * w_k * (ρ_tar_k - rho_traj[n + 1])
                    end
                end

                ρ_n1 = rho_traj[n + 1]
                dt_n = ctrl.pulse_dt[n]
                for k in 1:n_ctrl
                    H_ctrl_k = pwr .* ctrl.operators[k]
                    comm_k   = H_ctrl_k * ρ_n1 .- ρ_n1 * H_ctrl_k
                    grad_local[k, n] += -dt_n * real(1im * tr(Lambda' * comm_k))
                end

                Lambda = Ps[n]' * Lambda * Ps[n]
            end
        end  # state pairs

        fid_buf[idx] = fid_local
    end  # ensemble

    BLAS.set_num_threads(old_blas)

    # ── Reduction ──────────────────────────────────────────────────────────────
    fidelity = sum(fid_buf) / N_ens
    grad     = zeros(Float64, n_ctrl, n_t)
    for g in grad_buf
        grad .+= g
    end
    grad ./= N_ens

    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end

# ─── GPU implementation (Metal / CUDA) ───────────────────────────────────────

"""
    _grape_tracking_gpu(waveform, ctrl, gpu_sym) → (fidelity, grad)

GPU-accelerated tracking GRAPE kernel. Dispatches to `_grape_tracking_gpu_kernel`
after resolving the backend package and element type.

Falls back to CPU (`grape_tracking_kernel` with `:cpu` backend) if the
requested package is not loaded.

## Memory note

The GPU forward pass stores the full density-matrix trajectory
`rho_traj[dim, dim, n_outer, n_t+1]` on device before downloading in bulk
for the CPU backward pass. For large ensembles and long pulses this can reach
several hundred MB. The CPU path should be preferred when GPU memory is limited
or when `dim` is small (≤ 8), where kernel-launch overhead dominates.
"""
function _grape_tracking_gpu(waveform::Matrix{Float64}, ctrl, gpu_sym::Symbol)
    if gpu_sym == :metal && !_METAL_LOADED[]
        @warn "Pulsar: Metal.jl not loaded — falling back to CPU tracking kernel" maxlog=1
        return _grape_tracking_cpu_threaded(waveform, ctrl)
    end
    if gpu_sym == :cuda && !_CUDA_LOADED[]
        @warn "Pulsar: CUDA.jl not loaded — falling back to CPU tracking kernel" maxlog=1
        return _grape_tracking_cpu_threaded(waveform, ctrl)
    end

    if gpu_sym == :metal
        Metal  = Base.loaded_modules[Base.identify_package("Metal")]
        T      = Complex{Float32}
        to_gpu = x -> Metal.mtl(T.(x))
    else  # :cuda
        CUDA   = Base.loaded_modules[Base.identify_package("CUDA")]
        T      = ComplexF64
        to_gpu = x -> CUDA.cu(x)
    end

    return _grape_tracking_gpu_kernel(waveform, ctrl, to_gpu, T)
end

"""
    _batched_dm_prop_gpu(Ps_n, PsAdj_n, ρ_batch, dim, n_outer) → ρ_new

Batched density-matrix propagation on GPU:

    ρ_new[:,:,i] = Ps_n[:,:,i] · ρ_batch[:,:,i] · PsAdj_n[:,:,i]

All arguments are GPU arrays with shape `[dim, dim, n_outer]`.
Uses reshape+sum broadcasting — no NNlib dependency required.

# Step 1: tmp[r,p,i]   = Σ_q  Ps_n[r,q,i]    · ρ_batch[q,p,i]
# Step 2: ρ_new[r,c,i] = Σ_p  tmp[r,p,i]      · PsAdj_n[p,c,i]
"""
function _batched_dm_prop_gpu(Ps_n, PsAdj_n, ρ_batch, dim::Int, n_outer::Int)
    # Step 1: left-multiply  tmp = Ps_n * ρ_batch
    tmp = dropdims(sum(
            reshape(Ps_n,    dim, dim, n_outer, 1) .*
            reshape(ρ_batch, 1,   dim, dim,     n_outer);
            dims = 2); dims = 2)

    # Step 2: right-multiply  ρ_new = tmp * PsAdj_n
    ρ_new = dropdims(sum(
                reshape(tmp,      dim, dim, n_outer, 1) .*
                reshape(PsAdj_n,  1,   dim, dim,     n_outer);
                dims = 2); dims = 2)

    return ρ_new   # [dim, dim, n_outer] on GPU
end

"""
    _grape_tracking_gpu_kernel(waveform, ctrl, to_gpu, T) → (fidelity, grad)

GPU-accelerated implementation of the tracking GRAPE kernel.

## Algorithm

1. **CPU parallel propagator build** — same as the CPU kernel; uses LAPACK
   `eigen` on each of the `n_outer` Hamiltonians in parallel with `@threads`.

2. **Single bulk CPU → GPU transfer** — `Ps` and `PsAdj` as
   `[dim × dim × (n_outer × n_t)]` arrays.

3. **Batched DM forward pass on GPU** — at each of the `n_t` steps, applies
   `_batched_dm_prop_gpu` to advance all `n_outer` density matrices
   simultaneously. Stores the full trajectory `rho_traj_gpu[dim, dim, n_outer, n_t+1]`
   on device.

4. **Fidelity from checkpoints** — computed on GPU, returned as a scalar.

5. **Single bulk GPU → CPU transfer** — downloads `rho_traj` as a CPU array.

6. **CPU backward pass** — sequential co-state recursion (same formula as the
   CPU kernel), parallelised over the `n_outer` outer ensemble members.
"""
function _grape_tracking_gpu_kernel(waveform::Matrix{Float64}, ctrl, to_gpu, ::Type{T}) where {T}
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs
    dim     = size(ctrl.drifts[1], 1)

    ens_pairs = [(H, p) for H in ctrl.drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)

    # ── Build checkpoint lookup ───────────────────────────────────────────────
    ckpt_map = Dict{Int, Vector{Tuple{Float64, Matrix{ComplexF64}}}}()
    for tp in ctrl.tracking
        (tp.step < 1 || tp.step > n_t) &&
            error("TrackingPoint.step=$(tp.step) out of range [1, $n_t]")
        list = get!(ckpt_map, tp.step, Tuple{Float64, Matrix{ComplexF64}}[])
        push!(list, (tp.weight, tp.target_dm))
    end
    ckpt_steps = sort(collect(keys(ckpt_map)))

    # ── Step 1: CPU parallel propagator build ─────────────────────────────────
    Ps_cpu    = Array{T}(undef, dim, dim, n_outer, n_t)
    PsAdj_cpu = Array{T}(undef, dim, dim, n_outer, n_t)

    n_th  = Threads.maxthreadid()
    H_bufs  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    VD_bufs = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    P_bufs  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]

    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid          = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        H_buf        = H_bufs[tid]
        VD_buf       = VD_bufs[tid]
        P_buf        = P_bufs[tid]
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
            end
            _expm_neg_i_into!(P_buf, H_buf, ctrl.pulse_dt[n], VD_buf)
            Ps_cpu[:, :, idx, n]    = T.(P_buf)
            PsAdj_cpu[:, :, idx, n] = T.(P_buf')
        end
    end
    BLAS.set_num_threads(old_blas)

    # ── Step 2: Bulk CPU → GPU transfer ───────────────────────────────────────
    # Reshape to 3D [dim, dim, n_outer*n_t] for indexing as Ps_gpu[:,:, (n-1)*n_outer+i]
    Ps_gpu    = to_gpu(reshape(Ps_cpu,    dim, dim, n_outer * n_t))
    PsAdj_gpu = to_gpu(reshape(PsAdj_cpu, dim, dim, n_outer * n_t))

    fidelity_sum = 0.0
    grad_sum     = zeros(Float64, n_ctrl, n_t)

    for s in 1:n_pairs
        ψ₀ = ctrl.rho_init[s]
        ρ0_T = Matrix{T}(ψ₀ * ψ₀')   # [dim, dim]

        # ── Step 3: Batched DM forward pass on GPU ────────────────────────────
        # ρ_batch[:,:,i] = density matrix for ensemble member i
        ρ_batch = to_gpu(repeat(reshape(ρ0_T, dim, dim, 1), 1, 1, n_outer))  # [dim,dim,n_outer]

        # Store full trajectory on GPU: rho_traj_gpu[:,:,:,n] = ρ after step n-1
        # (n=1 is initial, n=n_t+1 is after last step)
        rho_traj_gpu = similar(ρ_batch, dim, dim, n_outer, n_t + 1)
        rho_traj_gpu[:, :, :, 1] = ρ_batch

        for n in 1:n_t
            # Slices for step n: shape [dim, dim, n_outer]
            base = (n - 1) * n_outer
            Ps_n    = Ps_gpu[:, :, base + 1 : base + n_outer]
            PsAdj_n = PsAdj_gpu[:, :, base + 1 : base + n_outer]
            ρ_batch = _batched_dm_prop_gpu(Ps_n, PsAdj_n, ρ_batch, dim, n_outer)
            rho_traj_gpu[:, :, :, n + 1] = ρ_batch
        end

        # ── Step 4: Checkpoint fidelity (on GPU, accumulate on CPU) ──────────
        fid_local = 0.0
        for n_ck in ckpt_steps
            ρ_ck_cpu = ComplexF64.(Array(rho_traj_gpu[:, :, :, n_ck + 1]))  # [dim,dim,n_outer]
            for (w_k, ρ_tar_k) in ckpt_map[n_ck]
                for i in 1:n_outer
                    fid_local += w_k * real(tr(ρ_tar_k * ρ_ck_cpu[:, :, i]))
                end
            end
        end
        fidelity_sum += fid_local

        # ── Step 5: Bulk GPU → CPU download ──────────────────────────────────
        rho_traj_cpu = ComplexF64.(Array(rho_traj_gpu))   # [dim, dim, n_outer, n_t+1]

        # ── Step 6: CPU backward pass ─────────────────────────────────────────
        # Parallelise over ensemble members; each thread owns its own Lambda and
        # gradient slice. Uses Ps_cpu (already on CPU) for backward propagation.
        grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:n_outer]

        Threads.@threads :static for idx in 1:n_outer
            _, pwr = ens_pairs[idx]
            Lambda = zeros(ComplexF64, dim, dim)
            gl     = grad_buf[idx]

            for n in n_t:-1:1
                # Inject tracking term at checkpoint (state after step n)
                if haskey(ckpt_map, n)
                    for (w_k, ρ_tar_k) in ckpt_map[n]
                        ρ_n1 = @view rho_traj_cpu[:, :, idx, n + 1]
                        @. Lambda += 2.0 * w_k * (ρ_tar_k - ρ_n1)
                    end
                end

                # Gradient contribution at step n
                ρ_n1 = @view rho_traj_cpu[:, :, idx, n + 1]
                dt_n = ctrl.pulse_dt[n]
                for k in 1:n_ctrl
                    H_ctrl_k = pwr .* ctrl.operators[k]
                    comm_k   = H_ctrl_k * ρ_n1 .- ρ_n1 * H_ctrl_k
                    gl[k, n] += -dt_n * real(1im * tr(Lambda' * comm_k))
                end

                # Backward-propagate co-state: Λ → Ps[n]† Λ Ps[n]
                # Use ComplexF64 Ps slice (convert from T)
                P_n    = ComplexF64.(Ps_cpu[:, :, idx, n])
                Lambda = P_n' * Lambda * P_n
            end
        end

        for gl in grad_buf
            grad_sum .+= gl
        end
    end  # state pairs

    fidelity = fidelity_sum / N_ens
    grad     = grad_sum ./ N_ens

    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end
