"""
    MR/GRAPEState.jl

GRAPE kernel for state-transfer optimisation in Hilbert space, following the
Khaneja et al. (2005) formulation adapted for ensemble averaging.

The waveform is normalised (dimensionless): actual RF contribution at step n
for ensemble member j is:

    H_ctrl(n,j) = pwr_levels[j] × Σ_k waveform[k,n] × operators[k]

Fidelity and gradient are averaged over ALL ensemble members
(n_drifts × n_pwr_levels × n_state_pairs).

## CPU parallelization (default, `ctrl.backend = :cpu`)

The outer loop over (drift, pwr) pairs is parallelised with `Threads.@threads`.
Each thread owns its own result buffers — no locks, no shared mutable state.
Start Julia with multiple threads to exploit this:

    julia -t auto --project=. examples/NMR/nmr_01_broadband_90_pulse.jl

## GPU parallelization (`ctrl.backend = :metal` or `:cuda`)

Matrix exponentials are computed on CPU (LAPACK eigendecomposition, fast for
small Hermitian matrices). The propagator matrices are then transferred to the
GPU in a batch, and all forward/backward propagation matrix-vector products run
on the GPU. The full sequences of intermediate states are transferred back to
CPU in one bulk call for gradient accumulation.

  - `:metal` — Apple Silicon, via Metal.jl. Uses Complex{Float32} arithmetic.
               Effective for matrix dim ≥ 32 (roughly a 3-qubit system).
  - `:cuda`  — NVIDIA GPU, via CUDA.jl. Uses ComplexF64 arithmetic.
               Effective for matrix dim ≥ 16.

Both GPU paths fall back to CPU with a runtime warning if the package is not
loaded.

Reference:
  Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse
  sequences by gradient ascent algorithms", J. Magn. Reson. 172 (2005) 296–305.
"""

using LinearAlgebra

# ─── Public entry point ───────────────────────────────────────────────────────

"""
    grape_state_kernel(waveform, ctrl) → (fidelity, grad)

Compute ensemble-averaged state-transfer fidelity and its gradient w.r.t. the
normalised control waveform.

Dispatches to a CPU-threaded or GPU implementation based on `ctrl.backend`
(`:cpu`, `:metal`, or `:cuda`).

# Arguments
- `waveform::Matrix{Float64}` — shape `[n_ctrl × n_t]`; normalised amplitudes
  in `[ctrl.l_bound, ctrl.u_bound]`.
- `ctrl::MRControl` — complete problem specification.

# Returns
- `fidelity::Float64` — ensemble-averaged fidelity ∈ [0, 1].
- `grad::Matrix{Float64}` — gradient ∂F/∂waveform, same shape as `waveform`.

# Physics (per ensemble member)
1. Forward:  ψ[n+1] = exp(−i H[n] dt[n]) ψ[n]
2. Overlap:  z = ⟨ψ_targ | ψ(T)⟩
3. Fidelity: |z|²  (`:square`) or Re(z) (`:real`)
4. Backward: λ[n] = P[n]† λ[n+1],   λ[N+1] = ψ_targ
5. Gradient (`:square`): ∂F/∂w[k,n] = 2 dt pwr Im(z̄ ⟨λ[n+1]|Op[k]|ψ[n]⟩)
"""
function grape_state_kernel(waveform::Matrix{Float64}, ctrl)
    backend = _resolve_backend_auto(ctrl, waveform)
    if backend == :cpu
        return _grape_cpu(waveform, ctrl)
    elseif backend == :metal
        return _grape_gpu(waveform, ctrl, :metal)
    elseif backend == :cuda
        return _grape_gpu(waveform, ctrl, :cuda)
    else
        throw(ArgumentError(
            "Unknown backend ':$(backend)'. Use :cpu (default), :metal, :cuda, or :auto."))
    end
end

# Lesson 4: when `ctrl.backend == :auto`, consult `plan_hybrid_execution`
# to pick `:cpu` or `:gpu` based on Hilbert-space dim, n_timesteps and the
# planner's CPU/GPU thresholds. Maps `:gpu` to whichever GPU extension is
# loaded (CUDA preferred over Metal). `:hybrid` falls back to `:cpu` here
# because the GRAPE state kernel does not currently support split execution.
function _resolve_backend_auto(ctrl, waveform::Matrix{Float64})
    backend = getfield(ctrl, :backend)
    backend === :auto || return backend

    cuda_ok = _CUDA_LOADED[]
    metal_ok = _METAL_LOADED[]
    gpu_ok = cuda_ok || metal_ok

    dim = length(ctrl.rho_init[1])
    n_t = size(waveform, 2)
    n_c = size(waveform, 1)
    planner = HybridExecutionPlanner()
    decision = plan_hybrid_execution(dim, n_t, gpu_ok, planner;
                                      op = "gradient", n_controls = n_c)
    if decision === :gpu
        return cuda_ok ? :cuda : :metal
    else
        return :cpu
    end
end

# ─── CPU implementation (multi-threaded) ─────────────────────────────────────

"""
    _grape_cpu(waveform, ctrl) → (fidelity, grad)

Thread-parallel GRAPE kernel.  The outer loop over `(drift, pwr)` ensemble
pairs is distributed across Julia threads via `Threads.@threads`.

Each thread writes into its own pre-allocated fidelity scalar and gradient
matrix (indices into `fid_buf` / `grad_buf`), so there is no shared mutable
state and no synchronisation needed beyond the final reduction.

## Optimisations (vs naïve implementation)
- **Hermitian eigendecomposition** for all matrix exponentials: `_expm_neg_i`
  uses `eigen(Hermitian(H))` rather than Padé approximation, giving 3–5×
  faster propagator building for the small matrices common in NMR/EPR.
- **In-place `mul!`** for forward and backward propagation: forward states
  stored as columns of a pre-allocated `[dim × (n_t+1)]` matrix; backward
  co-states likewise.  Eliminates `O(n_t × n_pairs × n_outer)` vector
  allocations per call.
- **Pre-allocated `tmp_vec`** for the `Op[k] × ψ_n` inner products in the
  gradient loop: one scratch vector per thread, reused across all `(n, k)`.
"""
function _grape_cpu(waveform::Matrix{Float64}, ctrl)
    # Single-spin (dim=2) Bloch-vector fast path. Dispatched when:
    #   • ctrl.fast_path !== false
    #   • problem is technically eligible (see _bloch_problem_eligible)
    # When fast_path === true but problem is ineligible, raise an error.
    if hasproperty(ctrl, :fast_path) && ctrl.fast_path !== false
        if _bloch_problem_eligible(ctrl)
            return _grape_cpu_2x2_bloch(waveform, ctrl)
        elseif ctrl.fast_path === true
            throw(ArgumentError(
                "MRControl: fast_path=true requested but problem is not eligible. " *
                "Requires single-spin (dim=2), fidelity=:square, no tracking, " *
                "and pure-Pauli control operators (zero identity component)."))
        end
    end

    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs
    dim     = length(ctrl.rho_init[1])

    # Flatten (drift, pwr) pairs → one entry per independent propagator set.
    ens_pairs = [(H, p) for H in ctrl.drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)

    # Per-task result buffers — indexed by task, not thread.
    fid_buf  = zeros(Float64, n_outer)
    grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:n_outer]

    # ── Pre-allocate per-thread scratch to eliminate hot-loop allocations ─────
    # Indexed by Threads.threadid() (1-based, static scheduler).
    n_th = Threads.maxthreadid()
    # Propagator matrices: one pre-allocated [dim×dim] matrix per (thread, step)
    Ps_bufs   = [[Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_t]
                  for _ in 1:n_th]
    # Scratch for in-place H assembly and expm intermediate: 2 per thread
    H_bufs    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    VD_bufs   = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    # State / co-state trajectory matrices
    ψ_bufs    = [Matrix{ComplexF64}(undef, dim, n_t + 1) for _ in 1:n_th]
    λ_bufs    = [Matrix{ComplexF64}(undef, dim, n_t + 1) for _ in 1:n_th]
    # Gradient inner-product scratch vector
    tmp_bufs  = [Vector{ComplexF64}(undef, dim) for _ in 1:n_th]

    # ── Prevent BLAS/LAPACK from spawning extra threads inside @threads ────────
    # Without this, each eigen() call multiplies the OS thread count by
    # BLAS_threads, causing severe oversubscription and near-zero speedup.
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads :static for idx in 1:n_outer
        tid = Threads.threadid()

        H_drift, pwr   = ens_pairs[idx]
        Ps             = Ps_bufs[tid]
        H_buf          = H_bufs[tid]
        VD_buf         = VD_bufs[tid]
        ψ_fwd          = ψ_bufs[tid]
        λ_mat          = λ_bufs[tid]
        tmp            = tmp_bufs[tid]
        grad_local     = grad_buf[idx]

        # ── Build propagators in-place: H_buf = H_drift + pwr·Σ w[k,n]·Op[k] ─
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
            end
            _expm_neg_i_into!(Ps[n], H_buf, ctrl.pulse_dt[n], VD_buf)
        end

        fid_local = 0.0

        for s in 1:n_pairs
            ψ₀   = ctrl.rho_init[s]
            ψ_tg = ctrl.rho_targ[s]

            # ── Forward propagation ───────────────────────────────────────────
            ψ_fwd[:, 1] .= ψ₀
            for n in 1:n_t
                mul!(view(ψ_fwd, :, n + 1), Ps[n], view(ψ_fwd, :, n))
            end

            # ── Overlap and fidelity ──────────────────────────────────────────
            ψ_T      = view(ψ_fwd, :, n_t + 1)
            z        = state_overlap(ψ_tg, ψ_T)
            F_member = state_fidelity(ψ_tg, ψ_T; type = ctrl.fidelity)
            fid_local += F_member

            # ── Backward propagation ──────────────────────────────────────────
            λ_mat[:, n_t + 1] .= ψ_tg
            for n in n_t:-1:1
                mul!(view(λ_mat, :, n), Ps[n]', view(λ_mat, :, n + 1))
            end

            # ── Gradient accumulation ─────────────────────────────────────────
            for n in 1:n_t
                dt_n = ctrl.pulse_dt[n]
                λ_v  = view(λ_mat, :, n + 1)
                ψ_v  = view(ψ_fwd, :, n)
                for k in 1:n_ctrl
                    mul!(tmp, ctrl.operators[k], ψ_v)
                    inner = dot(λ_v, tmp)
                    grad_local[k, n] += fidelity_grad_prefactor(
                        z, inner, dt_n * pwr; type = ctrl.fidelity)
                end
            end
        end

        fid_buf[idx] = fid_local
    end  # @threads

    BLAS.set_num_threads(old_blas)

    # ── Reduction ─────────────────────────────────────────────────────────────
    fidelity = sum(fid_buf) / N_ens
    grad     = zeros(Float64, n_ctrl, n_t)
    for g in grad_buf
        grad .+= g
    end
    grad ./= N_ens

    # ── Penalty terms ─────────────────────────────────────────────────────────
    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end

# ─── GPU implementation (Metal / CUDA) ───────────────────────────────────────

"""
    _grape_gpu(waveform, ctrl, gpu_sym) → (fidelity, grad)

GPU-accelerated GRAPE kernel. Dispatches to `_grape_gpu_kernel` after
resolving the backend package and element type.

Falls back to CPU (`_grape_cpu`) if the requested package is not loaded.
"""
function _grape_gpu(waveform::Matrix{Float64}, ctrl, gpu_sym::Symbol)
    if gpu_sym == :metal && !_METAL_LOADED[]
        @warn "Pulsar: Metal.jl not loaded — falling back to CPU backend" maxlog=1
        return _grape_cpu(waveform, ctrl)
    end
    if gpu_sym == :cuda && !_CUDA_LOADED[]
        @warn "Pulsar: CUDA.jl not loaded — falling back to CPU backend" maxlog=1
        return _grape_cpu(waveform, ctrl)
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

    return _grape_gpu_kernel(waveform, ctrl, to_gpu, T)
end

function _grape_gpu_kernel(waveform::Matrix{Float64}, ctrl, to_gpu, ::Type{T}) where {T}
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs
    dim     = size(ctrl.drifts[1], 1)

    # Flatten (drift, pwr) into a single ensemble index.
    ens_pairs = [(H, p) for H in ctrl.drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)   # n_drift × n_pwr

    # ── Build ALL propagators on CPU in parallel, then single bulk GPU transfer
    Ps_cpu    = Array{T}(undef, dim, dim, n_outer, n_t)
    PsAdj_cpu = Array{T}(undef, dim, dim, n_outer, n_t)

    n_th_gpu  = Threads.maxthreadid()
    H_bufs_g  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_gpu]
    VD_bufs_g = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_gpu]
    P_bufs_g  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_gpu]

    old_blas_g = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid     = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        H_buf   = H_bufs_g[tid]
        VD_buf  = VD_bufs_g[tid]
        P_buf   = P_bufs_g[tid]
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
    BLAS.set_num_threads(old_blas_g)

    # Single host→device transfer: [dim × dim × n_outer × n_t] in one call.
    # Reshape to 3D so each [:,:,j] slice is a concrete GPU matrix view.
    Ps_gpu    = to_gpu(reshape(Ps_cpu,    dim, dim, n_outer * n_t))
    PsAdj_gpu = to_gpu(reshape(PsAdj_cpu, dim, dim, n_outer * n_t))

    fidelity_sum = 0.0
    grad_sum     = zeros(Float64, n_ctrl, n_t)

    for s in 1:n_pairs
        ψ0_T  = Vector{T}(ctrl.rho_init[s])
        ψtg_T = Vector{T}(ctrl.rho_targ[s])

        # ── Batched forward propagation on GPU ─────────────────────────────
        # ψ_batch[:, i] = state for ensemble member i, shape [dim × n_outer].
        # All n_t+1 snapshots written into a single pre-allocated
        # [dim × n_outer × (n_t+1)] GPU tensor.
        ψ_batch    = to_gpu(repeat(reshape(ψ0_T, dim, 1), 1, n_outer))  # [dim × n_outer]
        ψ_fwd_all  = similar(ψ_batch, dim, n_outer, n_t + 1)
        ψ_fwd_all[:, :, 1] = ψ_batch
        for n in 1:n_t
            Ps_n    = Ps_gpu[:, :, (n - 1) * n_outer + 1 : n * n_outer]
            ψ_batch = reshape(sum(Ps_n .* reshape(ψ_batch, 1, dim, n_outer); dims = 2), dim, n_outer)
            ψ_fwd_all[:, :, n + 1] = ψ_batch
        end

        # ── Overlap: single transfer of final states ────────────────────────
        ψ_final_cpu = ComplexF64.(Array(ψ_fwd_all[:, :, n_t + 1]))  # [dim × n_outer]
        ψ_targ_cpu  = ctrl.rho_targ[s]
        z_vec = [state_overlap(ψ_targ_cpu, ψ_final_cpu[:, i]) for i in 1:n_outer]
        for i in 1:n_outer
            fidelity_sum += state_fidelity(ψ_targ_cpu, ψ_final_cpu[:, i]; type = ctrl.fidelity)
        end

        # ── Batched backward propagation on GPU ────────────────────────────
        λ_batch  = to_gpu(repeat(reshape(ψtg_T, dim, 1), 1, n_outer))  # [dim × n_outer]
        λ_all    = similar(λ_batch, dim, n_outer, n_t + 1)
        λ_all[:, :, n_t + 1] = λ_batch
        for n in n_t:-1:1
            PsAdj_n = PsAdj_gpu[:, :, (n - 1) * n_outer + 1 : n * n_outer]
            λ_batch  = reshape(sum(PsAdj_n .* reshape(λ_batch, 1, dim, n_outer); dims = 2), dim, n_outer)
            λ_all[:, :, n] = λ_batch
        end

        # ── Single bulk host←device transfer ────────────────────────────────
        # ψ_all_cpu[:, i, n]  = ψ[n]   (state before applying P[n])
        # λ_all_cpu[:, i, n]  = λ[n+1] (co-state used in gradient at step n)
        ψ_all_cpu = ComplexF64.(Array(ψ_fwd_all[:, :, 1:n_t]))        # [dim, n_outer, n_t]
        λ_all_cpu = ComplexF64.(Array(λ_all[:, :, 2:n_t + 1]))        # [dim, n_outer, n_t]

        # ── Gradient accumulation on CPU (parallel over time steps) ────────
        Threads.@threads for n in 1:n_t
            dt_n = ctrl.pulse_dt[n]
            tmp  = Vector{ComplexF64}(undef, dim)   # thread-local scratch
            for i in 1:n_outer
                _, pwr_i = ens_pairs[i]
                z_i   = z_vec[i]
                λ_v   = view(λ_all_cpu, :, i, n)
                ψ_v   = view(ψ_all_cpu, :, i, n)
                for k in 1:n_ctrl
                    mul!(tmp, ctrl.operators[k], ψ_v)
                    inner = dot(λ_v, tmp)
                    grad_sum[k, n] += fidelity_grad_prefactor(
                        z_i, inner, dt_n * pwr_i; type = ctrl.fidelity)
                end
            end
        end

    end  # state pairs

    fidelity = fidelity_sum / N_ens
    grad     = grad_sum ./ N_ens

    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end

# ─── Forward-only fidelity (no gradient) ─────────────────────────────────────

"""
    fidelity_forward(waveform, drifts, pwr_levels, operators,
                     rho_init, rho_targ, pulse_dt;
                     fidelity_type = :real,
                     backend       = get_device()) → Float64

GPU-accelerated ensemble-averaged state-transfer fidelity (forward pass only).

Use this as the objective function for **derivative-free** optimizers
(GA, PSO, DE, CMA-ES, Nelder-Mead, …) when a GPU backend is active.
It uses the same batched-GPU propagation strategy as `grape_state_kernel`
(CPU-parallel matrix exponentials → single bulk GPU transfer → GPU matvec
forward pass) without computing the gradient.

# Arguments
- `waveform`   : `[n_ctrl × n_t]` normalised control amplitudes.
- `drifts`     : `Vector{Matrix}` of drift Hamiltonians (one per ensemble member).
- `pwr_levels` : `Vector{Float64}` of RF power levels in rad/s.
- `operators`  : `Vector{Matrix}` of dimensionless control operators.
- `rho_init`   : `Vector{Vector}` of initial states.
- `rho_targ`   : `Vector{Vector}` of target states.
- `pulse_dt`   : `AbstractVector{Float64}` of time-step durations (s).
- `fidelity_type` : `:real` or `:square`.
- `backend`    : `:cpu`, `:metal`, or `:cuda`.

# Returns
`Float64` — ensemble-averaged fidelity (average over all drift × pwr × state pairs).
"""
function fidelity_forward(
    waveform   :: Matrix{Float64},
    drifts     :: Vector{<:Matrix},
    pwr_levels :: Vector{Float64},
    operators  :: Vector{<:Matrix},
    rho_init   :: Vector{<:AbstractVector},
    rho_targ   :: Vector{<:AbstractVector},
    pulse_dt   :: AbstractVector{Float64};
    fidelity_type :: Symbol = :real,
    backend       :: Symbol = get_device(),
)
    if backend == :metal && _METAL_LOADED[]
        Metal  = Base.loaded_modules[Base.identify_package("Metal")]
        T      = Complex{Float32}
        to_gpu = x -> Metal.mtl(T.(x))
        return _fidelity_gpu(waveform, drifts, pwr_levels, operators,
                             rho_init, rho_targ, pulse_dt, fidelity_type, to_gpu, T)
    elseif backend == :cuda && _CUDA_LOADED[]
        CUDA   = Base.loaded_modules[Base.identify_package("CUDA")]
        T      = ComplexF64
        to_gpu = x -> CUDA.cu(x)
        return _fidelity_gpu(waveform, drifts, pwr_levels, operators,
                             rho_init, rho_targ, pulse_dt, fidelity_type, to_gpu, T)
    else
        if backend ∉ (:cpu, :metal, :cuda)
            @warn "fidelity_forward: unknown backend :$backend — using CPU" maxlog=1
        elseif backend != :cpu
            @warn "fidelity_forward: $backend package not loaded — using CPU" maxlog=1
        end
        return _fidelity_cpu(waveform, drifts, pwr_levels, operators,
                             rho_init, rho_targ, pulse_dt, fidelity_type)
    end
end

# CPU implementation (multi-threaded over ensemble members, in-place propagation)
function _fidelity_cpu(waveform, drifts, pwr_levels, operators,
                       rho_init, rho_targ, pulse_dt, fidelity_type)
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_pairs = length(rho_init)
    ens_pairs = [(H, p) for H in drifts for p in pwr_levels]
    n_outer   = length(ens_pairs)
    N_ens     = n_outer * n_pairs

    n_th_f   = Threads.maxthreadid()
    dim_f    = size(drifts[1], 1)
    H_bufs_f = [Matrix{ComplexF64}(undef, dim_f, dim_f) for _ in 1:n_th_f]
    VD_bufs_f= [Matrix{ComplexF64}(undef, dim_f, dim_f) for _ in 1:n_th_f]
    Ps_bufs_f= [[Matrix{ComplexF64}(undef, dim_f, dim_f) for _ in 1:n_t] for _ in 1:n_th_f]

    fid_buf  = zeros(Float64, n_outer)
    old_blas_f = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid     = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        H_buf   = H_bufs_f[tid]
        VD_buf  = VD_bufs_f[tid]
        Ps      = Ps_bufs_f[tid]
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * operators[k]
            end
            _expm_neg_i_into!(Ps[n], H_buf, pulse_dt[n], VD_buf)
        end
        local_fid = 0.0
        for s in 1:n_pairs
            ψ_cur = copy(rho_init[s])
            ψ_nxt = similar(ψ_cur)
            for n in 1:n_t
                mul!(ψ_nxt, Ps[n], ψ_cur)
                ψ_cur, ψ_nxt = ψ_nxt, ψ_cur
            end
            local_fid += state_fidelity(rho_targ[s], ψ_cur; type = fidelity_type)
        end
        fid_buf[idx] = local_fid
    end
    BLAS.set_num_threads(old_blas_f)
    return sum(fid_buf) / N_ens
end

# GPU implementation (batched forward pass only, no snapshots needed)
function _fidelity_gpu(waveform, drifts, pwr_levels, operators,
                       rho_init, rho_targ, pulse_dt, fidelity_type, to_gpu, ::Type{T}) where {T}
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    dim     = size(drifts[1], 1)
    n_pairs = length(rho_init)
    ens_pairs = [(H, p) for H in drifts for p in pwr_levels]
    n_outer   = length(ens_pairs)
    N_ens     = n_outer * n_pairs

    # Build propagators on CPU in parallel, single bulk GPU transfer
    Ps_cpu   = Array{T}(undef, dim, dim, n_outer, n_t)
    n_th_fg  = Threads.maxthreadid()
    H_bfg    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_fg]
    VD_bfg   = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_fg]
    P_bfg    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_fg]
    old_fg   = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid     = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        for n in 1:n_t
            H_bfg[tid] .= H_drift
            for k in 1:n_ctrl
                @. H_bfg[tid] += pwr * waveform[k, n] * operators[k]
            end
            _expm_neg_i_into!(P_bfg[tid], H_bfg[tid], pulse_dt[n], VD_bfg[tid])
            Ps_cpu[:, :, idx, n] = T.(P_bfg[tid])
        end
    end
    BLAS.set_num_threads(old_fg)
    Ps_gpu = to_gpu(reshape(Ps_cpu, dim, dim, n_outer * n_t))

    fidelity_sum = 0.0
    for s in 1:n_pairs
        ψ0_T    = Vector{T}(rho_init[s])
        ψ_batch = to_gpu(repeat(reshape(ψ0_T, dim, 1), 1, n_outer))  # [dim × n_outer]
        # Forward propagation: one GPU broadcast-reduce per time step.
        # No snapshot storage needed — only the final state is used.
        for n in 1:n_t
            Ps_n    = Ps_gpu[:, :, (n - 1) * n_outer + 1 : n * n_outer]
            ψ_batch = reshape(sum(Ps_n .* reshape(ψ_batch, 1, dim, n_outer); dims = 2), dim, n_outer)
        end
        ψ_final = ComplexF64.(Array(ψ_batch))  # single GPU→CPU transfer
        for i in 1:n_outer
            fidelity_sum += state_fidelity(rho_targ[s], ψ_final[:, i]; type = fidelity_type)
        end
    end
    return fidelity_sum / N_ens
end

# ─── Shared helpers ───────────────────────────────────────────────────────────

# Apply penalty terms; modifies grad in-place, returns updated fidelity scalar.
function _apply_penalties!(fidelity::Float64, grad::Matrix{Float64}, waveform, ctrl)
    for (pen, weight) in zip(ctrl.penalties, ctrl.p_weights)
        pen == :none && continue
        weight ≈ 0.0 && continue
        F_pen, G_pen = penalty_value_and_gradient(waveform;
            type    = pen,
            weight  = Float64(weight),
            l_bound = ctrl.l_bound,
            u_bound = ctrl.u_bound)
        fidelity -= F_pen
        grad     .-= G_pen
    end
    return fidelity
end

# Build control Hamiltonian at time step n.
@inline function _ctrl_hamiltonian(waveform, operators, n, n_ctrl)
    H = waveform[1, n] .* operators[1]
    for k in 2:n_ctrl
        H = H .+ waveform[k, n] .* operators[k]
    end
    return H
end

# _expm_neg_i / _expm_neg_i_into! are defined in Core/Propagators.jl.

# Penalty functions are provided by Core/Fidelity.jl:
#   penalty_value_and_gradient(w; type, weight, l_bound, u_bound, dt)

# ─── Per-sample kernel (for EnsembleObjective per-sample closures) ───────────

"""
    grape_state_kernel_single(waveform, H_drift, pwr, rho_init, rho_targ, ctrl)
        → (F::Float64, grad::Matrix{Float64})

Single-sample Hilbert-space GRAPE kernel for one `(H_drift, pwr, ρ_init, ρ_targ)`
combination. Returns the fidelity and gradient without averaging and without
applying penalties — the caller (an `EnsembleObjective` aggregator) is
responsible for both.

Used by [`build_ensemble_from_mrcontrol`](@ref) to back `:worst_case` / `:cvar`
aggregators that cannot use the batched [`grape_state_kernel`](@ref) fast path.
For `:mean` aggregation prefer `grape_state_kernel` — it is the same computation
but with threaded/GPU batching over all `(drift, pwr, state_pair)` members.
"""
function grape_state_kernel_single(waveform::Matrix{Float64},
                                    H_drift::AbstractMatrix,
                                    pwr::Real,
                                    rho_init::AbstractVector,
                                    rho_targ::AbstractVector,
                                    ctrl)
    n_ctrl = size(waveform, 1)
    n_t    = size(waveform, 2)
    dim    = length(rho_init)

    Ps     = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_t]
    H_buf  = Matrix{ComplexF64}(undef, dim, dim)
    VD_buf = Matrix{ComplexF64}(undef, dim, dim)
    ψ_fwd  = Matrix{ComplexF64}(undef, dim, n_t + 1)
    λ_mat  = Matrix{ComplexF64}(undef, dim, n_t + 1)
    tmp    = Vector{ComplexF64}(undef, dim)

    for n in 1:n_t
        H_buf .= H_drift
        for k in 1:n_ctrl
            @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
        end
        _expm_neg_i_into!(Ps[n], H_buf, ctrl.pulse_dt[n], VD_buf)
    end

    ψ_fwd[:, 1] .= rho_init
    for n in 1:n_t
        mul!(view(ψ_fwd, :, n + 1), Ps[n], view(ψ_fwd, :, n))
    end

    ψ_T = view(ψ_fwd, :, n_t + 1)
    z   = state_overlap(rho_targ, ψ_T)
    F   = state_fidelity(rho_targ, ψ_T; type = ctrl.fidelity)

    λ_mat[:, n_t + 1] .= rho_targ
    for n in n_t:-1:1
        mul!(view(λ_mat, :, n), Ps[n]', view(λ_mat, :, n + 1))
    end

    grad = zeros(Float64, n_ctrl, n_t)
    for n in 1:n_t
        dt_n = ctrl.pulse_dt[n]
        λ_v  = view(λ_mat, :, n + 1)
        ψ_v  = view(ψ_fwd, :, n)
        for k in 1:n_ctrl
            mul!(tmp, ctrl.operators[k], ψ_v)
            inner = dot(λ_v, tmp)
            grad[k, n] = fidelity_grad_prefactor(
                z, inner, dt_n * pwr; type = ctrl.fidelity)
        end
    end

    return F, grad
end

# ─── Single-spin (dim=2) Bloch-vector fast path ─────────────────────────────
#
# Closed-system state-transfer GRAPE with `fidelity=:square` on a single spin
# (Hilbert dim=2) reduces to real 3D rotations on the Bloch sphere. This path:
#   • avoids LAPACK eigendecomposition (uses Rodrigues + scalar exp);
#   • uses real arithmetic (4× over complex);
#   • stores 4 reals per timestep instead of a 2×2 complex propagator;
#   • computes the gradient as a real triple product per (k, n).
#
# Eligibility (auto-detected):
#   - dim(state) == 2
#   - fidelity == :square (Bloch is invariant under global phase, so :real
#     would require explicit phase tracking)
#   - no tracking checkpoints (routed elsewhere before _grape_cpu)
#   - all control operators are pure Pauli (identity component ≈ 0); the
#     drift's identity component is allowed and silently dropped (global phase).

@inline function _bloch_problem_eligible(ctrl)::Bool
    isempty(ctrl.rho_init) && return false
    length(ctrl.rho_init[1]) == 2 || return false
    ctrl.fidelity === :square || return false
    isempty(ctrl.tracking) || return false
    for op in ctrl.operators
        size(op) == (2, 2) || return false
        α, _, _, _ = _pauli_coeffs(op)
        abs(α) > 1e-10 && return false
    end
    return true
end

@inline function _state_to_bloch(ψ::AbstractVector)::NTuple{3,Float64}
    @inbounds begin
        α = ComplexF64(ψ[1])
        β = ComplexF64(ψ[2])
        ab = conj(α) * β
        rx = 2.0 * real(ab)
        ry = 2.0 * imag(ab)
        rz = abs2(α) - abs2(β)
    end
    return (rx, ry, rz)
end

# Apply Rodrigues rotation R(n̂, θ) to (vx, vy, vz), write into out[:, out_col].
# n̂ is read from axis[:, axis_col]. cos/sin are taken from precomputed cs/sn.
@inline function _rodrigues_apply!(out::AbstractMatrix{Float64}, out_col::Int,
                                   axis::AbstractMatrix{Float64}, axis_col::Int,
                                   cs::Float64, sn::Float64,
                                   vx::Float64, vy::Float64, vz::Float64)
    @inbounds begin
        nx = axis[1, axis_col]
        ny = axis[2, axis_col]
        nz = axis[3, axis_col]
        omc = 1.0 - cs
        # n̂ × v
        cx = ny*vz - nz*vy
        cy = nz*vx - nx*vz
        cz = nx*vy - ny*vx
        # n̂ · v
        ndotv = nx*vx + ny*vy + nz*vz
        out[1, out_col] = cs*vx + sn*cx + omc*ndotv*nx
        out[2, out_col] = cs*vy + sn*cy + omc*ndotv*ny
        out[3, out_col] = cs*vz + sn*cz + omc*ndotv*nz
    end
    return nothing
end

function _grape_cpu_2x2_bloch(waveform::Matrix{Float64}, ctrl)
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs

    # Pre-compute Pauli decompositions (per call, not per ensemble member)
    op_pauli    = [(let (_, b, c, d) = _pauli_coeffs(op); (b, c, d) end)
                   for op in ctrl.operators]
    drift_pauli = [(let (_, b, c, d) = _pauli_coeffs(H);  (b, c, d) end)
                   for H in ctrl.drifts]
    init_blochs = [_state_to_bloch(ψ) for ψ in ctrl.rho_init]
    targ_blochs = [_state_to_bloch(ψ) for ψ in ctrl.rho_targ]

    # Flatten (drift_idx, pwr) → outer ensemble index
    ens_pairs = [(di, p) for di in 1:n_drift for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)

    # Per-outer-task result buffers
    fid_buf  = zeros(Float64, n_outer)
    grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:n_outer]

    # Per-thread scratch
    n_th       = Threads.maxthreadid()
    r_bufs     = [Matrix{Float64}(undef, 3, n_t + 1) for _ in 1:n_th]
    λ_bufs     = [Matrix{Float64}(undef, 3, n_t + 1) for _ in 1:n_th]
    axis_bufs  = [Matrix{Float64}(undef, 3, n_t)     for _ in 1:n_th]
    cs_bufs    = [Vector{Float64}(undef, n_t)         for _ in 1:n_th]
    sn_bufs    = [Vector{Float64}(undef, n_t)         for _ in 1:n_th]

    Threads.@threads :static for idx in 1:n_outer
        tid = Threads.threadid()
        di, pwr  = ens_pairs[idx]
        bd, cd, dd = drift_pauli[di]
        r_traj   = r_bufs[tid]
        λ_traj   = λ_bufs[tid]
        axis     = axis_bufs[tid]
        cs_arr   = cs_bufs[tid]
        sn_arr   = sn_bufs[tid]
        grad_local = grad_buf[idx]

        # ── Build per-timestep rotation axis n̂ and (cos, sin) of θ ──────────
        @inbounds for n in 1:n_t
            b = bd; c = cd; d = dd
            for k in 1:n_ctrl
                pwr_w = pwr * waveform[k, n]
                bk, ck, dk = op_pauli[k]
                b += pwr_w * bk
                c += pwr_w * ck
                d += pwr_w * dk
            end
            mag = sqrt(b*b + c*c + d*d)        # = |ω|/2 with ω = 2(b,c,d)
            θ   = 2.0 * mag * ctrl.pulse_dt[n]
            cs_arr[n] = cos(θ)
            sn_arr[n] = sin(θ)
            if mag < 1e-30
                axis[1, n] = 1.0; axis[2, n] = 0.0; axis[3, n] = 0.0
            else
                inv_m = 1.0 / mag
                axis[1, n] = b * inv_m
                axis[2, n] = c * inv_m
                axis[3, n] = d * inv_m
            end
        end

        fid_local = 0.0

        for s in 1:n_pairs
            r0 = init_blochs[s]
            rT = targ_blochs[s]

            # ── Forward sweep: r[n+1] = R(n̂_n, θ_n) · r[n] ──────────────────
            r_traj[1, 1] = r0[1]; r_traj[2, 1] = r0[2]; r_traj[3, 1] = r0[3]
            @inbounds for n in 1:n_t
                _rodrigues_apply!(r_traj, n + 1, axis, n,
                                  cs_arr[n], sn_arr[n],
                                  r_traj[1, n], r_traj[2, n], r_traj[3, n])
            end

            # ── Fidelity per state pair: F = (1 + r[N+1] · r_target) / 2 ────
            @inbounds dot_RT = r_traj[1, n_t+1]*rT[1] +
                               r_traj[2, n_t+1]*rT[2] +
                               r_traj[3, n_t+1]*rT[3]
            fid_local += 0.5 * (1.0 + dot_RT)

            # ── Backward sweep: λ[N+1] = r_target; λ[n] = Rᵀ · λ[n+1] ──────
            # Rᵀ(n̂, θ) = R(n̂, -θ) → flip sin, keep cos
            λ_traj[1, n_t+1] = rT[1]; λ_traj[2, n_t+1] = rT[2]; λ_traj[3, n_t+1] = rT[3]
            @inbounds for n in n_t:-1:1
                _rodrigues_apply!(λ_traj, n, axis, n,
                                  cs_arr[n], -sn_arr[n],
                                  λ_traj[1, n+1], λ_traj[2, n+1], λ_traj[3, n+1])
            end

            # ── Gradient: ∂F/∂u[k,n] += dt·pwr·(b_k,c_k,d_k)·(r_mid × λ_mid)
            #
            # The standard first-order GRAPE in Hilbert space uses ⟨λ[n+1]|H_j|ψ[n]⟩
            # (mismatched times), which works only because of the trace identity
            #   2 Im(z̄ ⟨λ|H_j|ψ⟩) = (b,c,d) · (r_ψ × r_λ)
            # holding when ⟨ψ|λ⟩ = z. In Bloch space we have no z, so we MUST use
            # consistent times. The midpoint average minimises O(θ²) per-step error
            # and matches the slow Hilbert path to better than 1e-3 for typical dt.
            @inbounds for n in 1:n_t
                rx = 0.5 * (r_traj[1, n] + r_traj[1, n+1])
                ry = 0.5 * (r_traj[2, n] + r_traj[2, n+1])
                rz = 0.5 * (r_traj[3, n] + r_traj[3, n+1])
                lx = 0.5 * (λ_traj[1, n] + λ_traj[1, n+1])
                ly = 0.5 * (λ_traj[2, n] + λ_traj[2, n+1])
                lz = 0.5 * (λ_traj[3, n] + λ_traj[3, n+1])
                cx = ry*lz - rz*ly
                cy = rz*lx - rx*lz
                cz = rx*ly - ry*lx
                dt_pwr = ctrl.pulse_dt[n] * pwr
                for k in 1:n_ctrl
                    bk, ck, dk = op_pauli[k]
                    grad_local[k, n] += dt_pwr * (bk*cx + ck*cy + dk*cz)
                end
            end
        end

        fid_buf[idx] = fid_local
    end  # @threads

    fidelity = sum(fid_buf) / N_ens
    grad     = zeros(Float64, n_ctrl, n_t)
    for g in grad_buf
        grad .+= g
    end
    grad ./= N_ens

    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end
