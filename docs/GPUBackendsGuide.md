# GPU Backends Guide

Pulsar's GPU path is now **genuinely batched**: instead of computing one matrix
exponential at a time in a Julia loop (one cuSOLVER eigendecomposition plus one
host sync per matrix — pathologically slow for the common case of *many small
matrices*), the whole batch is processed concurrently with a single
`heevjBatched` eigensolve and a single `gemm_strided_batched`.

On a Tesla T4, a batch of 763 560 Hermitian 4×4 matrices (a 420-member ensemble
× 1818 timesteps) drops from **~293 s** (old serial loop) to **~3 s** with the
batched kernel — about **98×**, matching a CPU `eigen` reference to ~1e-15.

## Zero-config setup (recommended)

Pulsar detects your GPU automatically and can install + wire up the right
backend for you with a single call:

```julia
using Pulsar
Pulsar.setup_gpu!()      # detects NVIDIA/Apple, installs CUDA.jl or Metal.jl,
                         # records the choice, and enables it now
```

`setup_gpu!()`:
1. **Detects** the hardware (`detect_gpu_hardware()` → `:cuda` / `:metal` /
   `:cpu`) — Apple Silicon → Metal, an NVIDIA driver → CUDA.
2. **Installs** the matching package into your active environment (skipped in
   CI / read-only / non-writable environments, and when `PULSAR_NO_GPU_INSTALL`
   is set; pass `force=true` to override).
3. **Records a preference** (via Preferences.jl) so that **every later
   `using Pulsar` auto-loads the GPU** — no need to `import CUDA` first.

After that first call, this is all you ever write:

```julia
using Pulsar             # GPU is detected, loaded, and selected automatically
```

Check the state any time with `Pulsar.gpu_setup_status()`. Until you run
`setup_gpu!`, Pulsar prints a one-line hint when it sees a GPU but otherwise
stays on CPU.

> Why a one-time explicit call instead of installing during precompilation?
> `Pkg.add` is unsafe in the precompilation sandbox and dangerous inside
> `__init__` (it can hang on a download, recurse, or mutate a shared/CI/read-only
> environment). Detection is always safe; installation is an explicit action.
> CUDA auto-load adds a few seconds per session; Metal is cheap.

### Manual activation

You can always skip the helper and load a GPU package yourself — the extension
mechanism activates either way:

```julia
import CUDA            # or: import Metal
using Pulsar
Pulsar.set_device!(:cuda)     # or :metal, or :cpu (default)
```

The extension's `__init__` sets the authoritative availability flag
(`Pulsar.is_cuda_available()` / `Pulsar.is_metal_available()`), guarded on
`CUDA.functional()`. With no GPU package loaded, `using Pulsar` still loads
cleanly and everything runs on the CPU — the extension only activates when its
trigger package is present.

> **Note (historical bug, fixed):** earlier versions bound `CUDA` with an
> `@eval using CUDA` block that ran only at precompile time, so the backend threw
> a swallowed `UndefVarError` and silently fell back to CPU even with a working
> GPU. The extension mechanism is the durable fix — the GPU method bodies live in
> `ext/PulsarCUDAExt.jl`, where `using CUDA` binds `CUDA` correctly.

## The batched primitive layer

All GPU acceleration flows through one device-dispatched seam,
`src/Backend/BatchedPrimitives.jl`. Each primitive is a generic function with a
threaded **CPU fallback** (used for any backend when the GPU extension is not
loaded) that the extensions override for `::CUDABackend` / `::MetalBackend`:

| Primitive | Purpose | GPU implementation |
|-----------|---------|--------------------|
| `batched_herm_propagators` | `exp(-i H dt)` for a Hermitian batch | `heevjBatched` + `gemm_strided_batched` |
| `batched_nonherm_propagators` | `exp(A dt)` for a general batch (Lindblad) | batched scaling-and-squaring |
| `batched_build_hamiltonian` | `H = drift + Σ_k coeff_k·op_k` for all columns | one on-device `gemm` |
| `batched_matvec` | forward/backward sweeps | `gemm_strided_batched` |

Because the CPU fallback is exact, the GPU result is validated by parity: the
production GRAPE state kernel (`_grape_batched`) is **bit-for-bit equal** to the
trusted CPU kernel (`_grape_cpu`) when run on a `CPUBackend`, so the GPU dispatch
inherits that correctness.

**Extension point for future applications.** A new application only needs to (a)
assemble its `H`/`L`/rotation batch and (b) pick which primitive family it uses;
scheduling, chunking, fidelity metrics, and ensemble aggregation are inherited.

## `dim ≤ 32` batched-eigensolver guard

cuSOLVER's batched Jacobi Hermitian eigensolver (`heevjBatched`) supports
matrices up to **32×32**. `batched_herm_propagators` uses it for `dim ≤ 32` and
falls back to the threaded per-matrix path for larger dimensions. For Hilbert
spaces this covers up to a 5-qubit system; most NMR/EPR/QC pulse-design problems
are well within range.

## Memory-budgeted chunking

Each `(D, D, N)` `ComplexF64` batch is `16·D²·N` bytes, and a batched eigensolve
needs several such buffers live at once. `batched_chunk_size` splits the batch
axis so each chunk fits a budget derived from `CUDA.available_memory()` and the
backend's `memory_limit_gb` (with a safety margin). Very large batches are
processed chunk-by-chunk transparently.

## Open systems (Lindblad)

Lindblad propagators are `exp(𝓛 dt)` where the Liouvillian `𝓛` is **non-Hermitian**
(dimension `D²`), so `heevjBatched` does not apply. These go through
`batched_nonherm_propagators`, which uses batched scaling-and-squaring with
`gemm_strided_batched` (CPU fallback: LAPACK `exp`). The `dim ≤ 32` Hermitian
guard does not apply here; the size/memory budget is taken on the `D²` Liouvillian.

## Metal (Apple Silicon)

Metal shaders are **FP32-only** and Metal.jl exposes no batched FP64 complex
Hermitian eigensolver (no cuSOLVER `heevjBatched` analogue). Rather than leave
the eigensolve on the CPU, the Metal extension ships a **custom GPU kernel** that
computes each propagator `exp(-i H dt)` by **FP32 scaling-and-squaring**, one
GPU thread per matrix — sidestepping the missing batched eigensolver entirely.

Measured on Apple Silicon for a batch of small Hermitian matrices vs the threaded
CPU eigensolve:

| dim | speedup vs threaded CPU | element accuracy |
|-----|-------------------------|------------------|
| 2   | ~32×  | ~1e-7 |
| 3   | ~34×  | ~2e-7 |
| 4   | ~33×  | ~2e-7 |
| 6   | ~15×  | ~4e-7 |
| 8   | ~7×   | ~5e-7 |

The win is largest for the small matrices typical of NMR/EPR/QC; by `dim > 8`
per-thread register pressure collapses GPU occupancy, so those fall back to the
threaded CPU path.

Because the kernel is **FP32-accurate (~1e-7)**, it is used **only when the
backend explicitly opts into FP32** — `metal_backend(use_fp32=true,
use_fp64_fallback=false)`. The default Metal backend (`metal_backend()` /
`resolve_backend(:metal)`) keeps the exact FP64 CPU eigensolve, honouring the
"never silently drop to FP32 for gradients" rule. The kernel also requires
`dim ≤ 8` and a batch of at least a few thousand matrices (below that, CPU launch
overhead wins). CUDA remains the path for exact FP64 batched eigensolves at any
`dim ≤ 32`.

## Batch-volume-aware device selection

The scheduler (`plan_hybrid_execution`) no longer decides on matrix dimension
alone. For *many small* matrices the **batch count**, not the dim, determines
whether the GPU wins, so the planner takes a `batch_count` and routes a large
batch to the GPU even at small dim — the 763k × 4×4 case now correctly goes to
the GPU, where the old dim-only heuristic forced it onto the CPU. Two planner
knobs control this: `cpu_batch_threshold` (small dim **and** small batch → CPU)
and `gpu_volume_threshold` (`batch_count·dim²` above which the GPU is forced).

## Precision

`Float64` is the default everywhere. An FP32 fast path is opt-in (forward-only
fidelity / metaheuristic screening) and is never used silently for gradients.

## Benchmark

`benchmark/gpu_batched_benchmark.jl` prints a before/after table (serial loop vs
batched primitive). Run it on a CUDA box after `import CUDA; using Pulsar` for
the headline numbers; on a CPU-only machine it still runs and shows the
threaded-batch structural win.
