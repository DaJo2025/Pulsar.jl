# SU2DurationSweep.jl — Duration-sweep utility for SU(2) phase-only pulse design
#
# Formalizes the pattern from UR_90y_pulse_20kHz_Re_trace.jl:
# for each pulse duration T in a grid, run multi-start SU(2) GRAPE,
# save a checkpoint, and collect the sweep log.
#
# Depends on: Physics/SU2Objectives.jl (SU2Kernel),
#             Optimization/Gradient/QOC/GRAPEFamily.jl (grape_su2_multistart),
#             IO/Checkpoint.jl (save_checkpoint, Checkpoint)

"""
    su2_duration_sweep(kernel_fn, T_grid, DT;
                       n_restarts     = 20,
                       memory         = 20,
                       max_iter       = 2000,
                       tol            = 1e-12,
                       checkpoint_dir = nothing,
                       rng            = Random.GLOBAL_RNG,
                       verbose        = true,
                       print_interval = 200) → Vector{NamedTuple}

Sweep pulse duration over `T_grid` (in seconds), running `grape_su2_multistart`
at each duration.

## Arguments
- `kernel_fn`  : `(N_TS::Int) → SU2Kernel` — called once per duration.
  The simplest form is `N_TS -> kernel` (same kernel for all durations).
  Pass a closure if the precomputed ensemble parameters should vary with `N_TS`
  (e.g. when `dt` is fixed but offsets are DT-adjusted).
- `T_grid`     : pulse duration grid in seconds (iterable of `Float64`).
- `DT`         : time step in seconds; `N_TS = round(Int, T / DT)`.

## Keyword arguments
- `n_restarts`     : random restarts per duration (parallelised with Threads.@threads)
- `memory`         : L-BFGS-B history length
- `max_iter`       : maximum L-BFGS-B iterations per restart
- `tol`            : projected-gradient convergence tolerance
- `checkpoint_dir` : if not `nothing`, a `.jls` checkpoint is saved in this directory
  after each duration; filename is `T_<T_us>us.jls`.
- `rng`            : master RNG used to seed per-restart random phases
- `verbose`        : print progress per duration
- `print_interval` : pass-through to `grape_lbfgsb_optimize`

## Returns
`Vector{NamedTuple}` with one entry per duration containing:
- `T_us`     : pulse duration in µs
- `N_TS`     : number of time steps
- `F_best`   : best fidelity achieved
- `best_run` : index of the winning restart (always 1 for multi-start result)
- `phi_best` : optimised phase vector `(N_TS,)`
- `w_opt`    : Cartesian waveform `[2 × N_TS]` = `vcat(cos.(phi)', sin.(phi)')`
- `time_s`   : elapsed wall time for this duration

## Example
```julia
kernel = SU2Kernel(range(-20e3, 20e3; length=101), 10e3, 0.5e-6,
                   π/2, π/2, π/2; metric=:real_trace)

results = su2_duration_sweep(
    N_TS -> kernel,
    collect(200e-6:-5e-6:5e-6), 0.5e-6;
    n_restarts     = 20,
    checkpoint_dir = "checkpoints/UR_90y_20kHz",
)
```
"""
function su2_duration_sweep(
    kernel_fn       ,
    T_grid          ,
    DT              :: Real;
    n_restarts      :: Int     = 20,
    memory          :: Int     = 20,
    max_iter        :: Int     = 2000,
    tol             :: Float64 = 1e-12,
    checkpoint_dir          = nothing,
    rng                     = Random.GLOBAL_RNG,
    verbose         :: Bool    = true,
    print_interval  :: Int     = 200,
)
    if checkpoint_dir !== nothing
        isdir(checkpoint_dir) || mkpath(checkpoint_dir)
    end

    T_list  = collect(Float64, T_grid)
    n_durs  = length(T_list)
    log     = Vector{NamedTuple}(undef, n_durs)

    verbose && begin
        println("="^68)
        println("Pulsar — SU(2) duration sweep  ($(n_durs) durations × $(n_restarts) restarts)")
        println("="^68)
    end

    for (idx, T) in enumerate(T_list)
        N_TS   = round(Int, T / DT)
        kernel = kernel_fn(N_TS)

        verbose && @printf("[%2d / %2d]  T = %6.1f µs   N_TS = %d\n",
                           idx, n_durs, T * 1e6, N_TS)

        t_dur = @elapsed result = grape_su2_multistart(kernel, N_TS;
                                                        n_restarts     = n_restarts,
                                                        memory         = memory,
                                                        max_iter       = max_iter,
                                                        tol            = tol,
                                                        rng            = rng,
                                                        verbose        = verbose,
                                                        print_interval = print_interval)

        phi_best = vec(result.controls)          # [1 × N_TS] → length-N_TS vector
        F_best   = result.fidelity
        w_opt    = vcat(cos.(phi_best)', sin.(phi_best)')   # [2 × N_TS] Cartesian

        # Checkpoint
        if checkpoint_dir !== nothing
            ckpt_file = joinpath(checkpoint_dir,
                                 @sprintf("T_%03dus.jls", round(Int, T * 1e6)))
            save_checkpoint(
                ckpt_file,
                Checkpoint(
                    w_opt, F_best, 2, N_TS;
                    domain       = :mr,
                    drive_max_hz = 0.0,
                    T_pulse      = T,
                    metadata     = Dict{String,Any}(
                        "kernel_metric"  => string(kernel.metric),
                        "N_ens"          => kernel.N_ens,
                        "n_restarts"     => n_restarts,
                        "fidelity_kind"  => "SU2_$(kernel.metric)",
                    ),
                ),
            )
        end

        verbose && @printf("  → F_best = %+.8f   time = %.1f s\n", F_best, t_dur)

        log[idx] = (T_us     = T * 1e6,
                    N_TS     = N_TS,
                    F_best   = F_best,
                    best_run = 1,
                    phi_best = phi_best,
                    w_opt    = w_opt,
                    time_s   = t_dur)
    end

    if verbose
        println("="^68)
        @printf("  %-10s  %-7s  %-12s  %-8s\n", "T (µs)", "N_TS", "F_best", "time(s)")
        println("  " * "-"^45)
        for r in log
            @printf("  %-10.1f  %-7d  %+.8f  %.1f\n",
                    r.T_us, r.N_TS, r.F_best, r.time_s)
        end
        println("="^68)
    end

    return log
end
