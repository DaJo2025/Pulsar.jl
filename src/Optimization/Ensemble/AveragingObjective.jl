# ============================================================================
# Optimization/Ensemble/AveragingObjective.jl
# ============================================================================
# An EnsembleAveraging (Types/EnsembleAveraging.jl) as an EnsembleObjective, so
# every optimiser that takes one — through ensemble_wrap / ensemble_wrap_fonly —
# can optimise it. The ensemble aggregates internally (any of its aggregators),
# so it is wired in as the batched fast path.
# ============================================================================

"""
    ensemble_objective(ens, system, controls, target = nothing) -> EnsembleObjective

The ensemble average `ens` of the fidelity of `system` under `controls` as an
[`EnsembleObjective`](@ref): θ is `vec` of the `[n_controls × n_timesteps]`
amplitudes (the layout of `build_ensemble_from_systems`), starting point
`vec(controls.controls)`. `target` as in [`ensemble_averaging`](@ref) — a
`QuantumTarget`, or `nothing` when `ens.probes` are the targets.

```julia
obj      = ensemble_objective(ens, sys, ctrl, target)
f, grad! = ensemble_wrap(obj)                 # minimiser convention (−F)
θ, _     = lbfgs_optimize(f, grad!, vec(ctrl.controls))
```

The `aggregator` field of the returned object reads `:mean` because the batched
path requires it; the ensemble's own `aggregate` is what is applied.
"""
function ensemble_objective(ens::EnsembleAveraging, system, controls::ControlSequence,
                            target = nothing)
    _ensemble_metric(ens.probes, target, ens.metric)       # validate up front
    dt, n_t = controls.dt, controls.n_steps
    n_c = size(controls.amplitudes, 2)
    batched = θ -> begin
        amps = permutedims(reshape(collect(Float64, θ), n_c, n_t))
        ensemble_averaging(ens, system, ControlSequence(amps, dt, n_t), target)
    end
    return EnsembleObjective(; batched_eval = batched, n_samples = length(ens))
end
