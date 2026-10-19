# Ensemble averaging

A robust pulse is optimized over a set of variants of the problem — resonance
offsets, B1 inhomogeneity, coupling strengths, phase or timing errors — and
often over several input states. `EnsembleAveraging` describes that set once,
for any MR or QC system; `ensemble_averaging` evaluates it and returns the
fidelity and its GRAPE gradient.

```julia
using Pulsar

sys = spin_system([0.5, 0.5], [0.0 140.0; 140.0 0.0], [0.0, 0.0])
ens = EnsembleAveraging(
    OffsetAxis(range(-5e3, 5e3; length = 11); operator = offset_operator(sys)),
    DriveScaleAxis([0.9, 1.0, 1.1]),                         # B1 / Rabi error
    CouplingAxis([-5.0, 0.0, 5.0]; operator = coupling_operator(sys, 1, 2)),
    aggregate = :mean)

F, G = ensemble_averaging(ens, sys, ctrl, target)            # G: [n_controls × n_steps]
result = grape_optimize(sys, target, ctrl; ensemble = ens)
```

## Axes

Each axis is a list of sample values with weights (uniform by default); the
members are every combination (`combine = :product`) or the i-th values of
all axes together (`combine = :zip`).

| Axis | Effect of a value `v` |
|------|------------------------|
| `OffsetAxis(v; operator)` | `H_drift += 2π·v·operator` — resonance offset / B0 / qubit detuning |
| `CouplingAxis(v; operator)` | `H_drift += 2π·v·operator` — J coupling / exchange / ZZ deviation |
| `TermAxis(K, v)` | `H_drift += 2π·v·K` for any Hermitian `K` |
| `DriveScaleAxis(v; channels)` | `H_controls[c] *= v` — B1 inhomogeneity / Rabi miscalibration |
| `DrivePhaseAxis(ε; pairs)` | the `(x, y)` drive turned by `ε` about z — RF / IQ phase error |
| `TimeScaleAxis(v)` | `dt *= v` — timing error |
| `DriftListAxis(drifts)` | replaces the drift (an explicit list) |
| `FunctionAxis(f, v)` | `(H_drift, H_controls) = f(H_drift, H_controls, v)` — anything else |

`offset_operator(sys)` and `coupling_operator(sys, i, j)` return the operator
each system's builder multiplies its frequency / coupling parameter by, so an
axis value is the same as editing that parameter (at a fixed carrier):

| System | Offset operator | Coupling operator |
|--------|-----------------|-------------------|
| `SpinSystem`, `MRSpinSystem`, `SpinQubitSystem` | `Σ Iz_i` | `I_i·I_j` (`kind = :zz`: `Iz_i Iz_j`) |
| `TrappedIonSystem` | `Σ σz_i/2` | — |
| `TransmonSystem` | `Σ n̂_i` | `a_i†a_j + a_i a_j†` |
| `NeutralAtomSystem` | `Σ \|r⟩⟨r\|_i` | `\|rr⟩⟨rr\|` (blockade) |
| `NVCenterSystem` | `\|−1⟩⟨−1\|` | — |

## Aggregators

`:mean` (`Σω_k F_k / Σω_k`), `:sum` (signed weights allowed — pass and stop
bands), `:worst_case`, `:cvar` (mean of the worst `cvar_alpha` fraction of
weight) and `:softmin` (a smooth worst case). The gradient is exact for each.
`band_selective_fidelity` is an `OffsetAxis` with signed weights under `:sum`.

## Input states

With `probes = …` (and no target) each member is scored on a set of
(input, target) pairs: `StateProbes` (state vectors; State- or DM-row
metrics) or `OperatorProbes` (operators / density matrices, DM row on
`Tr(B†UAU†)/‖A‖‖B‖` — operator transfer). Build them from a target unitary:

| Probe set | Metric | Equals |
|-----------|--------|--------|
| `state_probes(U; set = :axes)` (qubit), `:axes6`, `:mub` (prime d) | `:state_squared` | `:state_squared_haar` — the average gate fidelity |
| any orthonormal basis | `:state_real` | Φ^P_R |
| `operator_probes(U; set = :pauli / :gell_mann)` | `:dm_real` | `(d²Φ^P_Q − 1)/(d² − 1)` |
| `state_probes(U; set = :basis)` | `:state_squared` | phase-blind — a Z error scores 1 (warns) |

```julia
ens = EnsembleAveraging(OffsetAxis(offs; operator = Iz); probes = state_probes(U; set = :axes))
result = grape_optimize(sys, ens, ctrl)           # the probes are the targets
```

## Front ends

- `grape_optimize(sys, target, ctrl; ensemble)` and `ensemble_objective(ens, sys,
  ctrl, target)` (an `EnsembleObjective` for every `ensemble_wrap` optimizer)
  take any ensemble.
- `MRControl(…; ensemble)`, `LindbladMRControl(…; ensemble)`,
  `SU2Kernel(ens, …)` and `SpinIKernel(ens, …)` average uniformly over a fixed
  product inside their kernels, so they take only a uniformly weighted `:mean`
  ensemble of the axes they can represent (offset / coupling terms, drive
  scale, probe pairs) and refuse anything else rather than approximate it.
