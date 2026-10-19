# Fidelity metrics

Fidelity functions live in
[`src/Physics/Objectives.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/Objectives.jl).
Pulsar exposes a typed algebra of metrics: every metric is a singleton
struct subtyping `AbstractFidelityMetric`, so the compiler specialises on
the metric without runtime symbol comparisons. A `type = :symbol` keyword
API delegates to the same dispatch.

## The nine metrics: Φ^X_Y

Every figure of merit is a choice of **what is compared** (the level `X`)
and **how the complex overlap `z` is reduced to a real number** (the
reduction `Y`). Three levels times three reductions gives nine metrics,
and Pulsar names each one after its cell:

| Level `X`                                  | Real part `Re z` (`R`) | Modulus `\|z\|` (`M`) | Modulus squared `\|z\|²` (`Q`) |
|--------------------------------------------|------------------------|------------------------|---------------------------------|
| **P** — propagator, `z = Tr(U_t† U)/d`      | `:prop_real` · Φ^P_R    | `:prop_modulus` · Φ^P_M | `:prop_squared` · Φ^P_Q          |
| **S** — state, `z = ⟨ψ_t \| ψ(T)⟩`         | `:state_real` · Φ^S_R   | `:state_modulus` · Φ^S_M | `:state_squared` · Φ^S_Q        |
| **D** — density matrix, `z = Tr(ρ_t† ρ)`    | `:dm_real` · Φ^D_R      | `:dm_modulus` · Φ^D_M   | `:dm_squared` · Φ^D_Q            |

In prose these are Prop-Real, Prop-Modulus, Prop-Squared, State-Real, and
so on. `Q` stands for "squared" because `S` is taken by *State*.

Off the grid:

| Symbol            | Formula                              | Notes                                   |
|-------------------|--------------------------------------|-----------------------------------------|
| `:state_squared_haar` | `(d · Φ^P_Q + 1)/(d + 1)`        | State-Squared averaged over Haar-random inputs = the average gate fidelity ⟨Φ^S_Q⟩_Haar; computed from the propagator overlap |
| `:dm_uhlmann`     | `(Tr √(√ρ σ √ρ))²`                   | Uhlmann–Jozsa; forward metric only      |
| `:prop_real_quad` | `1 − (1 − Φ^P_R)²`                   | Kobzar quadratic form; `grape_gate_kernel` only |

Each symbol has a matching type and exported singleton: `:state_squared` ↔
`StateSquared` ↔ `STATE_SQUARED`, `:prop_real` ↔ `PropReal` ↔ `PROP_REAL`, and
so on (`StateSquaredHaar`, `DMUhlmann` for the off-grid ones). The tuples
`PROP_METRICS`, `STATE_METRICS`, `DM_METRICS` and `FIDELITY_METRICS` list the
symbols, and `metric_label(:dm_squared)` returns `"Φ^D_Q (DM-Squared)"`.

### Where each level applies

The level is fixed by the object being optimized, and every entry point
checks it — asking for a density-matrix metric on a state-vector kernel
throws instead of being silently reinterpreted.

| Entry point                                   | Accepted metrics                                  | Default          |
|-----------------------------------------------|---------------------------------------------------|------------------|
| `unitary_target(U; metric)`, `gate_fidelity`  | `:prop_real`, `:prop_modulus`, `:prop_squared`, `:state_squared_haar` | `:prop_squared` |
| `state_target(ψ; metric)`                     | `:state_*`, and `:dm_*` evaluated on the pure states | `:state_squared` |
| `state_fidelity(ψ_t, ψ; type)`                | `:state_*`, `:dm_*`, `:dm_uhlmann`                | `:state_real`    |
| `MRControl(; fidelity)`                       | `:state_real`, `:state_modulus`, `:state_squared` | `:state_real`    |
| `LindbladMRControl(; fidelity)`               | `:dm_real`, `:dm_modulus`, `:dm_squared`          | `:dm_squared`    |
| `SU2Kernel(...; metric)`                      | propagator row (`:prop_real`, `:prop_modulus`, `:prop_squared`, `:state_squared_haar`, `:prop_real_quad`) and the \|↑⟩-probe state row (`:state_real`, `:state_modulus`, `:state_squared`) | `:prop_real` |
| `SpinIKernel(...; metric)`                    | state row and propagator row, as SU(2)            | `:state_squared` |
| `TrotterKernel(...; metric)`                  | `:state_real`, `:state_modulus`, `:state_squared` | `:state_squared` |
| `grape_gate_kernel(...; metric)`              | `:prop_real`, `:prop_real_quad`, `:prop_squared`, `:prop_modulus` | `:prop_real` |

The SU(2) kernel's `:state_squared` is `|⟨↑|U_t† U|↑⟩|²`, the state fidelity
of the `|↑⟩` probe. It is **not** Φ^P_Q: for SU(2), `Tr(U_t† U)/2` is the real
part of that overlap, so Φ^P_Q would be its real part squared.

### Older names

Before v0.3 the same symbol meant different metrics on different objects —
`:square` was State-Squared on `MRControl` but DM-Squared on
`LindbladMRControl`, and `:real` was Re⟨ψ_t|ψ⟩ on a state target but
Re Tr(U_t† U)/d on a gate target. The old names still work: each entry
point translates them at its own level, with a one-time deprecation warning.

| Old name                           | New name                                      |
|------------------------------------|-----------------------------------------------|
| `:normalized`                      | `:prop_squared`                               |
| `:average`, `:prop_average`        | `:state_squared_haar`                         |
| `:real`, `:modulus`, `:square`     | `:prop_*`, `:state_*` or `:dm_*`, by entry point |
| `:real_trace` (SU2/Spin-I kernels) | `:prop_real`                                  |
| `:squared` (SU2 kernel)            | `:state_squared`                              |
| `:state` (Spin-I/Trotter kernels)  | `:state_squared`                              |
| `:dm_linear`, `:dm_square`         | `:dm_real`, `:dm_squared`                     |
| `RealOverlap`, `SquaredOverlap`, `ModulusOverlap` | `StateReal`, `StateSquared`, `StateModulus` |
| `RealGate`, `NormalizedGate`, `ModulusGate`, `AverageGate` / `PropAverage` | `PropReal`, `PropSquared`, `PropModulus`, `StateSquaredHaar` |
| `LinearDMFidelity`, `SquaredDMFidelity`, `ModulusDMFidelity`, `UhlmannFidelity` | `DMReal`, `DMSquared`, `DMModulus`, `DMUhlmann` |

The singletons follow the types (`REAL_OVERLAP` → `STATE_REAL`,
`NORMALIZED_GATE` → `PROP_SQUARED`, `LINEAR_DM` → `DM_REAL`, …).
`canonical_metric(sym, supported; context)` performs the translation if you
need it in your own code.

### One implementation

Every kernel reduces its own complex overlap `z` with the same two functions,
`overlap_fidelity(metric, z; dim)` and `overlap_weight(metric, z; dim)`. The
weight `w` is defined by `dF = Re(w · dz)` — `1` for the real reductions,
`2z̄` for squared, `z̄/|z|` for modulus — so a kernel turns the derivative of
its overlap into a fidelity gradient as `Re(w · ∂z/∂u)`, and every kernel ×
metric combination is checked against finite differences in the test suite.
DM-Real is the raw `Re Tr(ρ_t† ρ)`: with traceless operators (operator
transfer) it is legitimately negative, and a perfect inversion scores −1.

## Using a metric

```julia
F = state_fidelity(ψ_target, ψ_final, STATE_SQUARED)          # typed
F = state_fidelity(ψ_target, ψ_final; type = :state_squared)  # by symbol
F = gate_fidelity(U, U_target, PROP_SQUARED)
F = gate_fidelity(U, U_target; type = :state_squared_haar)
F = state_fidelity(ρ, σ, DM_UHLMANN)
F = dm_fidelity(ρ, σ)                                          # alias for Uhlmann

tgt = unitary_target(U_target; metric = :prop_real)            # optimize Φ^P_R
tgt = state_target(ψ_target; psi_init = ψ_0, metric = :state_modulus)
```

`DMUhlmann` is computed via the spectral square root of `ρ`. Pulsar's
Lindblad GRAPE (see [Propagators](propagators.md)) builds the trajectory
`ρ(T)` via Liouville-space propagation and feeds the result back through
this dispatch.

## Subspace and process metrics

For leakage-aware optimization on multi-level platforms:

```julia
m = EssentialSubspaceGate([1, 2])             # qubit subspace inside a 3-level transmon
F = gate_fidelity(U_full, U_target, m)        # |Tr(Π U†_t U Π)|² / |Π|²
```

For `optim_target = "gate, file"`-style cooperative goals (a gate shape
**and** a specific state preparation in one cost function):

```julia
m = CooperativeTargetFidelity(PROP_SQUARED, STATE_SQUARED;
                              α = 0.6, β = 0.4)
F = cooperative_fidelity(U, U_target, ψ_init, ψ_target, m)
```

For process tomography reconstructed from basis-state transfers:

```julia
F = gate_fidelity(U, U_target, ProcessTomographyFidelity(d))
```

## Krotov co-state boundary

Krotov's algorithm needs `χ(T) = -∂J_T/∂⟨ψ(T)|`. Pulsar derives this
automatically per metric:

| Metric             | `χ(T)`                                |
|--------------------|---------------------------------------|
| `StateReal`        | `ψ_target`                            |
| `StateModulus`     | `ψ_target` (up to a phase)            |
| `StateSquared`     | `⟨ψ_target | ψ(T)⟩ · ψ_target`        |
| `PropReal`         | `U_target`                            |
| `PropSquared`      | `(Tr(U_target† U(T)) / d²) · U_target` |

`make_chi(metric, target, ψ_T)` returns the boundary; `krotov_optimize` calls
this when no `chi_constructor` is supplied.

## GRAPE gradient prefactor

The chain-rule coefficient `∂F/∂w[c, k]` is also dispatched on the metric:

| Metric             | `fidelity_grad_prefactor(z, inner, dt_pwr, m)`     |
|--------------------|----------------------------------------------------|
| `StateReal`, `PropReal`         | `dt_pwr · Im(inner)`                               |
| `StateSquared`, `PropSquared`   | `2 · dt_pwr · Im(z̄ · inner)`                      |
| `StateModulus`, `PropModulus`   | `dt_pwr · Im(z̄ · inner)/|z|`  (zero when `|z| < ε`) |

Density-matrix metrics use a Liouville-space adjoint gradient (see
[`Physics/Lindblad.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/Lindblad.jl))
rather than this Hilbert-space prefactor.

## Ensemble fidelity

For a parametric ensemble `{ψ_finals_i, ψ_targs_i}` with optional weights:

```julia
F = ensemble_fidelity(ψ_finals, ψ_targs;
                       weights  = nothing,
                       mode     = :mean,        # :mean, :weighted, :minimax
                       fid_type = :state_real)
```

Modes:

- `:mean` — `(1/N) Σ Fᵢ` or `Σ wᵢ Fᵢ` if weights are given
- `:weighted` — `Σ wᵢ Fᵢ` (requires weights)
- `:minimax` — `min Fᵢ` (worst-case / robust guarantee)

This is the foundation for [robust optimization](../algorithms/robust.md).

## Band-selective fidelity

For pulses with frequency-dependent passband / stopband requirements (see
[NMR](../domains/nmr.md), [MRI](../domains/mri.md)):

```julia
band_weights = [
    BandWeight(-500.0,  1.0),     # +1 weight at -500 Hz (passband)
    BandWeight(   0.0,  1.0),     # +1 weight on resonance
    BandWeight(+500.0,  1.0),
    BandWeight(+2500.0, -0.5),    # -0.5 weight at 2.5 kHz (stopband)
]

F = band_selective_fidelity(sys, ctrl, target, band_weights)
```

`band_selective_gradient` provides the matching analytic gradient. Both live
in
[`src/Physics/MRPhysics.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/MRPhysics.jl).

## Custom metrics

To introduce a non-standard objective, subtype `AbstractFidelityMetric` and
add methods for `state_fidelity` (or `gate_fidelity`) and, if you want
analytic gradients, `fidelity_grad_prefactor`. The optimizer machinery
specialises automatically.
