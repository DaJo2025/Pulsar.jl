# Pulsar.jl

```@raw html
<p align="center">
  <img src="assets/logo.png" alt="Pulsar logo" width="420"/>
</p>
```

**Pulse Design Library for Spin Control Algorithms and Rollout**

Pulsar is a Julia package for designing optimal control pulses for quantum
systems, with first-class support for both quantum-computing platforms
(transmon, trapped ion, neutral atom, spin qubit, NV center) and magnetic
resonance (NMR, EPR, MAS solid-state, MRI, DNP).

It provides a unified, layered API across:

- **System types** that capture the physics of qubits and spins
- **Computation primitives** — propagators, ensemble maps, MAS Wigner rotations,
  Bloch propagator
- **Optimization algorithms** — GRAPE family, second-order (BFGS, L-BFGS,
  Newton, trust-region), direct search (Nelder–Mead, NEWUOA, COBYLA),
  metaheuristic (CMA-ES, GA, SA, PSO, basin hopping), QOC-specific (Krotov,
  GOAT, GROUP, CRAB, T-GRAPE), and analytic composite pulses (BB1, SCROFULOUS,
  SK1, CORPSE, DRAG, STA, SLR, VERSE)
- **Physics extensions** — open-system Lindblad dynamics, automatic
  differentiation, uncertainty quantification, sensitivity analysis
- **Hardware backends** — CPU, CUDA, Metal, with hybrid execution planning
- **Pulse export** — Bruker, JEOL, EPR, Pulseq, Qiskit, QuilT, QUA, Pulser

## Layer model

Pulsar's source tree follows a strict five-layer dependency model. Higher
layers may only depend on lower ones.

| Layer | Subdirectory | Purpose |
|---|---|---|
| 1a | [`src/Types/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Types) | Core system, target, control types |
| 1b | [`src/Computation/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Computation) | Propagators, ensemble, MAS, Bloch |
| 1c | [`src/Backend/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Backend) | Hardware, parallelism, scheduling |
| 2  | [`src/Physics/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Physics) | Objectives, penalties, gradients, Lindblad |
| 3  | [`src/Optimization/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Optimization) | All optimizers |
| 4  | [`src/IO/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/IO), [`src/Runtime/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Runtime), [`src/Utilities/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Utilities) | Infrastructure |
| 5  | [`src/Application/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Application) | Domain wrappers (MR, QC) |

## Where to start

- New users: [Installation](installation.md) → [Quickstart](quickstart.md)
- Theory background: [Hamiltonians](theory/hamiltonians.md), [Propagators](theory/propagators.md), [Fidelity](theory/fidelity.md)
- Picking an algorithm: [GRAPE](algorithms/grape.md), [Second-order](algorithms/second_order.md), [QOC-specific](algorithms/qoc_specific.md)
- Domain-specific guides: [NMR](domains/nmr.md), [QC platforms](domains/qc_platforms.md), …

## What's new

- **v0.2.1** ([release notes](release_notes/v0.2.1.md)) — patch release
  finishing the Julia 1.12 threading fix sweep (`Threads.nthreads()` →
  `Threads.maxthreadid()` in PSO, GA, DE, and `ensemble_grad!`). v0.2.1 is
  the version to install on Julia 1.12+.
- **v0.2.0** ([release notes](release_notes/v0.2.0.md)) — native SU(2) / Spin-I /
  Trotter propagators, matching fidelity metrics, an SU(2) duration-sweep
  helper, GRAPE-family kernel refinements, and the first round of Julia 1.12
  LAPACK + threading compatibility fixes. Registry-ready: `[compat]` entries
  added, TagBot workflow installed.

## License

Apache License, Version 2.0.
