# API — Types

Core system, target, and control types. Source:
[`src/Types/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Types).

## Quantum systems

```@docs
QuantumSystem
QubitSystem
qubit_system
```

## Magnetic-resonance systems

```@docs
MRSpinSystem
mr_system
HeteronuclearSystem
heteronuclear_system
EPRSpinSystem
epr_system
MASSpinSystem
mas_spin_system
BlochSystem
bloch_system
DNPSpinSystem
dnp_system
```

## Quantum-computing platform systems

```@docs
TransmonSystem
transmon_system
TrappedIonSystem
trapped_ion_system
NeutralAtomSystem
neutral_atom_system
SpinQubitSystem
spin_qubit_system
NVCenterSystem
nv_center_system
```

## Targets and controls

```@docs
QuantumTarget
state_target
unitary_target
ControlSequence
random_controls
zero_controls
```

## Fidelity metric names

The Φ^X_Y grid — see [Fidelity metrics](../theory/fidelity.md).

```@docs
canonical_metric
metric_label
```

## Spin operators

```@docs
spin_op
spin_state
hamiltonian
```
