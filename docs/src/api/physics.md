# API — Physics

Objectives, penalties, gradients, Lindblad, autodiff, UQ, sensitivity, MR
physics. Source:
[`src/Physics/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Physics).

## Objectives

```@docs
state_fidelity
gate_fidelity
dm_fidelity
compute_fidelity
fidelity_grad_prefactor
AbstractFidelityMetric
PropReal
PropModulus
PropSquared
PropAverage
StateReal
StateModulus
StateSquared
DMReal
DMModulus
DMSquared
DMUhlmann
```

## Penalties

```@docs
AbstractPenalty
NormSquarePenalty
SpilloutPenalty
AmplitudeSpilloutPenalty
SmoothnessPenalty
EnergyPenalty
```

## Gradients

```@docs
compute_grape_gradient
finite_difference_gradient
grape_optimize_ensemble
```

## Lindblad

```@docs
lindblad_system_from_jump_ops
lindblad_grad_prefactor
build_drift_liouvillian
build_control_liouvillian
vec_rho
mat_rho
pure_state_to_vec_rho
```

## Automatic differentiation

```@docs
compute_gradient_autodiff
verify_gradient_autodiff
```

## Uncertainty & sensitivity

```@docs
estimate_uncertainty
UQConfig
UncertaintyResult
compute_sensitivity
SensitivityConfig
SensitivityResult
```

## MR-specific physics

```@docs
band_selective_fidelity
band_selective_gradient
shift_system
dnp_polarization_fidelity
electron_polarized_state
nuclear_polarization_operator
sar_penalty
sar_gradient
slew_rate_penalty
slew_rate_gradient
BandWeight
```
