# Propagators

Pulsar's optimization kernels rely on piecewise-constant propagation of the
time-dependent Schrödinger or Liouville equation.

## Piecewise-constant evolution

For a control sequence with `n_t` slices of duration `Δt = T / n_t`:

```math
\begin{aligned}
U_k \;&=\; \exp\!\left(-\,i\,H_k\,\Delta t\right), \\[2pt]
H_k \;&=\; H_{\text{drift}} \;+\; \sum_{c} u_{c}[k]\, H_c .
\end{aligned}
```

The full propagator is the time-ordered product `U_total = U_n ⋯ U_2 U_1`.

## Forward / backward passes

GRAPE-style gradients require both **forward** and **backward** propagators:

| Function | Returns | Use |
|---|---|---|
| `compute_propagator(H, dt)` | `exp(-i H Δt)` | Single slice |
| `build_total_hamiltonian(sys, ctrl, k)` | `H_k` for slice `k` | Symbolic build |
| `compute_forward_propagators(sys, ctrl)` | `U_k = exp(-i H_k Δt)` for all `k` | GRAPE forward |
| `compute_backward_propagators(sys, ctrl)` | Adjoint partials | GRAPE backward |

The forward/backward arrays let GRAPE compute the gradient
`∂F/∂u_c[k]` analytically without finite differences (see
[GRAPE](../algorithms/grape.md)).

## MAS (magic-angle spinning) propagators

For solid-state NMR with sample rotation, the Hamiltonian is time-periodic.
Pulsar provides a Wigner-rotation-based stepper:

| Function | Purpose |
|---|---|
| `build_mas_hamiltonian(sys, t)` | Time-dependent MAS drift |
| `rotate_spin_system(sys, Ω)` | Apply Wigner D² rotation |
| `wigner_d2`, `wigner_D2` | Rank-2 Wigner matrices |
| `powder_grid(n)` | Powder averaging Euler grid |
| `compute_grape_gradient_powder` | Gradient over a powder ensemble |

The integration step length is set internally so that one MAS revolution is
resolved.

## Bloch propagator

For MRI / single-spin slice-profile design, `BlochPropagator.jl` provides a
classical (3-vector) integrator:

| Function | Use |
|---|---|
| `bloch_forward_pass(sys::BlochSystem, ctrl::MRIControlSequence)` | Magnetization trajectory `[3 × n_iso]` at `T` |
| `bloch_adjoint_pass(...)` | Adjoint for gradient backprop |
| `bloch_fidelity(sys, M_final, M_target)` | Inner-product fidelity weighted by ρ₀ |
| `slice_profile_fidelity(sys, M_final, M_target)` | Transverse-only profile match |

The forward pass embeds T1 / T2 relaxation per isochromat (from
`BlochIsochromat.T1`, `T2`) and applies `(B1_x, B1_y, G·r)` per slice.

## Lindblad evolution

Open-system dynamics are propagated in Liouville space via column-stacking
vectorisation. The Lindblad master equation

```math
\begin{aligned}
\dot\rho \;=\;& -i\,[H,\rho]
   \;+\; \sum_{j} \gamma_j \left(
        L_j\,\rho\,L_j^{\dagger}
        \;-\; \tfrac{1}{2}\,\{L_j^{\dagger} L_j,\,\rho\}
   \right)
\end{aligned}
```

becomes a linear ODE on the vectorised density matrix `vec(ρ)`:

```math
\begin{aligned}
\dot{\mathrm{vec}}(\rho) \;=\;& \mathcal{L}\,\mathrm{vec}(\rho), \\[4pt]
\mathcal{L} \;=\;& -i\,\bigl(I \otimes H \;-\; H^{\top}\!\otimes I\bigr) \\
                 &+\; \sum_{j} \gamma_j \left(
                       \bar L_j \otimes L_j
                       \;-\; \tfrac{1}{2}\bigl(
                              I \otimes L_j^{\dagger} L_j
                              \;+\; (L_j^{\top} \bar L_j) \otimes I
                            \bigr)
                  \right).
\end{aligned}
```

The Liouvillian is assembled from a drift Hamiltonian, jump operators, and
control Hamiltonians:

```julia
L = lindblad_system_from_jump_ops(H_drift,
                                   jump_ops,
                                   decay_rates,
                                   H_controls)
```

Pulsar provides:

| Function | Purpose |
|---|---|
| `vec_rho(ρ)` / `mat_rho(v)`               | Column-stacking vec / inverse |
| `pure_state_to_vec_rho(ψ)`                | `|ψ⟩⟨ψ|` → vec |
| `build_drift_liouvillian(H, jumps, rates)` | Drift `L₀` |
| `build_control_liouvillian(H_c)`          | Hamiltonian control in `L`-space |
| `lindblad_grad_prefactor(...)`            | Adjoint chain-rule prefactor |

GRAPE in Liouville space then proceeds exactly as in Hilbert space,
replacing `U_k = exp(-i H_k Δt)` by `Φ_k = exp(L_k Δt)`. The matrix
exponential of the (non-Hermitian) Liouvillian is computed by the same
back-end registry (Padé / eigen / Chebyshev / Newton / Magnus).

## Propagator back-end registry

`compute_propagator(H, dt)` defaults to a Padé approximant. For
specialised regimes you can dispatch on a propagator type:

| Type | Best for |
|---|---|
| `PadePropagator()`      | General dense matrices (default) |
| `EigenPropagator()`     | Hermitian, repeated `dt`, small dim |
| `ChebyshevPropagator()` | Sparse / banded `H`, smooth spectra |
| `NewtonPropagator()`    | Sparse `H` with selective Krylov subspace |
| `MagnusPropagator(order)` | Time-dependent `H(t)` to high order |

```julia
U = compute_propagator(H, dt, ChebyshevPropagator())
```

Time-dependent MAS Hamiltonians are typically integrated with
`MagnusPropagator(2)` (second-order Magnus) per rotor sub-period.

## Specialised low-dimensional propagators (v0.2.0+)

For spin-1/2 and single-spin problems, Pulsar ships three specialised
propagators that avoid the full matrix-exponential O(dim³) cost:

- **`SU2Propagator`** — Cayley–Klein scalar propagator for phase-only
  constant-Rabi pulses on spin-1/2. The propagator factors as
  `P_k(φ_n) = [[α_k, -β_n_k*], [β_n_k, α_k*]]` with α/β precomputed once via
  [`precompute_su2_params`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Computation/SU2Propagator.jl).
  Per-step cost: 4 complex multiplications per offset (vs O(dim³) for matrix
  exp).
- **`SpinIPropagator`** — Single-spin scalar propagation for spin-I systems
  (any half-integer or integer spin).
- **`TrotterPropagator`** — Trotter–Suzuki decomposition option for cases
  where the default `compute_propagator` (LAPACK Hermitian eigendecomposition)
  is not the best fit; the order is configurable.

Matching fidelity metrics live in `src/Physics/SU2Objectives.jl` and
`src/Physics/SpinIObjectives.jl`. For sweeping the pulse duration of an
SU(2) problem to locate the shortest pulse meeting a target fidelity, use
[`sweep_su2_duration`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Utilities/SU2DurationSweep.jl).

See the [v0.2.0 release notes](../release_notes/v0.2.0.md) for the rationale.
