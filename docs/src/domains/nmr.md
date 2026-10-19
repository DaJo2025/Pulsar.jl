# NMR — solution-state pulse design

Solution NMR is Pulsar's most developed application domain. The MR-layer
([`src/Application/MR/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Application/MR))
provides a high-level interface that ultimately calls the same
`compute_grape_gradient` and `grape_lbfgsb_optimize` machinery used elsewhere.

Two complementary entry points are available:

1. **High-level `optimcon(sys, target, ctrl; …)`** — pass a system, a
   `QuantumTarget`, and a `ControlSequence`; the wrapper builds drifts,
   penalties, and optimiser and returns an `OptimizationResult`.
2. **Low-level `MRControl(...) ; optimcon(ctrl, guess)`** — a single pulse
   object carrying drifts, control operators, initial/target states, power
   levels, time grid, and method/iteration knobs.

Use option 1 for ordinary GRAPE-style runs; use option 2 when you need
multi-isochromat ensembles, custom penalties, tracking points, or callbacks.

## Building a system

`mr_system` accepts a vector of isotope strings:

```julia
sys_H   = mr_system("1H")              # single proton (dim = 2)
sys_HC  = mr_system(["1H", "13C"])    # 1H–13C pair  (dim = 4)
sys_2H  = mr_system(["1H", "1H"])     # two protons   (dim = 4)
```

Supported isotopes follow `SPIN_QUANTUM_NUMBER` and include `1H`, `2H`,
`13C`, `15N`, `19F`, `31P`, etc. Drift Hamiltonians are built via
`hamiltonian(sys; …)`:

```julia
H = hamiltonian(sys_HC;
                shifts_ppm   = [1.5, 30.0],
                B0_tesla     = 14.1,
                offset_hz    = 0.0,
                couplings_hz = [0.0 145.0; 145.0 0.0])
```

For a frequency-offset ensemble (broadband pulse design over ±6 kHz):

```julia
drifts = [hamiltonian(sys_H; offset_hz = Δf)
          for Δf in range(-6000, 6000, length = 25)]
```

## Single-spin pure states

```julia
ψ_init = spin_state(sys_H, :Iz)        # +z eigenstate
ψ_targ = spin_state(sys_H, :mIy)       # −y target after a 90°ₓ pulse
```

`spin_state(sys, :Iz | :mIz | :Ix | :mIx | :Iy | :mIy)` is defined for
single spin-½ systems. For multi-spin Hilbert spaces build the product
state directly with `kron`.

## Spin operators

```julia
Lx_H = spin_op(sys_HC, :Ix, "1H")     # Ix on the proton
Lz_C = spin_op(sys_HC, :Iz, "13C")    # Iz on the carbon
Lx_1 = spin_op(sys_HC, :Ix, 1)        # Ix on spin index 1
```

## Heteronuclear systems

For multi-carrier experiments (separate rotating frames for each species):

```julia
sys_H  = mr_system("1H")
sys_C  = mr_system("13C")
hsys   = heteronuclear_system([sys_H, sys_C], [600e6, 150e6])
H      = hamiltonian(hsys; offsets_hz   = [200.0, 50.0],
                            J_couplings = Dict((1, 2) => 145.0))
```

Each subsystem keeps its own `Iz`/`Ix`/`Iy` operators, embedded onto the
joint Hilbert space; `H_controls` is `[Ix₁, Iy₁, Ix₂, Iy₂, …]`.

## Closed-system optimisation — `optimcon` form

```julia
sys    = mr_system("1H")
target = state_target(spin_state(sys, :mIz))   # 180° inversion
ctrl   = random_controls(sys, 100e-6, 200; amplitude = 0.5)

config = LBFGSConfig(max_iter = 200, verbose = false)
result = optimcon(sys, target, ctrl; config = config)
```

`optimcon(::MRSpinSystem, ::QuantumTarget, ::ControlSequence)` accepts a
`bands` keyword to enable a band-selective sweep:

```julia
bands = [BandWeight(-500.0, 1.0),
         BandWeight(   0.0, 1.0),
         BandWeight(+500.0, 1.0),
         BandWeight(+2500.0, -0.5)]    # negative weight = stop band
result = optimcon(sys, target, ctrl; config = config, bands = bands)
```

For heteronuclear systems use the dedicated dispatch:

```julia
optimcon(hsys, target, ctrl;
         offsets_hz   = [0.0, 0.0],
         J_couplings  = Dict((1, 2) => 145.0),
         config       = config)
```

## Closed-system optimisation — `MRControl` form

`MRControl` is a kwargs-only pulse object that bundles everything the
optimiser needs:

```julia
Iz = spin_op(sys_H, :Iz)
Ix = spin_op(sys_H, :Ix)
Iy = spin_op(sys_H, :Iy)

H_drift = 2π * 0.0 .* Iz                       # on resonance
ψ0      = spin_state(sys_H, :Iz)
ψ1      = spin_state(sys_H, :mIz)              # 180° inversion

ctrl = MRControl(
    drifts     = [H_drift],
    operators  = [Ix, Iy],
    rho_init   = [ψ0],
    rho_targ   = [ψ1],
    pwr_levels = [2π * 5e3, 2π * 5e3],          # rad/s per channel
    pulse_dt   = fill(1e-6, 200),                # 200 × 1 µs slices
    method     = :lbfgs,
    max_iter   = 500,
    fidelity   = :state_squared,                 # Φ^S_Q, |⟨ψ1|ψ(T)⟩|²
    verbose    = false,
)

guess  = randn(2, 200) .* 0.1                    # [n_ctrl × n_ts]
result = optimcon(ctrl, guess)
```

`drifts` is a *vector* of Hamiltonians: a single entry runs nominal GRAPE,
multiple entries form an ensemble (broadband/robust optimisation). The
`fidelity` keyword takes a state-level metric — `:state_real` (Φ^S_R, the
default), `:state_modulus` (Φ^S_M) or `:state_squared` (Φ^S_Q); see
[Fidelity metrics](../theory/fidelity.md). `method` chooses
the inner gradient algorithm — `:lbfgs`, `:lbfgs_b`, `:bfgs`, `:gd`,
`:cg`, `:nelder_mead`, `:adam`.

`MRControl` accepts `tracking::Vector{TrackingPoint}` for state-trajectory
checkpoints during forward propagation, and a `callback::Function` for
per-iteration hooks (e.g. checkpointing, see
[Checkpointing](../advanced/checkpointing.md)).

## Open-system (relaxation) workflow — Lindblad

When relaxation cannot be ignored (typically `T1`, `T2` short relative to
pulse length), use `LindbladMRControl`:

```julia
ψ0    = spin_state(sys_H, :Iz)
ψ1    = spin_state(sys_H, :mIz)
γ_T2  = 1 / 0.1                                  # 100 ms T2
γ_T1  = 1 / 2.0                                  # 2 s T1

ctrl_L = LindbladMRControl(
    drifts      = [H_drift],
    operators   = [Ix, Iy],
    jump_ops    = [Ix - im*Iy, Iz],              # any 2D operators
    decay_rates = [γ_T2, γ_T1],
    rho_init    = [ψ0],
    rho_targ    = [ψ1],
    pwr_levels  = [2π * 5e3, 2π * 5e3],
    pulse_dt    = fill(1e-6, 200),
    method      = :lbfgs,
    max_iter    = 500,
    fidelity    = :dm_squared,                   # Φ^D_Q (or :dm_real, Φ^D_R)
    verbose     = false,
)

result = optimcon(ctrl_L, randn(2, 200) .* 0.1)
```

Internally this converts everything to Liouville space and calls
`grape_lindblad_kernel` (see
[`Physics/Lindblad.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/Lindblad.jl)).
`mr_relaxation` is available for building physically-motivated jump-op
sets from `T1`/`T2` parameters.

## Building blocks summary

| Function | Purpose |
|---|---|
| `mr_system(isotopes)` | Construct an `MRSpinSystem` |
| `heteronuclear_system(subs, carriers)` | Multi-frame heteronuclear system |
| `hamiltonian(sys; …)` | Drift Hamiltonian with shifts / J / offsets |
| `spin_op(sys, :Ix, k)` | Single-spin operator on the joint space |
| `spin_state(sys, :Iz)` | Pure spin-½ eigenstate |
| `mr_relaxation(...)` | Build T1/T2 jump operators |
| `density_matrix(...)` | Pure / mixed `ρ` constructor |
| `band_selective_fidelity` / `band_selective_gradient` | Frequency-band-targeted objectives |
| `shift_system(sys, Δ)` | Frequency-shifted copy for ensemble averaging |
| `MRControl` / `LindbladMRControl` | Closed/open-system pulse objects |
| `optimcon(ctrl[, guess])` | Run the bundled optimiser |
| `optimcon(sys, target, ctrl; …)` | High-level system-target wrapper |
| `grape_state_kernel`, `grape_lindblad_kernel` | Forward fidelity kernels |
| `grape_tracking_kernel` | Trajectory checkpoints |
| `fidelity_forward(w, ctrl)` | Derivative-free fidelity |

## Convergence checkpointing

Pass a `callback` to `MRControl` / `LindbladMRControl` to fire every
iteration. Combine with Pulsar's unified
[`Checkpoint`](../advanced/checkpointing.md) for safe long-running
optimisations (broadband 180°, INEPT, multi-band selective).

```julia
cb = (iter, F; grad = NaN, evals = 0) -> begin
    iter % 50 == 0 && save_checkpoint(
        "ckpt.jls",
        Checkpoint(w_ref[], F, n_ctrl, n_ts;
                   domain = :mr, drive_max_hz = 5e3, T_pulse = 200e-6),
    )
end
ctrl = MRControl(...; callback = cb)
```

## Worked patterns

- **Broadband 180° inversion** — L-BFGS-GRAPE over a 21-isochromat
  frequency ensemble (±6 kHz) with `[SmoothnessPenalty, SpilloutPenalty]`.
- **INEPT polarisation transfer** — heteronuclear ¹H–¹³C, 4 controls
  (¹Hₓ, ¹Hᵧ, ¹³Cₓ, ¹³Cᵧ), `J = 145 Hz`, `1 ms`.
- **Selective excitation** — band-selective fidelity weighted across
  multi-band passbands and stop-bands.
