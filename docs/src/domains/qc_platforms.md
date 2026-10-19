# Quantum-computing platforms

Pulsar ships first-class support for five qubit modalities, each with its own
system type, native gate set, noise models, and optimization wrapper.

| Platform | System type | Wrapper | Source |
|---|---|---|---|
| Superconducting | `TransmonSystem` | `optimcon(::TransmonSystem, ...)` | [`Application/QuantumComputing/Platforms/Superconducting.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/QuantumComputing/Platforms/Superconducting.jl) |
| Trapped ion | `TrappedIonSystem` | `optimcon(::TrappedIonSystem, ...)` | `Platforms/TrappedIon.jl` |
| Neutral atom | `NeutralAtomSystem` | `optimcon(::NeutralAtomSystem, ...)` | `Platforms/NeutralAtom.jl` |
| Spin qubit (QD) | `SpinQubitSystem` | `optimcon(::SpinQubitSystem, ...)` | `Platforms/SpinQubit.jl` |
| NV center | `NVCenterSystem` | `optimcon(::NVCenterSystem, ...)` | `Platforms/NVCenter.jl` |

## System constructors

Each platform has a domain-aware constructor with positional parameters
for the dominant physics and keyword arguments for everything else.

### Transmon

```julia
transmon = transmon_system(
    [5.1e9, 5.3e9],         # qubit frequencies (Hz)
    [-300e6, -300e6];       # anharmonicities (Hz)
    n_levels      = 3,
    coupling_hz   = [(1, 2) => 10e6],
    T1_s          = 50e-6,
    T2_s          = 30e-6,
    carrier_hz    = 5.2e9,
)
```

### Trapped ion

```julia
ion = trapped_ion_system(
    [10e6, 10e6],            # qubit frequencies (Hz)
    fill(0.1, 2, 1),         # Lamb–Dicke matrix η[ion, mode]
    [1e6];                   # axial mode frequency (Hz)
    Omega_hz   = 0.5e6,
    T1_s       = 1e-3,
    carrier_hz = 10e6,
)
```

### Neutral atom

Basis `|g⟩ = |0⟩`, `|r⟩ = |1⟩` (the Rydberg state is the upper level) in every
term: `H = Σ_i Δω_i |r⟩⟨r|_i + Σ_{i<j} V_ij |rr⟩⟨rr|_ij` with
`Δω_i = 2π(freq_hz[i] − carrier_hz)`, and `T1_s` decays `|r⟩ → |g⟩`. This is
Pulser's convention (`δ = −Δω`), so the `:pulser` exporter writes the phase
unchanged. Earlier versions used `Δω_i/2 σz_i` and a `|r⟩⟨g|` decay — the
detuning reversed relative to `V` and the decay pumping into `|r⟩`; runs with
a nonzero detuning or finite `T1_s` differ.

```julia
V = zeros(4, 4); V[1, 2] = V[2, 1] = 30e6      # Rydberg interaction (Hz)
ryd = neutral_atom_system(
    [0.0, 0.0, 0.0, 0.0],   # ground-Rydberg detunings (Hz)
    V;                       # interaction matrix
    blockade_regime = true,
    T1_s            = 100e-6,
    carrier_hz      = 0.0,
)
```

### Spin qubit (quantum dot)

```julia
J = zeros(2, 2); J[1, 2] = J[2, 1] = 1e9
sq = spin_qubit_system(
    [10e9, 10e9],            # Larmor frequencies (Hz)
    J;                        # exchange coupling matrix (Hz)
    g_factor   = 2.0,
    T2_s       = 10e-6,
    carrier_hz = 10e9,
)
```

### NV center

```julia
nv = nv_center_system(
    0.02;                                       # B0 (T)
    D_hz         = 2.87e9,
    E_hz         = 0.0,
    hyperfine_hz = 2.16e6,
    n_nuclei     = 1,
    T1_s         = 5e-3,
    T2_s         = 1e-3,
    subspace     = (-1, 0),
)
```

## Native gate sets

`native_gate_set(:platform)` returns a `NativeGateSet` listing the
natively-implementable single- and two-qubit gates with typical execution
times and fidelities. For an arbitrary SU(2) target, `zyz_decompose(U)` and
`zyz_sequence(U)` produce a native-basis decomposition.

```julia
gs = native_gate_set(:superconducting)
gs.single_qubit["SX"]                    # √X gate matrix
gs.two_qubit["CZ"]                       # CZ gate matrix
gs.gate_times_ns["CZ"]                   # ≈ 40 ns
gs.gate_fidelities["CZ"]                 # typical achieved fidelity

α, β, γ, δ = zyz_decompose(my_unitary)   # U = exp(iδ) Rz(α) Ry(β) Rz(γ)
seq = zyz_sequence(H_gate())             # [(:Rz,γ), (:Ry,β), (:Rz,α)]

inf = gate_infidelity(U_achieved, U_target)
```

Supported platforms: `:superconducting`, `:trapped_ion`, `:neutral_atom`,
`:spin_qubit`, `:nv_center`. The `single_qubit_gate_set()` and
`two_qubit_gate_set()` functions return the full library of pre-canned
gates (X/Y/Z/H/S/T/SX, Rx/Ry/Rz/Rn/U3, CNOT/CZ/SWAP/iSWAP/SQISWAP/MS/CRx/...).

| Function | Source |
|---|---|
| `single_qubit_gate_set`, `Rx`, `Ry`, `Rz`, `Rn`, `U3` | [`Gates/SingleQubitGates.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/QuantumComputing/Gates/SingleQubitGates.jl) |
| `two_qubit_gate_set`, `MS_gate`, `ZZθ_gate`, `CRx_gate` | [`Gates/TwoQubitGates.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/QuantumComputing/Gates/TwoQubitGates.jl) |
| `NativeGateSet`, `native_gate_set`, `zyz_decompose`, `zyz_sequence`, `gate_infidelity` | [`Gates/NativeGateSet.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/QuantumComputing/Gates/NativeGateSet.jl) |

## Noise models

Every platform integrates with three noise model categories.

### Quasi-static noise (slow parameter drift)

`QuasiStaticNoise(parameter, sigma; distribution, n_samples)` specifies a
slow stochastic parameter (frequency, coupling, amplitude) sampled by
Gauss-Hermite-like quadrature. `robust_optimcon_qs` builds a drift-Hamiltonian
ensemble and runs ensemble GRAPE:

```julia
σz = ComplexF64[1 0; 0 -1]
H_fn(p) = π * p[1] * σz                       # detuning noise

noise  = [QuasiStaticNoise(:freq, 100e3; n_samples = 7)]
result = robust_optimcon_qs(H_fn, [σz/2], noise,
                             unitary_target(X_gate()), ctrl;
                             config = GRAPEConfig(max_iter = 200))

mean_F, std_F, min_F = evaluate_qs_robustness(H_fn, [σz/2], noise,
                                               target, ctrl)
```

### Markovian (Lindblad) noise

`markovian_noise(dim; T1_s, T2_s, depol_rate)` assembles standard jump
operators. Building blocks `amplitude_damping(T1)`, `phase_damping(T2, T1)`,
`depolarizing_channel(p)` are available individually.

```julia
noise  = markovian_noise(2; T1_s = 50e-6, T2_s = 30e-6)
result = lindblad_optimcon(H_drift, [Hx, Hy], noise,
                            state_tgt, ctrl;
                            config = GRAPEConfig(max_iter = 200))
```

GRAPE then runs in N²-dimensional Liouville space (see
[Propagators](../theory/propagators.md#lindblad-evolution)).

### Non-Markovian noise (filter function)

For 1/f, ohmic, or arbitrary classical-noise spectra, the filter-function
formalism penalises the dephasing infidelity `χ = (1/π) ∫ S(ω) |F(ω)|² / ω² dω`:

```julia
spec   = pink_noise_spectrum(2π * 1e3, 2π * 1e7, 200; A = 1e6)
H_n    = ComplexF64[1 0; 0 -1]                  # σz dephasing axis
result = optimcon_ff(sys, target, ctrl, H_n, spec;
                     config    = GRAPEConfig(max_iter = 200),
                     ff_weight = 1.0)
```

Spectrum constructors: `pink_noise_spectrum`, `white_noise_spectrum`,
`ohmic_noise_spectrum`. Lower-level utilities `compute_filter_function(Us, H_noise, dt, ω)`
and `filter_function_infidelity(Us, H_noise, dt, spec)` evaluate the FF
directly.

## Verification

### Randomized benchmarking

`Verification/RandomizedBenchmarking.jl` simulates Clifford-RB and interleaved
RB for evaluating optimized pulses:

```julia
m_vals = [1, 2, 5, 10, 20, 50, 100]

# Survival p(m) under a user-supplied gate-noise model
noise_fn(U) = U                                # ideal — replace with channel
p = rb_survival_probability(m_vals, noise_fn; n_sequences = 50)

result = estimate_epc(m_vals, p)               # → RBResult
@show result.epc result.epg result.r_squared

# Interleaved RB for a specific gate
rb_ref, rb_int, epc_gate =
    interleaved_rb(X_gate(), m_vals; gate_fn = noise_fn)
```

| Function | Purpose |
|---|---|
| `rb_sequence(m; n_qubits, rng)` | Length-`m` random Clifford sequence (auto-recovery) |
| `rb_survival_probability(m_values, gate_fn; n_sequences)` | Mean p(m) over random sequences |
| `fit_rb_decay(m, p)` | LS fit of `p = A·rᵐ + B` |
| `estimate_epc(m, p; n_qubits)` | EPC + EPG via the depolarizing-channel formula |
| `interleaved_rb(U_gate, m; gate_fn)` | Per-gate EPC by comparing standard vs. interleaved decay |

Process tomography utilities live in
[`Verification/ProcessTomography.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/QuantumComputing/Verification/ProcessTomography.jl).

## Common patterns

- **Single-qubit gate set on a transmon with leakage suppression** — DRAG
  initial guess + GRAPE refinement + leakage penalty
- **Mølmer–Sørensen on trapped ions** — phase-modulated pulses optimized
  against motional-mode heating
- **Rydberg CZ on neutral atoms** — adiabatic + quench schemes with blockade
  constraint
- **NV-center gates under hyperfine drift** — `RobustConfig` over the
  `^14N` hyperfine ensemble
