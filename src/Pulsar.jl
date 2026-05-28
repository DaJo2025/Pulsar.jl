"""
    Pulsar — Pulse Design Library for Spin Control Algorithms and Rollout

Pulsar.jl is a Julia package for quantum control optimization.
It provides a comprehensive suite of algorithms and tools for designing optimal
control pulses that steer quantum systems toward desired target states or
operations with high fidelity.

# Overview

The package is structured around three central concepts:

1. **Quantum System** — a physical description of the system to be controlled,
   including its drift Hamiltonian, control Hamiltonians, and Hilbert-space
   dimension.
2. **Optimization Algorithm** — a strategy for finding control amplitudes that
   maximize (or minimize) a fidelity measure. Pulsar implements gradient-based
   methods (GRAPE, L-BFGS, Newton-CG, etc.), derivative-free methods (CMA-ES,
   Nelder-Mead, PSO, etc.), constrained and robust variants, and trust-region methods.
3. **Backend** — the computational substrate used to evaluate matrix exponentials
   and propagators. Supported backends are CPU (with SIMD/threading), CUDA, Metal,
   and a hybrid planner that automatically partitions work across devices.

# Quick Start

```julia
using Pulsar

# 1. Define a spin-1/2 system driven by two quadrature controls
sys = spin_system(
    drift  = 0.5 * [1 0; 0 -1],    # σ_z / 2
    controls = [[0 1; 1 0],          # σ_x
                [0 -im; im 0]],      # σ_y
    n_timesteps = 100,
    duration    = 10.0,
)

# 2. Set the target (X gate)
target = QuantumTarget(; U_target = [0 1; 1 0])

# 3. Run GRAPE
result = grape_optimize(sys, target; config = GRAPEConfig())

@show result.fidelity
@show result.iterations
```

# Backends

- [`CPUBackend`](@ref) — multi-threaded CPU evaluation with optional SIMD via
  LoopVectorization.
- [`CUDABackend`](@ref) — GPU acceleration via CUDA.jl (requires an NVIDIA GPU
  and the CUDA.jl extension).
- [`MetalBackend`](@ref) — GPU acceleration via Metal.jl (requires Apple Silicon
  and the Metal.jl extension).
- [`HybridBackend`](@ref) — automatic work-stealing planner that distributes
  ensemble members across all available devices.

# Algorithms

| Function                    | Method                                    |
|-----------------------------|-------------------------------------------|
| `grape_optimize`            | GRadient Ascent Pulse Engineering (GRAPE) |
| `bfgs_optimize`             | Broyden-Fletcher-Goldfarb-Shanno (BFGS)   |
| `lbfgs_optimize`            | Limited-memory BFGS                       |
| `newton_optimize`           | Newton-CG (Hessian-free)                  |
| `cmaes_optimize`            | Covariance Matrix Adaptation ES           |
| `nelder_mead_optimize`      | Nelder-Mead simplex                       |
| `pso_optimize`              | Particle Swarm Optimization               |
| `constrained_optimize`      | Augmented Lagrangian / penalty methods    |
| `robust_optimize`           | Ensemble / worst-case robustness          |
| `trust_region_optimize`     | Trust-region Newton                       |
| `multi_objective_optimize`  | Pareto-front multi-objective              |

# Advanced Features

- Automatic differentiation via ForwardDiff and Zygote.
- Uncertainty quantification (Monte-Carlo and quasi-Monte Carlo).
- Global sensitivity analysis (Sobol indices via Saltelli).
- Checkpoint / resume for long optimizations.
- Algorithm selection heuristic (`recommend_optimizer`).
- Performance monitoring and benchmark utilities.

See the online documentation for a complete API reference and tutorials.
"""
module Pulsar

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Dates
using Random
using Distributed

# ---------------------------------------------------------------------------
# Third-party imports (optional; loaded lazily where used)
# ---------------------------------------------------------------------------
# The following are loaded inside their respective subfiles via try/catch:
#   ForwardDiff, Zygote  → Advanced/AutomaticDifferentiation.jl
#   JSON3                → Advanced/CheckpointAndResume.jl
#   Plots                → Utilities/VisualizationUtilities.jl
#   BenchmarkTools       → benchmark/BenchmarkSuite.jl (not in module)
#   FFTW, Optim          → not currently used in core; available to users

# ---------------------------------------------------------------------------
# Layer 1a: Type definitions (no computation, no external deps)
# ---------------------------------------------------------------------------
include("Types/QuantumSystem.jl")
include("Types/Targets.jl")
include("Types/ControlSequence.jl")

# ---------------------------------------------------------------------------
# Layer 1b/1c interleave: parallelism primitives must be available to
# `Computation/EnsembleMap.jl` (which uses `@threadsif`), so CPU parallelism
# is loaded before the computation layer rather than after CPUBackend.
# ---------------------------------------------------------------------------
include("Backend/Parallelism/CPUParallelization.jl")

# ---------------------------------------------------------------------------
# Layer 1b: Numerical computation primitives
# ---------------------------------------------------------------------------
include("Computation/Propagators.jl")
include("Computation/PropagatorRegistry.jl")
include("Computation/PropagatorCache.jl")
include("Computation/EnsembleMap.jl")
include("Computation/SU2Propagator.jl")
include("Computation/SpinIPropagator.jl")
include("Computation/TrotterPropagator.jl")

# ---------------------------------------------------------------------------
# Layer 1c: Backend — hardware abstraction (remaining sub-concerns)
# ---------------------------------------------------------------------------
include("Backend/Hardware/CPUBackend.jl")
include("Backend/Hardware/CUDABackend.jl")
include("Backend/Hardware/MetalBackend.jl")
include("Backend/Scheduling/HybridExecution.jl")

# ---------------------------------------------------------------------------
# Layer 2: Physics — objective functions, gradients, open-system models
# ---------------------------------------------------------------------------
include("Physics/Objectives.jl")
include("Physics/SU2Objectives.jl")
include("Physics/SpinIObjectives.jl")
include("Physics/Penalties.jl")
include("Physics/Gradients.jl")
include("Physics/Lindblad.jl")
include("Physics/AutoDiff.jl")
# Unified noise / ensemble abstraction (Theme 5)
include("Physics/NoiseModels.jl")
# Note: Physics/UncertaintyQuantification.jl and Physics/Sensitivity.jl depend
# on OptimizationResult / GRAPEConfig (defined in Optimization/GRAPE.jl) and
# are therefore included after the optimization layer below.

# ---------------------------------------------------------------------------
# Layer 3: Optimization algorithms
# ---------------------------------------------------------------------------
# Runtime self-check helpers (used by optimizers guarded by check_invariants)
include("Optimization/Invariants.jl")

# Generic ensemble-objective wrapper (core type + :mean/:worst_case/:cvar
# aggregators + `build_ensemble_from_systems`). `grape_optimize_ensemble`
# delegates through this core below.
include("Optimization/Ensemble/EnsembleObjective.jl")
include("Optimization/Ensemble/SystemBuilder.jl")

# Control parameterisation hierarchy (Theme 2)
include("Optimization/Parameterization.jl")

# Core GRAPE
include("Optimization/GRAPE.jl")

# Second-order methods
include("Optimization/SecondOrder/SecondOrderMethods.jl")

# Derivative-free / direct search
include("Optimization/DirectSearchMethods.jl")

# Shared finite-difference utility (used by RobustOpt.jl)
include("Utilities/FiniteDifference.jl")

# Constrained and robust
include("Optimization/Constrained/ConstrainedOpt.jl")
include("Optimization/Robust/RobustOpt.jl")

# Perturbation-based ensemble builder (needs RobustOpt helpers — must load
# after `Robust/RobustOpt.jl`).
include("Optimization/Ensemble/PerturbationBuilder.jl")

# Gradient sub-components (step size schedules live here)
include("Optimization/Gradient/AdaptiveStepSize.jl")

# Multi-objective
include("Optimization/MultiObjective/MultiObjectiveOptimization.jl")

# Direct / local derivative-free
include("Optimization/Direct/SimplexSearch.jl")
include("Optimization/Direct/PatternSearch.jl")
include("Optimization/Direct/QuadraticModels.jl")
include("Optimization/Direct/ConstrainedDirect.jl")

# Metaheuristic
include("Optimization/Metaheur/GA.jl")
include("Optimization/Metaheur/SA.jl")
include("Optimization/Metaheur/Swarm.jl")
include("Optimization/Metaheur/CMAES.jl")
include("Optimization/Metaheur/MC.jl")
include("Optimization/Metaheur/BasinHopping.jl")

# Shared gradient-method helpers (must precede Generic + QOC gradient files)
include("Optimization/Gradient/_LineSearch.jl")
include("Optimization/Gradient/_SharedHelpers.jl")

# Generic gradient-based
include("Optimization/Gradient/Generic/FirstOrder.jl")
include("Optimization/Gradient/Generic/ConjugateGradient.jl")
include("Optimization/Gradient/Generic/QuasiNewton.jl")
include("Optimization/Gradient/Generic/SecondOrder.jl")

# QOC-specific gradient methods
include("Optimization/Gradient/QOC/GRAPEFamily.jl")
include("Optimization/Gradient/QOC/Krotov.jl")
include("Optimization/Gradient/QOC/BasisMethods.jl")
include("Optimization/Gradient/QOC/HighOrderOC.jl")
include("Optimization/Gradient/QOC/TGRAPE.jl")
include("Optimization/Gradient/QOC/CRAB.jl")

# Analytic pulse design
include("Optimization/Analytic/Composite.jl")
include("Optimization/Analytic/SmallTipAngle.jl")
include("Optimization/Analytic/SLR.jl")
include("Optimization/Analytic/VERSE.jl")

# Physics files that depend on OptimizationResult / GRAPEConfig (post-optimization)
include("Physics/UncertaintyQuantification.jl")
include("Physics/Sensitivity.jl")
# Theme 9 — depends on CompositePulseSegment (Optimization/Analytic/Composite.jl)
include("Physics/PulseComposition.jl")

# ---------------------------------------------------------------------------
# Layer 4: Infrastructure — I/O, runtime, utilities
# ---------------------------------------------------------------------------

# I/O subsystem
include("IO/Output.jl")
using .Output
include("IO/Checkpoint.jl")
include("IO/PulseExport.jl")

# Runtime instrumentation
include("Runtime/PerformanceMonitoring.jl")
include("Runtime/IterationCallback.jl")
include("Runtime/AlgorithmRegistry.jl")
include("Runtime/AlgorithmSelection.jl")

# Pure utilities
include("Utilities/ParameterValidation.jl")
include("Utilities/VisualizationUtilities.jl")
include("Utilities/SU2DurationSweep.jl")
# Theme 12 — pulse-spectrum and Bloch-sweep analysis utilities
include("Utilities/PulseAnalysis.jl")
# Canonical problem library (single-qubit, two-qubit, INEPT, …)
include("Utilities/ProblemLibrary.jl")

# ---------------------------------------------------------------------------
# GPU device registry (uses _METAL_LOADED/_CUDA_LOADED from Hardware files)
# ---------------------------------------------------------------------------
include("Backend/Scheduling/DeviceRegistry.jl")

# ---------------------------------------------------------------------------
# Layer 1a extension: NMR spin system types (solution NMR + heteronuclear)
# ---------------------------------------------------------------------------
include("Types/NMRSpinSystem.jl")
include("Types/EPRSpinSystem.jl")
include("Types/MASSpinSystem.jl")
include("Types/BlochSystem.jl")
include("Types/DNPSpinSystem.jl")
include("Types/TransmonSystem.jl")
include("Types/TrappedIonSystem.jl")
include("Types/NeutralAtomSystem.jl")
include("Types/SpinQubitSystem.jl")
include("Types/NVCenterSystem.jl")

# ---------------------------------------------------------------------------
# Layer 5: Application — domain-specific thin wrappers
# ---------------------------------------------------------------------------
include("Application/MR/LindbladMR.jl")
include("Application/MR/GRAPEState.jl")
include("Application/MR/GRAPEPhase.jl")
include("Application/MR/GRAPELindblad.jl")
include("Application/MR/GRAPETracking.jl")
include("Application/MR/OptControl.jl")

# MR-aware ensemble builder (needs MRControl / LindbladMRControl + MR kernels)
include("Application/MR/EnsembleBuilder.jl")

# New computation (after new types)
include("Computation/MASPropagators.jl")
include("Computation/WignerRotations.jl")
include("Computation/BlochPropagator.jl")

# New physics layer — depends on new types
include("Physics/MRPhysics.jl")

# New application layers
include("Application/MR/OptControlExtensions.jl")
include("Application/MR/_OrientationAggregation.jl")
include("Application/MR/SolidStateNMR/MASOptControl.jl")
include("Application/MR/EPR/EPROptControl.jl")
include("Application/MR/MRI/MRIOptControl.jl")
include("Application/MR/DNP/DNPLindblad.jl")
include("Application/MR/DNP/DNPOptControl.jl")

# ---------------------------------------------------------------------------
# Layer 5b: Quantum Computing application layer
# ---------------------------------------------------------------------------

# Gate libraries (no dependencies on platform types — load first)
include("Application/QuantumComputing/Gates/SingleQubitGates.jl")
include("Application/QuantumComputing/Gates/TwoQubitGates.jl")
include("Application/QuantumComputing/Gates/NativeGateSet.jl")

# Physical platform optimcon overloads (types are in Types/)
include("Application/QuantumComputing/Platforms/Common.jl")
include("Application/QuantumComputing/Platforms/Superconducting.jl")
include("Application/QuantumComputing/Platforms/TrappedIon.jl")
include("Application/QuantumComputing/Platforms/NeutralAtom.jl")
include("Application/QuantumComputing/Platforms/SpinQubit.jl")
include("Application/QuantumComputing/Platforms/NVCenter.jl")
# Unified QC optimcon context (after all platform types)
include("Application/QuantumComputing/OptControl.jl")
# Theme 6 — Open-system QC control (skeleton; depends on Physics/Lindblad.jl helpers)
include("Application/QuantumComputing/LindbladQCControl.jl")
# Shared AbstractOptimizationContext fallback (loaded last so all subtypes exist)
include("Application/OptControl.jl")

# Noise models (depend on lindblad_system_from_jump_ops from Physics/Lindblad.jl)
include("Application/QuantumComputing/NoiseModels/QuasiStatic.jl")
include("Application/QuantumComputing/NoiseModels/Markovian.jl")
include("Application/QuantumComputing/NoiseModels/NonMarkovian.jl")

# Verification tools (depend on gate libraries)
include("Application/QuantumComputing/Verification/RandomizedBenchmarking.jl")
include("Application/QuantumComputing/Verification/ProcessTomography.jl")

# ---------------------------------------------------------------------------
# Public API — System types
# ---------------------------------------------------------------------------
export QuantumSystem, SpinSystem, QubitSystem
export QuantumTarget
export ControlSequence
export OptimizationResult

# Algorithm configuration types
export GRAPEConfig, BFGSConfig, LBFGSConfig, NewtonConfig
export CMAESConfig, NelderMeadConfig, PSOConfig
export ConstrainedConfig, RobustConfig
export TrustRegionConfig
export AdamConfig, AdamState
export AlgorithmRecommendation
export AutoDiffConfig
export UQConfig, UncertaintyResult
export SensitivityConfig, SensitivityResult
export MultiObjectiveConfig, MultiObjectiveResult
export Checkpoint

# Optimizer runtime self-checks (used when `check_invariants = true`)
export InvariantViolationError
export check_armijo, check_wolfe_curvature
export check_bfgs_curvature, check_lbfgs_pair_positive
export check_monotone_ascent, check_trust_region_ratio
export check_penalty_weight_growth
export check_simplex_shape, check_cma_covariance, check_cvar_ordering
export check_unitary_invariant, check_pure_state_norm

# Backend types
export CPUBackend, CUDABackend, MetalBackend
export HybridExecutionPlanner
export plan_hybrid_execution, adaptive_backend_selection, estimate_operation_time
export TaskParallelizationStrategy, VectorizationStrategy, GradientParallelization

# Constraint types
export AbstractConstraint, BoundConstraint, PowerConstraint
export BandwidthConstraint, EnergyConstraint, CustomConstraint

# ---------------------------------------------------------------------------
# Public API — System constructors and helpers
# ---------------------------------------------------------------------------
export quantum_system, spin_system, qubit_system
export state_target, unitary_target
export random_controls, zero_controls
export validate_system, validate_controls, validate_target, validate_all

# ---------------------------------------------------------------------------
# Public API — Core computations
# ---------------------------------------------------------------------------
export compute_propagator, compute_fidelity
export compute_grape_gradient, compute_gradient_autodiff
export build_total_hamiltonian
export compute_forward_propagators, compute_backward_propagators
# Backward-compatible legacy aliases (Matrix + dt convenience wrappers)
export grape_gradient, evaluate_fidelity
export finite_difference_gradient, finite_diff_gradient
# Renamed-symbol aliases (kept for older script / test compatibility)
const propagator      = compute_propagator
const gate_target     = unitary_target
const operator_target = unitary_target
export propagator, gate_target, operator_target

# Propagator-backend hierarchy (Theme 1)
export AbstractPropagator
export EigenPropagator, PadePropagator, ChebyshevPropagator
export NewtonPropagator, MagnusPropagator

# Control parameterisation hierarchy (Theme 2)
export AbstractControlParameterization
export PiecewiseConstant, TanhParam, TanhSqParam, LogisticParam
export PhaseOnlyParam
export BSplineParam, HermiteParam, FourierParam
export ChebyshevParam, SlepianParam, CRABRandomParam
export to_waveform, from_waveform, waveform_jacobian
export apply_jacobian_transpose!

# Unified noise / ensemble abstraction (Theme 5)
export AbstractNoiseModel, NoiseSample
export ParametricDrift, PowderOrientation, DriveCalibration
export MarkovianDissipation, ColoredNoiseSpectrum, CompositeNoise
export sample_ensemble, n_samples

# Fidelity library — state, gate, ensemble, gradient pre-factor
export state_fidelity, gate_fidelity, dm_fidelity
export state_overlap, gate_fidelity_unnormalized
export state_transfer_fidelity, state_transfer_fidelity_unnormalized
export ensemble_fidelity, infidelity
export fidelity_grad_prefactor
export STATE_FIDELITY_TYPES, GATE_FIDELITY_TYPES
# Fidelity metric type hierarchy (type-stable dispatch)
export AbstractFidelityMetric
export RealOverlap, SquaredOverlap, ModulusOverlap, UhlmannFidelity, LinearDMFidelity
export NormalizedGate, RealGate, AverageGate
export ModulusGate, SquaredDMFidelity, ModulusDMFidelity
export REAL_OVERLAP, SQUARED_OVERLAP, MODULUS_OVERLAP, UHLMANN_FIDELITY, LINEAR_DM
export NORMALIZED_GATE, REAL_GATE, AVERAGE_GATE
export MODULUS_GATE, SQUARED_DM, MODULUS_DM
# Theme 4 — extended fidelity metrics
export EssentialSubspaceGate, CooperativeTargetFidelity, ProcessTomographyFidelity
export cooperative_fidelity

# Theme 9 — hardware-aware pulse composition
export PulseComposition, compose_hard_pulse_propagator, dead_time_propagator
export compose_effective_boundary

# Theme 12 — pulse analysis utilities
export pulse_spectrum, pulse_bandwidth, pulse_summary, bloch_sweep_fidelity
export parameter_jacobian

# Penalty functions
export penalty_value, penalty_gradient, penalty_value_and_gradient
export PENALTY_TYPES
# Penalty functor hierarchy (type-stable callable structs)
export AbstractPenalty
export NormSquarePenalty, SpilloutPenalty, AmplitudeSpilloutPenalty
export SmoothnessPenalty, EnergyPenalty
# Quandary / Spinach / qopt-inspired waveform-only penalties
export TotalEnergyBudget, MirrorSymmetryPenalty, AsymmetryPenalty
export CrossCouplingPenalty, InterpolatedTikhonov
export gradient, value_and_gradient
export make_penalty_fns, make_penalty_grad_fns

# Sensitivity and uncertainty
export compute_sensitivity, estimate_uncertainty
export verify_gradient_autodiff

# ---------------------------------------------------------------------------
# Public API — Optimization algorithms  (full lists in the generic/QOC sections below)
# ---------------------------------------------------------------------------
export grape_optimize_ensemble   # ensemble wrapper not repeated in QOC section

# Generic ensemble-objective wrapper — any aggregator × any optimizer
export EnsembleObjective
export ensemble_value, ensemble_value_and_grad
export ensemble_wrap, ensemble_wrap_fonly, ensemble_wrap_ascent
export build_ensemble_from_systems, build_ensemble_from_perturbations
export build_ensemble_from_mrcontrol

# Per-sample MR kernels (back the :worst_case / :cvar aggregators)
export grape_state_kernel_single, grape_lindblad_kernel_single

# RobustOpt helpers exposed for users building custom EnsembleObjectives
export sample_parametric_perturbations, sample_drift_trajectories
export robust_fidelity, cvar

# ---------------------------------------------------------------------------
# Public API — MR application layer
# ---------------------------------------------------------------------------
# Physical spin system
export MRSpinSystem
export mr_system, spin_op, spin_state
export hamiltonian                    # builds drift H from physical parameters
export GYRO_MHZ_PER_T, SPIN_QUANTUM_NUMBER

# Abstract supertype (allows dispatch on either MRControl or LindbladMRControl)
export AbstractMRControl

# Optimal control problem — closed system
export MRControl
export optimcon                       # high-level GRAPE driver for MRControl

# Optimal control problem — open system (Lindblad / Liouville space)
export LindbladMRControl
export grape_lindblad_kernel          # Liouville-space GRAPE kernel (advanced use)
# Physical relaxation helpers
export mr_relaxation, density_matrix
# Liouvillian builders and density-matrix utilities
export build_drift_liouvillian, build_control_liouvillian
export vec_rho, mat_rho, pure_state_to_vec_rho
export lindblad_grad_prefactor

# GRAPE state-transfer kernel (exposed for advanced use)
export grape_state_kernel
# Phase-only GRAPE kernel (polar parameterisation, fixed amplitude)
export grape_phase_kernel
# Tracking GRAPE (trajectory checkpoints)
export TrackingPoint
export grape_tracking_kernel
# Forward-only fidelity for derivative-free / metaheuristic optimizers
export fidelity_forward
export constrained_optimize, robust_optimize
export trust_region_optimize
export multi_objective_optimize
export auto_optimize

# ---------------------------------------------------------------------------
# Public API — Advanced features
# ---------------------------------------------------------------------------
export AbstractCheckpoint
export save_checkpoint, load_checkpoint
export resume_optimization, create_checkpoint
export auto_checkpoint_callback
export checkpoint_compatible
export list_checkpoints, checkpoint_summary

# Multi-objective objectives
export energy_objective, smoothness_objective
export peak_amplitude_objective, fidelity_objective

# ---------------------------------------------------------------------------
# Public API — Algorithm selection
# ---------------------------------------------------------------------------
export recommend_optimizer, describe_recommendation

# Algorithm-routing registry (Theme 8)
export OptimizerEntry, OptimizerSupports
export OPTIMIZER_REGISTRY
export register_optimizer!, get_optimizer, list_optimizers, is_registered

# ---------------------------------------------------------------------------
# Public API — Utilities
# ---------------------------------------------------------------------------
# Iteration progress callback
export IterationCallback, iteration_callback, reset!

export PerformanceMonitor, record_iteration!, get_summary, print_progress
export detect_stagnation, get_memory_usage_mb
export plot_convergence, plot_controls, plot_bloch_trajectory
export plot_sensitivity_heatmap, plot_pareto_front
export create_optimization_report

# Pulse export (spectrometer file formats)
export save_bruker_shape
export load_bruker_shape

# New unified export subsystem (Output module)
export OptimizedPulse
export export_pulse
export register_exporter, replace_exporter, list_exporters
# Return-type structs from Output module
export BrukerShape, JEOLShape, EPRShape
export PulseqSequence
export QiskitWaveformExport, QuilTExport, QUAExport, PulserExport
# Pulse format loaders (inverse of exporters)
export load_jeol_shape, load_epr_shape, load_pulseq
export load_qiskit_waveform, load_quil_t, load_qua, load_pulser

# Direct / local derivative-free optimizers (generic function interface)
export nelder_mead_optimize           # generic f(θ) variant (dispatch-safe)
export hooke_jeeves_optimize, compass_search_optimize, powell_dirset_optimize
export uobyqa_optimize, newuoa_optimize, bobyqa_optimize
export cobyla_optimize, lincoa_optimize

# Metaheuristic / global optimizers (generic function interface)
export ga_optimize
export sa_optimize, mcsa_optimize, ssmc_optimize
export cmaes_optimize, pso_optimize, de_optimize
export pscmaes_optimize
export mc_random_search, grid_search
export basin_hopping_optimize

# Gradient-based optimizers — generic (function interface)
# First-order
export gd_optimize, sgd_optimize, momentum_optimize, nag_optimize
export adagrad_optimize, rmsprop_optimize, adam_optimize
# Conjugate gradient
export cg_optimize
# Quasi-Newton
export bfgs_optimize, lbfgs_optimize, lbfgsb_optimize
# Second-order
export newton_optimize, gauss_newton_optimize, lm_optimize
export trust_region_newton_optimize, projected_gradient_optimize

# Gradient-based optimizers — QOC (generic Function dispatch, backward-compatible)
export grape_optimize, grape_cg_optimize, grape_lbfgsb_optimize
# SU(2) Cayley-Klein phase-only GRAPE (spin-1/2, constant Rabi, ~50-100x faster than matrix-exp)
export precompute_su2_params, precompute_su2_params_ensemble, validate_su2_params
export su2_fidelity_gradient, su2_fidelity_forward, su2_forward_pass!
export SU2Kernel, su2_fidelity_and_grad
export grape_su2_optimize, grape_su2_multistart
export su2_duration_sweep

# Spin-I (arbitrary single spin-I, phase-only)
export precompute_spinI_params, precompute_spinI_params_ensemble
export spinI_fidelity_gradient, spinI_fidelity_forward, spinI_forward_pass!
export SpinIKernel, spinI_fidelity_and_grad
export grape_spinI_optimize, grape_spinI_multistart

# Trotter (N coupled spin-1/2, Strang-split, phase-only)
export precompute_trotter_params, trotter_fidelity_gradient, trotter_fidelity_forward
export TrotterKernel, trotter_fidelity_and_grad
export grape_trotter_optimize, grape_trotter_multistart
# Real Krotov (monotonic co-state method; system/target/controls signature)
export krotov_optimize, krotov_second_order_optimize
export group_optimize, goat_optimize
export oc_trust_region_newton_optimize, oc_semismooth_newton_optimize
export tgrape_optimize
export crab_optimize

# Analytic / semi-analytic pulse design
export CompositePulseSegment, AnalyticPulse
export bb1, scrofulous, sk1, corpse, short_corpse, f1, g1, corpse_in_bb1
export small_tip_angle_fourier_1d
export sta_fourier_1d  # deprecated alias
export slr_1d
export verse, verse_min_time, verse_acoustic_noise

# Problem library
export hadamard_gate_problem, not_gate_problem, cnot_gate_problem
export state_transfer_0_to_1, state_transfer_0_to_plus
export inept_problem, spin_echo_problem
export robust_hadamard_problem, random_unitary_problem
export spin_half_operators, spin_operators, tensor_product_operators

# Backend availability checks
export is_cuda_available, is_metal_available

# ---------------------------------------------------------------------------
# New system types (Task 17 exports)
# ---------------------------------------------------------------------------
export HeteronuclearSystem, EPRSpinSystem, MASSpinSystem
export BlochSystem, BlochIsochromat, GradientSystem, MRIControlSequence
export DNPSpinSystem
export CSATensor, DipolarCoupling

# New constructors
export heteronuclear_system, epr_system, mas_spin_system, bloch_system
export mri_control_sequence, dnp_system

# New physics
export BandWeight
export band_selective_fidelity, band_selective_gradient
export shift_system
export bloch_fidelity, slice_profile_fidelity, bloch_forward_pass, bloch_adjoint_pass
export dnp_polarization_fidelity
export electron_polarized_state, nuclear_polarization_operator

# MAS
export build_mas_hamiltonian, rotate_spin_system, compute_grape_gradient_powder
export compute_propagators

# Wigner
export wigner_d2, wigner_D2, powder_grid

# MRI penalties
export sar_penalty, sar_gradient, slew_rate_penalty, slew_rate_gradient

# DNP optimal control
export optimcon_dnp
export grape_dnp_lindblad_kernel

# Device registry — global compute device for ensemble averaging
export set_device!, get_device, available_devices, with_device

# ---------------------------------------------------------------------------
# Public API — Quantum Computing application layer
# ---------------------------------------------------------------------------

# ── Shared lower-layer additions ─────────────────────────────────────────────
# Physics/Penalties.jl additions
export leakage_penalty, leakage_gradient
export ms_closure_penalty, ms_phase_penalty
export ms_closure_gradient, ms_phase_gradient
export filter_function_penalty, filter_function_gradient

# Physics/Lindblad.jl additions
export lindblad_system_from_jump_ops

# Optimization/Analytic/Composite.jl additions
export drag_pulse

# ── TransmonSystem ───────────────────────────────────────────────────────────
export TransmonSystem
export transmon_system

# ── Unified optimcon context ─────────────────────────────────────────────────
export AbstractOptimizationContext
export QCControl
export LindbladQCControl

# ── Hardware platform types and constructors ─────────────────────────────────
export TrappedIonSystem, trapped_ion_system
export NeutralAtomSystem, neutral_atom_system
export SpinQubitSystem, spin_qubit_system
export NVCenterSystem, nv_center_system

# ── Gate libraries ────────────────────────────────────────────────────────────
# Single-qubit gates
export X_gate, Y_gate, Z_gate, H_gate, I_gate
export S_gate, Sdg_gate, T_gate, Tdg_gate, SX_gate
export Rx, Ry, Rz, Rn, U3
export single_qubit_gate_set

# Two-qubit gates
export CNOT_gate, CX_gate, CZ_gate, CY_gate
export SWAP_gate, iSWAP_gate, SQISWAP_gate
export MS_gate, CRx, CRy, CRz, ZZθ_gate
export two_qubit_gate_set

# Native gate set
export NativeGateSet, native_gate_set
export zyz_decompose, zyz_sequence, gate_infidelity

# ── Noise models ──────────────────────────────────────────────────────────────
# Quasi-static
export QuasiStaticNoise
export quasi_static_ensemble, robust_optimcon_qs, evaluate_qs_robustness

# Markovian (Lindblad)
export MarkovianNoise, markovian_noise
export amplitude_damping, phase_damping, depolarizing_channel
export lindblad_optimcon

# Non-Markovian (filter function)
export NoiseSpectrum
export pink_noise_spectrum, white_noise_spectrum, ohmic_noise_spectrum
export compute_filter_function, filter_function_infidelity
export optimcon_ff

# ── Verification ──────────────────────────────────────────────────────────────
# Randomized benchmarking
export RBResult
export rb_sequence, rb_survival_probability
export fit_rb_decay, estimate_epc, interleaved_rb

# Process tomography
export QPTResult
export qpt_input_states, qpt_choi_matrix, qpt_reconstruct_linear
export process_fidelity, process_fidelity_to_avg, average_gate_fidelity
export print_qpt_summary

# Device-dispatched ensemble map and gradient averaging
export ensemble_map, ensemble_grad!, ensemble_fobj
# Strategy-based ensemble execution (replaces raw device symbols)
export AbstractEnsembleStrategy, ThreadedEnsemble, SequentialEnsemble, DistributedEnsemble
export default_ensemble_strategy
export ensemble_map_strategy, ensemble_gradient_accumulate!
# Pre-allocated scratch buffer pool (eliminates GC in gradient loops)
export ScratchBufferPool, acquire!, release!

"""
    is_metal_available() -> Bool

Return `true` if Metal.jl was successfully imported during module
initialisation (Apple Silicon macOS with Metal.jl present).
"""
is_metal_available()::Bool = _METAL_LOADED[]

"""
    is_cuda_available() -> Bool

Return `true` if CUDA.jl was successfully imported during module
initialisation (any platform with CUDA.jl and a functional NVIDIA driver).
"""
is_cuda_available()::Bool = _CUDA_LOADED[]

# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

"""
    _can_use_metal() -> Bool

Platform pre-check for Metal: only valid on aarch64 macOS (Apple Silicon).
"""
_can_use_metal()::Bool = Sys.isapple() && Sys.ARCH === :aarch64

"""
    _detect_and_load_backends()

Called once from `__init__()`. Passively checks whether GPU packages have
**already been loaded** by the caller before `using Pulsar`.  Pulsar never
triggers a new `import` here — doing so can stall for minutes or hours if the
package needs precompilation.

Usage pattern for GPU acceleration
────────────────────────────────────
Load the GPU package *before* Pulsar in your script or REPL session:

    import Metal          # or: import CUDA
    using Pulsar          # __init__ will see Metal/CUDA already loaded

Decision logic
──────────────
• Metal already in Base.loaded_modules (Apple Silicon)  → set _METAL_LOADED[]
• CUDA  already in Base.loaded_modules                  → set _CUDA_LOADED[]
• Neither present yet                                   → CPU-only mode

No imports are triggered; no precompilation is started; loading is instant.
"""
function _detect_and_load_backends()

    # ── Metal ─────────────────────────────────────────────────────
    # Only flag if Metal was already imported by the user/script.
    # Never call `import Metal` here — it can trigger precompilation
    # and hang inside __init__ for an arbitrarily long time.
    if _can_use_metal()
        metal_id = Base.identify_package("Metal")
        if !isnothing(metal_id) && haskey(Base.loaded_modules, metal_id)
            _METAL_LOADED[] = true
        end
    end

    # ── CUDA ──────────────────────────────────────────────────────
    # Same passive-only check.
    cuda_id = Base.identify_package("CUDA")
    if !isnothing(cuda_id) && haskey(Base.loaded_modules, cuda_id)
        _CUDA_LOADED[] = true
    end

    # ── Log outcome ───────────────────────────────────────────────
    if _METAL_LOADED[] && _CUDA_LOADED[]
        @debug "Pulsar: Metal + CUDA backends activated"
    elseif _METAL_LOADED[]
        @debug "Pulsar: Metal backend activated (Apple Silicon)"
    elseif _CUDA_LOADED[]
        @debug "Pulsar: CUDA backend activated"
    else
        @debug "Pulsar: no GPU backend found — CPU-only mode"
    end
end

# ---------------------------------------------------------------------------
# Module initializer
# ---------------------------------------------------------------------------

"""
    __init__()

Module initialisation hook — runs every time Pulsar is loaded into a
Julia session (not during precompilation).

1. Configure BLAS threads to match Julia thread count.
2. Detect and import available GPU backends (Metal / CUDA / both).
3. Print startup banner in interactive sessions.
"""
function __init__()
    # 1. Tune BLAS thread count
    try
        LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
    catch
        # Non-critical; some BLAS builds don't expose this
    end

    # 2. GPU detection — populates _METAL_LOADED and _CUDA_LOADED
    _detect_and_load_backends()

    # 3. Interactive banner
    if isinteractive()
        _print_banner()
    end
end

# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------

"""
    _print_banner()

Startup banner shown in interactive Julia sessions.
"""
function _print_banner()
    ver = pkgversion(@__MODULE__)
    n   = Threads.nthreads()
    gpu = _detect_gpu()
    @printf(
        "\n  Pulsar.jl v%s  |  Julia %s  |  %d thread%s  |  GPU: %s\n\n",
        ver, VERSION, n, n == 1 ? "" : "s", gpu,
    )
end

"""
    _detect_gpu() -> String

Human-readable summary of active GPU backend(s), using the flags set by
`_detect_and_load_backends()` during `__init__()`.
"""
function _detect_gpu()::String
    if _METAL_LOADED[] && _CUDA_LOADED[]
        return "Metal (Apple Silicon) + CUDA (NVIDIA)"
    elseif _METAL_LOADED[]
        return "Metal (Apple Silicon)"
    elseif _CUDA_LOADED[]
        return "CUDA (NVIDIA)"
    else
        return "none"
    end
end

end # module Pulsar
