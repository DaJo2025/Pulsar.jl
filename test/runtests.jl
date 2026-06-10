# Pulsar.jl Test Suite
# ====================
# Main test runner. Executes every test group in logical order:
#   1. Physics/mathematics foundations
#   2. Algorithm-level unit tests
#   3. Advanced feature tests
#   4. Full end-to-end integration tests
#
# Run from the repository root with:
#   julia --project=. test/runtests.jl
# or via Pkg:
#   julia> ] test Pulsar

using Test
using Pulsar
using LinearAlgebra

# ---------------------------------------------------------------------------
# Helper: print a section banner so output is easy to scan in CI logs
# ---------------------------------------------------------------------------
function section(title::String)
    bar = "=" ^ 70
    println("\n", bar)
    println("  ", title)
    println(bar)
end

# ---------------------------------------------------------------------------
# Top-level test set
# ---------------------------------------------------------------------------
@testset "Pulsar.jl Tests" begin

    section("Architecture – Layer Dependencies")
    include("Architecture/LayerDependencyTests.jl")

    section("Physics – Mathematical Correctness")
    include("Physics/MathematicalCorrectness.jl")

    section("Physics – Physics Validation")
    include("Physics/PhysicsValidation.jl")

    section("Physics – Penalty Functor Additions (Theme 3)")
    include("Physics/PenaltyTests.jl")

    section("Physics – Fidelity Metric Extensions (Theme 4)")
    include("Physics/FidelityMetricsTheme4Tests.jl")

    section("Physics – Fidelity Metric Coverage (all algorithms/applications)")
    include("Physics/FidelityMetricCoverageTests.jl")

    section("Physics – Hardware-Aware Pulse Composition (Theme 9)")
    include("Physics/PulseCompositionTests.jl")

    section("Computation – Propagator Registry (Theme 1)")
    include("Computation/PropagatorRegistryTests.jl")

    section("Optimization – Control Parameterisation (Theme 2)")
    include("Optimization/ParameterizationTests.jl")

    section("Runtime – Algorithm Registry (Theme 8)")
    include("Runtime/AlgorithmRegistryTests.jl")

    section("Physics – Unified Noise Models (Theme 5)")
    include("Physics/NoiseModelsTests.jl")

    section("IO – Unified Checkpoint Hierarchy (Theme 13)")
    include("IO/CheckpointHierarchyTests.jl")

    section("Application – LindbladQCControl Skeleton (Theme 6)")
    include("Application/QuantumComputing/LindbladQCControlTests.jl")

    section("Application – LindbladQCControl Integration (Theme 6b)")
    include("Application/QuantumComputing/LindbladQCControlIntegrationTests.jl")

    section("Application – LindbladMRControl Algorithm Sweep (Theme 6b)")
    include("Application/MR/LindbladMRAlgorithmSweepTests.jl")

    section("Algorithms – GRAPE")
    include("Algorithms/GRAPETests.jl")

    section("Algorithms – Second-Order Methods")
    include("Algorithms/SecondOrderTests.jl")

    section("Algorithms – Constrained Optimization")
    include("Algorithms/ConstrainedTests.jl")

    section("Algorithms – Direct Search")
    include("Algorithms/DirectSearchTests.jl")

    section("Algorithms – Runtime Invariants")
    include("Algorithms/AlgorithmInvariants.jl")

    section("Algorithms – tGRAPE (time-optimal)")
    include("Algorithms/TGRAPETests.jl")

    section("Algorithms – PhaseOnlyParam Integration")
    include("Algorithms/PhaseOnlyTests.jl")

    section("Algorithms – Real Krotov")
    include("Algorithms/KrotovTests.jl")

    section("Algorithms – Krotov Theme 7 Upgrades")
    include("Algorithms/KrotovTheme7Tests.jl")

    section("Utilities – Pulse Analysis (Theme 12)")
    include("Utilities/PulseAnalysisTests.jl")

    section("Algorithms – Ensemble Objective")
    include("Algorithms/EnsembleTests.jl")

    section("Advanced Features – Automatic Differentiation")
    include("AdvancedFeatures/AutoDiffTests.jl")

    section("Integration – End-to-End Workflows")
    include("Integration/EndToEndTests.jl")

    section("IO – Pulse Format Round-Trip")
    include("IO/PulseExportTests.jl")

    section("Parallelization – CPU Gradient Consistency")
    include("Parallelization/CPUParallelizationTests.jl")

    section("Parallelization – Backend Fallback")
    include("Parallelization/BackendFallbackTests.jl")

    section("Backend – Batched GPU Primitives")
    include("Backend/GPUBatchedTests.jl")

    section("Runtime – GPU Auto-Setup")
    include("Runtime/GPUSetupTests.jl")

end
