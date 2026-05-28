using Documenter
using Pulsar

DocMeta.setdocmeta!(Pulsar, :DocTestSetup, :(using Pulsar); recursive=true)

const Pulsar_ROOT = normpath(joinpath(@__DIR__, ".."))

makedocs(;
    modules  = [Pulsar],
    authors  = "Pulsar.jl Contributors",
    sitename = "Pulsar.jl",
    remotes  = Dict(
        Pulsar_ROOT => (Documenter.Remotes.GitHub("DaJo2025", "Pulsar.jl"), "main"),
    ),
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://DaJo2025.github.io/Pulsar.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quickstart" => "quickstart.md",
        "Theory" => [
            "Hamiltonians"           => "theory/hamiltonians.md",
            "Propagators"            => "theory/propagators.md",
            "Fidelity metrics"       => "theory/fidelity.md",
            "Penalties"              => "theory/penalties.md",
        ],
        "Algorithms" => [
            "GRAPE"                  => "algorithms/grape.md",
            "Second-order methods"   => "algorithms/second_order.md",
            "Direct search"          => "algorithms/direct_search.md",
            "Metaheuristic"          => "algorithms/metaheuristic.md",
            "QOC-specific"           => "algorithms/qoc_specific.md",
            "Analytic pulses"        => "algorithms/analytic.md",
            "Constrained"            => "algorithms/constrained.md",
            "Robust"                 => "algorithms/robust.md",
            "Multi-objective"        => "algorithms/multi_objective.md",
        ],
        "Backends" => [
            "CPU / GPU"              => "backends/cpu_gpu.md",
            "Parallelism"            => "backends/parallelism.md",
        ],
        "Domains" => [
            "NMR"                    => "domains/nmr.md",
            "EPR"                    => "domains/epr.md",
            "MAS solid-state"        => "domains/mas.md",
            "MRI"                    => "domains/mri.md",
            "DNP"                    => "domains/dnp.md",
            "QC platforms"           => "domains/qc_platforms.md",
        ],
        "Advanced" => [
            "Automatic differentiation" => "advanced/autodiff.md",
            "UQ & sensitivity"          => "advanced/uq_sensitivity.md",
            "Pulse export"              => "advanced/io_export.md",
            "Checkpointing"             => "advanced/checkpointing.md",
        ],
        "Comparisons" => [
            "Overview"                  => "comparisons/overview.md",
            "Drivers"                   => "comparisons/drivers.md",
            "Defining your own problem" => "comparisons/your_problem.md",
        ],
        "API reference" => [
            "Types"            => "api/types.md",
            "Computation"      => "api/computation.md",
            "Physics"          => "api/physics.md",
            "Optimization"     => "api/optimization.md",
            "IO"               => "api/io.md",
            "Runtime"          => "api/runtime.md",
            "Utilities"        => "api/utilities.md",
            "Application"      => "api/application.md",
        ],
        "Release notes" => [
            "v0.2.1"           => "release_notes/v0.2.1.md",
            "v0.2.0"           => "release_notes/v0.2.0.md",
        ],
    ],
    warnonly = true,
)

deploydocs(;
    repo         = "github.com/DaJo2025/Pulsar.jl",
    devbranch    = "main",
    push_preview = true,
)
