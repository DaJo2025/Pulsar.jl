@testset "Architecture – Layer Dependencies" begin

    src_root = joinpath(@__DIR__, "..", "..", "src")

    # Layer number for each source directory prefix (higher = depends on lower)
    LAYER = Dict(
        "Types"        => 1,
        "Computation"  => 2,
        "Backend"      => 2,
        "Physics"      => 3,
        "Optimization" => 4,
        "IO"           => 5,
        "Runtime"      => 5,
        "Utilities"    => 5,
        "Application"  => 6,
    )

    # Directories that a given layer may NOT reference (upward deps)
    # Key = layer number, value = set of forbidden directory prefixes
    FORBIDDEN = Dict(
        1 => ["Physics", "Optimization", "IO", "Runtime", "Utilities", "Application"],
        2 => ["Physics", "Optimization", "IO", "Runtime", "Utilities", "Application"],
        3 => ["Optimization", "IO", "Runtime", "Utilities", "Application"],
        4 => ["IO", "Runtime", "Utilities", "Application"],
        5 => ["Application"],
    )

    # Assign a layer to a file path relative to src/
    function file_layer(relpath)
        dir = split(relpath, "/")[1]
        get(LAYER, dir, 0)
    end

    # Collect all .jl source files under src/ (relative paths)
    all_src = String[]
    for (root, dirs, files) in walkdir(src_root)
        for f in files
            endswith(f, ".jl") || continue
            abs = joinpath(root, f)
            rel = relpath(abs, src_root)
            push!(all_src, replace(rel, "\\" => "/"))
        end
    end

    # --------------------------------------------------------------------------
    # Test 1: Pulsar.jl include order respects layer ordering
    # --------------------------------------------------------------------------
    @testset "Include order in Pulsar.jl" begin
        pulsar_jl = read(joinpath(src_root, "Pulsar.jl"), String)

        # Extract all include("...") paths
        includes = [m.match for m in eachmatch(r"""include\("([^"]+)"\)""", pulsar_jl)]
        paths    = [m.captures[1] for m in eachmatch(r"""include\("([^"]+)"\)""", pulsar_jl)]

        # Files that are intentionally re-ordered out of strict layer sequence;
        # see CLAUDE.md §1 for the architectural rationale.  Each entry depends
        # on a higher-layer file and is therefore included after that layer.
        ALLOWED_REORDERINGS = Set([
            "Physics/UncertaintyQuantification.jl",   # depends on Optimization/GRAPE
            "Physics/Sensitivity.jl",                  # depends on Optimization/GRAPE
            "Physics/PulseComposition.jl",             # depends on Optimization/GRAPE
            "Physics/MRPhysics.jl",                    # late-included (post-Application)
            "Backend/Scheduling/DeviceRegistry.jl",    # post-Application registration
            "Computation/MASPropagators.jl",           # late-included
            "Computation/WignerRotations.jl",          # late-included
            "Computation/BlochPropagator.jl",          # late-included
            "Types/NMRSpinSystem.jl",
            "Types/EPRSpinSystem.jl",
            "Types/MASSpinSystem.jl",
            "Types/BlochSystem.jl",
            "Types/DNPSpinSystem.jl",
            "Types/TransmonSystem.jl",
            "Types/TrappedIonSystem.jl",
            "Types/NeutralAtomSystem.jl",
            "Types/SpinQubitSystem.jl",
            "Types/NVCenterSystem.jl",
        ])

        max_layer_seen = 0
        violations     = String[]

        for path in paths
            layer = file_layer(path)
            layer == 0 && continue  # skip Pulsar.jl itself or unknown
            if layer < max_layer_seen - 1 && !(path in ALLOWED_REORDERINGS)
                push!(violations, "$(path) (layer $(layer)) included after layer $(max_layer_seen) files")
            end
            max_layer_seen = max(max_layer_seen, layer)
        end

        @test isempty(violations)

        if !isempty(violations)
            for v in violations
                @warn "Layer ordering violation: $v"
            end
        end
    end

    # --------------------------------------------------------------------------
    # Test 2: Source files in lower layers do not include() higher-layer files
    # --------------------------------------------------------------------------
    @testset "No upward include() in source files" begin
        violations = Tuple{String,String}[]

        for relpath in all_src
            layer = file_layer(relpath)
            layer == 0 && continue
            forbidden_dirs = get(FORBIDDEN, layer, String[])
            isempty(forbidden_dirs) && continue

            content = read(joinpath(src_root, relpath), String)

            for m in eachmatch(r"""include\("([^"]+)"\)""", content)
                inc_path = m.captures[1]
                for fd in forbidden_dirs
                    if startswith(inc_path, fd * "/") || contains(inc_path, "/" * fd * "/")
                        push!(violations, (relpath, inc_path))
                    end
                end
            end
        end

        @test isempty(violations)
        for (src, inc) in violations
            @warn "Upward include: $src includes $inc"
        end
    end

    # --------------------------------------------------------------------------
    # Test 3: Types/ files do not import using .Physics / .Optimization etc.
    # --------------------------------------------------------------------------
    @testset "Types layer has no upward using/import" begin
        forbidden_modules = ["Physics", "Optimization", "IO", "Runtime", "Utilities", "Application"]
        pattern = Regex("using\\s+\\.(" * join(forbidden_modules, "|") * ")|import\\s+\\.(" * join(forbidden_modules, "|") * ")")

        violations = Tuple{String,String}[]
        for relpath in all_src
            startswith(relpath, "Types/") || continue
            content = read(joinpath(src_root, relpath), String)
            for m in eachmatch(pattern, content)
                push!(violations, (relpath, m.match))
            end
        end

        @test isempty(violations)
        for (src, stmt) in violations
            @warn "Upward module import in Types/: $src — $stmt"
        end
    end

end
