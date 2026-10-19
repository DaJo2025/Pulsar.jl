"""
    comparisons/Translator/Subprocess.jl

Shared subprocess and waveform-file helpers for every driver.  Each driver
uses these in place of per-file `readlines / strip / filter` boilerplate.
"""

using Printf

"""
    run_subprocess(bin, args, workdir; stdout_file, stderr_file) -> exit_code

Run `bin` with `args` in directory `workdir`, redirecting stdout/stderr to the
given files.  Returns the exit code (0 on success).  Raises on non-zero
exit after including the last 40 lines of stderr in the error message.
"""
function run_subprocess(bin::String, args::Vector{String}, workdir::String;
                         stdout_file::String, stderr_file::String,
                         env = nothing)::Int
    cmd = if env === nothing
        Cmd(Cmd([bin, args...]); dir=workdir)
    else
        Cmd(Cmd([bin, args...]); dir=workdir, env=env)
    end
    try
        run(pipeline(cmd, stdout=stdout_file, stderr=stderr_file))
    catch
        err_tail = log_tail(stderr_file)
        out_tail = log_tail(stdout_file)
        error("Subprocess $(bin) failed.\nstderr tail:\n$err_tail\n\nstdout tail:\n$out_tail")
    end
    return 0
end

"""
    log_tail(path, n=40) -> String

Return the last `n` lines of a log file, or an empty-log notice.
"""
function log_tail(path::String, n::Int = 40)::String
    isfile(path) || return "(log file $path does not exist)"
    lines = readlines(path)
    isempty(lines) && return "(log empty)"
    return join(lines[max(1, end - n + 1):end], "\n")
end

"""
    parse_waveform_file(path, n_ctrl, n_t;
                         convention::Symbol = :rad_per_sec,
                         pwr_levels         = nothing,
                         delimiter          = r"\\s+")
    -> Matrix{Float64}

Parse a plain-text waveform written by a target program.  The file must have
at least `n_t` non-blank lines, each with `n_ctrl` whitespace-separated
numeric fields.

# Conventions
- `:rad_per_sec` — values are in rad/s; divided by `pwr_levels[k]` to
  normalise into `[l_bound, u_bound]`.
- `:hz`          — values are in Hz; multiplied by 2π then treated like
  `:rad_per_sec`.
- `:normalised`  — values are already dimensionless amplitudes in
  `[l_bound, u_bound]`; no rescaling.

Returns `[n_ctrl × n_t]` matrix matching `MRControl` / `LindbladMRControl`
conventions.
"""
function parse_waveform_file(path::String, n_ctrl::Int, n_t::Int;
                              convention::Symbol           = :rad_per_sec,
                              pwr_levels                   = nothing,
                              delimiter                    = r"\s+")::Matrix{Float64}
    convention ∈ (:rad_per_sec, :hz, :normalised) ||
        throw(ArgumentError("parse_waveform_file: unknown convention :$convention"))

    isfile(path) || error("Waveform file missing: $path")
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= n_t ||
        error("Waveform file has $(length(lines)) rows, expected ≥ $n_t: $path")

    pwr_per_ctrl = _resolve_pwr_per_ctrl(pwr_levels, n_ctrl)

    wf = Matrix{Float64}(undef, n_ctrl, n_t)
    for i in 1:n_t
        parts = filter(!isempty, split(lines[i], delimiter))
        length(parts) >= n_ctrl ||
            error("Waveform row $i has $(length(parts)) fields, expected $n_ctrl: $(lines[i])")
        for k in 1:n_ctrl
            v = parse(Float64, parts[k])
            if convention === :normalised
                wf[k, i] = v
            elseif convention === :rad_per_sec
                wf[k, i] = v / pwr_per_ctrl[k]
            elseif convention === :hz
                wf[k, i] = (2π * v) / pwr_per_ctrl[k]
            end
        end
    end
    return wf
end

function _resolve_pwr_per_ctrl(pwr_levels, n_ctrl::Int)::Vector{Float64}
    pwr_levels === nothing && return ones(Float64, n_ctrl)
    pwr = collect(Float64, pwr_levels)
    length(pwr) == 1 && return fill(pwr[1], n_ctrl)
    length(pwr) == n_ctrl && return pwr
    throw(ArgumentError(
        "pwr_levels length $(length(pwr)) ≠ n_ctrl $n_ctrl and ≠ 1"))
end

"""
    canonical_rescore(waveform, ctrl) -> Float64

Score an external-solver waveform using Pulsar's kernels with the
**phase-insensitive** `|⟨ψ_t|ψ_f⟩|²` metric (`:state_squared`, Φ^S_Q).  External packages
optimise in density-matrix space and do not preserve ket global phase, so
scoring their output with `MRControl`'s default `:state_real` metric gives garbage
(e.g. a perfect Rx(π) scores 0 because Rx(π)|+z⟩ = −i|−z⟩).

Pulsar's own driver keeps `grape_state_kernel` (which honours
`ctrl.fidelity = :state_real`) — it optimised that metric and is scored on it.
"""
function canonical_rescore(waveform::Matrix{Float64}, ctrl::MRControl)::Float64
    return fidelity_forward(waveform, ctrl.drifts, ctrl.pwr_levels,
                             ctrl.operators, ctrl.rho_init, ctrl.rho_targ,
                             ctrl.pulse_dt; fidelity_type = :state_squared)
end

canonical_rescore(waveform::Matrix{Float64}, ctrl::LindbladMRControl)::Float64 =
    first(grape_lindblad_kernel(waveform, ctrl))

"""
    write_waveform_file(path, w_normalised, pwr_levels;
                          convention::Symbol = :rad_per_sec) -> Nothing

Write a waveform to disk for consumption by a target program as its initial
guess.  Inverse of `parse_waveform_file`.
"""
function write_waveform_file(path::String, w_normalised::Matrix{Float64},
                              pwr_levels;
                              convention::Symbol = :rad_per_sec)::Nothing
    n_ctrl, n_t = size(w_normalised)
    pwr_per_ctrl = _resolve_pwr_per_ctrl(pwr_levels, n_ctrl)
    open(path, "w") do io
        for i in 1:n_t
            for k in 1:n_ctrl
                v = w_normalised[k, i]
                out = if convention === :normalised
                    v
                elseif convention === :rad_per_sec
                    v * pwr_per_ctrl[k]
                elseif convention === :hz
                    v * pwr_per_ctrl[k] / (2π)
                else
                    throw(ArgumentError("write_waveform_file: unknown convention :$convention"))
                end
                k > 1 && print(io, "  ")
                @printf(io, "%24.16e", out)
            end
            println(io)
        end
    end
    return nothing
end
