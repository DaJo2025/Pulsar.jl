"""
    Pulsar.Output

Export subsystem for optimised pulse waveforms.

Supports three application domains and eight vendor/format targets:

  :NMR_EPR   :bruker      → Bruker TopSpin JCAMP-DX shape file (.shape)
             :jeol        → JEOL Delta RF shape file (.jrf)
             :epr_bruker  → Bruker Xepr/Xenon AWG IQ table (.shp)

  :MRI       :pulseq      → Pulseq open MRI sequence file (.seq)

  :QC        :qiskit      → Qiskit Pulse waveform JSON (.json)
             :quil_t      → Quil-T DEFWAVEFORM snippet (.quil)
             :qua         → QUA I/Q play snippet (.py)
             :pulser      → Pasqal Pulser Waveform example script (.py)

All file I/O lives exclusively in this module.
The public entry point for users is `Pulsar.export_pulse` in Utilities/PulseExport.jl,
which is a zero-I/O dispatcher that calls `Pulsar.Output.export_pulse`.

## Canonical pulse type

    OptimizedPulse(samples, dt; meta...)

`samples` is a `Vector{ComplexF64}` where `samples[n] = I[n] + im*Q[n]`.
`dt` is the uniform time step in seconds.
`meta` is an optional `Dict{Symbol,Any}` for metadata (:flip_angle, :carrier_hz, etc.).

## Usage

    pulse = OptimizedPulse(complex_envelope, dt; flip_angle=90.0, carrier_hz=500e6)

    # Save Bruker shape file
    result = export_pulse(pulse; application=:NMR_EPR, vendor=:bruker,
                          save=true, output_dir="output", name="UR90")

    # In-memory Pulseq (no file I/O)
    seq = export_pulse(pulse; application=:MRI, vendor=:pulseq, save=false)

    # Quil-T snippet, saved to disk
    quil = export_pulse(pulse; application=:QC, vendor=:quil_t,
                        save=true, output_dir="output", name="xy_gate")
"""
module Output

using LinearAlgebra
using Printf
using Dates

export OptimizedPulse
export export_pulse
export register_exporter, replace_exporter, list_exporters
export BrukerShape, JEOLShape, EPRShape
export PulseqSequence
export QiskitWaveformExport, QuilTExport, QUAExport, PulserExport
# Loaders
export load_jeol_shape, load_epr_shape, load_pulseq
export load_qiskit_waveform, load_quil_t, load_qua, load_pulser

# ============================================================================
# Canonical pulse type
# ============================================================================

"""
    OptimizedPulse(samples, dt; meta...)

Canonical representation of an optimised RF pulse for export.

# Fields
- `samples::Vector{ComplexF64}` — complex envelope; `samples[n] = I[n] + im*Q[n]`
- `dt::Float64` — uniform time step in seconds
- `meta::Dict{Symbol,Any}` — optional metadata; common keys:
    - `:flip_angle`  — total rotation angle in degrees (default 180.0)
    - `:carrier_hz`  — carrier frequency in Hz (default 0.0)
    - `:bandwidth_hz`— pulse bandwidth in Hz (default 0.0)
    - `:name`        — pulse name string (default "pulsar_pulse")

# Construction from a [2 × N_TS] Cartesian waveform (rad/s)

    OptimizedPulse(w_mat::Matrix{Float64}, pwr_level::Float64, dt::Float64)

Converts `w_mat[1,:] = wx`, `w_mat[2,:] = wy` to complex envelope
`(wx .+ im .* wy) ./ pwr_level`.
"""
struct OptimizedPulse
    samples :: Vector{ComplexF64}
    dt      :: Float64
    meta    :: Dict{Symbol,Any}

    function OptimizedPulse(samples::AbstractVector, dt::Real; kwargs...)
        meta = Dict{Symbol,Any}(
            :flip_angle   => 180.0,
            :carrier_hz   => 0.0,
            :bandwidth_hz => 0.0,
            :name         => "pulsar_pulse",
        )
        for (k, v) in kwargs; meta[k] = v; end
        new(ComplexF64.(samples), Float64(dt), meta)
    end
end

"""
    OptimizedPulse(w_mat, pwr_level, dt; kwargs...)

Convenience constructor from a `[2 × N_TS]` Cartesian waveform in rad/s.

`pwr_level` (rad/s) is used to normalise amplitudes to [-1, 1].
"""
function OptimizedPulse(w_mat::Matrix{Float64}, pwr_level::Float64, dt::Float64; kwargs...)
    size(w_mat, 1) == 2 ||
        throw(ArgumentError("w_mat must be [2 × N_TS]; got size $(size(w_mat))"))
    pwr_level > 0 ||
        throw(ArgumentError("pwr_level must be positive"))
    samples = ComplexF64.(w_mat[1, :] ./ pwr_level .+ im .* (w_mat[2, :] ./ pwr_level))
    OptimizedPulse(samples, dt; kwargs...)
end

# ============================================================================
# Export context
# ============================================================================

"""
    ExportContext

Bundles all runtime options passed from `export_pulse` to an exporter function.
"""
struct ExportContext
    pulse      :: OptimizedPulse
    save       :: Bool
    output_dir :: String
    format     :: Any
    kwargs     :: Dict{Symbol,Any}
end

# ============================================================================
# Registry
# ============================================================================

const _EXPORTERS = Dict{Tuple{Symbol,Symbol}, Function}()

"""
    to_symbol(x) → Symbol

Normalise application/vendor identifiers to lowercase symbols.
"""
to_symbol(x::Symbol)          = Symbol(lowercase(String(x)))
to_symbol(x::AbstractString)  = Symbol(lowercase(strip(String(x))))

"""
    register_exporter(application, vendor, f)

Register `f` as the exporter for `(application, vendor)`.
Raises an error if the key is already registered.
"""
function register_exporter(application::Union{Symbol,AbstractString},
                            vendor     ::Union{Symbol,AbstractString},
                            f          ::Function)
    key = (to_symbol(application), to_symbol(vendor))
    haskey(_EXPORTERS, key) &&
        error("Exporter for $(key) is already registered; use replace_exporter to override")
    _EXPORTERS[key] = f
    return f
end

"""
    replace_exporter(application, vendor, f)

Like `register_exporter` but silently overwrites an existing entry.
Useful for user-provided custom exporters.
"""
function replace_exporter(application, vendor, f::Function)
    _EXPORTERS[(to_symbol(application), to_symbol(vendor))] = f
    return f
end

"""
    list_exporters() → Vector{Tuple{Symbol,Symbol}}

Return all registered `(application, vendor)` keys.
"""
list_exporters() = sort(collect(keys(_EXPORTERS)))

# ============================================================================
# Main dispatcher
# ============================================================================

"""
    export_pulse(pulse; application, vendor, save, output_dir, format, kwargs...)
        → vendor-specific result struct

Export `pulse` to the requested `(application, vendor)` format.

# Arguments
- `pulse::OptimizedPulse` — canonical optimised pulse
- `application::Symbol`   — one of `:NMR_EPR`, `:MRI`, `:QC`
- `vendor::Symbol`        — format identifier (see module docstring)
- `save::Bool=true`       — write files to `output_dir` if `true`; no I/O if `false`
- `output_dir::String`    — directory for output files (created if needed)
- `format`                — optional sub-format hint (vendor-specific)
- `kwargs...`             — forwarded to the exporter (e.g. `name="my_pulse"`)
"""
function export_pulse(
    pulse      :: OptimizedPulse;
    application:: Union{Symbol,AbstractString},
    vendor     :: Union{Symbol,AbstractString},
    save       :: Bool            = true,
    output_dir :: AbstractString  = "output",
    format                        = nothing,
    kwargs...
)
    key = (to_symbol(application), to_symbol(vendor))
    f   = get(_EXPORTERS, key, nothing)
    f === nothing &&
        error("""
No exporter registered for application=$(application), vendor=$(vendor).
Registered exporters: $(join(["$(a)/$(v)" for (a,v) in list_exporters()], ", "))
""")
    ctx = ExportContext(pulse, save, String(output_dir), format,
                        Dict{Symbol,Any}(kwargs))
    return f(ctx)
end

# ============================================================================
# Internal helpers shared across exporters
# ============================================================================

# Amplitude normalised to [0, 1] and phase in degrees [0, 360)
function _amp_phase(samples::Vector{ComplexF64})
    amp   = abs.(samples)
    mx    = maximum(amp)
    mx < 1e-30 && (mx = 1.0)
    amp_n = amp ./ mx
    phase = mod.(rad2deg.(angle.(samples)), 360.0)
    return amp_n, phase, mx
end

_pulse_name(ctx::ExportContext) =
    get(ctx.kwargs, :name, get(ctx.pulse.meta, :name, "pulsar_pulse"))

function _ensure_dir(ctx::ExportContext)
    ctx.save && mkpath(ctx.output_dir)
end

_bfmt(x::Float64) = replace(@sprintf("%.6E", x), "E+" => "E")

# ============================================================================
# Return types
# ============================================================================

"""
    BrukerShape

Result of the Bruker TopSpin NMR shape exporter.

# Fields
- `content::String`                    — full file content as a string
- `filepath::Union{String,Nothing}`    — path written to disk (or `nothing`)
- `amp_pct::Vector{Float64}`           — amplitude in percent (0–100)
- `phase_deg::Vector{Float64}`         — phase in degrees (0–360)
"""
struct BrukerShape
    content   :: String
    filepath  :: Union{String,Nothing}
    amp_pct   :: Vector{Float64}
    phase_deg :: Vector{Float64}
end

"""
    JEOLShape

Result of the JEOL Delta NMR shape exporter.

# Fields
- `content::String`                 — full .jrf file content
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
- `amp_norm::Vector{Float64}`       — normalised amplitude [0, 1]
- `phase_deg::Vector{Float64}`      — phase in degrees [0, 360)
"""
struct JEOLShape
    content   :: String
    filepath  :: Union{String,Nothing}
    amp_norm  :: Vector{Float64}
    phase_deg :: Vector{Float64}
end

"""
    EPRShape

Result of the Bruker Xepr/Xenon EPR shape exporter.

# Fields
- `content::String`                 — AWG IQ table content
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
- `I::Vector{Float64}`              — in-phase channel, normalised [-1, 1]
- `Q::Vector{Float64}`              — quadrature channel, normalised [-1, 1]
"""
struct EPRShape
    content  :: String
    filepath :: Union{String,Nothing}
    I        :: Vector{Float64}
    Q        :: Vector{Float64}
end

"""
    PulseqRFShape

Minimal Pulseq [SHAPES] entry.
"""
struct PulseqRFShape
    id    :: Int
    mag   :: Vector{Float64}   # magnitude [0, 1]
    phase :: Vector{Float64}   # phase [0, 360)
end

"""
    PulseqRFEvent

Minimal Pulseq RF event referencing a shape.
"""
struct PulseqRFEvent
    id            :: Int
    shape_id      :: Int
    amp_hz        :: Float64    # peak amplitude in Hz
    freq_offset   :: Float64    # frequency offset in Hz
    phase_offset  :: Float64    # phase offset in degrees
    duration_us   :: Float64    # duration in microseconds
end

"""
    PulseqSequence

Result of the Pulseq MRI exporter.

# Fields
- `content::String`                 — full .seq file content
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
- `shapes::Vector{PulseqRFShape}`
- `events::Vector{PulseqRFEvent}`
"""
struct PulseqSequence
    content  :: String
    filepath :: Union{String,Nothing}
    shapes   :: Vector{PulseqRFShape}
    events   :: Vector{PulseqRFEvent}
end

"""
    QiskitWaveformExport

Result of the Qiskit Pulse exporter.

# Fields
- `samples::Vector{ComplexF64}`     — complex envelope (same as input)
- `name::String`                    — waveform name
- `dt::Float64`                     — time step in seconds
- `json_dict::Dict{String,Any}`     — JSON-serialisable dict matching Qiskit `Waveform` init args
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
"""
struct QiskitWaveformExport
    samples   :: Vector{ComplexF64}
    name      :: String
    dt        :: Float64
    json_dict :: Dict{String,Any}
    filepath  :: Union{String,Nothing}
end

"""
    QuilTExport

Result of the Quil-T exporter.

# Fields
- `snippet::String`                 — `DEFWAVEFORM … PULSE …` text
- `name::String`                    — waveform identifier
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
"""
struct QuilTExport
    snippet  :: String
    name     :: String
    filepath :: Union{String,Nothing}
end

"""
    QUAExport

Result of the QUA (Quantum Machines) exporter.

# Fields
- `snippet::String`                 — Python-compatible I/Q declaration + `play()` call
- `name::String`                    — waveform identifier
- `I::Vector{Float64}`              — in-phase samples normalised to [-0.5, 0.5]
- `Q::Vector{Float64}`              — quadrature samples normalised to [-0.5, 0.5]
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
"""
struct QUAExport
    snippet  :: String
    name     :: String
    I        :: Vector{Float64}
    Q        :: Vector{Float64}
    filepath :: Union{String,Nothing}
end

"""
    PulserExport

Result of the Pasqal Pulser exporter.

# Fields
- `times_us::Vector{Float64}`       — step start times in microseconds
- `amp_norm::Vector{Float64}`       — amplitude per step, normalised to its peak
- `phase_rad::Vector{Float64}`      — phase per step in radians (unwrapped)
- `omega_max_rad_per_us::Float64`   — peak Rabi frequency Ω (rad/µs)
- `dt_ns::Int`                      — step length in ns
- `script::String`                  — Python Pulser script
- `filepath::Union{String,Nothing}` — path written (or `nothing`)
"""
struct PulserExport
    times_us             :: Vector{Float64}
    amp_norm             :: Vector{Float64}
    phase_rad            :: Vector{Float64}
    omega_max_rad_per_us :: Float64
    dt_ns                :: Int
    script               :: String
    filepath             :: Union{String,Nothing}
end

# ============================================================================
# Exporter 1 — NMR/EPR · Bruker TopSpin JCAMP-DX
# ============================================================================

function _export_bruker_nmr(ctx::ExportContext)
    p        = ctx.pulse
    N        = length(p.samples)
    amp_n, phase, mx = _amp_phase(p.samples)
    amp_pct  = amp_n .* 100.0
    T_pulse  = N * p.dt

    flip_deg  = Float64(get(p.meta, :flip_angle,   180.0))
    bw_hz     = Float64(get(p.meta, :bandwidth_hz, 0.0))
    exmode    = String(get(p.meta, :shape_exmode,  ""))
    title     = String(get(p.meta, :title,         "Pulsar optimised pulse"))
    shape_type= String(get(p.meta, :shape_type,    "Optimal_control_pulse"))

    name      = _pulse_name(ctx)

    flip_rad  = flip_deg * π / 180.0
    bwfac     = T_pulse * bw_hz
    integfac  = flip_rad / max(mx * 2π * T_pulse, 1e-30)

    now_dt   = now()
    date_str = Dates.format(now_dt, "yyyy/mm/dd")
    time_str = Dates.format(now_dt, "HH:MM:SS")

    io = IOBuffer()
    write(io, "##TITLE= $(title)\n")
    write(io, "##JCAMP-DX= 5.00 Bruker JCAMP library\n")
    write(io, "##DATA TYPE= Shape Data\n")
    write(io, "##ORIGIN= Pulsar.jl\n")
    write(io, "##OWNER= \n")
    write(io, "##DATE= $(date_str)\n")
    write(io, "##TIME= $(time_str)\n")
    write(io, "##\$SHAPE_PARAMETERS= Type: Pulsar optimised pulse; N=$(N); dt=$(round(p.dt*1e9,digits=2)) ns\n")
    write(io, "##MINX= $(_bfmt(minimum(amp_pct)))\n")
    write(io, "##MAXX= $(_bfmt(maximum(amp_pct)))\n")
    write(io, "##MINY= $(_bfmt(minimum(phase)))\n")
    write(io, "##MAXY= $(_bfmt(maximum(phase)))\n")
    write(io, "##\$SHAPE_EXMODE= $(exmode)\n")
    write(io, "##\$SHAPE_TOTROT= $(_bfmt(flip_deg))\n")
    write(io, "##\$SHAPE_TYPE= $(shape_type)\n")
    write(io, "##\$SHAPE_USER_DEF= \n")
    write(io, "##\$SHAPE_REPHFAC= \n")
    write(io, "##\$SHAPE_BWFAC= $(_bfmt(bwfac))\n")
    write(io, "##\$SHAPE_BWFAC50= \n")
    write(io, "##\$SHAPE_INTEGFAC= $(_bfmt(integfac))\n")
    write(io, "##\$SHAPE_MODE= 0\n")
    write(io, "##NPOINTS= $(N)\n")
    write(io, "##XYPOINTS= (XY..XY)\n")
    for n in 1:N
        write(io, "$(_bfmt(amp_pct[n])), $(_bfmt(phase[n]))\n")
    end
    write(io, "##END")

    content = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name).shape")
        write(fp, content)
    end
    return BrukerShape(content, fp, amp_pct, phase)
end

# ============================================================================
# Exporter 2 — NMR/EPR · JEOL Delta
# ============================================================================

function _export_jeol(ctx::ExportContext)
    p         = ctx.pulse
    N         = length(p.samples)
    amp_n, phase, _ = _amp_phase(p.samples)
    name      = _pulse_name(ctx)
    flip_deg  = Float64(get(p.meta, :flip_angle,   180.0))
    T_pulse_us = N * p.dt * 1e6

    now_dt   = now()
    date_str = Dates.format(now_dt, "dd-MMM-yyyy HH:MM:SS")

    io = IOBuffer()
    # JEOL Delta RF shape format header
    write(io, "# JEOL Delta RF Shape File\n")
    write(io, "# Generated by Pulsar.jl  $(date_str)\n")
    write(io, "# Name: $(name)\n")
    write(io, "# Points: $(N)\n")
    write(io, "# Duration (us): $(round(T_pulse_us, digits=3))\n")
    write(io, "# Flip angle (deg): $(round(flip_deg, digits=2))\n")
    write(io, "# Columns: amplitude(0-1)  phase(deg)\n")
    write(io, "#\n")
    for n in 1:N
        @printf(io, "%.8f  %.6f\n", amp_n[n], phase[n])
    end

    content = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name).jrf")
        write(fp, content)
    end
    return JEOLShape(content, fp, amp_n, phase)
end

# ============================================================================
# Exporter 3 — NMR/EPR · Bruker EPR (Xepr/Xenon AWG IQ table)
# ============================================================================

function _export_epr_bruker(ctx::ExportContext)
    p    = ctx.pulse
    N    = length(p.samples)
    name = _pulse_name(ctx)

    # EPR AWG tables: I and Q normalised to [-1, 1]
    mx = maximum(abs.(p.samples))
    mx < 1e-30 && (mx = 1.0)
    I_arr = real.(p.samples) ./ mx
    Q_arr = imag.(p.samples) ./ mx

    # Default EPR sample rate: 1 GSa/s or use :sample_rate_gs from meta/kwargs
    sr_gs = Float64(get(ctx.kwargs, :sample_rate_gs,
                    get(p.meta, :sample_rate_gs, 1.0)))
    now_dt   = now()
    date_str = Dates.format(now_dt, "dd-MMM-yyyy HH:MM:SS")

    io = IOBuffer()
    write(io, "# Bruker EPR AWG IQ Shape File\n")
    write(io, "# Generated by Pulsar.jl  $(date_str)\n")
    write(io, "# Name: $(name)\n")
    write(io, "# Points: $(N)\n")
    write(io, "# Sample rate (GSa/s): $(sr_gs)\n")
    write(io, "# Duration (ns): $(round(N / sr_gs, digits=3))\n")
    write(io, "# Columns: I(-1 to 1)  Q(-1 to 1)\n")
    write(io, "#\n")
    for n in 1:N
        @printf(io, "%+.8f  %+.8f\n", I_arr[n], Q_arr[n])
    end

    content = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name).shp")
        write(fp, content)
    end
    return EPRShape(content, fp, I_arr, Q_arr)
end

# ============================================================================
# Exporter 4 — MRI · Pulseq (.seq)
# ============================================================================

function _export_pulseq(ctx::ExportContext)
    p    = ctx.pulse
    N    = length(p.samples)
    name = _pulse_name(ctx)

    amp_n, phase, mx_abs = _amp_phase(p.samples)

    # Pulseq .seq format version 1.4
    # [SHAPES] section uses run-length compressed magnitude/phase
    # We emit the full (uncompressed) shape as two separate shape arrays (mag, phase)

    shape_id_mag   = 1
    shape_id_phase = 2
    rf_event_id    = 1

    amp_hz   = Float64(get(ctx.kwargs, :amp_hz,
               get(p.meta, :amp_hz, mx_abs / (2π))))   # peak amp in Hz
    freq_off = Float64(get(ctx.kwargs, :freq_offset_hz,
               get(p.meta, :carrier_hz, 0.0)))
    phase_off= Float64(get(ctx.kwargs, :phase_offset_deg, 0.0))
    dur_us   = N * p.dt * 1e6

    shape_mag   = PulseqRFShape(shape_id_mag,   amp_n,  phase)
    shape_phase = PulseqRFShape(shape_id_phase,  phase, amp_n)  # phase shape: phase column
    event       = PulseqRFEvent(rf_event_id, shape_id_mag,
                                amp_hz, freq_off, phase_off, dur_us)

    # Build .seq file string
    io = IOBuffer()

    # File header
    write(io, "# Pulseq sequence file\n")
    write(io, "# Generated by Pulsar.jl\n")
    write(io, "# Name: $(name)\n")
    write(io, "\n")
    write(io, "[VERSION]\n")
    write(io, "major 1\n")
    write(io, "minor 4\n")
    write(io, "revision 1\n")
    write(io, "\n")

    # [DEFINITIONS]
    write(io, "[DEFINITIONS]\n")
    write(io, "Name $(name)\n")
    @printf(io, "Nx %d\n", N)
    @printf(io, "FOV 0 0 0\n")
    write(io, "\n")

    # [RF] events table: id amp_hz mag_id phase_id time_shape_id delay_us freq_hz phase_deg
    write(io, "[RF]\n")
    @printf(io, "# id  amp(Hz)       mag_id  phase_id  time_id  delay(us)  freq(Hz)  phase(deg)\n")
    @printf(io, "%d  %.6e  %d  %d  0  0  %.6f  %.6f\n",
            rf_event_id, amp_hz, shape_id_mag, shape_id_phase,
            freq_off, phase_off)
    write(io, "\n")

    # [BLOCKS]: single block containing the RF event
    write(io, "[BLOCKS]\n")
    write(io, "# id  delay  rf  gx  gy  gz  adc  ext\n")
    write(io, "1  0  $(rf_event_id)  0  0  0  0  0\n")
    write(io, "\n")

    # [SHAPES]: compressed RLE format
    # Pulsar emits uncompressed (each sample on its own line) for clarity
    write(io, "[SHAPES]\n")
    write(io, "\n")
    write(io, "shape_id $(shape_id_mag)\n")
    write(io, "num_samples $(N)\n")
    for v in amp_n
        @printf(io, "%.9f\n", v)
    end
    write(io, "\n")
    write(io, "shape_id $(shape_id_phase)\n")
    write(io, "num_samples $(N)\n")
    for v in phase
        # Pulseq stores phase shape as normalised [0, 1] fraction of 360°
        @printf(io, "%.9f\n", v / 360.0)
    end
    write(io, "\n")

    content = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name).seq")
        write(fp, content)
    end
    return PulseqSequence(content, fp,
                          [shape_mag, shape_phase],
                          [event])
end

# ============================================================================
# Exporter 5 — QC · Qiskit Pulse
# ============================================================================

function _export_qiskit(ctx::ExportContext)
    p    = ctx.pulse
    name = _pulse_name(ctx)

    # Qiskit Waveform: complex samples normalised so max amplitude ≤ 1
    mx = maximum(abs.(p.samples))
    mx < 1e-30 && (mx = 1.0)
    samples_norm = p.samples ./ mx

    # JSON-serialisable dict matching qiskit.pulse.library.Waveform keyword args
    json_dict = Dict{String,Any}(
        "name"       => name,
        "dt"         => p.dt,
        "samples"    => [[real(s), imag(s)] for s in samples_norm],
        "epsilon"    => get(ctx.kwargs, :epsilon, 1e-7),
        "zero_amp_t" => get(ctx.kwargs, :zero_amp_tol, 1e-10),
    )

    # Build a minimal Python snippet for reference
    io = IOBuffer()
    write(io, "# Qiskit Pulse — generated by Pulsar.jl\n")
    write(io, "# Load with:\n")
    write(io, "#   import json, numpy as np\n")
    write(io, "#   from qiskit.pulse.library import Waveform\n")
    write(io, "#   with open('$(name).json') as f: d = json.load(f)\n")
    write(io, "#   wf = Waveform(np.array([complex(s[0], s[1]) for s in d['samples']]),\n")
    write(io, "#                 name=d['name'])\n")

    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name).json")
        # Write minimal JSON without requiring JSON3
        open(fp, "w") do io_f
            write(io_f, "{\n")
            write(io_f, "  \"name\": \"$(name)\",\n")
            write(io_f, "  \"dt\": $(p.dt),\n")
            write(io_f, "  \"num_samples\": $(length(samples_norm)),\n")
            write(io_f, "  \"samples\": [\n")
            for (k, s) in enumerate(samples_norm)
                sep = k < length(samples_norm) ? "," : ""
                @printf(io_f, "    [%.10f, %.10f]%s\n", real(s), imag(s), sep)
            end
            write(io_f, "  ]\n}\n")
        end
    end
    return QiskitWaveformExport(samples_norm, name, p.dt, json_dict, fp)
end

# ============================================================================
# Exporter 6 — QC · Quil-T
# ============================================================================

function _export_quil_t(ctx::ExportContext)
    p    = ctx.pulse
    N    = length(p.samples)
    name = _pulse_name(ctx)
    qubit = get(ctx.kwargs, :qubit, 0)
    frame = get(ctx.kwargs, :frame, "\"rf\"")

    mx = maximum(abs.(p.samples))
    mx < 1e-30 && (mx = 1.0)
    samples_norm = p.samples ./ mx

    io = IOBuffer()
    write(io, "# Quil-T waveform — generated by Pulsar.jl\n")
    write(io, "DEFWAVEFORM $(name):\n")
    for (k, s) in enumerate(samples_norm)
        sep = k < N ? "," : ""
        @printf(io, "    (%.10f%+.10fi)%s\n", real(s), imag(s), sep)
    end
    write(io, "\n")
    write(io, "PULSE $(qubit) $(frame) $(name)\n")

    snippet = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name).quil")
        write(fp, snippet)
    end
    return QuilTExport(snippet, name, fp)
end

# ============================================================================
# Exporter 7 — QC · QUA (Quantum Machines OPX)
# ============================================================================

function _export_qua(ctx::ExportContext)
    p    = ctx.pulse
    N    = length(p.samples)
    name = _pulse_name(ctx)
    element = String(get(ctx.kwargs, :element, "qubit"))

    # QUA uses I/Q in [-0.5, 0.5]
    mx = maximum(abs.(p.samples))
    mx < 1e-30 && (mx = 1.0)
    scale  = 0.5 / mx
    I_arr  = real.(p.samples) .* scale
    Q_arr  = imag.(p.samples) .* scale

    io = IOBuffer()
    write(io, "# QUA snippet — generated by Pulsar.jl\n")
    write(io, "# Paste into a QUA program or config dict\n")
    write(io, "\n")
    write(io, "from qm.qua import *\n")
    write(io, "\n")
    # I and Q arrays
    write(io, "$(name)_I = [")
    join(io, (@sprintf("%.10f", v) for v in I_arr), ", ")
    write(io, "]\n")
    write(io, "$(name)_Q = [")
    join(io, (@sprintf("%.10f", v) for v in Q_arr), ", ")
    write(io, "]\n")
    write(io, "\n")
    write(io, "# Inside a QUA program:\n")
    write(io, "with program() as prog:\n")
    write(io, "    play(pulse.Waveform($(name)_I, $(name)_Q), \"$(element)\")\n")

    snippet = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name)_qua.py")
        write(fp, snippet)
    end
    return QUAExport(snippet, name, I_arr, Q_arr, fp)
end

# ============================================================================
# Exporter 8 — QC · Pasqal Pulser
# ============================================================================

# Pulser's ground-rydberg convention (|r⟩ upper, Ω and δ in rad/µs):
#     H = Ω/2 (e^{iφ}|r⟩⟨g| + e^{−iφ}|g⟩⟨r|) − δ|r⟩⟨r|
# Pulsar's NeutralAtomSystem drive u_x σx/2 + u_y σy/2 in the basis
# {|g⟩ = |0⟩, |r⟩ = |1⟩} is the same with Ω = |u|, φ = atan2(u_y, u_x), and its
# detuning Δω·|r⟩⟨r| is δ = −Δω — so the sample phase exports unchanged.
#
# A pulse with a time-varying phase is Pulse.ArbitraryPhase (a plain Pulse takes
# one constant phase); Pulser realises it as a detuning δ = −dφ/dt. The waveforms
# are CustomWaveform, one sample per ns, so a piecewise-constant pulse is exact
# when dt is a whole number of ns. The phase is unwrapped and held through
# zero-amplitude steps, so no step becomes a spurious 2π or "phase of zero"
# detuning spike.
function _export_pulser(ctx::ExportContext)
    p    = ctx.pulse
    N    = length(p.samples)
    name = _pulse_name(ctx)
    channel = String(get(ctx.kwargs, :channel, "rydberg_global"))
    register_size = Int(get(ctx.kwargs, :register_size, 1))

    dt_ns_f = p.dt * 1e9
    dt_ns   = round(Int, dt_ns_f)
    (dt_ns >= 1 && abs(dt_ns_f - dt_ns) <= 1e-6 * max(1.0, dt_ns_f)) || throw(ArgumentError(
        "Pulser samples waveforms at 1 ns: dt = $(p.dt) s is $(dt_ns_f) ns, not a whole " *
        "number of ns. Resample the pulse onto a whole-ns step first."))

    amp_abs = abs.(p.samples)
    mx = maximum(amp_abs; init = 0.0)
    amp_n = mx < 1e-30 ? zeros(N) : amp_abs ./ mx
    # Peak Rabi frequency: given, or the samples read as rad/s (as for Pulseq).
    omega_max = Float64(get(ctx.kwargs, :omega_max_rad_per_us,
                            get(p.meta, :omega_max_rad_per_us, mx * 1e-6)))

    phase_rad = zeros(N)
    prev = 0.0
    for k in 1:N
        if amp_n[k] > 1e-12
            φ = angle(p.samples[k])
            φ += 2π * round((prev - φ) / (2π))          # unwrap against the last phase
            prev = φ
        end
        phase_rad[k] = prev                             # held through zero amplitude
    end
    times_us = collect(range(0.0; step = p.dt * 1e6, length = N))

    io = IOBuffer()
    write(io, "# Pasqal Pulser — generated by Pulsar.jl\n")
    write(io, "# Requires: pip install pulser\n")
    write(io, "#\n")
    write(io, "# Piecewise-constant pulse: $(N) steps of DT_NS ns; step k has Rabi frequency\n")
    write(io, "# OMEGA_MAX * amp_$(name)[k] (rad/µs) and phase phase_$(name)[k] (rad).\n")
    write(io, "# Pulser convention, = Pulsar NeutralAtomSystem with |r> = |1>:\n")
    write(io, "#   H = Omega/2 (e^{i phi}|r><g| + h.c.) - delta |r><r|\n")
    write(io, "# Pulse.ArbitraryPhase realises the time-varying phase as a detuning\n")
    write(io, "# delta = -dphi/dt: each phase step is a 1-ns detuning spike, which a\n")
    write(io, "# device's detuning limits may reject (Sequence.add raises if so).\n")
    write(io, "# The atoms' own detuning (freq_hz - carrier in Pulsar) is not part of\n")
    write(io, "# this pulse.\n")
    write(io, "\n")
    write(io, "import numpy as np\n")
    write(io, "from pulser import Pulse, Register, Sequence\n")
    write(io, "from pulser.devices import AnalogDevice\n")
    write(io, "from pulser.waveforms import CustomWaveform\n")
    write(io, "\n")
    write(io, "# Step start times (µs)\n")
    write(io, "times_us = np.array([")
    join(io, (@sprintf("%.6f", t) for t in times_us), ", ")
    write(io, "])\n\n")
    write(io, "DT_NS = $(dt_ns)\n")
    write(io, "OMEGA_MAX = $(@sprintf("%.10g", omega_max))  # rad/µs, peak Rabi frequency\n\n")
    write(io, "# Amplitude per step, normalised to OMEGA_MAX\n")
    write(io, "amp_$(name) = np.array([")
    join(io, (@sprintf("%.10f", a) for a in amp_n), ", ")
    write(io, "])\n\n")
    write(io, "# Phase per step (rad, unwrapped)\n")
    write(io, "phase_$(name) = np.array([")
    join(io, (@sprintf("%.10f", φ) for φ in phase_rad), ", ")
    write(io, "])\n\n")
    write(io, "omega_wf = CustomWaveform(np.repeat(OMEGA_MAX * amp_$(name), DT_NS))\n")
    write(io, "phase_wf = CustomWaveform(np.repeat(phase_$(name), DT_NS))\n")
    write(io, "pulse = Pulse.ArbitraryPhase(omega_wf, phase_wf)\n")
    write(io, "\n")
    write(io, "# Minimal register and sequence\n")
    if register_size == 1
        write(io, "reg = Register.square(1, prefix=\"q\")\n")
    else
        write(io, "reg = Register.square($(round(Int, sqrt(register_size))), prefix=\"q\")\n")
    end
    write(io, "seq = Sequence(reg, AnalogDevice)\n")
    write(io, "seq.declare_channel(\"ch0\", \"$(channel)\")\n")
    write(io, "seq.add(pulse, \"ch0\")\n")

    script = String(take!(io))
    _ensure_dir(ctx)
    fp = nothing
    if ctx.save
        fp = joinpath(ctx.output_dir, "$(name)_pulser.py")
        write(fp, script)
    end
    return PulserExport(times_us, amp_n, phase_rad, omega_max, dt_ns, script, fp)
end

# ============================================================================
# Register all built-in exporters at module load
# ============================================================================

register_exporter(:NMR_EPR, :bruker,     _export_bruker_nmr)
register_exporter(:NMR_EPR, :jeol,       _export_jeol)
register_exporter(:NMR_EPR, :epr_bruker, _export_epr_bruker)
register_exporter(:MRI,     :pulseq,     _export_pulseq)
register_exporter(:QC,      :qiskit,     _export_qiskit)
register_exporter(:QC,      :quil_t,     _export_quil_t)
register_exporter(:QC,      :qua,        _export_qua)
register_exporter(:QC,      :pulser,     _export_pulser)

# ============================================================================
# Pulse format loaders (inverse of exporters)
# All return OptimizedPulse so round-trip export → load → export is possible.
# ============================================================================

"""
    load_jeol_shape(filepath; max_amp=1.0, dt=nothing) → OptimizedPulse

Load a JEOL Delta RF shape file (`.jrf`).
Reads `(amp_norm, phase_deg)` columns and reconstructs complex samples.
`max_amp` rescales amplitudes from the normalized [0,1] range back to rad/s.
`dt` (seconds) is not stored in the file; pass it explicitly or accept `dt=0.0`.
"""
function load_jeol_shape(
    filepath :: String;
    max_amp  :: Float64              = 1.0,
    dt       :: Union{Float64,Nothing} = nothing,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))
    amp_n = Float64[];  phase_d = Float64[]
    n_points = 0;  name_str = ""
    for line in eachline(filepath)
        ls = strip(line)
        isempty(ls) && continue
        if startswith(ls, "# Name:") || startswith(ls, "# name:")
            name_str = strip(ls[8:end])
        elseif startswith(ls, "# Points:") || startswith(ls, "# points:")
            try n_points = parse(Int, strip(ls[10:end])) catch end
        elseif startswith(ls, "#")
            continue
        else
            parts = split(ls)
            length(parts) >= 2 || continue
            try
                push!(amp_n,   parse(Float64, parts[1]))
                push!(phase_d, parse(Float64, parts[2]))
            catch end
        end
    end
    isempty(amp_n) && error("No data found in $filepath")
    if dt === nothing
        @warn "load_jeol_shape: dt not in file; setting dt=0.0. Pass dt=<seconds>."
    end
    samples = [amp_n[i] * max_amp * exp(im * phase_d[i] * π / 180.0) for i in eachindex(amp_n)]
    OptimizedPulse(ComplexF64.(samples), dt === nothing ? 0.0 : dt;
        name=isempty(name_str) ? "jeol_pulse" : name_str, n_points=length(samples))
end

"""
    load_epr_shape(filepath; max_amp=1.0, dt=nothing) → OptimizedPulse

Load a Bruker Xepr/Xenon AWG IQ shape file (`.shp`).
Reads signed `(I, Q)` columns in `[-1, 1]` range. `max_amp` rescales to rad/s.
`dt` is not stored in the file.
"""
function load_epr_shape(
    filepath :: String;
    max_amp  :: Float64              = 1.0,
    dt       :: Union{Float64,Nothing} = nothing,
    sample_rate_gsa :: Float64       = 1.0,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))
    I_vals = Float64[];  Q_vals = Float64[]
    name_str = "";  sr = sample_rate_gsa
    for line in eachline(filepath)
        ls = strip(line)
        isempty(ls) && continue
        if startswith(ls, "# Name:") || startswith(ls, "# name:")
            name_str = strip(ls[8:end])
        elseif startswith(ls, "# Sample rate")
            m = match(r"[\d.]+", ls[14:end])
            m !== nothing && try sr = parse(Float64, m.match) catch end
        elseif startswith(ls, "#")
            continue
        else
            parts = split(ls)
            length(parts) >= 2 || continue
            try
                push!(I_vals, parse(Float64, parts[1]))
                push!(Q_vals, parse(Float64, parts[2]))
            catch end
        end
    end
    isempty(I_vals) && error("No data found in $filepath")
    dt_val = dt === nothing ? (sr > 0 ? 1.0 / (sr * 1e9) : 0.0) : dt
    dt === nothing && sr > 0 &&
        @info "load_epr_shape: inferred dt=$(dt_val*1e9) ns from sample rate $(sr) GSa/s"
    samples = ComplexF64.((I_vals .+ im .* Q_vals) .* max_amp)
    OptimizedPulse(samples, dt_val;
        name=isempty(name_str) ? "epr_pulse" : name_str, n_points=length(samples))
end

"""
    load_pulseq(filepath; dt=nothing) → OptimizedPulse

Load a Pulseq sequence file (`.seq`, v1.4) and extract the first RF pulse.
Reconstructs `samples = mag * exp(2π*im * phase_frac)`.
`dt` is inferred from the `[DEFINITIONS]` block when available.
"""
function load_pulseq(
    filepath :: String;
    dt       :: Union{Float64,Nothing} = nothing,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))
    content = read(filepath, String)

    # Extract dt from [DEFINITIONS] section (in µs — Pulseq convention)
    dt_val = dt
    m_dt = match(r"\[DEFINITIONS\].*?Duration\s+([\d.eE+-]+)", content; flags=Base.PCRE.DOTALL)
    if m_dt !== nothing && dt === nothing
        try
            total_us = parse(Float64, m_dt.captures[1])
            # dt will be set after we know n_points
        catch end
    end

    # Extract shapes from [SHAPES] section
    # Each shape block: "num_samples N\n" followed by N float lines
    shape_blocks = Dict{Int,Vector{Float64}}()
    shape_section = match(r"\[SHAPES\](.*?)(?=\[|\z)", content; flags=Base.PCRE.DOTALL)
    if shape_section !== nothing
        for m_block in eachmatch(r"shape_id\s+(\d+)\s*\nnum_samples\s+(\d+)\s*\n([\d.\n eE+-]+?)(?=shape_id|\z)",
                                  shape_section.captures[1]; flags=Base.PCRE.DOTALL)
            id   = parse(Int, m_block.captures[1])
            n    = parse(Int, m_block.captures[2])
            vals = [parse(Float64, strip(l)) for l in split(strip(m_block.captures[3]), "\n")
                    if !isempty(strip(l))]
            shape_blocks[id] = vals[1:min(n, length(vals))]
        end
    end

    # Find first RF event from [RF] section
    rf_section = match(r"\[RF\](.*?)(?=\[|\z)", content; flags=Base.PCRE.DOTALL)
    mag_id = phase_id = amp_hz = 0.0
    n_mag_id = n_phase_id = 0
    if rf_section !== nothing
        # columns: id amp_hz mag_id phase_id time_id delay freq phase
        for line in split(rf_section.captures[1], "\n")
            ls = strip(line)
            isempty(ls) || startswith(ls, "#") && continue
            parts = split(ls)
            length(parts) >= 4 || continue
            try
                amp_hz    = parse(Float64, parts[2])
                n_mag_id  = parse(Int, parts[3])
                n_phase_id= parse(Int, parts[4])
                break
            catch end
        end
    end

    mag_shape   = get(shape_blocks, n_mag_id,   Float64[])
    phase_shape = get(shape_blocks, n_phase_id, Float64[])

    isempty(mag_shape) && error("No RF shape data found in $filepath")

    n = length(mag_shape)
    phase_s = length(phase_shape) == n ? phase_shape : zeros(n)
    samples = ComplexF64[mag_shape[i] * exp(2π * im * phase_s[i]) for i in 1:n]

    if dt_val === nothing
        @warn "load_pulseq: dt not determined; setting dt=0.0. Pass dt=<seconds>."
        dt_val = 0.0
    end

    OptimizedPulse(samples, dt_val; n_points=n, amp_hz=amp_hz)
end

"""
    load_qiskit_waveform(filepath) → OptimizedPulse

Load a Qiskit Pulse waveform JSON file. Extracts `dt` and complex `samples`
directly from the JSON fields written by `export_pulse(...; vendor=:qiskit)`.
"""
function load_qiskit_waveform(filepath::String)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))
    content = read(filepath, String)

    # Parse dt
    m_dt = match(r""""dt"\s*:\s*([\d.eE+\-]+)""", content)
    dt_val = m_dt !== nothing ? parse(Float64, m_dt.captures[1]) : 0.0

    # Parse name
    m_name = match(r"\"name\"\s*:\s*\"([^\"]+)\"", content)
    name_str = m_name !== nothing ? m_name.captures[1] : "qiskit_pulse"

    # Parse samples array: [[re, im], [re, im], ...]
    samples = ComplexF64[]
    for m in eachmatch(r"\[\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\]", content)
        re = parse(Float64, m.captures[1])
        im_val = parse(Float64, m.captures[2])
        push!(samples, re + im * im_val)
    end

    isempty(samples) && error("No samples found in $filepath")
    OptimizedPulse(samples, dt_val; name=name_str, n_points=length(samples))
end

"""
    load_quil_t(filepath; max_amp=1.0, dt=nothing) → OptimizedPulse

Load a Quil-T waveform file (`.quil`). Parses the `DEFWAVEFORM` block and
extracts complex samples in `(real+imagj)` format. `dt` is not stored in the
file and must be supplied by the caller.
"""
function load_quil_t(
    filepath :: String;
    max_amp  :: Float64              = 1.0,
    dt       :: Union{Float64,Nothing} = nothing,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))
    content = read(filepath, String)

    m_block = match(r"DEFWAVEFORM\s+(\S+)[^\n]*\n([\s\S]+?)(?=\n\S|\z)", content)
    m_block === nothing && error("No DEFWAVEFORM block found in $filepath")

    name_str = m_block.captures[1]
    block    = m_block.captures[2]

    samples = ComplexF64[]
    # Accept either Python ('j') or Julia ('i'/'im') imaginary suffix.
    for m in eachmatch(r"\(\s*([\d.eE+\-]+)\s*([+\-]\s*[\d.eE+\-]+)\s*(?:j|im|i)\s*\)", block)
        re  = parse(Float64, m.captures[1])
        im_str = replace(m.captures[2], " " => "")
        im_val = parse(Float64, im_str)
        push!(samples, (re + im * im_val) * max_amp)
    end

    isempty(samples) && error("No sample data found in DEFWAVEFORM block of $filepath")
    if dt === nothing
        @warn "load_quil_t: dt not in file; setting dt=0.0. Pass dt=<seconds>."
    end
    OptimizedPulse(samples, dt === nothing ? 0.0 : dt;
        name=name_str, n_points=length(samples))
end

"""
    load_qua(filepath; dt=nothing) → OptimizedPulse

Load a QUA OPX waveform file (`_qua.py`). Reads I/Q arrays from the Python
snippet (or a sidecar `_data.json` if present) and rescales from `[-0.5, 0.5]`.
`dt` must be supplied (not stored in the Python file).
"""
function load_qua(
    filepath :: String;
    dt       :: Union{Float64,Nothing} = nothing,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))

    # Try sidecar JSON first
    json_path = replace(filepath, r"_qua\.py$" => "_qua_data.json")
    if isfile(json_path)
        return _load_qua_from_json(json_path, dt)
    end

    content = read(filepath, String)

    # Extract name from variable names: <name>_I = [...]
    m_name = match(r"(\w+)_I\s*=\s*\[", content)
    name_str = m_name !== nothing ? m_name.captures[1] : "qua_pulse"

    # Extract I array
    m_I = match(r"$(name_str)_I\s*=\s*\[([\d.,\s+\-eE]+)\]", content)
    m_I === nothing && (m_I = match(r"_I\s*=\s*\[([\d.,\s+\-eE]+)\]", content))

    # Extract Q array
    m_Q = match(r"$(name_str)_Q\s*=\s*\[([\d.,\s+\-eE]+)\]", content)
    m_Q === nothing && (m_Q = match(r"_Q\s*=\s*\[([\d.,\s+\-eE]+)\]", content))

    (m_I === nothing || m_Q === nothing) && error("Could not parse I/Q arrays from $filepath")

    parse_array(s) = [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
    I_arr = parse_array(m_I.captures[end])
    Q_arr = parse_array(m_Q.captures[end])

    length(I_arr) == length(Q_arr) ||
        error("I and Q arrays have different lengths in $filepath")

    # Rescale from [-0.5, 0.5] → full amplitude
    scale = maximum(abs, vcat(I_arr, Q_arr))
    scale = scale > 0 ? 0.5 / scale : 1.0
    samples = ComplexF64.((I_arr .+ im .* Q_arr) ./ 0.5)

    dt === nothing &&
        @warn "load_qua: dt not in file; setting dt=0.0. Pass dt=<seconds>."
    OptimizedPulse(samples, dt === nothing ? 0.0 : dt;
        name=name_str, n_points=length(samples))
end

function _load_qua_from_json(json_path::String, dt::Union{Float64,Nothing})::OptimizedPulse
    content = read(json_path, String)
    m_dt   = match(r""""dt"\s*:\s*([\d.eE+\-]+)""", content)
    m_name = match(r"\"name\"\s*:\s*\"([^\"]+)\"", content)
    dt_val = (dt !== nothing) ? dt :
             (m_dt !== nothing ? parse(Float64, m_dt.captures[1]) : 0.0)
    name_str = m_name !== nothing ? m_name.captures[1] : "qua_pulse"
    I_arr = [parse(Float64, m.captures[1])
             for m in eachmatch(r""""I"\s*:\s*([\d.eE+\-]+)""", content)]
    if isempty(I_arr)
        m_I = match(r""""I"\s*:\s*\[([\d.,\s+\-eE]+)\]""", content)
        m_I !== nothing && (I_arr = [parse(Float64, strip(x)) for x in split(m_I.captures[1], ",")])
    end
    Q_arr = [parse(Float64, m.captures[1])
             for m in eachmatch(r""""Q"\s*:\s*([\d.eE+\-]+)""", content)]
    if isempty(Q_arr)
        m_Q = match(r""""Q"\s*:\s*\[([\d.,\s+\-eE]+)\]""", content)
        m_Q !== nothing && (Q_arr = [parse(Float64, strip(x)) for x in split(m_Q.captures[1], ",")])
    end
    isempty(I_arr) && error("Could not parse I array from $json_path")
    samples = ComplexF64.(I_arr .+ im .* Q_arr)
    OptimizedPulse(samples, dt_val; name=name_str, n_points=length(samples))
end

"""
    load_pulser(filepath; max_amp=nothing, dt=nothing) → OptimizedPulse

Load a Pasqal Pulser script (`_pulser.py`). Extracts `times_us`, `amp`, and
`phase` arrays (or sidecar `_data.json`) and reconstructs complex samples.
`dt` is inferred from `times_us` spacing when possible. The amplitudes are
scaled by `max_amp`; when it is not given, a script's `OMEGA_MAX` (rad/µs) is
used and the samples come back in rad/s — the inverse of `export_pulse`
(otherwise 1.0, i.e. normalised).
"""
function load_pulser(
    filepath :: String;
    max_amp  :: Union{Float64,Nothing} = nothing,
    dt       :: Union{Float64,Nothing} = nothing,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))

    # Try sidecar JSON first
    json_path = replace(filepath, r"_pulser\.py$" => "_pulser_data.json")
    if isfile(json_path)
        return _load_pulser_from_json(json_path, dt; max_amp = something(max_amp, 1.0))
    end

    content = read(filepath, String)
    if max_amp === nothing
        m_Ω = match(r"OMEGA_MAX\s*=\s*([\d.eE+\-]+)", content)
        max_amp = m_Ω === nothing ? 1.0 : parse(Float64, m_Ω.captures[1]) * 1e6   # rad/µs → rad/s
    end

    # Extract times_us
    m_t = match(r"times_us\s*=\s*np\.array\(\[([\d.,\s+\-eE]+)\]\)", content)
    # Extract amplitude array (amp_<name> = np.array([...]))
    m_a = match(r"amp_\w+\s*=\s*np\.array\(\[([\d.,\s+\-eE]+)\]\)", content)
    # Extract phase array (phase_<name> = np.array([...]))
    m_p = match(r"phase_\w+\s*=\s*np\.array\(\[([\d.,\s+\-eE]+)\]\)", content)

    (m_a === nothing || m_p === nothing) &&
        error("Could not parse amplitude/phase arrays from $filepath")

    parse_arr(s) = [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
    amp_arr   = parse_arr(m_a.captures[1])
    phase_arr = parse_arr(m_p.captures[1])
    times_arr = m_t !== nothing ? parse_arr(m_t.captures[1]) : Float64[]

    length(amp_arr) == length(phase_arr) ||
        error("amp and phase arrays have different lengths in $filepath")

    dt_val = if dt !== nothing
        dt
    elseif length(times_arr) >= 2
        (times_arr[2] - times_arr[1]) * 1e-6  # µs → s
    else
        @warn "load_pulser: dt not determined; setting dt=0.0. Pass dt=<seconds>."
        0.0
    end

    samples = ComplexF64.(amp_arr .* max_amp .* exp.(im .* phase_arr))
    OptimizedPulse(samples, dt_val; n_points=length(samples))
end

function _load_pulser_from_json(json_path::String, dt::Union{Float64,Nothing};
                                  max_amp::Float64=1.0)::OptimizedPulse
    content = read(json_path, String)
    m_dt = match(r""""dt"\s*:\s*([\d.eE+\-]+)""", content)
    dt_val = dt !== nothing ? dt :
             m_dt !== nothing ? parse(Float64, m_dt.captures[1]) : 0.0
    m_t = match(r""""times_us"\s*:\s*\[([\d.,\s+\-eE]+)\]""", content)
    m_a = match(r""""amp"\s*:\s*\[([\d.,\s+\-eE]+)\]""", content)
    m_p = match(r""""phase"\s*:\s*\[([\d.,\s+\-eE]+)\]""", content)
    (m_a === nothing || m_p === nothing) &&
        error("Could not parse amp/phase from $json_path")
    parse_arr(s) = [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
    amp_arr   = parse_arr(m_a.captures[1])
    phase_arr = parse_arr(m_p.captures[1])
    if dt_val == 0.0 && m_t !== nothing
        t = parse_arr(m_t.captures[1])
        length(t) >= 2 && (dt_val = (t[2] - t[1]) * 1e-6)
    end
    samples = ComplexF64.(amp_arr .* max_amp .* exp.(im .* phase_arr))
    OptimizedPulse(samples, dt_val; n_points=length(samples))
end

end  # module Output
