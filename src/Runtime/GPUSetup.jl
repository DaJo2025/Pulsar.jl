# Runtime/GPUSetup.jl — zero-config GPU detection, install, and auto-setup.
#
# Design (see docs/GPUBackendsGuide.md "Zero-config setup"):
#   • detect_gpu_hardware()  — pure, side-effect-free hardware probe.
#   • setup_gpu!()           — explicit, one-time installer: Pkg.add the right
#                              GPU package into the ACTIVE environment and record
#                              a Preference.  Never runs during precompile/__init__
#                              (that is unsafe), and auto-skips CI/read-only envs.
#   • load_gpu!()            — import the installed GPU package (activating the
#                              extension) and set the device.
#   • __init__ hook          — if the user has run setup_gpu! (preference
#                              `auto_load_gpu`), auto-load on every `using Pulsar`;
#                              otherwise print a one-time "GPU detected" hint.
#
# Why not auto-install at precompile/init: `Pkg.add` is unsafe in the
# precompilation sandbox and dangerous inside `__init__` (it can hang on a
# download, recurse, or mutate a shared/CI/read-only environment).  Detection is
# always safe; installation is an explicit user action.

using Preferences

const _PULSAR_UUID = Base.UUID("54ac5e6d-f0b9-4c2a-a5c2-b0909a9fa55a")

# ---------------------------------------------------------------------------
# Hardware detection (pure)
# ---------------------------------------------------------------------------

# Pure decision rule — Metal wins on Apple Silicon (where CUDA is unavailable
# anyway), otherwise CUDA if an NVIDIA driver is present, else CPU.  Factored
# out so the logic is unit-testable without GPU hardware.
_decide_backend(has_metal::Bool, has_cuda::Bool)::Symbol =
    has_metal ? :metal : (has_cuda ? :cuda : :cpu)

# Cheap NVIDIA probe that does NOT require CUDA.jl: prefer `nvidia-smi` (its
# presence implies a working driver), then Linux device nodes / proc.
function _detect_nvidia()::Bool
    try
        Sys.which("nvidia-smi") !== nothing && return true
    catch
    end
    if Sys.islinux()
        try
            isfile("/proc/driver/nvidia/version") && return true
            isdir("/dev") && any(startswith("nvidia"), readdir("/dev")) && return true
        catch
        end
    end
    return false
end

"""
    detect_gpu_hardware() -> Symbol

Detect the GPU backend this machine can use, **without loading any GPU package**:
`:metal` on Apple Silicon, `:cuda` if an NVIDIA driver is present, else `:cpu`.
Pure and side-effect-free — safe to call anywhere (including a build script).
"""
detect_gpu_hardware()::Symbol = _decide_backend(_can_use_metal(), _detect_nvidia())

# ---------------------------------------------------------------------------
# Package helpers
# ---------------------------------------------------------------------------

_gpu_pkg_name(dev::Symbol) = dev === :cuda ? "CUDA" :
                             dev === :metal ? "Metal" : nothing

# Is the GPU package loadable in the current environment (project or a stacked
# env)?  Used to decide whether `setup_gpu!` needs to install it.
function _gpu_pkg_loadable(name::AbstractString)::Bool
    try
        return Base.identify_package(name) !== nothing
    catch
        return false
    end
end

# Lazily load Pkg only when actually installing (keeps it off the `using Pulsar`
# hot path).
_load_pkg() = Base.require(Base.PkgId(
    Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))

# Guard against installing into an environment we should not mutate.
function _install_blocked()::Tuple{Bool,String}
    get(ENV, "PULSAR_NO_GPU_INSTALL", "") in ("1", "true") &&
        return (true, "PULSAR_NO_GPU_INSTALL is set")
    get(ENV, "CI", "") in ("1", "true") &&
        return (true, "CI environment detected")
    proj = Base.active_project()
    if proj !== nothing
        d = dirname(proj)
        writable = try
            t = joinpath(d, ".pulsar_write_test_$(getpid())")
            touch(t); rm(t; force=true); true
        catch
            false
        end
        writable || return (true, "active project directory is not writable ($d)")
    end
    return (false, "")
end

# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

_preferred_device()::Symbol =
    Symbol(Preferences.load_preference(_PULSAR_UUID, "gpu_backend", "cpu"))

_auto_load_pref()::Bool =
    Preferences.load_preference(_PULSAR_UUID, "auto_load_gpu", false) === true

function _store_gpu_prefs!(dev::Symbol, auto_load::Bool)
    Preferences.set_preferences!(_PULSAR_UUID,
        "gpu_backend" => string(dev),
        "auto_load_gpu" => auto_load;
        force = true)
    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    setup_gpu!(; device=:auto, install=true, set_default=true,
                 auto_load=true, force=false) -> Symbol

One-time GPU setup for the **active environment**.  Detects the GPU (unless
`device` is given explicitly), installs the matching package (`CUDA` or `Metal`)
if it is not already available, records the choice as a persistent preference,
and — when `set_default` is true and the package is available — loads it and
makes it the default device for this session.

After running this once, every later `using Pulsar` will auto-load the GPU
(see [`load_gpu!`]).  Re-running is idempotent.

Installation is skipped automatically in CI / read-only / non-writable
environments (and when `PULSAR_NO_GPU_INSTALL` is set); pass `force=true` to
override that guard.

Returns the configured device (`:cuda`, `:metal`, or `:cpu`).
"""
function setup_gpu!(; device::Symbol = :auto,
                      install::Bool = true,
                      set_default::Bool = true,
                      auto_load::Bool = true,
                      force::Bool = false)::Symbol
    dev = device === :auto ? detect_gpu_hardware() : device
    dev in (:cpu, :cuda, :metal) ||
        throw(ArgumentError("device must be :auto, :cpu, :cuda, or :metal (got :$device)"))

    if dev === :cpu
        @info "Pulsar.setup_gpu!: no GPU selected — configuring CPU backend."
        _store_gpu_prefs!(:cpu, false)
        set_default && set_device!(:cpu)
        return :cpu
    end

    pkg = _gpu_pkg_name(dev)
    if install && !_gpu_pkg_loadable(pkg)
        blocked, why = force ? (false, "") : _install_blocked()
        if blocked
            @warn "Pulsar.setup_gpu!: not installing $pkg ($why). " *
                  "Add it manually with `] add $pkg`, or pass force=true."
        else
            @info "Pulsar.setup_gpu!: installing $pkg into the active environment…"
            try
                _load_pkg().add(pkg)
            catch e
                @warn "Pulsar.setup_gpu!: `Pkg.add(\"$pkg\")` failed ($e). " *
                      "Install it manually, then re-run setup_gpu!()."
            end
        end
    end

    _store_gpu_prefs!(dev, auto_load)

    if set_default && _gpu_pkg_loadable(pkg)
        loaded = load_gpu!(; device = dev, quiet = true)
        if loaded === dev
            @info "Pulsar.setup_gpu!: $dev is ready and is now the default device. " *
                  "Future `using Pulsar` will load it automatically."
        else
            @info "Pulsar.setup_gpu!: preference saved ($dev). It will activate on the " *
                  "next `using Pulsar` (restart your session)."
        end
    else
        @info "Pulsar.setup_gpu!: preference saved ($dev). Run `using Pulsar` in a fresh " *
              "session to activate it."
    end
    return dev
end

"""
    load_gpu!(; device=<preference>, quiet=false) -> Symbol

Import the installed GPU package for `device` (activating its Pulsar extension)
and make it the active device.  Falls back to `:cpu` with a warning if the
package is not installed or the GPU is not functional.  Returns the device that
ended up active.
"""
function load_gpu!(; device::Symbol = _preferred_device(), quiet::Bool = false)::Symbol
    if device === :cpu
        set_device!(:cpu)
        return :cpu
    end
    pkg = _gpu_pkg_name(device)
    id = Base.identify_package(pkg)
    if id === nothing
        quiet || @warn "Pulsar.load_gpu!: $pkg is not installed. " *
                       "Run `Pulsar.setup_gpu!()`. Staying on CPU."
        return :cpu
    end
    try
        Base.require(id)   # loads the package → triggers the Pulsar extension
    catch e
        quiet || @warn "Pulsar.load_gpu!: failed to load $pkg ($e). Staying on CPU."
        return :cpu
    end
    functional = device === :cuda ? is_cuda_available() : is_metal_available()
    if functional
        set_device!(device)
        return device
    else
        # The package loaded but its Pulsar extension has not run its __init__
        # yet — this happens when `load_gpu!` is called from *inside* Pulsar's
        # own __init__ (the extension activates only afterwards).  Record the
        # preferred device as the global default directly; it takes effect the
        # moment the extension flips the availability flag.  (Downstream code
        # still falls back to CPU if the GPU turns out non-functional.)
        _DEVICE_DEFAULT[] = device
        return device
    end
end

"""
    gpu_setup_status() -> NamedTuple

Summary of the current GPU configuration: detected hardware, the saved
preference, whether auto-load is enabled, which GPU packages are installable,
and the currently active device.
"""
function gpu_setup_status()
    return (
        detected      = detect_gpu_hardware(),
        preference    = _preferred_device(),
        auto_load     = _auto_load_pref(),
        cuda_installed = _gpu_pkg_loadable("CUDA"),
        metal_installed = _gpu_pkg_loadable("Metal"),
        cuda_active   = is_cuda_available(),
        metal_active  = is_metal_available(),
        active_device = get_device(),
    )
end

# ---------------------------------------------------------------------------
# __init__ hook
# ---------------------------------------------------------------------------

# Called from Pulsar's __init__ AFTER passive backend detection.  Honours the
# user's saved preference: auto-load the GPU if they ran setup_gpu!, else print
# a one-time discovery hint.  Wrapped so any failure is non-fatal to loading.
function _maybe_autoload_gpu()
    try
        if _auto_load_pref()
            dev = _preferred_device()
            dev in (:cuda, :metal) && load_gpu!(; device = dev, quiet = true)
        elseif !is_cuda_available() && !is_metal_available()
            hw = detect_gpu_hardware()
            if hw !== :cpu
                name = hw === :cuda ? "NVIDIA CUDA" : "Apple Metal"
                @info "Pulsar: $name GPU detected. Run `Pulsar.setup_gpu!()` once to " *
                      "install and enable GPU acceleration." maxlog = 1
            end
        end
    catch e
        @debug "Pulsar: GPU auto-setup skipped ($e)."
    end
    return nothing
end
