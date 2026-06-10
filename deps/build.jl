# deps/build.jl — GPU hardware recognition at install/build time.
#
# Runs on `Pkg.build("Pulsar")`.  It only DETECTS the GPU and prints a one-line
# hint; it never installs anything (installation is the explicit, user-invoked
# `Pulsar.setup_gpu!()`).  Detection is inlined here because Pulsar itself may
# not be loadable yet at build time.  Everything is wrapped so a build never
# fails on account of GPU detection.

function _detect()
    try
        if Sys.isapple() && Sys.ARCH === :aarch64
            return :metal
        end
        if Sys.which("nvidia-smi") !== nothing
            return :cuda
        end
        if Sys.islinux()
            isfile("/proc/driver/nvidia/version") && return :cuda
            isdir("/dev") && any(startswith("nvidia"), readdir("/dev")) && return :cuda
        end
    catch
    end
    return :cpu
end

try
    dev = _detect()
    if dev === :metal
        @info "Pulsar: Apple Metal GPU detected. After loading, run " *
              "`Pulsar.setup_gpu!()` once to enable GPU acceleration."
    elseif dev === :cuda
        @info "Pulsar: NVIDIA GPU detected. After loading, run " *
              "`Pulsar.setup_gpu!()` once to install CUDA.jl and enable GPU acceleration."
    else
        @info "Pulsar: no GPU detected — CPU-only build. (Run `Pulsar.setup_gpu!()` " *
              "later if you add a GPU.)"
    end
catch
    # Detection must never break the build.
end
