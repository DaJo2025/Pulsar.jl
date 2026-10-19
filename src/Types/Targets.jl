# Types/Targets.jl
# QuantumTarget type and constructor functions.
# Extracted from Core/QuantumSystem.jl.

using LinearAlgebra

# ============================================================================
# Target type
# ============================================================================

"""
    QuantumTarget

Specifies the desired outcome of a quantum control experiment.

# Fields
- `type           :: String` — one of `"state"`, `"unitary"`, or `"subspace"`
- `target_state   :: Union{Vector{ComplexF64}, Nothing}` — target state vector |ψ_f⟩;
  non-`nothing` when `type == "state"`
- `target_unitary :: Union{Matrix{ComplexF64}, Nothing}` — target unitary U_target;
  non-`nothing` when `type == "unitary"`
- `initial_state  :: Union{Vector{ComplexF64}, Nothing}` — explicit initial
  state |ψ_init⟩ for `"state"` targets; if `nothing`, downstream kernels fall
  back to using `target_state` as both initial and final (legacy fixed-point
  fidelity, kept for backward compatibility)
- `dim            :: Int` — Hilbert space dimension
- `metric         :: Symbol` — fidelity metric, under its Φ^X_Y name (see
  [`FIDELITY_METRICS`](@ref)); `:auto` selects the per-type default

# Physics
For `type == "unitary"` the default (`:auto` = `:prop_squared`, Φ^P_Q) is

    F = |Tr(U_target† U)|² / dim²

For `type == "state"` the default (`:auto` = `:state_squared`, Φ^S_Q) is

    F = |⟨ψ_target | U | ψ_init⟩|²
"""
# ============================================================================
# Fidelity-metric names — the Φ^X_Y grid
# ============================================================================
#
# Every figure of merit is a choice of WHAT is compared (level X) and HOW the
# complex overlap z is reduced to a real number (reduction Y):
#
#                       Re z  (R)        |z|  (M)          |z|²  (Q)
#   P  propagator   :prop_real       :prop_modulus     :prop_squared     z = Tr(U_t† U)/dim
#   S  state        :state_real      :state_modulus    :state_squared    z = ⟨ψ_t|ψ⟩
#   D  density mat. :dm_real         :dm_modulus       :dm_squared       z = Tr(ρ_t† ρ)
#
# Off-grid: :prop_average   = (dim·Φ^P_Q + 1)/(dim + 1)  (Haar-average gate fidelity)
#           :prop_real_quad = 1 − (1 − Φ^P_R)²            (Kobzar quadratic form)
#           :dm_uhlmann     = (Tr√(√ρ σ √ρ))²             (Uhlmann–Jozsa)
#
# These names are the only ones used inside Pulsar.  Every public entry point
# that takes a metric resolves its argument through `canonical_metric`, which
# also accepts the pre-Φ^X_Y names (`:square`, `:normalized`, `:real_trace`, …)
# with a one-time deprecation warning.  The Symbol → metric-object mapping
# lives in Physics/Objectives.jl; Types must not depend upward.

const PROP_METRICS     = (:prop_real,  :prop_modulus,  :prop_squared)
const STATE_METRICS    = (:state_real, :state_modulus, :state_squared)
const DM_METRICS       = (:dm_real,    :dm_modulus,    :dm_squared)
const FIDELITY_METRICS = (PROP_METRICS..., STATE_METRICS..., DM_METRICS...,
                          :prop_average, :prop_real_quad, :dm_uhlmann)

const _METRIC_LABELS = Dict{Symbol,String}(
    :prop_real      => "Φ^P_R (Prop-Real)",
    :prop_modulus   => "Φ^P_M (Prop-Modulus)",
    :prop_squared   => "Φ^P_Q (Prop-Squared)",
    :state_real     => "Φ^S_R (State-Real)",
    :state_modulus  => "Φ^S_M (State-Modulus)",
    :state_squared  => "Φ^S_Q (State-Squared)",
    :dm_real        => "Φ^D_R (DM-Real)",
    :dm_modulus     => "Φ^D_M (DM-Modulus)",
    :dm_squared     => "Φ^D_Q (DM-Squared)",
    :prop_average   => "Φ^P_avg (Prop-Average)",
    :prop_real_quad => "1 − (1 − Φ^P_R)² (Prop-Real-Quadratic)",
    :dm_uhlmann     => "Φ^D_Uhl (DM-Uhlmann)",
)

"""
    metric_label(metric::Symbol) -> String

Paper symbol and plain name of a fidelity metric, e.g.
`metric_label(:state_squared) == "Φ^S_Q (State-Squared)"`.
"""
metric_label(metric::Symbol)::String = get(_METRIC_LABELS, metric, ":$metric")

# :prop / :state / :dm, or `nothing` for anything off the grid (e.g. :auto).
function _metric_level(metric::Symbol)
    s = String(metric)
    startswith(s, "prop_")  && return :prop
    startswith(s, "state_") && return :state
    startswith(s, "dm_")    && return :dm
    return nothing
end

# Pre-Φ^X_Y names.  The same old symbol meant different metrics on different
# objects (`:square` was State-Squared on MRControl but DM-Squared on
# LindbladMRControl), so the lookup is keyed by level as well.
#
# Only names that actually existed are listed.  Note that the SU(2) kernel's
# old `:squared` is |⟨↑|U_t†U|↑⟩|², the state fidelity of the |↑⟩ probe, not
# Φ^P_Q = (Re Tr(U_t†U)/2)² — hence it maps to :state_squared.
const _LEGACY_METRIC_NAMES = Dict{Tuple{Symbol,Symbol},Symbol}(
    (:prop,  :real)         => :prop_real,       # unitary_target, gate_fidelity, grape_gate_kernel
    (:prop,  :real_trace)   => :prop_real,       # SU2Kernel, SpinIKernel
    (:prop,  :modulus)      => :prop_modulus,    # unitary_target, grape_gate_kernel
    (:prop,  :gate_modulus) => :prop_modulus,    # fidelity_grad_prefactor
    (:prop,  :normalized)   => :prop_squared,    # unitary_target, gate_fidelity
    (:prop,  :square)       => :prop_squared,    # grape_gate_kernel
    (:prop,  :average)      => :prop_average,    # unitary_target, gate_fidelity
    (:prop,  :real_quad)    => :prop_real_quad,  # grape_gate_kernel
    (:state, :real)         => :state_real,      # state_target, MRControl, state_fidelity
    (:state, :modulus)      => :state_modulus,   # state_target, MRControl, state_fidelity
    (:state, :square)       => :state_squared,   # state_target, MRControl, state_fidelity
    (:state, :squared)      => :state_squared,   # SU2Kernel (|↑⟩ probe)
    (:state, :state)        => :state_squared,   # SpinIKernel, TrotterKernel
    (:dm,    :real)         => :dm_real,         # LindbladMRControl, lindblad_grad_prefactor
    (:dm,    :square)       => :dm_squared,      # LindbladMRControl, lindblad_grad_prefactor
    (:dm,    :dm_linear)    => :dm_real,         # state_target, state_fidelity
    (:dm,    :dm_square)    => :dm_squared,      # state_target, state_fidelity
)

_format_metrics(ms) = join((m === :auto ? ":auto" : ":$m" for m in ms), ", ")

"""
    canonical_metric(metric, supported; context = "") -> Symbol

Resolve `metric` to its Φ^X_Y name and check that `supported` (a tuple of
canonical names, optionally including `:auto`) contains it.

A pre-Φ^X_Y name is translated at the level of `supported`: `:square` becomes
`:state_squared` where `supported` holds state metrics and `:dm_squared` where it
holds density-matrix metrics.  If `supported` spans several levels, the first
level listed wins.  Each translation warns once per `context` and old name.

Throws `ArgumentError` for an unknown name, or for a valid name at a level the
caller does not handle (e.g. `:dm_squared` on a state-vector kernel).
"""
function canonical_metric(metric::Symbol, supported::Tuple;
                          context::AbstractString = "")::Symbol
    metric in supported && return metric
    where = isempty(context) ? "" : " for $context"
    if metric in FIDELITY_METRICS
        throw(ArgumentError(
            "Fidelity metric :$metric — $(metric_label(metric)) — is not " *
            "supported$where. Supported: $(_format_metrics(supported))."))
    end
    levels = unique(filter(!isnothing, map(_metric_level, collect(supported))))
    for lvl in levels
        new = get(_LEGACY_METRIC_NAMES, (lvl, metric), nothing)
        new === nothing && continue
        new in supported || throw(ArgumentError(
            "Fidelity metric :$metric means :$new — $(metric_label(new)) — " *
            "here, which is not supported$where. " *
            "Supported: $(_format_metrics(supported))."))
        @warn("Fidelity metric :$metric is deprecated$where; use :$new — " *
              "$(metric_label(new)). The old name still works for now.",
              maxlog = 1, _id = Symbol("pulsar_legacy_metric_", context, "_", metric))
        return new
    end
    throw(ArgumentError(
        "Unknown fidelity metric :$metric$where. " *
        "Supported: $(_format_metrics(supported))."))
end

# Metrics a pure-state `QuantumTarget` accepts: the state overlaps, plus the
# density-matrix metrics, which reduce to an overlap for a pure state.  The
# Uhlmann fidelity (`:dm_uhlmann`) needs an actual density matrix, so it
# belongs to the open-system / Lindblad path, not a pure-state target.
const VALID_STATE_METRICS = (:auto, STATE_METRICS..., DM_METRICS...)
const VALID_GATE_METRICS  = (:auto, PROP_METRICS..., :prop_average)

struct QuantumTarget
    type           :: String
    target_state   :: Union{Vector{ComplexF64}, Nothing}
    target_unitary :: Union{Matrix{ComplexF64}, Nothing}
    initial_state  :: Union{Vector{ComplexF64}, Nothing}
    dim            :: Int
    # Figure-of-merit selector, always stored under its Φ^X_Y name.  `:auto` →
    # the per-type default (`:prop_squared` for gates, `:state_squared` for
    # states); any other symbol picks an explicit fidelity metric (see the
    # metric table above and Physics/Objectives.jl).  Stored as a `Symbol` so
    # the Types layer stays free of any Physics dependency.
    metric         :: Symbol

    # Single inner constructor with `metric` defaulting to `:auto`, so every
    # existing 5-argument positional call keeps working unchanged.  Pre-Φ^X_Y
    # metric names are translated here, so downstream code sees only the
    # canonical ones.
    function QuantumTarget(type::AbstractString,
                           target_state::Union{Vector{ComplexF64}, Nothing},
                           target_unitary::Union{Matrix{ComplexF64}, Nothing},
                           initial_state::Union{Vector{ComplexF64}, Nothing},
                           dim::Integer,
                           metric::Symbol = :auto)
        m = type == "unitary" ?
                canonical_metric(metric, VALID_GATE_METRICS;  context = "a unitary target") :
            type == "state" ?
                canonical_metric(metric, VALID_STATE_METRICS; context = "a state target") :
                metric
        new(String(type), target_state, target_unitary, initial_state,
            Int(dim), m)
    end
end

# ============================================================================
# Target constructors
# ============================================================================

"""
    state_target(state::AbstractVector; psi_init=nothing, metric=:auto) -> QuantumTarget

Construct a `QuantumTarget` representing a desired final state |ψ_target⟩.

# Arguments
- `state`    — target state vector; will be normalized if not already unit norm
- `psi_init` — optional initial state vector |ψ_init⟩ (same dimension as
  `state`). When provided, the fidelity optimised is the proper state-transfer
  overlap `F = |⟨ψ_target | U | ψ_init⟩|²`. When omitted (`nothing`),
  downstream kernels use `state` as both initial and final, optimising for
  controls that hold |ψ_target⟩ as a fixed point.
- `metric`   — `:auto` (= `:state_squared`, Φ^S_Q), `:state_real`,
  `:state_modulus`, or a density-matrix metric `:dm_real`, `:dm_modulus`,
  `:dm_squared` (evaluated on the pure states).  See [`FIDELITY_METRICS`](@ref).

# Returns
A `QuantumTarget` with `type == "state"`.

# Throws
- `ArgumentError` if state is empty, contains NaN/Inf, or `psi_init` has wrong dimension.

# Examples
```julia
# Fixed-point: find U such that ⟨1|U|1⟩ ≈ 1
tgt = state_target(ComplexF64[0, 1])

# State transfer |0⟩ → |1⟩
tgt = state_target(ComplexF64[0, 1]; psi_init = ComplexF64[1, 0])
```
"""
function state_target(state::AbstractVector;
                       psi_init::Union{AbstractVector, Nothing} = nothing,
                       metric::Symbol = :auto)::QuantumTarget
    # Validate up front so a bad metric fails before any state checks.
    metric = canonical_metric(metric, VALID_STATE_METRICS; context = "a state target")
    psi = ComplexF64.(state)
    d = length(psi)
    if d == 0
        throw(ArgumentError("Target state vector must be non-empty"))
    end
    if any(isnan.(psi)) || any(isinf.(psi))
        throw(ArgumentError("Target state contains NaN or Inf values"))
    end
    nrm = norm(psi)
    if nrm < eps(Float64)
        throw(ArgumentError("Target state has zero norm"))
    end

    psi_i::Union{Vector{ComplexF64}, Nothing} = nothing
    if psi_init !== nothing
        psi_i = ComplexF64.(psi_init)
        if length(psi_i) != d
            throw(ArgumentError(
                "psi_init has length $(length(psi_i)); expected $d"))
        end
        if any(isnan.(psi_i)) || any(isinf.(psi_i))
            throw(ArgumentError("psi_init contains NaN or Inf values"))
        end
        ni = norm(psi_i)
        ni < eps(Float64) && throw(ArgumentError("psi_init has zero norm"))
    end

    return QuantumTarget("state", psi, nothing, psi_i, d, metric)
end

"""
    unitary_target(U::AbstractMatrix; metric=:auto) -> QuantumTarget

Construct a `QuantumTarget` representing a desired unitary gate U_target.

# Arguments
- `U` — target unitary matrix of size `dim × dim`; must be square and approximately
  unitary within a tolerance of `1e-10`.
- `metric` — `:auto` (= `:prop_squared`, Φ^P_Q), `:prop_real`, `:prop_modulus`,
  or `:prop_average`.  See [`FIDELITY_METRICS`](@ref).

# Returns
A `QuantumTarget` with `type == "unitary"`.

# Throws
- `ArgumentError` if U is not square or not unitary.

# Example
```julia
# CNOT gate target
U = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0] .+ 0im
tgt = unitary_target(U)
```
"""
function unitary_target(U::AbstractMatrix; metric::Symbol = :auto)::QuantumTarget
    # Validate up front so a bad metric fails before the unitarity check.
    metric = canonical_metric(metric, VALID_GATE_METRICS; context = "a unitary target")
    Uc = ComplexF64.(U)
    m, n = size(Uc)
    if m != n
        throw(ArgumentError("Target unitary must be square, got $m × $n"))
    end
    # Check unitarity: U† U ≈ I
    deviation = norm(Uc' * Uc - I) / m
    if deviation > 1e-8
        throw(ArgumentError(
            "Target matrix is not unitary: ||U†U - I|| / dim = $deviation > 1e-8"))
    end
    return QuantumTarget("unitary", nothing, Uc, nothing, m, metric)
end

# ============================================================================
# Utility: pretty printing
# ============================================================================

function Base.show(io::IO, tgt::QuantumTarget)
    print(io, "QuantumTarget(type=\"$(tgt.type)\", dim=$(tgt.dim))")
end
