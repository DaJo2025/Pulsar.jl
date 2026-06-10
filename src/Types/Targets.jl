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

# Physics
For `type == "unitary"` the figure of merit optimized is the gate fidelity

    F = |Tr(U_target† U)| / dim

For `type == "state"` the fidelity is the state-overlap fidelity

    F = |⟨ψ_target | U | ψ_init⟩|²
"""
# Layer-1-safe sets of valid fidelity-metric symbols (the Symbol → metric-object
# mapping itself lives in Physics/Objectives.jl; Types must not depend upward).
# Pure-state (vector) targets: the overlap-based metrics, plus the density-matrix
# metrics that reduce to an overlap for a pure state.  The Uhlmann fidelity
# (`:dm_uhlmann`) needs an actual density matrix, so it belongs to the
# open-system / Lindblad path, not a pure-state `QuantumTarget`.
const VALID_STATE_METRICS = (:auto, :real, :square, :modulus,
                             :dm_linear, :dm_real, :dm_square, :dm_modulus)
const VALID_GATE_METRICS  = (:auto, :normalized, :real, :average, :modulus)

struct QuantumTarget
    type           :: String
    target_state   :: Union{Vector{ComplexF64}, Nothing}
    target_unitary :: Union{Matrix{ComplexF64}, Nothing}
    initial_state  :: Union{Vector{ComplexF64}, Nothing}
    dim            :: Int
    # Figure-of-merit selector.  `:auto` → the per-type default (`:normalized`
    # for gates, `:square` for states); any other symbol picks an explicit
    # fidelity metric (see Physics/Objectives.jl).  Stored as a `Symbol` so the
    # Types layer stays free of any Physics dependency.
    metric         :: Symbol

    # Single inner constructor with `metric` defaulting to `:auto`, so every
    # existing 5-argument positional call keeps working unchanged.
    function QuantumTarget(type::AbstractString,
                           target_state::Union{Vector{ComplexF64}, Nothing},
                           target_unitary::Union{Matrix{ComplexF64}, Nothing},
                           initial_state::Union{Vector{ComplexF64}, Nothing},
                           dim::Integer,
                           metric::Symbol = :auto)
        new(String(type), target_state, target_unitary, initial_state,
            Int(dim), metric)
    end
end

# ============================================================================
# Target constructors
# ============================================================================

"""
    state_target(state::AbstractVector; psi_init=nothing) -> QuantumTarget

Construct a `QuantumTarget` representing a desired final state |ψ_target⟩.

# Arguments
- `state`    — target state vector; will be normalized if not already unit norm
- `psi_init` — optional initial state vector |ψ_init⟩ (same dimension as
  `state`). When provided, the fidelity optimised is the proper state-transfer
  overlap `F = |⟨ψ_target | U | ψ_init⟩|²`. When omitted (`nothing`),
  downstream kernels use `state` as both initial and final, optimising for
  controls that hold |ψ_target⟩ as a fixed point.

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
    metric in VALID_STATE_METRICS || throw(ArgumentError(
        "metric :$metric is not a valid state fidelity metric. " *
        "Valid: $(VALID_STATE_METRICS)"))
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
    unitary_target(U::AbstractMatrix) -> QuantumTarget

Construct a `QuantumTarget` representing a desired unitary gate U_target.

# Arguments
- `U` — target unitary matrix of size `dim × dim`; must be square and approximately
  unitary within a tolerance of `1e-10`.

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
    metric in VALID_GATE_METRICS || throw(ArgumentError(
        "metric :$metric is not a valid gate fidelity metric. " *
        "Valid: $(VALID_GATE_METRICS)"))
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
