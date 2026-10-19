# Types/EnsembleAveraging.jl
#
# Specification of an ensemble average — data only; the evaluation is
# `ensemble_averaging` in Physics/EnsembleAveraging.jl.
#
# An ensemble is a set of members k, each a variant of the nominal problem with
# a weight ω_k, scored as
#
#     F = aggregate over k of  ω_k · f_m(z_k)
#
# (f_m is a Φ^X_Y fidelity metric, z_k the member's overlap). Members are the
# product — or the element-wise zip — of AXES, each a list of sample values
# with weights and a rule for how a value changes the problem, optionally
# combined with an INPUT-STATE axis (probe pairs: which states or operators are
# sent through the propagator and which outputs they should reach).
#
# The same axes describe MR and QC robustness: a frequency offset is a
# resonance offset / B0 inhomogeneity or a qubit detuning; a drive scale is B1
# inhomogeneity or a Rabi miscalibration; a coupling term is a J coupling or a
# ZZ crosstalk; `TermAxis` adds any Hermitian term and `FunctionAxis` any
# change at all.

"""
    AbstractEnsembleAxis

One axis of an [`EnsembleAveraging`](@ref): sample values, their weights, and
how a value changes the problem. Concrete axes: [`TermAxis`](@ref) (and its
[`OffsetAxis`](@ref) / [`CouplingAxis`](@ref) forms), [`DriveScaleAxis`](@ref),
[`DrivePhaseAxis`](@ref), [`TimeScaleAxis`](@ref), [`DriftListAxis`](@ref),
[`FunctionAxis`](@ref).
"""
abstract type AbstractEnsembleAxis end

function _axis_weights(n::Int, weights)
    n > 0 || throw(ArgumentError("an ensemble axis needs at least one value"))
    w = weights === nothing ? fill(1.0 / n, n) : Float64.(collect(weights))
    length(w) == n ||
        throw(ArgumentError("ensemble axis has $n values but $(length(w)) weights"))
    all(isfinite, w) || throw(ArgumentError("ensemble axis weights must be finite"))
    return w
end

"""
    TermAxis(operator, values_hz; weights = nothing, label = :term)

Adds `2π · v · operator` to the drift Hamiltonian for each sample value `v`
(Hz). `operator` must be Hermitian. Weights default to uniform.
"""
struct TermAxis <: AbstractEnsembleAxis
    operator :: Matrix{ComplexF64}
    values   :: Vector{Float64}
    weights  :: Vector{Float64}
    label    :: Symbol
    function TermAxis(operator::AbstractMatrix, values; weights = nothing, label::Symbol = :term)
        K = Matrix{ComplexF64}(operator)
        size(K, 1) == size(K, 2) || throw(ArgumentError("TermAxis operator must be square"))
        K ≈ K' || throw(ArgumentError("TermAxis operator must be Hermitian"))
        v = Float64.(collect(values))
        new(K, v, _axis_weights(length(v), weights), label)
    end
end

"""
    OffsetAxis(values_hz; operator, weights = nothing)

Frequency-offset axis: adds `2π · δ · operator`, where `operator` is the total
z operator (`Σ_i Iz_i` for spins; for a qubit, `σz/2`). This is a resonance
offset / B0 inhomogeneity in MR and a qubit detuning in QC. For spin systems
`offset_operator(sys)` builds the operator.
"""
OffsetAxis(values; operator::AbstractMatrix, weights = nothing) =
    TermAxis(operator, values; weights, label = :offset)

"""
    CouplingAxis(values_hz; operator, weights = nothing)

Coupling-strength axis: adds `2π · ΔJ · operator` (a deviation from the
nominal coupling, in Hz). `operator` is the coupling form, e.g. `I1·I2`
(isotropic J) or `I1z·I2z` (weak coupling, ZZ crosstalk); for spin systems
`coupling_operator(sys, i, j)` builds it.
"""
CouplingAxis(values; operator::AbstractMatrix, weights = nothing) =
    TermAxis(operator, values; weights, label = :coupling)

"""
    DriveScaleAxis(factors; weights = nothing, channels = :all)

Scales the control Hamiltonians of `channels` (all by default) by each factor:
B1 inhomogeneity / miscalibration in MR, Rabi-rate miscalibration in QC.
"""
struct DriveScaleAxis <: AbstractEnsembleAxis
    values   :: Vector{Float64}
    weights  :: Vector{Float64}
    channels :: Union{Symbol, Vector{Int}}
    function DriveScaleAxis(values; weights = nothing, channels = :all)
        v = Float64.(collect(values))
        ch = channels === :all ? :all : Int.(collect(channels))
        new(v, _axis_weights(length(v), weights), ch)
    end
end

"""
    DrivePhaseAxis(errors_rad; weights = nothing, pairs = [(1, 2)])

Rotates the drive of each `(x, y)` control pair by a phase error ε:
`Hx → Hx cos ε + Hy sin ε`, `Hy → Hy cos ε − Hx sin ε` (the field
`u_x Ix + u_y Iy` turned by ε about z): RF phase errors in MR, IQ phase
imbalance in QC.
"""
struct DrivePhaseAxis <: AbstractEnsembleAxis
    values  :: Vector{Float64}
    weights :: Vector{Float64}
    pairs   :: Vector{Tuple{Int,Int}}
    function DrivePhaseAxis(values; weights = nothing, pairs = [(1, 2)])
        v = Float64.(collect(values))
        new(v, _axis_weights(length(v), weights), [Tuple{Int,Int}(p) for p in pairs])
    end
end

"""
    TimeScaleAxis(factors; weights = nothing)

Scales the time step by each factor (clock / timing error).
"""
struct TimeScaleAxis <: AbstractEnsembleAxis
    values  :: Vector{Float64}
    weights :: Vector{Float64}
    function TimeScaleAxis(values; weights = nothing)
        v = Float64.(collect(values))
        all(>(0), v) || throw(ArgumentError("TimeScaleAxis factors must be positive"))
        new(v, _axis_weights(length(v), weights))
    end
end

"""
    DriftListAxis(drifts; weights = nothing)

An explicit list of drift Hamiltonians (rad/s) that REPLACE the nominal drift —
e.g. the `drifts` of an `MRControl`, or one drift per system of a
`grape_optimize_ensemble` call. Apply it before any `TermAxis` that should add
to it.
"""
struct DriftListAxis <: AbstractEnsembleAxis
    values  :: Vector{Matrix{ComplexF64}}
    weights :: Vector{Float64}
    function DriftListAxis(drifts; weights = nothing)
        v = [Matrix{ComplexF64}(H) for H in drifts]
        new(v, _axis_weights(length(v), weights))
    end
end

"""
    FunctionAxis(f, values; weights = nothing, label = :function)

Any other change: `f(H_drift, H_controls, v) -> (H_drift′, H_controls′)` for
each sample value `v` (a number, tuple, or anything `f` understands). Covers
quasi-static noise models, anharmonicity, field maps, ….
"""
struct FunctionAxis <: AbstractEnsembleAxis
    f       :: Function
    values  :: Vector{Any}
    weights :: Vector{Float64}
    label   :: Symbol
    function FunctionAxis(f::Function, values; weights = nothing, label::Symbol = :function)
        v = collect(Any, values)
        new(f, v, _axis_weights(length(v), weights), label)
    end
end

_axis_length(a::AbstractEnsembleAxis) = length(a.values)

# ─── Operators for the offset / coupling axes of spin systems ────────────────
# (MRSpinSystem methods are in Types/NMRSpinSystem.jl.)

"""
    offset_operator(sys; spins = :all) -> Matrix{ComplexF64}

The total z operator Σ_i Iz_i over `spins` — the operator a frequency offset
couples to (pass it to [`OffsetAxis`](@ref)). Methods for `SpinSystem` and
`MRSpinSystem`.
"""
function offset_operator(sys::SpinSystem; spins = :all)
    idx = spins === :all ? eachindex(sys.spins) : spins
    return sum(embed_operator(spin_Sz(sys.spins[i]), sys.spins, i) for i in idx)
end

"""
    coupling_operator(sys, i, j; kind = :isotropic) -> Matrix{ComplexF64}

The operator a coupling between spins `i` and `j` multiplies: `I_i·I_j`
(`kind = :isotropic`, strong J) or `I_iz·I_jz` (`:zz`, weak J / ZZ crosstalk).
Pass it to [`CouplingAxis`](@ref) (values are deviations in Hz). Methods for
`SpinSystem` and `MRSpinSystem`.
"""
function coupling_operator(sys::SpinSystem, i::Int, j::Int; kind::Symbol = :isotropic)
    e(op, k) = embed_operator(op(sys.spins[k]), sys.spins, k)
    kind === :zz && return e(spin_Sz, i) * e(spin_Sz, j)
    kind === :isotropic || throw(ArgumentError("kind must be :isotropic or :zz"))
    return sum(e(op, i) * e(op, j) for op in (spin_Sx, spin_Sy, spin_Sz))
end

# `op` (d×d) on site `i` of `n` identical d-level sites (site 1 = leftmost
# kron factor, the ordering every QC platform builder uses).
_embed_site(op::AbstractMatrix, i::Int, n::Int, d::Int) =
    foldl(kron, (k == i ? Matrix{ComplexF64}(op) : Matrix{ComplexF64}(I, d, d) for k in 1:n))

# ─── Input-state axis ─────────────────────────────────────────────────────────

"""
    AbstractProbeSet

The input-state axis of an [`EnsembleAveraging`](@ref): a weighted set of
(input, target) pairs sent through each member's propagator.
[`StateProbes`](@ref) holds state vectors (state-row and, on the pure states,
DM-row metrics); [`OperatorProbes`](@ref) holds density matrices or operators
(DM-row metrics). Build them from a target unitary with `state_probes` /
`operator_probes`.
"""
abstract type AbstractProbeSet end

"""
    StateProbes(inputs, targets; weights = nothing)

Pairs `ψ_in → ψ_target` (normalized). The per-member fidelity is the weighted
mean over pairs of the metric of `z = ⟨ψ_target|U|ψ_in⟩`.
"""
struct StateProbes <: AbstractProbeSet
    inputs  :: Vector{Vector{ComplexF64}}
    targets :: Vector{Vector{ComplexF64}}
    weights :: Vector{Float64}
    function StateProbes(inputs, targets; weights = nothing)
        length(inputs) == length(targets) ||
            throw(ArgumentError("StateProbes: $(length(inputs)) inputs but $(length(targets)) targets"))
        ins = [normalize(Vector{ComplexF64}(ψ)) for ψ in inputs]
        tgs = [normalize(Vector{ComplexF64}(ψ)) for ψ in targets]
        all(length(ψ) == length(ins[1]) for ψ in vcat(ins, tgs)) ||
            throw(ArgumentError("StateProbes: all states must have the same dimension"))
        new(ins, tgs, _axis_weights(length(ins), weights))
    end
end

"""
    OperatorProbes(inputs, targets; weights = nothing)

Pairs `A → B` of density matrices or operators. The per-member fidelity is the
weighted mean over pairs of the DM-row metric of the Hilbert–Schmidt overlap
`z = Tr(B† U A U†) / (‖A‖ ‖B‖)` — real for Hermitian pairs, and signed for
traceless operators (operator transfer).
"""
struct OperatorProbes <: AbstractProbeSet
    inputs  :: Vector{Matrix{ComplexF64}}
    targets :: Vector{Matrix{ComplexF64}}
    weights :: Vector{Float64}
    function OperatorProbes(inputs, targets; weights = nothing)
        length(inputs) == length(targets) ||
            throw(ArgumentError("OperatorProbes: $(length(inputs)) inputs but $(length(targets)) targets"))
        ins = [Matrix{ComplexF64}(A) for A in inputs]
        tgs = [Matrix{ComplexF64}(B) for B in targets]
        all(norm(A) > 0 for A in vcat(ins, tgs)) ||
            throw(ArgumentError("OperatorProbes: operators must be non-zero"))
        new(ins, tgs, _axis_weights(length(ins), weights))
    end
end

# ─── The ensemble ─────────────────────────────────────────────────────────────

const ENSEMBLE_AGGREGATES = (:mean, :sum, :worst_case, :cvar, :softmin)

"""
    EnsembleAveraging(axes...; probes = nothing, combine = :product,
                      aggregate = :mean, metric = :auto,
                      cvar_alpha = 0.2, softmin_beta = 50.0)

An ensemble average over `axes` (see [`AbstractEnsembleAxis`](@ref)) and an
optional input-state axis `probes` (an [`AbstractProbeSet`](@ref)). Evaluate it
with `ensemble_averaging(ens, system, controls, target)`.

- `combine = :product` — every combination of axis values; member weight =
  product of the axis weights. `:zip` — the i-th values of all axes together
  (axes of equal length, e.g. joint random samples).
- `aggregate` — how member fidelities F_k with weights ω_k become one number:
  `:mean` Σω_k F_k / Σω_k (default); `:sum` Σω_k F_k with the weights as given
  (signed weights: band-selective pass / stop bands); `:worst_case` min F_k;
  `:cvar` the weighted mean of the worst `cvar_alpha` fraction of weight;
  `:softmin` −log(Σ ω̂_k e^{−β F_k})/β, a smooth worst case (β = `softmin_beta`).
- `metric` — the Φ^X_Y fidelity of one member (see `FIDELITY_METRICS`).
  `:auto` takes the target's metric, or `:state_squared` for `StateProbes` and
  `:dm_real` for `OperatorProbes`.
- Within a member the probe pairs are averaged with their own weights.

Front ends: `grape_optimize(system, target, controls; ensemble)` (or
`grape_optimize(system, ens, controls)` when the probes are the targets),
`ensemble_objective` for any optimiser, and — for the uniform means their
kernels compute — `MRControl(…; ensemble)`, `SU2Kernel(ens, …)`,
`SpinIKernel(ens, …)`.
"""
struct EnsembleAveraging
    axes         :: Vector{AbstractEnsembleAxis}
    probes       :: Union{Nothing, AbstractProbeSet}
    combine      :: Symbol
    aggregate    :: Symbol
    metric       :: Symbol
    cvar_alpha   :: Float64
    softmin_beta :: Float64
    function EnsembleAveraging(axes::AbstractEnsembleAxis...;
                               probes::Union{Nothing, AbstractProbeSet} = nothing,
                               combine::Symbol = :product, aggregate::Symbol = :mean,
                               metric::Symbol = :auto,
                               cvar_alpha::Real = 0.2, softmin_beta::Real = 50.0)
        combine in (:product, :zip) ||
            throw(ArgumentError("combine must be :product or :zip (got :$combine)"))
        aggregate in ENSEMBLE_AGGREGATES ||
            throw(ArgumentError("aggregate must be one of $(ENSEMBLE_AGGREGATES) (got :$aggregate)"))
        0 < cvar_alpha <= 1 || throw(ArgumentError("cvar_alpha must be in (0, 1]"))
        softmin_beta > 0 || throw(ArgumentError("softmin_beta must be positive"))
        m = canonical_metric(metric, (:auto, FIDELITY_METRICS...); context = "EnsembleAveraging")
        ax = AbstractEnsembleAxis[axes...]
        if combine === :zip && !isempty(ax)
            n = _axis_length(ax[1])
            all(_axis_length(a) == n for a in ax) ||
                throw(ArgumentError("combine = :zip needs axes of equal length"))
        end
        new(ax, probes, combine, aggregate, m, Float64(cvar_alpha), Float64(softmin_beta))
    end
end

"""
    ensemble_members(ens) -> Vector{NamedTuple{(:index, :weight)}}

The members of `ens`: for each, the value index into every axis and the
member weight (product of the axis weights; not normalized). An ensemble with
no axes has one member — the nominal problem — of weight 1.
"""
function ensemble_members(ens::EnsembleAveraging)
    ax = ens.axes
    isempty(ax) && return [(index = (), weight = 1.0)]
    if ens.combine === :zip
        return [(index = ntuple(_ -> i, length(ax)), weight = prod(a.weights[i] for a in ax))
                for i in 1:_axis_length(ax[1])]
    end
    idx = Iterators.product((1:_axis_length(a) for a in ax)...)
    return [(index = Tuple(t), weight = prod(ax[j].weights[t[j]] for j in eachindex(ax)))
            for t in vec(collect(idx))]
end

Base.length(ens::EnsembleAveraging) = length(ensemble_members(ens))

function Base.show(io::IO, ens::EnsembleAveraging)
    print(io, "EnsembleAveraging(", length(ens), " members over ",
          isempty(ens.axes) ? "no axes" : join((nameof(typeof(a)) for a in ens.axes), " × "),
          ens.probes === nothing ? "" : " × $(length(ens.probes.inputs)) probe pairs",
          "; aggregate = :", ens.aggregate,
          ens.metric === :auto ? "" : ", metric = :$(ens.metric)", ")")
end

# ─── Front-end check ──────────────────────────────────────────────────────────
#
# Kernels with a built-in ensemble (MRControl: drifts × pwr_levels × state
# pairs; SU2 / spin-I: offsets × amp_factors) compute an UNWEIGHTED mean over a
# full product. An EnsembleAveraging maps onto them only if it is one; anything
# else is refused rather than approximated.

_uniform_weights(w) = all(x -> isapprox(x, w[1]; rtol = 1e-12), w)

function _require_uniform_mean(ens::EnsembleAveraging, allowed::Tuple, context::AbstractString)
    hint = ". Evaluate other ensembles with ensemble_averaging, or optimise them with " *
           "grape_optimize(system, target, controls; ensemble) / ensemble_objective."
    ens.aggregate === :mean || throw(ArgumentError(
        "$context averages uniformly: aggregate must be :mean (got :$(ens.aggregate))$hint"))
    (ens.combine === :product || length(ens.axes) <= 1) || throw(ArgumentError(
        "$context averages over a full product of its axes: combine must be :product$hint"))
    for a in ens.axes
        any(T -> a isa T, allowed) || throw(ArgumentError(
            "$context cannot represent a $(nameof(typeof(a))) " *
            "(it takes $(join(map(nameof, allowed), ", ")))$hint"))
        _uniform_weights(a.weights) || throw(ArgumentError(
            "$context averages uniformly, but a $(nameof(typeof(a))) has non-uniform weights$hint"))
    end
    ens.probes === nothing || _uniform_weights(ens.probes.weights) || throw(ArgumentError(
        "$context averages its state pairs uniformly, but the probe weights are not uniform$hint"))
    return nothing
end

# The metric a front end uses: the ensemble's unless it is :auto; a different
# explicit `metric` argument is a conflict, not an override.
function _front_end_metric(ens::EnsembleAveraging, metric, default::Symbol, context)
    explicit = !(metric === nothing || metric === :auto)
    ens.metric === :auto && return explicit ? metric : default
    (!explicit || metric === ens.metric) || throw(ArgumentError(
        "$context: the ensemble's metric :$(ens.metric) conflicts with metric = :$metric"))
    return ens.metric
end

# Offsets (Hz) and drive-scale factors of an offsets × B1 ensemble whose offset
# terms are real multiples of `Iz` — the ensemble the SU(2) and spin-I kernels
# average natively. `scales === nothing` when there is no DriveScaleAxis.
function _offset_scale_lists(ens::EnsembleAveraging, Iz::AbstractMatrix, context::AbstractString)
    _require_uniform_mean(ens, (TermAxis, DriveScaleAxis), context)
    ens.probes === nothing || throw(ArgumentError(
        "$context has its own input state / target; probe pairs are not supported " *
        "(use ensemble_averaging on a QuantumSystem)"))
    offsets = [0.0]
    scales  = nothing
    for a in ens.axes
        if a isa TermAxis
            size(a.operator) == size(Iz) || throw(DimensionMismatch(
                "$context: offset operator $(size(a.operator)) ≠ spin dimension $(size(Iz))"))
            c = real(a.operator[1, 1]) / real(Iz[1, 1])
            isapprox(a.operator, c .* Iz; atol = 1e-12 * max(1, abs(c))) || throw(ArgumentError(
                "$context: an offset axis must couple to Iz (operator = c·Iz)"))
            offsets = [o + c * v for v in a.values for o in offsets]
        else
            a.channels === :all || throw(ArgumentError(
                "$context scales the whole RF field: DriveScaleAxis channels must be :all"))
            scales = scales === nothing ? copy(a.values) : [s * v for v in a.values for s in scales]
        end
    end
    return offsets, scales
end
