"""
    comparisons/Translator/TransmonAnnotation.jl

Solver-agnostic description of a superconducting-qubit (transmon) problem.
Consumed by `QuandaryEmitter` (emits a `standardmodel=True` Python script)
and reconstructed as an `MRControl` for the Pulsar driver.

A `TransmonAnnotation` carries the same four elements Quandary's native
interface expects:
  * `freq01_hz`   — per-qubit 01 transition frequency (Hz)
  * `anharm_hz`   — per-qubit anharmonicity (Hz, negative for transmon)
  * `coupling_hz` — pairwise cross-Kerr / exchange coupling (Hz) — upper
    triangle of an `n_q × n_q` matrix
  * target: either `:state_transfer` with `initial_state` / `target_state`
    on the essential computational subspace, or `:gate` with
    `target_unitary` on the same subspace

Guard levels (one extra level per qubit in the emitted model) are standard
in Quandary's leakage-aware optimisation.  Pulsar's `MRControl` rebuild
uses `n_levels = essential + guard` per qudit to match.

See BM07 (single-transmon X gate) and BM08 (2-transmon CNOT) for canonical
uses.
"""

using LinearAlgebra

"""
    TransmonAnnotation(; freq01_hz, anharm_hz, coupling_hz, n_essential=2,
                        n_guard=1, omega_max_hz, target_kind, initial_state,
                        target_state, target_unitary, total_time_s,
                        n_time_steps, dt_s, method, max_iter, guess_seed)

Transmon-style QC benchmark annotation.  `freq01_hz` and `anharm_hz` must
have the same length `n_q`; `coupling_hz` is an `n_q × n_q` matrix (only
the upper triangle is used).  `target_kind` is `:state_transfer` or
`:gate`.  `initial_state` / `target_state` live on the essential subspace
(dimension `n_essential^n_q`); `target_unitary` is `2^n_q × 2^n_q`.
"""
Base.@kwdef struct TransmonAnnotation <: AbstractPhysicsAnnotation
    freq01_hz      :: Vector{Float64}
    anharm_hz      :: Vector{Float64}
    coupling_hz    :: Matrix{Float64}                = zeros(length(freq01_hz), length(freq01_hz))
    n_essential    :: Int                             = 2
    n_guard        :: Int                             = 1
    omega_max_hz   :: Float64
    target_kind    :: Symbol                          = :state_transfer
    initial_state  :: Vector{ComplexF64}              = ComplexF64[]
    target_state   :: Vector{ComplexF64}              = ComplexF64[]
    target_unitary :: Union{Nothing,Matrix{ComplexF64}} = nothing
    total_time_s   :: Float64
    n_time_steps   :: Int
    dt_s           :: Float64
    method         :: Symbol                          = :lbfgs
    max_iter       :: Int                             = 500
    guess_seed     :: Int                             = 42
end

n_qubits(ann::TransmonAnnotation) = length(ann.freq01_hz)
n_levels(ann::TransmonAnnotation) = ann.n_essential + ann.n_guard
total_dim(ann::TransmonAnnotation) = n_levels(ann)^n_qubits(ann)

"""
    build_transmon_drift(ann) -> Matrix{ComplexF64}

Construct the rotating-frame drift Hamiltonian at the carrier = mean(freq01):

    H = Σ_i [Δω_i n̂_i + (α_i/2) n̂_i(n̂_i − 1)]
      + Σ_{i<j} ξ_ij n̂_i n̂_j

where the cross-Kerr convention matches Quandary's standardmodel (`crosskerr`).
Exchange (Jaynes–Cummings) `g_ij (a_i† a_j + a_i a_j†)` is intentionally
omitted here to keep parity with the default `Jkl = 0` Quandary path;
coupling is passed through as cross-Kerr.
"""
function build_transmon_drift(ann::TransmonAnnotation)::Matrix{ComplexF64}
    nq  = n_qubits(ann)
    nl  = n_levels(ann)
    dim = nl^nq

    a_one(nl) = begin
        a = zeros(ComplexF64, nl, nl)
        for k in 1:nl-1; a[k, k+1] = sqrt(Float64(k)); end
        a
    end
    n_one(nl) = diagm(0 => Float64.(0:nl-1) .+ 0im)
    Id(n)     = Matrix{ComplexF64}(I, n, n)

    function embed(op, q)
        mats = [k == q ? op : Id(nl) for k in 1:nq]
        foldl(kron, mats)
    end

    ns = [embed(n_one(nl), q) for q in 1:nq]

    ω_c = 2π * (sum(ann.freq01_hz) / nq)
    H   = zeros(ComplexF64, dim, dim)

    for q in 1:nq
        Δω = 2π * ann.freq01_hz[q] - ω_c
        α  = 2π * ann.anharm_hz[q]
        H .+= Δω .* ns[q] .+ (α / 2) .* (ns[q] * ns[q] .- ns[q])
    end

    for i in 1:nq, j in i+1:nq
        ξ = 2π * ann.coupling_hz[i, j]
        iszero(ξ) && continue
        H .+= ξ .* (ns[i] * ns[j])
    end

    return H
end

"""
    build_transmon_controls(ann) -> Vector{Matrix{ComplexF64}}

Per-qubit `(a+a†)/2`, `i(a†−a)/2` operators embedded into the full
Hilbert space.  Two controls per qubit, in the order
`[Ix_q1, Iy_q1, Ix_q2, Iy_q2, …]` — matches Pulsar's `TransmonSystem` and
Quandary's native `Hc_re` / `Hc_im` pair.
"""
function build_transmon_controls(ann::TransmonAnnotation)::Vector{Matrix{ComplexF64}}
    nq  = n_qubits(ann)
    nl  = n_levels(ann)
    Id(n) = Matrix{ComplexF64}(I, n, n)
    a_one(nl) = begin
        a = zeros(ComplexF64, nl, nl)
        for k in 1:nl-1; a[k, k+1] = sqrt(Float64(k)); end
        a
    end
    function embed(op, q)
        mats = [k == q ? op : Id(nl) for k in 1:nq]
        foldl(kron, mats)
    end
    ops = Matrix{ComplexF64}[]
    for q in 1:nq
        a = embed(a_one(nl), q)
        push!(ops, (a .+ a') ./ 2)          # x quadrature
        push!(ops, (1im .* (a' .- a)) ./ 2) # y quadrature
    end
    return ops
end

"""
    lift_essential_to_full(v_ess, ann) -> Vector{ComplexF64}

Pad an essential-subspace state vector (length `n_essential^n_q`) into the
full Hilbert space (length `n_levels^n_q`) by placing it on the indices
corresponding to qudit digits `< n_essential`.  Other (guard) indices are
zero.
"""
function lift_essential_to_full(v_ess::AbstractVector{<:Number},
                                  ann::TransmonAnnotation)::Vector{ComplexF64}
    nq   = n_qubits(ann)
    nl   = n_levels(ann)
    ne   = ann.n_essential
    length(v_ess) == ne^nq ||
        throw(ArgumentError("essential state length $(length(v_ess)) ≠ $(ne^nq)"))
    v_full = zeros(ComplexF64, nl^nq)
    for idx_ess in 0:ne^nq-1
        digits_ess = reverse(digits(idx_ess; base=ne, pad=nq))
        idx_full = 0
        for d in digits_ess
            idx_full = idx_full * nl + d
        end
        v_full[idx_full + 1] = v_ess[idx_ess + 1]
    end
    return v_full
end

"""
    build_ctrl_from_transmon(ann; kwargs...) -> MRControl

Reconstruct the Pulsar matrix form for a `TransmonAnnotation`: a single-drift,
single-state-pair `MRControl` with the transmon rotating-frame drift,
`2·n_q` real control operators (x/y per qubit), and shared `pwr_levels =
[2π · omega_max_hz]`.  Used by the Pulsar driver so the benchmark runs
without any changes to `pulsar_driver.jl`.

For `target_kind = :gate`, the MRControl encodes one state pair per column
of the target unitary (Reich/Goerz/Koch convention), so Pulsar optimises
the gate via the full basis-state transfer set.
"""
function build_ctrl_from_transmon(ann::TransmonAnnotation; kwargs...)
    drift = build_transmon_drift(ann)
    ops   = build_transmon_controls(ann)

    rho_init, rho_targ = if ann.target_kind === :state_transfer
        ψ0 = lift_essential_to_full(ann.initial_state, ann)
        ψT = lift_essential_to_full(ann.target_state,  ann)
        ([ψ0], [ψT])
    elseif ann.target_kind === :gate
        ann.target_unitary === nothing &&
            throw(ArgumentError("target_kind = :gate requires target_unitary"))
        nq   = n_qubits(ann)
        ne   = ann.n_essential
        d_ess = ne^nq
        size(ann.target_unitary) == (d_ess, d_ess) ||
            throw(ArgumentError("target_unitary size $(size(ann.target_unitary)) ≠ ($d_ess,$d_ess)"))
        inits = Vector{ComplexF64}[]
        targs = Vector{ComplexF64}[]
        for k in 1:d_ess
            e_k            = zeros(ComplexF64, d_ess); e_k[k] = 1
            push!(inits, lift_essential_to_full(e_k, ann))
            push!(targs, lift_essential_to_full(ann.target_unitary[:, k], ann))
        end
        (inits, targs)
    else
        throw(ArgumentError("target_kind must be :state_transfer or :gate (got :$(ann.target_kind))"))
    end

    return MRControl(
        drifts     = [drift],
        operators  = ops,
        rho_init   = rho_init,
        rho_targ   = rho_targ,
        pwr_levels = [2π * ann.omega_max_hz],
        pulse_dt   = fill(ann.dt_s, ann.n_time_steps),
        method     = ann.method,
        max_iter   = ann.max_iter,
        fidelity   = :state_squared,
        verbose    = false;
        kwargs...,
    )
end
