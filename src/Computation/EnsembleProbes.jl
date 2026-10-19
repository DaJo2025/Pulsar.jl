# Computation/EnsembleProbes.jl
#
# Builders for the input-state axis of an EnsembleAveraging: (input, target)
# pairs generated from a target unitary U.
#
# Which input set reproduces which propagator-level number (verified to
# machine precision, test/Physics/EnsembleAveragingTests.jl):
#   State-Squared averaged over a 2-design            = ⟨Φ^S_Q⟩_Haar = (d·Φ^P_Q + 1)/(d + 1)
#     (qubit: the three axis states, or all six; prime d: d + 1 mutually
#      unbiased bases)
#   State-Real averaged over any orthonormal basis     = Φ^P_R
#   DM-Real averaged over a complete traceless operator basis
#     (Paulis, generalized Gell-Mann)                  = (d²·Φ^P_Q − 1)/(d² − 1)
#   The computational basis alone is NOT enough for State-Squared: it is blind
#   to relative phases (a Z error scores 1, against 1/3 for the gate).

using Random

"""
    state_probes(U; set = :axes, n = 0, seed = 1, weights = nothing) -> StateProbes

Input states ψ_j and targets `U·ψ_j` for a target unitary `U`.

- `:axes` — qubit only: |+z⟩, |+x⟩, |+y⟩. Averaging State-Squared over them
  equals the average gate fidelity ⟨Φ^S_Q⟩_Haar exactly (for any U(2)
  propagator); averaging State-Real gives Φ^P_R.
- `:axes6` — qubit: all six axis states (the octahedron, a 3-design).
- `:mub` — a complete set of d + 1 mutually unbiased bases (d(d+1) states, a
  2-design) for prime `d` (the qubit case is `:axes6`).
- `:basis` — the computational basis; blind to relative phases (warns).
- `:random` — `n` Haar-random states (seeded): an unbiased estimate for any d.
"""
function state_probes(U::AbstractMatrix; set::Symbol = :axes, n::Integer = 0,
                      seed::Integer = 1, weights = nothing)
    Uc = Matrix{ComplexF64}(U)
    d = size(Uc, 1)
    size(Uc, 2) == d || throw(ArgumentError("state_probes: U must be square"))
    ins = _probe_inputs(set, d, n, seed)
    return StateProbes(ins, [Uc * ψ for ψ in ins]; weights)
end

function _probe_inputs(set::Symbol, d::Int, n::Integer, seed::Integer)
    s2 = 1 / sqrt(2)
    if set === :axes || set === :axes6
        d == 2 || throw(ArgumentError("state_probes set = :$set is for a qubit (d = 2); got d = $d — use :mub or :random"))
        three = [ComplexF64[1, 0], ComplexF64[s2, s2], ComplexF64[s2, im * s2]]
        set === :axes && return three
        return vcat(three, [ComplexF64[0, 1], ComplexF64[s2, -s2], ComplexF64[s2, -im * s2]])
    elseif set === :mub
        d == 2 && return _probe_inputs(:axes6, 2, n, seed)
        _isprime(d) || throw(ArgumentError(
            "state_probes set = :mub is implemented for prime d (got d = $d); use :random, " *
            "or the closed-form metric :state_squared_haar"))
        ω = cis(2π / d)
        states = [(v = zeros(ComplexF64, d); v[i] = 1; v) for i in 1:d]
        for b in 0:d-1, k in 0:d-1
            push!(states, [ω^mod(b * j^2 + k * j, d) for j in 0:d-1] ./ sqrt(d))
        end
        return states
    elseif set === :basis
        @warn("state_probes(set = :basis): the computational basis is blind to relative " *
              "phases — a pure phase error scores 1. Use :axes / :mub for a gate.", maxlog = 1)
        return [(v = zeros(ComplexF64, d); v[i] = 1; v) for i in 1:d]
    elseif set === :random
        n > 0 || throw(ArgumentError("state_probes set = :random needs n > 0"))
        rng = MersenneTwister(seed)
        return [normalize(randn(rng, ComplexF64, d)) for _ in 1:n]
    end
    throw(ArgumentError("state_probes: unknown set :$set (use :axes, :axes6, :mub, :basis, :random)"))
end

_isprime(n::Int) = n >= 2 && all(n % k != 0 for k in 2:isqrt(n))

"""
    operator_probes(U; set = :pauli, weights = nothing) -> OperatorProbes
    operator_probes(U, inputs; weights = nothing) -> OperatorProbes

Input operators A_j and targets `U·A_j·U†`.

- `:pauli` — d = 2ⁿ: all n-qubit Pauli strings except the identity (for one
  spin: Ix, Iy, Iz — the product-operator probes of the paper's "rot" form).
- `:gell_mann` — the d² − 1 generalized Gell-Mann matrices, any d.

Both are complete traceless operator bases: averaging DM-Real over them gives
(d²·Φ^P_Q − 1)/(d² − 1).
"""
function operator_probes(U::AbstractMatrix; set::Symbol = :pauli, weights = nothing)
    d = size(U, 1)
    ops = set === :pauli     ? _probe_pauli_basis(d) :
          set === :gell_mann ? _probe_gell_mann_basis(d) :
          throw(ArgumentError("operator_probes: unknown set :$set (use :pauli or :gell_mann)"))
    return operator_probes(U, ops; weights)
end

function operator_probes(U::AbstractMatrix, inputs::AbstractVector; weights = nothing)
    Uc = Matrix{ComplexF64}(U)
    ins = [Matrix{ComplexF64}(A) for A in inputs]
    return OperatorProbes(ins, [Uc * A * Uc' for A in ins]; weights)
end

function _probe_pauli_basis(d::Int)
    nq = round(Int, log2(d))
    2^nq == d || throw(ArgumentError("operator_probes set = :pauli needs d = 2ⁿ (got d = $d); use :gell_mann"))
    σ = (ComplexF64[1 0; 0 1], ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1])
    ops = Matrix{ComplexF64}[]
    for idx in Iterators.product(ntuple(_ -> 1:4, nq)...)
        all(==(1), idx) && continue                      # skip the identity
        push!(ops, reduce(kron, (σ[i] for i in idx)) ./ 2)
    end
    return ops
end

function _probe_gell_mann_basis(d::Int)
    E(i, j) = (M = zeros(ComplexF64, d, d); M[i, j] = 1; M)
    ops = Matrix{ComplexF64}[]
    for i in 1:d, j in i+1:d
        push!(ops, E(i, j) + E(j, i))
        push!(ops, -im * E(i, j) + im * E(j, i))
    end
    for l in 1:d-1
        D = zeros(ComplexF64, d, d)
        for i in 1:l; D[i, i] = 1; end
        D[l+1, l+1] = -l
        push!(ops, D .* sqrt(2 / (l * (l + 1))))
    end
    return ops
end
