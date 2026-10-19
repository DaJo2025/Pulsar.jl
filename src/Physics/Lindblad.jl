"""
    Core/LindbladSystem.jl

Liouville-space representation of open quantum systems governed by the
Lindblad master equation.

The Lindblad equation for a density matrix ρ (N×N) is:

    dρ/dt = −i[H(t), ρ]  +  Σ_k γ_k ( L_k ρ L_k† − ½{L_k†L_k, ρ} )

Vectorising ρ into vec(ρ) ∈ ℂ^(N²) using the column-stacking convention
    vec(AXB) = (B^T ⊗ A) vec(X)
converts the matrix ODE into a linear vector ODE:

    d/dt vec(ρ) = 𝓛(t) vec(ρ)

where the Liouvillian superoperator 𝓛 (N²×N²) is:

    𝓛 = 𝓛_coherent + 𝓛_dissipative
    𝓛_coherent  = −i(I⊗H − H^T⊗I)          (commutator part)
    𝓛_dissipative = Σ_k γ_k D_k              (Lindblad dissipators)
    D_k = conj(L_k)⊗L_k − ½(I⊗L_k†L_k) − ½(conj(L_k†L_k)⊗I)

The propagator for one piecewise-constant step is exp(𝓛 dt), a general
(non-unitary) N²×N² complex matrix.

Provides:
  vec_rho / mat_rho            — density-matrix ↔ vector conversion
  pure_state_to_vec_rho        — |ψ⟩ → vec(|ψ⟩⟨ψ|)
  build_drift_liouvillian      — coherent + dissipative superoperator
  build_control_liouvillian    — coherent superoperator for one control op

Reference:
  Lindblad, G. (1976). "On the generators of quantum dynamical semigroups."
  Commun. Math. Phys. 48(2), 119–130.
"""

using LinearAlgebra

# ─── Density-matrix / vector conversion ──────────────────────────────────────

"""
    vec_rho(ρ) → Vector{ComplexF64}

Vectorise a density matrix ρ (N×N) into a length-N² column vector using
the column-stacking convention: `vec(ρ)[i + N*(j-1)] = ρ[i, j]`.

In Julia this is just `vec(ρ)`, but using this function makes the intent
explicit and ensures `ComplexF64` output regardless of input element type.

# Example
```julia
ρ = ComplexF64[1 0; 0 0]   # |+z⟩⟨+z|
v = vec_rho(ρ)              # [1+0i, 0+0i, 0+0i, 0+0i]
```
"""
vec_rho(ρ::Matrix{ComplexF64})::Vector{ComplexF64} = vec(ρ)
vec_rho(ρ::AbstractMatrix)::Vector{ComplexF64}      = vec(ComplexF64.(ρ))

"""
    mat_rho(v, N) → Matrix{ComplexF64}

Reshape a length-N² superoperator vector back into an N×N density matrix.
Inverse of [`vec_rho`](@ref).

# Example
```julia
v = [1+0im, 0, 0, 0]
ρ = mat_rho(v, 2)    # ComplexF64[1 0; 0 0]
```
"""
mat_rho(v::AbstractVector, N::Int)::Matrix{ComplexF64} =
    reshape(ComplexF64.(v), N, N)

"""
    pure_state_to_vec_rho(ψ) → Vector{ComplexF64}

Convert a pure-state vector |ψ⟩ (length N) to `vec(|ψ⟩⟨ψ|)` (length N²).

Used internally by `LindbladMRControl` to convert user-supplied pure state
vectors (from `spin_state`) into Liouville-space density-matrix vectors.

# Example
```julia
ψ = [1.0+0im, 0.0+0im]          # |+z⟩
v = pure_state_to_vec_rho(ψ)    # vec([[1,0],[0,0]]) = [1,0,0,0]
```
"""
function pure_state_to_vec_rho(ψ::AbstractVector)::Vector{ComplexF64}
    c = ComplexF64.(ψ)
    return vec_rho(c * c')
end

# ─── Liouvillian builders ─────────────────────────────────────────────────────

"""
    build_drift_liouvillian(H_drift, jump_ops, decay_rates) → Matrix{ComplexF64}

Construct the N²×N² Liouvillian superoperator for the time-independent
(drift) part of the Lindblad master equation.

    𝓛_drift = −i(I⊗H_drift − H_drift^T⊗I)
             + Σ_k γ_k [ conj(L_k)⊗L_k − ½(I⊗L_k†L_k) − ½(conj(L_k†L_k)⊗I) ]

The result acts on vec(ρ): `d/dt vec(ρ) = (𝓛_drift + Σ_k w_k 𝓛_ctrl_k) vec(ρ)`.

For a **closed system** (no relaxation) pass `jump_ops = []`, `decay_rates = []`.

# Arguments
- `H_drift`     — N×N Hermitian drift Hamiltonian in rad/s
- `jump_ops`    — Vector of N×N Lindblad jump operators L_k (can be empty)
- `decay_rates` — Vector of rates γ_k in rad/s (same length as `jump_ops`)

# Returns
N²×N² complex matrix (generally non-Hermitian; real part of eigenvalues ≤ 0
for a valid CPTP Lindbladian).

# Example
```julia
sys  = mr_system("13C")
H    = hamiltonian(sys; offset_hz = 500.0)
L_m  = sys.Im[1]                             # I₋ (T1 emission)
𝓛    = build_drift_liouvillian(H, [L_m], [0.5])   # T1 = 2 s  →  γ = 0.5 rad/s
```
"""
function build_drift_liouvillian(
    H_drift     :: Matrix{ComplexF64},
    jump_ops    :: Vector{Matrix{ComplexF64}},
    decay_rates :: Vector{Float64},
)::Matrix{ComplexF64}
    N  = size(H_drift, 1)
    IN = Matrix{ComplexF64}(I, N, N)

    # Coherent part: −i(I⊗H − H^T⊗I)
    # Derivation: vec(−i[H,ρ]) = −i(I⊗H)vec(ρ) + i(H^T⊗I)vec(ρ)
    K_IH = kron(IN, H_drift)
    K_HI = kron(transpose(H_drift), IN)
    𝓛   = similar(K_IH)
    @. 𝓛 = -im * K_IH + im * K_HI

    # Dissipative part: Σ_k γ_k D_k
    # D_k = conj(L_k)⊗L_k  −  ½(I⊗A_k)  −  ½(conj(A_k)⊗I)
    # where A_k = L_k†L_k (Hermitian, so conj(A_k) = A_k^T = A_k*)
    for (L, γ) in zip(jump_ops, decay_rates)
        (γ == 0.0 || !isfinite(γ)) && continue
        γ < 0.0 && throw(ArgumentError(
            "Lindblad decay rate must be non-negative; got γ = $γ. " *
            "Negative rates yield a non-CPTP map and unphysical growth."))
        A    = L' * L                             # N×N, Hermitian
        K_LL = kron(conj(L), L)
        K_IA = kron(IN, A)
        K_AI = kron(conj(A), IN)
        @. 𝓛 += γ * (K_LL - 0.5 * K_IA - 0.5 * K_AI)
    end

    return 𝓛
end

"""
    build_control_liouvillian(H_ctrl) → Matrix{ComplexF64}

Construct the N²×N² superoperator for one dimensionless control Hamiltonian.

Control operators affect only the **coherent** (commutator) dynamics; the
dissipation is state-independent and therefore belongs in the drift Liouvillian.

    𝓛_ctrl = −i(I⊗H_ctrl − H_ctrl^T⊗I)

At time step n, the total Liouvillian for ensemble member j is:

    𝓛[n,j] = 𝓛_drift[j] + Σ_k waveform[k,n] × pwr_level × 𝓛_ctrl[k]

Precompute these once (stored in `LindbladMRControl._L_controls`) to avoid
repeated Kronecker products in the GRAPE inner loop.

# Arguments
- `H_ctrl` — N×N dimensionless control operator (e.g. `Ix = 0.5σx`)

# Returns
N²×N² complex matrix.

# Example
```julia
sys   = mr_system("13C")
Lx    = spin_op(sys, :Ix)
𝓛_x  = build_control_liouvillian(Lx)    # 4×4 superoperator for Ix
```
"""
function build_control_liouvillian(H_ctrl::Matrix{ComplexF64})::Matrix{ComplexF64}
    N    = size(H_ctrl, 1)
    IN   = Matrix{ComplexF64}(I, N, N)
    K_IH = kron(IN, H_ctrl)
    K_HI = kron(transpose(H_ctrl), IN)
    out  = similar(K_IH)
    @. out = -im * K_IH + im * K_HI
    return out
end

# ─── Lindblad GRAPE gradient prefactor ───────────────────────────────────────

"""
    lindblad_grad_prefactor(z, inner, dt_pwr; type) → Float64

Chain-rule coefficient ∂F/∂w[k,n] for the Lindblad GRAPE adjoint method.

# Arguments
- `z`      — complex Hilbert-Schmidt overlap ⟨σ_targ | σ(T)⟩
- `inner`  — ⟨λ[n+1] | 𝓛_ctrl[k] | σ[n]⟩  (Liouville space inner product)
- `dt_pwr` — dt[n] × pwr_level (s × rad/s = dimensionless)
- `type`   — `:dm_real` (Φ^D_R), `:dm_modulus` (Φ^D_M) or `:dm_squared` (Φ^D_Q)

Note: Unlike the Hilbert-space `fidelity_grad_prefactor`, this uses **Re()**
because the −i factor is already embedded inside 𝓛_ctrl = −i(commutator).

`dt_pwr · Re(w · inner)` with the chain weight `w = overlap_weight(type, z)`:

| `type`        | Formula                                  |
|:------------- |:---------------------------------------- |
| `:dm_real`    | `dt_pwr · Re(inner)`                     |
| `:dm_modulus` | `dt_pwr · Re(conj(z) · inner)/\\|z\\|`   |
| `:dm_squared` | `2 · dt_pwr · Re(conj(z) · inner)`       |
"""
@inline function lindblad_grad_prefactor(z::ComplexF64,
                                          inner::ComplexF64,
                                          dt_pwr::Float64;
                                          type::Symbol)::Float64
    type in DM_METRICS ||
        return lindblad_grad_prefactor(z, inner, dt_pwr;
            type = canonical_metric(type, DM_METRICS; context = "lindblad_grad_prefactor"))
    return dt_pwr * real(overlap_weight(type, z) * inner)
end

# ─── Convenience constructor ──────────────────────────────────────────────────

"""
    lindblad_system_from_jump_ops(H_drift, jump_ops, decay_rates,
                                  H_controls; metadata) -> QuantumSystem

Construct a [`QuantumSystem`](@ref) whose drift Hamiltonian is replaced by the
full Liouville-space superoperator built from `H_drift` and the supplied
Lindblad jump operators, and whose control operators are the corresponding
control Liouvillians.

This is the recommended entry point for open-system optimal control with the
standard GRAPE / L-BFGS optimizers operating in Liouville space.  The returned
`QuantumSystem` has dimension N² (where N = size(H_drift, 1)); to recover the
density matrix from a propagated state vector use [`mat_rho`](@ref).

# Arguments
- `H_drift`      — N×N Hermitian drift Hamiltonian (rad/s)
- `jump_ops`     — Vector of N×N Lindblad jump operators L_k
- `decay_rates`  — Vector{Float64} of rates γ_k (rad/s); same length as `jump_ops`
- `H_controls`   — Vector of N×N Hermitian control Hamiltonians
- `metadata`     — optional `Dict{String,Any}` stored in the returned system

# Returns
`QuantumSystem` with `dim = N²`, operating in Liouville (superoperator) space.

# Example
```julia
sys = mr_system("13C")
H   = hamiltonian(sys; offset_hz = 100.0)
Lm  = sys.Im[1]            # T1 relaxation
Lz  = sys.Iz[1]            # T2 dephasing
qs  = lindblad_system_from_jump_ops(H, [Lm, Lz], [0.5, 0.2],
                                     sys.H_controls)
# qs.dim == 4  (2² Liouville space for a single spin-1/2)
```
"""
function lindblad_system_from_jump_ops(
    H_drift     :: Matrix{ComplexF64},
    jump_ops    :: Vector{Matrix{ComplexF64}},
    decay_rates :: Vector{Float64},
    H_controls  :: Vector{Matrix{ComplexF64}};
    metadata    :: Dict{String,Any} = Dict{String,Any}(),
)::QuantumSystem
    N  = size(H_drift, 1)
    @assert length(jump_ops) == length(decay_rates) "jump_ops and decay_rates must have equal length"

    𝓛_drift = build_drift_liouvillian(H_drift, jump_ops, decay_rates)
    𝓛_ctrls = [build_control_liouvillian(Hk) for Hk in H_controls]

    return QuantumSystem(𝓛_drift, 𝓛_ctrls, N^2, length(H_controls), metadata)
end
