"""
    Propagators.jl

Time-evolution propagators for piecewise-constant quantum control sequences.

All evolution follows the Schrödinger equation (ħ = 1):

    i d/dt U(t) = H(t) U(t),    U(0) = I

For a piecewise-constant Hamiltonian H[k] active during time slice [t_{k-1}, t_k]
with step size dt = t_k - t_{k-1}, the propagator for that slice is:

    U[k] = exp(-i H[k] dt)

computed exactly via eigendecomposition of the Hermitian matrix H[k].

The full propagator after n steps is the time-ordered product:

    U_total = U[n] * U[n-1] * … * U[1]
"""

using LinearAlgebra

# ============================================================================
# Single-step propagator
# ============================================================================

"""
    compute_propagator(H::Matrix{ComplexF64}, dt::Float64) -> Matrix{ComplexF64}

Compute the exact unitary propagator U = exp(-i H dt) for a Hermitian H.

# Arguments
- `H`  — Hermitian matrix of size `dim × dim` (rad/s)
- `dt` — time step duration in seconds; may be zero (returns identity)

# Returns
Unitary matrix `U = exp(-i H dt)` of size `dim × dim`.

# Algorithm
Uses the eigendecomposition H = V D V† where D is real diagonal, giving

    exp(-i H dt) = V diag(exp(-i λ_k dt)) V†

This is exact for any Hermitian H and avoids the truncation error of Padé or
Taylor series approximations.  The eigendecomposition is computed via
`LinearAlgebra.eigen` (calls LAPACK `dsyevd` or `zheevd`).

# Numerical properties
- Exact for any Hermitian H (no approximation).
- Resulting matrix is unitary to machine precision (round-off ~ dim * ε_mach).
- Cost: O(dim³) for the eigendecomposition.

# Throws
- `ArgumentError` if H is not square.
- `ArgumentError` if H is not Hermitian within tolerance 1e-10.

# Example
```julia
H = [0.0+0im 1.0+0im; 1.0+0im 0.0+0im]   # σx (rad/s)
U = compute_propagator(H, π/2)              # π/2 rotation: U = -i σx ... up to phase
```
"""
function compute_propagator(H::AbstractMatrix{ComplexF64}, dt::Real)::Matrix{ComplexF64}
    dt = Float64(dt)
    m, n = size(H)
    if m != n
        throw(ArgumentError("H must be square, got $m × $n"))
    end

    # Short-circuit for zero dt: propagator is the identity
    if dt == 0.0
        return Matrix{ComplexF64}(I, m, m)
    end

    # Validate Hermiticity
    _propagator_check_hermitian(H, "H")

    # Single-spin (dim=2) fast path: closed-form Pauli-decomposition exponential.
    if m == 2
        return _propagator_2x2_pauli(H, dt)
    end

    # Eigendecomposition of Hermitian matrix: H = V * Diagonal(λ) * V†
    # Julia 1.12 reroutes the Hermitian eigen path through LAPACK.lacpy! / chkstride1,
    # which requires contiguous columns. Strided views (e.g. H_total[k, :, :] of a
    # [n_steps × dim × dim] array) are not contiguous along the first axis, so we
    # materialize to a dense matrix before dispatching. On 1.9–1.11 this was implicit;
    # on 1.12+ it is mandatory. TODO(v0.3): swap to H[:,:,k] layout to remove this copy.
    F = eigen(Hermitian(Matrix(H)))  # ensures real eigenvalues; uses LAPACK *heevd
    λ = F.values                     # real eigenvalues (rad/s)
    V = F.vectors                    # unitary matrix of eigenvectors

    # Build exp(-i λ_k dt) for each eigenvalue
    phases = exp.((-im * dt) .* λ)  # Vector{ComplexF64}

    # Reconstruct: U = V * diag(phases) * V†
    U = V * Diagonal(phases) * V'
    return U
end

# ============================================================================
# Multi-step propagators
# ============================================================================

"""
    compute_propagators(H_total::Array{ComplexF64,3}, dt::Float64)
    -> Array{ComplexF64,3}

Compute individual propagators U[k] = exp(-i H[k] dt) for each time step.

# Arguments
- `H_total` — 3-D array of Hamiltonians with layout `[n_timesteps × dim × dim]`;
  `H_total[k, :, :]` is the Hamiltonian during time slice k.
- `dt`       — uniform time step in seconds (> 0)

# Returns
Array `U` of shape `[n_timesteps × dim × dim]` where `U[k, :, :]` is the
propagator for time slice k.

# Notes
Each propagator is computed independently via `compute_propagator`, so the
total cost is O(n_timesteps * dim³).

# Throws
- `ArgumentError` if `H_total` does not have three dimensions or if the last
  two dimensions are not equal (non-square matrices).

# Example
```julia
n, d = 50, 4
H_total = zeros(ComplexF64, n, d, d)
# ... fill H_total ...
U_steps = compute_propagators(H_total, 1e-5)
```
"""
function compute_propagators(H_total::Array{ComplexF64,3},
                              dt::Real)::Array{ComplexF64,3}
    dt = Float64(dt)
    if ndims(H_total) != 3
        throw(ArgumentError(
            "H_total must be a 3-D array [n_timesteps × dim × dim], " *
            "got $(ndims(H_total))-D"))
    end
    n_timesteps, d1, d2 = size(H_total)
    if d1 != d2
        throw(ArgumentError(
            "H_total last two dimensions must be equal (square matrices), " *
            "got $d1 × $d2"))
    end
    if dt <= 0.0
        throw(ArgumentError("dt must be positive, got $dt"))
    end

    dim = d1
    propagators = Array{ComplexF64,3}(undef, n_timesteps, dim, dim)
    @inbounds for k in 1:n_timesteps
        Hk = @view H_total[k, :, :]    # dim × dim slice (view, no alloc)
        Uk = compute_propagator(Hk, dt)
        copyto!(@view(propagators[k, :, :]), Uk)
    end
    return propagators
end

"""
    compute_propagators!(propagators, H_total, dt; H_buf=similar(H_total[1,:,:]),
                          tmp=similar(H_total[1,:,:]))

In-place variant of `compute_propagators`: fills the pre-allocated `propagators`
array (shape `[n_timesteps × dim × dim]`) with `exp(-i H[k] dt)` for each k.
Uses `_expm_neg_i_into!` (LAPACK Hermitian eigendecomposition) to avoid the
generic Padé path used by the immutable `compute_propagator`.

`H_buf` and `tmp` are scratch buffers reused across timesteps.
"""
function compute_propagators!(propagators::Array{ComplexF64,3},
                              H_total::Array{ComplexF64,3},
                              dt::Real;
                              H_buf::Matrix{ComplexF64} =
                                  Matrix{ComplexF64}(undef, size(H_total,2), size(H_total,3)),
                              tmp::Matrix{ComplexF64}   =
                                  Matrix{ComplexF64}(undef, size(H_total,2), size(H_total,3)))
    dt = Float64(dt)
    n_timesteps, dim, _ = size(H_total)
    size(propagators) == size(H_total) ||
        throw(DimensionMismatch("propagators size $(size(propagators)) ≠ H_total size $(size(H_total))"))
    Uk = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_timesteps
        @views copyto!(H_buf, H_total[k, :, :])
        _expm_neg_i_into!(Uk, H_buf, dt, tmp)
        @views propagators[k, :, :] .= Uk
    end
    return propagators
end

"""
    compute_total_propagator(propagators::Array{ComplexF64,3}) -> Matrix{ComplexF64}

Compute the total time-ordered propagator as the sequential product of step propagators.

# Arguments
- `propagators` — array of shape `[n_timesteps × dim × dim]`; `propagators[k, :, :]`
  is the propagator U[k] for time slice k.

# Returns
Total propagator `U_total = U[n] * U[n-1] * … * U[1]` of size `dim × dim`.

# Mathematical detail
The time-ordered product satisfies the Schrödinger equation: the state at time T
is |ψ(T)⟩ = U_total |ψ(0)⟩.  The ordering U[n] * … * U[1] ensures that U[1]
acts first (rightmost = first in time).

# Edge cases
- If `n_timesteps == 0`, returns the `dim × dim` identity matrix (empty product).
- If `n_timesteps == 1`, returns `propagators[1, :, :]` directly.

# Example
```julia
U_total = compute_total_propagator(U_steps)
```
"""
function compute_total_propagator(propagators::Array{ComplexF64,3})::Matrix{ComplexF64}
    if ndims(propagators) != 3
        throw(ArgumentError(
            "propagators must be a 3-D array, got $(ndims(propagators))-D"))
    end
    n_timesteps, dim, d2 = size(propagators)
    if dim != d2
        throw(ArgumentError(
            "propagators last two dims must be equal, got $dim × $d2"))
    end
    if n_timesteps == 0
        # Empty time-ordered product equals the identity operator.
        return Matrix{ComplexF64}(I, dim, dim)
    end

    # Start from identity and apply propagators left-to-right in reverse time order
    # U_total = U[n] * ... * U[1]
    U_total = Matrix{ComplexF64}(I, dim, dim)
    U_k     = Matrix{ComplexF64}(undef, dim, dim)
    tmp     = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_timesteps
        @views copyto!(U_k, propagators[k, :, :])
        mul!(tmp, U_k, U_total)
        U_total, tmp = tmp, U_total     # swap workspace buffers
    end
    return U_total
end

# ============================================================================
# Forward and backward propagators for GRAPE
# ============================================================================

"""
    compute_forward_propagators(propagators::Array{ComplexF64,3})
    -> Array{ComplexF64,3}

Compute cumulative forward propagators P[k] used in GRAPE gradient computation.

# Arguments
- `propagators` — step propagators of shape `[n_timesteps × dim × dim]`

# Returns
Array `P` of shape `[(n_timesteps+1) × dim × dim]` where:

    P[1, :, :] = I                   (before any evolution)
    P[k, :, :] = U[k-1] * … * U[1]  (forward evolution up to the end of step k-1)

Indexing convention: `P[k]` is the propagator *into* time slice k, i.e. it
maps the initial state to the state just before step k begins.  In particular
`P[n_timesteps+1]` equals the full propagator U_total.

# Notes
This convention matches the GRAPE paper by Khaneja et al. (2005): the forward
propagator to the left of slice k is used when computing the gradient with
respect to u[k].

# Cost
O(n_timesteps * dim²) matrix multiplications after the initial propagators are known.

# Example
```julia
P = compute_forward_propagators(U_steps)
# P[1] = I, P[end] = U_total
```
"""
function compute_forward_propagators(propagators::Array{ComplexF64,3})::Array{ComplexF64,3}
    if ndims(propagators) != 3
        throw(ArgumentError(
            "propagators must be 3-D [n_timesteps × dim × dim]"))
    end
    n_timesteps, dim, d2 = size(propagators)
    if dim != d2
        throw(ArgumentError(
            "propagators last two dims must be equal, got $dim × $d2"))
    end

    # P has n_timesteps+1 entries: P[1]=I, P[k]=U[k-1]…U[1] for k>1
    P = Array{ComplexF64,3}(undef, n_timesteps + 1, dim, dim)
    P[1, :, :] = Matrix{ComplexF64}(I, dim, dim)
    U_k   = Matrix{ComplexF64}(undef, dim, dim)
    P_k   = Matrix{ComplexF64}(undef, dim, dim)
    P_out = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_timesteps
        @views copyto!(U_k, propagators[k, :, :])
        @views copyto!(P_k, P[k, :, :])
        mul!(P_out, U_k, P_k)
        @views P[k+1, :, :] .= P_out
    end
    return P
end

"""
    compute_forward_propagators!(P, propagators)

In-place variant: fills the pre-allocated `P` array (shape
`[(n_timesteps+1) × dim × dim]`) with `P[1] = I` and `P[k+1] = U[k] · P[k]`.
"""
function compute_forward_propagators!(P::Array{ComplexF64,3},
                                       propagators::Array{ComplexF64,3})
    n_timesteps, dim, _ = size(propagators)
    size(P) == (n_timesteps + 1, dim, dim) ||
        throw(DimensionMismatch("P shape $(size(P)) ≠ ($(n_timesteps+1), $dim, $dim)"))
    @views P[1, :, :] .= Matrix{ComplexF64}(I, dim, dim)
    U_k   = Matrix{ComplexF64}(undef, dim, dim)
    P_k   = Matrix{ComplexF64}(undef, dim, dim)
    P_out = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_timesteps
        @views copyto!(U_k, propagators[k, :, :])
        @views copyto!(P_k, P[k, :, :])
        mul!(P_out, U_k, P_k)
        @views P[k+1, :, :] .= P_out
    end
    return P
end

"""
    compute_backward_propagators(propagators::Array{ComplexF64,3})
    -> Array{ComplexF64,3}

Compute cumulative backward propagators Q[k] used in GRAPE gradient computation.

# Arguments
- `propagators` — step propagators of shape `[n_timesteps × dim × dim]`

# Returns
Array `Q` of shape `[(n_timesteps+1) × dim × dim]` where:

    Q[n_timesteps+1, :, :] = I
    Q[k, :, :]             = U[k+1]† * … * U[n]†    for k ≤ n_timesteps

Equivalently, Q[k] is the adjoint of the forward propagator from step k+1 to
the end:  Q[k] = (U[n] * … * U[k+1])†.

# GRAPE interpretation
For the gate fidelity gradient, the backward co-state at step k is

    λ[k] = U_target† * Q[k]†  (up to an overall phase factor)

and the gradient is proportional to Tr( λ[k] * (-i H_j dt) * P[k] ).

# Example
```julia
Q = compute_backward_propagators(U_steps)
# Q[n+1] = I, Q[1] = (U[n]…U[2])†
```
"""
function compute_backward_propagators(propagators::Array{ComplexF64,3})::Array{ComplexF64,3}
    if ndims(propagators) != 3
        throw(ArgumentError(
            "propagators must be 3-D [n_timesteps × dim × dim]"))
    end
    n_timesteps, dim, d2 = size(propagators)
    if dim != d2
        throw(ArgumentError(
            "propagators last two dims must be equal, got $dim × $d2"))
    end

    # Q[n_timesteps+1] = I, Q[k] = U[k+1]† * Q[k+1]
    Q = Array{ComplexF64,3}(undef, n_timesteps + 1, dim, dim)
    Q[n_timesteps + 1, :, :] = Matrix{ComplexF64}(I, dim, dim)
    U_k   = Matrix{ComplexF64}(undef, dim, dim)
    Q_nxt = Matrix{ComplexF64}(undef, dim, dim)
    Q_out = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in n_timesteps:-1:1
        @views copyto!(U_k,   propagators[k, :, :])
        @views copyto!(Q_nxt, Q[k+1, :, :])
        mul!(Q_out, U_k', Q_nxt)
        @views Q[k, :, :] .= Q_out
    end
    return Q
end

"""
    compute_backward_propagators!(Q, propagators)

In-place variant of the unseeded backward propagator: writes `Q[n+1] = I` and
`Q[k] = U[k+1]† · Q[k+1]` for k from n down to 1.
"""
function compute_backward_propagators!(Q::Array{ComplexF64,3},
                                        propagators::Array{ComplexF64,3})
    n_timesteps, dim, _ = size(propagators)
    size(Q) == (n_timesteps + 1, dim, dim) ||
        throw(DimensionMismatch("Q shape $(size(Q)) ≠ ($(n_timesteps+1), $dim, $dim)"))
    @views Q[n_timesteps + 1, :, :] .= Matrix{ComplexF64}(I, dim, dim)
    U_k   = Matrix{ComplexF64}(undef, dim, dim)
    Q_nxt = Matrix{ComplexF64}(undef, dim, dim)
    Q_out = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in n_timesteps:-1:1
        @views copyto!(U_k,   propagators[k, :, :])
        @views copyto!(Q_nxt, Q[k+1, :, :])
        mul!(Q_out, U_k', Q_nxt)
        @views Q[k, :, :] .= Q_out
    end
    return Q
end

"""
    compute_backward_propagators(propagators, seed) -> Array{ComplexF64,3}

Seeded variant: `Q[n+1] = seed` and `Q[k] = U[k+1]† · Q[k+1]`.

Used by penalty gradients (e.g. leakage) where the adjoint is initialized at
`U_total† · Π` instead of the identity.
"""
function compute_backward_propagators(propagators :: Array{ComplexF64,3},
                                       seed        :: AbstractMatrix)::Array{ComplexF64,3}
    n_timesteps, dim, d2 = size(propagators)
    if dim != d2
        throw(ArgumentError(
            "propagators last two dims must be equal, got $dim × $d2"))
    end
    if size(seed, 1) != dim || size(seed, 2) != dim
        throw(ArgumentError(
            "seed must be $dim × $dim, got $(size(seed))"))
    end

    Q = Array{ComplexF64,3}(undef, n_timesteps + 1, dim, dim)
    @views Q[n_timesteps + 1, :, :] .= seed
    U_k   = Matrix{ComplexF64}(undef, dim, dim)
    Q_nxt = Matrix{ComplexF64}(undef, dim, dim)
    Q_out = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in n_timesteps:-1:1
        @views copyto!(U_k,   propagators[k, :, :])
        @views copyto!(Q_nxt, Q[k+1, :, :])
        mul!(Q_out, U_k', Q_nxt)
        @views Q[k, :, :] .= Q_out
    end
    return Q
end

# ============================================================================
# Build total Hamiltonian array from system and controls
# ============================================================================

"""
    build_total_hamiltonian(system::AbstractQuantumSystem,
                             controls::ControlSequence) -> Array{ComplexF64,3}

Build the array of total Hamiltonians H[k] for each time slice k.

# Arguments
- `system`   — quantum system with fields `H_drift` and `H_controls`
- `controls` — control sequence with `controls.amplitudes[k, j]` = amplitude of
  control j at time step k (layout `[n_steps × n_controls]`)

# Returns
Array `H_total` of shape `[n_steps × dim × dim]` where:

    H_total[k, :, :] = H_drift + Σ_j u_j[k] * H_controls[j]

# Physics
This implements the standard linear-response coupling between classical
electromagnetic fields u_j(t) and the quantum system via the coupling
Hamiltonians H_controls[j].

# Notes
- All matrices are promoted to `ComplexF64`.
- Amplitudes are read from `controls.amplitudes[k, j]` (`[n_steps × n_controls]`
  layout); the inner j-loop is stride-1 for fixed k in column-major Julia.
- The function does not validate Hermiticity of H_total (it follows from
  the Hermiticity of H_drift and H_controls combined with real amplitudes).

# Throws
- `ArgumentError` if `system.n_controls` does not match the number of control
  channels in `controls.amplitudes`.

# Example
```julia
H_total = build_total_hamiltonian(sys, seq)
U_steps = compute_propagators(H_total, seq.dt)
U_full  = compute_total_propagator(U_steps)
```
"""
function build_total_hamiltonian(system::AbstractQuantumSystem,
                                  controls::ControlSequence)::Array{ComplexF64,3}
    n_steps    = controls.n_steps
    n_controls = size(controls.amplitudes, 2)   # [n_steps × n_controls]
    if n_controls != system.n_controls
        throw(ArgumentError(
            "controls has $n_controls control channels but system has " *
            "$(system.n_controls) control fields"))
    end

    dim    = system.dim
    H_total = Array{ComplexF64,3}(undef, n_steps, dim, dim)
    amps   = controls.amplitudes   # [n_steps × n_controls], cache-friendly access
    H_drift    = system.H_drift
    H_controls = system.H_controls
    Hk = similar(H_drift)          # contiguous scratch buffer for BLAS axpy!

    @inbounds for k in 1:n_steps
        copyto!(Hk, H_drift)
        for j in 1:n_controls
            LinearAlgebra.axpy!(amps[k, j], H_controls[j], Hk)
        end
        @views H_total[k, :, :] .= Hk
    end

    return H_total
end

"""
    build_total_hamiltonian!(H_total, system, controls)

In-place variant: fills the pre-allocated `H_total` (shape `[n_steps × dim × dim]`)
with `H_total[k] = H_drift + Σ_j u_j[k] H_controls[j]`. Reuses the buffer across
optimizer iterations.
"""
function build_total_hamiltonian!(H_total::Array{ComplexF64,3},
                                   system::AbstractQuantumSystem,
                                   controls::ControlSequence)
    n_steps    = controls.n_steps
    n_controls = size(controls.amplitudes, 2)
    n_controls == system.n_controls ||
        throw(ArgumentError("controls has $n_controls channels but system has $(system.n_controls)"))
    dim = system.dim
    size(H_total) == (n_steps, dim, dim) ||
        throw(DimensionMismatch("H_total shape $(size(H_total)) ≠ ($n_steps, $dim, $dim)"))

    amps       = controls.amplitudes
    H_drift    = system.H_drift
    H_controls = system.H_controls
    Hk = Matrix{ComplexF64}(undef, dim, dim)

    @inbounds for k in 1:n_steps
        copyto!(Hk, H_drift)
        for j in 1:n_controls
            LinearAlgebra.axpy!(amps[k, j], H_controls[j], Hk)
        end
        @views H_total[k, :, :] .= Hk
    end
    return H_total
end

# ============================================================================
# Convenience: full propagation pipeline
# ============================================================================

"""
    propagate(system::AbstractQuantumSystem, controls::ControlSequence)
    -> Matrix{ComplexF64}

Convenience function: build total Hamiltonians, compute step propagators, and
return the total propagator.

# Arguments
- `system`   — quantum system
- `controls` — control sequence

# Returns
Total propagator `U_total = U[n] * … * U[1]`.

# Example
```julia
U = propagate(sys, seq)
```
"""
function propagate(system::AbstractQuantumSystem,
                   controls::ControlSequence)::Matrix{ComplexF64}
    H_total = build_total_hamiltonian(system, controls)
    U_steps = compute_propagators(H_total, controls.dt)
    return compute_total_propagator(U_steps)
end

"""
    propagate_state(system::AbstractQuantumSystem, controls::ControlSequence,
                    psi_init::Vector{ComplexF64}) -> Vector{ComplexF64}

Propagate an initial state vector under the given control sequence.

# Arguments
- `system`    — quantum system
- `controls`  — control sequence
- `psi_init`  — initial state vector of length `system.dim`

# Returns
Final state |ψ(T)⟩ = U_total |ψ_init⟩.

# Example
```julia
psi0 = [1.0+0im, 0.0+0im]
psi_final = propagate_state(sys, seq, psi0)
```
"""
function propagate_state(system::AbstractQuantumSystem,
                          controls::ControlSequence,
                          psi_init::Vector{ComplexF64})::Vector{ComplexF64}
    if length(psi_init) != system.dim
        throw(ArgumentError(
            "psi_init has length $(length(psi_init)) but system.dim = $(system.dim)"))
    end
    U_total = propagate(system, controls)
    return U_total * psi_init
end

# ============================================================================
# Internal helper
# ============================================================================

"""
    _propagator_check_hermitian(H::Matrix{ComplexF64}, name::String; tol::Float64=1e-10)

Check that `H` is Hermitian; throw `ArgumentError` if not.
Used internally to avoid importing the same check from QuantumSystem.jl.
"""
function _propagator_check_hermitian(H::AbstractMatrix{ComplexF64}, name::String;
                                      tol::Float64=1e-10)
    dev = maximum(abs.(H - H'))
    if dev > tol
        throw(ArgumentError(
            "$name is not Hermitian: max|H - H†| = $dev > tol = $tol"))
    end
end

# ============================================================================
# Fast Hermitian propagator (MR layer — in-place, allocation-minimised)
# ============================================================================

"""
    _expm_neg_i(H, dt) → Matrix{ComplexF64}

Compute the propagator `exp(-i H dt)` for a Hermitian matrix `H`.

Uses LAPACK eigendecomposition (`eigen(Hermitian(H))`) rather than the generic
Padé / Schur algorithm used by `exp(M)`.  For the small Hermitian matrices
typical in NMR/EPR/MRI (dim = 2–64), this is 3–5× faster and numerically
more accurate (real eigenvalues → no cancellation error in the exponent).

`cis(x) = exp(im*x)` is used for the scalar exponentials; it avoids the
intermediate `exp(Complex(0, x))` conversion in the generic path.
"""
@inline function _expm_neg_i(H::Matrix{ComplexF64}, dt::Float64)::Matrix{ComplexF64}
    out = Matrix{ComplexF64}(undef, size(H, 1), size(H, 1))
    tmp = Matrix{ComplexF64}(undef, size(H, 1), size(H, 1))
    _expm_neg_i_into!(out, H, dt, tmp)
    return out
end

"""
    _expm_neg_i_into!(out, H, dt, tmp)

In-place version of `_expm_neg_i`: writes `exp(-i H dt)` into pre-allocated
`out`, using `tmp` as a scratch matrix.  Avoids all heap allocations except
those internal to LAPACK's `eigen`.

Called from MR GRAPE kernels where `out` and `tmp` are thread-private buffers.
"""
@inline function _expm_neg_i_into!(out::Matrix{ComplexF64},
                                   H::Matrix{ComplexF64},
                                   dt::Float64,
                                   tmp::Matrix{ComplexF64})
    # Single-spin (dim=2) fast path: closed-form Pauli-decomposition exponential
    # avoids LAPACK eigendecomposition entirely. ~5× faster on 2×2 matrices.
    if size(H, 1) == 2
        return _propagator_2x2_pauli!(out, H, dt)
    end
    F = eigen(Hermitian(H))           # LAPACK allocates F.values, F.vectors
    # Scale each column of F.vectors by cis(-λ_j·dt), write into tmp
    @inbounds for j in eachindex(F.values)
        c = cis(-F.values[j] * dt)
        @simd for i in axes(F.vectors, 1)
            tmp[i, j] = F.vectors[i, j] * c
        end
    end
    # out = tmp * F.vectors'  →  exp(-i H dt)
    mul!(out, tmp, F.vectors')
    return out
end

# ============================================================================
# Single-spin (2×2) closed-form Pauli-decomposition propagator
# ============================================================================

"""
    _pauli_coeffs(H::AbstractMatrix) -> NTuple{4,Float64}

Decompose a 2×2 Hermitian `H` into its Pauli coefficients `(α, b, c, d)` such that
`H = α·I + b·σ_x + c·σ_y + d·σ_z`. The imaginary part of `H[1,1]` and `H[2,2]`
is discarded (assumed Hermitian); `α` is the real trace/2.
"""
@inline function _pauli_coeffs(H::AbstractMatrix)::NTuple{4,Float64}
    @inbounds begin
        h11 = H[1, 1]
        h22 = H[2, 2]
        h12 = H[1, 2]
        α = 0.5 * real(h11 + h22)
        d = 0.5 * real(h11 - h22)
        b = real(h12)
        c = -imag(h12)
    end
    return (α, b, c, d)
end

"""
    _propagator_2x2_pauli!(out, H, dt) -> out

Closed-form `exp(-i H dt)` for a 2×2 Hermitian `H`, written into `out`.
Uses `H = α·I + b·σ_x + c·σ_y + d·σ_z` with

    exp(-iH·dt) = e^{-iα·dt} [cos(r·dt) I − i sin(r·dt) (b·σ_x + c·σ_y + d·σ_z)/r]

where `r = √(b² + c² + d²)`. The `r ≈ 0` branch returns `e^{-iα·dt}·I`.

No allocations, no LAPACK calls. ~5× faster than the eigendecomposition path
on 2×2 matrices.
"""
@inline function _propagator_2x2_pauli!(out::AbstractMatrix{ComplexF64},
                                       H::AbstractMatrix{ComplexF64},
                                       dt::Real)
    α, b, c, d = _pauli_coeffs(H)
    r = sqrt(b*b + c*c + d*d)
    phase = cis(-α * dt)              # e^{-iα·dt}
    @inbounds if r < 1e-30
        out[1, 1] = phase
        out[1, 2] = zero(ComplexF64)
        out[2, 1] = zero(ComplexF64)
        out[2, 2] = phase
    else
        cs = cos(r * dt)
        sn = sin(r * dt)
        # Components of -i sin(r·dt) (b σ_x + c σ_y + d σ_z) / r
        nx = b / r
        ny = c / r
        nz = d / r
        # σ_z diag entries: ±1; σ_x off-diag: (1, 1); σ_y off-diag: (-i, +i)
        # Combined off-diagonal coefficient for (1,2) entry: nx - i·ny  (since σ_y[1,2] = -i)
        # Combined off-diagonal coefficient for (2,1) entry: nx + i·ny
        out[1, 1] = phase * (cs - im * sn * nz)
        out[2, 2] = phase * (cs + im * sn * nz)
        # off-diagonals: -i·sn·(nx σ_x + ny σ_y)[i,j]
        # σ_x[1,2]=1, σ_y[1,2]=-i  → coeff (nx - i·ny) gives factor for (1,2)
        # but we are multiplying by -i·sn:  -i·sn·(nx - i·ny) = -i·sn·nx + i²·sn·ny·(-1)
        # let's just compute directly: M = nx σ_x + ny σ_y has M[1,2] = nx - i·ny, M[2,1] = nx + i·ny
        # then U_offdiag[1,2] = -i·sn·(nx - i·ny) = -i·sn·nx - sn·ny
        out[1, 2] = phase * (-sn * ny - im * sn * nx)
        out[2, 1] = phase * ( sn * ny - im * sn * nx)
    end
    return out
end

"""
    _propagator_2x2_pauli(H, dt) -> Matrix{ComplexF64}

Out-of-place variant: returns a freshly allocated 2×2 propagator.
"""
@inline function _propagator_2x2_pauli(H::AbstractMatrix{ComplexF64},
                                      dt::Real)::Matrix{ComplexF64}
    out = Matrix{ComplexF64}(undef, 2, 2)
    _propagator_2x2_pauli!(out, H, dt)
    return out
end

# MAS/DNP propagator overloads are in Computation/MASPropagators.jl
