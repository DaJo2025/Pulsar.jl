# Application/MR/DNP/DNPLindblad.jl
#
# Lindblad-space GRAPE kernel for DNP optimal control.
# Models polarization transfer dynamics via explicit relaxation operators
# for the solid effect (SE) and cross effect (CE) mechanisms.
#
# Algorithm follows the same structure as GRAPELindblad.jl:
#   Forward:  σ[n+1] = exp(𝓛[n] dt) σ[n]          (vectorised density matrix)
#   Backward: λ[n]   = exp(𝓛[n] dt)† λ[n+1]       (co-state)
#   Gradient: G[k,n] = lindblad_grad_prefactor(z, ⟨λ[n+1]|𝓛_ctrl[k]|σ[n]⟩, dt)

using LinearAlgebra

# ─── Jump operator builders ───────────────────────────────────────────────────

"""
    _dnp_jump_ops(sys::DNPSpinSystem; T1_electron_s, T2_electron_s,
                  T1_nuclear_s, cross_relax_rate)
        -> (jump_ops::Vector{Matrix{ComplexF64}}, rates::Vector{Float64})

Build Lindblad jump operators and corresponding decay rates for a DNP
spin system.

Jump operators added (when the corresponding rate is finite and positive):

1. **Electron T1** — lowering `S⁻ = Sx − iSy`, rate `1/T1_e`
2. **Electron pure dephasing** — `Sz/2`, rate `max(1/T2_e − 1/(2T1_e), 0)`
3. **Cross-relaxation (SE/CE)** — `S⁺ I⁻_k` and `S⁻ I⁺_k` per nucleus,
   rate `cross_relax_rate`
4. **Nuclear T1** — `I⁻_k` per nucleus, rate `1/T1_n_k`
"""
function _dnp_jump_ops(
    sys              :: DNPSpinSystem;
    T1_electron_s    :: Float64                        = Inf,
    T2_electron_s    :: Float64                        = Inf,
    T1_nuclear_s     :: Union{Float64,Vector{Float64}} = Inf,
    cross_relax_rate :: Float64                        = 0.0,
)
    Sx, Sy, Sz = sys.S_ops
    n_nuc  = length(sys.I_nuclei)
    T1_n   = T1_nuclear_s isa Float64 ? fill(T1_nuclear_s, n_nuc) : T1_nuclear_s
    length(T1_n) == n_nuc ||
        throw(ArgumentError(
            "T1_nuclear_s must be scalar or Vector of length $n_nuc"))

    jump_ops = Matrix{ComplexF64}[]
    rates    = Float64[]

    # Electron T1 (lowering S⁻)
    if isfinite(T1_electron_s) && T1_electron_s > 0.0
        push!(jump_ops, Sx .- im .* Sy)
        push!(rates, 1.0 / T1_electron_s)
    end

    # Electron pure dephasing
    γ_φ = 0.0
    if isfinite(T2_electron_s) && T2_electron_s > 0.0
        γ2 = 1.0 / T2_electron_s
        γ1 = isfinite(T1_electron_s) && T1_electron_s > 0.0 ? 1.0/T1_electron_s : 0.0
        γ_φ = max(γ2 - γ1/2.0, 0.0)
    end
    if γ_φ > 0.0
        push!(jump_ops, Sz ./ 2.0)
        push!(rates, γ_φ)
    end

    # Cross-relaxation per nucleus (S⁺ I⁻ and S⁻ I⁺)
    if cross_relax_rate > 0.0
        S_plus  = Sx .+ im .* Sy
        S_minus = Sx .- im .* Sy
        for k in 1:n_nuc
            Ix, Iy, _ = sys.I_ops[k]
            I_plus  = Ix .+ im .* Iy
            I_minus = Ix .- im .* Iy
            push!(jump_ops, S_plus  * I_minus)
            push!(rates,    cross_relax_rate)
            push!(jump_ops, S_minus * I_plus)
            push!(rates,    cross_relax_rate)
        end
    end

    # Nuclear T1 per spin
    for k in 1:n_nuc
        if isfinite(T1_n[k]) && T1_n[k] > 0.0
            Ix, Iy, _ = sys.I_ops[k]
            push!(jump_ops, Ix .- im .* Iy)
            push!(rates, 1.0 / T1_n[k])
        end
    end

    return jump_ops, rates
end

# ─── Lindblad kernel ─────────────────────────────────────────────────────────

"""
    grape_dnp_lindblad_kernel(sys::DNPSpinSystem, ctrl::ControlSequence;
                               T1_electron_s, T2_electron_s,
                               T1_nuclear_s, cross_relax_rate,
                               config) -> OptimizationResult

Run GRAPE in Liouville space to maximise nuclear polarization for a DNP
spin system.

The objective is `F = Re(⟨vec_ρ_target | vec_ρ(T)⟩)` where
`vec_ρ_target ∝ Σ_k I_{z,k}` (nuclear polarization operator) and
`vec_ρ(T)` is the final vectorised density matrix propagated through the
Lindblad master equation.

Propagation uses the full matrix exponential `exp(𝓛_total dt)` for each
time step (Padé/Schur method via `LinearAlgebra.exp`), identical to
`GRAPELindblad.jl`.

# Arguments
- `sys`               — `DNPSpinSystem` from `dnp_system(...)`
- `ctrl`              — initial `ControlSequence`
- `T1_electron_s`     — electron T₁ (s); default `Inf` (no relaxation)
- `T2_electron_s`     — electron T₂ (s); default `Inf`
- `T1_nuclear_s`      — nuclear T₁ (s); scalar or per-nucleus vector; default `Inf`
- `cross_relax_rate`  — zero-quantum cross-relaxation rate (rad/s); default 0
- `config`            — `GRAPEConfig` (default `GRAPEConfig()`)

# Returns
`OptimizationResult` with the controls that maximise nuclear polarization.

# Example
```julia
sys  = dnp_system(1//2, [1//2], 9.4; mw_freq_hz=263e9, omega_r_hz=10e3)
ctrl = ControlSequence(0.01 .* randn(sys.n_controls, 100), 1e-6, 1e-4, 100)

result = grape_dnp_lindblad_kernel(sys, ctrl;
    T1_electron_s    = 1e-3,
    T2_electron_s    = 100e-6,
    cross_relax_rate = 1e3,
)
@show result.fidelity
```
"""
function grape_dnp_lindblad_kernel(
    sys              :: DNPSpinSystem,
    ctrl             :: ControlSequence;
    T1_electron_s    :: Float64                        = Inf,
    T2_electron_s    :: Float64                        = Inf,
    T1_nuclear_s     :: Union{Float64,Vector{Float64}} = Inf,
    cross_relax_rate :: Float64                        = 0.0,
    config           :: GRAPEConfig                   = GRAPEConfig(),
)::OptimizationResult

    jump_ops, rates = _dnp_jump_ops(sys;
        T1_electron_s    = T1_electron_s,
        T2_electron_s    = T2_electron_s,
        T1_nuclear_s     = T1_nuclear_s,
        cross_relax_rate = cross_relax_rate,
    )

    if isempty(jump_ops)
        @warn "grape_dnp_lindblad_kernel: no finite relaxation rates; " *
              "falling back to closed-system optimcon_dnp."
        return optimcon_dnp(sys, ctrl; config=config)
    end

    # Build Liouvillians (precomputed once)
    L_drift    = build_drift_liouvillian(sys.H_drift, jump_ops, rates)
    L_controls = [build_control_liouvillian(H) for H in sys.H_controls]
    N2         = size(L_drift, 1)   # dim² (Liouville space dimension)

    # Vectorised initial and target density matrices
    rho0       = electron_polarized_state(sys)
    vec_σ0     = vec_rho(rho0)
    Op_I       = nuclear_polarization_operator(sys)
    σ_norm     = norm(vec_rho(Op_I))
    vec_σ_targ = σ_norm > 0 ? vec_rho(Op_I) ./ σ_norm : vec_rho(Op_I)

    # ── Custom fidelity: F = Re(⟨vec_σ_targ | σ(T)⟩) ───────────────────────
    function dnp_lindblad_fidelity(_, c, __)
        n_t = c.n_steps
        dt  = c.dt
        σ   = copy(vec_σ0)
        for n in 1:n_t
            L = copy(L_drift)
            for k in 1:sys.n_controls
                L .+= c.amplitudes[n, k] .* L_controls[k]
            end
            σ = exp(L .* dt) * σ
        end
        return real(dot(vec_σ_targ, σ))
    end

    # ── Custom gradient: GRAPE adjoint in Liouville space ────────────────────
    function dnp_lindblad_gradient(_, c, __)
        n_t    = c.n_steps
        dt     = c.dt
        nc     = sys.n_controls

        # Build and cache propagators Φ[n] = exp(𝓛_total[n] dt)
        Phi = [begin
            L = copy(L_drift)
            for k in 1:nc
                L .+= c.amplitudes[n, k] .* L_controls[k]
            end
            exp(L .* dt)
        end for n in 1:n_t]

        # Forward trajectory
        σ_fwd = Matrix{ComplexF64}(undef, N2, n_t + 1)
        σ_fwd[:, 1] .= vec_σ0
        for n in 1:n_t
            σ_fwd[:, n+1] .= Phi[n] * σ_fwd[:, n]
        end

        # Backward co-state (adjoint propagator = Φ†)
        λ_mat = Matrix{ComplexF64}(undef, N2, n_t + 1)
        λ_mat[:, n_t + 1] .= vec_σ_targ
        for n in n_t:-1:1
            λ_mat[:, n] .= Phi[n]' * λ_mat[:, n + 1]
        end

        # Fidelity overlap z = ⟨σ_targ | σ(T)⟩ (for the grad prefactor)
        z = ComplexF64(dot(vec_σ_targ, σ_fwd[:, n_t + 1]))

        # Gradient [n_controls × n_steps] — matches ctrl.controls shape
        G = zeros(nc, n_t)
        for n in 1:n_t
            for k in 1:nc
                tmp   = L_controls[k] * σ_fwd[:, n]
                inner = ComplexF64(dot(λ_mat[:, n + 1], tmp))
                G[k, n] += lindblad_grad_prefactor(z, inner, dt; type=:dm_real)
            end
        end
        return G
    end

    return grape_optimize(sys, nothing, ctrl;
        fidelity_fn = dnp_lindblad_fidelity,
        gradient_fn = dnp_lindblad_gradient,
        config      = config,
    )
end
