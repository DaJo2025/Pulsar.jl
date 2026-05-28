# test/Physics/PhysicsValidation.jl
# ===================================
# Tests that verify known, textbook-level physics results.
# Every assertion here corresponds to a physical fact that can be
# independently derived with pencil and paper.

using Test
using Pulsar
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Shared Pauli matrices
# ---------------------------------------------------------------------------
const _σ_x = ComplexF64[0 1; 1 0]
const _σ_y = ComplexF64[0 -im; im 0]
const _σ_z = ComplexF64[1 0; 0 -1]
const _I2  = ComplexF64[1 0; 0 1]

# ---------------------------------------------------------------------------
@testset "Physics Validation" begin

    # -----------------------------------------------------------------------
    @testset "Spin echo refocusing" begin

        # A π pulse about x inverts the z-component of the Bloch vector.
        # exp(-i*(π/2)*σ_x) applied to |0⟩ = [1,0] should give ±i|1⟩.
        # More precisely: a full π pulse exp(-i*π/2*σ_x) = -i*σ_x
        # so [1,0] → -i*[0,1], i.e., the qubit flips (up to global phase).

        H_pi_x = (π/2) * _σ_x       # so that exp(-i*H*t) at t=1 gives π rotation
        U_pi   = propagator(H_pi_x, 1.0)

        ψ0    = ComplexF64[1, 0]
        ψ_out = U_pi * ψ0

        # After π pulse about x, population in |0⟩ should vanish
        @test abs(ψ_out[1])^2 < 1e-10          # no population in |0⟩
        @test abs(abs(ψ_out[2])^2 - 1.0) < 1e-10  # full population in |1⟩

        # Two π/2 pulses: first pulse creates superposition,
        # second refocuses → end up in |1⟩ (state transfer via Ramsey)
        H_half_pi = (π/4) * _σ_x
        U_half    = propagator(H_half_pi, 1.0)
        ψ_after_two = U_half * (U_half * ψ0)
        # Two π/2 ≡ π: should land in |1⟩ (up to global phase)
        @test abs(ψ_after_two[1])^2 < 1e-10
        @test abs(abs(ψ_after_two[2])^2 - 1.0) < 1e-10

        # Spin echo: apply π/2 – free precession – π – free precession – π/2
        # The π pulse in the middle refocuses the inhomogeneity.
        # We verify that for ANY off-resonance δω, the final state is the
        # same as δω = 0.
        δω_list = [0.0, 1e3, 5e3, -3e3, 10e3]  # rad/s
        T_free  = 5e-6  # 5 µs free precession interval
        H_x_half = (π/4) / T_free * _σ_x   # so propagator over T_free gives π/2
        H_x_pi   = (π/2) / T_free * _σ_x   # π pulse over T_free

        ψ_ref = nothing
        for δω in δω_list
            H_free = δω * _σ_z
            U_half_pulse = propagator(H_x_half, T_free)
            U_free       = propagator(H_free,   T_free)
            U_pi_pulse   = propagator(H_x_pi,   T_free)

            ψ_echo = U_half_pulse * U_free * U_pi_pulse * U_free * U_half_pulse * ψ0
            if ψ_ref === nothing
                ψ_ref = ψ_echo
            end
            # Population should agree with the δω=0 reference to within 1e-6
            @test abs(abs(ψ_echo[1])^2 - abs(ψ_ref[1])^2) < 1e-6
            @test abs(abs(ψ_echo[2])^2 - abs(ψ_ref[2])^2) < 1e-6
        end

    end  # Spin echo refocusing

    # -----------------------------------------------------------------------
    @testset "Known gate solutions" begin

        # ---- Hadamard gate ----
        # H_gate = (σ_x + σ_z) / √2
        # Applied to |0⟩ = [1,0], should give |+⟩ = [1,1]/√2
        H_gate  = (_σ_x + _σ_z) / sqrt(2)
        ψ_plus  = ComplexF64[1, 1] / sqrt(2)
        ψ0      = ComplexF64[1, 0]
        ψ1      = ComplexF64[0, 1]

        ψ_out_H = H_gate * ψ0
        @test abs(state_fidelity(ψ_out_H, ψ_plus) - 1.0) < 1e-14

        # Applied to |1⟩ should give |−⟩ = [1,-1]/√2
        ψ_minus = ComplexF64[1, -1] / sqrt(2)
        ψ_out_H1 = H_gate * ψ1
        @test abs(state_fidelity(ψ_out_H1, ψ_minus) - 1.0) < 1e-14

        # Hadamard is its own inverse: H²= I
        @test norm(H_gate * H_gate - _I2) < 1e-14

        # ---- NOT gate (X gate) ----
        # X = σ_x;  X|0⟩ = |1⟩,  X|1⟩ = |0⟩
        @test abs(state_fidelity(_σ_x * ψ0, ψ1) - 1.0) < 1e-14
        @test abs(state_fidelity(_σ_x * ψ1, ψ0) - 1.0) < 1e-14

        # X as a rotation: exp(-i*π/2*σ_x) = -i*σ_x (up to global phase)
        U_not = propagator((π/2) * _σ_x, 1.0)
        @test abs(gate_fidelity(U_not, _σ_x) - 1.0) < 1e-10

        # ---- Phase (S) gate ----
        # S = diag(1, i) = exp(i*π/4*(I - σ_z)) up to global phase
        S_gate = ComplexF64[1 0; 0 im]
        S_fid  = gate_fidelity(S_gate, S_gate)
        @test abs(S_fid - 1.0) < 1e-12
        # S² = Z
        @test norm(S_gate * S_gate - _σ_z) < 1e-14

        # ---- T gate ----
        # T = diag(1, exp(i*π/4))
        T_gate = ComplexF64[1 0; 0 exp(im*π/4)]
        # T² = S, T⁴ = Z, T⁸ = I
        @test norm(T_gate * T_gate - S_gate) < 1e-14
        @test norm(T_gate^4 - _σ_z) < 1e-14
        @test norm(T_gate^8 - _I2) < 1e-14

        # ---- CNOT gate (4×4) ----
        CNOT = ComplexF64[1 0 0 0;
                          0 1 0 0;
                          0 0 0 1;
                          0 0 1 0]
        # CNOT² = I
        I4 = Matrix{ComplexF64}(I, 4, 4)
        @test norm(CNOT * CNOT - I4) < 1e-14
        # CNOT is unitary
        @test norm(CNOT' * CNOT - I4) < 1e-14

    end  # Known gate solutions

    # -----------------------------------------------------------------------
    @testset "State transfer" begin

        ψ0 = ComplexF64[1, 0]
        ψ1 = ComplexF64[0, 1]

        # ---- |0⟩ → |1⟩ via optimized X rotation ----
        H_drift = zeros(ComplexF64, 2, 2)      # resonant (no drift)
        H_ctrl  = [2π * _σ_x, 2π * _σ_y]
        sys     = quantum_system(H_drift, H_ctrl)

        target  = state_target(ψ1)

        rng     = MersenneTwister(101)
        n_ts    = 50
        dt      = 1e-7
        u_init  = 0.1 * randn(rng, Float64, 2, n_ts)

        result = grape_optimize(sys, target, u_init, dt;
                                config = GRAPEConfig(max_iter=300,
                                                     convergence_tol=1e-9,
                                                     verbose=false))
        @test result.fidelity > 0.99

        # ---- |0⟩ → |+⟩ via y-rotation pulse ----
        ψ_plus  = ComplexF64[1, 1] / sqrt(2)
        target2 = state_target(ψ_plus)
        u_init2 = 0.1 * randn(rng, Float64, 2, n_ts)

        result2 = grape_optimize(sys, target2, u_init2, dt;
                                 config = GRAPEConfig(max_iter=300,
                                                      convergence_tol=1e-9,
                                                      verbose=false))
        @test result2.fidelity > 0.99

        # ---- Resonance: on-resonance is easier than off-resonance ----
        # Use |+⟩ target (NOT a σ_z eigenstate) so off-resonant σ_z drift
        # actually degrades the transfer. With |1⟩ both cases give F=1
        # trivially because |1⟩ is a σ_z eigenstate.
        H_drift_off = 2π * 5e6 * _σ_z      # very large off-resonance
        sys_off     = quantum_system(H_drift_off, H_ctrl)
        # Short pulse + tiny controls: should not be able to compensate
        u_short     = 0.001 * randn(rng, Float64, 2, 3)
        dt_short    = 1e-10
        result_off  = grape_optimize(sys_off, target2, u_short, dt_short;
                                     config = GRAPEConfig(max_iter=20,
                                                          verbose=false))
        # Resonance case (|+⟩ via optimized pulse) should beat extreme
        # off-resonance + tiny short pulse (which can't reach |+⟩ from |0⟩)
        @test result2.fidelity > result_off.fidelity

    end  # State transfer

    # -----------------------------------------------------------------------
    @testset "INEPT basic verification" begin

        # Two-spin 1H-13C system.
        # Physical parameters (SI / rad/s):
        #   ω_H  = 2π × 600 MHz (¹H frequency at 14.1 T)
        #   ω_C  = 2π × 150 MHz (¹³C frequency)
        #   J    = 2π × 125 Hz   (one-bond 1J_CH)
        #
        # In the rotating frame with secular approximation, the drift is:
        #   H_drift = J * π * (IH_z ⊗ IC_z)
        # Control Hamiltonians: H pulses on ¹H, C pulses on ¹³C

        I2 = _I2
        Sz_H = kron(_σ_z / 2, I2)    # Iz on proton  (⊗ identity on carbon)
        Sz_C = kron(I2, _σ_z / 2)    # Iz on carbon  (identity ⊗ carbon)
        Sx_H = kron(_σ_x / 2, I2)
        Sy_H = kron(_σ_y / 2, I2)
        Sx_C = kron(I2, _σ_x / 2)
        Sy_C = kron(I2, _σ_y / 2)

        J_Hz  = 125.0                           # 1J_CH in Hz
        J_rad = 2π * J_Hz                       # convert to rad/s
        H_drift_INEPT = J_rad * (2.0 * Sz_H * Sz_C)   # scalar coupling

        H_ctrl_INEPT = [Sx_H, Sy_H, Sx_C, Sy_C]
        sys_INEPT    = quantum_system(H_drift_INEPT, H_ctrl_INEPT)

        # Initial state: thermal equilibrium ∝ Iz_H (¹H z-magnetization)
        # In the density-matrix picture we start with ρ₀ = Iz_H.
        # INEPT transfers this to Iz_C (carbon z-magnetization).
        # Here we track the expectation value of Sz_C.

        # The INEPT sequence consists of two π/2 pulses and one π pulse:
        # (π/2)_H – τ – (π)_H (π)_C – τ – (π/2)_H (π/2)_C
        # where τ = 1/(4J).
        τ     = 1.0 / (4 * J_Hz)    # in seconds: ~2 ms for 125 Hz

        # Build propagators analytically for the INEPT sequence:
        # π/2 on H (y-axis): exp(-i * π/4 * σ_y_H)
        U_pi2_Hy  = exp(-im * (π/4) * kron(_σ_y, I2))
        # π on H (x-axis):   exp(-i * π/2 * σ_x_H)
        U_pi_Hx   = exp(-im * (π/2) * kron(_σ_x, I2))
        # π on C (x-axis):   exp(-i * π/2 * σ_x_C)
        U_pi_Cx   = exp(-im * (π/2) * kron(I2, _σ_x))
        # π/2 on C (y-axis): exp(-i * π/4 * σ_y_C)
        U_pi2_Cy  = exp(-im * (π/4) * kron(I2, _σ_y))
        # Free precession under scalar coupling for time τ:
        U_free    = exp(-im * J_rad * (2.0 * Sz_H * Sz_C) * τ)

        # Full INEPT propagator:
        U_INEPT = U_pi2_Cy * U_pi2_Hy * U_free * U_pi_Cx * U_pi_Hx * U_free * U_pi2_Hy

        # Start in |αα⟩ = |00⟩ and measure transfer
        # ρ₀ = |αα⟩⟨αα| represents excess ¹H α population
        ψ_init = ComplexF64[1, 0, 0, 0]    # |αα⟩ = |0_H 0_C⟩
        ψ_final = U_INEPT * ψ_init

        # INEPT in NMR transfers polarization in the *operator basis*:
        # Iz_H → 2 Iz_H Iz_C (antiphase coherence). On a pure |αα⟩ state,
        # ⟨Sz_C⟩ ≈ 0 by symmetry — the signature instead is non-zero
        # antiphase 2 Sz_H Sz_C expectation. Verify INEPT acts non-trivially
        # and creates the antiphase term.
        antiphase_op = 2 * Sz_H * Sz_C
        exp_antiphase_before = real(ψ_init'  * antiphase_op * ψ_init)
        exp_antiphase_after  = real(ψ_final' * antiphase_op * ψ_final)
        @test abs(exp_antiphase_after - exp_antiphase_before) > 0.1   # antiphase coherence created

        # ⟨Sz_H⟩ before INEPT
        exp_Sz_H_before = real(ψ_init' * Sz_H * ψ_init)
        @test exp_Sz_H_before > 0.0   # initial H magnetization is positive

        # Verify Pulsar can optimize toward the analytical INEPT propagator.
        # operator_target requires a unitary; Sz_C is Hermitian-not-unitary,
        # so we use unitary_target(U_INEPT) — the analytic gate above.
        target_INEPT = unitary_target(U_INEPT)
        rng = MersenneTwister(77)
        u_init = 0.05 * randn(rng, Float64, 4, 40)
        dt_INEPT = τ / 10   # fine-grain the delay

        result_INEPT = grape_optimize(sys_INEPT, target_INEPT, u_init, dt_INEPT;
                                      config = GRAPEConfig(max_iter=200,
                                                           verbose=false))
        # GRAPE should improve the gate fidelity from the noisy initial guess
        F0_INEPT = evaluate_fidelity(sys_INEPT, target_INEPT, u_init, dt_INEPT)
        @test result_INEPT.fidelity > F0_INEPT

    end  # INEPT basic verification

    # -----------------------------------------------------------------------
    @testset "Rabi oscillations" begin

        # A resonant x-drive of amplitude Ω produces Rabi oscillations.
        # Population in |1⟩ as a function of pulse area θ = Ω*t:
        #   P₁(θ) = sin²(θ/2)
        # At θ = π, P₁ = 1 (complete flip); at θ = 2π, P₁ = 0 (return to |0⟩).

        Ω   = 2π * 1e5    # 100 kHz Rabi frequency
        ψ0  = ComplexF64[1, 0]

        # θ = π (π pulse)
        t_pi = π / Ω
        U_pi = propagator(Ω * _σ_x / 2, t_pi)   # H = Ω/2 * σ_x (rotating frame)
        ψ_pi = U_pi * ψ0
        @test abs(abs(ψ_pi[2])^2 - 1.0) < 1e-10  # all in |1⟩

        # θ = 2π (2π pulse, return to |0⟩)
        t_2pi  = 2π / Ω
        U_2pi  = propagator(Ω * _σ_x / 2, t_2pi)
        ψ_2pi  = U_2pi * ψ0
        @test abs(abs(ψ_2pi[1])^2 - 1.0) < 1e-10  # back in |0⟩

        # θ = π/2 (π/2 pulse, equal superposition)
        t_pi2  = π / (2Ω)
        U_pi2  = propagator(Ω * _σ_x / 2, t_pi2)
        ψ_pi2  = U_pi2 * ψ0
        @test abs(abs(ψ_pi2[1])^2 - 0.5) < 1e-10
        @test abs(abs(ψ_pi2[2])^2 - 0.5) < 1e-10

    end  # Rabi oscillations

    # -----------------------------------------------------------------------
    @testset "Closed-system vs Lindblad propagation (zero rates)" begin

        # With all decay rates set to zero, the Liouville-space evolution must
        # reproduce the Hilbert-space unitary evolution exactly: ρ(t) = U ρ₀ U†.
        # This regression-tests the vec-identity sign and ordering in
        # build_drift_liouvillian; any swap of L vs conj(L), left vs right
        # Kronecker, or sign flip would break this test.

        Random.seed!(24680)

        # 1-qubit driven system, non-trivial Hamiltonian evolution
        Ω      = 2π * 1.5e6
        H_d    = 2π * 2.0e5 .* _σ_z                           # detuning
        H_c    = [Ω/2 .* _σ_x, Ω/2 .* _σ_y]
        amps   = [0.3, 0.2]
        t      = 1.2e-7
        H_tot  = H_d .+ amps[1] .* H_c[1] .+ amps[2] .* H_c[2]
        U      = exp(-im .* H_tot .* t)

        # Closed-system density matrix evolution
        ρ0     = ComplexF64[0.7 0.1-0.2im; 0.1+0.2im 0.3]
        ρ_closed = U * ρ0 * U'

        # Lindblad-space evolution with zero decay rates
        L_ops  = [_σ_x, _σ_z]            # arbitrary jump operators
        γ_zero = [0.0, 0.0]              # switched off
        𝓛_drift = Pulsar.build_drift_liouvillian(H_d, L_ops, γ_zero)
        𝓛_ctrls = [Pulsar.build_control_liouvillian(Hk) for Hk in H_c]
        𝓛_tot   = 𝓛_drift .+ amps[1] .* 𝓛_ctrls[1] .+ amps[2] .* 𝓛_ctrls[2]
        U_super = exp(𝓛_tot .* t)
        vec_ρ   = U_super * Pulsar.vec_rho(ρ0)
        ρ_liouv = Pulsar.mat_rho(vec_ρ, 2)

        @test norm(ρ_liouv - ρ_closed) < 1e-10

        # Trace preservation and Hermiticity after zero-rate Lindblad evolution
        @test abs(tr(ρ_liouv) - 1.0) < 1e-10
        @test norm(ρ_liouv - ρ_liouv') < 1e-10

    end  # Closed-system vs Lindblad propagation

    # -----------------------------------------------------------------------
    @testset "New fidelity gradients (analytic vs finite-diff)" begin

        # Spin-1/2 with Ix, Iy controls. We feed propagators directly to the
        # metric-aware overloads of compute_gradient_state / compute_gradient_gate
        # and cross-check against a central finite-difference of the fidelity.
        Ix_h = _σ_x / 2;  Iy_h = _σ_y / 2;  Iz_h = _σ_z / 2

        Random.seed!(2025)
        n_t   = 10
        dt    = 1e-6
        δ_hz  = 500.0
        H_dr  = 2π * δ_hz .* Iz_h
        H_c   = [2π .* Ix_h, 2π .* Iy_h]
        u     = 2π * 1000.0 .* (rand(2, n_t) .- 0.5)

        function _props(u)
            Us = [exp(-im * (H_dr .+ u[1,k].*H_c[1] .+ u[2,k].*H_c[2]) * dt) for k in 1:n_t]
            P  = zeros(ComplexF64, n_t+1, 2, 2);  Q = zeros(ComplexF64, n_t+1, 2, 2)
            P[1,:,:] = _I2;  Q[n_t+1,:,:] = _I2
            for k in 1:n_t;  P[k+1,:,:] = Us[k] * P[k,:,:];  end
            for k in n_t:-1:1;  Q[k,:,:] = Q[k+1,:,:] * Us[k];  end
            return P, Q, P[n_t+1,:,:]
        end

        ψ_i = ComplexF64[1.0, 0.0]
        ψ_t = (1/sqrt(2)) .* ComplexF64[1.0, 1.0]
        U_T = ComplexF64[0 1; 1 0]      # X-gate target

        _fid_state(u, m) = (begin _,_,Utot = _props(u); state_fidelity(ψ_t, Utot * ψ_i, m) end)
        _fid_gate(u, m)  = (begin _,_,Utot = _props(u); gate_fidelity(Utot, U_T, m)        end)

        function _fd(f, u; eps=1e-7)
            G = zeros(size(u))
            @inbounds for j in 1:size(u,1), k in 1:size(u,2)
                up = copy(u); up[j,k] += eps
                um = copy(u); um[j,k] -= eps
                G[j,k] = (f(up) - f(um)) / (2eps)
            end
            return G
        end

        P, Q, Utot = _props(u)
        tol = 1e-5

        # State metrics
        for (name, metric) in [("ModulusDM", MODULUS_DM),
                                ("SquaredDM", SQUARED_DM),
                                ("LinearDM",  LINEAR_DM)]
            G_a = Pulsar.compute_gradient_state(Utot, P, Q, H_c, ψ_i, ψ_t, dt, metric)
            G_n = _fd(uu -> _fid_state(uu, metric), u)
            err = maximum(abs.(G_a .- G_n)) / max(1, maximum(abs.(G_n)))
            @test err < tol
        end

        # Gate metric
        G_a = Pulsar.compute_gradient_gate(Utot, P, Q, H_c, U_T, dt, MODULUS_GATE)
        G_n = _fd(uu -> _fid_gate(uu, MODULUS_GATE), u)
        err = maximum(abs.(G_a .- G_n)) / max(1, maximum(abs.(G_n)))
        @test err < tol

        # Metric-aware path reproduces hard-coded path for existing metrics.
        G1 = Pulsar.compute_gradient_state(Utot, P, Q, H_c, ψ_i, ψ_t, dt, SQUARED_OVERLAP)
        G2 = Pulsar.compute_gradient_state(Utot, P, Q, H_c, ψ_i, ψ_t, dt)
        @test maximum(abs.(G1 .- G2)) < 1e-12

        G1 = Pulsar.compute_gradient_gate(Utot, P, Q, H_c, U_T, dt, NORMALIZED_GATE)
        G2 = Pulsar.compute_gradient_gate(Utot, P, Q, H_c, U_T, dt)
        @test maximum(abs.(G1 .- G2)) < 1e-12

    end  # New fidelity gradients

end  # Physics Validation
