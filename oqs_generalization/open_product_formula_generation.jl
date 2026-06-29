using ITensors, ITensorMPS
using LinearAlgebra
include("liouville_space.jl")

# =============================================================================
# NOTE ON SCOPE OF THIS FILE
# =============================================================================
#
# This file builds the open-system step channel S(dt) and its adjoint
# S(dt)^dagger entirely via DIRECT exponentiation of dense local generators
# in the vectorized Liouville space (Eq. 43 of main.pdf) -- no Kraus
# operators anywhere. This is intentional: it is the "brute-force" baseline
# requested for benchmarking, expected to be LESS efficient (larger MPO bond
# dimension and/or more expensive gate construction) than a Kraus-operator-
# based construction. The Kraus-based alternative (kraus_channel_gate, in
# liouville_space.jl) is kept available but unused here, reserved for a
# follow-up comparison once this direct route's bond-dimension behavior has
# been characterized.

# =============================================================================
# Two-site Heisenberg gates as dense unitaries (re-derived here so this file
# is self-contained; numerically identical to the RXX/RYY/RZZ op overloads in
# product_formula_generation.jl, just packaged as one combined 4x4 unitary
# per bond so we only need one call to unitary_channel_gates per bond).
# =============================================================================

function heisenberg_bond_unitary(alpha::Float64, beta::Float64)
    # exp(-i*alpha*(XX+YY)/1 ... ) composed exactly as RXX(alpha)*RYY(alpha)*RZZ(beta)
    # acting on a 2-qubit Hilbert space, basis order |00>,|01>,|10>,|11>.
    c = cos(alpha / 2)
    s = -im * sin(alpha / 2)

    RXX = ComplexF64[
        c 0 0 s
        0 c s 0
        0 s c 0
        s 0 0 c
    ]
    RYY = ComplexF64[
        c 0 0 -s
        0 c s 0
        0 s c 0
        -s 0 0 c
    ]
    RZZ = ComplexF64[
        exp(-im*beta/2) 0 0 0
        0 exp(im*beta/2) 0 0
        0 0 exp(im*beta/2) 0
        0 0 0 exp(-im*beta/2)
    ]
    return RZZ * RYY * RXX
end

# =============================================================================
# Single-qubit dissipator -- DIRECT / BRUTE-FORCE route
# =============================================================================
#
# This file deliberately builds the dissipator gate the same way it builds
# the unitary gates: by exponentiating the relevant LOCAL TERM of the dense
# vectorized Lindbladian L_vec (Eq. 43 of main.pdf) directly, with no Kraus
# decomposition anywhere. This is the "brute-force" baseline the user asked
# for, to be benchmarked against a Kraus-operator-based construction later.
#
# For a single Lindblad operator L acting on one qubit with rate gamma, the
# corresponding LOCAL super-operator term of Eq. 43 (restricting to just
# this dissipator, dropping the Hamiltonian part which is handled by the
# separate unitary layers) is the dense 4x4 matrix
#
#   L_diss = gamma * ( conj(L) ⊗ L  -  (1/2) * Id ⊗ (L^† L)  -  (1/2) * (L^† L)^T ⊗ Id )
#
# acting on the same (bra,ket)-stacked 4-dimensional space used everywhere
# else in this file (bra slower, ket faster -- see liouville_space.jl). The
# one-step dissipator CHANNEL over a time interval dt is then literally
#
#   S_diss(dt) = exp(dt * L_diss)
#
# i.e. a dense 4x4 matrix exponential, with NO Kraus operators involved at
# any point. This is exact for a single isolated dissipator term (no
# Trotter splitting needed within the dissipator itself), and is the direct
# open-system analogue of how the unitary bond gates are built by
# exponentiating their own generator (here we just write the result of that
# exponentiation in closed form via heisenberg_bond_unitary instead of
# calling `exp`, but for the dissipator there is no comparably simple closed
# form in general, so we call `exp` on the dense generator directly).
#
# Generalizes to ANY single-qubit Lindblad operator L (not just amplitude
# damping), since L_diss above is well-defined for any 2x2 matrix L.

function single_qubit_dissipator_generator(L::AbstractMatrix, gamma::Float64)
    @assert size(L) == (2, 2) "single_qubit_dissipator_generator only supports single-qubit (2x2) Lindblad operators."
    Ld = Matrix{ComplexF64}(L)
    LdL = Ld' * Ld
    Id2_ = Matrix{ComplexF64}(I, 2, 2)
    return gamma .* (kron(conj(Ld), Ld) .- 0.5 .* kron(Id2_, LdL) .- 0.5 .* kron(transpose(LdL), Id2_))
end

# Forward dissipator channel over time dt, as a dense 4x4 matrix, built by
# directly exponentiating the generator above -- the brute-force route.
function single_qubit_dissipator_channel(L::AbstractMatrix, gamma::Float64, dt::Float64)
    return exp(dt .* single_qubit_dissipator_generator(L, gamma))
end

# Turn a dense 4x4 superoperator matrix (in the bra-slow/ket-fast convention
# used throughout this file) into the corresponding ITensor gate at physical
# site j. Used for both the forward channel and (with Sdag = S' passed in)
# the adjoint -- see dissipator_gate_from_matrix below.
function dissipator_gate_from_matrix(S::AbstractMatrix, j::Int, lsites::LiouvilleSites)
    d = 2
    ket_s = lsites.ket[j]
    bra_s = lsites.bra[j]
    Stensor = reshape(Matrix{ComplexF64}(S), d, d, d, d)  # (ket_out, bra_out, ket_in, bra_in)
    return ITensor(Stensor, ket_s', bra_s', ket_s, bra_s)
end

# Forward dissipator gate at site j, amplitude-damping case (L = sigma_-),
# built via direct exponentiation of the dense generator -- no Kraus
# operators anywhere in this construction.
function amplitude_damping_gate(gamma::Float64, dt::Float64, j::Int, lsites::LiouvilleSites)
    S = single_qubit_dissipator_channel(SIGMA_MINUS, gamma, dt)
    return dissipator_gate_from_matrix(S, j, lsites)
end

# The *adjoint* (Hilbert-Schmidt sense) of the dissipator channel. As noted
# in the project notes (Section 2), this is NOT itself a physical channel
# once gamma > 0 (S^dagger != S^{-1}), so it is built directly as the
# conjugate transpose of the dense forward-channel matrix -- exactly the
# operator analogue of `swapprime(dag(.))` used for closed-system MPOs, but
# applied here to a genuinely non-unitary dense matrix.
function amplitude_damping_dag_gate(gamma::Float64, dt::Float64, j::Int, lsites::LiouvilleSites)
    S = single_qubit_dissipator_channel(SIGMA_MINUS, gamma, dt)
    Sdag = S'   # conjugate transpose of the dense 4x4 superoperator matrix
    return dissipator_gate_from_matrix(Sdag, j, lsites)
end

# =============================================================================
# Layer generation (edge-coloring), Liouville-space version
# =============================================================================

function odd_layer_channel_gates(n, J, dt, lsites::LiouvilleSites)
    alpha = [x * dt for x in J]
    beta  = [2 * x * dt for x in J]
    gates = ITensor[]
    for j in 1:2:n-1
        Umat = heisenberg_bond_unitary(alpha[j], beta[j])
        append!(gates, unitary_channel_gates(Umat, [j, j+1], lsites))
    end
    return gates
end

function even_layer_channel_gates(n, J, dt, lsites::LiouvilleSites)
    alpha = [x * dt for x in J]
    beta  = [2 * x * dt for x in J]
    gates = ITensor[]
    if n > 2
        for j in 2:2:n-1
            Umat = heisenberg_bond_unitary(alpha[j], beta[j])
            append!(gates, unitary_channel_gates(Umat, [j, j+1], lsites))
        end
    end
    return gates
end

function dissipator_layer_channel_gates(n, gammas, dt, lsites::LiouvilleSites)
    gates = ITensor[]
    for j in 1:n
        gammas[j] == 0.0 && continue
        push!(gates, amplitude_damping_gate(gammas[j], dt, j, lsites))
    end
    return gates
end

function dissipator_layer_channel_gates_dag(n, gammas, dt, lsites::LiouvilleSites)
    gates = ITensor[]
    for j in 1:n
        gammas[j] == 0.0 && continue
        push!(gates, amplitude_damping_dag_gate(gammas[j], dt, j, lsites))
    end
    return gates
end

# =============================================================================
# Product formulas combining Hamiltonian + dissipator layers
# =============================================================================
#
# Order-1 (Lie-Trotter): unitary odd/even layers followed by one combined
# dissipator layer per step, matching the structure of Eq. 33 in main.pdf,
#   S(dt) = e^{dt L0} e^{dt L1} ... e^{dt L_{m-1}}
# with each L_alpha taken to be either a two-site Heisenberg bond term or a
# single-site dissipator.
#
# Order-2 (symmetric Trotter / Strang splitting): half-step odd unitary
# layer, full-step even unitary layer, full-step dissipator layer, half-step
# odd unitary layer again. This is the direct open-system analogue of
# get_step_gates_order2 in product_formula_generation.jl.

function get_open_step_gates(n, J, gammas, dt, lsites::LiouvilleSites; order::Int=1)
    if order == 1
        return get_open_step_gates_order1(n, J, gammas, dt, lsites)
    elseif order == 2
        return get_open_step_gates_order2(n, J, gammas, dt, lsites)
    else
        error("Unsupported open product-formula order = $order. Use 1 or 2.")
    end
end

function get_open_step_gates_order1(n, J, gammas, dt, lsites::LiouvilleSites)
    return vcat(
        odd_layer_channel_gates(n, J, dt, lsites),
        even_layer_channel_gates(n, J, dt, lsites),
        dissipator_layer_channel_gates(n, gammas, dt, lsites),
    )
end

function get_open_step_gates_order2(n, J, gammas, dt, lsites::LiouvilleSites)
    return vcat(
        odd_layer_channel_gates(n, J, dt/2, lsites),
        even_layer_channel_gates(n, J, dt, lsites),
        dissipator_layer_channel_gates(n, gammas, dt, lsites),
        odd_layer_channel_gates(n, J, dt/2, lsites),
    )
end

# =============================================================================
# Adjoint product formula
# =============================================================================
#
# IMPORTANT (see project notes, Section 2): for the open system, the adjoint
# of the *full* step S(dt) is NOT simply "run the same gate list backwards
# with dt -> -dt", because (a) the dissipator gate is not unitary, so
# S_dissipator(dt)^{-1} != S_dissipator(dt)^dagger, and (b) operator adjoint
# of a product reverses order and adjoints each factor:
#   (ABC)^† = C^† B^† A^†
# So we build S(dt)^† gate-by-gate: take the SAME gate list used to build
# the forward step (in the same order), reverse it, and replace each gate by
# its own Hilbert-Schmidt adjoint (unitary layers: U -> U^† via dt -> -dt,
# which IS still valid since U is unitary; dissipator layers: replaced by
# the genuinely different non-CP adjoint built in
# amplitude_damping_dag_gate, NOT by reversing gamma*dt).

function get_open_step_gates_dag(n, J, gammas, dt, lsites::LiouvilleSites; order::Int=1)
    if order == 1
        return get_open_step_gates_order1_dag(n, J, gammas, dt, lsites)
    elseif order == 2
        return get_open_step_gates_order2_dag(n, J, gammas, dt, lsites)
    else
        error("Unsupported open product-formula order = $order. Use 1 or 2.")
    end
end

function get_open_step_gates_order1_dag(n, J, gammas, dt, lsites::LiouvilleSites)
    fwd_unitary_part = vcat(
        odd_layer_channel_gates(n, J, dt, lsites),
        even_layer_channel_gates(n, J, dt, lsites),
    )
    unitary_dag = reverse([unitary_channel_gate_dag(g) for g in fwd_unitary_part])
    diss_dag = dissipator_layer_channel_gates_dag(n, gammas, dt, lsites)
    # (U_odd U_even D)^† = D^† U_even^† U_odd^†
    return vcat(diss_dag, unitary_dag)
end

function get_open_step_gates_order2_dag(n, J, gammas, dt, lsites::LiouvilleSites)
    odd_half_1 = odd_layer_channel_gates(n, J, dt/2, lsites)
    even_full  = even_layer_channel_gates(n, J, dt, lsites)
    diss_full  = dissipator_layer_channel_gates(n, gammas, dt, lsites)
    odd_half_2 = odd_layer_channel_gates(n, J, dt/2, lsites)

    odd_half_1_dag = reverse([unitary_channel_gate_dag(g) for g in odd_half_1])
    even_full_dag  = reverse([unitary_channel_gate_dag(g) for g in even_full])
    diss_full_dag  = dissipator_layer_channel_gates_dag(n, gammas, dt, lsites)
    odd_half_2_dag = reverse([unitary_channel_gate_dag(g) for g in odd_half_2])

    # (A B C A')^† = A'^† C^† B^† A^†
    return vcat(odd_half_2_dag, diss_full_dag, even_full_dag, odd_half_1_dag)
end

# Helper: Hilbert-Schmidt adjoint of a single unitary-channel ITensor gate.
# Because unitary_channel_gates returns *two* gates (one acting on ket legs
# with U, one on bra legs with conj(U)), and (U)^† = U^† while
# (conj(U))^† = conj(U^†) = conj(U)^†, the adjoint of each individual gate
# tensor is obtained simply by complex-conjugating its matrix elements
# and swapping primed/unprimed indices (dag + swapprime), exactly as for an
# ordinary unitary ITensor MPO tensor -- no need to actually recompute U^†
# from alpha,beta, we just dagger the ITensor directly.
function unitary_channel_gate_dag(g::ITensor)
    return swapprime(dag(g), 0 => 1)
end

# =============================================================================
# Building the step MPO by contracting the gate list with the identity
# =============================================================================
#
# Direct analogue of get_step_MPO / get_step_MPO_dag in
# product_formula_generation.jl. Each call to `apply` truncates the MPO via
# SVD with the given cutoff/maxdim, exactly as in the closed-system code.

function get_open_step_MPO(n, J, gammas, dt, lsites::LiouvilleSites, cutoff, maxdim; order::Int=1)
    gates = get_open_step_gates(n, J, gammas, dt, lsites; order=order)
    S = identity_liouville_mpo(lsites)
    S = apply(gates, S; cutoff=cutoff, maxdim=maxdim)
    return S
end

function get_open_step_MPO_dag(n, J, gammas, dt, lsites::LiouvilleSites, cutoff, maxdim; order::Int=1)
    gates_dag = get_open_step_gates_dag(n, J, gammas, dt, lsites; order=order)
    S = identity_liouville_mpo(lsites)
    S = apply(gates_dag, S; cutoff=cutoff, maxdim=maxdim)
    return S
end
