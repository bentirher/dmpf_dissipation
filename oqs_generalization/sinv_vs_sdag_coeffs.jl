# sinv_vs_sdag_coeffs.jl
# Tests whether F' (built with S^{-1} on the bra side) yields DMPF coefficients
# close to F (built with the correct adjoint S^dag), at n=5.
#
# Physics: F and F' are IDENTICAL in the Hamiltonian sector (unitary => S^dag =
# S^{-1}). They differ ONLY in the dissipator gates:
#   F  uses  S_diss^dag  = (dense forward channel)'      [correct, physical]
#   F' uses  S_diss^{-1} = inv(dense forward channel)    [unphysical time-reversal]
# Since exp(-dt L)exp(+dt L)=I exactly, F' should recover the near-identity
# cancellation the closed system enjoys -- the hypothesis is that F' has lower
# bond growth AND, if we're lucky, nearly the same fitted coefficients.
#
# This run focuses on COEFFICIENT AGREEMENT (the crux). It builds M and L both
# ways and compares the resulting c. It also reports F' bond growth and a
# conditioning number for the composed inverse dissipator, as secondary signals.
#
# Usage: julia sinv_vs_sdag_coeffs.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

# =============================================================================
# Local INVERSE step-MPO construction: mirror of get_open_step_gates_dag, but
# with the dissipator adjoint (S') replaced by the dissipator inverse inv(S).
# Unitary layers are unchanged (their dt->-dt adjoint already equals inverse).
# =============================================================================

# inverse dissipator gate: inv(forward channel) instead of forward'
function amplitude_damping_inv_gate(gamma::Float64, dt::Float64, j::Int, lsites::LiouvilleSites)
    S = single_qubit_dissipator_channel(SIGMA_MINUS, gamma, dt)
    Sinv = inv(S)                    # <-- the only change vs the _dag gate (which uses S')
    return dissipator_gate_from_matrix(Sinv, j, lsites)
end

function dissipator_layer_channel_gates_inv(n, gammas, dt, lsites::LiouvilleSites)
    gates = ITensor[]
    for j in 1:n
        gammas[j] == 0.0 && continue
        push!(gates, amplitude_damping_inv_gate(gammas[j], dt, j, lsites))
    end
    return gates
end

# order-2 inverse gate list, mirroring get_open_step_gates_order2_dag exactly
# but with diss_full_dag -> diss_full_inv
function get_open_step_gates_order2_inv(n, J, gammas, dt, lsites::LiouvilleSites)
    odd_half_1 = odd_layer_channel_gates(n, J, dt/2, lsites)
    even_full  = even_layer_channel_gates(n, J, dt, lsites)
    odd_half_2 = odd_layer_channel_gates(n, J, dt/2, lsites)

    odd_half_1_dag = reverse([unitary_channel_gate_dag(g) for g in odd_half_1])
    even_full_dag  = reverse([unitary_channel_gate_dag(g) for g in even_full])
    diss_full_inv  = dissipator_layer_channel_gates_inv(n, gammas, dt, lsites)
    odd_half_2_dag = reverse([unitary_channel_gate_dag(g) for g in odd_half_2])

    return vcat(odd_half_2_dag, diss_full_inv, even_full_dag, odd_half_1_dag)
end

function get_open_step_MPO_inv(n, J, gammas, dt, lsites::LiouvilleSites, cutoff, maxdim;
                               dissipation::Bool=true)
    eff = dissipation ? gammas : zeros(length(gammas))
    gates = get_open_step_gates_order2_inv(n, J, eff, dt, lsites)
    S = identity_liouville_mpo(lsites)
    return apply(gates, S; cutoff=cutoff, maxdim=maxdim)
end

# conditioning signal: how much does the DENSE single-site inverse dissipator
# amplify, and how does that compound over k steps?
function inverse_conditioning(gamma, t, k)
    dt = t / k
    S = single_qubit_dissipator_channel(SIGMA_MINUS, gamma, dt)
    Sinv = inv(S)
    smax = maximum(svdvals(Sinv))       # per-step amplification
    return smax, smax^k                 # per-step, and compounded over k steps
end

# =============================================================================
# Build F and F' Gram matrices "by hand" so both share identical structure,
# differing only in which reference/adjoint operator is used on the bra side.
# We compute the DMPF coefficients from the OPT-level (cheap) M and L, exactly
# as the sweep does, at a single candidate maxdim.
# =============================================================================

Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(0.05, n)
t      = 3.0
lsites = liouville_siteinds(n)
ks         = [3, 8]
k_ref_opt  = 40
maxdim     = 256          # generous; coefficients come from the opt step
cutoff     = 1e-12
order      = 2

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# --- conditioning report (cheap, do first) ---
println("\n=== inverse-dissipator conditioning (per-step smax, compounded ^k) ===")
for kk in vcat(ks, k_ref_opt)
    smax, comp = inverse_conditioning(0.05, t, kk)
    println("  k=$kk: per-step smax=$(round(smax,digits=4))  compounded=$(round(comp,digits=3))")
end
flush(stdout)

# --- coefficients the CORRECT way (S^dag), via existing routines ---
println("\n[$(now())] building M_opt, L_opt with S^dag (correct)..."); flush(stdout)
M_dag, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=cutoff, maxdim=maxdim, order=order, dissipation=true)
L_dag, _ = open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
                         cutoff=cutoff, maxdim=maxdim, order=order, order_ref=1,
                         dissipation=true)
c_dag, _ = dynamic_mpf_coefficients(M_dag, L_dag)
println("[$(now())] c_dag = $c_dag"); flush(stdout)

# --- coefficients the S^{-1} way: rebuild M and L using the inverse step MPO ---
# We reproduce the F_ij = [S_ki^{inv}] [S_kj] sandwich structure by evolving the
# bra side with the inverse step MPO. Simplest faithful route: build each F_ij
# as <<rho0| (S_ki^{inv})^{ki} (S_kj)^{kj} |rho0>> using explicit MPO products,
# mirroring build_open_F but with the inverse on the left.
function gram_and_L_inv(ks, k_ref)
    r = length(ks)
    Minv = zeros(Float64, r, r)
    Linv = zeros(Float64, r)
    # forward step MPOs (correct forward channel) for each k
    fwd = Dict(k => get_open_step_MPO(n, J, gammas, t/k, lsites, cutoff, maxdim;
                                      order=order, dissipation=true) for k in ks)
    invs = Dict(k => get_open_step_MPO_inv(n, J, gammas, t/k, lsites, cutoff, maxdim;
                                           dissipation=true) for k in ks)
    inv_ref = get_open_step_MPO_inv(n, J, gammas, t/k_ref, lsites, cutoff, maxdim;
                                    dissipation=true)

    # helper: <<rho0| A^{a} B^{b} |rho0>> via middle-out-style accumulation
    function sandwich(left_mpo, a, right_mpo, b)
        F = identity_liouville_mpo(lsites)
        # apply right (ket side) b times, left (bra side) a times, lockstep where possible
        for _ in 1:b; F = right_multiply(F, right_mpo; cutoff=cutoff, maxdim=maxdim); end
        for _ in 1:a; F = left_multiply(left_mpo, F;   cutoff=cutoff, maxdim=maxdim); end
        return real(inner(rho0', F, rho0))
    end

    for i in 1:r, j in 1:r
        Minv[i,j] = sandwich(invs[ks[i]], ks[i], fwd[ks[j]], ks[j])
    end
    for j in 1:r
        Linv[j] = sandwich(inv_ref, k_ref, fwd[ks[j]], ks[j])
    end
    return Minv, Linv
end

println("\n[$(now())] building M, L with S^{-1} (F')..."); flush(stdout)
M_inv, L_inv = gram_and_L_inv(ks, k_ref_opt)
c_inv, _ = dynamic_mpf_coefficients(M_inv, L_inv)
println("[$(now())] c_inv = $c_inv"); flush(stdout)

# =============================================================================
# compare
# =============================================================================
println("\n==== COEFFICIENT COMPARISON (n=$n, gamma=0.05, ks=$ks) ====")
println("c_dag (correct, S^dag) = $c_dag")
println("c_inv (S^{-1}, F')     = $c_inv")
println("abs difference         = $(abs.(c_dag .- c_inv))")
println("max abs difference     = $(maximum(abs.(c_dag .- c_inv)))")
println()
println("M_dag = $M_dag")
println("M_inv = $M_inv")
println("L_dag = $L_dag")
println("L_inv = $L_inv")
println("\n[$(now())] done."); flush(stdout)
