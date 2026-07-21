# sinv_gamma_scan.jl
# Cheap scan (n=4, maxdim=64) of how the DMPF coefficients from S^dag (correct)
# and S^{-1} (F') diverge as the dissipation rate gamma increases from 0 to 0.05.
#
# At gamma=0 the two constructions are IDENTICAL (empty dissipator layer), so
# c_dag == c_inv there -- used as a built-in correctness check.
#
# Writes sinv_scan_results/gamma_scan.csv with, per gamma:
#   gamma, c_dag..., c_inv..., max_abs_coeff_diff, mean_M_dag, mean_M_inv
#
# Usage: julia sinv_gamma_scan.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

# ---- inverse step-MPO machinery (mirrors get_open_step_gates_order2_dag, but
#      dissipator adjoint S' replaced by dissipator inverse inv(S)) ----
function amplitude_damping_inv_gate(gamma::Float64, dt::Float64, j::Int, lsites::LiouvilleSites)
    S = single_qubit_dissipator_channel(SIGMA_MINUS, gamma, dt)
    return dissipator_gate_from_matrix(inv(S), j, lsites)
end
function dissipator_layer_channel_gates_inv(n, gammas, dt, lsites::LiouvilleSites)
    gates = ITensor[]
    for j in 1:n
        gammas[j] == 0.0 && continue
        push!(gates, amplitude_damping_inv_gate(gammas[j], dt, j, lsites))
    end
    return gates
end
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
function get_open_step_MPO_inv(n, J, gammas, dt, lsites::LiouvilleSites, cutoff, maxdim; dissipation::Bool=true)
    eff = dissipation ? gammas : zeros(length(gammas))
    gates = get_open_step_gates_order2_inv(n, J, eff, dt, lsites)
    return apply(gates, identity_liouville_mpo(lsites); cutoff=cutoff, maxdim=maxdim)
end

# ---- fixed setup (cheap: n=4, maxdim=64) ----
Random.seed!(1234)
n      = 4
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
t      = 3.0
lsites = liouville_siteinds(n)
ks         = [3, 8]
k_ref_opt  = 40
maxdim     = 64
cutoff     = 1e-12
order      = 2

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

gamma_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

# sandwich <<rho0| left^a right^b |rho0>>
function sandwich(left_mpo, a, right_mpo, b)
    F = identity_liouville_mpo(lsites)
    for _ in 1:b; F = right_multiply(F, right_mpo; cutoff=cutoff, maxdim=maxdim); end
    for _ in 1:a; F = left_multiply(left_mpo, F;   cutoff=cutoff, maxdim=maxdim); end
    return real(inner(rho0', F, rho0))
end

function coeffs_for_gamma(gamma)
    gammas = fill(gamma, n)
    diss = gamma > 0.0

    # correct S^dag route via existing routines
    M_dag, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
    L_dag, _ = open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
                             cutoff=cutoff, maxdim=maxdim, order=order, order_ref=1, dissipation=diss)
    c_dag, _ = dynamic_mpf_coefficients(M_dag, L_dag)

    # S^{-1} route (F')
    fwd = Dict(k => get_open_step_MPO(n, J, gammas, t/k, lsites, cutoff, maxdim;
                                      order=order, dissipation=diss) for k in ks)
    invs = Dict(k => get_open_step_MPO_inv(n, J, gammas, t/k, lsites, cutoff, maxdim;
                                           dissipation=diss) for k in ks)
    inv_ref = get_open_step_MPO_inv(n, J, gammas, t/k_ref_opt, lsites, cutoff, maxdim; dissipation=diss)

    r = length(ks)
    M_inv = zeros(Float64, r, r); L_inv = zeros(Float64, r)
    for i in 1:r, j in 1:r
        M_inv[i,j] = sandwich(invs[ks[i]], ks[i], fwd[ks[j]], ks[j])
    end
    for j in 1:r
        L_inv[j] = sandwich(inv_ref, k_ref_opt, fwd[ks[j]], ks[j])
    end
    c_inv, _ = dynamic_mpf_coefficients(M_inv, L_inv)

    return c_dag, c_inv, M_dag, M_inv
end

outdir = "sinv_scan_results"; mkpath(outdir)
rows = []

for gamma in gamma_values
    println("[$(now())] gamma=$gamma ..."); flush(stdout)
    t_g = @elapsed begin
        c_dag, c_inv, M_dag, M_inv = coeffs_for_gamma(gamma)
    end
    maxdiff = maximum(abs.(c_dag .- c_inv))
    push!(rows, (gamma=gamma, c_dag=c_dag, c_inv=c_inv,
                 maxdiff=maxdiff, meanMdag=sum(M_dag)/length(M_dag),
                 meanMinv=sum(M_inv)/length(M_inv)))
    println("[$(now())] gamma=$gamma done ($(round(t_g,digits=1))s): " *
            "c_dag=$(round.(c_dag,digits=4)) c_inv=$(round.(c_inv,digits=4)) maxdiff=$(round(maxdiff,digits=5))")
    flush(stdout)

    # correctness check at gamma=0
    if gamma == 0.0
        if maxdiff < 1e-8
            println("   [check] gamma=0: c_dag == c_inv as expected (maxdiff=$maxdiff).")
        else
            println("   [WARNING] gamma=0 but coefficients differ by $maxdiff -- construction mismatch!")
        end
    end
end

open(joinpath(outdir, "gamma_scan.csv"), "w") do io
    println(io, "gamma,c_dag_1,c_dag_2,c_inv_1,c_inv_2,max_coeff_diff,mean_M_dag,mean_M_inv")
    for r in rows
        println(io, "$(r.gamma),$(r.c_dag[1]),$(r.c_dag[2]),$(r.c_inv[1]),$(r.c_inv[2])," *
                    "$(r.maxdiff),$(r.meanMdag),$(r.meanMinv)")
    end
end

println("\n==== GAMMA SCAN SUMMARY (n=$n, maxdim=$maxdim, ks=$ks) ====")
println("gamma   | max|c_dag-c_inv| | mean(M_dag) | mean(M_inv)")
for r in rows
    println("$(rpad(r.gamma,7)) | $(rpad(round(r.maxdiff,digits=5),16)) | " *
            "$(rpad(round(r.meanMdag,digits=4),11)) | $(round(r.meanMinv,digits=4))")
end
println("\n[$(now())] wrote sinv_scan_results/gamma_scan.csv"); flush(stdout)
