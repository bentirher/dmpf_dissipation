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

# Faithful copy of build_open_F's clock-synchronized contraction, with the ONLY
# change being that the left/bra operator is the INVERSE step MPO (step_i_inv)
# instead of the adjoint (step_i_dag). The while-loop clock sync and the final
# inner(rho0', F, rho0) contraction are byte-identical to the library, so at
# gamma=0 this MUST reproduce the S^dag result exactly (built-in check).
function build_open_F_inv_between(gammas, ks_left, ks_right, diss)
    Id = identity_liouville_mpo(lsites)
    comps = MPO[]
    for i in eachindex(ks_left)
        dt_i = t / ks_left[i]
        step_i_inv = get_open_step_MPO_inv(n, J, gammas, dt_i, lsites, cutoff, maxdim; dissipation=diss)
        for j in eachindex(ks_right)
            dt_j = t / ks_right[j]
            step_j = get_open_step_MPO(n, J, gammas, dt_j, lsites, cutoff, maxdim; order=order, dissipation=diss)
            F = deepcopy(Id)
            time_i = 0.0; time_j = 0.0
            while (time_i < t - 1e-12) || (time_j < t - 1e-12)
                if (time_j <= time_i) && (time_j < t - 1e-12)
                    F = right_multiply(F, step_j; cutoff=cutoff, maxdim=maxdim)
                    time_j += dt_j
                elseif time_i < t - 1e-12
                    F = left_multiply(step_i_inv, F; cutoff=cutoff, maxdim=maxdim)
                    time_i += dt_i
                end
            end
            push!(comps, F)
        end
    end
    return comps
end

function coeffs_for_gamma(gamma)
    gammas = fill(gamma, n)
    diss = gamma > 0.0
    r = length(ks)

    # correct S^dag route via existing library routines
    M_dag, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
    # NB: order_ref=2 here (not the library default of 1) so it matches the
    # order-2 inverse reference below -- this isolates the dag-vs-inverse
    # difference at gamma=0 rather than an order-1-vs-2 Trotter mismatch.
    L_dag, _ = open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
                             cutoff=cutoff, maxdim=maxdim, order=order, order_ref=2, dissipation=diss)
    c_dag, lam_dag = dynamic_mpf_coefficients(M_dag, L_dag)

    # S^{-1} route (F'): identical contraction, inverse operator on the bra side
    # M_inv: ks x ks   (mirrors build_open_F -> open_gram_matrix)
    Ms = build_open_F_inv_between(gammas, ks, ks, diss)
    M_inv = zeros(Float64, r, r)
    idx = 1
    for i in 1:r, j in 1:r
        M_inv[i, j] = real(inner(rho0', Ms[idx], rho0)); idx += 1
    end
    # L_inv: [k_ref] x ks   (mirrors build_open_F_between_lists -> open_L_vector)
    Ls = build_open_F_inv_between(gammas, [k_ref_opt], ks, diss)
    L_inv = [real(inner(rho0', Ls[j], rho0)) for j in 1:r]

    c_inv, lam_inv = dynamic_mpf_coefficients(M_inv, L_inv)

    cond_dag = cond(M_dag)
    cond_inv = cond(M_inv)
    return c_dag, c_inv, M_dag, M_inv, cond_dag, cond_inv, lam_dag, lam_inv
end

outdir = "sinv_scan_results"; mkpath(outdir)
rows = []

for gamma in gamma_values
    println("[$(now())] gamma=$gamma ..."); flush(stdout)
    local c_dag, c_inv, M_dag, M_inv, cond_dag, cond_inv, lam_dag, lam_inv
    t_g = @elapsed begin
        c_dag, c_inv, M_dag, M_inv, cond_dag, cond_inv, lam_dag, lam_inv = coeffs_for_gamma(gamma)
    end
    maxdiff = maximum(abs.(c_dag .- c_inv))
    push!(rows, (gamma=gamma, c_dag=c_dag, c_inv=c_inv,
                 maxdiff=maxdiff, meanMdag=sum(M_dag)/length(M_dag),
                 meanMinv=sum(M_inv)/length(M_inv),
                 cond_dag=cond_dag, cond_inv=cond_inv,
                 lam_dag=lam_dag, lam_inv=lam_inv))
    println("[$(now())] gamma=$gamma done ($(round(t_g,digits=1))s): " *
            "c_dag=$(round.(c_dag,digits=4)) c_inv=$(round.(c_inv,digits=4)) maxdiff=$(round(maxdiff,digits=5))")
    println("           cond(M_dag)=$(round(cond_dag,digits=1)) cond(M_inv)=$(round(cond_inv,digits=1)) " *
            "lambda_dag=$(round(lam_dag,sigdigits=4)) lambda_inv=$(round(lam_inv,sigdigits=4))")
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
    println(io, "gamma,c_dag_1,c_dag_2,c_inv_1,c_inv_2,max_coeff_diff,mean_M_dag,mean_M_inv,cond_M_dag,cond_M_inv,lambda_dag,lambda_inv")
    for r in rows
        println(io, "$(r.gamma),$(r.c_dag[1]),$(r.c_dag[2]),$(r.c_inv[1]),$(r.c_inv[2])," *
                    "$(r.maxdiff),$(r.meanMdag),$(r.meanMinv),$(r.cond_dag),$(r.cond_inv),$(r.lam_dag),$(r.lam_inv)")
    end
end

println("\n==== GAMMA SCAN SUMMARY (n=$n, maxdim=$maxdim, ks=$ks) ====")
println("gamma   | max|dc| | cond(M_dag) | cond(M_inv) | mean(M_dag) | mean(M_inv)")
for r in rows
    println("$(rpad(r.gamma,7)) | $(rpad(round(r.maxdiff,digits=3),7)) | " *
            "$(rpad(round(r.cond_dag,digits=1),11)) | $(rpad(round(r.cond_inv,digits=1),11)) | " *
            "$(rpad(round(r.meanMdag,digits=4),11)) | $(round(r.meanMinv,digits=4))")
end
println("\n[$(now())] wrote sinv_scan_results/gamma_scan.csv"); flush(stdout)
