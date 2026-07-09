# time_ref.jl
# Times each expensive building block of ONE maxdim_truncation_sweep task
# separately, so we can see exactly where the wall-clock time goes before
# committing to a full production sweep.
#
# Usage: julia time_ref.jl
#
# Each block is wrapped so that even if the job is killed mid-way, the log
# shows which blocks completed and how long they took.

import Distributions
import Random
include("spectrum_truncation_analysis.jl")   # full include chain

using LinearAlgebra
using Dates

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads(), " / Julia CPU threads: ", Sys.CPU_THREADS)
flush(stdout)

# --- setup, identical to run_sweep_case.jl ---
Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(0.05, n)
t      = 3.0
lsites = liouville_siteinds(n)
ks         = [3, 8]
k_ref_opt  = 40
k_ref_eval = 50
maxdim_ref = 256
cutoff     = 1e-12
initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

md = 16   # cheapest candidate; the maxdim=256 reference blocks dominate anyway

function timed(f, label)
    println("[$(now())] START  $label"); flush(stdout)
    dt = @elapsed (val = f())
    println("[$(now())] DONE   $label  ($(round(dt, digits=1))s)"); flush(stdout)
    return val
end
 
# 1) The sweep's own reference gram at maxdim=256 (built once per task)
timed("M_ref_full  open_gram_matrix(maxdim=256)") do
    open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=2, dissipation=true)
end
 
# 2) M_opt at the candidate maxdim (cheap: small md)
timed("M_opt       open_gram_matrix(maxdim=$md)") do
    open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff, maxdim=md, order=2, dissipation=true)
end
 
# 3) L_opt at candidate maxdim, k_ref_opt=40 (small md, but ~40 layers)
timed("L_opt       open_L_vector(maxdim=$md, k_ref=$k_ref_opt)") do
    open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
        cutoff=cutoff, maxdim=md, order=2, order_ref=1, dissipation=true)
end
 
# 4) M_ref inside test_dynamic_mpf_open: SECOND maxdim=256 gram (recomputed)
timed("M_ref       open_gram_matrix(maxdim=256)  [recompute]") do
    open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=2, dissipation=true)
end
 
# 5) L_ref: the suspected monster -- ~50 layers at chi=256
timed("L_ref       open_L_vector(maxdim=256, k_ref=$k_ref_eval)") do
    open_L_vector(n, J, gammas, t, ks, k_ref_eval, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=2, order_ref=2, dissipation=true)
end
 
# 6) reference_purity: 50 MPS applies at maxdim=256 (cheaper, MPS not MPO)
timed("purity      reference_purity(maxdim=256, k_ref=$k_ref_eval)") do
    reference_purity(n, J, gammas, t, k_ref_eval, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=2, dissipation=true)
end
 
println("[$(now())] ALL BLOCKS COMPLETE"); flush(stdout)
