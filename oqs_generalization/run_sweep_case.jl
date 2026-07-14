# run_sweep_case.jl
# De-duplicated maxdim truncation sweep for the open-system DMPF study.
#
# Usage: julia run_sweep_case.jl
#
# KEY OPTIMIZATION vs. maxdim_truncation_sweep:
#   The original sweep rebuilt the expensive maxdim_ref Gram matrix (M_ref),
#   L_ref, and the purity ONCE PER md -- but those three objects do not depend
#   on md at all. With 4 md values that meant ~5-6 redundant ~70-min Gram builds
#   per task. Here they are computed exactly ONCE and reused for every md.
#
#   Only the cheap, md-dependent pieces (M_opt, L_opt, coefficients) run inside
#   the md loop.
#
# maxdim_ref stays at 256: the Gram-matrix convergence test showed M is still
# drifting at the ~0.2% level between 192 and 256, so a lower reference would be
# under-converged. md=256 is dropped from the candidate list because comparing a
# 256 candidate against a 256 reference is degenerate.
#
# All primitives are already in scope via the include chain -- no project files
# are edited.

import Distributions
import Random
include("spectrum_truncation_analysis.jl")   # full include chain

using LinearAlgebra
using Dates

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads(), " / Julia CPU threads: ", Sys.CPU_THREADS)
flush(stdout)

# --- setup, identical seed/params to the notebook cell ---
Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(0.05, n)
t      = 3.0
lsites = liouville_siteinds(n)

ks             = [3, 8]
k_ref_opt      = 40
k_ref_eval     = 50
maxdim_ref     = 256
cutoff         = 1e-12
order          = 2
order_ref_opt  = 1
order_ref_eval = 2

maxdims = [16, 32, 64, 128]     # md=256 dropped: degenerate vs a 256 reference

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# =============================================================================
# md-INDEPENDENT reference objects -- computed ONCE (this is the expensive part)
# =============================================================================

println("[$(now())] building reference M_ref (maxdim=$maxdim_ref)..."); flush(stdout)
t_Mref = @elapsed begin
    M_ref, _ = open_gram_matrix(
        n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=order, dissipation=true
    )
end
println("[$(now())] M_ref done ($(round(t_Mref, digits=1))s)"); flush(stdout)

println("[$(now())] building reference L_ref (maxdim=$maxdim_ref, k_ref=$k_ref_eval)..."); flush(stdout)
t_Lref = @elapsed begin
    L_ref, _ = open_L_vector(
        n, J, gammas, t, ks, k_ref_eval, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=order, order_ref=order_ref_eval,
        dissipation=true
    )
end
println("[$(now())] L_ref done ($(round(t_Lref, digits=1))s)"); flush(stdout)

println("[$(now())] building reference purity (maxdim=$maxdim_ref, k_ref=$k_ref_eval)..."); flush(stdout)
t_pur = @elapsed begin
    purity = reference_purity(
        n, J, gammas, t, k_ref_eval, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=order_ref_eval, dissipation=true
    )
end
println("[$(now())] purity done ($(round(t_pur, digits=1))s)  value=$purity"); flush(stdout)

# =============================================================================
# md-DEPENDENT loop -- only the cheap pieces run here
# =============================================================================

results = NamedTuple[]

for md in maxdims
    global results
    println("[$(now())] --- md=$md ---"); flush(stdout)

    t_md = @elapsed begin
        # coarse optimization Gram + overlap at the candidate maxdim (cheap)
        M_opt, _ = open_gram_matrix(
            n, J, gammas, t, ks, lsites, rho0;
            cutoff=cutoff, maxdim=md, order=order, dissipation=true
        )
        L_opt, _ = open_L_vector(
            n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
            cutoff=cutoff, maxdim=md, order=order, order_ref=order_ref_opt,
            dissipation=true
        )

        # coefficients from the cheap opt step
        c, lambda = dynamic_mpf_coefficients(M_opt, L_opt)

        # errors evaluated against the ONCE-computed references
        E_mpf  = open_dynamic_mpf_error(M_ref, L_ref, c, purity)
        E_trot = open_single_trotter_errors(M_ref, L_ref, purity)

        M_dist = norm(M_opt .- M_ref)
    end

    push!(results, (maxdim=md, M_dist=M_dist, coeffs=c,
                    E_mpf=E_mpf, E_trot=E_trot, M_opt=M_opt))
    println("[$(now())] md=$md done ($(round(t_md, digits=1))s): M_dist=$M_dist E_mpf=$E_mpf"); flush(stdout)
end

sweep = (maxdims=maxdims, results=results, M_ref=M_ref)

# =============================================================================
# Write results to CSV (one row per md)
# =============================================================================

outdir = "sweep_results"
mkpath(outdir)
fname = joinpath(outdir, "sweep.csv")

open(fname, "w") do io
    println(io, "maxdim,M_dist,E_mpf,coeffs")
    for res in sweep.results
        coeffs_str = join(res.coeffs, ";")
        println(io, "$(res.maxdim),$(res.M_dist),$(res.E_mpf),\"$coeffs_str\"")
    end
end

println("[$(now())] sweep complete. wrote $fname"); flush(stdout)
print_sweep_summary(sweep)
