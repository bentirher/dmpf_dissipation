# run_sweep_case_v2.jl
# De-duplicated sweep + OPTION-1 FIX for the negative-E_mpf artifact.
#
# The fix: purity Tr(rho^2(t)) is now extracted from the SAME MPO reference
# sandwich used to build L_ref, instead of a separate MPS forward evolution.
# Both then share ONE consistently-truncated reference rho(t), so the
# truncation errors cancel in  E_mpf = purity + c^T M c - 2 L^T c,  keeping it
# non-negative up to genuine round-off rather than the ~1e-3 mismatch floor.
#
# Concretely: purity = <<rho(0)| [S_ref]^dag S_ref |rho(0)>>
#           = the [k_ref] x [k_ref] diagonal element of
#             build_open_F_between_lists(..., [k_ref], [k_ref], ...).
#
# Everything else (de-dup of references, maxdim_ref=256, md in {16,32,64,128})
# is unchanged from run_sweep_case.jl. No project files are edited.

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads(), " / Julia CPU threads: ", Sys.CPU_THREADS)
flush(stdout)

# --- setup ---
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

maxdims = [16, 32, 64, 128]

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# =============================================================================
# md-INDEPENDENT reference objects -- computed ONCE
# =============================================================================

println("[$(now())] building reference M_ref (maxdim=$maxdim_ref)..."); flush(stdout)
t_Mref = @elapsed begin
    M_ref, _ = open_gram_matrix(
        n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=order, dissipation=true)
end
println("[$(now())] M_ref done ($(round(t_Mref, digits=1))s)"); flush(stdout)

println("[$(now())] building reference L_ref (maxdim=$maxdim_ref, k_ref=$k_ref_eval)..."); flush(stdout)
t_Lref = @elapsed begin
    L_ref, _ = open_L_vector(
        n, J, gammas, t, ks, k_ref_eval, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=order, order_ref=order_ref_eval,
        dissipation=true)
end
println("[$(now())] L_ref done ($(round(t_Lref, digits=1))s)"); flush(stdout)

# --- OPTION-1 FIX: purity from the SAME MPO sandwich as L_ref ---
# [k_ref] x [k_ref] diagonal of build_open_F_between_lists, so the reference
# rho(t) is truncated identically to the one implicit in L_ref.
println("[$(now())] building consistent purity ([k_ref]x[k_ref] MPO sandwich)..."); flush(stdout)
t_pur = @elapsed begin
    Fpp = build_open_F_between_lists(
        n, J, gammas, t, [k_ref_eval], [k_ref_eval], lsites, cutoff, maxdim_ref;
        order_left=order_ref_eval, order_right=order_ref_eval, dissipation=true)
    purity = real(inner(rho0', Fpp[1], rho0))
end
println("[$(now())] purity done ($(round(t_pur, digits=1))s)  value=$purity"); flush(stdout)

# =============================================================================
# md-DEPENDENT loop
# =============================================================================

results = NamedTuple[]

for md in maxdims
    global results
    println("[$(now())] --- md=$md ---"); flush(stdout)

    t_md = @elapsed begin
        M_opt, _ = open_gram_matrix(
            n, J, gammas, t, ks, lsites, rho0;
            cutoff=cutoff, maxdim=md, order=order, dissipation=true)
        L_opt, _ = open_L_vector(
            n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
            cutoff=cutoff, maxdim=md, order=order, order_ref=order_ref_opt,
            dissipation=true)

        c, lambda = dynamic_mpf_coefficients(M_opt, L_opt)

        E_mpf  = open_dynamic_mpf_error(M_ref, L_ref, c, purity)
        E_trot = open_single_trotter_errors(M_ref, L_ref, purity)

        M_dist = norm(M_opt .- M_ref)
    end

    push!(results, (maxdim=md, M_dist=M_dist, coeffs=c,
                    E_mpf=E_mpf, E_trot=E_trot, M_opt=M_opt))
    println("[$(now())] md=$md done ($(round(t_md, digits=1))s): M_dist=$M_dist E_mpf=$E_mpf"); flush(stdout)
end

sweep = (maxdims=maxdims, results=results, M_ref=M_ref)

outdir = "sweep_results"
mkpath(outdir)
fname = joinpath(outdir, "sweep_v2.csv")
open(fname, "w") do io
    println(io, "maxdim,M_dist,E_mpf,coeffs")
    for res in sweep.results
        coeffs_str = join(res.coeffs, ";")
        println(io, "$(res.maxdim),$(res.M_dist),$(res.E_mpf),\"$coeffs_str\"")
    end
end

println("[$(now())] sweep complete. wrote $fname"); flush(stdout)
print_sweep_summary(sweep)
