# run_sweep_case_v3.jl
# Final sweep: DIRECT-NORM error estimator.
#
# The decomposition E_mpf = purity + c^T M_ref c - 2 L_ref^T c suffers
# catastrophic cancellation at truncation precision -- the three independently
# truncated MPO sandwiches disagree by ~2e-3, larger than the true error (~5e-4),
# which drove E_mpf negative. Verified via direct_norm_diag.jl:
#   md=128:  E_decomp = -1.494e-3   vs   E_direct = +5.48e-4  (correct).
#
# This version computes the error DIRECTLY as ||rho_ref(t) - mu(t)||^2 from
# explicit vectorized-state MPS, which is both numerically stable (guaranteed
# >= 0) AND far cheaper: MPS evolution is O(chi^2) vs the O(chi^4) sandwiches,
# so the expensive M_ref / L_ref / purity builds are eliminated entirely.
#
# md-independent objects (rho_ref, candidate states rho_kj) are built ONCE and
# reused across all md. Only the coefficients + linear combination + norm run
# inside the md loop.
#
# M_dist is retained as a diagnostic; it needs the maxdim_ref Gram matrix, which
# is the one remaining expensive build. Set COMPUTE_MDIST=false to skip it and
# make the whole sweep run in minutes.
#
# No project files are edited.

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

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

COMPUTE_MDIST = true   # set false to skip the one remaining expensive build

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# evolve rho0 by k Trotter steps of S(t/k) -> explicit vectorized-state MPS
function evolve_state(k, mdim, ord)
    dt = t / k
    S = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, mdim; order=ord, dissipation=true)
    psi = deepcopy(rho0)
    for _ in 1:k
        psi = apply(S, psi; cutoff=cutoff, maxdim=mdim)
    end
    return psi
end

# =============================================================================
# md-INDEPENDENT states -- built ONCE
# =============================================================================

println("[$(now())] building rho_ref(t) (k_ref=$k_ref_eval, maxdim=$maxdim_ref)..."); flush(stdout)
t_ref = @elapsed rho_ref = evolve_state(k_ref_eval, maxdim_ref, order_ref_eval)
println("[$(now())] rho_ref built ($(round(t_ref,digits=1))s)"); flush(stdout)

println("[$(now())] building candidate states rho_kj(t) at maxdim=$maxdim_ref..."); flush(stdout)
candidate_states = Dict{Int,MPS}()
for kj in ks
    t_c = @elapsed candidate_states[kj] = evolve_state(kj, maxdim_ref, order)
    println("[$(now())]   k=$kj built ($(round(t_c,digits=1))s)"); flush(stdout)
end

# optional: reference Gram for the M_dist diagnostic (the one costly build)
M_ref = nothing
if COMPUTE_MDIST
    println("[$(now())] building M_ref for M_dist diagnostic (maxdim=$maxdim_ref)..."); flush(stdout)
    t_M = @elapsed begin
        M_ref, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                    cutoff=cutoff, maxdim=maxdim_ref, order=order,
                                    dissipation=true)
    end
    println("[$(now())] M_ref done ($(round(t_M,digits=1))s)"); flush(stdout)
end

# =============================================================================
# md-DEPENDENT loop -- all cheap
# =============================================================================

results = NamedTuple[]

for md in maxdims
    global results
    println("[$(now())] --- md=$md ---"); flush(stdout)

    t_md = @elapsed begin
        # coefficients from the cheap opt step (unchanged)
        M_opt, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                    cutoff=cutoff, maxdim=md, order=order, dissipation=true)
        L_opt, _ = open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
                                 cutoff=cutoff, maxdim=md, order=order, order_ref=order_ref_opt,
                                 dissipation=true)
        c, _ = dynamic_mpf_coefficients(M_opt, L_opt)

        # DIRECT error: mu = sum_j c_j rho_kj  (reusing the once-built states)
        mu = nothing
        for (j, kj) in enumerate(ks)
            term = c[j] * candidate_states[kj]
            mu = (mu === nothing) ? term : +(mu, term; cutoff=cutoff, maxdim=maxdim_ref)
        end
        diff = +(rho_ref, -1.0 * mu; cutoff=cutoff, maxdim=maxdim_ref)
        E_mpf = real(inner(diff, diff))

        M_dist = COMPUTE_MDIST ? norm(M_opt .- M_ref) : NaN
    end

    push!(results, (maxdim=md, M_dist=M_dist, coeffs=c, E_mpf=E_mpf))
    println("[$(now())] md=$md done ($(round(t_md,digits=1))s): E_mpf=$E_mpf  M_dist=$M_dist  c=$c"); flush(stdout)
end

# =============================================================================
# write CSV
# =============================================================================

outdir = "sweep_results"
mkpath(outdir)
fname = joinpath(outdir, "sweep_v3.csv")
open(fname, "w") do io
    println(io, "maxdim,M_dist,E_mpf,coeffs")
    for res in results
        coeffs_str = join(res.coeffs, ";")
        println(io, "$(res.maxdim),$(res.M_dist),$(res.E_mpf),\"$coeffs_str\"")
    end
end

println("\n[$(now())] sweep complete. wrote $fname"); flush(stdout)
println("\nmaxdim | E_mpf        | M_dist    | coeffs")
for res in results
    println("$(rpad(res.maxdim,6)) | $(rpad(round(res.E_mpf,digits=6),12)) | $(rpad(round(res.M_dist,digits=6),9)) | $(round.(res.coeffs,digits=4))")
end
flush(stdout)
