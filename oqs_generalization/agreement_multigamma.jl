# agreement_multigamma.jl
# Confirms that the SANDWICH and DIRECT routes agree on M, L, and the DMPF
# coefficients across multiple gamma values at n=5 (the study size).
#
# Context: direct_vs_sandwich.jl showed agreement at a SINGLE point
# (gamma=0.05, one seed): max|dc| = 2.7e-3, both k=8-dominant, equal achieved
# error. This script checks the agreement is not a fluke of that point.
#
# NOTE ON THE DIRECT ROUTE: it is used here ONLY as a validation oracle. It is
# disqualified as a production method because efficient classical state
# evolution voids the reason to use DMPF/a QPU at all. See the Outlook in
# findings_thematic.md.
#
# One gamma per SLURM array task (the sandwich M build is ~2.9 h at n=5).
#
# Usage: julia agreement_multigamma.jl <gamma>

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

gamma_val = parse(Float64, ARGS[1])

# --- setup: n=5, matching the sweep ---
Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma_val, n)
t      = 3.0
lsites = liouville_siteinds(n)
ks       = [3, 8]
k_ref    = 40
maxdim   = 256
cutoff   = 1e-12
order    = 2
diss     = gamma_val > 0.0
r        = length(ks)

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

function evolve_state(k, ord)
    S = get_open_step_MPO(n, J, gammas, t/k, lsites, cutoff, maxdim; order=ord, dissipation=diss)
    psi = deepcopy(rho0)
    for _ in 1:k; psi = apply(S, psi; cutoff=cutoff, maxdim=maxdim); end
    return psi
end
dist2(a,b) = (d = +(a, -1.0*b; cutoff=cutoff, maxdim=maxdim); real(inner(d,d)))

println("\n[$(now())] gamma=$gamma_val : SANDWICH route..."); flush(stdout)
tS = @elapsed begin
    M_sand, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                 cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
    L_sand, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                              cutoff=cutoff, maxdim=maxdim, order=order, order_ref=order, dissipation=diss)
end
c_sand, lam_sand = dynamic_mpf_coefficients(M_sand, L_sand)
println("[$(now())] SANDWICH done ($(round(tS,digits=1))s)"); flush(stdout)

println("[$(now())] gamma=$gamma_val : DIRECT route (validation oracle)..."); flush(stdout)
tD = @elapsed begin
    cand    = Dict(k => evolve_state(k, order) for k in ks)
    rho_ref = evolve_state(k_ref, order)
    M_dir = zeros(Float64, r, r)
    for i in 1:r, j in 1:r
        M_dir[i,j] = real(inner(cand[ks[i]], cand[ks[j]]))
    end
    L_dir = [real(inner(rho_ref, cand[k])) for k in ks]
end
c_dir, lam_dir = dynamic_mpf_coefficients(M_dir, L_dir)
println("[$(now())] DIRECT done ($(round(tD,digits=1))s)"); flush(stdout)

# achieved DMPF error for each coefficient set (direct ground truth)
function mpf_err(c)
    mu = nothing
    for (j,k) in enumerate(ks)
        term = c[j]*cand[k]
        mu = (mu===nothing) ? term : +(mu, term; cutoff=cutoff, maxdim=maxdim)
    end
    return dist2(rho_ref, mu)
end
E_sand = mpf_err(c_sand); E_dir = mpf_err(c_dir)
E_k = [dist2(rho_ref, cand[k]) for k in ks]

dM = maximum(abs.(M_sand .- M_dir))
dL = maximum(abs.(L_sand .- L_dir))
dc = maximum(abs.(c_sand .- c_dir))

println("\n==== gamma = $gamma_val (n=$n, maxdim=$maxdim, k_ref=$k_ref) ====")
println("  max|dM| = $dM")
println("  max|dL| = $dL")
println("  c_sandwich = $(round.(c_sand,digits=5))")
println("  c_direct   = $(round.(c_dir,digits=5))")
println("  max|dc|    = $dc")
println("  E achieved: sandwich=$(round(E_sand,sigdigits=5))  direct=$(round(E_dir,sigdigits=5))")
println("  single-Trotter: E_k3=$(round(E_k[1],sigdigits=5))  E_k8=$(round(E_k[2],sigdigits=5))  " *
        (E_k[2] < E_k[1] ? "(k8<k3 OK)" : "(k8>=k3 !!)"))
println("  agreement verdict: " * (dc < 1e-2 ? "AGREE (max|dc| < 1e-2)" : "DISAGREE -- investigate"))

outdir = "agreement_results"; mkpath(outdir)
open(joinpath(outdir, "agreement_gamma$(gamma_val).csv"), "w") do io
    println(io, "gamma,dM,dL,dc,c_sand_1,c_sand_2,c_dir_1,c_dir_2,E_sand,E_dir,E_k3,E_k8,t_sandwich,t_direct")
    println(io, "$gamma_val,$dM,$dL,$dc,$(c_sand[1]),$(c_sand[2]),$(c_dir[1]),$(c_dir[2])," *
                "$E_sand,$E_dir,$(E_k[1]),$(E_k[2]),$tS,$tD")
end
println("\n[$(now())] wrote agreement_results/agreement_gamma$(gamma_val).csv"); flush(stdout)
