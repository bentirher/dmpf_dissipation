# direct_vs_sandwich.jl
# Head-to-head at n=5: compute the DMPF Gram matrix M and overlap vector L
# BOTH ways and compare entries + resulting coefficients.
#
#   SANDWICH route (current library): M via open_gram_matrix (build_open_F,
#     O(chi^4) MPO sandwiches <<rho0| F_ij |rho0>>), L via open_L_vector
#     (build_open_F_between_lists).
#
#   DIRECT route: evolve each state as an explicit vectorized MPS and take
#     inner products, matching the exact definitions:
#       M_ij = <<rho_ki(t) | rho_kj(t)>>
#       L_j  = <<rho_ref(t) | rho_kj(t)>>
#     This is O(chi^2) and, per the kref_convergence study, numerically stable.
#
# The kref study showed the SANDWICH L gives k_ref-unstable, physically wrong
# coefficients while the DIRECT L gives stable, k8-dominant (correct) ones.
# This confirms the same failure at the actual study size n=5, and establishes
# whether the earlier n=5 sweep coefficients (sandwich-based) need revising.
#
# Usage: julia direct_vs_sandwich.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

# --- setup: n=5, matching the sweep ---
Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(0.05, n)
t      = 3.0
lsites = liouville_siteinds(n)
ks       = [3, 8]
k_ref    = 40                 # same as the sweep's k_ref_opt
maxdim   = 256
cutoff   = 1e-12
order    = 2
diss     = true
r        = length(ks)

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

function evolve_state(k, ord)
    S = get_open_step_MPO(n, J, gammas, t/k, lsites, cutoff, maxdim; order=ord, dissipation=diss)
    psi = deepcopy(rho0)
    for _ in 1:k; psi = apply(S, psi; cutoff=cutoff, maxdim=maxdim); end
    return psi
end

# =============================================================================
# SANDWICH route (current library)
# =============================================================================
println("\n[$(now())] SANDWICH: building M via open_gram_matrix..."); flush(stdout)
tS = @elapsed begin
    M_sand, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                 cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
    L_sand, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                              cutoff=cutoff, maxdim=maxdim, order=order, order_ref=order, dissipation=diss)
end
c_sand, lam_sand = dynamic_mpf_coefficients(M_sand, L_sand)
println("[$(now())] SANDWICH done ($(round(tS,digits=1))s)"); flush(stdout)

# =============================================================================
# DIRECT route: state overlaps
# =============================================================================
println("\n[$(now())] DIRECT: evolving candidate + reference states as MPS..."); flush(stdout)
tD = @elapsed begin
    cand = Dict(k => evolve_state(k, order) for k in ks)
    rho_ref = evolve_state(k_ref, order)

    M_dir = zeros(Float64, r, r)
    for i in 1:r, j in 1:r
        M_dir[i,j] = real(inner(cand[ks[i]], cand[ks[j]]))
    end
    L_dir = [real(inner(rho_ref, cand[k])) for k in ks]
end
c_dir, lam_dir = dynamic_mpf_coefficients(M_dir, L_dir)
println("[$(now())] DIRECT done ($(round(tD,digits=1))s)"); flush(stdout)

# =============================================================================
# compare
# =============================================================================
println("\n" * "="^70)
println("COMPARISON (n=$n, gamma=0.05, ks=$ks, k_ref=$k_ref, maxdim=$maxdim)")
println("="^70)
println("\nM (Gram matrix):")
println("  sandwich = $M_sand")
println("  direct   = $M_dir")
println("  max|dM|  = $(maximum(abs.(M_sand .- M_dir)))")
println("\nL (overlap vector):")
println("  sandwich = $L_sand")
println("  direct   = $L_dir")
println("  max|dL|  = $(maximum(abs.(L_sand .- L_dir)))")
println("\nCoefficients:")
println("  c_sandwich = $(round.(c_sand,digits=5))   lambda=$(round(lam_sand,sigdigits=4))")
println("  c_direct   = $(round.(c_dir,digits=5))   lambda=$(round(lam_dir,sigdigits=4))")
println("  max|dc|    = $(maximum(abs.(c_sand .- c_dir)))")

# physical check + which set actually gives lower Trotter error
dist2(a,b) = (d = +(a, -1.0*b; cutoff=cutoff, maxdim=maxdim); real(inner(d,d)))
println("\nGround-truth single-Trotter errors (direct ||rho_ref - rho_kj||^2):")
for (j,k) in enumerate(ks)
    println("  E_k$k = $(round(dist2(rho_ref, cand[k]), sigdigits=5))")
end
# DMPF error achieved by each coefficient set
function mpf_err(c)
    mu = nothing
    for (j,k) in enumerate(ks)
        term = c[j]*cand[k]
        mu = (mu===nothing) ? term : +(mu, term; cutoff=cutoff, maxdim=maxdim)
    end
    return dist2(rho_ref, mu)
end
println("\nDMPF error ||rho_ref - mu||^2 achieved by each coefficient set (direct):")
println("  with c_sandwich : $(round(mpf_err(c_sand), sigdigits=5))")
println("  with c_direct   : $(round(mpf_err(c_dir),  sigdigits=5))")
println("  => the smaller one is the better (correct) coefficient set.")

println("\n[$(now())] done."); flush(stdout)
