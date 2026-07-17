# direct_norm_diag.jl  (slimmed)
# Computes ONLY the direct norm ||rho_ref(t) - mu(t)||^2, since the decomposition
# value is already known from the v2 sweep (E_decomp at md=128 = -0.001494).
#
# rho_ref(t) = S(t/k_ref)^{k_ref}|rho0>>  and  mu(t) = sum_j c_j rho_kj(t),
# all built as explicit vectorized-state MPS and truncated at maxdim_ref, then
# E_direct = ||rho_ref - mu||^2  (guaranteed >= 0 up to MPS round-off).
#
# Compare the printed E_direct against the known E_decomp = -0.001494:
#   * E_direct small POSITIVE  -> negativity is a decomposition-truncation
#                                 artifact; E_direct is the trustworthy error.
#   * E_direct ALSO negative   -> maxdim=256 is below the resolution floor at
#                                 n=5; the error is unresolvable there.
#
# Usage: julia direct_norm_diag.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

# --- same setup as the sweep ---
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
order          = 2
order_ref_opt  = 1
order_ref_eval = 2

md = 128
E_decomp_known = -0.001494   # from the v2 sweep, for reference in the printout

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

function evolve_state(k, mdim, ord)
    dt = t / k
    S = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, mdim; order=ord, dissipation=true)
    psi = deepcopy(rho0)
    for _ in 1:k
        psi = apply(S, psi; cutoff=cutoff, maxdim=mdim)
    end
    return psi
end

# --- coefficients c: cheap opt step (same as sweep) ---
println("[$(now())] building M_opt, L_opt at md=$md for coefficients..."); flush(stdout)
M_opt, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=cutoff, maxdim=md, order=order, dissipation=true)
L_opt, _ = open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
                         cutoff=cutoff, maxdim=md, order=order, order_ref=order_ref_opt,
                         dissipation=true)
c, _ = dynamic_mpf_coefficients(M_opt, L_opt)
println("[$(now())] c = $c"); flush(stdout)

# --- DIRECT norm only ---
println("[$(now())] building rho_ref(t) (k_ref=$k_ref_eval, maxdim=$maxdim_ref)..."); flush(stdout)
rho_ref = evolve_state(k_ref_eval, maxdim_ref, order_ref_eval)
println("[$(now())] rho_ref built."); flush(stdout)

mu = nothing
for (j, kj) in enumerate(ks)
    println("[$(now())]   building candidate k=$kj ..."); flush(stdout)
    psi_j = evolve_state(kj, maxdim_ref, order)
    term = c[j] * psi_j
    global mu = (mu === nothing) ? term : +(mu, term; cutoff=cutoff, maxdim=maxdim_ref)
end
println("[$(now())] mu(t) assembled."); flush(stdout)

diff = +(rho_ref, -1.0 * mu; cutoff=cutoff, maxdim=maxdim_ref)
E_direct = real(inner(diff, diff))

println("\n==== RESULT (md=$md) ====")
println("E_decomp (known, from v2 sweep) = $E_decomp_known")
println("E_direct (||rho_ref - mu||^2)   = $E_direct")
println("gap (decomp - direct)           = $(E_decomp_known - E_direct)")
println()
if E_direct >= 0
    println("=> E_direct is NON-NEGATIVE: negativity is a decomposition-truncation")
    println("   artifact. Trustworthy error at md=$md is E_direct = $E_direct.")
else
    println("=> E_direct is ALSO negative: maxdim=$maxdim_ref is below the")
    println("   resolution floor at n=$n. The error is unresolvable there.")
end
flush(stdout)
