# coeff_sanity.jl
# Interrogates whether the DMPF coefficient computation is CORRECT, driven by
# the physical objection: the k=8 circuit (finest Trotter) should be the most
# faithful single approximation to rho(t), yet the fit sometimes weights k=3
# far more. This script checks every link in the chain, at n=4 (cheap) and a
# couple of gammas, using ONLY the library routines (no S^{-1} machinery).
#
# Checks performed, per gamma:
#   [A] Single-Trotter errors E_kj = ||rho(t) - rho_kj||^2 for each kj, computed
#       via the library formula purity + M[j,j] - 2 L[j]. PHYSICAL EXPECTATION:
#       E_k8 < E_k3 (finer Trotter = smaller error). If violated, either the
#       reference is bad or M/L is wrong.
#   [B] Direct single-Trotter errors: build rho_ref, rho_k3, rho_k8 as explicit
#       MPS and compute ||rho_ref - rho_kj||^2 DIRECTLY, bypassing M/L entirely.
#       Cross-check against [A]. If [A] and [B] disagree, the M/L construction
#       (or its purity/reference matching) is inconsistent.
#   [C] k_ref convergence: recompute L and c at k_ref = 40, 80, 160. If c moves,
#       the reference was not converged and coefficients were fit to a moving
#       (wrong) target -- this would explain physically-nonsensical weights.
#   [D] The resulting DMPF coefficients and the error they achieve, vs the best
#       single circuit. A valid DMPF should do at least as well as the best
#       single Trotter circuit; if E_mpf > min(E_kj), the fit is misbehaving.
#
# Usage: julia coeff_sanity.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

Random.seed!(1234)
n      = 4
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
t      = 3.0
lsites = liouville_siteinds(n)
ks       = [3, 8]
maxdim   = 64
cutoff   = 1e-12
order    = 2

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

gammas_to_test = [0.03, 0.05]      # the two that looked physically wrong
krefs          = [40, 80, 160]     # for the convergence check

# evolve rho0 by k steps -> explicit MPS
function evolve_state(k, ord, gammas, diss)
    S = get_open_step_MPO(n, J, gammas, t/k, lsites, cutoff, maxdim; order=ord, dissipation=diss)
    psi = deepcopy(rho0)
    for _ in 1:k; psi = apply(S, psi; cutoff=cutoff, maxdim=maxdim); end
    return psi
end

# direct squared Frobenius distance ||a - b||^2 between two vectorized-state MPS
function dist2(a, b)
    d = +(a, -1.0 * b; cutoff=cutoff, maxdim=maxdim)
    return real(inner(d, d))
end

for gamma in gammas_to_test
    gammas = fill(gamma, n)
    diss = gamma > 0.0
    println("\n" * "="^70)
    println("GAMMA = $gamma")
    println("="^70); flush(stdout)

    # ---- library M and purity (reference at k_ref=40, order_ref=2 for L) ----
    kref0 = 40
    M, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
    purity = reference_purity(n, J, gammas, t, kref0, lsites, rho0;
                              cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
    L0, _ = open_L_vector(n, J, gammas, t, ks, kref0, lsites, rho0;
                          cutoff=cutoff, maxdim=maxdim, order=order, order_ref=order, dissipation=diss)
    c0, lam0 = dynamic_mpf_coefficients(M, L0)

    println("\n[A] Single-Trotter errors via library formula purity + M[j,j] - 2 L[j]:")
    E_formula = [purity + M[j,j] - 2*L0[j] for j in 1:length(ks)]
    for (j,k) in enumerate(ks)
        println("    E_k$k (formula) = $(round(E_formula[j], sigdigits=6))   [M[$j,$j]=$(round(M[j,j],digits=5)), L[$j]=$(round(L0[j],digits=5))]")
    end
    println("    purity(ref) = $(round(purity, digits=6))")
    println("    => PHYSICAL CHECK: expect E_k8 < E_k3.  " *
            (E_formula[2] < E_formula[1] ? "OK (k8 < k3)" : "VIOLATED (k8 >= k3) !!"))
    flush(stdout)

    # ---- [B] direct single-Trotter errors, bypassing M/L ----
    println("\n[B] Direct single-Trotter errors ||rho_ref - rho_kj||^2 (explicit MPS):")
    rho_ref = evolve_state(kref0, order, gammas, diss)
    purity_direct = real(inner(rho_ref, rho_ref))
    println("    purity(direct <<rho_ref|rho_ref>>) = $(round(purity_direct, digits=6))  " *
            "(formula purity was $(round(purity,digits=6)); should match)")
    for (j,k) in enumerate(ks)
        rho_k = evolve_state(k, order, gammas, diss)
        Ed = dist2(rho_ref, rho_k)
        println("    E_k$k (direct) = $(round(Ed, sigdigits=6))   " *
                "(formula gave $(round(E_formula[j], sigdigits=6)))")
    end
    flush(stdout)

    # ---- [C] k_ref convergence of L and c ----
    println("\n[C] k_ref convergence (does the reference / coefficients stabilize?):")
    for kref in krefs
        Lk, _ = open_L_vector(n, J, gammas, t, ks, kref, lsites, rho0;
                              cutoff=cutoff, maxdim=maxdim, order=order, order_ref=order, dissipation=diss)
        ck, lamk = dynamic_mpf_coefficients(M, Lk)
        println("    k_ref=$kref: L=$(round.(Lk,digits=5))  c=$(round.(ck,digits=4))  lambda=$(round(lamk,sigdigits=4))")
    end
    flush(stdout)

    # ---- [D] does the DMPF beat the best single circuit? ----
    E_mpf = purity + dot(c0, M*c0) - 2*dot(L0, c0)
    best_single = minimum(E_formula)
    println("\n[D] DMPF vs best single circuit (k_ref=$kref0):")
    println("    c = $(round.(c0, digits=4)),  lambda=$(round(lam0,sigdigits=4))")
    println("    E_mpf         = $(round(E_mpf, sigdigits=6))")
    println("    best E_single = $(round(best_single, sigdigits=6))")
    println("    => VALIDITY CHECK: E_mpf should be <= best single.  " *
            (E_mpf <= best_single + 1e-9 ? "OK" : "VIOLATED (DMPF worse than best single) !!"))
    if E_mpf < 0
        println("    NOTE: E_mpf < 0 -- decomposition-truncation artifact (known issue).")
    end
    flush(stdout)
end

println("\n[$(now())] sanity checks complete."); flush(stdout)
