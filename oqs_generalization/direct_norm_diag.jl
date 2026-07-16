# direct_norm_diag.jl
# Decisive diagnostic for the negative-E_mpf issue.
#
# The decomposition  E_mpf = purity + c^T M c - 2 L^T c  equals ||rho_ref - mu||^2
# ONLY in exact arithmetic. Each of M, L, purity is an independently-truncated
# middle-out MPO sandwich at maxdim=256, so their truncation errors need not
# satisfy the inequality that keeps the square >= 0.
#
# Here we compute ||rho_ref(t) - mu(t)||^2 DIRECTLY:
#   - build rho_ref(t) = S(t/k_ref)^{k_ref} |rho0>>   as an explicit MPS
#   - build each candidate rho_kj(t) = S(t/kj)^{kj} |rho0>>  as an MPS
#   - form mu(t) = sum_j c_j rho_kj(t)  as an MPS linear combination
#   - E_direct = || rho_ref - mu ||^2      (guaranteed >= 0)
# and compare against the decomposition value from M/L/purity.
#
# If E_direct is a small POSITIVE number while E_decomp is negative, the
# negativity is confirmed as decomposition-truncation error, and E_direct is
# the trustworthy error. We run this at md=128 (cleanest candidate).
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

md = 128   # candidate maxdim under test

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# helper: evolve rho0 by k Trotter steps of S(t/k), returning an MPS
function evolve_state(k, mdim, ord)
    dt = t / k
    S = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, mdim; order=ord, dissipation=true)
    psi = deepcopy(rho0)
    for _ in 1:k
        psi = apply(S, psi; cutoff=cutoff, maxdim=mdim)
    end
    return psi
end

# --- 1) coefficients c from the SAME cheap opt step the sweep uses ---
println("[$(now())] building M_opt, L_opt at md=$md for coefficients..."); flush(stdout)
M_opt, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=cutoff, maxdim=md, order=order, dissipation=true)
L_opt, _ = open_L_vector(n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
                         cutoff=cutoff, maxdim=md, order=order, order_ref=order_ref_opt,
                         dissipation=true)
c, _ = dynamic_mpf_coefficients(M_opt, L_opt)
println("[$(now())] c = $c"); flush(stdout)

# --- 2) decomposition value (matches run_sweep_case_v2.jl) ---
println("[$(now())] building M_ref, L_ref, purity for decomposition..."); flush(stdout)
M_ref, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=cutoff, maxdim=maxdim_ref, order=order, dissipation=true)
L_ref, _ = open_L_vector(n, J, gammas, t, ks, k_ref_eval, lsites, rho0;
                         cutoff=cutoff, maxdim=maxdim_ref, order=order, order_ref=order_ref_eval,
                         dissipation=true)
Fpp = build_open_F_between_lists(n, J, gammas, t, [k_ref_eval], [k_ref_eval],
                                 lsites, cutoff, maxdim_ref;
                                 order_left=order_ref_eval, order_right=order_ref_eval,
                                 dissipation=true)
purity = real(inner(rho0', Fpp[1], rho0))
E_decomp = purity + dot(c, M_ref * c) - 2.0 * dot(L_ref, c)
println("[$(now())] E_decomp = $E_decomp   (purity=$purity)"); flush(stdout)

# --- 3) DIRECT norm ---
println("[$(now())] building rho_ref(t) directly (k_ref=$k_ref_eval)..."); flush(stdout)
rho_ref = evolve_state(k_ref_eval, maxdim_ref, order_ref_eval)

println("[$(now())] building candidate states rho_kj(t)..."); flush(stdout)
# mu(t) = sum_j c_j rho_kj(t), each candidate evolved at the REFERENCE maxdim
# so the comparison is at the reference's accuracy, matching M_ref/L_ref.
mu = nothing
for (j, kj) in enumerate(ks)
    psi_j = evolve_state(kj, maxdim_ref, order)
    term = c[j] * psi_j
    global mu = (mu === nothing) ? term : +(mu, term; cutoff=cutoff, maxdim=maxdim_ref)
end

diff = +(rho_ref, -1.0 * mu; cutoff=cutoff, maxdim=maxdim_ref)
E_direct = real(inner(diff, diff))
println("[$(now())] E_direct = $E_direct"); flush(stdout)

# --- verdict ---
println("\n==== RESULT (md=$md) ====")
println("E_decomp (purity + cMc - 2Lc) = $E_decomp")
println("E_direct (||rho_ref - mu||^2) = $E_direct")
println("gap (decomp - direct)         = $(E_decomp - E_direct)")
println()
if E_direct >= 0 && E_decomp < 0
    println("=> Negativity is a DECOMPOSITION-TRUNCATION artifact.")
    println("   The trustworthy error at md=$md is E_direct = $E_direct.")
elseif E_direct < 0
    println("=> E_direct is ALSO negative -- impossible in exact arithmetic,")
    println("   so the MPS linear-combination truncation itself is the floor.")
    println("   This points to a reference-resolution limit at maxdim=256.")
else
    println("=> Both non-negative; decomposition and direct agree.")
end
flush(stdout)
