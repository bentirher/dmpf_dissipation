# verify_purity.jl
# Confirms the source of the negative E_mpf: purity computed two different ways.
#
#   (A) current route: reference_purity() -- applies the k_ref forward channel
#       to rho0 as an MPS, then <<rho(t)|rho(t)>>.  (O(chi^2) MPS truncation)
#
#   (B) consistent route: the [k_ref]x[k_ref] diagonal of the SAME MPO sandwich
#       machinery used to build L_ref, i.e.
#       <<rho(0)| [S_ref]^dag S_ref |rho(0)>>.   (O(chi^4) MPO truncation)
#
# If the diagnosis is right, (A) and (B) differ at the ~1e-3 level -- the same
# magnitude as the E_mpf values that went negative -- because they truncate the
# reference rho(t) differently at finite maxdim.
#
# Usage: julia verify_purity.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates

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
k_ref_eval = 50
maxdim_ref = 256
cutoff     = 1e-12
order_ref_eval = 2

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# (A) current MPS-route purity
println("[$(now())] (A) reference_purity (MPS route)..."); flush(stdout)
tA = @elapsed purity_mps = reference_purity(
    n, J, gammas, t, k_ref_eval, lsites, rho0;
    cutoff=cutoff, maxdim=maxdim_ref, order=order_ref_eval, dissipation=true)
println("[$(now())] (A) done ($(round(tA,digits=1))s): purity_MPS = $purity_mps"); flush(stdout)

# (B) consistent MPO-sandwich purity: [k_ref] x [k_ref] diagonal element,
#     built with the SAME build_open_F_between_lists used for L_ref.
println("[$(now())] (B) MPO-sandwich purity ([k_ref]x[k_ref])..."); flush(stdout)
tB = @elapsed begin
    Fpp = build_open_F_between_lists(
        n, J, gammas, t, [k_ref_eval], [k_ref_eval], lsites, cutoff, maxdim_ref;
        order_left=order_ref_eval, order_right=order_ref_eval, dissipation=true)
    purity_mpo = real(inner(rho0', Fpp[1], rho0))
end
println("[$(now())] (B) done ($(round(tB,digits=1))s): purity_MPO = $purity_mpo"); flush(stdout)

# --- compare ---
d  = purity_mps - purity_mpo
println("\n==== RESULT ====")
println("purity_MPS (current)      = $purity_mps")
println("purity_MPO (consistent)   = $purity_mpo")
println("difference (A - B)        = $d")
println("relative difference       = $(d / purity_mpo)")
println("\nFor reference, the sweep's negative E_mpf values were ~ -1e-3.")
println("If |difference| is of that order, the mismatch explains the sign flip.")
flush(stdout)
