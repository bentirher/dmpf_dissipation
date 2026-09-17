# =============================================================================
# family_scan.jl   --   CHOOSE THE CANDIDATE FAMILY
#
# WHY THIS RUNS BEFORE THE n-SWEEP
# --------------------------------
# In the first Step 1 run the DMPF answer bottomed out at
#
#     err_proj = 5.0e-3      (the error with PERFECT coefficients)
#
# while the purely classical simulation reached 1.5e-3 at chi=128 and was exact
# at chi=256. err_proj is a floor no amount of bond dimension can cross, so that
# comparison was lost before it started -- and the floor was not fundamental. It
# came from the candidate family:
#
#   * ks = [3, 8]: only TWO candidates, so the fit has one degree of freedom.
#     With sum(c) = 1 and r = 2, dc is forced along (1,-1)/sqrt(2) -- there is
#     exactly ONE direction it can point. The near-null-eigenvector argument of
#     L-MPF Appendix D is vacuous at r = 2; it needs r >= 3 to say anything.
#   * splitting = :project, which Step 0 measured at p_eff = 1.00 to three
#     digits for order 4 and decaying to ~1.3 for order 2. DMPF was being asked
#     to rescue a first-order formula.
#   * k = 3 means dt = 1.0. From the Step 0 order ladder that is a ~43% error in
#     state norm. E_k3 = 8.5e-2 against E_k8 = 1.5e-3: the two candidates are
#     fifty-fold apart in quality, so the fit is nearly a one-term formula.
#
# Guessing a replacement is not necessary. Candidate evolutions are short (k
# steps, not k0), so the whole exact pass is cheap and several families can be
# scored directly. That is all this script does.
#
# WHAT IT REPORTS, AND HOW TO CHOOSE
# ----------------------------------
#   E_mpf         the achieved error with perfect coefficients. LOWER IS BETTER,
#                 and it is the floor on everything downstream.
#   err_proj      the same thing at the level of an observable. This is the
#                 number the n-sweep has to beat a classical simulation with.
#   proj/single   err_proj divided by the best single Trotter circuit's error.
#                 Below 1 means the multiproduct fit is buying something at all.
#   cond(N)       conditioning of the fit. L-MPF warns this fails from BOTH
#                 ends: candidates too close together and the Gram matrix goes
#                 singular (all the error vectors parallel); too far apart and
#                 the off-diagonals go to zero. There is a sweet spot and it is
#                 empirical.
#   r             number of candidates. r >= 3 is required for dc to have more
#                 than one direction available, hence for the Appendix D
#                 mechanism to be testable at all.
#
# Pick the family with the lowest err_proj subject to cond(N) staying manageable
# (say under ~1e4) and r >= 3. Do not simply minimise E_mpf: a family can reach
# a tiny E_mpf with a cond(N) so large that truncation noise swamps it.
#
# CONSTRAINT: k0 must be an integer multiple of lcm(ks) for these coefficients
# to be cross-checkable against the MOC/N-route, whose B_j blocks need
# k0 % k_j == 0. Flagged per family below; K0=48 satisfies lcm = 2,3,4,6,8,12,16,24,48.
#
# Environment: N_QUBITS GAMMA TVAL K0 ORDER SPLITTING ORDER_REF SPLITTING_REF
#              FAMILIES CUTOFF TAG
# Output: family_scan<_tag>.csv
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("liouville_state_tools.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n             = parse(Int,     getenv("N_QUBITS",  6))
gamma         = parse(Float64, getenv("GAMMA",     0.05))
t             = parse(Float64, getenv("TVAL",      3.0))
k0            = parse(Int,     getenv("K0",        48))
order         = parse(Int,     getenv("ORDER",     2))
order_ref     = parse(Int,     getenv("ORDER_REF", 4))
splitting     = Symbol(getenv("SPLITTING",     "strang"))
splitting_ref = Symbol(getenv("SPLITTING_REF", "strang"))
evo_mode      = Symbol(getenv("EVO_MODE", "gates"))
ct            = parse(Float64, getenv("CUTOFF", EXACT_CUTOFF))
tag           = getenv("TAG", "")
sfx           = isempty(tag) ? "" : "_" * tag

# Families, semicolon-separated. Defaults span the axes that matter: number of
# candidates, spacing, and how coarse the coarsest member is.
families = map(f -> parse.(Int, split(strip(f), ",")),
               split(getenv("FAMILIES",
                   "3,8;" *            # the current family, for reference
                   "4,12;" *           # same size, both members finer
                   "4,8,16;" *         # r=3, geometric
                   "4,6,8,12;" *       # r=4, moderate spacing
                   "6,8,12,16;" *      # r=4, all members fine
                   "2,4,8,16;" *       # r=4, wide (includes a very coarse dt=1.5)
                   "4,6,8,12,16,24"),  # r=6, dense
               ";"))

chi = state_max_bond_dim(n)

@printf("family_scan\n")
@printf("  n=%d gamma=%.3f t=%.1f k0=%d\n", n, gamma, t, k0)
@printf("  candidates: splitting=%s order=%d    reference: splitting=%s order=%d\n",
        splitting, order, splitting_ref, order_ref)
@printf("  exact at the chi=%d ceiling; %d families\n\n", chi, length(families))
flush(stdout)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

mid  = n ÷ 2
O_id = identity_observable(lsites)
obs_names = vcat(["Z$m" for m in 1:n], ["Z$(mid)Z$(mid+1)"])
obs_mps   = vcat([z_observable(lsites, m) for m in 1:n],
                 [zz_observable(lsites, mid, mid + 1)])
n_obs = length(obs_names)

# ---- reference, once --------------------------------------------------------
e_ref = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                       maxdim=chi, order=order_ref, cutoff=ct,
                       splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
rho_ref = e_ref.rho
@printf("reference k0=%d: chi_rho=%d eps=%.2e raw Tr=%+.12f\n\n",
        k0, e_ref.chi, e_ref.eps, real(e_ref.trace))
O_exact = [real(expval(obs_mps[a], rho_ref)) for a in 1:n_obs]
flush(stdout)

# ---- candidates, cached across families ------------------------------------
# Families overlap heavily (4 appears in five of the defaults), so evolving each
# distinct k once is a large saving and costs nothing but a Dict.
all_ks = sort(unique(vcat(families...)))
cache  = Dict{Int,MPS}()
println("candidate states (each k evolved once, shared across families):")
for kj in all_ks
    ek = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                        maxdim=chi, order=order, cutoff=ct,
                        splitting=splitting, mode=evo_mode, id_mps=O_id)
    cache[kj] = ek.rho
    d = mps_difference(ek.rho, rho_ref)
    @printf("  k=%-3d dt=%.4f  ||Delta||^2 = %.4e   |<Z%d> - exact| = %.4e\n",
            kj, t / kj, real(inner(d, d)), mid,
            abs(real(expval(obs_mps[mid], ek.rho)) - O_exact[mid]))
    flush(stdout)
end
println()

# ---- score each family ------------------------------------------------------
rows = ["family,r,lcm,k0_ok,E_mpf,cond_N,lam_min,E_best_single,err_proj_mean,err_single_mean,proj_over_single,c_max_abs"]
results = []

println("="^118)
println("FAMILY SCAN")
println("="^118)
@printf("%-22s %3s %6s %5s | %-11s %-10s | %-11s %-11s %-7s | %-9s\n",
        "ks", "r", "lcm", "k0%", "E_mpf", "cond(N)", "err_proj", "err_single", "ratio", "max|c|")
println("-"^118)

for ks in families
    rhos = [cache[kj] for kj in ks]
    N, _ = trotter_error_gram_from_states(rhos, rho_ref)
    sol  = coefficients_from_N(N)
    c    = sol.coeffs
    r    = length(ks)

    O_kj = [real(expval(obs_mps[a], rhos[j])) for a in 1:n_obs, j in 1:r]
    ep   = [abs(sum(c[j] * O_kj[a, j] for j in 1:r) - O_exact[a]) for a in 1:n_obs]
    es   = [minimum(abs(O_kj[a, j] - O_exact[a]) for j in 1:r) for a in 1:n_obs]

    L  = lcm(ks...)
    ok = (k0 % L == 0)
    epm, esm = sum(ep) / n_obs, sum(es) / n_obs

    push!(results, (ks=ks, E_mpf=sol.E_mpf, cond=sol.cond, epm=epm, esm=esm,
                    ratio=epm / max(esm, 1e-300), cmax=maximum(abs.(c)), r=r, ok=ok))
    push!(rows, @sprintf("\"%s\",%d,%d,%s,%.8e,%.6e,%.6e,%.8e,%.8e,%.8e,%.6f,%.6f",
                         join(ks, "-"), r, L, ok ? "yes" : "NO", sol.E_mpf, sol.cond,
                         minimum(sol.eigvals), minimum(sol.E_trot), epm, esm,
                         epm / max(esm, 1e-300), maximum(abs.(c))))

    @printf("%-22s %3d %6d %5s | %.4e  %.3e  | %.4e  %.4e  %6.3f  | %8.3f%s\n",
            join(ks, ","), r, L, ok ? "ok" : "NO", sol.E_mpf, sol.cond,
            epm, esm, epm / max(esm, 1e-300), maximum(abs.(c)),
            r < 3 ? "   (r<3: dc has only one direction)" : "")
    flush(stdout)
end

# ---- recommendation ---------------------------------------------------------
println()
println("="^118)
println("RECOMMENDATION")
println("="^118)

viable = filter(x -> x.r >= 3 && x.ok && x.cond < 1e4 && x.ratio < 1.0, results)
if isempty(viable)
    println("  No family satisfies all of: r>=3, k0 divisible by lcm, cond(N)<1e4, proj<single.")
    println("  Relax in this order -- cond(N) first (it only costs truncation robustness),")
    println("  then the k0 divisibility (it only costs MOC cross-checkability), and only")
    println("  last r>=3 (which costs the Appendix D mechanism entirely).")
    best = results[argmin([x.epm for x in results])]
else
    best = viable[argmin([x.epm for x in viable])]
end

@printf("\n  BEST: ks = %s\n", join(best.ks, ","))
@printf("    export KS=%s\n", join(best.ks, ","))
@printf("    E_mpf     = %.4e\n", best.E_mpf)
@printf("    err_proj  = %.4e   (this is the FLOOR the n-sweep must beat a classical\n", best.epm)
@printf("                          simulation with -- nothing downstream can go below it)\n")
@printf("    cond(N)   = %.3e\n", best.cond)
@printf("    vs best single circuit: %.3f\n", best.ratio)

cur = findfirst(x -> x.ks == [3, 8], results)
if cur !== nothing
    @printf("\n  Improvement over the current ks=3,8: err_proj %.4e -> %.4e  (%.1fx)\n",
            results[cur].epm, best.epm, results[cur].epm / max(best.epm, 1e-300))
end

println()
println("  SANITY CHECK BEFORE COMMITTING: err_proj must be comfortably below the")
println("  accuracy a classical simulation reaches at an AFFORDABLE chi. In the first")
println("  Step 1 run the classical route hit 1.5e-3 at chi=128 (n=8), so an err_proj")
println("  of 5e-3 lost outright. If the best family here is not well under that, the")
println("  candidate family is still the binding constraint and no n-sweep will help.")

write("family_scan$sfx.csv", join(rows, "\n") * "\n")
println("\nwrote family_scan$sfx.csv")
