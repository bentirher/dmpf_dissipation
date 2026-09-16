# =============================================================================
# step0_reference_convergence.jl   --   CONVERGE THE REFERENCE
#
# THE PROBLEM THIS FIXES
# ----------------------
# README.md, open issue 3: "The reference is not converged. k0: 48 -> 96 shifts
# E_k8 by +6.3%. The n=4 'exact' point is exact = untruncated, not physically
# correct."
#
# Every number in the project -- c_exact, E_mpf, E_kj, the M-vs-N comparison,
# the gamma sweep -- is scored against rho_ref = S(t/k0)^k0 |rho0>>. If that
# object still carries 6% of its own Trotter error, then "error" means "distance
# to a slightly wrong state" and a 6% systematic sits under every curve. Step 1
# asks whether the DMPF observable beats a direct simulation by a factor of a
# few; a 6% floor in the reference is large enough to manufacture or destroy
# that result. So this runs first, and nothing else is trusted until it passes.
#
# WHY IT IS CHEAP
# ---------------
# Convergence in k0 is a question about the STATE, and a Liouville MPS at n <= 8
# is exact at chi = 4^(n/2) <= 256. So we evolve |rho>> directly with no
# truncation at all and vary only k0. No MPOs, no MOC, no four-object recursion.
# At n = 6 the whole ladder takes minutes.
#
# WHAT IT MEASURES
# ----------------
# For each k0 in a geometric ladder of multiples of lcm(ks) -- the multiple is
# required by the MOC recursion's B_j blocks, and kept here so the two routes
# stay comparable -- and for each reference order in ORDERS_REF:
#
#   Tr(rho_ref)        must be 1.0; end-to-end check on the vectorization
#   <Z_m>, <Z_m Z_m+1> observable-level convergence
#   E_kj = ||rho_kj - rho_ref||^2   the quantity that moved by 6.3%
#   c, E_mpf                        the products of the algorithm
#   ||rho_ref(k0) - rho_ref(k0_max)||   self-convergence (Cauchy)
#
# and then reports the SMALLEST k0 at which every tracked scalar is stable to
# REL_TOL against the finest k0 on the ladder.
#
# THE SECOND QUESTION IT ANSWERS
# ------------------------------
# ORDERS_REF defaults to "2,4". An order-4 reference converges in k0 far faster
# than order-2, and get_open_step_gates_order4 already exists. If order_ref = 4
# at k0 = 48 matches order_ref = 2 at k0 = 768, every downstream run gets 16x
# cheaper for free. Worth knowing before committing the cluster time in Step 1.
#
# CAVEAT ON MIXING ORDERS: the candidates rho_kj always use ORDER (the product
# formula whose error DMPF is correcting). Only the reference uses ORDER_REF.
# That is the existing order/order_ref split in the codebase, not a new one.
#
# Environment: N_QUBITS GAMMA TVAL KS ORDER ORDERS_REF N_LADDER REL_TOL TAG
# Output: step0_reference<_tag>.csv
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("liouville_state_tools.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n          = parse(Int,     getenv("N_QUBITS",   6))
gamma      = parse(Float64, getenv("GAMMA",      0.05))
t          = parse(Float64, getenv("TVAL",       3.0))
order      = parse(Int,     getenv("ORDER",      2))
reltol     = parse(Float64, getenv("REL_TOL",    1e-3))
nladder    = parse(Int,     getenv("N_LADDER",   6))
tag        = getenv("TAG", "")
ks         = parse.(Int, split(getenv("KS", "3,8"), ","))
orders_ref = parse.(Int, split(getenv("ORDERS_REF", "2,4"), ","))

sfx   = isempty(tag) ? "" : "_" * tag
chi   = state_max_bond_dim(n)          # EXACT: no truncation possible here
L     = lcm(ks...)
k0s   = [L * 2^i for i in 0:(nladder - 1)]
r     = length(ks)
ct    = 1e-16

@printf("step0_reference_convergence\n")
@printf("  n=%d gamma=%.3f t=%.1f ks=%s order=%d orders_ref=%s\n",
        n, gamma, t, string(ks), order, string(orders_ref))
@printf("  Liouville MPS ceiling = %d  (4^min(%d,%d)) -- every run below is UNTRUNCATED\n",
        chi, n ÷ 2, n - n ÷ 2)
@printf("  k0 ladder (multiples of lcm(ks)=%d): %s\n\n", L, string(k0s))
flush(stdout)

Random.seed!(1234)                     # same draw as every other script
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

mid    = n ÷ 2
O_id   = identity_observable(lsites)
O_z    = z_observable(lsites, mid)
O_zz   = zz_observable(lsites, mid, mid + 1)

# -----------------------------------------------------------------------------
# Candidates: computed ONCE. They do not depend on k0 or order_ref.
# -----------------------------------------------------------------------------
println("candidates rho_kj (exact, chi = $chi):")
rhos = MPS[]
for kj in ks
    e = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                       maxdim=chi, order=order, cutoff=ct)
    push!(rhos, e.rho)
    @printf("  k=%-4d chi_S=%-4d chi_rho=%-4d Tr=%+.12f  <Z%d>=%+.8f\n",
            kj, e.chi_S, e.chi, real(expval(O_id, e.rho)), mid, real(expval(O_z, e.rho)))
end
println()
flush(stdout)

# -----------------------------------------------------------------------------
# Reference ladder
# -----------------------------------------------------------------------------
rows = ["order_ref,k0,trace,Z_mid,ZZ_mid,E_k" * join(string.(ks), ",E_k") *
        ",c1,E_mpf,selfconv,time_s"]

# Everything we require to be converged, per order_ref, indexed by k0.
tracked = Dict{Int,Dict{Int,Vector{Float64}}}()
refstates = Dict{Int,Dict{Int,MPS}}()

for oref in orders_ref
    println("="^96)
    @printf("REFERENCE ORDER %d\n", oref)
    println("="^96)
    println("   k0 |   trace      <Z_mid>     <ZZ_mid>   |  " *
            join([@sprintf("E_k%-8d", kj) for kj in ks]) * " |     c1        E_mpf      time")
    println("-"^96)

    tracked[oref]   = Dict{Int,Vector{Float64}}()
    refstates[oref] = Dict{Int,MPS}()

    for k0 in k0s
        t0 = time()
        e = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                           maxdim=chi, order=oref, cutoff=ct)
        el = time() - t0
        rho_ref = e.rho
        refstates[oref][k0] = rho_ref

        tr   = real(expval(O_id, rho_ref))
        zval = real(expval(O_z,  rho_ref))
        zzv  = real(expval(O_zz, rho_ref))

        N, _ = trotter_error_gram_from_states(rhos, rho_ref)
        sol  = coefficients_from_N(N)
        Ek   = [N[j, j] for j in 1:r]

        # Tracked scalars, in a fixed order, for the convergence verdict below.
        tracked[oref][k0] = vcat(Ek, [sol.coeffs[1], sol.E_mpf, zval, zzv])

        push!(rows, @sprintf("%d,%d,%.12f,%.10f,%.10f,%s,%.10f,%.8e,,%.1f",
                             oref, k0, tr, zval, zzv,
                             join([@sprintf("%.8e", x) for x in Ek], ","),
                             sol.coeffs[1], sol.E_mpf, el))

        @printf("%5d | %+.8f %+.8f %+.8f  | %s | %+.6f %.4e %6.0fs\n",
                k0, tr, zval, zzv,
                join([@sprintf("%.4e ", x) for x in Ek]),
                sol.coeffs[1], sol.E_mpf, el)
        flush(stdout)
    end
    println()
end

# -----------------------------------------------------------------------------
# Self-convergence: ||rho_ref(k0) - rho_ref(k0_max)|| / ||rho_ref(k0_max)||
# -----------------------------------------------------------------------------
#
# Independent of the candidates and of the fit -- it asks only whether the
# reference has stopped moving. Reported for the finest order_ref available,
# and CROSS-order (order 2 vs order 4 at the finest k0) as the strongest single
# check: two different product formulas converging to the same state is much
# better evidence than one formula converging to itself.

println("="^96)
println("SELF-CONVERGENCE  ||rho_ref(k0) - rho_ref(k0_max)|| / ||rho_ref(k0_max)||")
println("="^96)
k0max = k0s[end]
for oref in orders_ref
    fine = refstates[oref][k0max]
    nf   = sqrt(max(real(inner(fine, fine)), 0.0))
    @printf("order_ref = %d   (k0_max = %d, ||rho|| = %.8f)\n", oref, k0max, nf)
    for k0 in k0s[1:end-1]
        d = mps_difference(refstates[oref][k0], fine)
        @printf("   k0 = %-5d  %.4e\n", k0, sqrt(max(real(inner(d, d)), 0.0)) / nf)
    end
end

if length(orders_ref) >= 2
    a, b = orders_ref[1], orders_ref[2]
    d = mps_difference(refstates[a][k0max], refstates[b][k0max])
    nf = sqrt(max(real(inner(refstates[b][k0max], refstates[b][k0max])), 0.0))
    @printf("\nCROSS-ORDER at k0 = %d:  ||rho_ref(order %d) - rho_ref(order %d)|| / ||rho|| = %.4e\n",
            k0max, a, b, sqrt(max(real(inner(d, d)), 0.0)) / nf)
    println("  (this is the number that says whether k0_max is converged in an")
    println("   absolute sense, rather than merely self-consistent.)")
end
println()

# -----------------------------------------------------------------------------
# VERDICT: smallest k0 stable to REL_TOL against the finest k0
# -----------------------------------------------------------------------------

labels = vcat(["E_k$kj" for kj in ks], ["c1", "E_mpf", "Z_mid", "ZZ_mid"])

println("="^96)
@printf("VERDICT  (rel. tolerance %.1e against k0 = %d)\n", reltol, k0max)
println("="^96)

recommended = Dict{Int,Int}()
for oref in orders_ref
    ref = tracked[oref][k0max]
    @printf("\norder_ref = %d\n", oref)
    println("   k0 | " * join([rpad(l, 11) for l in labels]) * "| verdict")
    println("-"^96)
    best = -1
    for k0 in k0s
        v   = tracked[oref][k0]
        rel = [abs(v[i] - ref[i]) / max(abs(ref[i]), 1e-300) for i in eachindex(v)]
        ok  = all(rel .< reltol)
        (ok && best < 0) && (best = k0)
        @printf("%5d | %s| %s\n", k0,
                join([@sprintf("%-10.2e ", x) for x in rel]),
                ok ? "PASS" : "fail")
    end
    recommended[oref] = best
    if best > 0
        @printf("  -> smallest converged k0 = %d\n", best)
    else
        println("  -> NO k0 on this ladder converged. Increase N_LADDER and rerun;")
        println("     until then no downstream 'exact' number means anything.")
    end
end

println()
println("="^96)
println("RECOMMENDATION")
println("="^96)
for oref in orders_ref
    b = recommended[oref]
    if b > 0
        @printf("  ORDER_REF=%d  ->  K0=%d   (export K0=%d ORDER_REF=%d for step1)\n", oref, b, b, oref)
    end
end
println()
println("  Pick the (ORDER_REF, K0) pair with the LOWEST k0 that passes, and check")
println("  it against the CROSS-ORDER number above. If order 4 passes at a much")
println("  smaller k0, use it: k0 sets the length of every evolution in Step 1.")
println()
println("  NOTE: k0 must remain an integer multiple of lcm(ks) = $L if these")
println("  coefficients are ever to be cross-checked against the MOC/N-route,")
println("  whose B_j blocks require k0 % k_j == 0.")

write("step0_reference$sfx.csv", join(rows, "\n") * "\n")
println("\nwrote step0_reference$sfx.csv")
