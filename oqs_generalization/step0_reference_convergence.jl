# =============================================================================
# step0_reference_convergence.jl   --   CONVERGE THE REFERENCE (v2)
#
# WHAT CHANGED IN v2, AND WHY
# ---------------------------
# v1 ran the ladder and reported that NOTHING converged: at n=6, gamma=0.05,
# untruncated at the chi=64 ceiling, E_mpf was still ~1% off at k0=768 and the
# self-convergence ratio per doubling was ~2.3 for BOTH order 2 and order 4.
#
# That is not a k0 problem. It is a product-formula bug:
#
#   get_open_step_gates_order2  =  odd(dt/2), even(dt), diss(dt), odd(dt/2)
#
# is NOT palindromic -- `even` and `diss` are composed as a bare first-order
# product -- so it is globally FIRST order whenever [diss, even] != 0, i.e.
# whenever gamma > 0. At gamma = 0 the dissipator layer is empty, it collapses
# to plain Strang, and it is genuinely second order; that is why the whole
# closed-system half of the project never saw this.
#
# And order 4 inherits the failure: get_open_step_gates_order4 applies the
# Yoshida triple-jump to five order-2 sub-steps, which reaches fourth order ONLY
# if the base is second order AND symmetric. The v1 log shows the order-2 and
# order-4 self-convergence curves lying on top of each other, which is exactly
# that failure.
#
# See symmetric_splitting.jl for the derivation, the independent measurement
# from vectorized_evolution.jl (effective order 1.6 for :project vs 2.0 for
# :strang), and the fix.
#
# So v2 does two things v1 did not:
#
#   1. Sweeps SCHEMES = (splitting, order) pairs, so :project and :strang are
#      measured side by side in one run rather than argued about.
#   2. Reports the EFFECTIVE ORDER directly, from successive self-convergence
#      differences: p_eff = log2( e(k0/2) / e(k0) ). This is the number that
#      makes the bug undeniable, and the number that says which scheme to use.
#      Order 2 should give p_eff ~ 2; order 4 should give ~4.
#
# It also divides out Tr(rho). Every gate is exactly trace-preserving, so the
# monotonic drift to 1.1e-6 at k0=768 seen in the v1 log (~1.4e-9 per step) is
# pure accumulated `apply` error, and it is a ~1% contamination of quantities of
# order 1e-4. The raw trace is still recorded -- watch it, do not hide it.
#
# WHAT IS AND IS NOT BEING CHANGED
# --------------------------------
# The CANDIDATES rho_kj stay on :project by default. DMPF corrects whatever
# formula the candidates use, the base order is not part of the claim, and
# keeping them preserves continuity with every number already computed. Only the
# REFERENCE needs to move, because the reference has to be CONVERGED, and at
# 1/k0 that is unreachable.
#
# WHY IT IS CHEAP
# ---------------
# Convergence in k0 is a question about the STATE, and a Liouville MPS at n <= 8
# is exact at chi = 4^(n/2) <= 256. No MPOs beyond the step channel, no MOC.
#
# Environment: N_QUBITS GAMMA TVAL KS ORDER CAND_SPLITTING REF_SCHEMES
#              N_LADDER REL_TOL TAG
# Output: step0_reference<_tag>.csv
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("liouville_state_tools.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n        = parse(Int,     getenv("N_QUBITS",  6))
gamma    = parse(Float64, getenv("GAMMA",     0.05))
t        = parse(Float64, getenv("TVAL",      3.0))
order    = parse(Int,     getenv("ORDER",     2))      # candidates
reltol   = parse(Float64, getenv("REL_TOL",   1e-3))
nladder  = parse(Int,     getenv("N_LADDER",  6))
tag      = getenv("TAG", "")
ks       = parse.(Int, split(getenv("KS", "3,8"), ","))
cand_spl = Symbol(getenv("CAND_SPLITTING", "project"))
evo_mode = Symbol(getenv("EVO_MODE", "gates"))   # :gates (default) or :mpo
# Small-k0 ladder used ONLY to measure the effective order against a converged
# gold reference. The main ladder cannot do this for a 4th-order scheme: at
# dt = 3/24 its Trotter error is already below the numerical floor, so the
# self-convergence differences are noise and p_eff comes out meaningless
# (the v2 log gave -0.50, -0.55, +1.30, -1.53 for strang:4). To SEE order 4 you
# need LARGER dt, i.e. SMALLER k0.
order_ladder = parse.(Int, split(getenv("ORDER_LADDER", "2,3,4,6,8,12,24"), ","))

# Reference schemes to compare, as "splitting:order" pairs.
schemes = map(split(getenv("REF_SCHEMES", "project:2,project:4,strang:2,strang:4"), ",")) do s
    a, b = split(strip(s), ":")
    (splitting=Symbol(a), order=parse(Int, b))
end
scheme_name(sc) = "$(sc.splitting):$(sc.order)"

sfx  = isempty(tag) ? "" : "_" * tag
chi  = state_max_bond_dim(n)          # EXACT: no truncation possible here
L    = lcm(ks...)
k0s  = [L * 2^i for i in 0:(nladder - 1)]
r    = length(ks)
ct   = 1e-16

@printf("step0_reference_convergence (v2: splitting sweep + effective order)\n")
@printf("  n=%d gamma=%.3f t=%.1f ks=%s\n", n, gamma, t, string(ks))
@printf("  candidates: splitting=%s order=%d    evolution mode=%s\n", cand_spl, order, evo_mode)
@printf("  reference schemes: %s\n", join(scheme_name.(schemes), ", "))
@printf("  Liouville MPS ceiling = %d  (4^min(%d,%d)) -- every run below is UNTRUNCATED\n",
        chi, n ÷ 2, n - n ÷ 2)
@printf("  k0 ladder (multiples of lcm(ks)=%d): %s\n\n", L, string(k0s))
flush(stdout)

Random.seed!(1234)                     # same draw as every other script
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

mid  = n ÷ 2
O_id = identity_observable(lsites)
O_z  = z_observable(lsites, mid)
O_zz = zz_observable(lsites, mid, mid + 1)

relnorm(a::MPS, b::MPS) = begin
    d = mps_difference(a, b)
    sqrt(max(real(inner(d, d)), 0.0)) / max(sqrt(max(real(inner(b, b)), 0.0)), 1e-300)
end

# -----------------------------------------------------------------------------
# Candidates: computed ONCE. Independent of k0 and of the reference scheme.
# -----------------------------------------------------------------------------
println("candidates rho_kj (exact, chi = $chi, splitting = $cand_spl, order = $order):")
rhos = MPS[]
for kj in ks
    e = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                       maxdim=chi, order=order, cutoff=ct,
                       splitting=cand_spl, mode=evo_mode, id_mps=O_id)
    push!(rhos, e.rho)
    @printf("  k=%-4d chi_S=%-5d chi_rho=%-5d raw Tr=%+.12f  <Z%d>=%+.8f\n",
            kj, e.chi_S, e.chi, real(e.trace), mid, real(expval(O_z, e.rho)))
end
println("  (raw Tr is BEFORE renormalisation; every gate is exactly trace-preserving,")
println("   so any deviation from 1 is accumulated `apply` error and nothing else.)\n")
flush(stdout)

# -----------------------------------------------------------------------------
# Reference ladder, per scheme
# -----------------------------------------------------------------------------
rows = ["splitting,order_ref,k0,raw_trace,Z_mid,ZZ_mid,E_k" * join(string.(ks), ",E_k") *
        ",c1,E_mpf,selfconv,p_eff,time_s"]

tracked   = Dict{String,Dict{Int,Vector{Float64}}}()
refstates = Dict{String,Dict{Int,MPS}}()
selfconv  = Dict{String,Vector{Float64}}()

for sc in schemes
    nm = scheme_name(sc)
    println("="^104)
    @printf("REFERENCE SCHEME  %s\n", nm)
    println("="^104)
    println("   k0 | sizeS |  raw Tr       <Z_mid>     <ZZ_mid>   |  " *
            join([@sprintf("E_k%-8d", kj) for kj in ks]) * " |     c1        E_mpf      time")
    println("-"^112)
    # sizeS = number of gates (:gates mode) or step-MPO bond dimension (:mpo).
    # In :mpo mode a value sitting exactly at min(MPO_MAXDIM, 16^(n/2)) means the
    # step operator itself was truncated -- a silent systematic on EVERY step,
    # and the prime suspect for the ~5e-6 floor in the v2 log (at n=6 the middle
    # cut is an odd bond, so strang:4's ten odd-layer crossings admit rank up to
    # 4096 while MPO_MAXDIM defaults to 512). Flagged explicitly below.

    tracked[nm]   = Dict{Int,Vector{Float64}}()
    refstates[nm] = Dict{Int,MPS}()
    traces        = Dict{Int,Float64}()

    for k0 in k0s
        t0 = time()
        e = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                           maxdim=chi, order=sc.order, cutoff=ct,
                           splitting=sc.splitting, mode=evo_mode, id_mps=O_id)
        el = time() - t0
        rho_ref = e.rho
        refstates[nm][k0] = rho_ref
        traces[k0] = real(e.trace)

        zval = real(expval(O_z,  rho_ref))
        zzv  = real(expval(O_zz, rho_ref))

        N, _ = trotter_error_gram_from_states(rhos, rho_ref)
        sol  = coefficients_from_N(N)
        Ek   = [N[j, j] for j in 1:r]

        tracked[nm][k0] = vcat(Ek, [sol.coeffs[1], sol.E_mpf, zval, zzv])

        @printf("%5d | %5d%s| %+.10f %+.8f %+.8f  | %s | %+.6f %.4e %6.0fs\n",
                k0, e.chi_S, e.step_truncated ? "!" : " ", traces[k0], zval, zzv,
                join([@sprintf("%.4e ", x) for x in Ek]),
                sol.coeffs[1], sol.E_mpf, el)
        e.step_truncated && @warn "step MPO truncated at the cap: this is a systematic on every step. Raise MPO_MAXDIM or use EVO_MODE=gates." scheme=nm k0=k0 chi_S=e.chi_S
        flush(stdout)
    end

    # ---- self-convergence and effective order -------------------------------
    fine = refstates[nm][k0s[end]]
    sc_vals = [relnorm(refstates[nm][k0], fine) for k0 in k0s[1:end-1]]
    selfconv[nm] = sc_vals

    println()
    println("  self-convergence ||rho(k0) - rho(k0_max)|| / ||rho(k0_max)||, and")
    println("  EFFECTIVE ORDER  p_eff = log2( e(k0/2) / e(k0) ):")
    peff = fill(NaN, length(k0s) - 1)
    for i in eachindex(sc_vals)
        if i > 1 && sc_vals[i] > 0
            peff[i] = log2(sc_vals[i-1] / sc_vals[i])
        end
        @printf("     k0 = %-5d  e = %.4e   p_eff = %s\n", k0s[i], sc_vals[i],
                isnan(peff[i]) ? "  --" : @sprintf("%5.2f", peff[i]))
    end
    println("  (trust the EARLY p_eff values; the last ones are contaminated because")
    println("   e(k0) approaches the reference's own error. Target: ~2 for order 2,")
    println("   ~4 for order 4. Anything near 1 means the scheme is first order.)")
    println()

    for (i, k0) in enumerate(k0s)
        v = tracked[nm][k0]
        push!(rows, @sprintf("%s,%d,%d,%.12f,%.10f,%.10f,%s,%.10f,%.8e,%s,%s,",
                             string(sc.splitting), sc.order, k0, traces[k0],
                             v[r+3], v[r+4],
                             join([@sprintf("%.8e", v[j]) for j in 1:r], ","),
                             v[r+1], v[r+2],
                             i <= length(sc_vals) ? @sprintf("%.8e", sc_vals[i]) : "",
                             (i <= length(peff) && !isnan(peff[i])) ? @sprintf("%.4f", peff[i]) : ""))
    end
end

# -----------------------------------------------------------------------------
# Cross-scheme agreement at the finest k0
# -----------------------------------------------------------------------------
#
# The strongest single check available. Two DIFFERENT product formulas agreeing
# at k0_max is far better evidence of convergence than one formula agreeing with
# itself -- self-convergence cannot detect a systematic shared by every k0.

k0max = k0s[end]
names = scheme_name.(schemes)

# The gold reference: the highest-order symmetric scheme available. Everything
# below is scored against THIS, never against a scheme's own k0_max -- scoring a
# scheme against itself is what let v1 report PASS at k0=768 for a formula that
# had not converged at all.
best_scheme = let cands = filter(sc -> sc.splitting === :strang, schemes)
    isempty(cands) ? schemes[end] : cands[argmax([sc.order for sc in cands])]
end

println("="^104)
@printf("CROSS-SCHEME AGREEMENT at k0 = %d   ||rho[row] - rho[col]|| / ||rho||\n", k0max)
println("="^104)
println("  " * rpad("", 14) * join([rpad(m, 12) for m in names]))
for a in names
    print("  " * rpad(a, 14))
    for b in names
        print(rpad(a == b ? "    --" : @sprintf("%.3e", relnorm(refstates[a][k0max], refstates[b][k0max])), 12))
    end
    println()
end
println()
println("  Read this as a clustering, not a list. Schemes that agree with each other")
println("  but differ from another cluster share a systematic. In the v2 log the two")
println("  :project schemes agreed to 2.2e-5 while both sat 1.8e-4 from both :strang")
println("  schemes -- exactly what you expect if project:4 is built from project:2 and")
println("  inherits its first-order defect. That 1.8e-4 IS project:2's remaining error")
println("  at k0 = 768.")
println()

# -----------------------------------------------------------------------------
# EFFECTIVE ORDER, measured properly
# -----------------------------------------------------------------------------
#
# The self-convergence tables above cannot measure the order of a high-order
# scheme, for a simple reason: they compare rho(k0) against rho(k0_max) OF THE
# SAME SCHEME, and once the Trotter error drops below the numerical floor both
# are the same state plus independent noise. p_eff then becomes the log-ratio of
# two noise realizations. That is precisely what the v2 log showed for strang:4
# (e bouncing over 4e-6 to 1e-5 with no trend; p_eff = -0.50, -0.55, +1.30,
# -1.53) -- not a broken formula, a saturated measurement.
#
# Fixing it needs two changes:
#   1. Score against a GOLD reference from the best scheme, not against self.
#   2. Use LARGER dt (smaller k0), so the Trotter error is well above the floor.
#
# The floor is estimated empirically as the gold scheme's own last
# self-convergence value: both of those states are converged, so their
# difference is pure numerical noise and nothing else.

gold_name = scheme_name(best_scheme)
rho_gold  = refstates[gold_name][k0max]
floor_est = isempty(selfconv[gold_name]) ? 0.0 : selfconv[gold_name][end]

println("="^104)
@printf("EFFECTIVE ORDER against the %s gold reference at k0 = %d\n", gold_name, k0max)
@printf("estimated numerical floor = %.3e  (the gold scheme's own last self-convergence value)\n", floor_est)
println("="^104)

ord_rows = String[]
for sc in schemes
    nm = scheme_name(sc)
    @printf("\nscheme %s\n", nm)
    println("     k0 |    dt    |  e = ||rho(k0)-gold||/||rho||  | p_eff  | status")
    println("-"^104)
    prev_e = NaN
    for k0 in order_ladder
        ev = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                            maxdim=chi, order=sc.order, cutoff=ct,
                            splitting=sc.splitting, mode=evo_mode, id_mps=O_id)
        e = relnorm(ev.rho, rho_gold)
        p = (isnan(prev_e) || e <= 0) ? NaN : log2(prev_e / e)
        at_floor = e < 5 * floor_est
        @printf("  %5d | %.6f |          %.4e            | %6s | %s\n",
                k0, t / k0, e, isnan(p) ? "  --" : @sprintf("%5.2f", p),
                at_floor ? "AT FLOOR (p_eff meaningless)" : "ok")
        push!(ord_rows, @sprintf("%s,%d,%d,%.8f,%.8e,%s,%s", string(sc.splitting), sc.order,
                                 k0, t / k0, e, isnan(p) ? "" : @sprintf("%.4f", p),
                                 at_floor ? "floor" : "ok"))
        prev_e = e
        flush(stdout)
    end
end
println()
println("  Read ONLY the rows marked ok, and only consecutive ok pairs. Targets:")
println("    order 2 -> p_eff ~ 2      order 4 -> p_eff ~ 4")
println("  If a scheme never leaves the floor on this ladder its error is below the")
println("  floor everywhere, which is a PASS, not a failure -- push ORDER_LADDER to")
println("  smaller k0 (larger dt) if you want to see the exponent itself.")
println()
write("step0_order$sfx.csv",
      "splitting,order,k0,dt,rel_err_vs_gold,p_eff,status\n" * join(ord_rows, "\n") * "\n")
println("  wrote step0_order$sfx.csv")
println()

# -----------------------------------------------------------------------------
# VERDICT: smallest k0 stable to REL_TOL, per scheme
# -----------------------------------------------------------------------------
#
# Scored against the BEST available reference -- the finest k0 of the
# highest-order symmetric scheme present -- not against each scheme's own
# k0_max. Scoring a scheme against itself is what let v1 report "PASS" at
# k0 = 768 for a formula that had not converged at all.

gold = tracked[gold_name][k0max]
labels = vcat(["E_k$kj" for kj in ks], ["c1", "E_mpf", "Z_mid", "ZZ_mid"])

println("="^104)
@printf("VERDICT  (rel. tolerance %.1e against the %s reference at k0 = %d)\n",
        reltol, scheme_name(best_scheme), k0max)
println("="^104)

recommended = Dict{String,Int}()
for sc in schemes
    nm = scheme_name(sc)
    @printf("\nscheme %s\n", nm)
    println("   k0 | " * join([rpad(l, 11) for l in labels]) * "| verdict")
    println("-"^104)
    best = -1
    for k0 in k0s
        v   = tracked[nm][k0]
        rel = [abs(v[i] - gold[i]) / max(abs(gold[i]), 1e-300) for i in eachindex(v)]
        ok  = all(rel .< reltol)
        (ok && best < 0) && (best = k0)
        @printf("%5d | %s| %s\n", k0,
                join([@sprintf("%-10.2e ", x) for x in rel]),
                ok ? "PASS" : "fail")
    end
    recommended[nm] = best
    best > 0 ? @printf("  -> smallest converged k0 = %d\n", best) :
               println("  -> never converges on this ladder")
end

println()
println("="^104)
println("RECOMMENDATION")
println("="^104)
passing = [(nm, k) for (nm, k) in recommended if k > 0]
if isempty(passing)
    println("  NOTHING PASSED. Two possible causes, in order of likelihood:")
    println("    1. Even :strang has not converged on this ladder -> raise N_LADDER.")
    println("    2. The cross-scheme numbers above are large, meaning the schemes")
    println("       disagree and the 'gold' reference is itself wrong.")
    println("  Do NOT proceed to Step 1 until one passes: the whole point of Step 1 is")
    println("  a factor-of-a-few comparison, and a percent-level systematic under the")
    println("  reference can manufacture or destroy it outright.")
else
    sort!(passing; by=x -> x[2])
    for (nm, k) in passing
        spl, ordr = split(nm, ":")
        @printf("  %-12s ->  K0=%-5d   (export K0=%d ORDER_REF=%s SPLITTING_REF=%s)\n",
                nm, k, k, ordr, spl)
    end
    println()
    println("  Take the pair with the LOWEST k0 that passes. k0 sets the length of every")
    println("  evolution in Step 1, so this is the single biggest lever on its cost.")
end
println()
println("  Expect :strang to beat :project by a wide margin. If it does not, the")
println("  splitting was not the problem and the next suspect is the `apply` accuracy")
println("  (watch the raw trace column: it should stop drifting, not just get divided out).")
println()
@printf("  NOTE: k0 must remain an integer multiple of lcm(ks) = %d if these coefficients\n", L)
println("  are ever to be cross-checked against the MOC/N-route, whose B_j blocks require")
println("  k0 % k_j == 0.")

write("step0_reference$sfx.csv", join(rows, "\n") * "\n")
println("\nwrote step0_reference$sfx.csv")
