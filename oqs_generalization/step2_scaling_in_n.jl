# =============================================================================
# step2_scaling_in_n.jl   --   THE EXPERIMENT THAT CAN ACTUALLY BE WON
#
# WHY STEP 1 COULD NOT SETTLE THIS
# --------------------------------
# Step 1 swept chi at fixed n = 8. The Liouville MPS ceiling there is 4^4 = 256,
# so at chi = 256 the classical route is EXACT and beats DMPF by whatever margin
# you like. The crossover came out near chi ~ 50 and above it the classical
# route won -- which was guaranteed by the ceiling, not discovered. At any n
# where an exact reference is affordable, the exact answer is affordable, so the
# fixed-n sweep asks a question whose answer is fixed in advance.
#
# The right axis is n. Fix chi at a value that stays affordable as the system
# grows, and ask how each route degrades:
#
#   err_direct   should degrade FAST. The classical route needs the state's full
#                operator entanglement, and that grows with n until dissipation
#                caps it. At fixed chi it falls further behind every time n
#                grows.
#   err_dmpf     should stay FLAT, pinned near err_proj. err_proj is a Trotter
#                error, not an entanglement error: it depends on dt and on the
#                candidate family, and only weakly on n.
#
# If that holds, the bond dimension DMPF needs to reach a given accuracy stays
# roughly constant in n while the bond dimension a classical simulation needs
# grows. That is the claim, and it is the honest form of "at matched bond
# dimension the coefficients are an easier classical target than the observable".
#
# THE HEADLINE NUMBER
# -------------------
# Not the raw errors -- the bond dimension each route needs to REACH A COMMON
# ACCURACY TARGET, as a function of n:
#
#   chi_direct*(n)   smallest chi at which err_direct <= target
#   chi_dmpf*(n)     smallest chi at which err_dmpf   <= target
#   target           set at TARGET_FACTOR x err_proj(n), since err_proj is the
#                    floor DMPF cannot cross. Comparing at DMPF's own floor is
#                    the fairest matched-accuracy question available.
#
# chi_direct*/chi_dmpf* growing with n IS the result. If both grow together, or
# if chi_direct* saturates, there is no win and that is worth knowing in a day
# rather than a quarter.
#
# WHAT WOULD FALSIFY IT
# ---------------------
# Dissipation caps operator entanglement. At large enough gamma*t the Liouville
# MPS stops growing with n, chi_direct* saturates, and the classical route wins
# at every n. At gamma=0.05, t=3, n=8 the exact state still SATURATED the ceiling
# (chi_rho = 256 in the Step 1 log), so we are not in that regime yet -- but the
# script reports chi_rho at every n precisely so the onset is visible. If chi_rho
# stops tracking the ceiling, the experiment has answered no.
#
# COST
# ----
# n = 10 needs an exact reference at chi = 4^5 = 1024. The central SVDs are
# 1024x1024 complex; expect tens of minutes. n = 12 (ceiling 4096) is out of
# reach, which caps this design at four points of n.
#
# Environment: N_LIST GAMMA TVAL KS K0 ORDER SPLITTING ORDER_REF SPLITTING_REF
#              CHI_LIST TARGET_FACTOR CUTOFF TAG
# Output: step2_scaling<_tag>.csv, step2_summary<_tag>.csv
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("liouville_state_tools.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
gamma         = parse(Float64, getenv("GAMMA",     0.05))
t             = parse(Float64, getenv("TVAL",      3.0))
k0            = parse(Int,     getenv("K0",        48))
order         = parse(Int,     getenv("ORDER",     2))
order_ref     = parse(Int,     getenv("ORDER_REF", 4))
splitting     = Symbol(getenv("SPLITTING",     "strang"))
splitting_ref = Symbol(getenv("SPLITTING_REF", "strang"))
evo_mode      = Symbol(getenv("EVO_MODE", "gates"))
ct            = parse(Float64, getenv("CUTOFF", EXACT_CUTOFF))
tfac          = parse(Float64, getenv("TARGET_FACTOR", 1.5))
tag           = getenv("TAG", "")
sfx           = isempty(tag) ? "" : "_" * tag
# SEVERAL FAMILIES IN ONE JOB. The reference evolution dominates the cost and
# candidate evolutions are short, so every family added is nearly free: one
# rho_ref and one shared pool of candidate states per (n, chi) serves them all.
#
# This is the right design because conditioning robustness CANNOT be predicted
# from cond(N). The naive estimate dc'N dc ~ p^2/lam_min missed the Step 1
# measurement by six orders of magnitude -- the perturbation to N is structured
# (truncation errors correlated across entries, corr ~ 0.5) and largely misses
# the small eigenvalue. Which family survives truncation is an empirical
# question, so ask it of several at once.
families = map(f -> parse.(Int, split(strip(f), ",")),
               split(getenv("FAMILIES", "3,8;4,8,16;2,4,8,16;4,6,8,12"), ";"))
n_list   = parse.(Int, split(getenv("N_LIST", "4,6,8,10"), ","))
chi_grid = parse.(Int, split(getenv("CHI_LIST", "4,8,16,32,64,128"), ","))
all_ks   = sort(unique(vcat(families...)))
ref_check = getenv("REF_CHECK", "1") == "1"
fam_name(ks) = join(ks, "-")

@printf("step2_scaling_in_n\n")
@printf("  n in %s   gamma=%.3f t=%.1f k0=%d\n", string(n_list), gamma, t, k0)
@printf("  families: %s   (candidate k values evolved once each: %s)\n",
        join(fam_name.(families), "  "), string(all_ks))
@printf("  candidates: %s order %d      reference: %s order %d      mode=%s\n",
        splitting, order, splitting_ref, order_ref, evo_mode)
@printf("  chi grid %s (capped per n at the ceiling 4^(n/2))\n", string(chi_grid))
@printf("  accuracy target = %.2f x err_proj(n)\n", tfac)
for fks in families
    k0 % lcm(fks...) == 0 || @printf("  WARNING: k0=%d not a multiple of lcm(%s)=%d -- these coefficients will not\n           be cross-checkable against the MOC route.\n", k0, fam_name(fks), lcm(fks...))
end
println()
flush(stdout)

rows = ["n,family,r,chi,at_ceiling,eps_chi,obs,O_exact,err_direct,err_dmpf,ratio,err_proj,err_best_single,cond_N,singular,relerr_N,dc_max,dcNdc,dc_frac_null,corr_min,time_s"]
summary = ["n,family,r,chi_ceiling,chi_rho_exact,ref_selferr,cond_N,lam_min,singular,E_mpf,err_proj,err_best_single,target,chi_direct_star,chi_dmpf_star,speedup,ratio_at_chi32,ratio_at_chi64"]

# Smallest grid chi whose error is at or below `target`; linear interpolation in
# log(chi) between the bracketing points, so the answer is not quantised to the
# grid. Returns Inf when the target is never reached on the grid.
# THE FIX: points with err == 0 are the ceiling run scored against ITSELF, not a
# measurement. Including them put log(0) = -Inf in the denominator, driving the
# interpolation weight to 0 and returning the LOWER bracket -- so chi_direct*
# always came back as the last grid point before the ceiling (128.0 at both n=8
# and n=10, for a target three orders below anything actually achieved there).
# Zero-error points are dropped; if the target is reached only there, the true
# chi* is at or just below the ceiling and the grid cannot resolve it, which is
# reported as -1 rather than a fabricated number.
function chi_star(chis, errs, target)
    idx = [i for i in eachindex(chis) if errs[i] > 0]
    for (j, i) in enumerate(idx)
        if errs[i] <= target
            j == 1 && return Float64(chis[i])
            ip = idx[j-1]
            errs[ip] <= target && return Float64(chis[i])
            f = (log(errs[ip]) - log(target)) / (log(errs[ip]) - log(errs[i]))
            return exp(log(chis[ip]) + f * (log(chis[i]) - log(chis[ip])))
        end
    end
    # target met only at a zero-error (ceiling) point?
    any(i -> errs[i] <= 0, eachindex(errs)) && return -1.0
    return Inf
end
fmt_chi(x) = x < 0 ? "ceil-only" : (isfinite(x) ? @sprintf("%.1f", x) : ">grid")

for n in n_list
    ceil_n = state_max_bond_dim(n)
    grid   = filter(<=(ceil_n), chi_grid)
    isempty(grid) && (grid = [ceil_n])
    ceil_n in grid || push!(grid, ceil_n)
    sort!(grid)

    println("="^114)
    @printf("n = %d    ceiling = %d    chi grid = %s\n", n, ceil_n, string(grid))
    println("="^114)
    flush(stdout)

    Random.seed!(1234)   # reseed per n so the couplings are the same draw
    J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
    gammas = fill(gamma, n)
    lsites = liouville_siteinds(n)
    rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

    mid  = n ÷ 2
    O_id = identity_observable(lsites)
    # Observables chosen to be COMPARABLE ACROSS n: the middle site, the middle
    # bond, and the mean over all sites. Site-indexed observables would compare
    # different physics at different n.
    # Single-observable errors pass through zero as the sign flips, which made
    # err_direct non-monotone by an order of magnitude between adjacent chi
    # (n=8: 1.4e-1, 2.6e-1, 4.9e-2, 9.4e-2, ...). Averaging |error| over all n
    # single-site Z damps that, at essentially no cost since the Z operators are
    # built anyway. Z_MAE is the metric to trust for chi*; Z_mid is kept for
    # continuity with the earlier runs.
    all_z     = [z_observable(lsites, m) for m in 1:n]
    obs_mps   = [z_observable(lsites, mid), zz_observable(lsites, mid, mid + 1)]
    obs_names = vcat(["Z_mid", "ZZ_mid", "Z_mean"], ["Z$m" for m in 1:n])
    n_obs     = length(obs_names)
    meas(rho) = vcat([real(expval(obs_mps[1], rho)), real(expval(obs_mps[2], rho)),
                      sum(real(expval(z, rho)) for z in all_z) / n],
                     [real(expval(z, rho)) for z in all_z])
    zrange    = 4:(3 + n)      # the per-site Z rows, averaged into Z_MAE

    # ---- exact pass ---------------------------------------------------------
    t0 = time()
    e_ref = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                           maxdim=ceil_n, order=order_ref, cutoff=ct,
                           splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
    rho_ref = e_ref.rho
    @printf("  exact reference: chi_rho=%-5d (ceiling %d)  eps=%.2e  Tr=%+.12f  (%.0f s)\n",
            e_ref.chi, ceil_n, e_ref.eps, real(e_ref.trace), time() - t0)
    e_ref.chi < ceil_n && @printf("  NOTE: the exact state does NOT saturate the ceiling. Entanglement is capped;\n        check whether chi_rho is still growing with n in the summary table.\n")
    flush(stdout)

    # ---- how accurate is the reference itself? ------------------------------
    # The fit minimises ||sum c_j rho_kj - rho_ref||, so driving that to zero
    # reproduces rho_ref INCLUDING ITS OWN TROTTER ERROR. Any err_proj below the
    # reference's error is not a measurement of accuracy -- it is the fit
    # successfully copying a slightly wrong state. The family scan hit exactly
    # this: err_proj values of 1.3e-8 against a reference good to ~1.8e-7.
    #
    # Measured directly by halving dt. strang:4 has p_eff = 4.01 (Step 0), so
    # ||rho(k0) - rho(2*k0)|| overestimates the true error by 16/15, i.e. it is
    # a tight upper bound.
    ref_selferr = NaN
    if ref_check
        t0 = time()
        e2 = evolve_trotter(n, J, gammas, t, 2 * k0, lsites, rho0;
                            maxdim=ceil_n, order=order_ref, cutoff=ct,
                            splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
        d = mps_difference(rho_ref, e2.rho)
        nrm = sqrt(max(real(inner(rho_ref, rho_ref)), 0.0))
        ref_selferr = sqrt(max(real(inner(d, d)), 0.0)) / max(nrm, 1e-300)
        @printf("  reference self-error ||rho(k0)-rho(2k0)||/||rho|| = %.3e   (%.0f s)\n",
                ref_selferr, time() - t0)
        println("  -> any err_proj BELOW this is not resolvable: the fit would be copying")
        println("     the reference's own Trotter error, not approaching the truth.")
        flush(stdout)
    end

    rhos_exact = Dict{Int,MPS}()
    for kj in all_ks
        rhos_exact[kj] = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                                        maxdim=ceil_n, order=order, cutoff=ct,
                                        splitting=splitting, mode=evo_mode, id_mps=O_id).rho
    end

    O_exact = meas(rho_ref)
    O_k     = Dict(kj => meas(rhos_exact[kj]) for kj in all_ks)

    # Per-family exact quantities.
    fam = Dict{String,Any}()
    println()
    @printf("  %-14s %3s | %-10s %-10s %-10s | %-11s %-11s %-7s %s\n",
            "family", "r", "E_mpf", "cond(N)", "lam_min", "err_proj", "err_single", "ratio", "flags")
    println("  " * "-"^104)
    for fks in families
        nm = fam_name(fks); rf = length(fks)
        Nx, _ = trotter_error_gram_from_states([rhos_exact[kj] for kj in fks], rho_ref)
        sx    = coefficients_from_N(Nx)
        cx    = sx.coeffs
        Okj   = hcat([O_k[kj] for kj in fks]...)          # n_obs x rf
        ep    = [abs(sum(cx[j] * Okj[a, j] for j in 1:rf) - O_exact[a]) for a in 1:n_obs]
        es    = [minimum(abs(Okj[a, j] - O_exact[a]) for j in 1:rf) for a in 1:n_obs]
        below = isfinite(ref_selferr) && ep[1] < ref_selferr
        fam[nm] = (ks=fks, r=rf, N=Nx, sol=sx, c=cx, Okj=Okj, ep=ep, es=es, below=below,
                   ed=Float64[], eh=Float64[])
        @printf("  %-14s %3d | %.3e  %.3e  %.3e | %.4e  %.4e  %6.3f  %s%s%s\n",
                nm, rf, sx.E_mpf, sx.cond, sx.lam_min, ep[1], es[1],
                ep[1] / max(es[1], 1e-300),
                sx.singular ? "SINGULAR " : "", below ? "BELOW-REF " : "", rf < 3 ? "r<3" : "")
    end
    println()
    flush(stdout)

    # ---- chi sweep ----------------------------------------------------------
    # rho_ref and the candidate pool are evolved ONCE per chi and shared by every
    # family, so extra families cost only linear algebra on r x r matrices.
    # err_best_single -- the error from the BEST SINGLE Trotter circuit, i.e. the
    # answer with ZERO classical computation -- is printed alongside. It is the
    # bar that actually matters, and it was missing from the previous table.
    # With sum(c)=1 and |c| bounded, any normalised combination of candidates
    # lands near the candidates themselves, so DMPF cannot do much WORSE than
    # this no matter how bad the coefficients are; beating it is the real test.
    println("   chi | eps_chi  | err_direct | " *
            join([rpad("err_dmpf[" * fam_name(f) * "]", 15) for f in families]) *
            "| free baseline")
    println("  " * "-"^(30 + 15 * length(families)))

    for chi in grid
        tA = time()
        er = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                            maxdim=chi, order=order_ref, cutoff=ct,
                            splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
        rhos_chi = Dict(kj => evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                                             maxdim=chi, order=order, cutoff=ct,
                                             splitting=splitting, mode=evo_mode, id_mps=O_id).rho
                        for kj in all_ks)
        el = time() - tA

        O_dir = meas(er.rho)
        ed1   = abs(O_dir[1] - O_exact[1])

        line = @sprintf("  %4d | %.2e | %.4e | ", chi, er.eps, ed1)
        for fks in families
            nm = fam_name(fks); F = fam[nm]; rf = F.r
            Nc  = trotter_error_gram_from_states([rhos_chi[kj] for kj in fks], er.rho)[1]
            sc  = coefficients_from_N(Nc)
            dc  = sc.coeffs .- F.c
            dcN = induced_error(F.N, dc)
            rel = norm(Nc .- F.N) / max(norm(F.N), 1e-300)
            pr  = abs.(F.sol.eigvecs' * dc)
            frc = pr[1] / max(norm(dc), 1e-300)
            cor = minimum([truncation_error_correlation(er.rho, rho_ref,
                           rhos_chi[kj], rhos_exact[kj]).corr for kj in fks])

            for a in 1:n_obs
                ed  = abs(O_dir[a] - O_exact[a])
                ehw = abs(sum(sc.coeffs[j] * F.Okj[a, j] for j in 1:rf) - O_exact[a])
                a == 1 && (push!(F.ed, ed); push!(F.eh, ehw))
                push!(rows, @sprintf("%d,%s,%d,%d,%s,%.6e,%s,%.10f,%.8e,%.8e,%.6f,%.8e,%.8e,%.6e,%s,%.6e,%.6e,%.6e,%.6f,%.6f,%.1f",
                                     n, nm, rf, chi, chi == ceil_n ? "yes" : "no", er.eps,
                                     obs_names[a], O_exact[a], ed, ehw, ed / max(ehw, 1e-300),
                                     F.ep[a], F.es[a], sc.cond, sc.singular, rel,
                                     maximum(abs.(dc)), dcN, frc, cor, el))
            end
            line *= @sprintf("%.4e%s   ", F.eh[end], sc.singular ? "*" : " ")
        end
        println(line, @sprintf("| %.4e", fam[fam_name(families[1])].es[1]),
                chi == ceil_n ? "  (exact)" : "")
        flush(stdout)
    end
    println("  (* = N was numerically singular at this chi: the coefficients are noise)")

    # ---- chi* at a common accuracy target, per family -----------------------
    println()
    @printf("  %-14s %-11s | %-10s %-10s %-8s | %-9s %-9s\n",
            "family", "target", "chi_direct*", "chi_dmpf*", "speedup", "r@chi=32", "r@chi=64")
    println("  " * "-"^84)
    for fks in families
        nm = fam_name(fks); F = fam[nm]
        target = tfac * F.ep[1]
        csd = chi_star(grid, F.ed, target)
        csh = chi_star(grid, F.eh, target)
        i32 = findfirst(==(32), grid); i64 = findfirst(==(64), grid)
        r32 = i32 === nothing ? NaN : F.ed[i32] / max(F.eh[i32], 1e-300)
        r64 = i64 === nothing ? NaN : F.ed[i64] / max(F.eh[i64], 1e-300)
        @printf("  %-14s %.4e | %-10s %-10s %-8s | %9.3f %9.3f\n", nm, target,
                fmt_chi(csd), fmt_chi(csh),
                (csd > 0 && csh > 0 && isfinite(csd) && isfinite(csh)) ? @sprintf("%.2fx", csd / csh) : "n/a",
                r32, r64)
        push!(summary, @sprintf("%d,%s,%d,%d,%d,%.6e,%.6e,%.6e,%s,%.8e,%.8e,%.8e,%.8e,%s,%s,%s,%.6f,%.6f",
                                n, nm, F.r, ceil_n, e_ref.chi, ref_selferr,
                                F.sol.cond, F.sol.lam_min, F.sol.singular, F.sol.E_mpf,
                                F.ep[1], F.es[1], target,
                                fmt_chi(csd), fmt_chi(csh),
                                (csd > 0 && csh > 0 && isfinite(csd) && isfinite(csh)) ? @sprintf("%.4f", csd / csh) : "NaN",
                                r32, r64))
    end
    println()
end

write("step2_scaling$sfx.csv", join(rows, "\n") * "\n")
write("step2_summary$sfx.csv", join(summary, "\n") * "\n")

println("="^114)
println("SUMMARY   (this table is the result)")
println("="^114)
for l in summary; println("  ", l); end
println()
println("  READ THE TRENDS, NOT THE VALUES:")
println()
println("   SINGULAR / BELOW-REF flags")
println("     a family whose N is singular is reporting coefficients that are noise;")
println("     one whose err_proj is below the reference self-error is measuring how well")
println("     the fit copies rho_ref, not how accurate it is. Discard both before reading")
println("     anything else. If every good family is BELOW-REF, raise K0 (the reference")
println("     error falls as k0^-4) rather than degrading the family.")
println()
println("   chi_rho_exact vs chi_ceiling")
println("     tracking the ceiling -> the state's operator entanglement is still growing")
println("     with n, so a classical simulation keeps getting harder. Falling away from")
println("     the ceiling -> dissipation has capped it and the classical route stops")
println("     getting harder. The second case is the honest failure mode.")
println()
println("   speedup = chi_direct* / chi_dmpf*")
println("     GROWING with n is the result. Flat means both routes scale the same way and")
println("     there is no advantage. Below 1 means the classical route is simply better.")
println()
println("   err_proj vs n")
println("     should be roughly flat -- it is a Trotter error, set by dt and the candidate")
println("     family. If it grows with n, the family needs redesigning at each size and")
println("     the comparison is not clean.")
println()
println("   corr near 1 and dc_max >> sqrt(dc'Ndc) in the per-n tables are the MECHANISM.")
println("   With r>=3, dc_frac_null becomes meaningful too: near 1 means the coefficient")
println("   error points along the near-null direction of N, where it does least damage.")
println()
println("  wrote step2_scaling$sfx.csv, step2_summary$sfx.csv")
