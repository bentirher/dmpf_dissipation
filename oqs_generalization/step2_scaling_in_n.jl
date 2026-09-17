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
ks       = parse.(Int, split(getenv("KS", "4,6,8,12"), ","))
n_list   = parse.(Int, split(getenv("N_LIST", "4,6,8,10"), ","))
chi_grid = parse.(Int, split(getenv("CHI_LIST", "4,8,16,32,64,128"), ","))
r        = length(ks)

@printf("step2_scaling_in_n\n")
@printf("  n in %s   gamma=%.3f t=%.1f k0=%d ks=%s\n", string(n_list), gamma, t, k0, string(ks))
@printf("  candidates: %s order %d      reference: %s order %d      mode=%s\n",
        splitting, order, splitting_ref, order_ref, evo_mode)
@printf("  chi grid %s (capped per n at the ceiling 4^(n/2))\n", string(chi_grid))
@printf("  accuracy target = %.2f x err_proj(n)\n", tfac)
k0 % lcm(ks...) == 0 || @printf("  WARNING: k0=%d not a multiple of lcm(ks)=%d -- fine here, but these\n           coefficients will not be cross-checkable against the MOC route.\n", k0, lcm(ks...))
println()
flush(stdout)

rows = ["n,chi,at_ceiling,eps_chi,obs,O_exact,err_direct,err_dmpf,ratio,err_proj,err_best_single,relerr_N,dc_max,dcNdc,dc_frac_null,corr_min,time_s"]
summary = ["n,chi_ceiling,chi_rho_exact,cond_N,E_mpf,err_proj,err_best_single,target,chi_direct_star,chi_dmpf_star,speedup"]

# Smallest grid chi whose error is at or below `target`; linear interpolation in
# log(chi) between the bracketing points, so the answer is not quantised to the
# grid. Returns Inf when the target is never reached on the grid.
function chi_star(chis, errs, target)
    for i in eachindex(chis)
        if errs[i] <= target
            i == 1 && return Float64(chis[i])
            (errs[i-1] <= target || !isfinite(errs[i-1])) && return Float64(chis[i])
            f = (log(errs[i-1]) - log(target)) / (log(errs[i-1]) - log(errs[i]))
            return exp(log(chis[i-1]) + f * (log(chis[i]) - log(chis[i-1])))
        end
    end
    return Inf
end

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
    obs_names = ["Z_mid", "ZZ_mid", "Z_mean"]
    obs_mps   = [z_observable(lsites, mid), zz_observable(lsites, mid, mid + 1)]
    all_z     = [z_observable(lsites, m) for m in 1:n]
    n_obs     = length(obs_names)
    meas(rho) = [real(expval(obs_mps[1], rho)), real(expval(obs_mps[2], rho)),
                 sum(real(expval(z, rho)) for z in all_z) / n]

    # ---- exact pass ---------------------------------------------------------
    t0 = time()
    e_ref = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                           maxdim=ceil_n, order=order_ref, cutoff=ct,
                           splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
    rho_ref = e_ref.rho
    @printf("  exact reference: chi_rho=%-5d (ceiling %d)  eps=%.2e  Tr=%+.12f  (%.0f s)\n",
            e_ref.chi, ceil_n, e_ref.eps, real(e_ref.trace), time() - t0)
    # chi_rho < ceiling means the state's operator entanglement has saturated
    # BELOW the maximum -- dissipation capping it. That is the regime in which
    # a classical simulation stops getting harder with n, and in which this
    # experiment answers no.
    e_ref.chi < ceil_n && @printf("  NOTE: the exact state does NOT saturate the ceiling. Entanglement is capped;\n        check whether chi_rho is still growing with n in the summary table.\n")
    flush(stdout)

    rhos_exact = MPS[]
    for kj in ks
        ek = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                            maxdim=ceil_n, order=order, cutoff=ct,
                            splitting=splitting, mode=evo_mode, id_mps=O_id)
        push!(rhos_exact, ek.rho)
    end

    N_exact, _ = trotter_error_gram_from_states(rhos_exact, rho_ref)
    sol_exact  = coefficients_from_N(N_exact)
    c_exact    = sol_exact.coeffs

    O_exact = meas(rho_ref)
    O_kj    = hcat([meas(rhos_exact[j]) for j in 1:r]...)   # n_obs x r
    err_proj = [abs(sum(c_exact[j] * O_kj[a, j] for j in 1:r) - O_exact[a]) for a in 1:n_obs]
    err_single = [minimum(abs(O_kj[a, j] - O_exact[a]) for j in 1:r) for a in 1:n_obs]

    @printf("  cond(N)=%.3e  E_mpf=%.4e  c=%s\n", sol_exact.cond, sol_exact.E_mpf,
            string(round.(c_exact; digits=5)))
    for a in 1:n_obs
        @printf("    %-8s <O>=%+.8f  err_proj=%.4e  err_single=%.4e  ratio=%.3f\n",
                obs_names[a], O_exact[a], err_proj[a], err_single[a],
                err_proj[a] / max(err_single[a], 1e-300))
    end
    println()
    flush(stdout)

    # ---- chi sweep ----------------------------------------------------------
    println("   chi | eps_chi  | relerr_N | dc_max   | dc'Ndc   | corr   | err_direct | err_dmpf   | ratio")
    println("  " * "-"^106)
    ed_track = Dict(a => Float64[] for a in 1:n_obs)
    eh_track = Dict(a => Float64[] for a in 1:n_obs)

    for chi in grid
        tA = time()
        er = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                            maxdim=chi, order=order_ref, cutoff=ct,
                            splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
        rhos_chi = [evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                                   maxdim=chi, order=order, cutoff=ct,
                                   splitting=splitting, mode=evo_mode, id_mps=O_id).rho
                    for kj in ks]
        el = time() - tA

        N_chi    = trotter_error_gram_from_states(rhos_chi, er.rho)[1]
        sol_chi  = coefficients_from_N(N_chi)
        dc       = sol_chi.coeffs .- c_exact
        dcNdc    = induced_error(N_exact, dc)
        relerr_N = norm(N_chi .- N_exact) / max(norm(N_exact), 1e-300)
        projs    = abs.(sol_exact.eigvecs' * dc)
        frac     = projs[1] / max(norm(dc), 1e-300)
        corrs    = [truncation_error_correlation(er.rho, rho_ref, rhos_chi[j], rhos_exact[j]).corr
                    for j in 1:r]

        O_dir = meas(er.rho)
        for a in 1:n_obs
            ed  = abs(O_dir[a] - O_exact[a])
            ehw = abs(sum(sol_chi.coeffs[j] * O_kj[a, j] for j in 1:r) - O_exact[a])
            push!(ed_track[a], ed); push!(eh_track[a], ehw)
            push!(rows, @sprintf("%d,%d,%s,%.6e,%s,%.10f,%.8e,%.8e,%.6f,%.8e,%.8e,%.6e,%.6e,%.6e,%.6f,%.6f,%.1f",
                                 n, chi, chi == ceil_n ? "yes" : "no", er.eps, obs_names[a],
                                 O_exact[a], ed, ehw, ed / max(ehw, 1e-300),
                                 err_proj[a], err_single[a], relerr_N, maximum(abs.(dc)),
                                 dcNdc, frac, minimum(corrs), el))
        end

        @printf("  %4d | %.2e | %.2e | %.2e | %.2e | %+.3f | %.4e | %.4e | %6.2f%s\n",
                chi, er.eps, relerr_N, maximum(abs.(dc)), dcNdc, minimum(corrs),
                ed_track[1][end], eh_track[1][end],
                ed_track[1][end] / max(eh_track[1][end], 1e-300),
                chi == ceil_n ? "  (exact)" : "")
        flush(stdout)
    end

    # ---- chi* at a common accuracy target -----------------------------------
    # Scored on Z_mid. err_proj is DMPF's floor, so the target is set just above
    # it: any lower and DMPF cannot reach it at any chi, which would make the
    # comparison meaningless rather than negative.
    target = tfac * err_proj[1]
    csd = chi_star(grid, ed_track[1], target)
    csh = chi_star(grid, eh_track[1], target)
    @printf("\n  target = %.2f x err_proj = %.4e\n", tfac, target)
    @printf("  chi needed:  classical %s     DMPF %s     speedup %s\n",
            isfinite(csd) ? @sprintf("%8.1f", csd) : "  >grid",
            isfinite(csh) ? @sprintf("%8.1f", csh) : "  >grid",
            (isfinite(csd) && isfinite(csh)) ? @sprintf("%.2fx", csd / csh) : "   n/a")
    println()

    push!(summary, @sprintf("%d,%d,%d,%.6e,%.8e,%.8e,%.8e,%.8e,%s,%s,%s",
                            n, ceil_n, e_ref.chi, sol_exact.cond, sol_exact.E_mpf,
                            err_proj[1], err_single[1], target,
                            isfinite(csd) ? @sprintf("%.4f", csd) : "Inf",
                            isfinite(csh) ? @sprintf("%.4f", csh) : "Inf",
                            (isfinite(csd) && isfinite(csh)) ? @sprintf("%.4f", csd / csh) : "NaN"))
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
