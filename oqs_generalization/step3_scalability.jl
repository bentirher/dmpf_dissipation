# =============================================================================
# step3_scalability.jl   --   DOES THE METHOD RUN, AND CONVERGE, AT LARGE n?
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# Everything so far has been capped at n <= 10, because the comparison needed an
# exact reference and the Liouville MPS ceiling 4^(n/2) is the last affordable
# one there. That is fine for measuring an advantage, but it says nothing about
# the regime the method is actually for: chains of 50+ spins, where no exact
# state exists and the ceiling is unreachable by many orders of magnitude.
#
# Two things had to change before that question could even be asked.
#
#   1. PROTOCOL. The earlier runs applied the gates at the CEILING and truncated
#      to chi once per step. A two-site gate raises a Liouville bond to chi*d
#      with d = 4, so from chi = 16 at n = 10 the middle bond reaches the
#      ceiling within three gate layers, and a fourth-order step has ~25. Every
#      "chi = 16" step ran at bond dimension up to 1024 internally. That makes
#      the reported chi meaningless as a cost AND makes the method unrunnable
#      past the sizes where the ceiling is affordable. PROTOCOL=capped applies
#      every gate at maxdim = chi, as a production TEBD would: O(n chi^2) memory,
#      O(n chi^3) time, no reference to the ceiling anywhere.
#
#   2. NO EXACT REFERENCE. Accuracy cannot be measured against truth at n = 32.
#      But it does not have to be. The quantity we care about is whether the
#      COEFFICIENTS have converged, and that is a self-consistency question:
#      compare c(chi) with c(chi_top) and convert the difference into an error
#      through the exact identity
#
#           E(c* + dc) - E(c*) = dc^T N dc ,
#
#      which needs only N, not the exact state. This is the payoff of the error
#      formulation: the figure of merit is computable without knowing the answer.
#
# WHAT IT MEASURES, per n and chi, all against the largest chi on the grid:
#
#   selfconv_direct   |<O>(rho_ref(chi)) - <O>(rho_ref(chi_top))|
#                     how far the purely classical estimate still has to travel
#   selfconv_coeff    sqrt( dc^T N(chi_top) dc ),  dc = c(chi) - c(chi_top)
#                     how far the coefficients still have to travel, in the
#                     metric that determines the achieved error
#   selfconv_dmpf     |sum_j dc_j <O>_kj(chi_top)|
#                     the same thing at the level of the observable
#   selfconv_cand     max_j |<O>_kj(chi) - <O>_kj(chi_top)|
#                     whether the candidates themselves have converged; they are
#                     shallow (k_j <= 16 steps) and should converge first
#
# THE CLAIM THIS CAN SUPPORT: the coefficients stop moving at a chi where the
# direct simulation has not. That is a statement about scalability which needs
# no exact state, and it is the one that matters at n = 50.
#
# THE CLAIM IT CANNOT SUPPORT: absolute accuracy. Self-convergence is necessary,
# not sufficient -- both routes could be converging to something wrong together.
# At n <= 10 we know they are not (step2 checks against the exact state); beyond
# that this is an extrapolation of trust, and should be said as such.
#
# Environment: N_LIST CHI_LIST GAMMA TVAL KS K0 ORDER SPLITTING ORDER_REF
#              SPLITTING_REF PROTOCOL CUTOFF TARGET TAG
# Output: step3_scalability<_tag>.csv, step3_summary<_tag>.csv
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("liouville_state_tools.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
gamma         = parse(Float64, getenv("GAMMA",     0.05))
t             = parse(Float64, getenv("TVAL",      3.0))
k0            = parse(Int,     getenv("K0",        96))
order         = parse(Int,     getenv("ORDER",     2))
order_ref     = parse(Int,     getenv("ORDER_REF", 4))
splitting     = Symbol(getenv("SPLITTING",     "strang"))
splitting_ref = Symbol(getenv("SPLITTING_REF", "strang"))
evo_mode      = Symbol(getenv("EVO_MODE", "gates"))
protocol      = Symbol(getenv("PROTOCOL", "capped"))
ct            = parse(Float64, getenv("CUTOFF", EXACT_CUTOFF))
target        = parse(Float64, getenv("TARGET", 1e-4))
tag           = getenv("TAG", "")
sfx           = isempty(tag) ? "" : "_" * tag
ks       = parse.(Int, split(getenv("KS", "4,8,16"), ","))
n_list   = parse.(Int, split(getenv("N_LIST", "12,16,20,24"), ","))
chi_grid = sort(parse.(Int, split(getenv("CHI_LIST", "32,64,96,128,192,256"), ",")))
r        = length(ks)

@printf("step3_scalability\n")
@printf("  n in %s   chi in %s\n", string(n_list), string(chi_grid))
@printf("  gamma=%.3f t=%.1f k0=%d ks=%s   protocol=%s\n",
        gamma, t, k0, string(ks), protocol)
if protocol !== :capped
    println("  WARNING: protocol is not :capped. The gates will be applied at the")
    println("           ceiling, which is neither the honest cost nor runnable at")
    println("           the sizes this script is for.")
end
@printf("  self-convergence target = %.1e ; reference chi = %d (largest on the grid)\n\n",
        target, chi_grid[end])
flush(stdout)

rows = ["n,chi,obs,O_direct,selfconv_direct,selfconv_dmpf,selfconv_cand," *
        "sqrt_dcNdc,dc_max,cond_N,lam_min,singular,chi_actual,trace_dev,time_s"]
summary = ["n,chi_top,target,chi_direct_conv,chi_coeff_conv,ratio," *
           "cond_N_top,E_mpf_top,time_top_s,chi_actual_top"]

# smallest chi on the grid from which a self-convergence curve stays below the
# target; the same sustained rule used elsewhere, so a lucky dip cannot pass.
function first_sustained(chis, vals, target)
    for i in eachindex(chis)
        all(v -> (isfinite(v) && v <= target), vals[i:end]) && return Float64(chis[i])
    end
    return NaN
end

for n in n_list
    println("="^108)
    @printf("n = %d    ceiling would be 4^%d (unused)    chi grid = %s\n",
            n, n ÷ 2, string(chi_grid))
    println("="^108)
    flush(stdout)

    Random.seed!(1234)
    J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
    gammas = fill(gamma, n)
    lsites = liouville_siteinds(n)
    rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

    mid   = n ÷ 2
    O_id  = identity_observable(lsites)
    all_z = [z_observable(lsites, m) for m in 1:n]
    O_zz  = zz_observable(lsites, mid, mid + 1)
    obs_names = vcat(["Z_MAE", "Z_mid", "ZZ_mid"], ["Z$m" for m in 1:n])
    # Z_MAE is a derived slot: its "value" is unused, only differences of the
    # per-site entries are averaged into it afterwards.
    meas(rho) = vcat([0.0, real(expval(all_z[mid], rho)), real(expval(O_zz, rho))],
                     [real(expval(z, rho)) for z in all_z])
    zrange = 4:(3 + n)
    n_obs  = length(obs_names)

    Odir = Dict{Int,Vector{Float64}}()     # chi -> observable vector (reference)
    Okj  = Dict{Int,Matrix{Float64}}()     # chi -> n_obs x r (candidates)
    Cs   = Dict{Int,Vector{Float64}}()     # chi -> coefficients
    Ns   = Dict{Int,Matrix{Float64}}()     # chi -> Gram matrix
    Sol  = Dict{Int,Any}()
    Tm   = Dict{Int,Float64}(); Ch = Dict{Int,Int}(); Tr = Dict{Int,Float64}()

    @printf("  %5s | %8s %7s %9s | %-11s %-11s %-11s\n",
            "chi", "time(s)", "chi_act", "|1-Tr|", "cond(N)", "E_mpf", "max|c|")
    println("  " * "-"^76)
    for chi in chi_grid
        tA = time()
        er = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                            maxdim=chi, order=order_ref, cutoff=ct,
                            splitting=splitting_ref, mode=evo_mode,
                            protocol=protocol, id_mps=O_id)
        rhos = [evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                               maxdim=chi, order=order, cutoff=ct,
                               splitting=splitting, mode=evo_mode,
                               protocol=protocol, id_mps=O_id).rho for kj in ks]
        Tm[chi] = time() - tA; Ch[chi] = er.chi; Tr[chi] = abs(1 - real(er.trace))

        Ns[chi] = trotter_error_gram_from_states(rhos, er.rho)[1]
        Sol[chi] = coefficients_from_N(Ns[chi])
        Cs[chi] = Sol[chi].coeffs
        Odir[chi] = meas(er.rho)
        Okj[chi] = hcat([meas(rh) for rh in rhos]...)

        @printf("  %5d | %8.1f %7d %9.2e | %.4e  %.4e  %8.3f%s\n",
                chi, Tm[chi], Ch[chi], Tr[chi], Sol[chi].cond, Sol[chi].E_mpf,
                maximum(abs.(Cs[chi])), Sol[chi].singular ? "  SINGULAR" : "")
        flush(stdout)
    end

    # ---- self-convergence against the largest chi ---------------------------
    top = chi_grid[end]; Ntop = Ns[top]; ctop = Cs[top]
    println()
    println("  self-convergence against chi = $top  (metric Z_MAE)")
    @printf("  %5s | %-12s %-12s %-12s | %-12s %-10s\n", "chi",
            "classical", "DMPF coeff", "candidates", "sqrt(dcNdc)", "max|dc|")
    println("  " * "-"^76)

    sc_dir = Float64[]; sc_dmp = Float64[]
    for chi in chi_grid
        dc  = Cs[chi] .- ctop
        dcN = sqrt(max(induced_error(Ntop, dc), 0.0))
        # site-averaged self-convergence of each route
        sdir = sum(abs(Odir[chi][a] - Odir[top][a]) for a in zrange) / n
        scan = maximum(sum(abs(Okj[chi][a, j] - Okj[top][a, j]) for a in zrange) / n
                       for j in 1:r)
        sdmp = sum(abs(sum(dc[j] * Okj[top][a, j] for j in 1:r)) for a in zrange) / n
        push!(sc_dir, sdir); push!(sc_dmp, sdmp)

        @printf("  %5d | %.4e   %.4e   %.4e   | %.4e   %.3e\n",
                chi, sdir, sdmp, scan, dcN, maximum(abs.(dc)))

        for (a, nm) in enumerate(obs_names)
            v_dir = a in zrange || a > 3 ? abs(Odir[chi][a] - Odir[top][a]) :
                    (nm == "Z_MAE" ? sdir : abs(Odir[chi][a] - Odir[top][a]))
            v_dmp = nm == "Z_MAE" ? sdmp : abs(sum(dc[j] * Okj[top][a, j] for j in 1:r))
            v_can = nm == "Z_MAE" ? scan : maximum(abs(Okj[chi][a, j] - Okj[top][a, j]) for j in 1:r)
            push!(rows, @sprintf("%d,%d,%s,%.10f,%.8e,%.8e,%.8e,%.8e,%.6e,%.6e,%.6e,%s,%d,%.6e,%.1f",
                                 n, chi, nm, Odir[chi][a], v_dir, v_dmp, v_can,
                                 dcN, maximum(abs.(dc)), Sol[chi].cond, Sol[chi].lam_min,
                                 Sol[chi].singular, Ch[chi], Tr[chi], Tm[chi]))
        end
    end

    cdir = first_sustained(chi_grid, sc_dir, target)
    ccof = first_sustained(chi_grid, sc_dmp, target)
    @printf("\n  chi at which each self-converges below %.1e:  classical %s   coefficients %s",
            target, isnan(cdir) ? ">grid" : string(Int(cdir)),
            isnan(ccof) ? ">grid" : string(Int(ccof)))
    @printf("   ratio %s\n\n",
            (isnan(cdir) || isnan(ccof)) ? "n/a" : @sprintf("%.2f", cdir / ccof))
    println("  (the last row of each column is identically zero by construction and is")
    println("   excluded from the rule; a '>grid' for the classical route with a finite")
    println("   value for the coefficients is the scalability statement.)")
    println()

    push!(summary, @sprintf("%d,%d,%.1e,%s,%s,%s,%.6e,%.8e,%.1f,%d",
                            n, top, target,
                            isnan(cdir) ? "Inf" : @sprintf("%.0f", cdir),
                            isnan(ccof) ? "Inf" : @sprintf("%.0f", ccof),
                            (isnan(cdir) || isnan(ccof)) ? "NaN" : @sprintf("%.4f", cdir / ccof),
                            Sol[top].cond, Sol[top].E_mpf, Tm[top], Ch[top]))
    flush(stdout)
end

write("step3_scalability$sfx.csv", join(rows, "\n") * "\n")
write("step3_summary$sfx.csv", join(summary, "\n") * "\n")

println("="^108)
println("SUMMARY")
println("="^108)
for l in summary; println("  ", l); end
println()
println("  chi_actual_top below chi_top means the state never needed the full bond")
println("  dimension; chi_actual == chi_top means the cap was binding and the largest")
println("  chi on the grid is not yet converged -- extend CHI_LIST before concluding.")
println()
println("  time_top_s is measured wall time at the largest chi, so the cost model")
println("  time ~ chi^3 used in the draft can be replaced by measurement.")
println()
println("  wrote step3_scalability$sfx.csv, step3_summary$sfx.csv")
