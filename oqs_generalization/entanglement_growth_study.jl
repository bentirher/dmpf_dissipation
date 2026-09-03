# =============================================================================
# entanglement_growth_study.jl  (round 2)
#
# Locates the classically expensive corner of (n, gamma, t) for vectorized
# MPS/TEBD, to motivate the parameters of the ancilla-based circuit experiment.
#
# -----------------------------------------------------------------------------
# WHAT ROUND 1 TAUGHT US, AND WHAT CHANGED HERE
# -----------------------------------------------------------------------------
#
# 1. THE TIME WINDOW SCALED WITH THE WRONG VARIABLE.
#    Round 1 set TMAX ~ 2/gamma on the assumption that the operator-entanglement
#    barrier peaks at t ~ 1/gamma. It does not. Measured peaks were at
#    t = 2.5, 4, 5, 4 for gamma = 0.2, 0.1, 0.05, 0.01 against 1/gamma =
#    5, 10, 20, 100. For small gamma the peak is where the CLOSED system
#    saturates, i.e. t_peak ~ n/2, and is set by system size, not by gamma.
#    Consequence: the gamma=0.01 job ran to t=200 to capture a peak at t=4 and
#    spent ~95% of its walltime on a flat tail, which is why the small-gamma
#    ladders never reached large n.
#    FIX: times are built per n as TMAX = TMAX_FACTOR * n (default 1.5), which
#    brackets the peak plus the beginning of the decay.
#
# 2. THE CLOSED BASELINE WAS UNPHYSICAL AT n >= 12.
#    Propagating a pure state as a density matrix at maxdim=256 gave purity
#    Tr(rho^2) = 1.69 (n=12), 4.19 (n=16), 24.2 (n=20). Purity is identically 1
#    for a pure state, so those runs were garbage past t ~ 2.5.
#    FIX: closed runs go through evolve_closed (closed_evolution.jl), which
#    propagates |psi> at local dimension 2 and reconstructs the operator-space
#    quantities EXACTLY from the pure-state Schmidt spectrum. Purity cannot be
#    violated, and energy conservation gives a sharp truncation flag.
#
# 3. maxdim BOUND EVERYWHERE, SO cutoff WAS NEVER ENFORCED.
#    Every n >= 12 run had linkdim pinned at maxdim. chi_req then tracked maxdim
#    (64 -> 127 -> 249 as maxdim went 64 -> 128 -> 256) instead of converging.
#    FIX: `saturated` now fires whenever maxdim binds at all, and the ladder
#    always extends ABOVE the production maxdim -- round 1's ladder topped out
#    AT it, which made "converged at 256" a tautology.
#
# 4. S_op AND chi_req CONVERGE AT COMPLETELY DIFFERENT RATES.
#    S_op was converged to 1e-4 already at maxdim = 64; chi_req was not
#    converged at 256. Chasing both on one expensive grid spends the budget on
#    the cheap quantity.
#    FIX: three modes.
#      MODE=scaling  cheap maxdim, wide (n, gamma) grid -> S_op(n,gamma,t) and
#                    the area-law onset n_sat(gamma). This is the physics.
#      MODE=ladder   one (n, gamma), maxdim ladder up to 2x production, a few
#                    times near the peak -> the chi_req question, answered as
#                    "converged to X" or honestly as "> X".
#      MODE=dt       Trotter-step check. Round 1 used dt=0.02 everywhere without
#                    ever testing whether 0.05 or 0.1 would do.
#
# Environment: MODE, N_LIST, GAMMA_LIST, TMAX_FACTOR, NT, DT, ORDER, MAXDIM,
#              CUTOFF, JCOUP, DISORDER, SEED, TAG, OUTDIR, SKIP_CLOSED
# =============================================================================
using LinearAlgebra, Printf, Dates
import Random, Distributions

getenv(k, d) = get(ENV, k, string(d))

outdir = getenv("OUTDIR", "results")
mkpath(outdir)
println("[stage] outdir=$(abspath(outdir))"); flush(stdout)
println("[stage] loading ..."); flush(stdout)
include(joinpath(@__DIR__, "vectorized_evolution.jl"))
include(joinpath(@__DIR__, "closed_evolution.jl"))
println("[stage] load OK"); flush(stdout)

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

mode        = getenv("MODE", "scaling")
n_list      = parse.(Int, split(getenv("N_LIST", "8,12,16,20,24"), ","))
gamma_list  = parse.(Float64, split(getenv("GAMMA_LIST", "0.02,0.05,0.10"), ","))
tmax_factor = parse(Float64, getenv("TMAX_FACTOR", 1.5))
nt          = parse(Int,     getenv("NT",     30))
dt          = parse(Float64, getenv("DT",     0.05))
order       = parse(Int,     getenv("ORDER",  2))
maxdim      = parse(Int,     getenv("MAXDIM", 256))
cutoff      = parse(Float64, getenv("CUTOFF", 1e-12))
jcoup       = parse(Float64, getenv("JCOUP",  0.5))
disorder    = parse(Bool,    getenv("DISORDER", false))
seed        = parse(Int,     getenv("SEED",   1234))
skip_closed = parse(Bool,    getenv("SKIP_CLOSED", false))
splitting   = Symbol(getenv("SPLITTING", "project"))   # :project (codebase default) or :strang
tag         = getenv("TAG", "")
sfx         = isempty(tag) ? "" : "_" * tag

# The peak sits near t ~ n/2, so the window is set by n, not by gamma.
# TMAX_FACTOR=1.5 covers the peak and the start of the fall; raise it to ~3 only
# if you specifically want the approach to the steady state.
times_for(n) = collect(range(tmax_factor*n/nt, tmax_factor*n; length=nt))

function coupling(n)
    disorder || return jcoup
    Random.seed!(seed)
    return rand(Distributions.Uniform(1/4, 3/4), n - 1)
end

@printf("=== entanglement growth study, MODE=%s ===\n", mode)
@printf("n_list=%s  gamma_list=%s\n", string(n_list), string(gamma_list))
@printf("TMAX = %.2f * n  (n=%d -> t up to %.1f)  nt=%d  dt=%.4g\n",
        tmax_factor, maximum(n_list), tmax_factor*maximum(n_list), nt, dt)
@printf("order=%d maxdim=%d cutoff=%.1e J=%s\n\n", order, maxdim, cutoff,
        disorder ? "U[1/4,3/4] (seed $seed)" : @sprintf("%.3f uniform", jcoup))
flush(stdout)

manifest = joinpath(outdir, "manifest.csv")
open(manifest, "w") do io
    println(io, "key,value")
    for (k,v) in [("mode",mode), ("n_list",join(n_list," ")), ("gamma_list",join(gamma_list," ")),
                  ("tmax_factor",tmax_factor), ("nt",nt), ("dt",dt), ("order",order),
                  ("maxdim",maxdim), ("cutoff",cutoff), ("jcoup",jcoup),
                  ("disorder",disorder), ("seed",seed), ("tag",tag),
                  ("host",gethostname()), ("started",Dates.now()), ("cwd",pwd())]
        println(io, "$k,$v")
    end
end
println("[stage] wrote $manifest"); flush(stdout)


# =============================================================================
# MODE = scaling
# =============================================================================
function run_scaling()
    all_rows = String[]; prof_rows = String[]; first = true
    function collect!(res, lbl)
        append!(all_rows,  split(chomp(rows_to_csv(res;    label=lbl, header=first)), "\n"))
        append!(prof_rows, split(chomp(profile_to_csv(res; label=lbl, header=first)), "\n"))
        first = false
        save_run(res; dir=outdir, label="$(lbl)$(sfx)")
    end

    for n in n_list
        J = coupling(n); ts = times_for(n)

        # Closed baseline once per n. It does not depend on gamma, and the
        # pure-state route is cheap enough that there is no reason to recompute
        # it inside the gamma loop the way round 1 did.
        if !skip_closed
            @printf("\n--- n=%d  CLOSED (pure-state route) ---\n", n); flush(stdout)
            resc = evolve_closed(n, J, ts; dt=dt, order=order, cutoff=cutoff,
                                 maxdim=maxdim, initial=:neel, tols=[1e-6,1e-10],
                                 verbose=true)
            collect!(resc, "closed_n$(n)")
        end

        for gm in gamma_list
            gtag = replace(@sprintf("%.3f", gm), "." => "p")
            @printf("\n--- n=%d  gamma=%.3f  OPEN ---\n", n, gm); flush(stdout)
            res = evolve_vectorized(n, J, gm, ts; dissipation=true, dt=dt, order=order,
                                    splitting=splitting, cutoff=cutoff, maxdim=maxdim,
                                    initial=:neel, tols=[1e-6,1e-10], verbose=true)
            collect!(res, "open_n$(n)_g$(gtag)")
        end
    end

    ts_file   = joinpath(outdir, "scaling$(sfx).csv")
    prof_file = joinpath(outdir, "scaling$(sfx)_profile.csv")
    write(ts_file,   join(all_rows,  "\n") * "\n")
    write(prof_file, join(prof_rows, "\n") * "\n")
    @printf("\nwrote %s (%d rows)\nwrote %s (%d rows)\n",
            ts_file, length(all_rows)-1, prof_file, length(prof_rows)-1)
end


# =============================================================================
# MODE = ladder
# =============================================================================
function run_ladder()
    n  = n_list[1]
    gm = gamma_list[1]
    J  = coupling(n)

    # A handful of times bracketing the peak (t ~ n/2), not the whole window:
    # the ladder costs one full run per rung and the answer lives at the peak.
    tp = n/2
    # Was [0.5, 0.75, 1.0, 1.25, 1.75]*tp. The ladder costs one full run per
    # rung and the run length is set by the LAST time, so the 1.75 point cost
    # 40% of the job to sample the flat post-peak tail. Dropped.
    check_times = round.(sort(unique(filter(t -> t > 0,
                        [0.5tp, 0.75tp, tp, 1.25tp]))); digits=4)

    # Always extend ABOVE the production maxdim, or the top rung is its own
    # reference and every point trivially reports as converged.
    # Cap at the exact MPDO ceiling: beyond 4^(n/2) there is nothing left to
    # resolve, so a higher rung is pure waste (at n=8 the ceiling is 256).
    ceil_n = 4^min(n÷2, n - n÷2)
    ladder = sort(unique(vcat(filter(<=(maxdim), [64,128,256,512,1024]),
                              [maxdim, 2*maxdim])))
    ladder = filter(<=(ceil_n), ladder)
    isempty(ladder) && (ladder = [ceil_n])
    length(ladder) < 2 && (ladder = sort(unique([max(8, ceil_n ÷ 4), ceil_n])))
    @printf("\n=== maxdim ladder, n=%d gamma=%.3f ===\n", n, gm)
    @printf("times: %s\nladder: %s (production=%d)\n\n",
            string(check_times), string(ladder), maxdim); flush(stdout)

    conv = maxdim_convergence(n, J, gm, check_times, ladder;
                              dt=dt, order=order, splitting=splitting, cutoff=cutoff,
                              initial=:neel, tols=[1e-6,1e-10], verbose=true)

    gtag = replace(@sprintf("%.3f", gm), "." => "p")
    f = joinpath(outdir, "ladder_n$(n)_g$(gtag)$(sfx).csv")
    write_convergence_csv(f, conv; label_prefix="ladder_n$(n)_g$(gtag)")
    @printf("wrote %s\n", f)

    # Verdict, stated explicitly so it does not have to be eyeballed later.
    lo, hi = conv.runs[end-1], conv.runs[end]
    println("\n--- verdict at the top two rungs (maxdim $(ladder[end-1]) vs $(ladder[end])) ---")
    for i in eachindex(hi.t)
        dS = abs(lo.S_op_mid[i] - hi.S_op_mid[i])
        c1, c2 = lo.chi_req_mid[1][i], hi.chi_req_mid[1][i]
        ratio = c1 == 0 ? NaN : c2/c1
        verdict = (dS < 1e-3 && ratio < 1.1) ? "CONVERGED" :
                  (dS < 1e-3 ? "S_op ok, chi_req NOT converged (only chi >= $(c2))" :
                               "NOT converged")
        @printf("  t=%6.2f  S_op %.5f vs %.5f (dS=%.1e)  chi_req %4d -> %4d (x%.2f)  %s\n",
                hi.t[i], lo.S_op_mid[i], hi.S_op_mid[i], dS, c1, c2, ratio, verdict)
    end
end


# =============================================================================
# MODE = dt
# =============================================================================
function run_dtcheck()
    n = n_list[1]; gm = gamma_list[1]; J = coupling(n)
    tp = n/2
    check_times = round.([0.5tp, tp, 1.5tp]; digits=4)
    dts = [0.2, 0.1, 0.05, 0.02]
    @printf("\n=== Trotter check, n=%d gamma=%.3f, maxdim=%d ===\n", n, gm, maxdim)
    conv = dt_convergence(n, J, gm, check_times, dts;
                          order=order, splitting=splitting, cutoff=cutoff, maxdim=maxdim,
                          initial=:neel, tols=[1e-6,1e-10], verbose=true)
    ref = conv.runs[argmin(dts)]
    rows = ["label,n,gamma,dt,order,maxdim,t,S_op_mid,z_mid_re,dS_vs_finest,dz_vs_finest"]
    for (r,d) in zip(conv.runs, dts), i in eachindex(r.t)
        push!(rows, join(["dt$(d)", n, gm, d, order, maxdim,
            @sprintf("%.4f", r.t[i]), @sprintf("%.10f", r.S_op_mid[i]),
            @sprintf("%.10f", real(r.z_mid[i])),
            @sprintf("%.4e", abs(r.S_op_mid[i]-ref.S_op_mid[i])),
            @sprintf("%.4e", abs(real(r.z_mid[i])-real(ref.z_mid[i])))], ","))
    end
    f = joinpath(outdir, "dtcheck_n$(n)$(sfx).csv")
    write(f, join(rows,"\n")*"\n"); @printf("wrote %s\n", f)
end


# =============================================================================
# MODE = order
# =============================================================================
#
# Measures the actual Trotter order, cleanly. The n=12 dt check was ambiguous
# because maxdim=256 was binding there, so truncation error was mixed into the
# dt error. At n=8 the MPDO ceiling is 4^4 = 256, so maxdim=256 truncates
# NOTHING and the only error left is the product formula. Whatever exponent
# comes out of this run is the real one.
#
# Reports the fitted local exponent from successive error ratios:
#   ratio ~ 2 => first order, ratio ~ 4 => second order, ~16 => fourth.
# Runs both the project's composition and the symmetric one, so the comparison
# is like-for-like at identical dt, times and truncation (none).
function run_ordercheck()
    n = n_list[1]; gm = gamma_list[1]; J = coupling(n)
    ceil_n = 4^min(n÷2, n - n÷2)
    if maxdim < ceil_n
        @printf("WARNING: maxdim=%d is below the exact ceiling %d for n=%d.\n", maxdim, ceil_n, n)
        println("         Truncation will contaminate the order measurement. Set MAXDIM=$ceil_n.")
    end
    tp = n/2
    check_times = round.([0.5tp, tp, 1.5tp, 2.5tp]; digits=4)
    dts = [0.2, 0.1, 0.05, 0.025]

    rows = ["label,n,gamma,splitting,dt,order,maxdim,t,S_op_mid,z_mid_re,dS_vs_finest,dz_vs_finest"]
    for spl in (:project, :strang)
        @printf("\n=== order check: splitting=%s, n=%d gamma=%.3f, maxdim=%d (exact ceiling %d) ===\n",
                spl, n, gm, maxdim, ceil_n); flush(stdout)
        runs = [evolve_vectorized(n, J, gm, check_times; dt=d, order=2, splitting=spl,
                                  cutoff=cutoff, maxdim=maxdim, initial=:neel,
                                  tols=[1e-6,1e-10], verbose=false) for d in dts]
        ref = runs[end]
        println("      t | " * join([@sprintf("%10s", "dt=$d") for d in dts], "") *
                " |  err ratios (2=1st order, 4=2nd)")
        for i in eachindex(ref.t)
            ref.t[i] == 0 && continue
            e = [abs(r.S_op_mid[i] - ref.S_op_mid[i]) for r in runs]
            rat = [e[k]/e[k+1] for k in 1:length(dts)-2 if e[k+1] > 0]
            @printf("%7.3f |%s | %s\n", ref.t[i],
                    join([@sprintf("%10.5f", r.S_op_mid[i]) for r in runs], ""),
                    join([@sprintf("%.2f", x) for x in rat], "  "))
        end
        for (r,d) in zip(runs, dts), i in eachindex(r.t)
            push!(rows, join([string(spl), n, gm, string(spl), d, 2, maxdim,
                @sprintf("%.4f", r.t[i]), @sprintf("%.10f", r.S_op_mid[i]),
                @sprintf("%.10f", real(r.z_mid[i])),
                @sprintf("%.4e", abs(r.S_op_mid[i]-ref.S_op_mid[i])),
                @sprintf("%.4e", abs(real(r.z_mid[i])-real(ref.z_mid[i])))], ","))
        end
    end
    f = joinpath(outdir, "ordercheck_n$(n)$(sfx).csv")
    write(f, join(rows,"\n")*"\n"); @printf("\nwrote %s\n", f)
    println("\nIf :project gives ratios near 2 and :strang gives ratios near 4,")
    println("then get_open_step_gates(...; order=2) is first order in the open case,")
    println("and that finding matters well beyond this study.")
end


# =============================================================================
if mode == "scaling"
    run_scaling()
elseif mode == "ladder"
    run_ladder()
elseif mode == "dt"
    run_dtcheck()
elseif mode == "order"
    run_ordercheck()
else
    error("MODE must be one of: scaling, ladder, dt, order. Got '$mode'.")
end
println("\n[stage] done"); flush(stdout)
