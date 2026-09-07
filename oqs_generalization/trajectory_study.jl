# =============================================================================
# trajectory_study.jl
#
# The other half of the classical-hardness study. The MPDO runs established that
# the vectorised route needs chi ~ 1e7 at the n=24 operating point
# (gamma = 1.4/n ~ 0.058, t = 0.45n ~ 11). That bounds ONE classical method.
# This driver measures the cost of the other one.
#
# The output that matters is the last block printed: N_traj * chi_traj^3 versus
# chi_MPDO^3. If the trajectory route is cheaper -- and it very likely is --
# then "classically hard" is not yet established at these parameters, and the
# operating line has to move to wherever BOTH methods fail.
#
# Environment: N, GAMMA, TMAX_FACTOR, NT, DT, NTRAJ, MAXDIM, CUTOFF, JCOUP,
#              SEED0, TARGET_SEM, CHI_MPDO, OUTDIR, TAG
# =============================================================================
using Printf, Statistics, LinearAlgebra, Dates

getenv(k,d) = get(ENV, k, string(d))
outdir = getenv("OUTDIR","results_traj"); mkpath(outdir)
println("[stage] outdir=$(abspath(outdir))"); flush(stdout)
include(joinpath(@__DIR__, "trajectory_evolution.jl"))
println("[stage] load OK  (threads = $(Threads.nthreads()))"); flush(stdout)

n         = parse(Int,     getenv("N", 16))
gamma     = parse(Float64, getenv("GAMMA", 1.4/parse(Int,getenv("N",16))))
tmax_fac  = parse(Float64, getenv("TMAX_FACTOR", 0.6))
nt        = parse(Int,     getenv("NT", 20))
dt        = parse(Float64, getenv("DT", 0.05))
Ntraj     = parse(Int,     getenv("NTRAJ", 200))
maxdim    = parse(Int,     getenv("MAXDIM", 512))
cutoff    = parse(Float64, getenv("CUTOFF", 1e-10))
jcoup     = parse(Float64, getenv("JCOUP", 0.5))
seed0     = parse(Int,     getenv("SEED0", 1000))
target    = parse(Float64, getenv("TARGET_SEM", 0.01))
chi_mpdo  = parse(Float64, getenv("CHI_MPDO", 0))   # 0 => use the fitted law
tag       = getenv("TAG",""); sfx = isempty(tag) ? "" : "_"*tag

times = collect(range(tmax_fac*n/nt, tmax_fac*n; length=nt))

# If no measured MPDO chi is supplied, use the study's fitted laws along the
# operating line: S_op ~ 1.39 t, chi ~ 6 * 2^(1.61 S_op).
if chi_mpdo <= 0
    S_op = 1.39 * maximum(times)
    chi_mpdo = 6 * 2.0^(1.61*S_op)
    @printf("CHI_MPDO not set; using fitted law at t=%.1f: S_op=%.1f bits -> chi=%.3e\n",
            maximum(times), S_op, chi_mpdo)
end

@printf("\n=== trajectory study ===\n")
@printf("n=%d gamma=%.4f (1.4/n = %.4f)  t up to %.1f (0.45n = %.1f)\n",
        n, gamma, 1.4/n, maximum(times), 0.45n)
@printf("Ntraj=%d dt=%.3g maxdim=%d cutoff=%.1e J=%.3f\n\n",
        Ntraj, dt, maxdim, cutoff, jcoup); flush(stdout)

open(joinpath(outdir,"manifest.csv"),"w") do io
    println(io,"key,value")
    for (k,v) in [("n",n),("gamma",gamma),("tmax",maximum(times)),("nt",nt),("dt",dt),
                  ("Ntraj",Ntraj),("maxdim",maxdim),("cutoff",cutoff),("jcoup",jcoup),
                  ("seed0",seed0),("target_sem",target),("chi_mpdo",chi_mpdo),
                  ("threads",Threads.nthreads()),("host",gethostname()),
                  ("started",Dates.now())]
        println(io,"$k,$v")
    end
end

res = run_trajectories(n, jcoup, gamma, times, Ntraj;
                       dt=dt, cutoff=cutoff, maxdim=maxdim, seed0=seed0,
                       tols=[1e-6], initial=:neel, verbose=true)

f = joinpath(outdir, "trajectory_n$(n)$(sfx).csv")
open(f,"w") do io
    println(io, "n,gamma,Ntraj,maxdim,dt,t,S_mid_mean,S_max_mean,S_max_p95," *
                "chi_mean,chi_p95,chi_max,linkdim_max,z_mid,z_sem,z_var,saturated")
    for r in res
        println(io, join([n,gamma,Ntraj,maxdim,dt,
            @sprintf("%.6f",r.t), @sprintf("%.8f",r.S_mid_mean),
            @sprintf("%.8f",r.S_max_mean), @sprintf("%.8f",r.S_max_p95),
            @sprintf("%.2f",r.chi_mean), @sprintf("%.2f",r.chi_p95), r.chi_max,
            r.linkdim_max, @sprintf("%.8f",r.z_mid), @sprintf("%.8f",r.z_sem),
            @sprintf("%.8e",r.z_var), r.saturated], ","))
    end
end
@printf("\nwrote %s\n", f)

cmp = cost_comparison(res, chi_mpdo, target)
open(joinpath(outdir,"cost_comparison$(sfx).csv"),"w") do io
    println(io,"n,gamma,target_sem,Ntraj_needed,chi_traj_p95,chi_mpdo,advantage")
    println(io, join([n,gamma,target,cmp.Ntraj,@sprintf("%.1f",cmp.chi_traj),
                      @sprintf("%.4e",chi_mpdo), @sprintf("%.4e",cmp.ratio)],","))
end

if any(r.saturated for r in res)
    println("\nWARNING: maxdim bound on at least one trajectory. chi_traj is a LOWER")
    println("         bound, so the trajectory advantage printed above is an OVER-estimate")
    println("         of the trajectory method's favour only if chi is underestimated --")
    println("         rerun at larger MAXDIM before quoting the ratio either way.")
end
println("\n[stage] done")
