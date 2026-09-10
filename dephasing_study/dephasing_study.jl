# =============================================================================
# dephasing_study.jl
#
# Dephasing analogue of trajectory_study.jl. Same operating line, same metrics,
# same CSV shape (plus an `unravel` column), so the dephasing and amplitude-
# damping CSVs can be concatenated and plotted with the existing scripts.
#
# WHAT THIS RUN IS FOR. The AD study asked "is the trajectory route cheaper than
# MPDO". That question had one trajectory route. Dephasing has at least two that
# cost wildly different amounts while giving the same rho, so this driver asks
# TWO questions:
#
#   (a) which unravelling binds, and by how much  -> unravelling gap
#   (b) is the binding one more expensive than the AD trajectory run at the
#       same n and matched transverse damping     -> the hardness hypothesis
#
# (b) is the point of the exercise. (a) has to be answered first, because
# quoting the expensive arm would be measuring a choice rather than the channel.
#
# Environment: N, GAMMA_PHI (or GAMMA_AD, from which GAMMA_PHI = GAMMA_AD/2),
#              TMAX_FACTOR, NT, DT, NTRAJ, NTRAJ_UNIT, MAXDIM, MAXDIM_UNIT,
#              CUTOFF, JCOUP, SEED0, TARGET_SEM, CHI_MPDO, CHI_AD_TRAJ,
#              UNRAVELLINGS, OUTDIR, TAG
# =============================================================================
using Printf, Statistics, LinearAlgebra, Dates

getenv(k,d) = get(ENV, k, string(d))
outdir = getenv("OUTDIR","results_deph"); mkpath(outdir)
println("[stage] outdir=$(abspath(outdir))"); flush(stdout)
include(joinpath(@__DIR__, "dephasing_evolution.jl"))
println("[stage] load OK  (threads = $(Threads.nthreads()))"); flush(stdout)

n         = parse(Int,     getenv("N", 16))
gamma_ad  = parse(Float64, getenv("GAMMA_AD", 1.4/parse(Int,getenv("N",16))))
# T2 matching: AD damps the transverse components at gamma/2, dephasing at
# gamma_phi. Equal transverse damping therefore means gamma_phi = gamma_ad/2.
# Override GAMMA_PHI directly to scan the strength instead.
gamma_phi = parse(Float64, getenv("GAMMA_PHI", gamma_ad/2))
tmax_fac  = parse(Float64, getenv("TMAX_FACTOR", 0.6))
nt        = parse(Int,     getenv("NT", 20))
dt        = parse(Float64, getenv("DT", 0.05))
Ntraj     = parse(Int,     getenv("NTRAJ", 200))
Ntraj_u   = parse(Int,     getenv("NTRAJ_UNIT", 16))
maxdim    = parse(Int,     getenv("MAXDIM", 512))
maxdim_u  = parse(Int,     getenv("MAXDIM_UNIT", maxdim))
cutoff    = parse(Float64, getenv("CUTOFF", 1e-10))
jcoup     = parse(Float64, getenv("JCOUP", 0.5))
seed0     = parse(Int,     getenv("SEED0", 1000))
target    = parse(Float64, getenv("TARGET_SEM", 0.01))
chi_mpdo  = parse(Float64, getenv("CHI_MPDO", 0))      # 0 => skip, never faked
chi_ad    = parse(Float64, getenv("CHI_AD_TRAJ", 0))   # 0 => skip
tag       = getenv("TAG",""); sfx = isempty(tag) ? "" : "_"*tag
unravs    = Symbol.(split(getenv("UNRAVELLINGS","projective,pauli"), ","))

times = collect(range(tmax_fac*n/nt, tmax_fac*n; length=nt))

@printf("\n=== dephasing trajectory study ===\n")
@printf("n=%d  gamma_phi=%.5f  (AD reference gamma=%.5f, 1.4/n=%.5f)\n",
        n, gamma_phi, gamma_ad, 1.4/n)
@printf("t up to %.2f (0.45n = %.2f)   NT=%d  dt=%.3g  J=%.3f\n",
        maximum(times), 0.45n, nt, dt, jcoup)
@printf("unravellings: %s   NTRAJ=%d (unitary arms %d)  MAXDIM=%d (unitary %d)\n\n",
        join(unravs,", "), Ntraj, Ntraj_u, maxdim, maxdim_u)
println("--- noise matching at one Trotter step (dt = $(dt)) ---")
nd = noise_diagnostics(gamma_phi, gamma_ad, dt)
println(); flush(stdout)

open(joinpath(outdir,"manifest.csv"),"w") do io
    println(io,"key,value")
    for (k,v) in [("n",n),("gamma_phi",gamma_phi),("gamma_ad_ref",gamma_ad),
                  ("tmax",maximum(times)),("nt",nt),("dt",dt),
                  ("Ntraj",Ntraj),("Ntraj_unit",Ntraj_u),
                  ("maxdim",maxdim),("maxdim_unit",maxdim_u),
                  ("cutoff",cutoff),("jcoup",jcoup),("seed0",seed0),
                  ("target_sem",target),("chi_mpdo",chi_mpdo),
                  ("chi_ad_traj",chi_ad),("unravellings",join(unravs,"|")),
                  ("D_X_deph",nd.D_X_deph),("c_deph",nd.c_deph),
                  ("D_X_ad",nd.D_X_ad),("c_ad",nd.c_ad),
                  ("threads",Threads.nthreads()),("host",gethostname()),
                  ("started",Dates.now())]
        println(io,"$k,$v")
    end
end

const CSV_HEADER = "n,gamma_phi,unravel,Ntraj,maxdim,dt,t,S_mid_mean,S_max_mean," *
                   "S_max_p95,chi_mean,chi_std,chi_sem,chi3_mean,chi_p95," *
                   "chi_max,linkdim_max,z_mid,z_sem,z_var,saturated"

arms = Dict{Symbol,Any}()
walltimes = Dict{Symbol,Any}()

for u in unravs
    # The unitary arms (:pauli, :gaussian) carry no ancillas, so every
    # trajectory is a pure unitary circuit and chi tracks the NOISELESS chain.
    # They need fewer trajectories (chi barely varies) and a separate maxdim,
    # because they will saturate long before the projective arm does.
    isunit = u !== :projective
    Nt = isunit ? Ntraj_u : Ntraj
    md = isunit ? maxdim_u : maxdim
    @printf("\n[stage] arm %s: Ntraj=%d maxdim=%d  (%s)\n", u, Nt, md,
            isunit ? "bare chain, n sites" : "ancilla chain, 2n sites")
    flush(stdout)

    out = run_trajectories(n, jcoup, gamma_phi, times, Nt;
                           unravel=u, dt=dt, cutoff=cutoff, maxdim=md,
                           seed0=seed0, tols=[1e-6], initial=:neel, verbose=true)
    arms[u] = out.series; walltimes[u] = out.walltime

    f = joinpath(outdir, "dephasing_n$(n)_$(u)$(sfx).csv")
    open(f,"w") do io
        println(io, CSV_HEADER)
        for r in out.series
            println(io, join([n,gamma_phi,u,Nt,md,dt,
                @sprintf("%.6f",r.t), @sprintf("%.8f",r.S_mid_mean),
                @sprintf("%.8f",r.S_max_mean), @sprintf("%.8f",r.S_max_p95),
                @sprintf("%.3f",r.chi_mean), @sprintf("%.3f",r.chi_std),
                @sprintf("%.3f",r.chi_sem), @sprintf("%.6e",r.chi3_mean),
                @sprintf("%.2f",r.chi_p95), @sprintf("%.2f",r.chi_max),
                r.linkdim_max, @sprintf("%.8f",r.z_mid), @sprintf("%.8f",r.z_sem),
                @sprintf("%.8e",r.z_var), r.saturated], ","))
        end
    end
    @printf("wrote %s\n", f); flush(stdout)
end

cmp = unravelling_comparison(arms, chi_mpdo, target; chi_ad_traj=chi_ad)

open(joinpath(outdir,"cost_comparison$(sfx).csv"),"w") do io
    println(io,"n,gamma_phi,gamma_ad_ref,unravel,binding,Ntraj_run,Ntraj_needed," *
               "t_peak,chi_mean,chi_sem,chi_p95,chi3_mean,inflation,cost_true," *
               "cost_naive,saturated,chi_mpdo,mpdo_ratio,chi_ad_traj,ad_chi_ratio," *
               "walltime_mean_s,walltime_max_s,walltime_total_coreh")
    for (u, c) in cmp.costs
        wt = walltimes[u]
        println(io, join([n,gamma_phi,gamma_ad,u,(u == cmp.binding),
            (u === :projective ? Ntraj : Ntraj_u), c.Ntraj,
            @sprintf("%.6f",c.t), @sprintf("%.3f",c.chi_mean),
            @sprintf("%.3f",c.chi_sem), @sprintf("%.2f",c.chi_p95),
            @sprintf("%.6e",c.chi3), @sprintf("%.3f",c.inflation),
            @sprintf("%.6e",c.cost_true), @sprintf("%.6e",c.cost_naive),
            c.saturated, @sprintf("%.4e",chi_mpdo),
            @sprintf("%.4e",cmp.mpdo_ratio), @sprintf("%.4e",chi_ad),
            @sprintf("%.4e",cmp.ad_chi_ratio),
            @sprintf("%.2f",mean(wt)), @sprintf("%.2f",maximum(wt)),
            @sprintf("%.4f",sum(wt)/3600)], ","))
    end
end

for (u, res) in arms
    if any(r.saturated for r in res)
        println("\nWARNING [$u]: maxdim bound hit. chi is a LOWER bound there, so any")
        println("         hardness claim resting on this arm is an UNDER-estimate.")
        println("         For the unitary arms this is expected and harmless (they are")
        println("         not the binding arm); for :projective it is not, rerun larger.")
    end
end
println("\n[stage] done")
