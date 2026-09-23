# =============================================================================
# circuit_study.jl  --  ONE driver for the whole hardware-aware study
#
# Everything from Step 2 onwards runs through this file, selected by MODE.
# No further .jl files should be needed.
#
#   MODE=theta     Step 2a. Sweep the gate angle at fixed (n, p, k).
#                  Where is the circuit hardest?
#   MODE=fidelity  Step 2b. Sweep theta at small n and measure how far the
#                  circuit is from the master equation it discretises.
#                  Where does it stop being physics?
#   MODE=damping   Step 3.  Sweep p at the chosen theta.
#   MODE=scaling   Step 4.  Sweep n at the chosen (theta, p).
#   MODE=map       The 2D (theta, p) colormap. Subsumes MODE=theta and
#                  MODE=damping -- both are one-dimensional slices of it.
#   MODE=headtohead  A self-contained MPDO-vs-trajectory comparison at small n,
#                  where BOTH methods can be run UNTRUNCATED. Stands alone: it
#                  does not ask the reader to accept any earlier result.
#   MODE=mpdoladder         Is the MPDO converged in maxdim, or reporting the
#                           cap back at us? Run at one (n, theta, p).
#
# MAXDIM=0 in theta/scaling mode means the EXACT Liouville ceiling 4^(n/2):
# no truncation, no caveat. Affordable to n=12 (ceiling 4096).
#
# -----------------------------------------------------------------------------
# PARAMETERISATION (settled in Step 1)
# -----------------------------------------------------------------------------
# The circuit is (n, theta, p, k) and nothing else.
#
#     theta = 2*J*dt      the rxx angle, the only entangling knob
#     p     = 1-exp(-gamma*dt)   per-qubit jump probability per step
#     k     = number of Trotter steps -- SET BY HARDWARE DEPTH, not chosen
#
# dt = t/k is the physics knob; t = k*dt is a derived label, reported for
# context only. J and gamma appear nowhere below: JREF and GREF exist purely to
# print the implied (dt, t, gamma/J) alongside each point so the circuit
# parameters can be mapped back to the manuscript's units.
#
# -----------------------------------------------------------------------------
# WHAT TO EXPECT IN THE THETA SWEEP
# -----------------------------------------------------------------------------
# The two-qubit block RZZ(2*theta) RYY(theta) RXX(theta) has entangling power
#     theta   0.05   0.50   0.79   1.00   pi/2   2.00   pi
#     power  0.003  0.174  0.167  0.126  0.223  0.131  0.000
# It is a PERFECT ENTANGLER at theta = pi/2 (matching CNOT's 2/9) and trivial at
# theta = pi. The landscape is NOT monotonic -- there are dips near theta = 1
# and 2 -- so the sweep is denser there than a log grid would be. Whether the
# many-body chi follows the single-gate curve or saturates earlier is exactly
# what this mode measures.
#
# Environment: MODE, N, NLIST, THETA, THETAS, P, PLIST, K, MAXDIM, CUTOFF,
#              NTRAJ, EXCITED, KREF, JREF, GREF, OUTDIR, TAG
# =============================================================================
using Printf, Statistics, LinearAlgebra, Dates

getenv(k,d) = get(ENV,k,string(d))
outdir = getenv("OUTDIR","results_circuit"); mkpath(outdir)
println("[stage] outdir=$(abspath(outdir))"); flush(stdout)
include(joinpath(@__DIR__,"vectorized_evolution.jl"))
include(joinpath(@__DIR__,"trajectory_evolution.jl"))
include(joinpath(@__DIR__,"circuit_native.jl"))
println("[stage] load OK (threads = $(Threads.nthreads()))"); flush(stdout)

mode    = getenv("MODE","theta")
n       = parse(Int, getenv("N",16))
nlist   = parse.(Int, split(getenv("NLIST","8,12,16,20,24"),","))
k       = parse(Int, getenv("K",10))
pfix    = parse(Float64, getenv("P",0.05))
plist   = parse.(Float64, split(getenv("PLIST","0.0,0.01,0.02,0.05,0.10,0.15,0.25"),","))
thetafx = parse(Float64, getenv("THETA",1.5708))
maxdim  = parse(Int, getenv("MAXDIM",1024))
# SEPARATE CAPS FOR THE TWO ROUTES.
#
# The MPDO is an MPS of local dimension 4, so a cap of X means SVDs of size
# 4X. At n=24 with MAXDIM=4096 that is 16384 x 16384 -- about 4e12 flops per
# SVD, 460 of them per theta point, with BLAS pinned to one thread. A single
# theta point is then thousands of core-hours, which is why task 10 sat on its
# first point for four hours looking hung.
#
# Raising the cap was meant for the TRAJECTORY route, which is an MPS of local
# dimension 2 and was genuinely censored at 256. The MPDO does not need it: it
# is known to lose by many orders of magnitude and is carried here for context.
# So the two are capped independently, and the MPDO can be skipped outright.
maxdim_mpdo = parse(Int, getenv("MAXDIM_MPDO", min(maxdim, 256)))
maxdim_traj = parse(Int, getenv("MAXDIM_TRAJ", maxdim))
skip_mpdo   = parse(Bool, getenv("SKIP_MPDO", false))
blas_thr    = parse(Int, getenv("BLAS_THREADS", 1))
# CUTOFF FOR THE TRAJECTORY ROUTE, separate from the MPDO's.
#
# The stored bond dimension is set by the CUTOFF, not by what we report. Task 10
# ran with 1e-10 and stored 4096 Schmidt values to report chi_req(1e-6) = 954 --
# cost goes as chi^3, so that is roughly a 75x penalty for precision that is then
# discarded. 1e-8 is still a hundred times tighter than the tolerance we quote.
cutoff_traj = parse(Float64, getenv("CUTOFF_TRAJ", 1e-8))
# 1 thread suits trajectory work (many small independent problems). Raise it via
# BLAS_THREADS for MPDO-dominated modes, where one big SVD is the whole job.
# This call must come AFTER blas_thr is parsed.
BLAS.set_num_threads(blas_thr)
cutoff  = parse(Float64, getenv("CUTOFF",1e-12))
ntraj   = parse(Int, getenv("NTRAJ",64))
kref    = parse(Int, getenv("KREF",1000))
JREF    = parse(Float64, getenv("JREF",0.25))
GREF    = parse(Float64, getenv("GREF",0.0625))
excited = Symbol(getenv("EXCITED","neel"))
tag     = getenv("TAG",""); sfx = isempty(tag) ? "" : "_"*tag

# Denser where the entangling power is structured; pi is excluded because the
# gate is exactly trivial there.
default_thetas = "0.1,0.2,0.35,0.5,0.65,0.8,0.95,1.1,1.25,1.4,1.5708,1.7,1.9,2.1,2.4,2.7,3.0"
thetas = parse.(Float64, split(getenv("THETAS", default_thetas),","))

@printf("=== circuit study, MODE=%s ===\n", mode)
@printf("k=%d  cutoff=%.1e  excited=%s  Ntraj=%d  BLAS threads=%d\n",
        k, cutoff, excited, ntraj, blas_thr)
@printf("maxdim: MPDO=%d  trajectory=%d%s\n", maxdim_mpdo, maxdim_traj,
        skip_mpdo ? "   (MPDO SKIPPED)" : "")
@printf("cutoff: MPDO=%.1e  trajectory=%.1e\n", cutoff, cutoff_traj)
@printf("reference units for the printed (dt,t): J=%.4f  gamma=%.4f\n\n", JREF, GREF)
flush(stdout)

open(joinpath(outdir,"manifest.csv"),"w") do io
    println(io,"key,value")
    for (a,b) in [("mode",mode),("n",n),("nlist",join(nlist," ")),("k",k),("p",pfix),
                  ("plist",join(plist," ")),("theta",thetafx),("thetas",join(thetas," ")),
                  ("maxdim",maxdim),("cutoff",cutoff),("ntraj",ntraj),("kref",kref),
                  ("excited",excited),("Jref",JREF),("Gref",GREF),
                  ("threads",Threads.nthreads()),("host",gethostname()),("started",Dates.now())]
        println(io,"$a,$b")
    end
end

# dt and t implied by a given theta, in the reference units above
dt_of(th) = th/(2*JREF)
t_of(th)  = k*dt_of(th)

# MAXDIM=0 means "no truncation at all": use the exact Liouville ceiling
# 4^(n/2) for this n. That is affordable only for n <= 12 (256 / 1024 / 4096),
# but where it is affordable the MPDO result carries no caveat whatsoever --
# it is the exact cost of the circuit, not a lower bound.
exact_ceiling(nn) = state_bond_dim_ceiling(nn)
md_for(nn) = maxdim == 0 ? exact_ceiling(nn) : maxdim


# =============================================================================
function run_theta()
    rows = ["n,k,theta,p,dt,t,gamma_over_J,S_op,chi_mpdo,linkdim,trace,sat_mpdo," *
            "S_traj_mean,S_traj_p95,chi_traj_mean,chi_traj_sem,chi_traj_max,sat_traj,wall_s"]
    # BOTH saturation flags. The first version printed only the MPDO's, and the
    # n=24 runs came back with chi_traj pinned at exactly 256 = MAXDIM without
    # anything in the table saying so.
    @printf("%7s %8s %8s | %8s %9s %4s | %8s %9s %4s\n",
            "theta","dt","t","S_op","chi_MPDO","sat","S_traj","chi_traj","sat")
    println("-"^80)
    for th in thetas
        # This function previously used md_for(n) for BOTH routes and never
        # consulted SKIP_MPDO, MAXDIM_MPDO or MAXDIM_TRAJ -- an earlier patch
        # matched text that was not here and silently did nothing. With MAXDIM
        # unset, md_for(24) = 1024, so the n=24 run computed the MPDO at 1024
        # despite SKIP_MPDO=true, and capped the trajectory at max(1024,1024) =
        # 1024 while the header announced 4096. chi_traj = 1009 was therefore
        # censored, and ~half the 7.4 h went on an MPDO that should not have run.
        c = skip_mpdo ? (S_op=NaN, chi=0, linkdim=0, trace=NaN, saturated=false) :
            circuit_cost_point(n, th, pfix, k; cutoff=cutoff, maxdim=maxdim_mpdo,
                               excited=excited)
        e = circuit_trajectory_ensemble(n, th, pfix, k, ntraj; cutoff=cutoff_traj,
                                        maxdim=maxdim_traj, excited=excited,
                                        verbose=false)
        f = e.series[end]
        gJ = pfix > 0 ? (-log(1-pfix)/dt_of(th))/JREF : 0.0
        @printf("%7.4f %8.4f %8.3f | %8.4f %9d %4s | %8.4f %9.1f %4s\n",
                th, dt_of(th), t_of(th), c.S_op, c.chi, c.saturated ? "!" : " ",
                f.S_mean, f.chi_mean, (f.censored ? "CEN" : (f.saturated ? "!!" : " "))); flush(stdout)
        push!(rows, join([n,k,@sprintf("%.6f",th),@sprintf("%.6f",pfix),
            @sprintf("%.6f",dt_of(th)),@sprintf("%.6f",t_of(th)),@sprintf("%.6f",gJ),
            @sprintf("%.8f",c.S_op),c.chi,c.linkdim,@sprintf("%.8f",c.trace),c.saturated,
            @sprintf("%.8f",f.S_mean),@sprintf("%.8f",f.S_p95),
            @sprintf("%.4f",f.chi_mean),@sprintf("%.4f",f.chi_sem),
            @sprintf("%.0f",f.chi_max),f.saturated,
            @sprintf("%.2f",mean(e.walltime))],","))
    end
    fpath = joinpath(outdir,"theta_sweep_n$(n)$(sfx).csv")
    write(fpath, join(rows,"\n")*"\n"); @printf("\nwrote %s\n", fpath)
end

# =============================================================================
function run_fidelity()
    # Small n on purpose: the reference needs KREF steps, and Trotter error is a
    # short-range property, so n=8 is representative and n=8 MPDO is EXACT
    # (Liouville ceiling 4^4 = 256).
    rows = ["n,k,kref,theta,p,dt,t,infidelity_dZ,infidelity_HS,trace_distance,S_op,chi_mpdo,sat"]
    @printf("reference: %d steps for the same (J*t, gamma*t)\n\n", kref)
    @printf("%7s %8s %8s | %10s %10s %10s | %8s %8s\n",
            "theta","dt","t","max|dZ|","HS dist","trace dist","S_op","chi")
    println("-"^80)
    for th in thetas
        infz, infhs, inftd, _, _ = trotter_infidelity(n, th, pfix, k; k_ref=kref,
                                cutoff=cutoff, maxdim=maxdim_mpdo, excited=excited)
        c = circuit_cost_point(n, th, pfix, k; cutoff=cutoff, maxdim=maxdim_mpdo,
                               excited=excited)
        @printf("%7.4f %8.4f %8.3f | %10.3e %10.3e %10.4f | %8.4f %8d\n",
                th, dt_of(th), t_of(th), infz, infhs, inftd, c.S_op, c.chi); flush(stdout)
        push!(rows, join([n,k,kref,@sprintf("%.6f",th),@sprintf("%.6f",pfix),
            @sprintf("%.6f",dt_of(th)),@sprintf("%.6f",t_of(th)),
            @sprintf("%.6e",infz),@sprintf("%.6e",infhs),@sprintf("%.6f",inftd),
            @sprintf("%.8f",c.S_op),c.chi,c.saturated],","))
    end
    fpath = joinpath(outdir,"fidelity_n$(n)$(sfx).csv")
    write(fpath, join(rows,"\n")*"\n"); @printf("\nwrote %s\n", fpath)
    println("\nPlot S_op (or chi) against infidelity: that curve is the")
    println("hardness-versus-faithfulness trade-off, and the operating point")
    println("should be a stated choice on it rather than an implicit one.")
end

# =============================================================================
function run_damping()
    rows = ["n,k,theta,p,total_damping,S_op,chi_mpdo,sat_mpdo,S_traj_mean,chi_traj_mean,chi_traj_sem"]
    @printf("theta=%.4f fixed\n\n", thetafx)
    @printf("%8s %14s | %8s %9s %4s | %8s %9s %4s\n",
            "p","1-(1-p)^k","S_op","chi_MPDO","sat","S_traj","chi_traj","sat")
    println("-"^72)
    for pp in plist
        c = skip_mpdo ? (S_op=NaN, chi=0, linkdim=0, trace=NaN, saturated=false) :
            circuit_cost_point(n, thetafx, pp, k; cutoff=cutoff,
                               maxdim=maxdim_mpdo, excited=excited)
        e = circuit_trajectory_ensemble(n, thetafx, pp, k, ntraj; cutoff=cutoff_traj,
                                        maxdim=maxdim_traj, excited=excited, verbose=false)
        f = e.series[end]; tot = 1-(1-pp)^k
        @printf("%8.4f %14.4f | %8.4f %9d %4s | %8.4f %9.1f %4s\n",
                pp, tot, c.S_op, c.chi, c.saturated ? "!" : " ",
                f.S_mean, f.chi_mean, (f.censored ? "CEN" : (f.saturated ? "!!" : " "))); flush(stdout)
        push!(rows, join([n,k,@sprintf("%.6f",thetafx),@sprintf("%.6f",pp),
            @sprintf("%.6f",tot),@sprintf("%.8f",c.S_op),c.chi,c.saturated,
            @sprintf("%.8f",f.S_mean),@sprintf("%.4f",f.chi_mean),
            @sprintf("%.4f",f.chi_sem)],","))
    end
    fpath = joinpath(outdir,"damping_n$(n)$(sfx).csv")
    write(fpath, join(rows,"\n")*"\n"); @printf("\nwrote %s\n", fpath)
end

# =============================================================================
function run_scaling()
    rows = ["n,k,theta,p,S_op,chi_mpdo,sat_mpdo,ceiling," *
            "S_traj_mean,chi_traj_mean,chi_traj_sem,chi3_traj,chi_traj_max,sat_traj,wall_s"]
    @printf("theta=%.4f  p=%.4f  k=%d%s\n\n", thetafx, pfix, k,
            maxdim == 0 ? "   [MPDO EXACT: maxdim = 4^(n/2), no truncation]" : "")
    @printf("%5s | %8s %10s %4s | %8s %10s %4s %10s\n",
            "n","S_op","chi_MPDO","sat","S_traj","chi_traj","sat","wall/traj")
    println("-"^76)
    for nn in nlist
        # Same bug as run_theta, caught by auditing every call site: md_for(nn)
        # for both routes, SKIP_MPDO ignored. Left alone, every Step 4 task would
        # have capped the trajectory at 1024 and paid for a full MPDO -- i.e. a
        # scaling curve censored at precisely the n values that set the exponent.
        c = skip_mpdo ? (S_op=NaN, chi=0, linkdim=0, trace=NaN, saturated=false) :
            circuit_cost_point(nn, thetafx, pfix, k; cutoff=cutoff,
                               maxdim=maxdim_mpdo, excited=excited)
        e = circuit_trajectory_ensemble(nn, thetafx, pfix, k, ntraj; cutoff=cutoff_traj,
                                        maxdim=maxdim_traj,
                                        excited=excited, verbose=false)
        f = e.series[end]
        @printf("%5d | %8.4f %10d %4s | %8.4f %10.1f %4s %10.1f\n",
                nn, c.S_op, c.chi, c.saturated ? "!" : " ", f.S_mean, f.chi_mean,
                (f.censored ? "CEN" : (f.saturated ? "!!" : " ")), mean(e.walltime)); flush(stdout)
        push!(rows, join([nn,k,@sprintf("%.6f",thetafx),@sprintf("%.6f",pfix),
            @sprintf("%.8f",c.S_op),c.chi,c.saturated,state_bond_dim_ceiling(nn),
            @sprintf("%.8f",f.S_mean),@sprintf("%.4f",f.chi_mean),
            @sprintf("%.4f",f.chi_sem),@sprintf("%.6e",f.chi3_mean),
            @sprintf("%.0f",f.chi_max),f.saturated,
            @sprintf("%.2f",mean(e.walltime))],","))
    end
    fpath = joinpath(outdir,"scaling$(sfx).csv")
    write(fpath, join(rows,"\n")*"\n"); @printf("\nwrote %s\n", fpath)
end

# =============================================================================
function run_mpdoladder()
    # Is the MPDO converged, or is it just reporting the cap back at us?
    #
    # chi is read off the STORED Schmidt spectrum, so it can never exceed
    # maxdim. If chi tracks the ladder, the measurement is censored and the only
    # honest statement is "chi_MPDO > (largest rung)". If it flattens, that
    # value is the real requirement. S_op converges in maxdim far faster than
    # chi does, so expect S_op to settle while chi keeps climbing.
    #
    # The trajectory route is run once at the top of the ladder for contrast.
    mds = parse.(Int, split(getenv("MAXDIMS","128,256,512,1024,2048"),","))
    ceil_n = exact_ceiling(n)
    @printf("n=%d theta=%.4f p=%.4f k=%d   exact ceiling 4^%d = %d\n\n",
            n, thetafx, pfix, k, n÷2, ceil_n)
    rows = ["n,k,theta,p,maxdim,ceiling,S_op,chi_mpdo,linkdim,saturated,exact"]
    println("   maxdim |     S_op   chi_MPDO   linkdim | status")
    println("-"^62)
    for md in mds
        md > ceil_n && (println("   (skipping $md: above the exact ceiling $ceil_n)"); continue)
        c = circuit_cost_point(n, thetafx, pfix, k; cutoff=cutoff, maxdim=md,
                               excited=excited)
        st = md >= ceil_n ? "EXACT" : (c.saturated ? "censored: chi > $(c.chi)" : "converged?")
        @printf("%9d | %8.4f %10d %9d | %s\n", md, c.S_op, c.chi, c.linkdim, st)
        flush(stdout)
        push!(rows, join([n,k,@sprintf("%.6f",thetafx),@sprintf("%.6f",pfix),md,ceil_n,
            @sprintf("%.8f",c.S_op),c.chi,c.linkdim,c.saturated,md>=ceil_n],","))
    end
    e = circuit_trajectory_ensemble(n, thetafx, pfix, k, ntraj; cutoff=cutoff_traj,
                                    maxdim=4096, excited=excited, verbose=false)
    f = e.series[end]
    @printf("\n  trajectory route at the same point: chi = %.1f +/- %.1f  (maxdim 4096, %s)\n",
            f.chi_mean, f.chi_sem, f.saturated ? "SATURATED" : "not saturated")
    fpath = joinpath(outdir,"mpdoladder_n$(n)$(sfx).csv")
    write(fpath, join(rows,"\n")*"\n"); @printf("wrote %s\n", fpath)
end

# =============================================================================
function run_headtohead()
    # WHY SMALL n IS THE POINT, NOT A LIMITATION.
    #
    # The vectorised density matrix is an MPS of local dimension 4, so its bond
    # dimension cannot exceed 4^(n/2): 256 at n=8, 1024 at n=10, 4096 at n=12.
    # Setting maxdim to that ceiling makes the MPDO EXACT -- no truncation at
    # all. The trajectory route is bounded by 2^(n/2) per system cut, far
    # smaller, and is likewise exact here. So this table is a direct, untruncated
    # comparison of the two methods on identical circuits, with no extrapolation
    # and no censored numbers anywhere. That is something the large-n runs can
    # never give, because there both methods have to be truncated.
    #
    # maxdim binds cost only when it is REACHED, so setting it to the ceiling is
    # free whenever the physical chi stays below -- which at k=10 it does.
    #
    # Three things come out:
    #   1. the two methods agree on the physics (<Z_j>, checked in units of the
    #      trajectory standard error) -- the trust-building row;
    #   2. chi_MPDO vs chi_traj, and their ratio against chi_traj^2, which is
    #      what the relation would be if rho stayed pure (the operator Schmidt
    #      values of |psi><psi| are the pairwise products of those of |psi>);
    #   3. the honest cost ratio  chi_MPDO^3 / (N * <chi_traj^3>), with N taken
    #      from the measured per-trajectory variance rather than assumed.
    rows = ["n,k,theta,p,ceiling_mpdo,maxdim_used,exact_mpdo,S_op,chi_mpdo," *
            "S_traj,chi_traj_mean,chi_traj_sem,chi_traj_max,chi3_traj,sat_traj," *
            "chi_ratio,chi_over_chitraj_sq,Ntraj_needed,cost_mpdo,cost_traj," *
            "mpdo_over_traj,max_dZ,max_dZ_in_sem,wall_mpdo_s,wall_traj_s"]
    @printf("theta=%.4f  p=%.4f  k=%d  excited=%s  target SEM on <Z> = 0.01\n\n",
            thetafx, pfix, k, excited)
    @printf("%4s | %9s %6s | %8s %8s | %9s %9s | %10s | %8s\n",
            "n","ceil","exact","chi_MPDO","chi_traj","ratio","/chi_t^2",
            "cost ratio","dZ[sem]")
    println("-"^92)
    for nn in nlist
        ceil_n = 4^min(nn÷2, nn-nn÷2)
        md = min(ceil_n, maxdim)
        wm = @elapsed c = circuit_cost_point(nn, thetafx, pfix, k; cutoff=cutoff,
                                             maxdim=md, excited=excited)
        wt = @elapsed e = circuit_trajectory_ensemble(nn, thetafx, pfix, k, ntraj;
                                cutoff=1e-12, maxdim=md, excited=excited,
                                verbose=false)
        f = e.series[end]
        zm = mpdo_expectation_Z_all(nn, thetafx, pfix, k; cutoff=cutoff,
                                    maxdim=md, excited=excited)
        dz  = maximum(abs.(f.z_all .- zm))
        dzs = maximum(abs.(f.z_all .- zm) ./ max.(f.z_all_sem, 1e-12))
        N   = Ntraj_for(e.series, 0.01)
        cost_m = float(c.chi)^3
        cost_t = N * f.chi3_mean
        exact  = md >= ceil_n && !c.saturated
        @printf("%4d | %9d %6s | %8d %8.1f | %9.2f %9.2f | %10.3e | %8.1f\n",
                nn, ceil_n, exact ? "yes" : "NO", c.chi, f.chi_mean,
                c.chi/max(f.chi_mean,1e-9), c.chi/max(f.chi_mean^2,1e-9),
                cost_m/max(cost_t,1e-300), dzs); flush(stdout)
        push!(rows, join([nn,k,@sprintf("%.6f",thetafx),@sprintf("%.6f",pfix),
            ceil_n, md, exact, @sprintf("%.8f",c.S_op), c.chi,
            @sprintf("%.8f",f.S_mean), @sprintf("%.4f",f.chi_mean),
            @sprintf("%.4f",f.chi_sem), @sprintf("%.0f",f.chi_max),
            @sprintf("%.6e",f.chi3_mean), f.saturated,
            @sprintf("%.4f",c.chi/max(f.chi_mean,1e-9)),
            @sprintf("%.4f",c.chi/max(f.chi_mean^2,1e-9)), N,
            @sprintf("%.6e",cost_m), @sprintf("%.6e",cost_t),
            @sprintf("%.6e",cost_m/max(cost_t,1e-300)),
            @sprintf("%.3e",dz), @sprintf("%.2f",dzs),
            @sprintf("%.2f",wm), @sprintf("%.2f",wt)],","))
    end
    fpath = joinpath(outdir,"headtohead$(sfx).csv")
    write(fpath, join(rows,"\n")*"\n"); @printf("\nwrote %s\n", fpath)
    println("\nREADING IT:")
    println("  exact=yes   -> neither method was truncated; the row is a measurement,")
    println("                 not a bound, and needs no caveat.")
    println("  dZ[sem]     -> agreement on the physics, in trajectory standard errors.")
    println("                 Should sit around 1-2. If it grows with n, suspect the")
    println("                 simulation, not the methods.")
    println("  /chi_t^2    -> chi_MPDO / chi_traj^2. Equals 1 when rho is pure, since")
    println("                 the operator Schmidt values of |psi><psi| are the")
    println("                 pairwise products. Below 1 means mixing is genuinely")
    println("                 destroying correlations and helping the MPDO.")
    println("  cost ratio  -> chi_MPDO^3 / (N*<chi_traj^3>), N from the measured")
    println("                 variance. This is the number the method choice rests on.")
end

# =============================================================================
function run_map()
    # THE 2D SWEEP. One job replaces the theta sweep and the damping sweep:
    # both are slices of this. Axes are the two knobs the circuit actually has,
    #     theta  the rxx angle            = 2*J*dt
    #     p      per-step jump probability = 1 - exp(-gamma*dt)
    # at fixed n and k. Everything else about the circuit is determined.
    #
    # Cost is (grid points) x ceil(NTRAJ/threads) x (wall per trajectory), and
    # wall/traj was measured at 9 s (n=8), 13 s (n=12), 59 s (n=16). So a
    # 13 x 9 grid at n=16 with NTRAJ=32 is about 4 h -- one job, no ladder.
    rows = ["n,k,theta,p,dt,t,total_damping,S_traj_mean,S_traj_p95," *
            "chi_mean,chi_sem,chi_max,chi3_mean,censored,saturated,wall_s," *
            "S_op_mpdo,chi_mpdo,sat_mpdo"]
    @printf("grid: %d theta x %d p = %d points, Ntraj=%d\n\n",
            length(thetas), length(plist), length(thetas)*length(plist), ntraj)
    @printf("%8s |", "theta")
    for pp in plist; @printf(" p=%-6.3f", pp); end
    println("\n" * "-"^(10 + 9*length(plist)))
    done = 0
    for th in thetas
        @printf("%8.4f |", th)
        for pp in plist
            e = circuit_trajectory_ensemble(n, th, pp, k, ntraj; cutoff=cutoff_traj,
                                            maxdim=maxdim_traj, excited=excited,
                                            verbose=false)
            f = e.series[end]
            c = skip_mpdo ? (S_op=NaN, chi=0, saturated=false) :
                circuit_cost_point(n, th, pp, k; cutoff=cutoff, maxdim=maxdim_mpdo,
                                   excited=excited)
            @printf(" %7.1f%s", f.chi_mean, f.censored ? "*" : " ")
            push!(rows, join([n,k,@sprintf("%.6f",th),@sprintf("%.6f",pp),
                @sprintf("%.6f",dt_of(th)),@sprintf("%.6f",t_of(th)),
                @sprintf("%.6f",1-(1-pp)^k),
                @sprintf("%.6f",f.S_mean),@sprintf("%.6f",f.S_p95),
                @sprintf("%.4f",f.chi_mean),@sprintf("%.4f",f.chi_sem),
                @sprintf("%.0f",f.chi_max),@sprintf("%.6e",f.chi3_mean),
                f.censored,f.saturated,@sprintf("%.2f",mean(e.walltime)),
                @sprintf("%.6f",c.S_op),c.chi,c.saturated],","))
            done += 1
        end
        println(); flush(stdout)
        # Written after every row, so a timeout still leaves a usable partial map.
        write(joinpath(outdir,"map_n$(n)$(sfx).csv"), join(rows,"\n")*"\n")
    end
    @printf("\nwrote %s  (%d points)\n", joinpath(outdir,"map_n$(n)$(sfx).csv"), done)
    println("  * next to a value means the reported chi is within 10% of the cap.")
    println("  Plot with:  python3 plot_map.py <that csv>")
end

# =============================================================================
if     mode == "theta";    run_theta()
elseif mode == "mpdoladder"; run_mpdoladder()
elseif mode == "fidelity"; run_fidelity()
elseif mode == "damping";  run_damping()
elseif mode == "scaling";  run_scaling()
elseif mode == "headtohead"; run_headtohead()
elseif mode == "map";      run_map()
else error("MODE must be one of: theta, fidelity, damping, scaling, mpdoladder, headtohead, map. Got '$mode'.")
end
println("\n[stage] done"); flush(stdout)
