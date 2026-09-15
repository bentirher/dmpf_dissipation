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
BLAS.set_num_threads(1)

mode    = getenv("MODE","theta")
n       = parse(Int, getenv("N",16))
nlist   = parse.(Int, split(getenv("NLIST","8,12,16,20,24"),","))
k       = parse(Int, getenv("K",10))
pfix    = parse(Float64, getenv("P",0.05))
plist   = parse.(Float64, split(getenv("PLIST","0.0,0.01,0.02,0.05,0.10,0.15,0.25"),","))
thetafx = parse(Float64, getenv("THETA",1.5708))
maxdim  = parse(Int, getenv("MAXDIM",1024))
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
@printf("k=%d  maxdim=%d  cutoff=%.1e  excited=%s  Ntraj=%d\n", k, maxdim, cutoff, excited, ntraj)
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


# =============================================================================
function run_theta()
    rows = ["n,k,theta,p,dt,t,gamma_over_J,S_op,chi_mpdo,linkdim,trace,sat_mpdo," *
            "S_traj_mean,S_traj_p95,chi_traj_mean,chi_traj_sem,chi_traj_max,sat_traj,wall_s"]
    @printf("%7s %8s %8s | %8s %9s %6s | %8s %9s | %s\n",
            "theta","dt","t","S_op","chi_MPDO","sat","S_traj","chi_traj","")
    println("-"^86)
    for th in thetas
        c = circuit_cost_point(n, th, pfix, k; cutoff=cutoff, maxdim=maxdim,
                               excited=excited)
        e = circuit_trajectory_ensemble(n, th, pfix, k, ntraj; cutoff=1e-10,
                                        maxdim=maxdim, excited=excited,
                                        verbose=false)
        f = e.series[end]
        gJ = pfix > 0 ? (-log(1-pfix)/dt_of(th))/JREF : 0.0
        @printf("%7.4f %8.4f %8.3f | %8.4f %9d %6s | %8.4f %9.1f\n",
                th, dt_of(th), t_of(th), c.S_op, c.chi, c.saturated ? "!" : " ",
                f.S_mean, f.chi_mean); flush(stdout)
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
    rows = ["n,k,kref,theta,p,dt,t,infidelity,S_op,chi_mpdo,sat"]
    @printf("reference: %d steps for the same (J*t, gamma*t)\n\n", kref)
    @printf("%7s %8s %8s | %12s | %8s %9s\n","theta","dt","t","infidelity","S_op","chi")
    println("-"^62)
    for th in thetas
        inf, _, _ = trotter_infidelity(n, th, pfix, k; k_ref=kref, cutoff=cutoff,
                                       maxdim=maxdim, excited=excited)
        c = circuit_cost_point(n, th, pfix, k; cutoff=cutoff, maxdim=maxdim,
                               excited=excited)
        @printf("%7.4f %8.4f %8.3f | %12.3e | %8.4f %9d\n",
                th, dt_of(th), t_of(th), inf, c.S_op, c.chi); flush(stdout)
        push!(rows, join([n,k,kref,@sprintf("%.6f",th),@sprintf("%.6f",pfix),
            @sprintf("%.6f",dt_of(th)),@sprintf("%.6f",t_of(th)),
            @sprintf("%.6e",inf),@sprintf("%.8f",c.S_op),c.chi,c.saturated],","))
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
    @printf("%8s %14s | %8s %9s | %8s %9s\n","p","1-(1-p)^k","S_op","chi_MPDO","S_traj","chi_traj")
    println("-"^62)
    for pp in plist
        c = circuit_cost_point(n, thetafx, pp, k; cutoff=cutoff, maxdim=maxdim, excited=excited)
        e = circuit_trajectory_ensemble(n, thetafx, pp, k, ntraj; cutoff=1e-10,
                                        maxdim=maxdim, excited=excited, verbose=false)
        f = e.series[end]; tot = 1-(1-pp)^k
        @printf("%8.4f %14.4f | %8.4f %9d | %8.4f %9.1f\n",
                pp, tot, c.S_op, c.chi, f.S_mean, f.chi_mean); flush(stdout)
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
    @printf("theta=%.4f  p=%.4f  k=%d\n\n", thetafx, pfix, k)
    @printf("%5s | %8s %10s %6s | %8s %10s %10s\n",
            "n","S_op","chi_MPDO","sat","S_traj","chi_traj","wall/traj")
    println("-"^70)
    for nn in nlist
        c = circuit_cost_point(nn, thetafx, pfix, k; cutoff=cutoff, maxdim=maxdim, excited=excited)
        e = circuit_trajectory_ensemble(nn, thetafx, pfix, k, ntraj; cutoff=1e-10,
                                        maxdim=maxdim, excited=excited, verbose=false)
        f = e.series[end]
        @printf("%5d | %8.4f %10d %6s | %8.4f %10.1f %10.1f\n",
                nn, c.S_op, c.chi, c.saturated ? "!" : " ", f.S_mean, f.chi_mean,
                mean(e.walltime)); flush(stdout)
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
if     mode == "theta";    run_theta()
elseif mode == "fidelity"; run_fidelity()
elseif mode == "damping";  run_damping()
elseif mode == "scaling";  run_scaling()
else error("MODE must be theta, fidelity, damping or scaling. Got '$mode'.")
end
println("\n[stage] done"); flush(stdout)
