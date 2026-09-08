# =============================================================================
# validate_trajectory.jl
#
# The n=8 validation runs agreed with the exact MPDO curve to -1.3 sigma
# combined over two seeds, with 27 of 30 deviations negative (i.e. the
# trajectory ensemble slightly UNDER-damps). That is suggestive but not
# significant, and brute-forcing it needs ~1e4 trajectories.
#
# These two tests localise it instead, in minutes. Between them they cover
# every moving part, and each has ZERO systematic error of its own.
#
#   TEST A  gamma = 0.  Every reset outcome is 0 with probability 1, so the
#           trajectory is DETERMINISTIC and must reproduce the closed-system
#           wavefunction gate for gate. No statistics at all. This tests the
#           Hamiltonian layers, the q-a-q-a interleaving, the non-adjacent
#           two-site gates (which ITensor implements by swapping through the
#           ancilla), the Strang ordering and the site bookkeeping.
#
#   TEST B  J = 0, gamma > 0.  The qubits decouple, so each qubit-plus-ancilla
#           pair is an independent single-qubit amplitude-damping process with
#           the analytic solution <Z_j>(t) = 1 - 2 exp(-gamma t) for an
#           initially excited qubit and +1 for one starting in |0>. This tests
#           the Born-rule sampling, the reset, and the channel parameterisation.
#
#           Test B is nearly free and therefore very sharp: with J=0 the state
#           stays a product (chi = 1), so 20000 trajectories cost nothing, AND
#           the n sites are statistically INDEPENDENT, so the effective sample
#           size is n * Ntraj. It resolves a bias of 0.015 at many sigma.
#
# If A passes and B fails -> the problem is in reset_ancilla! / the sampling.
# If A fails               -> the problem is in the gate layers or the layout,
#                             and the stochastic part is a red herring.
# If both pass             -> the -1.3 sigma really was a fluctuation; proceed.
#
# Run on a login node: julia validate_trajectory.jl
# =============================================================================
using Printf, Statistics, LinearAlgebra

# The closed-system reference for the second half of Test A lives in the MPDO
# study's files. They pull the whole project include chain, which may not be
# present in this working directory, so load them opportunistically.
let d = @__DIR__
    if isfile(joinpath(d,"vectorized_evolution.jl")) &&
       isfile(joinpath(d,"closed_evolution.jl")) &&
       isfile(joinpath(d,"F_diagnostics.jl"))
        try
            include(joinpath(d,"vectorized_evolution.jl"))
            include(joinpath(d,"closed_evolution.jl"))
            println("[stage] closed-system reference available (full Test A)")
        catch e
            println("[stage] could not load the closed reference: ", e)
            println("[stage] Test A will run its seed-independence half only")
        end
    else
        println("[stage] closed reference files not in this directory;")
        println("[stage] Test A will run its seed-independence half only")
    end
end
include(joinpath(@__DIR__, "trajectory_evolution.jl"))

const J0   = 0.5
const NREP = parse(Int, get(ENV, "NREP", "6000"))

# -----------------------------------------------------------------------------
println("="^74)
println("TEST A -- gamma = 0: trajectory must be deterministic and match closed MPS")
println("="^74)

n  = 8
ts = collect(1.0:1.0:6.0)

# Two trajectories with DIFFERENT seeds. With gamma=0 they must be identical to
# each other (no randomness is ever consumed) and to the closed-system answer.
r1 = single_trajectory(n, fill(J0, n-1), zeros(n), ts; dt=0.05, cutoff=1e-14,
                       maxdim=1024, seed=1, tols=[1e-6], initial=:neel)
r2 = single_trajectory(n, fill(J0, n-1), zeros(n), ts; dt=0.05, cutoff=1e-14,
                       maxdim=1024, seed=999999, tols=[1e-6], initial=:neel)

dseed = maximum(abs(real(a.z[4]) - real(b.z[4])) for (a,b) in zip(r1,r2))
@printf("  seed-independence  : max |dZ| between seeds = %.3e   %s\n",
        dseed, dseed < 1e-12 ? "PASS" : "FAIL (randomness is being consumed!)")

# Reference: the pure-state closed evolution from the MPDO study, if available.
if @isdefined(evolve_closed)
    rc = evolve_closed(n, J0, ts; dt=0.05, order=2, cutoff=1e-14, maxdim=1024,
                       initial=:neel, tols=[1e-6], verbose=false)
    println("\n      t |   Z(traj)     Z(closed)      diff")
    worst = 0.0
    for (i, t) in enumerate(ts)
        zt = real(r1[i+1].z[4]); zc = real(rc.z_mid[i+1])
        worst = max(worst, abs(zt - zc))
        @printf("%7.2f | %10.7f  %10.7f  %+.2e\n", t, zt, zc, zt - zc)
    end
    @printf("\n  vs closed MPS      : max |dZ| = %.3e   %s\n", worst,
            worst < 1e-8 ? "PASS" : "FAIL -- Hamiltonian layers or layout are wrong")
else
    println("\n  (closed_evolution.jl not loaded; include it to get the second half of Test A)")
    println("  include(\"vectorized_evolution.jl\"); include(\"closed_evolution.jl\")")
end

# -----------------------------------------------------------------------------
println()
println("="^74)
println("TEST B -- J = 0: independent qubits, analytic <Z_j>(t) = 1 - 2exp(-gamma t)")
println("="^74)
@printf("  Ntraj = %d; the %d sites are statistically independent at J=0, and\n", NREP, n)
@printf("  equivalent sites are pooled, so the effective sample size is %d\n", NREP*(n÷2))
@printf("  per time point per group -- SEM ~ %.4f, sharp enough to resolve the\n",
        1/sqrt(NREP*(n÷2)))
println("  ~0.02 discrepancy seen in the n=8 validation at many sigma.\n")

# Two step sizes, deliberately. With J = 0 there is NO Trotter error -- applying
# the damping channel k times with step dt is EXACTLY the channel over k*dt --
# so dt=0.5 and dt=0.05 must give identical answers. If a bias appears at
# dt=0.05 but not at dt=0.5 it accumulates PER STEP; if it is the same at both
# it is per unit time (i.e. per jump). That distinction points straight at the
# offending line. dt=0.5 is also 10x cheaper, so it carries most of the
# statistics.
gamma = 0.30
ts2   = collect(1.0:1.0:6.0)
excited = collect(1:2:n)              # :neel -- odd sites start in |1>
ground  = setdiff(1:n, excited)

function testB(dt::Float64, NR::Int)
    acc  = zeros(length(ts2)+1, n); acc2 = zeros(length(ts2)+1, n)
    Threads.@threads for k in 1:NR
        r = single_trajectory(n, zeros(n-1), fill(gamma, n), ts2; dt=dt,
                              cutoff=1e-14, maxdim=4, seed=31337+k, tols=[1e-6],
                              initial=:neel)
        for i in eachindex(r), j in 1:n
            z = real(r[i].z[j]); acc[i,j] += z; acc2[i,j] += z*z
        end
    end
    return acc, acc2
end

worstsig = 0.0
for (dt, NR) in ((0.5, NREP), (0.05, max(NREP ÷ 5, 200)))
    acc, acc2 = testB(dt, NR)
    @printf("\n  --- dt = %.3g, Ntraj = %d, %d steps to t=6 ---\n",
            dt, NR, round(Int, 6/dt))
    println("      t | group    |   <Z> measured      analytic       diff     n_sigma")
    for (i, t) in enumerate(ts2)
        row = i + 1
        # Aggregate over EQUIVALENT sites: at J=0 they are independent and
        # identically distributed, so this multiplies the sample size by 4 and
        # is what makes this test sharp enough to matter.
        for (grp, name) in ((excited, "excited"), (ground, "ground "))
            m  = sum(acc[row,j]  for j in grp) / (NR*length(grp))
            m2 = sum(acc2[row,j] for j in grp) / (NR*length(grp))
            sem = sqrt(max(m2 - m^2, 0.0) / (NR*length(grp)))
            exact = name == "excited" ? 1 - 2*exp(-gamma*t) : 1.0
            nsig = sem > 1e-12 ? (m - exact)/sem : 0.0
            global worstsig = max(worstsig, abs(nsig))
            @printf("%7.2f | %s | %12.7f  %12.7f  %+.2e   %+6.2f\n",
                    t, name, m, exact, m-exact, nsig)
        end
    end
end
@printf("\n  worst |n_sigma| over both step sizes = %.2f   %s\n", worstsig,
        worstsig < 3 ? "PASS -- sampling and reset are unbiased" :
                       "FAIL -- bias is in reset_ancilla! / the channel angle")

println()
println("INTERPRETATION")
println("  A pass, B pass -> the -1.3 sigma in the n=8 validation was a fluctuation.")
println("                    Proceed to the operating-line runs.")
println("  A pass, B fail -> bias is in the stochastic part (sampling or reset).")
println("  A fail         -> bias is in the deterministic part; the stochastic")
println("                    agreement was masking a gate/layout error.")
