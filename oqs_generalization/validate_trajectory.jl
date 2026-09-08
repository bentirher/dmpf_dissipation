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
const NSITE = 8

# -----------------------------------------------------------------------------
# EVERYTHING BELOW RUNS INSIDE FUNCTIONS, ON PURPOSE.
#
# In a Julia SCRIPT, assigning to an existing global from inside a top-level
# `for` loop creates a NEW LOCAL rather than touching the global, and reading it
# before that assignment raises `UndefVarError: ... not defined in local scope`.
# That is exactly what killed the first run of this file (`worst` at line 91),
# and it killed an earlier driver in this project the same way. Wrapping each
# test in a function removes the soft-scope rule entirely, so accumulators are
# ordinary locals and every future one is safe by construction.
# -----------------------------------------------------------------------------

"""
TEST A -- gamma = 0. Every reset outcome is 0 with probability 1, so the
trajectory is deterministic: different seeds must agree bit for bit, and the
result must match the closed-system wavefunction gate for gate. Zero statistics.
Covers the Hamiltonian layers, the q-a-q-a interleaving, the non-adjacent
two-site gates (applied by swapping through the ancilla) and the Strang order.
"""
function test_A(n::Int)
    println("="^74)
    println("TEST A -- gamma = 0: trajectory must be deterministic and match closed MPS")
    println("="^74)
    ts = collect(1.0:1.0:6.0)
    kw = (dt=0.05, cutoff=1e-14, maxdim=1024, tols=[1e-6], initial=:neel)
    r1 = single_trajectory(n, fill(J0, n-1), zeros(n), ts; seed=1,      kw...)
    r2 = single_trajectory(n, fill(J0, n-1), zeros(n), ts; seed=999999, kw...)

    dseed = maximum(abs(real(a.z[4]) - real(b.z[4])) for (a,b) in zip(r1,r2))
    @printf("  seed-independence  : max |dZ| between seeds = %.3e   %s\n",
            dseed, dseed < 1e-12 ? "PASS" : "FAIL (randomness is being consumed!)")

    if !@isdefined(evolve_closed)
        println("\n  (closed reference not loaded -- second half of Test A skipped)")
        return dseed < 1e-12
    end

    rc = evolve_closed(n, J0, ts; dt=0.05, order=2, cutoff=1e-14, maxdim=1024,
                       initial=:neel, tols=[1e-6], verbose=false)
    println("\n      t |   Z(traj)     Z(closed)      diff")
    worst = 0.0
    for (i, t) in enumerate(ts)
        zt = real(r1[i+1].z[4]); zc = real(rc.z_mid[i+1])
        worst = max(worst, abs(zt - zc))
        @printf("%7.2f | %10.7f  %10.7f  %+.2e\n", t, zt, zc, zt - zc)
    end
    ok = worst < 1e-8
    @printf("\n  vs closed MPS      : max |dZ| = %.3e   %s\n", worst,
            ok ? "PASS" : "FAIL -- Hamiltonian layers or layout are wrong")
    return (dseed < 1e-12) && ok
end

"""
TEST B -- J = 0. The qubits decouple into independent single-qubit
amplitude-damping processes, <Z_j>(t) = 1 - 2exp(-gamma t) for an initially
excited qubit and +1 otherwise. Covers the Born sampling, the reset and the
channel angle.

Run at two step sizes: with J = 0 there is NO Trotter error (applying the
channel k times with step dt is exactly the channel over k*dt), so dt=0.5 and
dt=0.05 must agree. A bias at dt=0.05 but not at dt=0.5 accumulates PER STEP;
one equal at both is per jump. dt=0.5 is 10x cheaper and carries the statistics.
"""
function test_B(n::Int, NR::Int)
    println()
    println("="^74)
    println("TEST B -- J = 0: independent qubits, analytic <Z_j>(t) = 1 - 2exp(-gamma t)")
    println("="^74)
    gamma = 0.30
    ts2 = collect(1.0:1.0:6.0)
    excited = collect(1:2:n); ground = setdiff(1:n, excited)
    @printf("  Ntraj = %d; sites are independent at J=0 and equivalent sites are\n", NR)
    @printf("  pooled, so N_eff = %d per group -- SEM ~ %.4f.\n\n",
            NR*length(excited), 1/sqrt(NR*length(excited)))

    function sweep(dt, NRl)
        acc  = zeros(length(ts2)+1, n); acc2 = zeros(length(ts2)+1, n)
        locks = Threads.SpinLock()
        Threads.@threads for k in 1:NRl
            r = single_trajectory(n, zeros(n-1), fill(gamma, n), ts2; dt=dt,
                                  cutoff=1e-14, maxdim=4, seed=31337+k,
                                  tols=[1e-6], initial=:neel)
            lock(locks) do
                for i in eachindex(r), j in 1:n
                    z = real(r[i].z[j]); acc[i,j] += z; acc2[i,j] += z*z
                end
            end
        end
        return acc, acc2
    end

    worstsig = 0.0
    for (dt, NRl) in ((0.5, NR), (0.05, max(NR ÷ 5, 200)))
        acc, acc2 = sweep(dt, NRl)
        @printf("\n  --- dt = %.3g, Ntraj = %d, %d steps to t=6 ---\n",
                dt, NRl, round(Int, 6/dt))
        println("      t | group   |   <Z> measured      analytic       diff     n_sigma")
        for (i, t) in enumerate(ts2)
            row = i + 1
            for (grp, name, exact) in ((excited, "excited", 1 - 2*exp(-gamma*t)),
                                       (ground,  "ground ", 1.0))
                m  = sum(acc[row,j]  for j in grp) / (NRl*length(grp))
                m2 = sum(acc2[row,j] for j in grp) / (NRl*length(grp))
                sem = sqrt(max(m2 - m^2, 0.0) / (NRl*length(grp)))
                nsig = sem > 1e-12 ? (m - exact)/sem : 0.0
                worstsig = max(worstsig, abs(nsig))
                @printf("%7.2f | %s | %12.7f  %12.7f  %+.2e   %+6.2f\n",
                        t, name, m, exact, m - exact, nsig)
            end
        end
    end
    ok = worstsig < 3
    @printf("\n  worst |n_sigma| over both step sizes = %.2f   %s\n", worstsig,
            ok ? "PASS -- sampling and reset are unbiased" :
                 "FAIL -- bias is in reset_ancilla! / the channel angle")
    return ok
end

function main()
    okA = test_A(NSITE)
    okB = test_B(NSITE, NREP)
    println()
    println("="^74)
    println("VERDICT")
    if okA && okB
        println("  Both PASS -> the -1.3 sigma in the n=8 validation was a fluctuation.")
        println("  Proceed to submit_trajectory_study.sh --array=1-3.")
    elseif okA && !okB
        println("  A pass, B FAIL -> bias is in the stochastic part: reset_ancilla!,")
        println("  the Born sampling, or the channel angle theta = 2 asin(sqrt(p)).")
        println("  Compare the two step sizes above to see if it is per-step or per-jump.")
    else
        println("  A FAIL -> bias is in the deterministic part (gate layers or the")
        println("  q-a-q-a layout). The stochastic agreement was masking it.")
    end
    println("="^74)
    return okA && okB
end

main()
