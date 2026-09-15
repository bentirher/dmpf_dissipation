# =============================================================================
# validate_circuit_native.jl  --  STEP 1 validation
#
# Reproduces EXACTLY the curve in ha_circuit_study.ipynb, so you can overlay it
# on the Qiskit AerSimulator output and confirm the Julia simulator is running
# the same circuit before any of the hardness sweeps are trusted.
#
# Notebook parameters: n=4, J=1/4 (uniform), gamma=J/4, excited=[qubit 0],
# k in {2,8,15}, t in [0,25], observable <(I - Z_0)/2> = excited population.
#
# In circuit units:  theta = 2*J*t/k ,  p = 1 - exp(-gamma*t/k)
#
# THREE CHECKS
#   A  MPDO vs Qiskit   -- same circuit? (you overlay; n=4 MPDO is EXACT here,
#                          the ceiling 4^2=16 is far below maxdim)
#   B  trajectories vs MPDO -- does the unravelling reproduce the channel?
#   C  dissipation=false vs the closed circuit -- sanity on the gate layers
#
# Run: julia --threads=8 validate_circuit_native.jl
# =============================================================================
using Printf, Statistics
include(joinpath(@__DIR__, "vectorized_evolution.jl"))
include(joinpath(@__DIR__, "trajectory_evolution.jl"))
include(joinpath(@__DIR__, "circuit_native.jl"))

const N      = 4
const JJ     = 0.25
const GAMMA  = JJ/4
const EXC    = [1]                 # qubit 0 in Qiskit -> site 1 in Julia
const TLIST  = collect(0.0:2.5:25.0)
const KLIST  = [2, 8, 15]
const NTRAJ  = parse(Int, get(ENV, "NTRAJ", "2000"))

println("="^78)
println("CHECK A -- excited population of qubit 0, MPDO. Overlay on the notebook.")
println("="^78)
@printf("n=%d  J=%.4f  gamma=%.4f  excited=[q0]\n\n", N, JJ, GAMMA)

rows = ["k,t,theta,p,pop_q0_mpdo"]
for k in KLIST
    @printf("--- k = %d ---\n", k)
    println("      t |  theta     p     | pop(q0) = (1-<Z_1>)/2")
    for t in TLIST
        th = theta_of(JJ, t, k); pp = p_of(GAMMA, t, k)
        z  = t == 0 ? -1.0 : mpdo_expectation_Z(N, th, pp, k, 1;
                                                excited=EXC, maxdim=256)
        pop = (1 - z)/2
        @printf("%7.2f | %6.4f  %6.4f | %10.6f\n", t, th, pp, pop)
        push!(rows, join([k, @sprintf("%.4f",t), @sprintf("%.6f",th),
                          @sprintf("%.6f",pp), @sprintf("%.8f",pop)], ","))
    end
    println()
end
write("validate_mpdo_population.csv", join(rows,"\n")*"\n")
println("wrote validate_mpdo_population.csv  -> plot against all_trotter in the notebook\n")

println("="^78)
println("CHECK B -- trajectories vs MPDO at the same circuit (k=8, a few times)")
println("="^78)
@printf("Ntraj=%d\n\n", NTRAJ)
println("      t |   pop MPDO    pop traj ± sem   diff    n_sigma")
k = 8
for t in [5.0, 10.0, 15.0, 20.0, 25.0]
    th = theta_of(JJ, t, k); pp = p_of(GAMMA, t, k)
    zm = mpdo_expectation_Z(N, th, pp, k, 1; excited=EXC, maxdim=256)
    ens = circuit_trajectory_ensemble(N, th, pp, k, NTRAJ;
                                      excited=EXC, maxdim=64, verbose=false)
    last = ens.series[end]
    pm, pt = (1-zm)/2, (1-last.z_mid)/2
    sem = last.z_sem/2
    ns  = sem > 0 ? (pt-pm)/sem : 0.0
    @printf("%7.2f | %10.6f   %8.6f±%.4f  %+.2e  %+6.2f\n", t, pm, pt, sem, pt-pm, ns)
end

println()
println("="^78)
println("CHECK C -- closed circuit (dissipation=false): trajectories must be")
println("           deterministic and match the MPDO exactly.")
println("="^78)
k = 8
for t in [10.0, 20.0]
    th = theta_of(JJ, t, k)
    zm = mpdo_expectation_Z(N, th, 0.0, k, 1; excited=EXC, maxdim=256, dissipation=false)
    r1 = circuit_trajectory(N, th, 0.0, k; excited=EXC, maxdim=256, seed=1,   dissipation=false)
    r2 = circuit_trajectory(N, th, 0.0, k; excited=EXC, maxdim=256, seed=999, dissipation=false)
    zt = real(r1[end].z[1]); dseed = abs(zt - real(r2[end].z[1]))
    @printf("  t=%5.1f: MPDO <Z_1>=%+.9f  traj=%+.9f  diff=%.2e  seed-indep=%.2e  %s\n",
            t, zm, zt, abs(zt-zm), dseed,
            (abs(zt-zm) < 1e-8 && dseed < 1e-12) ? "PASS" : "CHECK")
end

println()
println("="^78)
println("If A overlays the Qiskit curves and B, C pass, the simulator is running")
println("your circuit and we can move to Step 2 (the theta sweep).")
println("="^78)
