# =============================================================================
# systematic_diagnostic.jl   --   PRIORITY 1
#
# At n=4, maxdim=256 = 16^min(2,2), NOTHING is truncated by maxdim. Both routes
# now target the same reference (k_ref = k0 = 48). The M-route reproduces the
# exact point to 2.2e-10. The N-route is off by dc = 9.36e-4, with E_k3 low by
# 0.25% and E_k8 low by 0.6%.
#
# A NON-UNIFORM error is the signature that matters: a uniform rescaling of N
# leaves c invariant, so the fact that E_k3 and E_k8 are off by DIFFERENT
# amounts is what moves the coefficients.
#
# Six tests, each isolating one hypothesis. Run in order; the first that fails
# is the answer.
#
#   T1  op_dag(S) vs get_open_step_MPO_dag(S)
#       -> is the adjoint we use the same as the one the codebase builds?
#   T2  reference consistency: S(tau)^48 built three ways
#       -> does PRE-COMPOSING B_j = S(tau)^{q_j} into one MPO lose accuracy
#          relative to applying S(tau) one step at a time? PRIME SUSPECT: the
#          M-route applies S(tau)^dag 48 times singly, while the N-route
#          composes B_3 = S(tau)^16 (a propagator over duration 1.0!) and then
#          applies it 3 times.
#   T3  defect magnitudes ||delta_j|| / ||A_j||
#       -> quantifies limitation L5 and the conditioning of A_j - B_j
#   T4  recursion identity  G + Xi + Psi + Phi == F  at every step
#       -> distinguishes "accumulated round-off" from "algebra bug"
#   T5  cutoff sweep at maxdim=256
#       -> is 1e-12 accumulation enough to explain 9.4e-4?
#   T6  k0 sweep at maxdim=256
#       -> is the reference itself the limit?
#
# Environment: N_QUBITS (4), GAMMA, TVAL, K0, ORDER, NSMALL (3, for T4)
# =============================================================================

import Distributions
import Random
using LinearAlgebra
using Printf
include("trotter_error_gram.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n       = parse(Int,     getenv("N_QUBITS", 4))
gamma   = parse(Float64, getenv("GAMMA",    0.05))
t       = parse(Float64, getenv("TVAL",     3.0))
k0      = parse(Int,     getenv("K0",       48))
order   = parse(Int,     getenv("ORDER",    2))
n_small = parse(Int,     getenv("NSMALL",   3))
ks      = [3, 8]

chi_max = theoretical_max_bond_dim(n)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

relerr(A::MPO, B::MPO; ct=1e-14) =
    sqrt(abs(real(inner(+(A, -1 * B; cutoff=ct, maxdim=chi_max),
                        +(A, -1 * B; cutoff=ct, maxdim=chi_max))))) /
    sqrt(abs(real(inner(B, B))))

@printf("n=%d gamma=%.3f t=%.1f ks=%s k0=%d order=%d chi_max=%d\n\n",
        n, gamma, t, string(ks), k0, order, chi_max)
flush(stdout)

# =============================================================================
# T1 -- is op_dag the same adjoint the codebase builds from gates?
# =============================================================================
println("="^72); println("T1  op_dag(S) vs get_open_step_MPO_dag(S)"); println("="^72)
for kj in ks
    dt = t / kj
    S     = get_open_step_MPO(n, J, gammas, dt, lsites, 1e-14, chi_max;
                              order=order, dissipation=true)
    Sdag  = get_open_step_MPO_dag(n, J, gammas, dt, lsites, 1e-14, chi_max;
                                  order=order, dissipation=true)
    @printf("  k=%d dt=%.4f : ||op_dag(S) - S_dag|| / ||S_dag|| = %.4e\n",
            kj, dt, relerr(op_dag(S), Sdag))
end
println("  -> should be ~1e-15. Anything larger means the recursion's adjoint")
println("     differs from the M-route's, which alone could explain the offset.")
flush(stdout)

# =============================================================================
# T2 -- the reference, built three ways   (PRIME SUSPECT)
# =============================================================================
println("\n", "="^72)
println("T2  reference S(tau)^$k0 built three ways"); println("="^72)

tau = t / k0
fine = get_open_step_MPO(n, J, gammas, tau, lsites, 1e-14, chi_max;
                         order=order, dissipation=true)

# (a) one fine step at a time, k0 times -- the M-route's path
E_single = identity_liouville_mpo(lsites)
for _ in 1:k0
    global E_single = left_multiply(fine, E_single; cutoff=1e-14, maxdim=chi_max)
end

# (b) and (c) pre-composed blocks B_j = S(tau)^{q_j}, applied k_j times
#     -- the N-route's path, for each k_j
E_blocked = Dict{Int,MPO}()
for kj in ks
    q  = k0 ÷ kj
    Bj = reference_block_MPO(n, J, gammas, t / kj, q, lsites, 1e-14, chi_max;
                             order=order, dissipation=true)
    E  = identity_liouville_mpo(lsites)
    for _ in 1:kj
        E = left_multiply(Bj, E; cutoff=1e-14, maxdim=chi_max)
    end
    E_blocked[kj] = E
    @printf("  k=%d: B_%d = S(tau)^%-3d (duration %.3f), chi(B) = %d\n",
            kj, kj, q, t / kj, middle_bond_dim(Bj))
end

println()
for kj in ks
    @printf("  ||E_blocked[k=%d] - E_single|| / ||E_single|| = %.4e\n",
            kj, relerr(E_blocked[kj], E_single))
end
@printf("  ||E_blocked[3] - E_blocked[8]|| / ||E||          = %.4e\n",
        relerr(E_blocked[ks[1]], E_blocked[ks[2]]))
println("""
  -> These are the SAME operator mathematically (q_j * k_j = k0 in every case).
     A difference here at maxdim=$chi_max is pure composition round-off, and if
     it is ~1e-3 it explains the whole systematic: the two k_j branches then
     carry DIFFERENT references, which is exactly the non-uniform error that
     moves c.""")
flush(stdout)

# =============================================================================
# T3 -- defect magnitudes (limitation L5)
# =============================================================================
println("\n", "="^72); println("T3  defect magnitude ||delta_j|| / ||A_j||"); println("="^72)
for kj in ks
    q = k0 ÷ kj
    d = step_defect_MPO(n, J, gammas, t / kj, q, lsites, 1e-14, chi_max;
                        order=order, order_ref=order, dissipation=true)
    nA, nB, nd = _fnorm(d.A), _fnorm(d.B), _fnorm(d.delta)
    @printf("  k=%2d dt=%.4f : ||A||=%.4e ||B||=%.4e ||delta||=%.4e  ratio=%.4e  chi(delta)=%d\n",
            kj, t / kj, nA, nB, nd, nd / nA, middle_bond_dim(d.delta))
end
println("""
  -> ratio ~1e-2 or below is healthy. For k=3 the block spans t/3 = 1.0, so a
     LARGE ratio here confirms L5: "single-step defect" is a poor description
     and the norm hierarchy is weak on that branch. Digits lost in forming
     A - B is roughly -log10(ratio).""")
flush(stdout)

# =============================================================================
# T4 -- recursion identity, per step
# =============================================================================
println("\n", "="^72)
println("T4  identity  G + Xi + Psi + Phi == F   (n=$n_small, exact)"); println("="^72)

Random.seed!(1234)
Js  = rand(Distributions.Uniform(1/4, 3/4), n_small - 1)
gs  = fill(gamma, n_small)
ls  = liouville_siteinds(n_small)
cs  = theoretical_max_bond_dim(n_small)
@printf("  n=%d ceiling=%d (untruncated)\n", n_small, cs)

for (ki, kj) in ((8, 8), (3, 8))
    r = build_Phi_pair(n_small, Js, gs, t, ki, kj, k0, ls;
                       cutoff=1e-14, maxdim=cs, maxdim_G=cs,
                       order=order, order_ref=order, dissipation=true,
                       identity_check=true)
    @printf("  (k_i=%d,k_j=%d): max residual = %.4e   final = %.4e\n",
            ki, kj, maximum(r.identity_residual), r.identity_residual[end])
end
println("""
  -> ~1e-14 means the recursion algebra is CORRECT and the systematic is
     round-off/composition, not a sign or ordering error. A residual that grows
     step by step localises the bug to the step where it first jumps.""")
flush(stdout)

# =============================================================================
# T5 / T6 -- cutoff and k0 sweeps at maxdim = chi_max (no maxdim truncation)
# =============================================================================
println("\n", "="^72)
println("T5/T6  cutoff and k0 sweeps at maxdim=$chi_max"); println("="^72)

# Ground truth from the completed sweep (n=4, gamma=0.05, t=3, k_ref=48,
# order=2, maxdim=256 = exact). Hard-coded rather than recomputed so this script
# does not have to include open_optimization_problem.jl, which would pull the
# include chain a second time and trigger constant-redefinition warnings.
c_exact  = [-0.144708, 1.144708]
Ek_exact = [1.334740e-02, 2.502581e-04]

println("cutoff  |  k0 |   c1        c2       | dc_max    | E_k3 err  | E_k8 err")
println("-"^76)
for ct in (1e-12, 1e-14, 1e-15), kk in (k0,)
    r = test_dmpf_open_N(n, J, gammas, t, ks, kk, lsites, rho0;
                         cutoff=ct, maxdim=chi_max, maxdim_G=chi_max,
                         order=order, order_ref=order, dissipation=true)
    @printf("%.0e | %3d | %+.6f %+.6f | %.3e | %+7.3f%% | %+7.3f%%\n",
            ct, kk, r.coeffs[1], r.coeffs[2],
            maximum(abs.(r.coeffs .- c_exact)),
            100 * (r.E_trot[1] - Ek_exact[1]) / Ek_exact[1],
            100 * (r.E_trot[2] - Ek_exact[2]) / Ek_exact[2])
    flush(stdout)
end

for kk in (96, 144), ct in (1e-14,)
    r = test_dmpf_open_N(n, J, gammas, t, ks, kk, lsites, rho0;
                         cutoff=ct, maxdim=chi_max, maxdim_G=chi_max,
                         order=order, order_ref=order, dissipation=true)
    @printf("%.0e | %3d | %+.6f %+.6f | %.3e | %+7.3f%% | %+7.3f%%\n",
            ct, kk, r.coeffs[1], r.coeffs[2],
            maximum(abs.(r.coeffs .- c_exact)),
            100 * (r.E_trot[1] - Ek_exact[1]) / Ek_exact[1],
            100 * (r.E_trot[2] - Ek_exact[2]) / Ek_exact[2])
    flush(stdout)
end

println("""
  NOTE: raising k0 changes the reference, so E_k3/E_k8 SHOULD drift a little --
  they measure distance from a (slightly) different rho(t). What must not drift
  is dc_max, which compares against the k_ref=48 exact point. If dc_max falls
  with tighter cutoff -> T5 is the cause. If it falls with larger k0 -> the
  blocked reference is the cause and T2 will already have shown it.""")
