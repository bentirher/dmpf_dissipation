# =============================================================================
# identity_localise.jl   --   follow-up to T4
#
# T4 reported max residual 1.02e-2 (8,8) and 8.88e-2 (3,8), where ~1e-14 was
# expected. `max == final` in both cases, so it grows monotonically. But the
# summary cannot distinguish two very different causes:
#
#   (A) THE CHECK IS BROKEN. Ssum = G + Xi + Psi + Phi is an MPO sum whose
#       norms span 4 orders (||G|| ~ 8, ||Phi|| ~ 1e-3). ITensor compresses the
#       sum relative to ITS OWN weight, i.e. relative to ||G||, so Phi and Xi
#       can be destroyed INSIDE Ssum while being perfectly fine as standalone
#       objects. This would mean the recursion is correct and only the
#       diagnostic is wrong -- consistent with dc = 1e-4 being good.
#
#   (B) THE RECURSION DRIFTS. One of the four objects accumulates error that
#       <<rho0|Phi|rho0>> happens not to see.
#
# These have opposite implications, so this script measures them apart:
#
#   P1  per-step residual, printed for EVERY step, with the norm of each term.
#       Step 1 is analytically exact (Ssum = B_j + delta_j = A_j), so a residual
#       there is proof of (A). A residual that starts at 1e-15 and grows is (B).
#
#   P2  the same residual computed WITHOUT any MPO addition: contract each of
#       the five objects against <<rho0| . |rho0>> first and compare SCALARS.
#       Scalars cannot suffer compression loss, so agreement here with
#       disagreement in P1 pins the blame on (A) definitively.
#
#   P3  a pure MPO-addition stress test: build X = G + Phi explicitly and check
#       whether <<rho0|X|rho0>> equals <<rho0|G|rho0>> + <<rho0|Phi|rho0>>.
#       This isolates the summation step with no recursion involved at all.
#
# Environment: N_QUBITS (3), GAMMA, TVAL, K0, ORDER, KI, KJ
# =============================================================================

import Distributions
import Random
using LinearAlgebra
using Printf
include("trotter_error_gram.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n     = parse(Int,     getenv("N_QUBITS", 3))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
order = parse(Int,     getenv("ORDER",    2))
ki    = parse(Int,     getenv("KI",       8))
kj    = parse(Int,     getenv("KJ",       8))
ct    = 1e-14

chi = theoretical_max_bond_dim(n)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("n=%d chi_ceiling=%d (untruncated)  k_i=%d k_j=%d k0=%d order=%d\n\n",
        n, chi, ki, kj, k0, order)

# =============================================================================
# P3 first -- it is the cheapest and most decisive if it fails
# =============================================================================
println("="^78)
println("P3  MPO-addition stress test (no recursion involved)")
println("="^78)

Sfine = get_open_step_MPO(n, J, gammas, t / k0, lsites, ct, chi;
                          order=order, dissipation=true)
Gtest = identity_liouville_mpo(lsites)
for _ in 1:6
    global Gtest = left_multiply(Sfine, Gtest; cutoff=ct, maxdim=chi)
end
d = step_defect_MPO(n, J, gammas, t / kj, k0 ÷ kj, lsites, ct, chi;
                    order=order, order_ref=order, dissipation=true)
Small = d.delta

for scale in (1.0, 1e-2, 1e-4, 1e-6)
    Sc  = scale * Small
    Sum = +(Gtest, Sc; cutoff=ct, maxdim=chi)
    lhs = real(expect_F(Sum, rho0))
    rhs = real(expect_F(Gtest, rho0)) + real(expect_F(Sc, rho0))
    @printf("  ||small||/||G|| = %.1e :  <<Sum>> = %+.12e   <<G>>+<<small>> = %+.12e   rel diff = %.3e\n",
            _fnorm(Sc) / _fnorm(Gtest), lhs, rhs, abs(lhs - rhs) / max(abs(rhs), eps()))
end
println("""
  -> If the rel diff grows as the small term shrinks, MPO addition is losing
     the small term relative to the large one, and the T4 residual is an
     artifact of the CHECK (cause A), not of the recursion.""")
flush(stdout)

# =============================================================================
# P1 / P2 -- per-step residual, operator form and scalar form
# =============================================================================
println("\n", "="^78)
println("P1/P2  per-step identity residual")
println("="^78)

dt_i, dt_j = t / ki, t / kj
Li = step_defect_MPO(n, J, gammas, dt_i, k0 ÷ ki, lsites, ct, chi;
                     order=order, order_ref=order, dissipation=true)
Rj = step_defect_MPO(n, J, gammas, dt_j, k0 ÷ kj, lsites, ct, chi;
                     order=order, order_ref=order, dissipation=true)

Ai_dag, Bi_dag, di_dag = op_dag(Li.A), op_dag(Li.B), op_dag(Li.delta)

G::MaybeMPO   = identity_liouville_mpo(lsites)
Xi::MaybeMPO  = nothing
Psi::MaybeMPO = nothing
Phi::MaybeMPO = nothing
F::MaybeMPO   = identity_liouville_mpo(lsites)

ti = tj = 0.0
step = 0

println("step side |   ||G||     ||Xi||    ||Psi||   ||Phi||   ||F||   | P1 op-resid | P2 scalar-resid")
println("-"^100)

while (ti < t - 1e-12) || (tj < t - 1e-12)
    global G, Xi, Psi, Phi, F, ti, tj, step
    step += 1
    local side
    if (tj <= ti) && (tj < t - 1e-12)
        Phi_n = _add(_rmul(Phi, Rj.A; cutoff=ct, maxdim=chi),
                     _rmul(Xi,  Rj.delta; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
        Psi_n = _add(_rmul(Psi, Rj.A; cutoff=ct, maxdim=chi),
                     _rmul(G,   Rj.delta; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
        Xi_n  = _rmul(Xi, Rj.B; cutoff=ct, maxdim=chi)
        G_n   = _rmul(G,  Rj.B; cutoff=ct, maxdim=chi)
        G, Xi, Psi, Phi = G_n, Xi_n, Psi_n, Phi_n
        F  = _rmul(F, Rj.A; cutoff=ct, maxdim=chi)
        tj += dt_j; side = "R"
    else
        Phi_n = _add(_lmul(Ai_dag, Phi; cutoff=ct, maxdim=chi),
                     _lmul(di_dag, Psi; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
        Xi_n  = _add(_lmul(Ai_dag, Xi;  cutoff=ct, maxdim=chi),
                     _lmul(di_dag, G;   cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
        Psi_n = _lmul(Bi_dag, Psi; cutoff=ct, maxdim=chi)
        G_n   = _lmul(Bi_dag, G;   cutoff=ct, maxdim=chi)
        G, Xi, Psi, Phi = G_n, Xi_n, Psi_n, Phi_n
        F  = _lmul(Ai_dag, F; cutoff=ct, maxdim=chi)
        ti += dt_i; side = "L"
    end

    # P1: operator-level residual (goes through MPO addition -- can lose small terms)
    Ssum  = _add(_add(G, Xi; cutoff=ct, maxdim=chi),
                 _add(Psi, Phi; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
    D     = _add(Ssum, -1 * F; cutoff=ct, maxdim=chi)
    r_op  = _fnorm(D) / max(_fnorm(F), eps())

    # P2: scalar-level residual (no MPO addition -- immune to compression loss)
    sG, sX = real(_expect(G, rho0)),   real(_expect(Xi, rho0))
    sP, sF = real(_expect(Psi, rho0)), real(_expect(Phi, rho0))
    sFF    = real(_expect(F, rho0))
    r_sc   = abs((sG + sX + sP + sF) - sFF) / max(abs(sFF), eps())

    @printf("%4d  %s   | %.3e %.3e %.3e %.3e %.3e | %.4e | %.4e\n",
            step, side, _fnorm(G), _fnorm(Xi), _fnorm(Psi), _fnorm(Phi), _fnorm(F),
            r_op, r_sc)
    flush(stdout)
end

println("""

READING THIS:
  step 1 is ANALYTICALLY exact: Ssum = B_j + delta_j = A_j = F. So
    - residual ~1e-15 at step 1, growing later      -> cause (B), real drift
    - residual already large at step 1              -> cause (A), broken check
    - P2 (scalar) ~1e-15 while P1 (operator) large  -> cause (A), CONFIRMED:
      the recursion is correct and MPO addition is losing Phi against ||G||
""")
