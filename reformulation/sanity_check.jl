# =============================================================================
# sanity_check.jl   --   run this after ANY edit to trotter_error_gram.jl
#
# Consolidates the five diagnostic scripts written during development into one
# fast regression test at n=3 (ceiling 16, everything untruncated). Each check
# here corresponds to a bug that actually occurred. Minutes, not hours.
#
#   C1  include chain complete
#   C2  B_j + delta_j == A_j          (the recursion's founding identity)
#   C3  op_dag(S) == get_open_step_MPO_dag(S)
#   C4  recursion identity, SCALAR form, per step
#   C5  closed-system limit reproduces the known DMPF coefficients
#   C6  N is symmetric PSD with correctly ordered diagonal
#
# MEASUREMENT RULE, learned the hard way: every difference of near-equal MPOs is
# measured with alg="directsum". ITensorMPS's default `+` is density-matrix
# based (sqrt(eps) ~ 1.5e-8 accurate) and a density-matrix subtraction LOSES the
# difference it is measuring -- it once reported 2.9e-15 for a true 1.5e-8.
# Note also that sqrt(inner(X,X)) cannot resolve below sqrt(eps)*||A||, so
# ~1e-8 is the FLOOR here, not a failure.
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n     = parse(Int,     getenv("N_QUBITS", 3))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
order = parse(Int,     getenv("ORDER",    2))
ks    = [3, 8]
ct    = 1e-14

chi = theoretical_max_bond_dim(n)
Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

# ALWAYS measure differences with directsum.
dsrel(X, Y) = _fnorm(+(X, -1 * Y; alg="directsum")) / max(_fnorm(Y), eps())

fails = String[]
report(name, val, tol) = begin
    ok = val < tol
    @printf("  [%s] %-46s %.3e  (tol %.0e)\n", ok ? "PASS" : "FAIL", name, val, tol)
    ok || push!(fails, name)
end

@printf("sanity_check  n=%d ceiling=%d gamma=%.3f t=%.1f k0=%d order=%d\n\n",
        n, chi, gamma, t, k0, order)

# --- C1 ---------------------------------------------------------------------
println("C1  include chain")
for f in (:liouville_siteinds, :identity_liouville_mpo, :get_open_step_MPO,
          :get_open_step_MPO_dag, :op_dag, :left_multiply, :right_multiply,
          :middle_bond_dim, :vectorized_initial_state_mps, :expect_F, :build_open_F)
    ok = isdefined(Main, f)
    @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", f)
    ok || push!(fails, "include:$f")
end

# --- C2 ---------------------------------------------------------------------
println("\nC2  B_j + delta_j == A_j   (load-bearing: every update rule assumes it)")
for kj in ks
    d = step_defect_MPO(n, J, gammas, t / kj, k0 ÷ kj, lsites, ct, chi;
                        order=order, order_ref=order, dissipation=true,
                        exact_delta=true)
    rec = +(d.B, d.delta; alg="directsum")
    @printf("      k=%d: ||delta||/||A|| = %.3e, chi(delta) = %d\n",
            kj, _fnorm(d.delta) / _fnorm(d.A), maxlinkdim(d.delta))
    report("k=$kj  ||B+delta-A||/||A||", dsrel(rec, d.A), 1e-7)
end

# --- C3 ---------------------------------------------------------------------
println("\nC3  op_dag(S) == get_open_step_MPO_dag(S)")
for kj in ks
    S  = get_open_step_MPO(n, J, gammas, t / kj, lsites, ct, chi; order=order, dissipation=true)
    Sd = get_open_step_MPO_dag(n, J, gammas, t / kj, lsites, ct, chi; order=order, dissipation=true)
    report("k=$kj  adjoint consistency", dsrel(op_dag(S), Sd), 1e-12)
end

# --- C4 ---------------------------------------------------------------------
# G + Xi + Psi + Phi == F, checked on SCALARS. The operator-level version is
# unusable: summing four MPOs whose norms span 4 orders loses Phi inside the
# sum (maxdim < chi(A)+chi(B) discards the small term outright).
println("\nC4  G + Xi + Psi + Phi == F   (scalar form, per step)")
for (ki, kj) in ((8, 8), (3, 8))
    dt_i, dt_j = t / ki, t / kj
    Li = step_defect_MPO(n, J, gammas, dt_i, k0 ÷ ki, lsites, ct, chi;
                         order=order, order_ref=order, dissipation=true, exact_delta=true)
    Rj = step_defect_MPO(n, J, gammas, dt_j, k0 ÷ kj, lsites, ct, chi;
                         order=order, order_ref=order, dissipation=true, exact_delta=true)
    Ai_dag, Bi_dag, di_dag = op_dag(Li.A), op_dag(Li.B), op_dag(Li.delta)

    G::MaybeMPO = identity_liouville_mpo(lsites)
    Xi::MaybeMPO = nothing; Psi::MaybeMPO = nothing; Phi::MaybeMPO = nothing
    F::MaybeMPO = identity_liouville_mpo(lsites)
    ti = tj = 0.0; worst = 0.0

    while (ti < t - 1e-12) || (tj < t - 1e-12)
        if (tj <= ti) && (tj < t - 1e-12)
            Phi_n = _add(_rmul(Phi, Rj.A; cutoff=ct, maxdim=chi),
                         _rmul(Xi,  Rj.delta; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
            Psi_n = _add(_rmul(Psi, Rj.A; cutoff=ct, maxdim=chi),
                         _rmul(G,   Rj.delta; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
            Xi_n = _rmul(Xi, Rj.B; cutoff=ct, maxdim=chi); G_n = _rmul(G, Rj.B; cutoff=ct, maxdim=chi)
            G, Xi, Psi, Phi = G_n, Xi_n, Psi_n, Phi_n
            F = _rmul(F, Rj.A; cutoff=ct, maxdim=chi); tj += dt_j
        else
            Phi_n = _add(_lmul(Ai_dag, Phi; cutoff=ct, maxdim=chi),
                         _lmul(di_dag, Psi; cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
            Xi_n  = _add(_lmul(Ai_dag, Xi;  cutoff=ct, maxdim=chi),
                         _lmul(di_dag, G;   cutoff=ct, maxdim=chi); cutoff=ct, maxdim=chi)
            Psi_n = _lmul(Bi_dag, Psi; cutoff=ct, maxdim=chi); G_n = _lmul(Bi_dag, G; cutoff=ct, maxdim=chi)
            G, Xi, Psi, Phi = G_n, Xi_n, Psi_n, Phi_n
            F = _lmul(Ai_dag, F; cutoff=ct, maxdim=chi); ti += dt_i
        end
        s = real(_expect(G, rho0)) + real(_expect(Xi, rho0)) +
            real(_expect(Psi, rho0)) + real(_expect(Phi, rho0))
        sF = real(_expect(F, rho0))
        worst = max(worst, abs(s - sF) / max(abs(sF), eps()))
    end
    report("(k_i=$ki,k_j=$kj)  worst per-step scalar residual", worst, 1e-6)
end

# --- C5 ---------------------------------------------------------------------
# gamma = 0 must reproduce the closed-system DMPF answer (main.pdf).
println("\nC5  closed-system limit (dissipation=false)")
r0 = test_dmpf_open_N(n, J, zeros(n), t, ks, k0, lsites, rho0;
                      cutoff=ct, maxdim=chi, order=order, order_ref=order,
                      dissipation=false, exact_delta=true)
@printf("      c = [%+.6f, %+.6f]   (main.pdf: [-0.1067, 1.1067])\n", r0.coeffs[1], r0.coeffs[2])
report("closed-limit c1 vs -0.1067", abs(r0.coeffs[1] + 0.1067), 5e-2)

# --- C6 ---------------------------------------------------------------------
println("\nC6  N symmetric PSD, diagonal correctly ordered")
r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                     cutoff=ct, maxdim=chi, order=order, order_ref=order,
                     dissipation=true, exact_delta=true)
@printf("      N        = %s\n", string(round.(r.N; sigdigits=6)))
@printf("      eig(N)   = %s   cond = %.4g\n", string(r.eigvals), r.cond)
@printf("      E_trot   = %s   (ks = %s)\n", string(r.E_trot), string(ks))
@printf("      c        = %s\n", string(round.(r.coeffs; digits=6)))
report("symmetry ||N-N'||", norm(r.N .- r.N'), 1e-12)
report("PSD: -min(0, lambda_min)/||N||", max(0.0, -minimum(r.eigvals)) / norm(r.N), 1e-3)
report("ordering: E_k3 > E_k8 (0 if ok)", r.E_trot[1] > r.E_trot[2] ? 0.0 : 1.0, 0.5)
report("positivity of E_trot (0 if ok)", all(r.E_trot .> 0) ? 0.0 : 1.0, 0.5)

# --- verdict ----------------------------------------------------------------
println("\n" * "="^64)
if isempty(fails)
    println("ALL CHECKS PASSED")
else
    println("FAILURES: ", join(fails, ", "))
    println("\nC2 fail  -> delta construction; check alg=\"directsum\" in step_defect_MPO")
    println("C4 fail  -> recursion algebra; the step where it first jumps localises it")
    println("C6 fail  -> N no longer PSD; maxdim too low, or a sign error")
end
println("="^64)
exit(isempty(fails) ? 0 : 1)
