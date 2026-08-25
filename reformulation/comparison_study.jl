# =============================================================================
# comparison_study.jl   --   M-route vs N-route, scored against EXACT
#
# At n=4, maxdim = 256 = 16^min(2,2) is exact (no truncation), so every curve is
# scored against ground truth rather than a high-maxdim proxy. This is the
# upgrade over deck slides 23/26: those asked "has it stopped moving?", these
# ask "how far from the true value?"
#
# TWO FIXES relative to the original version of this script, both of which
# depressed every N-route number:
#   (a) maxdim_G = maxdim  (was maxdim/2, so G was truncated at 128 while the
#       run was reported as untruncated at 256: dc 9.4e-4 -> 1.0e-4)
#   (b) exact_delta = true (alg="directsum"; the default density-matrix `+` is
#       sqrt(eps)-accurate and was discarding 65% of delta's weight)
#
# OUTPUTS (CSV, so plotting can be re-run without repeating the sweep):
#   cmp_coefficients.csv     c1, c2, dc_max vs maxdim, both routes
#   cmp_errors.csv           E_mpf, E_k3, E_k8, lambda_min, resolved flag
#
# Keep the SIGN of every error: the M-route returns NEGATIVE squared norms, and
# that is the headline. Never plot |E| without a sign flag.
#
# n=4 only. At n=5 the ceiling is also 256 but M(256) costs 426-988 s per call.
#
# Environment: N_QUBITS, GAMMA, TVAL, K0, K_REF, ORDER, TAG
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")          # -> F_diagnostics -> ... -> liouville_space
include("open_optimization_problem.jl")   # M-route; expect constant-redefinition
                                          # warnings (both chain to F_diagnostics).
                                          # Harmless: identical values.
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n     = parse(Int,     getenv("N_QUBITS", 4))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
k_ref = parse(Int,     getenv("K_REF",    48))   # MATCH k0: otherwise the two
                                                  # routes target different
                                                  # references and the comparison
                                                  # mixes truncation with a
                                                  # systematic offset
order = parse(Int,     getenv("ORDER",    2))
tag   = getenv("TAG", "")
ks    = [3, 8]
ct    = 1e-14

chi  = theoretical_max_bond_dim(n)
grid = filter(<=(chi), [16, 32, 48, 64, 96, 128, 192, 256])
sfx  = isempty(tag) ? "" : "_" * tag

@printf("n=%d gamma=%.3f t=%.1f ks=%s k0=%d k_ref=%d order=%d\n", n, gamma, t,
        string(ks), k0, k_ref, order)
@printf("exact point: maxdim = %d (16^min(%d,%d), no truncation)\n\n", chi, n ÷ 2, n - n ÷ 2)
flush(stdout)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

build_N(M, L, P) = [M[i,j] - L[i] - L[j] + P for i in 1:length(L), j in 1:length(L)]

function m_route(md)
    M, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=ct, maxdim=md, order=order, dissipation=true)
    L, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                         cutoff=ct, maxdim=md, order=order, order_ref=order, dissipation=true)
    P = reference_purity(n, J, gammas, t, k_ref, lsites, rho0;
                         cutoff=ct, maxdim=md, order=order, dissipation=true)
    c, _ = dynamic_mpf_coefficients(M, L)
    N = build_N(M, L, P)
    (c=c, E_mpf=open_dynamic_mpf_error(M, L, c, P), N=N,
     lam=minimum(eigen(Symmetric(0.5 * (N + N'))).values))
end

# ---- ground truth ----------------------------------------------------------
println("="^72); println("GROUND TRUTH at maxdim = $chi"); println("="^72)
ex = m_route(chi)
c_exact, Ek_exact = ex.c, [ex.N[j,j] for j in 1:length(ks)]
@printf("  c      = [%+.6f, %+.6f]\n", c_exact[1], c_exact[2])
@printf("  E_k3   = %.6e   E_k8 = %.6e\n", Ek_exact[1], Ek_exact[2])
@printf("  E_mpf  = %.6e   lambda_min = %.4e\n\n", ex.E_mpf, ex.lam)
println("CAVEAT: k_ref=$k_ref is exact in the sense of UNTRUNCATED, not physically")
println("converged -- k0 48->96 shifts E_k8 by +6.3%. Fine for an M-vs-N comparison")
println("(both use it), but E_k8 here is not the physical value.\n")
flush(stdout)

# ---- sweep -----------------------------------------------------------------
rc = ["maxdim,route,c1,c2,dc_max"]
re = ["maxdim,route,E_mpf,E_k3,E_k8,lambda_min,resolved,time_s"]

println("maxdim | M: c1        dc        E_k8       | N: c1        dc        E_k8       | t_M   t_N")
println("-"^104)
for md in grid
    t0 = time(); m = m_route(md); tM = time() - t0
    t0 = time()
    r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                         cutoff=ct, maxdim=md, maxdim_G=md,      # (a)
                         order=order, order_ref=order, dissipation=true,
                         exact_delta=true)                        # (b)
    tN = time() - t0

    push!(rc, @sprintf("%d,M,%.8f,%.8f,%.8e", md, m.c[1], m.c[2], maximum(abs.(m.c .- c_exact))))
    push!(rc, @sprintf("%d,N,%.8f,%.8f,%.8e", md, r.coeffs[1], r.coeffs[2],
                       maximum(abs.(r.coeffs .- c_exact))))
    push!(re, @sprintf("%d,M,%.8e,%.8e,%.8e,%.8e,%s,%.1f", md, m.E_mpf, m.N[1,1], m.N[2,2],
                       m.lam, m.lam > 0 ? "yes" : "no", tM))
    push!(re, @sprintf("%d,N,%.8e,%.8e,%.8e,%.8e,%s,%.1f", md, r.E_mpf, r.E_trot[1], r.E_trot[2],
                       minimum(r.eigvals), minimum(r.eigvals) > 0 ? "yes" : "no", tN))

    @printf("%6d | %+.5f %.2e %+.3e | %+.5f %.2e %+.3e | %5.0f %5.0f\n",
            md, m.c[1], maximum(abs.(m.c .- c_exact)), m.N[2,2],
            r.coeffs[1], maximum(abs.(r.coeffs .- c_exact)), r.E_trot[2], tM, tN)
    flush(stdout)
end

write("cmp_coefficients$sfx.csv", join(rc, "\n") * "\n")
write("cmp_errors$sfx.csv",       join(re, "\n") * "\n")
println("\nwrote cmp_coefficients$sfx.csv, cmp_errors$sfx.csv")
