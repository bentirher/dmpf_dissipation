# =============================================================================
# comparison_study.jl
#
# Head-to-head of the M-route (M, L, P) against the N-route (Trotter-error
# Gram matrix) at n=4, where maxdim = 256 = 16^min(2,2) is EXACT -- so every
# curve is scored against ground truth, not against a high-maxdim proxy.
#
# This is the upgrade over the original deck figures (slides 23, 26): those
# asked "has it stopped moving?", these ask "how far from the true value?"
#
# OUTPUTS (CSV, plotted separately so the plotting can be re-run cheaply):
#   cmp_coefficients.csv   c1, c2 vs maxdim, both routes, + exact
#   cmp_errors.csv         E_mpf, E_k3, E_k8 vs maxdim, both routes, + exact
#                          (signs kept -- the M-route goes NEGATIVE, that is
#                           the headline, so never plot |E| without a sign flag)
#   cmp_spectra.csv        operator-Schmidt spectra of F_88 and Phi_88
#   cmp_weight_vs_error.csv discarded Schmidt weight vs resulting error in c
#
# THE KEY PLOT is the last one. Here is why the naive spectrum comparison is
# misleading and this one is not:
#
#   For F, s_1 IS the G = E^dag E piece, which cancels identically out of the
#   coefficients. So log10(s_i/s_1) makes F's tail look negligible when the
#   tail is in fact the entire signal. For Phi there is no such passenger --
#   every singular value contributes to the answer. Comparing the two
#   normalized spectra side by side is therefore apples-to-oranges and can be
#   read as Phi looking WORSE.
#
#   Discarded-weight-vs-error is the apples-to-apples version: for each object,
#   how much Schmidt weight did truncation throw away, and what did that cost
#   in the final answer? That ratio IS the amplification factor, measured.
#
# Environment: N_QUBITS (default 4), GAMMA, TVAL, K0, K_REF, ORDER
# =============================================================================

import Distributions
import Random
using LinearAlgebra
using Printf
include("trotter_error_gram.jl")          # pulls the whole chain
include("open_optimization_problem.jl")   # M-route: expect redefinition warnings,
                                          # harmless -- both chain to F_diagnostics
include("spectrum_truncation_analysis.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n      = parse(Int,     getenv("N_QUBITS", 4))
gamma  = parse(Float64, getenv("GAMMA",    0.05))
t      = parse(Float64, getenv("TVAL",     3.0))
k0     = parse(Int,     getenv("K0",       48))
k_ref  = parse(Int,     getenv("K_REF",    48))   # MATCHED to k0 on purpose:
                                                   # otherwise the two routes
                                                   # target different references
                                                   # and the comparison mixes
                                                   # truncation with a systematic
                                                   # offset (~2.5% at k0=48 vs
                                                   # k_oracle=200)
order  = parse(Int,     getenv("ORDER",    2))
ks     = [3, 8]
cutoff = 1e-12

chi_max = theoretical_max_bond_dim(n)
grid    = filter(<=(chi_max), [16, 32, 48, 64, 96, 128, 192, 256])

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("n=%d  gamma=%.3f  t=%.1f  ks=%s  k0=%d  k_ref=%d  order=%d\n",
        n, gamma, t, string(ks), k0, k_ref, order)
@printf("exact point: maxdim = %d (= 16^min(%d,%d), no truncation)\n\n",
        chi_max, n ÷ 2, n - n ÷ 2)
flush(stdout)

# =============================================================================
# Ground truth
# =============================================================================
println("="^70); println("GROUND TRUTH at maxdim = $chi_max"); println("="^70)

M_ex, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                           cutoff=cutoff, maxdim=chi_max, order=order, dissipation=true)
L_ex, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                        cutoff=cutoff, maxdim=chi_max, order=order,
                        order_ref=order, dissipation=true)
P_ex = reference_purity(n, J, gammas, t, k_ref, lsites, rho0;
                        cutoff=cutoff, maxdim=chi_max, order=order, dissipation=true)

N_exact = [M_ex[i,j] - L_ex[i] - L_ex[j] + P_ex for i in 1:length(ks), j in 1:length(ks)]
sol_ex  = trotter_error_coefficients(N_exact)
c_exact, E_mpf_exact = sol_ex.coeffs, sol_ex.E_mpf
Ek_exact = [N_exact[j,j] for j in 1:length(ks)]

@printf("  c        = [%.6f, %.6f]\n", c_exact[1], c_exact[2])
@printf("  E_k3     = %.6e\n  E_k8     = %.6e\n", Ek_exact[1], Ek_exact[2])
@printf("  E_mpf    = %.6e\n", E_mpf_exact)
@printf("  eig(N)   = [%.4e, %.4e]   cond = %.3g\n\n",
        sol_ex.eigvals[1], sol_ex.eigvals[2], sol_ex.cond)
flush(stdout)

# =============================================================================
# Sweep both routes
# =============================================================================
rows_c, rows_e = String[], String[]
push!(rows_c, "maxdim,route,c1,c2,dc_max")
push!(rows_e, "maxdim,route,E_mpf,E_k3,E_k8,lambda_min,resolved")

# `resolved` flags whether E_mpf is meaningful. N is near rank-1 (all the
# Trotter errors are nearly parallel), so lambda_min is tiny and E_mpf sits at
# the noise floor long after c has converged: c needs only the EIGENVECTOR
# direction, E_mpf needs the EIGENVALUE magnitude. Reporting E_mpf without this
# flag would be misleading.
resolved(lmin) = lmin > 0 ? "yes" : "no"

for md in grid
    # ---- M-route ----
    Mm, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                             cutoff=cutoff, maxdim=md, order=order, dissipation=true)
    Lm, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                          cutoff=cutoff, maxdim=md, order=order,
                          order_ref=order, dissipation=true)
    Pm = reference_purity(n, J, gammas, t, k_ref, lsites, rho0;
                          cutoff=cutoff, maxdim=md, order=order, dissipation=true)
    cM, _  = dynamic_mpf_coefficients(Mm, Lm)
    EmM    = open_dynamic_mpf_error(Mm, Lm, cM, Pm)
    NM     = [Mm[i,j] - Lm[i] - Lm[j] + Pm for i in 1:length(ks), j in 1:length(ks)]
    lminM  = minimum(eigen(Symmetric(0.5*(NM+NM'))).values)

    push!(rows_c, @sprintf("%d,M,%.8f,%.8f,%.8e", md, cM[1], cM[2],
                           maximum(abs.(cM .- c_exact))))
    push!(rows_e, @sprintf("%d,M,%.8e,%.8e,%.8e,%.8e,%s", md, EmM,
                           NM[1,1], NM[2,2], lminM, resolved(lminM)))

    # ---- N-route ----
    r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                         cutoff=cutoff, maxdim=md, maxdim_G=max(md ÷ 2, 16),
                         order=order, order_ref=order, dissipation=true)
    lminN = minimum(r.eigvals)

    push!(rows_c, @sprintf("%d,N,%.8f,%.8f,%.8e", md, r.coeffs[1], r.coeffs[2],
                           maximum(abs.(r.coeffs .- c_exact))))
    push!(rows_e, @sprintf("%d,N,%.8e,%.8e,%.8e,%.8e,%s", md, r.E_mpf,
                           r.E_trot[1], r.E_trot[2], lminN, resolved(lminN)))

    @printf("maxdim %3d | M: c=[%+.5f,%+.5f] dc=%.2e Ek8=%+.3e | N: c=[%+.5f,%+.5f] dc=%.2e Ek8=%+.3e\n",
            md, cM[1], cM[2], maximum(abs.(cM .- c_exact)), NM[2,2],
            r.coeffs[1], r.coeffs[2], maximum(abs.(r.coeffs .- c_exact)), r.E_trot[2])
    flush(stdout)
end

write("cmp_coefficients.csv", join(rows_c, "\n") * "\n")
write("cmp_errors.csv",       join(rows_e, "\n") * "\n")

# =============================================================================
# Spectra of F_88 and Phi_88, both built UNTRUNCATED
# =============================================================================
println("\n", "="^70); println("SPECTRA (untruncated, k=8 diagonal)"); println("="^70)

F88   = build_open_F(n, J, gammas, t, 8, lsites, cutoff, chi_max;
                     order=order, dissipation=true)
pair  = build_Phi_pair(n, J, gammas, t, 8, 8, k0, lsites;
                       cutoff=cutoff, maxdim=chi_max, maxdim_G=chi_max,
                       order=order, order_ref=order, dissipation=true)
sF, sP = operator_schmidt_spectrum(F88), operator_schmidt_spectrum(pair.Phi)

rows_s = ["index,object,sigma,sigma_over_s1,cum_weight_frac"]
for (name, s) in (("F", sF), ("Phi", sP))
    w = s .^ 2; cw = cumsum(w) ./ sum(w)
    for i in eachindex(s)
        push!(rows_s, @sprintf("%d,%s,%.10e,%.10e,%.12f", i, name, s[i], s[i]/s[1], cw[i]))
    end
end
write("cmp_spectra.csv", join(rows_s, "\n") * "\n")

for (name, s) in (("F", sF), ("Phi", sP))
    @printf("%-4s chi=%3d  s1=%.4e  eff_rank(1-1e-3)=%s  eff_rank(1-1e-6)=%s\n",
            name, length(s), s[1],
            string(effective_rank(s; weight_fraction=1-1e-3)),
            string(effective_rank(s; weight_fraction=1-1e-6)))
end

# =============================================================================
# THE KEY PLOT: discarded Schmidt weight vs resulting error in c
# =============================================================================
# For each maxdim, the fraction of Frobenius weight truncation throws away from
# the object, paired with the error that produced in the final coefficients.
# The slope IS the amplification factor. Expect the M-route to sit far to the
# upper-left (tiny weight discarded, huge error) and the N-route far to the
# lower-right (large weight discarded, small error).

println("\n", "="^70); println("DISCARDED WEIGHT vs ERROR IN c"); println("="^70)
println("maxdim | F: disc.weight  dc(M-route) | Phi: disc.weight  dc(N-route)")

discarded(s, md) = md >= length(s) ? 0.0 : 1 - sum(s[1:md].^2) / sum(s.^2)

cvals = Dict{Tuple{Int,String},Float64}()
for line in rows_c[2:end]
    f = split(line, ",")
    cvals[(parse(Int, f[1]), String(f[2]))] = parse(Float64, f[5])
end

rows_w = ["maxdim,object,discarded_weight,dc_max,route"]
for md in grid
    wF, wP = discarded(sF, md), discarded(sP, md)
    dcM, dcN = cvals[(md, "M")], cvals[(md, "N")]
    push!(rows_w, @sprintf("%d,F,%.8e,%.8e,M",   md, wF, dcM))
    push!(rows_w, @sprintf("%d,Phi,%.8e,%.8e,N", md, wP, dcN))
    @printf("%6d | %13.3e  %11.3e | %14.3e  %11.3e\n", md, wF, dcM, wP, dcN)
end
write("cmp_weight_vs_error.csv", join(rows_w, "\n") * "\n")

println("\nwrote cmp_coefficients.csv, cmp_errors.csv, cmp_spectra.csv, cmp_weight_vs_error.csv")
