# =============================================================================
# gauge_diagnostic.jl   --   EXPERIMENT 1 (run this first)
#
# Uses ONLY the existing code base. No new algorithm, no new MPOs. It answers
# one question about results we already have:
#
#   When we truncate, does the resulting error in (M, L, P) land in the GAUGE
#   directions (harmless -- cancels exactly out of the coefficients) or in the
#   INVARIANT directions (lands 1:1 on the answer)?
#
# BACKGROUND (see findings_N_reformulation.md Section 3). On the constraint set
# sum(c) = 1, the perturbation
#
#     M -> M + u 1^T + 1 u^T + a 11^T ,   L -> L + u + a 1 ,   P -> P + a
#
# leaves the cost function EXACTLY unchanged, for any vector u and scalar a.
# There are therefore r+1 gauge directions in the (r(r+1)/2 + r + 1)-dimensional
# space of (M, L, P). What survives is exactly
#
#     N_ij = M_ij - L_i - L_j + P        (the Trotter-error Gram matrix)
#
# The gauge directions carry values ~0.57. N carries 1e-4 to 1e-2. So the
# question above decides whether our current coefficients are accurate by
# design or by luck.
#
# WHAT THIS SCRIPT DOES
#   1. builds M, L, P at maxdim = MD_LO and again at maxdim = MD_HI
#   2. forms dM, dL, dP (the truncation error, MD_LO relative to MD_HI)
#   3. least-squares-fits the gauge part (u, a) of that error
#   4. reports  ||gauge part||  vs  ||residual||  vs  ||dN||  vs  ||N||
#
# READING THE OUTPUT
#   residual << gauge      -> the M-route is accidentally fine at these settings
#   residual ~  gauge      -> the M-route is not resolving N; sections 7/8 of
#                             findings_thematic.md were not anomalies
#   ||dN|| >~ ||N||        -> the coefficients at these settings are noise
#
# PARAMETERS come from the environment so one job array can sweep cases:
#   N_QUBITS, MD_LO, MD_HI, GAMMA, TVAL, K_REF, ORDER
# =============================================================================

import Distributions
import Random
using LinearAlgebra
include("open_optimization_problem.jl")   # existing include chain

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
getenv(k, default) = get(ENV, k, string(default))

n       = parse(Int,     getenv("N_QUBITS", 4))
md_lo   = parse(Int,     getenv("MD_LO",    32))
md_hi   = parse(Int,     getenv("MD_HI",    64))
gamma   = parse(Float64, getenv("GAMMA",    0.05))
t       = parse(Float64, getenv("TVAL",     3.0))
k_ref   = parse(Int,     getenv("K_REF",    40))
order   = parse(Int,     getenv("ORDER",    1))
ks      = [3, 8]
cutoff  = 1e-12

# CRITICAL (findings_thematic.md section 4): maxdim must never exceed the true
# operator-Schmidt ceiling. A loose maxdim forces `apply` to build enormous
# intermediate tensors before truncating, even when the final rank is far below
# it -- this was the ">8 h timed out" vs "~4 minutes" bug at n=5.
theoretical_max_bond_dim(n) = 16^min(n ÷ 2, n - n ÷ 2)
chi_ceiling = theoretical_max_bond_dim(n)
@assert md_hi <= chi_ceiling "MD_HI=$md_hi exceeds the n=$n ceiling $chi_ceiling"
@assert md_lo < md_hi        "MD_LO must be strictly below MD_HI"

Random.seed!(1234)
J       = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas  = fill(gamma, n)
lsites  = liouville_siteinds(n)
rho0    = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

println("="^76)
println("GAUGE DIAGNOSTIC   n=$n  gamma=$gamma  t=$t  ks=$ks  k_ref=$k_ref  order=$order")
println("maxdim: $md_lo vs $md_hi     (ceiling for n=$n is $chi_ceiling)")
println("="^76)
flush(stdout)

# -----------------------------------------------------------------------------
# Build (M, L, P) at both accuracies
# -----------------------------------------------------------------------------
function build_MLP(md)
    println("\n--- building M, L, P at maxdim = $md ---"); flush(stdout)

    t0 = time()
    M, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=cutoff, maxdim=md, order=order, dissipation=true)
    println("  M done in $(round(time()-t0; digits=1)) s"); flush(stdout)

    t0 = time()
    L, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                         cutoff=cutoff, maxdim=md, order=order,
                         order_ref=order, dissipation=true)
    println("  L done in $(round(time()-t0; digits=1)) s"); flush(stdout)

    t0 = time()
    P = reference_purity(n, J, gammas, t, k_ref, lsites, rho0;
                         cutoff=cutoff, maxdim=md, order=order, dissipation=true)
    println("  P done in $(round(time()-t0; digits=1)) s"); flush(stdout)

    return M, L, P
end

M_lo, L_lo, P_lo = build_MLP(md_lo)
M_hi, L_hi, P_hi = build_MLP(md_hi)

# -----------------------------------------------------------------------------
# N and its error
# -----------------------------------------------------------------------------
build_N(M, L, P) = [M[i, j] - L[i] - L[j] + P for i in 1:length(L), j in 1:length(L)]

N_lo = build_N(M_lo, L_lo, P_lo)
N_hi = build_N(M_hi, L_hi, P_hi)

dM = M_lo .- M_hi
dL = L_lo .- L_hi
dP = P_lo -  P_hi
dN = N_lo .- N_hi

# -----------------------------------------------------------------------------
# Least-squares split of (dM, dL, dP) into gauge + residual
# -----------------------------------------------------------------------------
# Unknowns theta = [u; a] in R^{r+1}. The gauge model is
#     dM ~ u 1^T + 1 u^T + a 11^T ,   dL ~ u + a 1 ,   dP ~ a
# Stack all residuals into one vector and solve the (small, dense) normal
# equations. r is 2 or 3, so an explicit design matrix is cheapest and clearest.

function gauge_fit(dM, dL, dP)
    r = length(dL)
    nrows = r * r + r + 1
    ncols = r + 1
    A = zeros(Float64, nrows, ncols)
    b = zeros(Float64, nrows)

    row = 1
    for i in 1:r, j in 1:r                     # dM block
        A[row, i] += 1.0                       # u_i
        A[row, j] += 1.0                       # u_j
        A[row, ncols] = 1.0                    # a
        b[row] = dM[i, j]
        row += 1
    end
    for i in 1:r                               # dL block
        A[row, i] = 1.0
        A[row, ncols] = 1.0
        b[row] = dL[i]
        row += 1
    end
    A[row, ncols] = 1.0                        # dP block
    b[row] = dP

    theta = A \ b
    u = theta[1:r]
    a = theta[ncols]

    gauge_vec = A * theta
    resid_vec = b .- gauge_vec

    return (u=u, a=a,
            gauge_norm = norm(gauge_vec),
            resid_norm = norm(resid_vec),
            total_norm = norm(b))
end

fit = gauge_fit(dM, dL, dP)

# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
println()
println("="^76)
println("RESULTS")
println("="^76)

println("\nM($md_hi) =");        display(M_hi)
println("\nL($md_hi) = ",        L_hi)
println("P($md_hi) = ",          P_hi)

println("\nN($md_hi) = M - L1^T - 1L^T + P 11^T  (the quantity that matters):")
display(N_hi)
println("\n  ||N||_F                = ", norm(N_hi))
println("  diag(N) = E_kj         = ", [N_hi[j, j] for j in 1:length(ks)], "   (ks = $ks)")
println("  eigenvalues of N       = ", eigen(Symmetric(0.5 * (N_hi + N_hi'))).values)
println("  -> any NEGATIVE eigenvalue here is impossible in exact arithmetic")
println("     and is a direct measure of the truncation damage.")

println("\n--- truncation error, maxdim $md_lo relative to $md_hi ---")
println("  ||dM||_F               = ", norm(dM))
println("  ||dL||                 = ", norm(dL))
println("  |dP|                   = ", abs(dP))
println("  ||dN||_F               = ", norm(dN))

println("\n--- gauge decomposition of that error ---")
println("  fitted u               = ", fit.u)
println("  fitted a               = ", fit.a)
println("  ||total error||        = ", fit.total_norm)
println("  ||gauge part||         = ", fit.gauge_norm)
println("  ||residual part||      = ", fit.resid_norm)
println("  residual / gauge       = ", fit.resid_norm / max(fit.gauge_norm, eps()))

println("\n--- the verdict ---")
println("  ||dN|| / ||N||         = ", norm(dN) / max(norm(N_hi), eps()))
println("  (this is the RELATIVE error in the quantity that sets the coefficients)")

ratio = norm(dN) / max(norm(N_hi), eps())
if ratio > 1.0
    println("\n  >>> ||dN|| EXCEEDS ||N||: at maxdim=$md_lo the answer is noise.")
elseif ratio > 0.1
    println("\n  >>> ||dN|| is >10% of ||N||: the M-route is marginal at maxdim=$md_lo.")
else
    println("\n  >>> ||dN|| is <10% of ||N||: gauge protection is largely holding here.")
end

# Coefficient impact, the practically meaningful endpoint
c_lo, _ = dynamic_mpf_coefficients(M_lo, L_lo)
c_hi, _ = dynamic_mpf_coefficients(M_hi, L_hi)
println("\n  c(maxdim=$md_lo)  = ", round.(c_lo; digits=5))
println("  c(maxdim=$md_hi)  = ", round.(c_hi; digits=5))
println("  max |dc|          = ", maximum(abs.(c_lo .- c_hi)))

println("\n  E_mpf($md_lo) = ", open_dynamic_mpf_error(M_lo, L_lo, c_lo, P_lo))
println("  E_mpf($md_hi) = ", open_dynamic_mpf_error(M_hi, L_hi, c_hi, P_hi))
println("  (negative values here are the section-7 symptom, reproduced)")
flush(stdout)
