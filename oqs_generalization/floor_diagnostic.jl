# =============================================================================
# floor_diagnostic.jl -- where does the ~1e-4 "floor" come from?
#
# At maxdim = theoretical_max_bond_dim(n) NOTHING is truncated, so the M-route
# and the N-route are the same object and must agree to ~1e-14. They do not.
# This script localises the disagreement by building N three independent ways
# at the ceiling and comparing them entry by entry.
#
#   N_direct : evolve rho_kj and rho_ref as MPS, form delta_j = rho_kj - rho_ref,
#              take inner products. This is the ORACLE (trotter_error_gram.jl
#              line 436: "exactly right as ground truth at small n"). It is also
#              CHEAP: it evolves an MPS, whose ceiling is 4^min(l,n-l) = 16 at
#              n=4, not the MPO ceiling 16^min = 256.
#   N_M      : build_N(M, L, P) -- the M-route, subject to the catastrophic
#              cancellation this whole project exists to avoid.
#   N_N      : trotter_error_gram -- the N-route.
#
# c_exact in comparison_study.jl is currently dynamic_mpf_coefficients(M, L),
# i.e. the Lagrange solve in M and L. That is the UNRELIABLE route being used
# as ground truth. Part 1 tests exactly that.
#
# Environment: N_QUBITS, GAMMA, TVAL, K0, K_REF, ORDER
# Run:  julia floor_diagnostic.jl
# =============================================================================
import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")
include("open_optimization_problem.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n     = parse(Int,     getenv("N_QUBITS", 4))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
k_ref = parse(Int,     getenv("K_REF",    48))
order = parse(Int,     getenv("ORDER",    2))
ks    = [3, 8]
ct    = 1e-14
chi   = theoretical_max_bond_dim(n)          # MPO ceiling  = 16^min(l,n-l)
chi_mps = 4^min(n ÷ 2, n - n ÷ 2)            # MPS ceiling  =  4^min(l,n-l)

@assert k_ref == k0 "k_ref must equal k0 or the routes target different references"

# EXACTLY the same setup as comparison_study.jl -- same seed, same J, same rho0
Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("floor_diagnostic  n=%d gamma=%.3f t=%.1f ks=%s k0=%d order=%d\n",
        n, gamma, t, string(ks), k0, order)
@printf("MPO ceiling = %d (16^%d)    MPS ceiling = %d (4^%d)\n\n",
        chi, min(n ÷ 2, n - n ÷ 2), chi_mps, min(n ÷ 2, n - n ÷ 2))
flush(stdout)

build_N(M, L, P) = [M[i,j] - L[i] - L[j] + P for i in 1:length(L), j in 1:length(L)]

function m_route(md)
    M, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                            cutoff=ct, maxdim=md, order=order, dissipation=true)
    L, _ = open_L_vector(n, J, gammas, t, ks, k_ref, lsites, rho0;
                         cutoff=ct, maxdim=md, order=order, order_ref=order, dissipation=true)
    P = reference_purity(n, J, gammas, t, k_ref, lsites, rho0;
                         cutoff=ct, maxdim=md, order=order, dissipation=true)
    c_lagrange, _ = dynamic_mpf_coefficients(M, L)
    (M=M, L=L, P=P, N=build_N(M, L, P), c_lagrange=c_lagrange)
end

cof(N) = trotter_error_coefficients(N).coeffs

# =============================================================================
# PART 0 -- is the oracle converged? MPS ceiling is 4^min, not 16^min.
# =============================================================================
# NOTE: this is a function, not a bare top-level loop. In Julia, assigning to a
# variable inside a top-level `for` creates a NEW LOCAL rather than updating the
# global of the same name ("soft scope"), so `prev = Nd` here would be invisible
# on the next iteration and reading `prev` would throw UndefVarError. A function
# body uses hard scope and behaves as intended.
function oracle_scan(mds)
    N_ceiling = nothing
    prev = nothing
    for md in mds
        md < 2 && continue
        Nd = validate_N_against_direct(n, J, gammas, t, ks, k_ref, lsites, rho0;
                                       cutoff=ct, maxdim=md, order=order,
                                       order_ref=order, dissipation=true)
        c = cof(Nd)
        d = prev === nothing ? NaN : maximum(abs.(c .- cof(prev)))
        @printf("  maxdim=%4d  c1=%+.10f  N11=%.8e  N12=%.8e  dc vs prev=%.2e\n",
                md, c[1], Nd[1,1], Nd[1,2], d)
        prev = Nd
        md >= chi_mps && (N_ceiling = Nd)
        flush(stdout)
    end
    N_ceiling === nothing && error("oracle_scan produced no run at or above the MPS ceiling")
    return N_ceiling
end

println("="^76)
println("PART 0 : oracle self-convergence (MPS ceiling is $chi_mps, NOT $chi)")
println("="^76)
N_orc = oracle_scan(unique([chi_mps ÷ 2, chi_mps, 2 * chi_mps, chi]))
println("  -> if these agree from maxdim=$chi_mps up, the oracle is EXACT and is")
println("     the correct ground truth. It is also far cheaper than either route.\n")
c_orc = cof(N_orc)

# =============================================================================
# PART 1 -- three-way comparison at the ceiling
# =============================================================================
println("="^76)
println("PART 1 : all three constructions at maxdim = $chi (nothing truncated)")
println("="^76)
mr  = m_route(chi)
rN  = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                       cutoff=ct, maxdim=chi, maxdim_G=chi, order=order,
                       order_ref=order, dissipation=true, exact_delta=true)

labels = ["N_direct (ORACLE)", "N_M = M-L-L+P", "N_N (N-route)"]
mats   = [N_orc, mr.N, rN.N]
println("\nMatrix entries:")
@printf("  %-20s %14s %14s %14s\n", "", "N11", "N12", "N22")
for (lab, A) in zip(labels, mats)
    @printf("  %-20s %14.8e %14.8e %14.8e\n", lab, A[1,1], A[1,2], A[2,2])
end
println("\nRelative deviation from the oracle, entry by entry:")
for (lab, A) in zip(labels[2:end], mats[2:end])
    @printf("  %-20s %13.3e%% %13.3e%% %13.3e%%\n", lab,
            100*(A[1,1]-N_orc[1,1])/N_orc[1,1],
            100*(A[1,2]-N_orc[1,2])/N_orc[1,2],
            100*(A[2,2]-N_orc[2,2])/N_orc[2,2])
end

println("\nCoefficients (all four ways of getting c):")
cands = [("oracle           ", c_orc),
         ("N-route          ", rN.coeffs),
         ("N from M,L,P     ", cof(mr.N)),
         ("Lagrange in M,L  ", mr.c_lagrange)]
for (lab, c) in cands
    @printf("  %s c1=%+.10f   dc vs oracle = %.3e\n", lab, c[1],
            maximum(abs.(c .- c_orc)))
end
println("""
  READ THIS: 'Lagrange in M,L' is what comparison_study.jl calls c_exact.
  If its dc-vs-oracle is comparable to the reported N-route floor (~1e-4), then
  the floor is the GROUND TRUTH's error, not the N-route's, and every dc_max
  column published so far is measured against the wrong reference.
""")
flush(stdout)

# =============================================================================
# PART 2 -- the documented maxdim_G effect
# =============================================================================
println("="^76)
println("PART 2 : maxdim_G at fixed maxdim = $chi")
println("="^76)
println("trotter_error_gram.jl:220 states maxdim=256/maxdim_G=128 gave dc=9.4e-4")
println("and maxdim_G=256 gave 1.0e-4. The uploaded cmp_coefficients.csv reports")
println("9.36e-4 at maxdim=256, which is the maxdim_G=128 number.\n")
rows = ["maxdim,maxdim_G,c1,dc_vs_oracle,dc_vs_lagrange,N11,N12,N22,lambda_min"]
@printf("  %8s %14s %14s %14s\n", "maxdim_G", "c1", "dc vs oracle", "dc vs Lagrange")

function maxdim_G_scan!(rows, mgs)
    for mg in mgs
        r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                             cutoff=ct, maxdim=chi, maxdim_G=mg, order=order,
                             order_ref=order, dissipation=true, exact_delta=true)
        d_o = maximum(abs.(r.coeffs .- c_orc))
        d_l = maximum(abs.(r.coeffs .- mr.c_lagrange))
        @printf("  %8d %+14.10f %14.3e %14.3e\n", mg, r.coeffs[1], d_o, d_l)
        push!(rows, @sprintf("%d,%d,%.10f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                             chi, mg, r.coeffs[1], d_o, d_l,
                             r.N[1,1], r.N[1,2], r.N[2,2], minimum(r.eigvals)))
        # Flush to disk after EVERY point: this loop is the long pole (~6 runs at
        # the ceiling) and a wall-clock timeout should not cost the whole sweep.
        write("floor_diagnostic.csv", join(rows, "\n") * "\n")
        flush(stdout)
    end
end

maxdim_G_scan!(rows, filter(<=(chi), [16, 32, 64, 128, 192, 256]))

# reference rows so the CSV is self-contained
push!(rows, @sprintf("%d,%d,%.10f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                     chi, -1, c_orc[1], 0.0, maximum(abs.(c_orc .- mr.c_lagrange)),
                     N_orc[1,1], N_orc[1,2], N_orc[2,2],
                     minimum(eigen(Symmetric(0.5*(N_orc+N_orc'))).values)))
push!(rows, @sprintf("%d,%d,%.10f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                     chi, -2, mr.c_lagrange[1],
                     maximum(abs.(mr.c_lagrange .- c_orc)), 0.0,
                     mr.N[1,1], mr.N[1,2], mr.N[2,2],
                     minimum(eigen(Symmetric(0.5*(mr.N+mr.N'))).values)))
write("floor_diagnostic.csv", join(rows, "\n") * "\n")
println("\nwrote floor_diagnostic.csv  (maxdim_G = -1 is the oracle, -2 the Lagrange M-route)")
