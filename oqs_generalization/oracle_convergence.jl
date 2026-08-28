# =============================================================================
# oracle_convergence.jl -- settle the n=6 ground truth.
#
# THE PROBLEM. Job 6995914 reported
#     maxdim= 64  c1=-0.1357320200
#     maxdim=128  c1=-0.1362048373      drift = 4.728e-04
# At n=4 the same check gave drift = 0.000e+00 EXACTLY. It should here too: the
# oracle manipulates an MPS on 6 Liouville sites of local dimension 4, so the
# middle cut admits Schmidt rank at most 4^3 = 64. maxdim=64 is the ceiling and
# raising it to 128 cannot add a single state. A non-zero drift therefore is NOT
# a bond-dimension effect -- it is accumulated numerical error, and the prime
# suspect is `cutoff`: delta_j = rho_kj - rho_ref is a small difference of two
# O(1) states, so a cutoff applied relative to the LARGE intermediate is far too
# loose relative to the small result, over ~k0 steps per branch.
#
# WHY IT MATTERS. The drift is 2.1% of the n=6 signal (2.2886e-02), and the best
# N-route point is dc = 2.31e-03 -- only ~5x the ground truth's own uncertainty.
# The n=6 conclusions survive that, but not comfortably, and not for publication.
#
# THIS SCRIPT is cheap: the oracle is an MPS calculation, which is why it was
# made ground truth in the first place. It scans maxdim x cutoff and reports
# where c1 stops moving.
#
# Environment: N_QUBITS (default 6), GAMMA, TVAL, K0, K_REF, ORDER
# =============================================================================
import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")
include("open_optimization_problem.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n     = parse(Int,     getenv("N_QUBITS", 6))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
k_ref = parse(Int,     getenv("K_REF",    48))
order = parse(Int,     getenv("ORDER",    2))
ks    = [3, 8]
chi_mps = 4^min(n ÷ 2, n - n ÷ 2)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("oracle_convergence  n=%d  MPS ceiling = 4^%d = %d\n\n",
        n, min(n ÷ 2, n - n ÷ 2), chi_mps)
@printf("maxdim=%d is the ceiling: larger values CANNOT add states, so any\n", chi_mps)
println("variation across a row is pure numerical error.\n")

cof(N) = trotter_error_coefficients(N).coeffs

function scan()
    rows = ["cutoff,maxdim,c1,N11,N12,N22"]
    cutoffs = [1e-12, 1e-14, 1e-16, 0.0]
    maxdims = filter(<=(4 * chi_mps), [chi_mps ÷ 2, chi_mps, 2 * chi_mps, 4 * chi_mps])
    @printf("  %-9s", "cutoff")
    for md in maxdims; @printf("  maxdim=%-13d", md); end
    println()
    for ct in cutoffs
        @printf("  %-9.0e", ct)
        for md in maxdims
            Nd = validate_N_against_direct(n, J, gammas, t, ks, k_ref, lsites, rho0;
                                           cutoff=ct, maxdim=md, order=order,
                                           order_ref=order, dissipation=true)
            c1 = cof(Nd)[1]
            @printf("  %+.10f", c1)
            push!(rows, @sprintf("%.1e,%d,%.12f,%.10e,%.10e,%.10e",
                                 ct, md, c1, Nd[1,1], Nd[1,2], Nd[2,2]))
            flush(stdout)
        end
        println()
        write("oracle_convergence.csv", join(rows, "\n") * "\n")
    end
    return rows
end

scan()

println("\nHOW TO READ THIS")
println("  Across a ROW (fixed cutoff, rising maxdim): must be flat from maxdim=$chi_mps.")
println("    If it is not, the truncation is not the binding constraint -- cutoff is.")
println("  Down a COLUMN (fixed maxdim, tightening cutoff): the value it converges to")
println("    is the true ground truth. Use THAT for n=6, and quote the residual")
println("    spread as the ground-truth uncertainty.")
println("\nwrote oracle_convergence.csv")
