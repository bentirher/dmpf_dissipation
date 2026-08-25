# =============================================================================
# directsum_test.jl  (v2)
#
# v1 failed because alg="directsum" takes NO cutoff/maxdim -- it is an exact
# direct sum, so truncation parameters are meaningless and rejected.
#
# v1 also revealed that the DIAGNOSTIC was broken, not just the construction:
# measured with directsum, the k=3 reconstruction error is 1.46e-8, whereas V1
# (measuring with the default density-matrix `+`) reported 2.95e-15. The
# density-matrix subtraction was losing the difference it was measuring.
#
# Conclusion: ITensorMPS's default `+` carries sqrt(eps) ~ 1.5e-8 accuracy.
# This script confirms directsum fixes the construction and checks whether
# truncate! can then bring chi(delta) back below the ceiling losslessly --
# which matters because exact delta at chi(A)+chi(B) = 272 drove the chi=256
# runtime from ~80 s to 1232 s.
# =============================================================================
import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")
BLAS.set_num_threads(parse(Int, get(ENV,"SLURM_CPUS_PER_TASK","1")))

n, gamma, t, k0, order = 4, 0.05, 3.0, 48, 2
chi = theoretical_max_bond_dim(n)
Random.seed!(1234)
J = rand(Distributions.Uniform(1/4,3/4), n-1); gammas = fill(gamma,n)
lsites = liouville_siteinds(n)

# ALWAYS measure differences with directsum.
exrel(X,Y) = _fnorm(+(X, -1*Y; alg="directsum")) / max(_fnorm(Y), eps())

println("n=$n ceiling=$chi   all differences measured with alg=directsum\n")
println(" k | delta built by            | chi(delta) | ||d||/||A|| | ||B+d-A||/||A||")
println("-"^78)
for kj in (3, 8)
    dt, q = t/kj, k0 ÷ kj
    A = get_open_step_MPO(n, J, gammas, dt, lsites, 1e-14, chi; order=order, dissipation=true)
    B = reference_block_MPO(n, J, gammas, dt, q, lsites, 1e-14, chi; order=order, dissipation=true)
    nA = _fnorm(A)

    d_dm = +(A, -1*B; cutoff=0.0, maxdim=maxlinkdim(A)+maxlinkdim(B))
    @printf("%2d | %-25s | %10d | %.4e | %.4e\n", kj, "densitymatrix (default)",
            maxlinkdim(d_dm), _fnorm(d_dm)/nA, exrel(+(B, d_dm; alg="directsum"), A))
    flush(stdout)

    d_ds = +(A, -1*B; alg="directsum")
    @printf("%2d | %-25s | %10d | %.4e | %.4e\n", kj, "directsum",
            maxlinkdim(d_ds), _fnorm(d_ds)/nA, exrel(+(B, d_ds; alg="directsum"), A))
    flush(stdout)

    for cc in (1e-16, 1e-14, 1e-12, 1e-10)
        dc = deepcopy(d_ds); truncate!(dc; cutoff=cc)
        @printf("%2d | %-25s | %10d | %.4e | %.4e\n", kj, "  directsum+trunc $cc",
                maxlinkdim(dc), _fnorm(dc)/nA, exrel(+(B, dc; alg="directsum"), A))
        flush(stdout)
    end
end
println("""
-> directsum at ~1e-15 for BOTH k confirms the construction is clean.
   Then pick the LARGEST truncate! cutoff that holds ~1e-13: that is the
   compress_cutoff to set, and it recovers the runtime lost to chi(delta)=272.""")
