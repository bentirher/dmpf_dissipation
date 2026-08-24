# =============================================================================
# directsum_test.jl
#
# V1 showed ||B + delta - A||/||A|| = 3.03e-8 for k=8 even with cutoff=0.0 and
# maxdim = chi(A)+chi(B), while truncate! reproduced that error to all eight
# digits. So the defect is introduced BY `+`, not by any subsequent truncation.
#
# 3.03e-8 ~ 2*sqrt(eps). That is the fingerprint of a density-matrix algorithm:
# forming rho = M M^dag and diagonalizing squares the singular values and costs
# half the available digits. ITensorMPS's add/+ defaults to
# alg="densitymatrix"; alg="directsum" performs a genuine direct sum with no
# fitting.
#
# If alg="directsum" drops the k=8 reconstruction to ~1e-15, that is the last
# numerical defect in the delta construction. Cheap: minutes at n=4.
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

exrel(X,Y) = _fnorm(+(X, -1*Y; alg="directsum")) / max(_fnorm(Y), eps())

println("ITensorMPS add algorithms: densitymatrix (default) vs directsum\n")
println(" k | alg            | chi(delta) | ||d||/||A|| | ||B+d-A||/||A||")
println("-"^72)
for kj in (3, 8)
    dt, q = t/kj, k0 ÷ kj
    A = get_open_step_MPO(n, J, gammas, dt, lsites, 1e-14, chi; order=order, dissipation=true)
    B = reference_block_MPO(n, J, gammas, dt, q, lsites, 1e-14, chi; order=order, dissipation=true)
    nA = _fnorm(A)
    for alg in ("densitymatrix", "directsum")
        d = try
            +(A, -1*B; alg=alg, cutoff=0.0, maxdim=maxlinkdim(A)+maxlinkdim(B))
        catch e
            println("  alg=$alg unsupported: $e"); continue
        end
        rec = +(B, d; alg="directsum")
        @printf("%2d | %-14s | %10d | %.4e | %.4e\n", kj, alg, maxlinkdim(d), _fnorm(d)/nA, exrel(rec, A))
        flush(stdout)
        # does truncate! stay lossless on the directsum result?
        if alg == "directsum"
            for cc in (1e-14, 1e-12)
                dc = deepcopy(d); truncate!(dc; cutoff=cc)
                @printf("%2d | %-14s | %10d | %.4e | %.4e\n", kj, "  +trunc $cc",
                        maxlinkdim(dc), _fnorm(dc)/nA, exrel(+(B, dc; alg="directsum"), A))
                flush(stdout)
            end
        end
    end
end
println("\n-> directsum at ~1e-15 for BOTH k means the delta construction is finally clean.")
println("   If k=8 stays at 3e-8, the defect is upstream in A or B, not in the sum.")
