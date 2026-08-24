# =============================================================================
# delta_fix_verify.jl
#
# The delta-compression bug is confirmed (microtest M1/M2: exact direct sum
# gives 1e-12..1e-14, compressed gives 3.8e-3 / 4.3e-2). Two things remain:
#
#   V1  Can `truncate!` compress the exact delta back below the ceiling
#       LOSSLESSLY? The direct sum has chi(A)+chi(B) = 512 at n=4, above the
#       mathematical ceiling of 256, and delta is applied twice per step.
#       `truncate!` orthogonalizes into canonical form before its SVD, unlike
#       whatever `+` does internally, so it may succeed where `+` fails. If it
#       does, we get correctness AND the original cost.
#
#   V2  Re-run the head-to-head with BOTH fixes:
#         (a) exact_delta = true      (this bug)
#         (b) maxdim_G = maxdim       (the earlier bug: comparison_study used
#                                      maxdim_G = maxdim/2, so G was truncated
#                                      at 128 while the run was reported as
#                                      untruncated at 256)
#       Both push the same direction, so every N-route number in cmp_*.csv is
#       pessimistic. This measures by how much.
#
# NOTE on what was and was not affected: chi(delta_8) = 152 at n=4, BELOW the
# 256 cap, so delta_8 was never truncated and E_k8 was always sound. But
# chi(delta_3) = 256 hit the cap, so the k=3 branch was corrupted -- which is
# where the N_11 error (+26.9%) and much of what was attributed to L5 came from.
#
# Environment: N_QUBITS (4), GAMMA, TVAL, K0, ORDER, STAGES ("12")
# =============================================================================

import Distributions
import Random
using LinearAlgebra
using Printf
include("trotter_error_gram.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n      = parse(Int,     getenv("N_QUBITS", 4))
gamma  = parse(Float64, getenv("GAMMA",    0.05))
t      = parse(Float64, getenv("TVAL",     3.0))
k0     = parse(Int,     getenv("K0",       48))
order  = parse(Int,     getenv("ORDER",    2))
stages = getenv("STAGES", "12")
ks     = [3, 8]
ct     = 1e-14

chi = theoretical_max_bond_dim(n)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

exdiff(X, Y) = +(X, -1 * Y; cutoff=0.0, maxdim=maxlinkdim(X) + maxlinkdim(Y))
exrel(X, Y)  = _fnorm(exdiff(X, Y)) / max(_fnorm(Y), eps())

@printf("n=%d ceiling=%d k0=%d order=%d\n\n", n, chi, k0, order)

# =============================================================================
# V1 -- can truncate! compress delta losslessly?
# =============================================================================
if occursin("1", stages)
    println("="^94)
    println("V1  lossless compression of the exact delta")
    println("="^94)
    println(" k | variant                    | chi(delta) | ||d||/||A|| | ||B+d-A||/||A||")
    println("-"^94)

    for kj in ks
        dt, q = t / kj, k0 ÷ kj
        A = get_open_step_MPO(n, J, gammas, dt, lsites, ct, chi; order=order, dissipation=true)
        B = reference_block_MPO(n, J, gammas, dt, q, lsites, ct, chi;
                                order=order, dissipation=true)
        nA = _fnorm(A)

        d_exact = +(A, -1 * B; cutoff=0.0, maxdim=maxlinkdim(A) + maxlinkdim(B))
        rec = +(B, d_exact; cutoff=0.0, maxdim=maxlinkdim(B) + maxlinkdim(d_exact))
        @printf("%2d | %-26s | %10d | %.4e | %.4e\n", kj, "exact direct sum",
                maxlinkdim(d_exact), _fnorm(d_exact) / nA, exrel(rec, A))
        flush(stdout)

        for cc in (1e-16, 1e-14, 1e-12, 1e-10)
            dc = deepcopy(d_exact)
            truncate!(dc; cutoff=cc)
            r = +(B, dc; cutoff=0.0, maxdim=maxlinkdim(B) + maxlinkdim(dc))
            @printf("%2d | truncate! cutoff=%-9.0e | %10d | %.4e | %.4e\n",
                    kj, cc, maxlinkdim(dc), _fnorm(dc) / nA, exrel(r, A))
            flush(stdout)
        end
    end
    println("""
  -> If truncate! holds the reconstruction at ~1e-13 while cutting chi(delta)
     well below chi(A)+chi(B), set compress_delta=true and keep the original
     cost. If reconstruction degrades at every cutoff, carry the uncompressed
     delta and accept roughly 2x on the delta applications.""")
end

# =============================================================================
# V2 -- head-to-head rerun with both fixes
# =============================================================================
if occursin("2", stages)
    println("\n", "="^94)
    println("V2  N-route with exact_delta AND maxdim_G = maxdim")
    println("="^94)
    println("""
Reference (n=4, k_ref=48, order=2, untruncated M-route):
  c      = [-0.144708, 1.144708]
  E_k3   = 1.334740e-02      E_k8 = 2.502581e-04
CAVEAT: k_ref=48 is NOT converged -- T6 showed k0 48->96 moves E_k8 by +6.3%.
These are exact in the sense of untruncated, not physically correct. Fine for
an M-vs-N comparison (both use the same reference), but E_k8 = 2.5e-4 is not
the physical value.

Previous N-route (delta compressed, maxdim_G = maxdim/2), for comparison:
  chi= 16 dc=1.364e-01 | 32 dc=6.226e-03 | 64 dc=8.667e-04 | 256 dc=9.360e-04
""")
    c_exact  = [-0.144708, 1.144708]
    Ek_exact = [1.334740e-02, 2.502581e-04]

    println("maxdim | c1         c2        | dc_max    | E_k3 err | E_k8 err | lam_min     | time(s)")
    println("-"^94)
    for md in filter(<=(chi), [16, 32, 48, 64, 96, 128, 192, 256])
        t0 = time()
        r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                             cutoff=ct, maxdim=md, maxdim_G=md,
                             order=order, order_ref=order, dissipation=true,
                             exact_delta=true)
        @printf("%6d | %+.6f %+.6f | %.3e | %+7.2f%% | %+7.2f%% | %+.4e | %.1f\n",
                md, r.coeffs[1], r.coeffs[2], maximum(abs.(r.coeffs .- c_exact)),
                100 * (r.E_trot[1] - Ek_exact[1]) / Ek_exact[1],
                100 * (r.E_trot[2] - Ek_exact[2]) / Ek_exact[2],
                minimum(r.eigvals), time() - t0)
        flush(stdout)
    end
    println("""
  -> chi=256 should now be EXACT (dc ~ 1e-12). If it is, the recursion is
     vindicated and every earlier N-route number was depressed by these two
     bugs. If a floor remains, something upstream of the recursion is still
     wrong and no further sweeps are worth running.""")
end
