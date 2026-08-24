# =============================================================================
# delta_microtest.jl
#
# P2 at step 1 sums SCALARS, so compression of the four-way MPO sum cannot
# explain it. By linearity the only remaining possibility is
#
#     B_j + delta_j  !=  A_j
#
# which is fatal, because every update rule was derived by substituting
# A_j = B_j + delta_j:
#     G' + Xi' + Psi' + Phi' = (G+Xi)(B_j + delta_j) + (Psi+Phi) A_j
# collapses to F A_j ONLY if that holds numerically.
#
# This script tests it directly and finds what controls the error, instead of
# reasoning about ITensor internals. Four probes:
#
#   M1  ||(B + delta) - A|| / ||A||  for delta formed several ways
#       (compressed at cutoff/maxdim, vs exact direct sum)
#   M2  the same identity at the SCALAR level, matching P2's metric exactly
#   M3  is _rmul(Id, X) == X?  (step 1 applies exactly this to B and delta;
#       if the identity multiply is lossy, that alone explains P2)
#   M4  P3's constant ratio 1.29e-4 vs cutoff and maxdim -- which knob moves it?
#       It is neither cutoff (1e-14) nor sqrt(cutoff) (1e-7), so it is set by
#       something else and worth identifying before trusting any tolerance.
#
# Cheap: n=3, everything untruncated. Minutes.
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

chi = theoretical_max_bond_dim(n)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

# Exact (no-truncation) difference, used as the measuring stick everywhere.
exdiff(X::MPO, Y::MPO) = +(X, -1 * Y; cutoff=0.0, maxdim=maxlinkdim(X) + maxlinkdim(Y))
exrel(X::MPO, Y::MPO)  = _fnorm(exdiff(X, Y)) / max(_fnorm(Y), eps())

@printf("n=%d chi_ceiling=%d  k0=%d order=%d\n\n", n, chi, k0, order)

# =============================================================================
# M1 / M2 -- does B + delta reproduce A?
# =============================================================================
println("="^92)
println("M1/M2  B + delta == A ?")
println("="^92)
println(" k | delta formed as              | chi(A) chi(B) chi(d) | ||d||/||A|| | M1 op-rel  | M2 scalar-rel")
println("-"^92)

for kj in (8, 3)
    dt, q = t / kj, k0 ÷ kj
    A = get_open_step_MPO(n, J, gammas, dt, lsites, 1e-14, chi; order=order, dissipation=true)
    B = reference_block_MPO(n, J, gammas, dt, q, lsites, 1e-14, chi;
                            order=order, dissipation=true)

    variants = [
        ("cutoff=1e-14, maxdim=chi", +(A, -1 * B; cutoff=1e-14, maxdim=chi)),
        ("cutoff=1e-16, maxdim=chi", +(A, -1 * B; cutoff=1e-16, maxdim=chi)),
        ("cutoff=0,     maxdim=2chi", +(A, -1 * B; cutoff=0.0,
                                        maxdim=maxlinkdim(A) + maxlinkdim(B))),
    ]

    sA = real(_expect(A, rho0))
    for (label, d) in variants
        recon = +(B, d; cutoff=0.0, maxdim=maxlinkdim(B) + maxlinkdim(d))
        m1 = exrel(recon, A)
        m2 = abs((real(_expect(B, rho0)) + real(_expect(d, rho0))) - sA) / max(abs(sA), eps())
        @printf("%2d | %-27s | %5d %5d %6d | %.4e | %.4e | %.4e\n",
                kj, label, maxlinkdim(A), maxlinkdim(B), maxlinkdim(d),
                _fnorm(d) / _fnorm(A), m1, m2)
        flush(stdout)
    end
end
println("""
  -> If the exact direct sum gives ~1e-15 while the compressed ones do not, the
     recursion is correct and compressing delta was the bug. If ALL variants
     fail, the problem is upstream in A or B, not in the subtraction.""")

# =============================================================================
# M3 -- is right-multiplication by the identity lossless?
# =============================================================================
println("\n", "="^92)
println("M3  _rmul(Id, X) == X ?   (step 1 does exactly this to B and to delta)")
println("="^92)

Id = identity_liouville_mpo(lsites)
for kj in (8, 3)
    dt, q = t / kj, k0 ÷ kj
    d = step_defect_MPO(n, J, gammas, dt, q, lsites, 1e-14, chi;
                        order=order, order_ref=order, dissipation=true,
                        exact_delta=true)
    for (nm, X) in (("A", d.A), ("B", d.B), ("delta", d.delta))
        R = _rmul(Id, X; cutoff=1e-14, maxdim=chi)
        L = _lmul(Id, X; cutoff=1e-14, maxdim=chi)
        @printf("  k=%d %-6s : ||rmul(Id,X)-X||/||X|| = %.4e   ||lmul(Id,X)-X||/||X|| = %.4e\n",
                kj, nm, exrel(R, X), exrel(L, X))
        flush(stdout)
    end
end
println("  -> anything above ~1e-14 here means the identity multiply itself is lossy,")
println("     which would corrupt step 1 regardless of how delta is formed.")

# =============================================================================
# M4 -- what sets P3's constant 1.29e-4?
# =============================================================================
println("\n", "="^92)
println("M4  P3 revisited: relative error in the SMALL term vs cutoff and maxdim")
println("="^92)

Sfine = get_open_step_MPO(n, J, gammas, t / k0, lsites, 1e-14, chi;
                          order=order, dissipation=true)
Gt = identity_liouville_mpo(lsites)
for _ in 1:6
    global Gt = left_multiply(Sfine, Gt; cutoff=1e-14, maxdim=chi)
end
dref = step_defect_MPO(n, J, gammas, t / 8, k0 ÷ 8, lsites, 1e-14, chi;
                       order=order, order_ref=order, dissipation=true,
                       exact_delta=true)
Small = dref.delta

println("cutoff  maxdim | scale  | rel err in small term")
println("-"^56)
for ct in (1e-12, 1e-14, 1e-16, 0.0), md in (chi, 2 * chi)
    for scale in (1e-2, 1e-6)
        Sc  = scale * Small
        Sum = +(Gt, Sc; cutoff=ct, maxdim=md)
        lhs = real(_expect(Sum, rho0))
        rhs = real(_expect(Gt, rho0)) + real(_expect(Sc, rho0))
        small_contrib = abs(real(_expect(Sc, rho0)))
        @printf("%.0e  %6d | %.0e | %.4e\n",
                ct, md, scale, abs(lhs - rhs) / max(small_contrib, eps()))
    end
    flush(stdout)
end
println("""
  -> If the ratio is flat in cutoff but drops when maxdim doubles, the loss is
     RANK truncation of the direct sum, not the singular-value cutoff -- and the
     fix everywhere is to allow the direct-sum bond dimension, not a tighter
     cutoff. If it is flat in both, the loss is in _expect / apply, not in the
     addition, and the microtest has found the wrong suspect.""")
