# =============================================================================
# run_N_route.jl   --   EXPERIMENTS 2 and 3
#
# Run AFTER gauge_diagnostic.jl. Three stages, in order of increasing cost:
#
#   STEP 1  norm hierarchy and bond dimensions, one (i,j) pair, tracing on.
#           This is the measurement that decides whether the reformulation is a
#           cost win or only a correctness fix. Cheap. Read it before the rest.
#
#   STEP 2  full N, coefficients, validated against the direct state-overlap
#           oracle (disqualified for production, correct as ground truth).
#
#   STEP 3  maxdim sweep -- the actual efficiency claim.
#
# Environment parameters: N_QUBITS, GAMMA, TVAL, K0, ORDER, MAXDIM, MAXDIM_G,
#                         K_REF_ORACLE, STEPS ("1", "12", "123", ...)
# =============================================================================

import Distributions
import Random
using LinearAlgebra
include("trotter_error_gram.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, default) = get(ENV, k, string(default))

n        = parse(Int,     getenv("N_QUBITS",     5))
gamma    = parse(Float64, getenv("GAMMA",        0.05))
t        = parse(Float64, getenv("TVAL",         3.0))
k0       = parse(Int,     getenv("K0",           48))    # multiple of BOTH 3 and 8
order    = parse(Int,     getenv("ORDER",        1))
maxdim   = parse(Int,     getenv("MAXDIM",       128))
maxdim_G = parse(Int,     getenv("MAXDIM_G",     48))
k_oracle = parse(Int,     getenv("K_REF_ORACLE", 200))
steps    = getenv("STEPS", "123")
ks       = [3, 8]
cutoff   = 1e-12

chi_ceiling = theoretical_max_bond_dim(n)
maxdim      = min(maxdim,   chi_ceiling)
maxdim_G    = min(maxdim_G, chi_ceiling)
for kj in ks
    @assert k0 % kj == 0 "K0=$k0 must be an integer multiple of k_j=$kj"
end

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

println("="^78)
println("N-ROUTE   n=$n  gamma=$gamma  t=$t  ks=$ks  k0=$k0  order=$order")
println("maxdim=$maxdim  maxdim_G=$maxdim_G   (ceiling for n=$n is $chi_ceiling)")
println("steps requested: $steps")
println("="^78)
flush(stdout)

# =============================================================================
# STEP 1 -- the diagnostic. Do this first.
# =============================================================================
if occursin("1", steps)
    println("\n", "="^78)
    println("STEP 1: norm hierarchy and bond dimensions, pair (k_i=8, k_j=8)")
    println("="^78)
    println("""
Expect, if the analysis holds:
  ||G||  >>  ||Xi||, ||Psi||  >>  ||Phi||     (order 1 : eps : eps^2)
  final <<rho0|Phi|rho0>> ~ E_k8 ~ 1.9e-4     (the recorded n=5 value)
  chi_Phi materially below chi_G              <- the cost claim
""")
    flush(stdout)

    t0 = time()
    pair = build_Phi_pair(n, J, gammas, t, 8, 8, k0, lsites;
                          cutoff=cutoff, maxdim=maxdim, maxdim_G=maxdim_G,
                          order=order, order_ref=order, dissipation=true, trace=true)
    println("built in $(round(time() - t0; digits=1)) s\n"); flush(stdout)

    trace_report(pair.history)

    println("\nfinal ||Phi||_F           = ", _fnorm(pair.Phi))
    println("N_88 = <<rho0|Phi|rho0>>  = ", real(_expect(pair.Phi, rho0)))
    println("  (recorded E_k8 at n=5, gamma=0.05, t=3 was 1.9e-4)")
    println("\nfinal bond dims:  chi_G = ", pair.G === nothing ? 0 : middle_bond_dim(pair.G),
            "   chi_Phi = ", pair.Phi === nothing ? 0 : middle_bond_dim(pair.Phi))
    println("""
READING THIS:
  - if the norm hierarchy fails, k0=$k0 is not converged: Delta_j is measuring
    reference error rather than Trotter error. Raise K0 and re-run.
  - if chi_Phi << chi_G, the reformulation is a genuine cost win.
  - if chi_Phi also saturates, it is a correctness fix only, and the asymptotic
    claim has to come from the KMS or gamma-expansion routes.
""")
    flush(stdout)
end

# =============================================================================
# STEP 2 -- full N, validated against the direct oracle
# =============================================================================
N_direct = nothing

if occursin("2", steps)
    println("\n", "="^78)
    println("STEP 2: full N vs direct state-overlap oracle")
    println("="^78)
    flush(stdout)

    t0 = time()
    res = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                           cutoff=cutoff, maxdim=maxdim, maxdim_G=maxdim_G,
                           order=order, order_ref=order, dissipation=true)
    println("recursion route built in $(round(time() - t0; digits=1)) s"); flush(stdout)

    t0 = time()
    N_direct = validate_N_against_direct(n, J, gammas, t, ks, k_oracle, lsites, rho0;
                                         cutoff=cutoff, maxdim=chi_ceiling,
                                         order=order, order_ref=order, dissipation=true)
    println("direct oracle built in $(round(time() - t0; digits=1)) s\n"); flush(stdout)

    print_N_report(res; N_direct=N_direct)

    println("""

NOTE: do NOT validate against M - L_i - L_j + P. That combination is the
catastrophic cancellation this reformulation exists to avoid; agreement with it
would be bounded by its own ~2e-3 truncation floor, which likely exceeds N.
""")
    flush(stdout)
end

# =============================================================================
# STEP 3 -- the efficiency claim
# =============================================================================
if occursin("3", steps)
    println("\n", "="^78)
    println("STEP 3: maxdim stability of the coefficients")
    println("="^78)
    println("""
The M-route needed maxdim_ref = 256 at n=5 and still reached only ~2e-3
relative on M. If the error analysis holds, c should already be stable here at
maxdim ~ 32.
""")

    if N_direct === nothing
        println("(building the oracle once for comparison)"); flush(stdout)
        N_direct = validate_N_against_direct(n, J, gammas, t, ks, k_oracle, lsites, rho0;
                                             cutoff=cutoff, maxdim=chi_ceiling,
                                             order=order, order_ref=order, dissipation=true)
    end

    grid = [(16, 16), (32, 16), (48, 24), (64, 32), (96, 48), (128, 48)]
    grid = [(min(a, chi_ceiling), min(b, chi_ceiling)) for (a, b) in grid]

    println("maxdim | maxdim_G | c                        | E_mpf      | ||dN||/||N|| | time (s)")
    println("-"^92)
    for (md, mdG) in grid
        t0 = time()
        r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                             cutoff=cutoff, maxdim=md, maxdim_G=mdG,
                             order=order, order_ref=order, dissipation=true)
        el = time() - t0
        relerr = norm(r.N .- N_direct) / norm(N_direct)
        println(rpad(md, 6), " | ", rpad(mdG, 8), " | ",
                rpad(string(round.(r.coeffs; digits=5)), 24), " | ",
                rpad(round(r.E_mpf; sigdigits=4), 10), " | ",
                rpad(round(relerr; sigdigits=3), 12), " | ", round(el; digits=1))
        flush(stdout)
    end
end

println("\ndone.")
