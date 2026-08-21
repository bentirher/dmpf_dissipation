# =============================================================================
# spectra_study.jl
#
# The spectra + discarded-weight-vs-error section of comparison_study.jl, split
# out so it can run WITHOUT repeating the 48-minute maxdim sweep. It reads the
# dc values from cmp_coefficients.csv, which the sweep already wrote.
#
# Include chain note: spectrum_truncation_analysis.jl has NO include of its own
# (it assumes the caller loaded the chain), so including trotter_error_gram.jl
# first and it second gives a SINGLE load -- no `redefinition of constant
# Main.ID2` warnings, unlike comparison_study.jl which pulls the chain twice.
#
# Environment: N_QUBITS, GAMMA, TVAL, K0, ORDER, KDIAG (default 8)
# =============================================================================

import Distributions
import Random
using LinearAlgebra
using Printf
include("trotter_error_gram.jl")
include("spectrum_truncation_analysis.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n      = parse(Int,     getenv("N_QUBITS", 4))
gamma  = parse(Float64, getenv("GAMMA",    0.05))
t      = parse(Float64, getenv("TVAL",     3.0))
k0     = parse(Int,     getenv("K0",       48))
order  = parse(Int,     getenv("ORDER",    2))
kd     = parse(Int,     getenv("KDIAG",    8))    # which diagonal pair to profile
cutoff = 1e-12

chi_max = theoretical_max_bond_dim(n)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("n=%d  gamma=%.3f  t=%.1f  k=%d  k0=%d  order=%d  chi_max=%d\n\n",
        n, gamma, t, kd, k0, order, chi_max)
flush(stdout)

# =============================================================================
# Build F_kk and Phi_kk, both UNTRUNCATED
# =============================================================================
println("building F_$(kd)$(kd) at maxdim=$chi_max ..."); flush(stdout)
t0 = time()
# NOTE: build_open_F takes a LIST of k's and returns Vector{MPO}, one entry per
# (i,j) pair in row-major order. Passing the scalar `kd` silently "works"
# (eachindex(8) -> OneTo(1), 8[1] -> 8) but returns a 1-element VECTOR, which is
# what broke comparison_study.jl. Pass [kd] and index explicitly.
F_list = build_open_F(n, J, gammas, t, [kd], lsites, cutoff, chi_max;
                      order=order, dissipation=true)
F = F_list[1]
@printf("  done in %.1f s, chi = %d\n", time() - t0, middle_bond_dim(F)); flush(stdout)

println("building Phi_$(kd)$(kd) at maxdim=$chi_max ..."); flush(stdout)
t0 = time()
pair = build_Phi_pair(n, J, gammas, t, kd, kd, k0, lsites;
                      cutoff=cutoff, maxdim=chi_max, maxdim_G=chi_max,
                      order=order, order_ref=order, dissipation=true)
Phi = pair.Phi
@printf("  done in %.1f s, chi = %d\n\n", time() - t0, middle_bond_dim(Phi)); flush(stdout)

sF   = operator_schmidt_spectrum(F)
sPhi = operator_schmidt_spectrum(Phi)
sG   = pair.G  === nothing ? Float64[] : operator_schmidt_spectrum(pair.G)
sXi  = pair.Xi === nothing ? Float64[] : operator_schmidt_spectrum(pair.Xi)

# =============================================================================
# Spectra CSV
# =============================================================================
rows_s = ["index,object,sigma,sigma_over_s1,cum_weight_frac,tail_weight_frac"]
for (name, s) in (("F", sF), ("Phi", sPhi), ("G", sG), ("Xi", sXi))
    isempty(s) && continue
    w = s .^ 2; tot = sum(w); cw = cumsum(w) ./ tot
    for i in eachindex(s)
        push!(rows_s, @sprintf("%d,%s,%.10e,%.10e,%.12f,%.12e",
                               i, name, s[i], s[i]/s[1], cw[i], 1 - cw[i]))
    end
end
write("cmp_spectra.csv", join(rows_s, "\n") * "\n")

println("="^72)
println("SPECTRA")
println("="^72)
println("object |  chi | s_1        | eff_rank @1-1e-3 | @1-1e-6 | s_2/s_1")
for (name, s) in (("F", sF), ("Phi", sPhi), ("G", sG), ("Xi", sXi))
    isempty(s) && continue
    @printf("%-6s | %4d | %.4e | %16s | %7s | %.4e\n", name, length(s), s[1],
            string(effective_rank(s; weight_fraction=1-1e-3)),
            string(effective_rank(s; weight_fraction=1-1e-6)),
            length(s) > 1 ? s[2]/s[1] : NaN)
end

println("""

HOW TO READ THIS. Do not compare the normalized spectra of F and Phi directly:
for F, s_1 IS the G = E^dag E piece, which cancels identically out of the
coefficients -- so log10(s_i/s_1) makes F's tail look negligible when that tail
is the entire signal. Phi has no such passenger; every singular value of Phi
contributes. Phi's normalized spectrum will very likely look WORSE than F's,
and that is expected, not a problem. The apples-to-apples comparison is the
weight-vs-error table below.
""")

# =============================================================================
# Discarded weight vs error in c  -- THE KEY FIGURE
# =============================================================================
# The slope of this relation IS the amplification factor, measured. For the
# M-route expect: tiny discarded weight, huge error in c. For the N-route:
# large discarded weight, small error in c.

discarded(s, md) = md >= length(s) ? 0.0 : 1 - sum(s[1:md] .^ 2) / sum(s .^ 2)

# Read the dc values the sweep already computed.
dc = Dict{Tuple{Int,String},Float64}()
if isfile("cmp_coefficients.csv")
    for line in readlines("cmp_coefficients.csv")[2:end]
        f = split(strip(line), ",")
        length(f) >= 5 || continue
        dc[(parse(Int, f[1]), String(f[2]))] = parse(Float64, f[5])
    end
else
    println("WARNING: cmp_coefficients.csv not found -- run comparison_study.jl first.")
end

grid = sort(unique([k[1] for k in keys(dc)]))
isempty(grid) && (grid = filter(<=(chi_max), [16, 32, 48, 64, 96, 128, 192, 256]))

rows_w = ["maxdim,object,discarded_weight,dc_max,route,amplification"]
println("="^72)
println("DISCARDED WEIGHT vs ERROR IN c")
println("="^72)
println("maxdim |  F: disc.wt   dc(M)      amp     |  Phi: disc.wt  dc(N)      amp")
for md in grid
    wF, wP = discarded(sF, md), discarded(sPhi, md)
    dcM = get(dc, (md, "M"), NaN)
    dcN = get(dc, (md, "N"), NaN)
    # "amplification": error produced in c per unit of Schmidt weight discarded.
    ampM = wF > 0 ? dcM / wF : NaN
    ampN = wP > 0 ? dcN / wP : NaN
    push!(rows_w, @sprintf("%d,F,%.8e,%.8e,M,%.8e",   md, wF, dcM, ampM))
    push!(rows_w, @sprintf("%d,Phi,%.8e,%.8e,N,%.8e", md, wP, dcN, ampN))
    @printf("%6d | %.3e  %.3e  %8.2e | %.3e   %.3e  %8.2e\n",
            md, wF, dcM, ampM, wP, dcN, ampN)
end
write("cmp_weight_vs_error.csv", join(rows_w, "\n") * "\n")

println("\nwrote cmp_spectra.csv, cmp_weight_vs_error.csv")
println("(cmp_coefficients.csv and cmp_errors.csv were written by the earlier sweep)")
