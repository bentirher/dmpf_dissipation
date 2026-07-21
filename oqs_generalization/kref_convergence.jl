# kref_convergence.jl
# Finds where the DMPF coefficients converge as the reference k_ref -> exact.
#
# The coeff_sanity diagnostic showed c swings from [0.956,0.044] (k_ref=40) to
# [-0.424,1.424] (k_ref=160) -- i.e. k_ref=40 is FAR from converged and produces
# physically wrong weights (coarse k=3 circuit dominating the fine k=8 one).
# This study extends k_ref until c plateaus, and tracks the reference's own
# convergence directly, so we learn the minimum "safe" reference accuracy.
#
# Ground-truth errors are computed DIRECTLY (||rho_ref - rho_kj||^2 via explicit
# MPS), not via the purity + M[j,j] - 2 L[j] decomposition (which coeff_sanity
# showed is corrupted by independent truncation).
#
# Two convergence signals per k_ref:
#   (1) coefficient stability: how much does c change vs the previous k_ref?
#   (2) reference stability:   ||rho_ref(k_ref) - rho_ref(prev)||^2  -- is the
#       reference state itself still moving? Once this is tiny, k_ref is
#       effectively exact and c should stop changing.
#
# Usage: julia kref_convergence.jl
#
# Writes kref_results/kref_convergence.csv

import Distributions
import Random
include("spectrum_truncation_analysis.jl")

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

Random.seed!(1234)
n      = 4
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
t      = 3.0
lsites = liouville_siteinds(n)
ks       = [3, 8]
maxdim   = 64
cutoff   = 1e-12
order    = 2
gamma    = 0.05                 # the strongest dissipation -- worst case for convergence
gammas   = fill(gamma, n)
diss     = true

krefs = [40, 80, 160, 320, 640]

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

# evolve rho0 by k steps -> explicit MPS
function evolve_state(k, ord)
    S = get_open_step_MPO(n, J, gammas, t/k, lsites, cutoff, maxdim; order=ord, dissipation=diss)
    psi = deepcopy(rho0)
    for _ in 1:k; psi = apply(S, psi; cutoff=cutoff, maxdim=maxdim); end
    return psi
end
dist2(a, b) = (d = +(a, -1.0*b; cutoff=cutoff, maxdim=maxdim); real(inner(d, d)))

# candidate states are k_ref-INDEPENDENT: build once
println("[$(now())] building candidate states rho_k3, rho_k8 (once)..."); flush(stdout)
rho_cand = Dict(k => evolve_state(k, order) for k in ks)

# M is also k_ref-independent (Gram of candidates): build once via library
M, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                        cutoff=cutoff, maxdim=maxdim, order=order, dissipation=diss)
println("[$(now())] M (candidate Gram) = $M"); flush(stdout)

outdir = "kref_results"; mkpath(outdir)
rows = []
prev_c = nothing
prev_rho_ref = nothing

for kref in krefs
    println("\n[$(now())] --- k_ref = $kref ---"); flush(stdout)

    # build the reference state directly and its L vector (overlaps with candidates)
    t_build = @elapsed rho_ref = evolve_state(kref, order)
    purity = real(inner(rho_ref, rho_ref))

    # L_j = <<rho_ref | rho_kj>>  (direct overlaps, consistent with M's states)
    L = [real(inner(rho_ref, rho_cand[k])) for k in ks]
    c, lam = dynamic_mpf_coefficients(M, L)

    # direct single-Trotter errors (ground truth)
    E_single = [dist2(rho_ref, rho_cand[k]) for k in ks]
    # direct DMPF error: ||rho_ref - sum_j c_j rho_kj||^2
    mu = nothing
    for (j,k) in enumerate(ks)
        term = c[j] * rho_cand[k]
        mu = (mu === nothing) ? term : +(mu, term; cutoff=cutoff, maxdim=maxdim)
    end
    E_mpf_direct = dist2(rho_ref, mu)

    # convergence signals
    dc = prev_c === nothing ? NaN : maximum(abs.(c .- prev_c))
    dref = prev_rho_ref === nothing ? NaN : dist2(rho_ref, prev_rho_ref)

    push!(rows, (kref=kref, c=c, lam=lam, E_single=E_single, E_mpf=E_mpf_direct,
                 dc=dc, dref=dref, purity=purity))
    println("[$(now())] k_ref=$kref built ($(round(t_build,digits=1))s)")
    println("    c = $(round.(c,digits=4))  lambda=$(round(lam,sigdigits=4))")
    println("    E_single: k3=$(round(E_single[1],sigdigits=5)) k8=$(round(E_single[2],sigdigits=5))  " *
            (E_single[2] < E_single[1] ? "(k8<k3 OK)" : "(k8>=k3 !!)"))
    println("    E_mpf(direct) = $(round(E_mpf_direct,sigdigits=5))")
    println("    coeff change vs prev = $(dc===NaN ? "n/a" : round(dc,sigdigits=4))   " *
            "reference change vs prev = $(dref===NaN ? "n/a" : round(dref,sigdigits=4))")
    flush(stdout)

    global prev_c = c
    global prev_rho_ref = rho_ref
end

open(joinpath(outdir, "kref_convergence.csv"), "w") do io
    println(io, "kref,c_1,c_2,lambda,E_single_k3,E_single_k8,E_mpf,coeff_change,ref_change,purity")
    for r in rows
        println(io, "$(r.kref),$(r.c[1]),$(r.c[2]),$(r.lam),$(r.E_single[1]),$(r.E_single[2])," *
                    "$(r.E_mpf),$(r.dc),$(r.dref),$(r.purity)")
    end
end

println("\n==== k_ref CONVERGENCE SUMMARY (n=$n, gamma=$gamma, maxdim=$maxdim) ====")
println("k_ref | c            | coeff change | ref change  | E_mpf(direct)")
for r in rows
    println("$(rpad(r.kref,5)) | $(rpad(string(round.(r.c,digits=3)),12)) | " *
            "$(rpad(r.dc===NaN ? "n/a" : round(r.dc,sigdigits=3),12)) | " *
            "$(rpad(r.dref===NaN ? "n/a" : round(r.dref,sigdigits=3),11)) | $(round(r.E_mpf,sigdigits=4))")
end
println("\nInterpretation: once 'ref change' and 'coeff change' both drop toward")
println("zero, the reference is effectively exact and c is trustworthy. The k_ref")
println("at which that happens is the minimum safe reference for coefficient fits.")
println("\n[$(now())] wrote kref_results/kref_convergence.csv"); flush(stdout)
