# closed_moc_vs_plain.jl
# Closed-system (gamma=0) comparison of how the F_ij operator-Schmidt spectrum
# and bond dimension grow under two contraction strategies:
#
#   (i)  MOC  -- middle-out contraction: clocks kept in lockstep, alternating
#        a forward right-multiply by S_j and a backward left-multiply by S_i^dag.
#        At every synchronization point F = [S_i^dag]^m [S_j]^m is a NEAR-IDENTITY
#        (in the closed system S^dag is the exact inverse), so its spectrum
#        should decay fast and bond dimension stay small.
#
#   (ii) PLAIN -- build the whole forward evolution S_j^{k} first (a full,
#        highly-entangled evolution to time t), THEN left-multiply by S_i^dag
#        k times. The intermediate object is a full time-t propagator, so its
#        bond dimension should balloon before collapsing back at the very end.
#
# We use i=j (k_i=k_j=k) so the EXACT final F is precisely the identity; any
# bond growth is purely a property of the contraction path, making the MOC-vs-
# plain contrast clean. Set gamma=0 for the closed system.
#
# maxdim is kept generous so neither path is prematurely clipped -- we want to
# observe the TRUE spectrum, not a capped one.
#
# Outputs CSV of bond dimension per step for both methods, plus the full
# operator-Schmidt spectrum at each step (for spectrum-by-step plots).
#
# Usage: julia closed_moc_vs_plain.jl

import Distributions
import Random
include("bond_dimension_scaling_study.jl")   # full include chain

using LinearAlgebra
using Dates
using ITensors, ITensorMPS

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads()); flush(stdout)

# --- setup: CLOSED system (gamma = 0) ---
Random.seed!(1234 + 5)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gamma_val = 0.05
gammas = fill(gamma_val, n)
t      = 3.0
k      = 8
cutoff = 1e-12
maxdim = 4000                # generous: do not clip the true spectrum
lsites = liouville_siteinds(n)

# full operator-Schmidt spectrum across the middle cut (singular values,
# normalized to s_1). Mirrors operator_schmidt_rank but returns the values.
function middle_spectrum(M::MPO; svd_cutoff::Float64=1e-14)
    nn = length(M)
    mid = nn ÷ 2
    mid == 0 && return [1.0]
    Mo = deepcopy(M)
    orthogonalize!(Mo, mid)
    right_link = commonind(Mo[mid], Mo[mid+1])
    U, S, V = svd(Mo[mid], right_link; cutoff=svd_cutoff)
    nsv = ITensors.dim(commonind(U, S))
    svals = [S[i, i] for i in 1:nsv]
    sort!(svals, rev=true)
    return svals
end

dt = t / k
step_mpo     = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, maxdim; order=2, dissipation=true)
step_dag_mpo = get_open_step_MPO_dag(n, J, gammas, dt, lsites, cutoff, maxdim; order=2, dissipation=true)
Id = identity_liouville_mpo(lsites)

# storage: step -> (bond_dim, spectrum)
moc_bd   = Int[];   moc_spec   = Vector{Vector{Float64}}()
plain_bd = Int[];   plain_spec = Vector{Vector{Float64}}()

# =============================================================================
# (i) MOC: lockstep forward/backward, synchronization after each backward step
# =============================================================================
println("[$(now())] === MOC (lockstep, DISSIPATIVE) ==="); flush(stdout)
F = deepcopy(Id)
for m in 1:k
    global F
    F = right_multiply(F, step_mpo;     cutoff=cutoff, maxdim=maxdim)  # forward
    F = left_multiply(step_dag_mpo, F;  cutoff=cutoff, maxdim=maxdim)  # backward -> synchronized
    bd = middle_bond_dim(F)
    sp = middle_spectrum(F)
    push!(moc_bd, bd); push!(moc_spec, sp)
    println("[$(now())] MOC sync $m/$k: bond_dim=$bd (spectrum length $(length(sp)))"); flush(stdout)
end

# =============================================================================
# (ii) PLAIN: full forward evolution first, then all backward steps
# =============================================================================
println("[$(now())] === PLAIN (forward-then-backward, DISSIPATIVE) ==="); flush(stdout)
F = deepcopy(Id)
step = 0
# k forward steps
for m in 1:k
    global F, step
    F = right_multiply(F, step_mpo; cutoff=cutoff, maxdim=maxdim)
    step += 1
    bd = middle_bond_dim(F); sp = middle_spectrum(F)
    push!(plain_bd, bd); push!(plain_spec, sp)
    println("[$(now())] PLAIN fwd $m/$k (step $step): bond_dim=$bd"); flush(stdout)
end
# k backward steps
for m in 1:k
    global F, step
    F = left_multiply(step_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)
    step += 1
    bd = middle_bond_dim(F); sp = middle_spectrum(F)
    push!(plain_bd, bd); push!(plain_spec, sp)
    println("[$(now())] PLAIN bwd $m/$k (step $step): bond_dim=$bd"); flush(stdout)
end

# =============================================================================
# write results
# =============================================================================
outdir = "moc_results_open"; mkpath(outdir)

# bond dimension per step, both methods (plain has 2k steps, moc has k syncs)
open(joinpath(outdir, "bond_dim_compare.csv"), "w") do io
    println(io, "method,step,bond_dim")
    for (i, bd) in enumerate(moc_bd);   println(io, "moc,$i,$bd");   end
    for (i, bd) in enumerate(plain_bd); println(io, "plain,$i,$bd"); end
end

# full spectra: one row per (method, step, index, value)
open(joinpath(outdir, "spectra_compare.csv"), "w") do io
    println(io, "method,step,index,sval,log10_ratio")
    for (i, sp) in enumerate(moc_spec)
        s1 = sp[1]
        for (idx, s) in enumerate(sp); println(io, "moc,$i,$idx,$s,$(log10(s/s1))"); end
    end
    for (i, sp) in enumerate(plain_spec)
        s1 = sp[1]
        for (idx, s) in enumerate(sp); println(io, "plain,$i,$idx,$s,$(log10(s/s1))"); end
    end
end

println("\n==== SUMMARY (open system, gamma=0.05) ====")
println("MOC   max bond dim over syncs : $(maximum(moc_bd))   (final $(moc_bd[end]))")
println("PLAIN max bond dim over steps : $(maximum(plain_bd))  (peak of forward half)")
println("\nIf your hypothesis holds: MOC bond dim stays small throughout, while")
println("PLAIN peaks high at the end of the forward half (full time-t propagator)")
println("before collapsing back toward identity during the backward half.")
println("\n[$(now())] wrote moc_results/bond_dim_compare.csv, spectra_compare.csv"); flush(stdout)
