# gram_convergence.jl
# Computes the 2x2 open Gram matrix M at increasing maxdim values and reports
# each entry + build time, so we can find the cheapest maxdim at which M is
# already converged (and thus safe to use as maxdim_ref in the sweep).
#
# Runs maxdims in ASCENDING order and flushes after each, so the cheap points
# are logged even if the maxdim=256 build is slow or gets killed.
#
# Usage: julia gram_convergence.jl

import Distributions
import Random
include("spectrum_truncation_analysis.jl")   # full include chain

using LinearAlgebra
using Dates
using Printf

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
println("BLAS threads: ", BLAS.get_num_threads(), " / Julia CPU threads: ", Sys.CPU_THREADS)
flush(stdout)

# --- setup: identical to run_sweep_case.jl / time_ref.jl ---
Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(0.05, n)
t      = 3.0
lsites = liouville_siteinds(n)
ks     = [3, 8]
cutoff = 1e-12
initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

maxdims = [64, 96, 128, 192, 256]

# store M from each maxdim so we can print convergence deltas vs the previous one
prev_M = nothing

for md in maxdims
    println("[$(now())] START  M at maxdim=$md ..."); flush(stdout)
    local M
    dt = @elapsed begin
        M, _ = open_gram_matrix(n, J, gammas, t, ks, lsites, rho0;
                                cutoff=cutoff, maxdim=md, order=2, dissipation=true)
    end
    println("[$(now())] DONE   M at maxdim=$md  ($(round(dt, digits=1))s)")

    # print the 2x2 matrix entries
    @printf("           M = [ %.10e  %.10e ]\n", M[1,1], M[1,2])
    @printf("               [ %.10e  %.10e ]\n", M[2,1], M[2,2])

    # print max abs change vs previous maxdim (convergence signal)
    if prev_M !== nothing
        delta = maximum(abs.(M .- prev_M))
        reldelta = delta / maximum(abs.(M))
        @printf("           max|dM| vs maxdim=%d: %.3e   (relative: %.3e)\n",
                maxdims[findfirst(==(md), maxdims)-1], delta, reldelta)
    end
    global prev_M = M
    flush(stdout)
end

println("[$(now())] CONVERGENCE TEST COMPLETE"); flush(stdout)
