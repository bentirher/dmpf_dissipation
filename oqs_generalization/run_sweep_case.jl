# run_sweep_case.jl
# Usage: julia run_sweep_case.jl <maxdim>
import Distributions
import Random
include("spectrum_truncation_analysis.jl")   # pulls in the full include chain

md = parse(Int, ARGS[1])

# --- setup, matching the earlier notebook cell exactly ---
Random.seed!(1234)
n      = 5
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(0.05, n)
t      = 3.0
lsites = liouville_siteinds(n)

# --- sweep-specific setup ---
ks         = [3, 8]
k_ref_opt  = 40
# k_ref_eval = 100
k_ref_eval = 50
maxdim_ref = 256
cutoff     = 1e-12

initially_excited = rand(["0", "1"], n)
rho0 = vectorized_initial_state_mps(lsites, initially_excited)

sweep = maxdim_truncation_sweep(
    n, J, gammas, t, ks, k_ref_opt, k_ref_eval, lsites, rho0,
    [md];                       # <-- just this one maxdim
    maxdim_ref=maxdim_ref, cutoff=cutoff,
    order=1, order_ref_opt=1, order_ref_eval=2,
    dissipation=true
)

res = sweep.results[1]

outdir = "sweep_results"
mkpath(outdir)
fname = joinpath(outdir, "maxdim_$(md).csv")

open(fname, "w") do io
    println(io, "maxdim,M_dist,E_mpf,E_trot,coeffs")
    coeffs_str = join(res.coeffs, ";")   # semicolon so it survives being one CSV field
    println(io, "$(res.maxdim),$(res.M_dist),$(res.E_mpf),$(res.E_trot),\"$coeffs_str\"")
end

println("maxdim=$md done. M_dist=$(res.M_dist) E_mpf=$(res.E_mpf)")