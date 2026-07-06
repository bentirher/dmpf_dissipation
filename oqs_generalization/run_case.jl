# run_case.jl
# Usage: julia run_case.jl <n> <gamma> <maxdim>
include("bond_dimension_scaling_study.jl")   # pulls in the full include chain

n      = parse(Int, ARGS[1])
gamma  = parse(Float64, ARGS[2])
maxdim = parse(Int, ARGS[3])

t      = 3.0
k      = 8
cutoff = 1e-10

Random.seed!(1234 + n)              # same seeding convention as n_scaling_study
J      = rand(Uniform(1/4, 3/4), n - 1)
lsites = liouville_siteinds(n)
gammas = fill(gamma, n)

res = track_Fii_bond_dimension(
    n, J, gammas, t, k, lsites, cutoff, maxdim;
    order=2, track_exact_rank=false, dissipation=(gamma > 0.0)
)

outdir = "results"
mkpath(outdir)
fname  = joinpath(outdir, "n$(n)_gamma$(gamma).csv")

open(fname, "w") do io
    println(io, "layer,bond_dim")
    for (layer, bd) in enumerate(res.bond_dim)
        println(io, "$layer,$bd")
    end
end

capped = any(res.bond_dim .>= maxdim)
println("n=$n gamma=$gamma done. final bond_dim=$(res.bond_dim[end]) capped=$capped theoretical_max=$(theoretical_max_bond_dim(n))")