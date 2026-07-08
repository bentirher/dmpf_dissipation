# run_case.jl
# Usage: julia run_case.jl <n> <gamma> <maxdim>
include("bond_dimension_scaling_study.jl")   # pulls in the full include chain

using LinearAlgebra
using Dates

# --- report whether BLAS is actually using the cores SLURM gave us ---
println("BLAS threads: ", BLAS.get_num_threads(), " / Julia CPU threads: ", Sys.CPU_THREADS)
flush(stdout)

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

# =============================================================================
# Instrumented, inline copy of track_Fii_bond_dimension.
#
# This reproduces exactly what track_Fii_bond_dimension does (order=2,
# track_exact_rank=false), but with timing + flushed prints around the one-time
# MPO construction and each Trotter layer, so a short test run reveals where the
# wall-clock time actually goes. All primitives used here are already in scope
# via the include chain above -- no project files are edited.
# =============================================================================

dt = t / k
 
println("[$(now())] building step MPOs (n=$n, gamma=$gamma, maxdim=$maxdim)...")
flush(stdout)
 
t_build = @elapsed begin
    step_mpo     = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, maxdim;
                                     order=2, dissipation=(gamma > 0.0))
    step_dag_mpo = get_open_step_MPO_dag(n, J, gammas, dt, lsites, cutoff, maxdim;
                                         order=2, dissipation=(gamma > 0.0))
end
 
println("[$(now())] step MPOs built in $(round(t_build, digits=1))s")
flush(stdout)
 
F = identity_liouville_mpo(lsites)
bond_dims = Int[]
 
for layer in 1:k
    t_layer = @elapsed begin
        F = right_multiply(F, step_mpo; cutoff=cutoff, maxdim=maxdim)
        F = left_multiply(step_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)
    end
    bd = middle_bond_dim(F)
    push!(bond_dims, bd)
    println("[$(now())] layer $layer/$k: bond_dim=$bd, took $(round(t_layer, digits=1))s")
    flush(stdout)
end
 
res = (layer=collect(1:k), bond_dim=bond_dims, F_final=F)

# =============================================================================
# Write results (identical output format to the original run_case.jl)
# =============================================================================

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
flush(stdout)