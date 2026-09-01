# =============================================================================
# entanglement_growth_study.jl
#
# Locates the classically expensive corner of (n, gamma, t) for the vectorized
# MPS/TEBD simulation of the Heisenberg + amplitude damping chain, so that the
# quantum-circuit experiment (Hamiltonian + ancilla-mediated amplitude damping)
# can be sited somewhere defensible.
#
# For each (n, gamma) it evolves |rho(t)>> and records S_op(t) and the bond
# dimension actually required. Each gamma also gets a gamma=0 closed-system
# control run at the same n, so the "does dissipation help or hurt" claim is
# made against a like-for-like baseline rather than against intuition.
#
# WHAT THE OUTPUT SHOULD LOOK LIKE, AND WHAT IT MEANS
#
# Amplitude damping alone (sigma^- only, no absorption) has the trivial product
# steady state |0...0>, so S_op must return to zero at long times. The curve is
# a barrier: rise, peak, fall. The peak sits near t ~ 1/gamma and gets TALLER as
# gamma gets SMALLER. So:
#
#   - the hard regime is small gamma at t ~ 1/gamma, not long times;
#   - "evolve longer and it gets harder" is false for this model;
#   - the circuit experiment should be sited AT the peak, which is the only
#     place the classical baseline is genuinely strained.
#
# If instead chi_req stays flat while S_op falls, that is the Vilchez-Estevez
# effect and is the more interesting result of the two -- it is the separation
# between "the state is simple" and "getting there was not".
#
# Environment: N_LIST, GAMMA, TMAX, NT, DT, ORDER, MAXDIM, CUTOFF, JCOUP,
#              DISORDER, SEED, TAG
# =============================================================================
using LinearAlgebra, Printf, Dates
import Random, Distributions

getenv(k, d) = get(ENV, k, string(d))

# --- breadcrumbs, written BEFORE the heavy include -------------------------
# The output directory and a manifest are created first so that the contents of
# the directory identify how far the job got:
#   directory absent          -> the bash script never ran (module load, path)
#   directory empty           -> Julia died at load time (missing include, ITensors)
#   manifest.csv only         -> Julia loaded but the first evolution failed
#   manifest + *_timeseries   -> some runs finished; check which n are present
# Previously an empty directory was ambiguous between the first two, because the
# bash `mkdir -p` and Julia's `mkpath` both create it.
outdir = getenv("OUTDIR", "results")
mkpath(outdir)

println("[stage] outdir=$(abspath(outdir))"); flush(stdout)
println("[stage] loading vectorized_evolution.jl ..."); flush(stdout)
include(joinpath(@__DIR__, "vectorized_evolution.jl"))
println("[stage] load OK"); flush(stdout)

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
n_list   = parse.(Int, split(getenv("N_LIST", "8,12,16,20"), ","))
gamma    = parse(Float64, getenv("GAMMA",  0.05))
tmax     = parse(Float64, getenv("TMAX",   40.0))
nt       = parse(Int,     getenv("NT",     40))
dt       = parse(Float64, getenv("DT",     0.02))
order    = parse(Int,     getenv("ORDER",  2))
maxdim   = parse(Int,     getenv("MAXDIM", 512))
cutoff   = parse(Float64, getenv("CUTOFF", 1e-12))
jcoup    = parse(Float64, getenv("JCOUP",  0.5))
disorder = parse(Bool,    getenv("DISORDER", false))
seed     = parse(Int,     getenv("SEED",   1234))
tag      = getenv("TAG", "")
sfx      = isempty(tag) ? "" : "_" * tag
gtag     = replace(@sprintf("%.3f", gamma), "." => "p")

# Time grid. tmax defaults to ~2/gamma so the barrier peak is inside the window
# rather than just off the right-hand edge -- the mistake that makes every such
# plot look like monotone growth.
times = collect(range(tmax / nt, tmax; length=nt))

@printf("=== entanglement growth study ===\n")
@printf("n_list=%s gamma=%.4f tmax=%.2f (1/gamma=%.2f) nt=%d dt=%.4g\n",
        string(n_list), gamma, tmax, gamma > 0 ? 1/gamma : Inf, nt, dt)
@printf("order=%d maxdim=%d cutoff=%.1e J=%s\n\n",
        order, maxdim, cutoff, disorder ? "U[1/4,3/4] (seed $seed)" : @sprintf("%.3f uniform", jcoup))
flush(stdout)

outdir = outdir   # already created and announced above

# Manifest: every parameter of this job, written before any physics runs, so
# the directory is never ambiguously empty again.
manifest = joinpath(outdir, "manifest.csv")
open(manifest, "w") do io
    println(io, "key,value")
    for (k, v) in [("n_list", join(n_list, " ")), ("gamma", gamma), ("tmax", tmax),
                   ("nt", nt), ("dt", dt), ("order", order), ("maxdim", maxdim),
                   ("cutoff", cutoff), ("jcoup", jcoup), ("disorder", disorder),
                   ("seed", seed), ("tag", tag), ("host", gethostname()),
                   ("started", Dates.now()), ("cwd", pwd())]
        println(io, "$k,$v")
    end
end
println("[stage] wrote $manifest"); flush(stdout)

# -----------------------------------------------------------------------------
# Everything below runs inside a function ON PURPOSE.
#
# In a Julia SCRIPT (as opposed to the REPL), assigning to an existing global
# from inside a top-level `for` loop does NOT touch the global: the assignment
# creates a new local, and reading it before that assignment raises
# `UndefVarError: ... not defined in local scope`. That is exactly the failure
# that killed the first version of this file. Wrapping the sweep in a function
# removes the soft-scope rule entirely -- `first_run` is now an ordinary local,
# and every future accumulator added here is safe by construction.
#
# It is also faster: top-level globals are type-unstable, so any loop written at
# top level in Julia pays a dispatch cost on every iteration.
# -----------------------------------------------------------------------------

function run_sweep(; n_list, gamma, times, dt, order, maxdim, cutoff,
                     jcoup, disorder, seed, outdir, gtag, sfx)
    all_rows  = String[]   # one row per (run, time)
    prof_rows = String[]   # one row per (run, time, bond)
    first_run = true

    # Reseeded per call so the disorder realization for a given n is
    # reproducible and the n=8 couplings are a prefix of the n=20 ones.
    function coupling(n)
        disorder || return jcoup
        Random.seed!(seed)
        return rand(Distributions.Uniform(1/4, 3/4), n - 1)
    end

    for n in n_list
        J = coupling(n)
        for (diss, name) in ((true, "open"), (false, "closed"))
            @printf("\n--- n=%d  %s ---\n", n, name); flush(stdout)
            res = evolve_vectorized(n, J, gamma, times;
                                    dissipation = diss,
                                    dt = dt, order = order,
                                    cutoff = cutoff, maxdim = maxdim,
                                    initial = :neel,
                                    tols = [1e-6, 1e-10],
                                    verbose = true)
            lbl = "$(name)_n$(n)"
            # Header on the first run only; later runs append bare rows.
            append!(all_rows,  split(chomp(rows_to_csv(res;    label=lbl, header=first_run)), "\n"))
            append!(prof_rows, split(chomp(profile_to_csv(res; label=lbl, header=first_run)), "\n"))
            first_run = false

            # Per-run files written immediately, so a job that dies at n=20 still
            # leaves usable n=8,12,16 data behind rather than nothing.
            save_run(res; dir=outdir, label="g$(gtag)$(sfx)_$(lbl)")
        end
    end

    ts_file   = joinpath(outdir, "entanglement_growth_g$(gtag)$(sfx).csv")
    prof_file = joinpath(outdir, "entanglement_growth_g$(gtag)$(sfx)_profile.csv")
    write(ts_file,   join(all_rows,  "\n") * "\n")
    write(prof_file, join(prof_rows, "\n") * "\n")
    @printf("\nwrote %s (%d data rows)\n", ts_file,   length(all_rows)  - 1)
    @printf("wrote %s (%d data rows)\n",   prof_file, length(prof_rows) - 1)
    return (timeseries=ts_file, profile=prof_file, coupling=coupling)
end

# -----------------------------------------------------------------------------
# Convergence check on the largest n, around the barrier peak. Do not skip this:
# a chi_req curve from an unconverged run is a picture of your own maxdim.
# -----------------------------------------------------------------------------
function run_convergence(; n_list, gamma, times, tmax, dt, order, maxdim, cutoff,
                           coupling, outdir, gtag, sfx)
    n_big = maximum(n_list)
    peak_window = filter(t -> t <= min(tmax, 1.5 / max(gamma, 1e-9)), times)
    isempty(peak_window) && (peak_window = times)
    stride = max(1, length(peak_window) ÷ 4)
    check_times = collect(peak_window[stride:stride:end])
    isempty(check_times) && (check_times = [times[end]])

    ladder = filter(<=(maxdim), [64, 128, 256, 512, 1024])
    length(ladder) < 2 && (ladder = [max(8, maxdim ÷ 2), maxdim])

    println("\n=== maxdim convergence, n=$n_big ===")
    conv = maxdim_convergence(n_big, coupling(n_big), gamma, check_times, ladder;
                              dt=dt, order=order, cutoff=cutoff, initial=:neel,
                              tols=[1e-6, 1e-10], verbose=true)

    conv_file = joinpath(outdir, "entanglement_growth_g$(gtag)$(sfx)_convergence.csv")
    write_convergence_csv(conv_file, conv; label_prefix="conv_n$(n_big)")
    @printf("wrote %s\n", conv_file)

    lo, hi = conv.runs[1], conv.runs[end]
    nconv = count(i -> abs(lo.S_op_mid[i] - hi.S_op_mid[i]) < 1e-3, eachindex(lo.t))
    @printf("  (%d/%d times already converged at the SMALLEST maxdim=%d)\n",
            nconv, length(lo.t), ladder[1])
    return conv_file
end

# -----------------------------------------------------------------------------

sweep = run_sweep(; n_list=n_list, gamma=gamma, times=times, dt=dt, order=order,
                    maxdim=maxdim, cutoff=cutoff, jcoup=jcoup, disorder=disorder,
                    seed=seed, outdir=outdir, gtag=gtag, sfx=sfx)

if get(ENV, "SKIP_CONVERGENCE", "false") != "true"
    run_convergence(; n_list=n_list, gamma=gamma, times=times, tmax=tmax, dt=dt,
                      order=order, maxdim=maxdim, cutoff=cutoff,
                      coupling=sweep.coupling, outdir=outdir, gtag=gtag, sfx=sfx)
end

println("\n[stage] done"); flush(stdout)
