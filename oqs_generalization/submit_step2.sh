#!/bin/bash
# STEP 2 -- sweep SYSTEM SIZE at fixed bond dimension. This is the experiment
# that can actually be won.
#
# Step 1 swept chi at fixed n=8, where the ceiling is 4^4 = 256. At chi=256 the
# classical route is EXACT, so it wins by any margin you like and the crossover
# near chi~50 was guaranteed by the ceiling rather than discovered. At any n
# where an exact reference is affordable, so is the exact answer.
#
# Sweeping n instead: err_direct should degrade fast (the classical route needs
# the state's full operator entanglement, which grows with n), while err_dmpf
# stays pinned near err_proj (a Trotter error, nearly n-independent). The
# headline is chi_direct*/chi_dmpf* -- the bond dimension each route needs for a
# common accuracy target -- GROWING with n.
#
# Output: step2_scaling.csv, step2_summary.csv
#SBATCH --job-name=step2_n
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export GAMMA=0.05 TVAL=3.0
export EVO_MODE=gates CUTOFF=1e-32

# ---- from Step 0 (validated: p_eff = 4.01, E_mpf error 3e-10 at k0=768) -----
export ORDER_REF=4 SPLITTING_REF=strang
export K0=48        # 48, not 24: divisible by lcm of every sensible family, and
                    # strang:4 error there is ~2e-7, negligible against err_proj

# ---- from the FAMILY SCAN -- replace with whatever it recommends ------------
export ORDER=2 SPLITTING=strang
export KS=4,6,8,12

# ---- the sweep --------------------------------------------------------------
# n=10 needs an exact reference at chi = 4^5 = 1024; the central SVDs are
# 1024x1024 complex, so expect tens of minutes. n=12 (ceiling 4096) is out of
# reach, which caps this design at four points.
export N_LIST=4,6,8,10
export CHI_LIST=4,8,16,32,64,128
export TARGET_FACTOR=1.5   # accuracy target = 1.5 x err_proj(n). err_proj is
                           # DMPF's floor, so a target below it would be
                           # unreachable by construction.

echo "step2_scaling_in_n on $(hostname)"; echo "start: $(date)"
julia step2_scaling_in_n.jl
echo "end: $(date)"

# ---------------------------------------------------------------------------
# IF n=10 RUNS OUT OF TIME OR MEMORY: drop to N_LIST=4,6,8 and submit n=10 as a
# separate job (N_LIST=10 TAG=n10), then concatenate the CSVs. Three points of n
# is thin for a trend; four is the minimum worth plotting.
#
# THE FAILURE MODE TO WATCH: chi_rho_exact falling away from chi_ceiling. That
# means dissipation has capped the operator entanglement, the classical route
# stops getting harder with n, and the answer is no. At gamma=0.05, t=3, n=8 the
# state still saturated (chi_rho = 256), so we are not there yet -- but gamma is
# the axis that would take us there, and it is the next sweep either way.
# ---------------------------------------------------------------------------
