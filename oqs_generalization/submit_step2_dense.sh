#!/bin/bash
# STEP 2, PASS 2 -- fill the gap and beat the noise.
#
# Pass 1 established the result: at matched chi, DMPF beats direct classical
# simulation by 1-3 orders of magnitude at every chi below the ceiling, and the
# margin grows with n (chi=16: 58x at n=6, 88x at n=8, 509x at n=10).
#
# Three things stopped it being quotable:
#
#  1. chi_direct* was corrupted by log(0) at the ceiling point, so every value
#     read as the last grid point. Fixed in chi_star; recompute the OLD run with
#     `DATA=. python3 recompute_chi_star.py --csv` -- no cluster time needed.
#
#  2. The grid jumped 128 -> 1024 at n=10, so where classical overtakes DMPF is
#     unknown, and that gap spans the whole interesting region. CHI_LIST below
#     fills it.
#
#  3. Single-observable errors pass through zero as the sign flips, making
#     err_direct non-monotone by an order of magnitude between adjacent chi.
#     The patched script now averages |error| over all n single-site Z (Z_MAE),
#     which damps this for free.
#
# Only n=8 and n=10 are rerun: n=4 and n=6 hit the ceiling too early to add
# anything, and their pass-1 rows are already in step2_scaling.csv.
#SBATCH --job-name=step2_dense
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step2d_%j.out
#SBATCH --error=logs/step2d_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export GAMMA=0.05 TVAL=3.0
export EVO_MODE=gates CUTOFF=1e-32
export ORDER_REF=4 SPLITTING_REF=strang K0=48
export ORDER=2 SPLITTING=strang

# 4-6-8-12 is dropped: its err_proj at n=8 (4.0e-7) is only 2x above the
# reference self-error (2.1e-7), too close to the floor to trust. The two kept
# families sit comfortably in the resolvable window.
export FAMILIES="4,8,16;2,4,8,16"

export N_LIST=8,10
export CHI_LIST=16,32,48,64,96,128,192,256,384,512
export TARGET_FACTOR=1.5
export REF_CHECK=0   # ref_selferr was measured in pass 1 (2.1e-7 at n=8,
                     # 4.0e-7 at n=10) and costs a full extra reference
                     # evolution -- 2750 s at n=10. Reuse the pass-1 numbers.
export TAG=dense

echo "step2 dense on $(hostname)"; echo "start: $(date)"
julia step2_scaling_in_n.jl
echo "end: $(date)"

# The n=10 exact reference alone took 1388 s in pass 1, and the chi sweep now
# has 10 points instead of 7. If this runs long, split: N_LIST=8 TAG=dense8 and
# N_LIST=10 TAG=dense10 as separate jobs, then concatenate the CSVs.
