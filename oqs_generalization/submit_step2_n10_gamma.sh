#!/bin/bash
# n=10 GAMMA SWEEP -- the run that closes the two remaining gaps.
#
# WHY n=10 SPECIFICALLY, AND WHY THE PREVIOUS SWEEP COULD NOT ANSWER THIS
# ----------------------------------------------------------------------
# The gamma sweep ran at n=8, where the Liouville MPS ceiling is 4^4 = 256 and
# chi_dmpf* came out at 161-179 across every gamma. The measurable speedup is
# therefore capped at 256/165 = 1.55 BY CONSTRUCTION -- there is no room above
# chi_dmpf* for the classical curve to need. n=8 can measure the matched-chi
# ratio; it cannot measure chi*.
#
# n=10 has ceiling 4^5 = 1024, chi_dmpf* ~ 152, and gave chi_direct* ~ 403 (a
# 2.7x speedup) at gamma=0.05. It is the only size in reach where chi* is
# resolvable, so the gamma dependence of the speedup has to be measured here.
#
# n=12 is NOT reachable: ceiling 4^6 = 4096 means 4096x4096 complex SVDs and
# ~1 GB per site tensor. That caps this experimental design, and the cap should
# be stated in the paper rather than worked around.
#
# Six array tasks, one per gamma, matching the n=8 sweep exactly so the two are
# directly comparable.
#SBATCH --job-name=n10_gamma
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/n10g_%A_%a.out
#SBATCH --error=logs/n10g_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

GAMMAS=(0.01 0.02 0.05 0.10 0.20 0.40)
export GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
export TAG=n10g${GAMMA}

export TVAL=3.0
export EVO_MODE=gates CUTOFF=1e-32
export ORDER_REF=4 SPLITTING_REF=strang
export ORDER=2 SPLITTING=strang

# K0=96, not 48. At gamma=0.40 the n=8 sweep gave err_proj = 2.5e-7 against a
# reference self-error of 2.5e-8 -- only a factor of 10 of headroom, because
# dissipation damps the Trotter error as well as the entanglement. Doubling k0
# drops the reference error by 2^4 = 16 (strang:4 is genuinely fourth order,
# measured p_eff = 4.01), restoring ~2 orders of headroom at the high-gamma end.
# 96 is still divisible by lcm(4,8,16) = 16 and by lcm(3,8) = 24.
export K0=96

# 4-8-16 is the family that won; 3-8 is the control that LOST to the classical
# simulation at chi=64 at every gamma in the n=8 sweep (ratios 0.98 down to
# 0.14). Keeping both means the advantage must track the family and not the
# setup. The second family is nearly free: all families share one candidate pool.
export FAMILIES="4,8,16;3,8"

export N_LIST=10
export CHI_LIST=16,32,48,64,96,128,192,256,384,512,768
export TARGET_FACTOR=1.5
export REF_CHECK=1   # ~46 min at n=10, and worth it: err_proj shifts by two
                     # orders across the gamma range, so the reference floor has
                     # to be re-measured rather than assumed at each gamma.

echo "n=10 gamma sweep, GAMMA=$GAMMA on $(hostname)"; echo "start: $(date)"
julia step2_scaling_in_n.jl
echo "end: $(date)"

# Afterwards, per gamma:
#     DATA=. TAG=n10g0.05 python3 recompute_chi_star.py --csv
#
# EXPECTED COST: the n=10 exact reference took 1251 s at k0=48, so ~2500 s at
# k0=96, plus ~5000 s for REF_CHECK, plus 11 chi points. Budget 4-6 h per task;
# 16 h is generous. If tasks time out, drop REF_CHECK=0 for gamma <= 0.10 (the
# floor is comfortable there) and keep it for 0.20 and 0.40.
#
# OPTIONAL EXTRA SIZES: n=7 and n=9 have ceilings 4^3=64 and 4^4=256, so they are
# cheap next to n=10 and would thicken the matched-chi size trend (they cannot
# help the chi* trend -- same ceiling problem as n=8). Before adding them, check
# that vectorized_initial_state_mps handles ODD n: the call site passes
# collect(0:2:(n-1)), which has been exercised only for even n.
