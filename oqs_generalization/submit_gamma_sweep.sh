#!/bin/bash
# GAMMA SWEEP -- the robustness axis, and the one that can still kill the claim.
#
# Everything so far is at a single gamma = 0.05. The failure mode is specific and
# physical: dissipation CAPS operator entanglement. At large enough gamma*t the
# Liouville MPS stops growing with n, the exact state no longer saturates its
# ceiling, a classical simulation stops getting harder -- and the advantage
# evaporates. "Works only at weak dissipation" is a far narrower claim than what
# is currently on the table, so the boundary has to be located, not assumed.
#
# THE DIAGNOSTIC TO WATCH is chi_rho_exact against chi_ceiling. At gamma=0.05 it
# saturated at every n (16, 64, 256, 1024). The gamma at which it falls away is
# the edge of the regime where any of this matters.
#
# n=8 for the full range: the exact reference there took 117 s versus 1251 s at
# n=10, so the whole sweep costs about one n=10 point. Follow up at n=10 only at
# whichever gamma turn out to be interesting.
#
# Array indices map to gamma = 0.01, 0.02, 0.05, 0.10, 0.20, 0.40.
#SBATCH --job-name=gamma_sweep
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/gamma_%A_%a.out
#SBATCH --error=logs/gamma_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

GAMMAS=(0.01 0.02 0.05 0.10 0.20 0.40)
export GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
export TAG=g${GAMMA}

export TVAL=3.0
export EVO_MODE=gates CUTOFF=1e-32
export ORDER_REF=4 SPLITTING_REF=strang K0=48
export ORDER=2 SPLITTING=strang

# 4-8-16 is the family that won: err_proj 1.2e-5 at n=10, cond(N) 1.3e6, and the
# only one beating both the classical simulation AND the free baseline at every
# chi >= 64. 3-8 is carried as the control -- it LOST to classical at n=8
# (vs_classical 0.26 at chi=64), so if the advantage is real it must reappear for
# one family and not the other at every gamma.
export FAMILIES="4,8,16;3,8"

export N_LIST=8
export CHI_LIST=16,32,48,64,96,128,192,256
export TARGET_FACTOR=1.5
export REF_CHECK=1   # err_proj shifts with gamma, so the reference floor must be
                     # re-measured at each one rather than reused. Cheap at n=8.

echo "gamma sweep, GAMMA=$GAMMA on $(hostname)"; echo "start: $(date)"
julia step2_scaling_in_n.jl
echo "end: $(date)"

# Afterwards, per gamma:
#     DATA=. TAG=g0.10 python3 recompute_chi_star.py --csv
#
# WHAT WOULD NARROW THE CLAIM: chi_rho_exact dropping below the ceiling, or
# vs_free falling toward 1 at large gamma. Either means the honest statement is
# "in the weak-dissipation regime", which is still the device-relevant one but
# must be stated.
#
# WHAT WOULD BROADEN IT: the advantage holding to gamma=0.2-0.4. Then it covers
# the regime where the closed-system MOC argument provably fails, which is the
# whole motivation for the open-system generalisation.
