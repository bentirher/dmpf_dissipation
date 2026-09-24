#!/bin/bash
# RE-MEASUREMENT UNDER THE HONEST PROTOCOL.
#
# The earlier n=10 gamma sweep applied the gates at the CEILING and truncated to
# chi once per step, so the reported chi was never the resource actually used: a
# two-site gate raises a Liouville bond to chi*d with d=4, so from chi=16 at
# n=10 the middle bond reaches the 1024 ceiling within three gate layers, and a
# fourth-order step has about twenty-five of them. PROTOCOL=capped caps the bond
# dimension at chi at every gate, as a production TEBD does.
#
# PREDICTION, worth writing down before looking: the number of truncation events
# per step goes from 1 to (layers per step). For the reference that is ~25 x 96 =
# 2400 instead of 96; for the candidates ~5 x 28 = 140 instead of 28. The
# classical curve IS the reference, so it should degrade more than the hybrid,
# whose truncation reaches the answer only through dc and therefore at second
# order. If the advantage does NOT grow under the honest protocol, something in
# the argument is wrong.
#
# Same six gamma as before so the two protocols can be compared directly.
#SBATCH --job-name=n10_capped
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/capped_%A_%a.out
#SBATCH --error=logs/capped_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

GAMMAS=(0.01 0.02 0.05 0.10 0.20 0.40)
export GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
export TAG=cap${GAMMA}

export TVAL=3.0
export EVO_MODE=gates CUTOFF=1e-32
export PROTOCOL=capped          # <-- the whole point of this run
export ORDER_REF=4 SPLITTING_REF=strang K0=96
export ORDER=2 SPLITTING=strang
export FAMILIES="4,8,16;3,8"
export N_LIST=10
export CHI_LIST=16,32,48,64,96,128,192,256,384,512,768
export TARGET_FACTOR=1.5
export REF_CHECK=1

echo "capped-protocol n=10 sweep, GAMMA=$GAMMA on $(hostname)"; echo "start: $(date)"
julia step2_scaling_in_n.jl
echo "end: $(date)"

# NOTE ON COST: the exact pass still runs at the ceiling (chi=1024) because it
# defines the target and supplies the device values -- that is a measurement
# cost, charged to neither route, and it is what caps this study at n<=10. The
# per-chi sweep is now genuinely chi-limited. Expect these jobs to be FASTER
# than the previous ones at small chi, since those were secretly running at 1024.
#
# eps_chi is NaN under this protocol (the dissipator gates are not norm
# preserving, so per-gate discarded weight is not recoverable by the norm trick).
# consolidate.py falls back to fitting against chi.
