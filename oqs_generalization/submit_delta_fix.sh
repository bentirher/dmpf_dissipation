#!/bin/bash
#SBATCH --job-name=deltafix
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dfix_%j.out
#SBATCH --error=logs/dfix_%j.err

# Memory raised to 64G: the exact delta carries chi(A)+chi(B) = 512 at n=4,
# above the 256 ceiling, and it is applied twice per step.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2 STAGES=12

echo "delta_fix_verify on $(hostname)"
echo "start: $(date)"
julia delta_fix_verify.jl
echo "end: $(date)"
