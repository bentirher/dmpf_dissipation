#!/bin/bash
#SBATCH --job-name=bd_small
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --array=8-8
#SBATCH --output=logs/small_%A_%a.out
#SBATCH --error=logs/small_%A_%a.err

mkdir -p logs results

ns=(3 4 5)
gammas=(0.01 0.05 0.1 0.2)

n_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))
g_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))

n=${ns[$n_idx]}
gamma=${gammas[$g_idx]}

module load Julia/1.11.6-linux-x86_64     # replace with the exact name module spider gives you

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Task $SLURM_ARRAY_TASK_ID: n=$n gamma=$gamma on $(hostname)"
julia run_case.jl $n $gamma 4000