#!/bin/bash
# SCALABILITY: does the method run, and do the coefficients converge, at sizes
# where no exact state exists? Requires the honest chi-capped protocol.
#
# No exact reference is used or needed. Convergence is assessed by
# self-consistency in chi, and the coefficient error is converted into an
# achieved error through the exact identity E(c*+dc) - E(c*) = dc^T N dc, which
# needs only N. That is the practical payoff of the error formulation: the
# figure of merit is computable without knowing the answer.
#
# START SMALL. n=12 and 16 first to calibrate wall time, then extend.
#SBATCH --job-name=step3_scale
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step3_%j.out
#SBATCH --error=logs/step3_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export GAMMA=0.05 TVAL=3.0
export EVO_MODE=gates CUTOFF=1e-32
export PROTOCOL=capped
export ORDER_REF=4 SPLITTING_REF=strang K0=96
export ORDER=2 SPLITTING=strang KS=4,8,16

export N_LIST=12,16,20
export CHI_LIST=32,64,96,128,192,256
export TARGET=1e-4
export TAG=s1

echo "step3 scalability on $(hostname)"; echo "start: $(date)"
julia step3_scalability.jl
echo "end: $(date)"

# READ chi_actual_top FIRST. If it equals the largest chi on the grid, the cap
# was still binding and the "converged" reference is not converged -- extend
# CHI_LIST before believing any self-convergence number.
#
# THEN: N_LIST=24,32 with CHI_LIST=64,128,192,256,384 once wall times are known.
# n=32 at chi=256 is roughly 1e9 flops per gate, ~400 gates per step, 96 steps:
# order 4e13 flops for the reference, so a few hours. n=50 is the target and
# should be reachable at chi<=256 if the times scale as expected.
