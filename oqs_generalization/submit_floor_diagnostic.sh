#!/bin/bash
#SBATCH --job-name=floordiag
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/floor_%j.out
#SBATCH --error=logs/floor_%j.err

# Memory 64G for the same reason as submit_delta_fix.sh: exact_delta=true carries
# chi(A)+chi(B) = 512 at n=4, above the 256 MPO ceiling, applied twice per step.
#
# Time 08:00:00 rather than 06:00:00. Part 2 runs the N-route SIX times at
# maxdim = 256 (one per maxdim_G), and comparison_study logged ~770 s for a
# single n=4 ceiling run, so Part 2 alone is ~80 min before counting Part 1's
# m_route + N-route pair. Part 0 is nearly free -- it evolves an MPS, whose
# ceiling is 4^2 = 16 at n=4, not the MPO ceiling of 256.
#
# K_REF is set explicitly and MUST equal K0: floor_diagnostic.jl asserts this,
# because otherwise the two routes target different references and the whole
# comparison mixes truncation with a systematic offset.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2

echo "floor_diagnostic on $(hostname)"
echo "start: $(date)"
julia floor_diagnostic.jl
echo "end: $(date)"

echo
echo "wrote: floor_diagnostic.csv"
echo "The decisive number is in PART 1: 'Lagrange in M,L' vs the oracle."
echo "  ~1e-4  -> the floor belongs to c_exact, not to the N-route"
echo "  ~1e-14 -> the floor is in the N-route; look at step_defect_MPO next"
