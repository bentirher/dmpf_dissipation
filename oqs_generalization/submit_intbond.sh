#!/bin/bash
#SBATCH --job-name=intbond
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/intbond_%j.out
#SBATCH --error=logs/intbond_%j.err

# Splits the N-route's error into ACCUMULATED (recursion) and REPRESENTATIONAL
# (final object), and tests whether an error-limited truncation schedule beats a
# rank-limited one at equal bond dimension.
#
# HONEST FRAMING: arm A is a diagnostic, not a speed-up. Running the recursion at
# maxdim_int=256 costs what 256 costs no matter what you compress to afterwards.
# What it buys is knowing WHERE the 86x gap from check 2 lives. Arm B is the arm
# that could actually reduce cost.
#
# COST. Arm A is 14 (maxdim_int, maxdim_fin) pairs x 3 Phi builds each. Cost is
# set by maxdim_int, and at n=6 a single N-route point cost 293/825/2392/7505/
# 21406 s at chi = 48/64/96/128/192. The maxdim_int=256 block alone is roughly
# 5 x 30000 s. Arm B adds 3 rank-limited plus 4 error-limited runs, and the
# error-limited ones have maxdim at the CEILING, so their cost depends entirely
# on where the cutoff lets the bond dimension settle -- genuinely unpredictable.
#
# THEREFORE: start with ARM=A and N_QUBITS=4. n=4 is cheap (the whole n=4 sweep
# was 30 min), the 86x gap should still show as the chi=192 vs 256 jump that
# motivated this, and it tells you whether arm B is worth an expensive n=6 slot.
# Only move to n=6 once arm A has been read at n=4.
#
# Everything writes its CSV after every point, so a timeout costs only the tail.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2
export ARM=both

echo "intermediate_bond on $(hostname)"; echo "start: $(date)"
julia intermediate_bond.jl || { echo "FAILED"; exit 1; }
echo "end: $(date)"
echo
echo "ARM A -- read dc down each maxdim_int block:"
echo "  flat in maxdim_fin  -> error is ACCUMULATED; final compression is free,"
echo "                         but there is no saving: the recursion needs its"
echo "                         bond dimension."
echo "  falls with maxdim_fin -> the recursion can run cheap. That WOULD be a win."
echo "ARM B -- compare rows at EQUAL 'chi built'. An error-limited run reaching"
echo "  the same chi with a smaller dc means the schedule was the problem."
