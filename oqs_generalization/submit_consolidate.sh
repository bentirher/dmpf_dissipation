#!/bin/bash
# CONSOLIDATED ANALYSIS of every tagged Step 2 run. Post-processing only --
# seconds of CPU, standard library Python. Runs equally well on the login node:
#     python3 consolidate.py --csv
#
# Reads step2_scaling_g*.csv (n=8) and step2_scaling_n10g*.csv (n=10) and fixes
# the two problems the per-run summaries could not:
#
#  1. Sign-cancellation noise: averages |error| over every single-site Z_m
#     (Z_MAE), using per-site rows already present in every CSV since the dense
#     run. On Z_mid alone err_dmpf dips below its own floor, and err_proj at
#     gamma=0.20 breaks trend by an order of magnitude.
#
#  2. Confounded targets: scores chi* at FIXED absolute accuracies (1e-4, 3e-5,
#     1e-5) instead of 1.5 x err_proj, which differed by up to 25x between n=8
#     and n=10 at the same gamma.
#
# Also uses a SUSTAINED crossing for DMPF chi* (smallest chi from which the
# error stays below target), so a lucky dip cannot masquerade as convergence.
#
# Output: consolidated_chistar.csv, plus printed tables.
#SBATCH --job-name=consolidate
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/consolidate_%j.out
#SBATCH --error=logs/consolidate_%j.err
mkdir -p logs
# module load Python/3.11.3-GCCcore-12.3.0   # uncomment if python3 is not on PATH

export DATA=.
export FAMILY=4-8-16
export TARGETS=1e-4,3e-5,1e-5
export CHI_MIN=16

echo "consolidate on $(hostname)"; echo "start: $(date)"
python3 consolidate.py --csv
echo "end: $(date)"
