#!/bin/bash
# RE-ANALYSIS of the Step 2 pass-1 output. No simulation -- pure post-processing
# of step2_scaling.csv, seconds of CPU. Submitted as a job only so it lands in
# logs/ with everything else; it runs identically on the login node with
#
#     DATA=. python3 recompute_chi_star.py --csv
#
# WHY IT IS NEEDED
# ----------------
# Pass 1 established the result -- at matched chi, DMPF beat direct classical
# simulation by 1-3 orders of magnitude at every chi below the ceiling, and the
# margin grew with n (chi=16: 58x at n=6, 88x at n=8, 509x at n=10). But the
# printed summary had three gaps:
#
#  1. chi_direct* was corrupted. A point with err == 0 is the ceiling run scored
#     against ITSELF, so log(0) = -Inf drove the interpolation weight to zero and
#     returned the lower bracket. Every chi_direct* read as 128.0, and the whole
#     `speedup` column derived from it is meaningless. Fixed here.
#
#  2. err_best_single -- the answer with ZERO classical computation -- was in the
#     CSV but not the table. It is the bar that matters: with sum(c)=1 and |c|
#     bounded, any normalised combination of good candidates is already good, so
#     DMPF beating the classical simulation is not sufficient. It must beat this.
#
#  3. corr, max|dc| vs sqrt(dc'N dc) and dc_frac_null dropped out of the printed
#     table when the per-family columns went in. Those are the MECHANISM, and
#     max|dc|/sqrt(dc'N dc) is the number that rebuts main.pdf directly.
#
# Also cross-checks the advantage across observables, so a sign-flip artifact on
# one observable cannot masquerade as a result.
#
# Output: step2_matched.csv, step2_chistar.csv, step2_mechanism.csv
#SBATCH --job-name=step2_reanalysis
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/reanalysis_%j.out
#SBATCH --error=logs/reanalysis_%j.err
mkdir -p logs

# Standard library only -- no numpy, no matplotlib -- so any python3 works and
# no module needs loading. Uncomment if the bare interpreter is not on PATH.
# module load Python/3.11.3-GCCcore-12.3.0

export DATA=.
# export TAG=dense     # set when re-analysing a tagged run
export CHI_MIN=16      # chi=4 and chi=8 are noise: the truncated state is not
                       # approximating anything there, so those ratios compare
                       # against garbage rather than against a competitor

echo "step2 re-analysis on $(hostname)"; echo "start: $(date)"
python3 recompute_chi_star.py --csv
echo "end: $(date)"
