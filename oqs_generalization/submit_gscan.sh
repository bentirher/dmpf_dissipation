#!/bin/bash
#SBATCH --job-name=gscan
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gscan_%j.out
#SBATCH --error=logs/gscan_%j.err

# Joint maxdim x maxdim_G scan.  ->  cmp_gscan.csv
#
# WHY NOT JUST EDIT submit_comparison_v2.sh:
#   that script MOVES cmp_*.csv to stale_cmp_*.csv on entry. Running it a second
#   time would move the GOOD oracle-referenced CSVs from job 6977175 into
#   stale_*, overwriting the genuinely stale ones. This version COPIES into a
#   timestamped directory instead, so nothing is ever lost.
#
# TIME BUDGET. Job 6977175 took 40 min wall (not the 10 h I allowed). Breakdown:
#   main sweep sum(t_M)+sum(t_N) = 205 + 1614 s = 30 min
#   spectra_study 79 + 299 s     =  6 min
# GSCAN adds 26 N-route calls. Bounding each by the diagonal cost t_N(maxdim):
#   md=16:1, 32:2, 48:2, 64:3, 96:3, 128:4, 192:5, 256:6 runs
#   -> <= 15.6 + 46 + 77 + 168 + 328 + 832 + 2126 + 4429 s = 2 h 14 min
# So ~3 h total. 12 h is ample margin; going to 24 h only costs queue priority.
#
# The main sweep re-runs, which is deliberate: it gives a SECOND timing sample.
# findings_check0d §7 flags that the 11x M-vs-N cost ratio rests on single
# unrepeated runs, and this closes that caveat for free.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2 GSCAN=1

echo "gscan on $(hostname)"
echo "start: $(date)"

# Non-destructive: COPY the current CSVs aside, do not move them.
BACKUP="csv_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
for f in cmp_coefficients.csv cmp_errors.csv cmp_spectra.csv cmp_weight_vs_error.csv cmp_gscan.csv; do
    [ -f "$f" ] && cp -v "$f" "$BACKUP/"
done
echo "backed up to $BACKUP/"

julia comparison_study_v2.jl || { echo "comparison_study_v2 FAILED"; exit 1; }

echo "end: $(date)"
echo
echo "wrote: cmp_gscan.csv (plus regenerated cmp_coefficients.csv, cmp_errors.csv)"
echo
echo "WHAT TO LOOK FOR in the JOINT SCAN block:"
echo "  For each maxdim, does dc stay flat as maxdim_G drops? If dc(maxdim, 16)"
echo "  is close to dc(maxdim, maxdim), then G -- the expensive object -- can run"
echo "  far below the ceiling for free, which is where a cost saving would come"
echo "  from. Compare each row's 'x signal': anything under 1.0 beat the free"
echo "  formula and is good enough."
echo
echo "NOTE: spectra_study.jl is NOT re-run here (it does not depend on maxdim_G)."
echo "cmp_spectra.csv and cmp_weight_vs_error.csv from job 6977175 remain valid,"
echo "but cmp_weight_vs_error.csv WILL be stale if the regenerated dc_max values"
echo "differ from 6977175 -- re-run spectra_study.jl afterwards if so."
