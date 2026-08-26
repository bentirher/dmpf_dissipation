#!/bin/bash
#SBATCH --job-name=cmpv2
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=logs/cmpv2_%j.out
#SBATCH --error=logs/cmpv2_%j.err

# Regenerates ALL FOUR CSVs against the direct-overlap oracle.
#
# 64G for the same reason as submit_delta_fix.sh: exact_delta=true carries
# chi(A)+chi(B) = 512 at n=4, above the 256 MPO ceiling, applied twice per step.
#
# 10:00:00. Job 6974035 did ~8 ceiling-level runs in 2h30 (~18 min each). This
# sweep is 8 maxdims x (M-route + N-route), plus one extra m_route(256) for the
# v1 reference comparison, plus spectra_study. The low-maxdim points are cheap,
# so ~3h is the estimate; the margin is for spectra_study, which rebuilds F and
# Phi at the ceiling. The scripts write their CSVs after EVERY point, so a
# timeout costs only the unfinished tail, not the run.
#
# GSCAN: leave at 0 here. GSCAN=1 adds 26 more N-route calls (every maxdim_G <=
# maxdim at every maxdim) and will NOT fit in this allocation -- submit it as a
# separate job with --time=24:00:00.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2 GSCAN=1

echo "comparison_study_v2 on $(hostname)"
echo "start: $(date)"

# Keep the stale files rather than overwriting them: they are the only record of
# what the published numbers were measured against.
for f in cmp_coefficients.csv cmp_errors.csv cmp_spectra.csv cmp_weight_vs_error.csv; do
    [ -f "$f" ] && mv -v "$f" "stale_$f"
done

julia comparison_study_v2.jl || { echo "comparison_study_v2 FAILED"; exit 1; }

echo
echo "--- spectra_study (unmodified; reads dc_max from field 5) ---"
export KDIAG=8
julia spectra_study.jl || { echo "spectra_study FAILED"; exit 1; }

echo "end: $(date)"
echo
echo "wrote: cmp_coefficients.csv cmp_errors.csv cmp_spectra.csv cmp_weight_vs_error.csv"
echo "In the GROUND TRUTH block, check:"
echo "  - 'drift' is ~0       -> the oracle saturated at the MPS ceiling"
echo "  - '|v1 - oracle|'     -> how wrong the old reference was (expect ~7e-6)"
echo "  - 'signal'            -> expect ~1.43e-02"
