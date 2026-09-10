#!/bin/bash
# Dephasing operating line, matched to the amplitude-damping line already run.
#
#   sbatch --array=0-5 submit_dephasing_line.sh
#
# THE QUESTION. The AD line established chi_traj(n) ~ 13 e^{0.22 n}, reaching
# 268 at n=24, and crossed the N chi^3 = 1e18 feasibility ceiling around n=51.
# This run asks whether dephasing sits ABOVE that line at matched noise, which
# is what would move the hardware target to smaller n.
#
# WHAT IS DIFFERENT FROM submit_operating_line.sh, and why:
#
# 1. TWO ARMS PER POINT. Dephasing has two natural unravellings that give the
#    same rho at very different cost, and a classical competitor picks the
#    cheap one. :projective is CRy+reset (weak Z monitoring, disentangles);
#    :pauli is the random-Z circuit (every trajectory unitary, chi tracks the
#    NOISELESS chain). Quoting :pauli would be measuring our own choice. The
#    binding number is min over arms, and the gap between them is itself a
#    result worth recording. Add :gaussian to UNRAVELLINGS if you also want the
#    exact random-Rz arm the hardware implements -- same channel as :pauli, so
#    it is a cross-check rather than new information.
#
# 2. GAMMA IS MATCHED ON T2, NOT ON THE SYMBOL. Amplitude damping at rate g
#    damps the transverse Pauli components at g/2; dephasing at gamma_phi damps
#    them at gamma_phi. So gamma_phi = g/2 = 0.7/n reproduces the same
#    transverse damping per step, which is the thing that feeds operator
#    entanglement. The script prints D_X, D_Z and the contraction coefficient
#    c(N) for both channels at every run so this is auditable, not asserted.
#    Note c(dephasing) >= 1/3 at ANY strength because D_Z = 1 -- that is the
#    structural reason to expect it to be the harder channel, and the reason
#    this is worth 40 core-hours.
#
# 3. CHI_MPDO IS DELIBERATELY EMPTY. The AD study's fitted law
#    (S_op ~ 1.39 t, chi ~ 6*2^(1.61 S)) describes AD operator entanglement.
#    Reusing it here would assume exactly what is under test, so the driver
#    SKIPS the MPDO block rather than fabricating it. Fill CHI_MPDO in after
#    rerunning the MPDO study with the dephasing dissipator; until then this
#    run answers "dephasing vs AD trajectories", not "trajectories vs MPDO".
#
# 4. CHI_AD_TRAJ is supplied per n so the hardness question is arithmetic in
#    the log rather than a later merge. VALUES BELOW ARE READ OFF THE
#    operating_line_n24 FIGURE AND ARE APPROXIMATE -- replace them with the
#    chi_mean column of your op_n*/trajectory_n*.csv before quoting anything.
#
# 5. LONGER WALL LIMIT. The AD timings are a LOWER bound here: if the
#    hypothesis is right, chi is larger and the runs are slower. n=24 keeps the
#    shorter window for the same reason it did before.
#SBATCH --job-name=dephline
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=logs/dephline_%A_%a.out
#SBATCH --error=logs/dephline_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Identical to the AD runs so the numbers are comparable, not merely similar.
export DT=0.05 CUTOFF=1e-10 JCOUP=0.5 SEED0=1000 TARGET_SEM=0.01
export TMAX_FACTOR=0.6 NT=20
export UNRAVELLINGS=projective,pauli
export NTRAJ_UNIT=16          # the unitary arm's chi barely varies; 16 is plenty
export CHI_MPDO=0             # see note 3 -- do not fill from the AD law

# GAMMA_AD is the AD line's rate; GAMMA_PHI defaults to half of it (note 2).
case $SLURM_ARRAY_TASK_ID in
  0) export N=10 GAMMA_AD=0.140000 NTRAJ=256 MAXDIM=512  MAXDIM_UNIT=1024
     export CHI_AD_TRAJ=13   OUTDIR=deph_n10 TAG=n10 ;;
  1) export N=12 GAMMA_AD=0.116667 NTRAJ=256 MAXDIM=512  MAXDIM_UNIT=2048
     export CHI_AD_TRAJ=20   OUTDIR=deph_n12 TAG=n12 ;;
  2) export N=14 GAMMA_AD=0.100000 NTRAJ=256 MAXDIM=1024 MAXDIM_UNIT=2048
     export CHI_AD_TRAJ=30   OUTDIR=deph_n14 TAG=n14 ;;
  3) export N=16 GAMMA_AD=0.087500 NTRAJ=256 MAXDIM=1024 MAXDIM_UNIT=2048
     export CHI_AD_TRAJ=48   OUTDIR=deph_n16 TAG=n16 ;;
  4) export N=20 GAMMA_AD=0.070000 NTRAJ=128 MAXDIM=2048 MAXDIM_UNIT=2048
     export CHI_AD_TRAJ=130  OUTDIR=deph_n20 TAG=n20 ;;
  # n=24 keeps the shorter window, as in the AD run. The AD chi peak sat at
  # t/t* = 0.93; if the dephasing peak drifts past 1.0 in the n<=20 CSVs, put
  # TMAX_FACTOR back to 0.6 here and drop NTRAJ to 16 to stay inside the wall.
  5) export N=24 GAMMA_AD=0.058333 NTRAJ=32  MAXDIM=2048 MAXDIM_UNIT=2048
     export TMAX_FACTOR=0.5 NT=12
     export CHI_AD_TRAJ=268  OUTDIR=deph_n24 TAG=n24 ;;
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID N=$N GAMMA_AD=$GAMMA_AD (GAMMA_PHI defaults to half)"
echo "NTRAJ=$NTRAJ MAXDIM=$MAXDIM MAXDIM_UNIT=$MAXDIM_UNIT TMAX_FACTOR=$TMAX_FACTOR"
echo "OUTDIR=$OUTDIR threads=$JULIA_NUM_THREADS host=$(hostname) cwd=$(pwd)"
echo "start: $(date)"

for f in dephasing_evolution.jl dephasing_study.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

julia --threads="$JULIA_NUM_THREADS" dephasing_study.jl
JULIA_STATUS=$?
echo "end: $(date)"

echo "--- CSVs written ---"
ls -la "$OUTDIR"
NCSV=$(find "$OUTDIR" -name '*.csv' | wc -l)
echo "csv count: $NCSV"

if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
# projective csv + pauli csv + cost_comparison + manifest = 4
if [ "$NCSV" -le 2 ]; then
  echo "FATAL: julia exited 0 but produced no data beyond the manifest" >&2
  exit 3
fi
exit 0
