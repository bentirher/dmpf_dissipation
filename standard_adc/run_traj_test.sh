#!/bin/bash
# Login-node smoke test for the trajectory workflow.
#
#   bash run_traj_test.sh
#
# Small on purpose (n=8, 20 trajectories, 4 time points) -- fine interactively.
# Anything larger goes through sbatch.
#
# WHY THIS SCRIPT EXISTS RATHER THAN A ONE-LINER: an interactive login shell is
# NOT the environment your batch jobs run in. The submit script calls
# `module load` before julia; a bare interactive `julia ...` does not, so it
# picks up whatever julia happens to be on PATH, with a different depot and no
# ITensors. That is the "Package ITensors not found in current path" error.
# Everything below is the batch environment, reproduced.

set -uo pipefail

module load Julia/1.11.6-linux-x86_64 || {
  echo "FATAL: module load failed. Check the exact name with: module avail Julia" >&2
  exit 2
}

echo "--- environment ---"
echo "julia binary : $(which julia 2>/dev/null || echo NOT-FOUND)"
julia --version || { echo "FATAL: julia not on PATH after module load" >&2; exit 2; }

# If the project keeps its own Project.toml, use it. `--project=@.` walks up
# from the current directory looking for one and falls back to the default
# environment if there is none, which is what the batch jobs effectively do.
JLPROJ="--project=@."
[ -f Project.toml ] && echo "project      : $(pwd)/Project.toml" \
                    || echo "project      : none here, using default depot env"

echo "depot / env  :"
julia $JLPROJ -e 'println("  active: ", Base.active_project()); println("  depot : ", DEPOT_PATH[1])'

# Fail fast and legibly if the packages are missing, rather than 20 lines of
# stacktrace from inside an include.
echo "--- checking packages ---"
julia $JLPROJ -e 'using ITensors, ITensorMPS; println("  ITensors OK")' || {
  echo >&2
  echo "FATAL: ITensors is not available in this environment." >&2
  echo "  Find the environment your batch jobs use:" >&2
  echo "    julia -e 'using Pkg; Pkg.status()'" >&2
  echo "  If it is a named/shared env, point this script at it, e.g." >&2
  echo "    JLPROJ=--project=/scratch/bentirher/envs/itensors" >&2
  echo "  Or install into the current one:" >&2
  echo "    julia $JLPROJ -e 'using Pkg; Pkg.add([\"ITensors\",\"ITensorMPS\"])'" >&2
  exit 3
}

echo "--- required files ---"
for f in trajectory_evolution.jl trajectory_study.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

# --- run parameters ---------------------------------------------------------
# gamma = 0.05 deliberately: it matches open_n8_g0p050_* from the MPDO study,
# and at n=8 the MPDO ceiling 4^4 = 256 means that run was EXACT. Comparing
# <Z_mid>(t) against it is the validation.
export N=8
export GAMMA=0.05
export NTRAJ=20
export NT=4
export TMAX_FACTOR=0.5      # t up to 4.0
export DT=0.05              # same step as the MPDO runs
export MAXDIM=64
export CUTOFF=1e-10
export JCOUP=0.5
export OUTDIR=traj_test
export TAG=test

echo "--- running ---"
# -t auto is required: run_trajectories parallelises over trajectories with
# Threads.@threads and silently uses one thread without it.
julia $JLPROJ -t auto trajectory_study.jl
STATUS=$?

echo
if [ $STATUS -ne 0 ]; then
  echo "julia exited $STATUS" >&2
  exit $STATUS
fi
echo "--- output ---"
ls -la "$OUTDIR"
