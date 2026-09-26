#!/bin/bash
#SBATCH --job-name=scen_exact
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Branch and bound. Task 0 measures the optimality gap on the real rebalancing
# dates; task 1 grows the universe until the solver stops finishing.
#
# This is the only job that needs the long queue. Three days of wall clock is
# deliberate: the point of the scaling run is to find where the MILP stops being
# usable, so a task that times out at n=200 is a result, not a failure. Per-instance
# time is capped so one hard instance cannot eat the whole allocation.

# Inside a batch job $0 is the copy SLURM spooled under /var/spool, not this file
# in the repository, so dirname "$0" finds nothing. SLURM_SUBMIT_DIR is where
# sbatch was run, which submit_all.sh makes the repository root.
SCENPORT_ROOT="${SCENPORT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/scenport}}"
export SCENPORT_ROOT
if [ ! -f "$SCENPORT_ROOT/slurm/env.sh" ]; then
    echo "cannot find slurm/env.sh under '$SCENPORT_ROOT'" >&2
    echo "set SCENPORT_ROOT or submit from the repository root" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$SCENPORT_ROOT/slurm/env.sh"

# HiGHS gets the whole allocation for its own parallel branch and bound. This is
# the one job where env.sh's one-thread-per-library rule is wrong.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

case ${SLURM_ARRAY_TASK_ID:-0} in
  0)
    python scripts/run_exact.py gaps \
        --config configs/base.yaml \
        --forecaster garch \
        --every 3 \
        --seeds 10 \
        --max-scenarios 400 \
        --time-limit 900
    ;;
  1)
    python scripts/run_exact.py scaling \
        --config configs/base.yaml \
        --sizes 10 15 20 30 50 75 100 150 200 \
        --max-scenarios 400 \
        --time-limit 3600
    ;;
esac
