#!/bin/bash
#SBATCH --job-name=scen_exact
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Branch and bound. Task 0 measures the optimality gap on the real rebalancing
# dates; task 1 grows the universe until the solver stops finishing.
#
# The two-day wall clock is the cpu partition maximum and is deliberate: the point
# of the scaling run is to find where the MILP stops being usable, so a task that
# times out at n=150 is a result, not a failure. Per-instance time is capped so one
# hard instance cannot eat the whole allocation.

source "$(dirname "$0")/env.sh"

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
        --sizes 10 15 20 30 50 75 100 150 \
        --max-scenarios 400 \
        --time-limit 1800
    ;;
esac
