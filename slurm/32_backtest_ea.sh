#!/bin/bash
#SBATCH --job-name=scen_bt_ea
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-43%44
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# The metaheuristics, one array task per forecaster and allocator.
#
# These are the whole cost of the backtest stage and everything else is noise
# beside them. Measured on the reference machine at the production scenario count,
# one GA solve takes about 19 seconds, so 85 rebalancing dates is around 27 minutes
# per seed and five seeds is a little over two hours. Four metaheuristics for one
# forecaster in a single task would be nine hours, which is why the earlier shape
# of this job would have timed out.
#
# 11 forecasters x 4 metaheuristics = 44 tasks, all of them at once: 88 cores,
# which fits under the medium queue's group QOS cap of 112. The ablations run
# alongside and the scheduler interleaves them against that same cap.
#
# env.sh pins every library to one thread, so two cores per task is the allocation
# plus somewhere for the parquet writes to go.
#
#   sbatch --dependency=afterok:<forecast jobs> slurm/32_backtest_ea.sh

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

FORECASTERS=(rw rw_nodrift arima ets garch prophet prophet_rmse lgbm lstm deepar transformer)
ALLOCATORS=(ga de pso cmaes)

idx=${SLURM_ARRAY_TASK_ID:-0}
forecaster=${FORECASTERS[$((idx / ${#ALLOCATORS[@]}))]}
allocator=${ALLOCATORS[$((idx % ${#ALLOCATORS[@]}))]}

echo "task $idx -> forecaster=$forecaster allocator=$allocator"

python scripts/run_backtest.py \
    --config configs/base.yaml \
    --forecaster "$forecaster" \
    --allocator "$allocator" \
    --seeds 5
