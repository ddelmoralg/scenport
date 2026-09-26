#!/bin/bash
#SBATCH --job-name=scen_fc_cpu
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Scenario simulation for the eight CPU forecasters.
#
# The array index is unpacked into (forecaster, shard): 8 forecasters x 16 shards
# of the rebalancing calendar = 128 tasks. Shards are interleaved rather than
# contiguous, so every task gets a mix of early and late dates and they finish at
# roughly the same time.
#
# Prophet dominates the cost: eight grid points, three inner folds and a final fit
# per asset per date. The eight worker processes are used to run several dates in
# parallel inside a task, which is much more effective than trying to parallelise a
# single Stan fit.
#
# Twelve at a time is 96 cores. The node has 224 but the medium queue carries a
# group QOS cap of 112 shared with everyone else on it, so anything above that
# just queues. This leaves the GPU job its 24 and a little for other people.
#
#   sbatch slurm/10_forecast_cpu.sh

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

FORECASTERS=(rw rw_nodrift arima ets garch prophet prophet_rmse lgbm)
N_SHARDS=16

idx=${SLURM_ARRAY_TASK_ID:-0}
forecaster=${FORECASTERS[$((idx / N_SHARDS))]}
shard=$((idx % N_SHARDS))

echo "task $idx -> forecaster=$forecaster shard=$shard/$N_SHARDS"

python scripts/run_forecasts.py \
    --config configs/base.yaml \
    --forecaster "$forecaster" \
    --seed 0 \
    --shard "$shard" \
    --n-shards "$N_SHARDS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
