#!/bin/bash
#SBATCH --job-name=scen_nf_bt
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-43%44
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# The genetic algorithm on each replicate scenario set of 33_noise_forecasts.sh.
# Replicate r uses backtest seed r, so the coupling template and the search are
# redrawn along with the generator. One run per task.
#
#   sbatch --dependency=afterok:<cpu job>:<gpu job> slurm/34_noise_backtest.sh

SCENPORT_ROOT="${SCENPORT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/scenport}}"
export SCENPORT_ROOT
# shellcheck disable=SC1091
source "$SCENPORT_ROOT/slurm/env.sh"

FORECASTERS=(rw rw_nodrift arima ets garch prophet prophet_rmse lgbm lstm deepar transformer)
N_SEEDS=4

idx=${SLURM_ARRAY_TASK_ID:-0}
forecaster=${FORECASTERS[$((idx / N_SEEDS))]}
seed=$((1 + idx % N_SEEDS))

echo "task $idx -> forecaster=$forecaster replicate=$seed"

python scripts/run_backtest.py \
    --config configs/base.yaml \
    --set paths.runs=runs/noise_floor \
    --forecaster "$forecaster" \
    --forecast-seed "$seed" \
    --allocator ga \
    --seeds 1 \
    --first-seed "$seed"
