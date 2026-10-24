#!/bin/bash
#SBATCH --job-name=scen_nf_fc
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-43%16
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Replicate scenario sets for the noise floor: every generator refitted and
# resimulated with forecast seeds 1 to 4, over the whole calendar. Seed 0 is the
# main run and is not touched. Everything goes under runs/noise_floor, so the
# scores, scenarios and backtests of the main study are never mixed with these.
#
# 11 generators x 4 seeds = 44 tasks. The neural generators need a GPU, so the
# script asks for one only on the tasks that run them; submit it twice:
#
#   sbatch --array=0-31%16 slurm/33_noise_forecasts.sh
#   sbatch --array=32-43%6 --gres=gpu:h100:1 slurm/33_noise_forecasts.sh
#
# then slurm/34_noise_backtest.sh and slurm/35_noise_floor.sh after both.

SCENPORT_ROOT="${SCENPORT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/scenport}}"
export SCENPORT_ROOT
# shellcheck disable=SC1091
source "$SCENPORT_ROOT/slurm/env.sh"

FORECASTERS=(rw rw_nodrift arima ets garch prophet prophet_rmse lgbm lstm deepar transformer)
N_SEEDS=4
# SCENPORT_SHARDS splits each replicate's calendar over several tasks, which the
# neural generators need on CPU: about four minutes a date, so 85 dates do not
# fit one task. With 4 shards the neural tasks are 128 to 175.
N_SHARDS="${SCENPORT_SHARDS:-1}"

idx=${SLURM_ARRAY_TASK_ID:-0}
forecaster=${FORECASTERS[$((idx / (N_SEEDS * N_SHARDS)))]}
seed=$((1 + (idx / N_SHARDS) % N_SEEDS))
shard=$((idx % N_SHARDS))

extra=()
workers="${SLURM_CPUS_PER_TASK:-8}"
case "$forecaster" in
    lstm|deepar|transformer)
        # SCENPORT_DEVICE=cpu runs them without a GPU when none is free; the
        # procedure is the same, only the arithmetic device changes.
        extra=(--param "device=${SCENPORT_DEVICE:-cuda}")
        workers=1
        ;;
esac

echo "task $idx -> forecaster=$forecaster forecast seed=$seed shard=$shard/$N_SHARDS"

python scripts/run_forecasts.py \
    --config configs/base.yaml \
    --set paths.runs=runs/noise_floor \
    --forecaster "$forecaster" \
    --seed "$seed" \
    --shard "$shard" \
    --n-shards "$N_SHARDS" \
    --workers "$workers" \
    "${extra[@]}"
