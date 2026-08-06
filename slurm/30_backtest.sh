#!/bin/bash
#SBATCH --job-name=scen_bt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Walk-forward backtest: one array task per forecaster, all allocators inside.
#
# Every task replays the scenario sets written by the forecasting jobs, so this
# must wait for both of them:
#
#   fc_cpu=$(sbatch --parsable slurm/10_forecast_cpu.sh)
#   fc_gpu=$(sbatch --parsable slurm/20_forecast_gpu.sh)
#   sbatch --dependency=afterok:$fc_cpu:$fc_gpu slurm/30_backtest.sh

source "$(dirname "$0")/env.sh"

FORECASTERS=(rw rw_nodrift arima ets garch prophet prophet_rmse lgbm lstm deepar transformer)
forecaster=${FORECASTERS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "backtesting every allocator on $forecaster"

python scripts/run_backtest.py \
    --config configs/base.yaml \
    --forecaster "$forecaster" \
    --allocator all \
    --seeds 5
