#!/bin/bash
#SBATCH --job-name=scen_bt
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Walk-forward backtest for everything except the metaheuristics: one array task
# per forecaster.
#
# These allocators are cheap. The whole 85-date calendar runs in seconds each,
# because they are closed forms or a linear programme. The metaheuristics are
# hours and they live in 32_backtest_ea.sh, which is a finer array. Running both
# here would put nine hours of work in a single task, which is what the earlier
# shape of this job did.
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

echo "backtesting the cheap allocators on $forecaster"

python scripts/run_backtest.py \
    --config configs/base.yaml \
    --forecaster "$forecaster" \
    --allocator equal mcap random invvol single minvar maxsharpe riskparity hrp cvar_lp \
    --seeds 5
