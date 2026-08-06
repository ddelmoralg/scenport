#!/bin/bash
# Pull the latest code, submit the whole pipeline, print the job ids.
#
#   bash slurm/submit_all.sh
#
# Run it on the login node. The last job pushes the results back to the remote, so
# once squeue is empty you just `git pull` on the other end.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "working tree is dirty; commit or stash before submitting" >&2
    git status --short
    exit 1
fi

git pull --rebase --autostash
echo "at $(git log -1 --oneline)"

if [ ! -f data/panel/price.parquet ]; then
    echo "data/panel/price.parquet is missing." >&2
    echo "Run scripts/fetch_data.py somewhere with network access and commit the panel." >&2
    exit 1
fi

fc_cpu=$(sbatch --parsable slurm/10_forecast_cpu.sh)
echo "forecast (cpu)  : $fc_cpu"

fc_gpu=$(sbatch --parsable slurm/20_forecast_gpu.sh)
echo "forecast (gpu)  : $fc_gpu"

bt=$(sbatch --parsable --dependency="afterok:${fc_cpu}:${fc_gpu}" slurm/30_backtest.sh)
echo "backtest        : $bt"

abl=$(sbatch --parsable --dependency="afterok:${fc_cpu}" slurm/31_ablations.sh)
echo "ablations       : $abl"

exact=$(sbatch --parsable --dependency="afterok:${fc_cpu}" slurm/40_exact.sh)
echo "exact           : $exact"

agg=$(sbatch --parsable --dependency="afterok:${bt}:${abl}:${exact}" slurm/50_aggregate.sh)
echo "aggregate       : $agg"

pub=$(sbatch --parsable --dependency="afterok:${agg}" slurm/60_publish.sh)
echo "publish         : $pub"

echo
echo "watch:   squeue -u \$USER -o '%.10i %.14j %.9P %.8T %.10M %R'"
echo "cancel:  scancel $fc_cpu $fc_gpu $bt $abl $exact $agg $pub"
