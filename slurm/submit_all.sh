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

for queue in short medium long; do
    if ! sinfo -h -p "$queue" -o %P >/dev/null 2>&1 || [ -z "$(sinfo -h -p "$queue" -o %P)" ]; then
        echo "queue '$queue' does not exist on this cluster." >&2
        echo "Available:" >&2
        sinfo -h -o '  %P  maxtime %l  nodes %D' >&2
        echo "Fix the --partition lines in slurm/*.sh before submitting." >&2
        exit 1
    fi
done

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

bt_ea=$(sbatch --parsable --dependency="afterok:${fc_cpu}:${fc_gpu}" slurm/32_backtest_ea.sh)
echo "backtest (ea)   : $bt_ea"

abl=$(sbatch --parsable --dependency="afterok:${fc_cpu}" slurm/31_ablations.sh)
echo "ablations       : $abl"

exact=$(sbatch --parsable --dependency="afterok:${fc_cpu}" slurm/40_exact.sh)
echo "exact           : $exact"

# The ablations are sensitivity checks, not headline results, so they wait with
# afterany: one infeasible variant should not throw away a completed backtest by
# cancelling the aggregation. The jobs the tables are actually built from stay on
# afterok.
agg=$(sbatch --parsable     --dependency="afterok:${bt}:${bt_ea}:${exact},afterany:${abl}"     slurm/50_aggregate.sh)
echo "aggregate       : $agg"

echo
echo "The publish step is not in the chain. The compute nodes on this cluster"
echo "cannot resolve github.com, so 60_publish.sh spends five retries failing."
echo "Once squeue is empty, from the login node:"
echo
echo "    bash scripts/publish.sh"

echo
echo "watch:   squeue -u \$USER -o '%.10i %.14j %.9P %.8T %.10M %R'"
echo "cancel:  scancel $fc_cpu $fc_gpu $bt $bt_ea $abl $exact $agg"
