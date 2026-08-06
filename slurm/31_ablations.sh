#!/bin/bash
#SBATCH --job-name=scen_abl
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Sensitivity runs. Same scenarios, one knob moved at a time.
#
# Each entry below is a tag plus the overrides that define it. Everything replays
# the stored scenario sets, so none of this costs a new simulation -- which is the
# whole reason the two stages were separated.

source "$(dirname "$0")/env.sh"

FORECASTER=${SCENPORT_ABLATION_FORECASTER:-garch}

# tag|override|override|...
VARIANTS=(
  "gamma00|alloc.gamma=0.0"
  "gamma25|alloc.gamma=0.25"
  "gamma75|alloc.gamma=0.75"
  "gamma100|alloc.gamma=1.0"
  "card05|alloc.cardinality=5"
  "card10|alloc.cardinality=10"
  "card15|alloc.cardinality=15"
  "wmax10|alloc.w_max=0.10"
  "wmax33|alloc.w_max=0.33"
  "wmax100|alloc.w_max=1.0|alloc.w_min=0.0"
  "cost00|backtest.cost_bps=0"
  "cost50|backtest.cost_bps=50"
  "cost100|backtest.cost_bps=100"
  "alpha90|alloc.cvar_alpha=0.90"
  "alpha99|alloc.cvar_alpha=0.99"
  "nocouple|scenarios.coupling=independent"
  "empirical|scenarios.coupling=empirical"
  "sharpeobj|alloc.objective=sharpe"
)

entry=${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}
IFS='|' read -r -a parts <<< "$entry"
tag=${parts[0]}
overrides=()
for ((i = 1; i < ${#parts[@]}; i++)); do
    overrides+=(--set "${parts[$i]}")
done

echo "ablation $tag on $FORECASTER: ${overrides[*]}"

python scripts/run_backtest.py \
    --config configs/base.yaml \
    --forecaster "$FORECASTER" \
    --allocator ga de cmaes cvar_lp equal hrp \
    --seeds 10 \
    --tag "$tag" \
    "${overrides[@]}"
