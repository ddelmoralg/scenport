#!/bin/bash
#SBATCH --job-name=scen_fc_gpu
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Scenario simulation for the three neural forecasters.
#
# Three forecasters x two shards = six tasks, one H100 each, three at a time. The
# node has eight cards and no per-user cap right now, but the models are small (a
# two-layer encoder over 64 steps) so taking more would only starve other people
# for no gain here.
#
# Runs alongside 10_forecast_cpu.sh on the same node: 3 x 8 = 24 cores here plus
# 20 x 8 = 160 there leaves room on the 224 the node has.
#
# Depends on the panel from 00_fetch.sh. Independent of 10_forecast_cpu.sh.
#
#   sbatch slurm/20_forecast_gpu.sh

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

FORECASTERS=(lstm deepar transformer)
N_SHARDS=2

idx=${SLURM_ARRAY_TASK_ID:-0}
forecaster=${FORECASTERS[$((idx / N_SHARDS))]}
shard=$((idx % N_SHARDS))

echo "task $idx -> forecaster=$forecaster shard=$shard/$N_SHARDS"
echo "GPU assigned: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# One date at a time: the workers would each want the GPU.
python scripts/run_forecasts.py \
    --config configs/base.yaml \
    --forecaster "$forecaster" \
    --seed 0 \
    --shard "$shard" \
    --n-shards "$N_SHARDS" \
    --workers 1 \
    --param device=cuda
