#!/bin/bash
#SBATCH --job-name=scen_fc_gpu
#SBATCH --partition=gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Scenario simulation for the three neural forecasters.
#
# One GPU each and at most two tasks at a time, which is the per-user limit on
# gpu-small. Three forecasters x two shards = six tasks. The models are small
# (a two-layer encoder over 64 steps), so a single GPU is plenty and asking for
# more would only lengthen the queue for everyone.
#
# Depends on the panel from 00_fetch.sh. Independent of 10_forecast_cpu.sh, so the
# two can run at the same time.
#
#   sbatch slurm/20_forecast_gpu.sh

source "$(dirname "$0")/env.sh"

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
