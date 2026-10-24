#!/bin/bash
#SBATCH --job-name=scen_nf_agg
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Noise floor from the replicates, written to artifacts/noise_floor.{csv,json}.
#
#   sbatch --dependency=afterok:<backtest job> slurm/35_noise_floor.sh

SCENPORT_ROOT="${SCENPORT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/scenport}}"
export SCENPORT_ROOT
# shellcheck disable=SC1091
source "$SCENPORT_ROOT/slurm/env.sh"

python scripts/noise_floor.py
