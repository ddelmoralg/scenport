#!/bin/bash
#SBATCH --job-name=scen_agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Builds every table and figure in the paper from whatever is in runs/.
# The bootstrap tests are the slow part, which is why this asks for four hours.

source "$(dirname "$0")/env.sh"

python scripts/aggregate.py --config configs/base.yaml
python scripts/make_figures.py --config configs/base.yaml
python scripts/export_instances.py --config configs/base.yaml --forecaster garch --every 3

echo
echo "artifacts:"
ls -lh artifacts
