#!/bin/bash
#SBATCH --job-name=scen_fetch
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Downloads the Coin Metrics panel and the FRED risk-free series, then builds the
# price panel and the point-in-time universe.
#
# NOTE: this is the only step that needs outbound network access. If the compute
# nodes are firewalled, run it on the login node instead:
#
#   conda activate scenport && python scripts/fetch_data.py --config configs/base.yaml
#
# Everything downloaded is cached under data/raw, so a rerun costs nothing.

source "$(dirname "$0")/env.sh"

python scripts/fetch_data.py --config configs/base.yaml

echo "panel written to data/panel"
ls -lh data/panel
