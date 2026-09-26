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

python scripts/fetch_data.py --config configs/base.yaml

echo "panel written to data/panel"
ls -lh data/panel
