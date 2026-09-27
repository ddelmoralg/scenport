#!/bin/bash
#SBATCH --job-name=scen_agg
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Builds every table and figure in the paper from whatever is in runs/.
# The bootstrap tests are the slow part, which is why this asks for four hours.

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

python scripts/aggregate.py --config configs/base.yaml
python scripts/make_figures.py --config configs/base.yaml
python scripts/export_instances.py --config configs/base.yaml

# Last thing before anything leaves this machine. The first full run finished
# green and produced results that were measuring the wrong thing, so the chain
# now refuses to publish output it can tell is impossible.
python scripts/check_results.py --config configs/base.yaml

echo
echo "artifacts:"
ls -lh artifacts
