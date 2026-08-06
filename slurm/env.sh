# Sourced by every job script. Edit this file, not the job scripts.
#
# The cluster does not have a system Python new enough for the pinned versions, so
# everything lives in one conda environment created from environment.yml.

set -euo pipefail

export SCENPORT_ROOT="${SCENPORT_ROOT:-$HOME/scenport}"

module purge
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scenport

cd "$SCENPORT_ROOT"
mkdir -p logs

# Every library here will happily grab every core on the node and then fight the
# other array tasks for them. One thread each, and let SLURM do the scheduling.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# cmdstanpy writes its scratch files next to the model by default, which on a
# shared filesystem means 160 array tasks stepping on each other.
export TMPDIR="${TMPDIR:-/tmp}/scenport-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

python -c "import scenport; print('scenport', scenport.__version__)"
