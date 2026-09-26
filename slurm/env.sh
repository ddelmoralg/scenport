# Sourced by every job script. Edit this file, not the job scripts.
#
# Queues on this cluster: short (4 h, default), medium (24 h), long (7 days).
# One compute node with 224 cores and 8 H100s, but the queues carry group QOS caps
# that bind long before the hardware does: medium allows 112 cores and 4 GPUs,
# long allows 56 cores and 2 GPUs, shared with everyone else on the queue. Size
# the %N on each array against those, not against the node.
#
# The cluster does not have a system Python new enough for the pinned versions, so
# everything lives in one conda environment created from environment.yml.

set -euo pipefail

export SCENPORT_ROOT="${SCENPORT_ROOT:-$HOME/scenport}"
export SCENPORT_ENV="${SCENPORT_ENV:-scenport}"

module purge 2>/dev/null || true
module load miniconda3 2>/dev/null \
  || module load anaconda3 2>/dev/null \
  || module load conda 2>/dev/null \
  || true

# The node was reinstalled and the module names moved around, so fall back to a
# conda installed under $HOME if none of the above took.
if ! command -v conda >/dev/null 2>&1; then
    for candidate in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Load the right module here or install miniforge under \$HOME." >&2
    exit 1
fi

# Conda's own hooks are not written for a strict shell. prophet ships an
# activate.d script that reads $CMDSTAN before setting it, which under `set -u`
# aborts the activation and leaves the job running against the base interpreter.
# Relax nounset across the activation only.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCENPORT_ENV"
set -u

if [ "${CONDA_DEFAULT_ENV:-}" != "$SCENPORT_ENV" ]; then
    echo "conda activate '$SCENPORT_ENV' did not take (now in '${CONDA_DEFAULT_ENV:-none}')" >&2
    exit 1
fi

cd "$SCENPORT_ROOT"
mkdir -p logs

# Every library here will happily grab every core on the node and then fight the
# other array tasks for them. One thread each, and let SLURM do the scheduling.
# 40_exact.sh overrides this: the MILP solver is the one place we want the threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# cmdstanpy writes its scratch files next to the model by default, which on a
# shared filesystem means 128 array tasks stepping on each other.
#
# Do not trust an inherited TMPDIR. A login profile here points it at a directory
# that does not exist, and SLURM only silently repairs that for the job it starts,
# not for anything the job then runs.
scratch_base="${TMPDIR:-/tmp}"
if [ ! -d "$scratch_base" ] || [ ! -w "$scratch_base" ]; then
    echo "TMPDIR '$scratch_base' is missing or not writable; using /tmp" >&2
    scratch_base=/tmp
fi
export TMPDIR="$scratch_base/scenport-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

python -c "import scenport; print('scenport', scenport.__version__)"
