#!/bin/bash
# Push results back to the remote. Runs on the cluster, at the end of the job chain.
#
#   bash scripts/publish.sh [message]
#
# Only artifacts/ and the small summary parquets go back. The scenario sets and the
# per-run backtest tree are gigabytes and stay where they were produced.

set -euo pipefail
cd "$(dirname "$0")/.."

MESSAGE="${1:-results from job ${SLURM_JOB_ID:-manual} on $(hostname -s)}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git config user.name  >/dev/null 2>&1 || git config user.name  "scenport runner"
git config user.email >/dev/null 2>&1 || git config user.email "noreply@$(hostname -f)"

git add -A artifacts runs/summary runs/exact 2>/dev/null || true

if git diff --cached --quiet; then
    echo "nothing new to publish"
    exit 0
fi

git commit -q -m "$MESSAGE"
echo "committed: $(git log -1 --oneline)"

# The laptop may have pushed while the jobs were running. Rebase on top of it and
# retry a few times rather than failing the whole chain over a race.
for attempt in 1 2 3 4 5; do
    if git pull --rebase --autostash origin "$BRANCH" && git push origin "$BRANCH"; then
        echo "pushed to origin/$BRANCH"
        exit 0
    fi
    echo "push attempt $attempt failed, retrying in $((attempt * 15))s" >&2
    sleep $((attempt * 15))
done

echo "could not push after 5 attempts; the commit is still here, push it by hand" >&2
exit 1
