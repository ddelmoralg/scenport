#!/bin/bash
# Wires the cluster to the repo, from here, in one go. Idempotent.
#
#   bash scripts/setup_cluster_from_laptop.sh [user@host]
#
# Uses the two things you already have: your own SSH access to the cluster, and a
# logged-in gh. Nothing to type, and no GitHub credential ever lands on the cluster
# beyond a deploy key scoped to this one repo.
#
# What it does:
#   1. copies bootstrap_cluster.sh over
#   2. runs it, which generates a deploy key there and prints the public half
#   3. registers that half on the repo through the GitHub API
#   4. runs it again, which clones the repo and checks that pushing works

set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:-${SCENPORT_HPC:-hpc-bisite.usal.es}}"
REPO="${SCENPORT_REPO:-ralorin/scenport}"

command -v gh >/dev/null || { echo "gh is not installed" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh is not logged in; run 'gh auth login'" >&2; exit 1; }

echo "cluster : $HOST"
echo "repo    : $REPO"
echo

echo "== copying the bootstrap script =="
scp -q scripts/bootstrap_cluster.sh "$HOST:~/bootstrap_cluster.sh"

echo "== first run: generate the key =="
first="$(ssh "$HOST" 'bash ~/bootstrap_cluster.sh' 2>&1)"
echo "$first" | sed 's/^/    /'

pubkey="$(printf '%s\n' "$first" | grep -m1 '^ssh-ed25519 ' || true)"

if [ -z "$pubkey" ]; then
    if printf '%s' "$first" | grep -q "deploy key works"; then
        echo
        echo "already wired up, nothing to register."
        exit 0
    fi
    echo "could not find a public key in the output above" >&2
    exit 1
fi

echo
echo "== registering the deploy key =="

title="$(echo "$HOST" | tr -cd 'A-Za-z0-9._@-') $(date +%Y-%m-%d)"

existing="$(gh api "repos/$REPO/keys" --jq '.[].key' 2>/dev/null || true)"
keybody="$(printf '%s' "$pubkey" | cut -d' ' -f1,2)"

if printf '%s\n' "$existing" | grep -qF "$keybody"; then
    echo "    already registered"
else
    gh api "repos/$REPO/keys" \
        -f "title=$title" \
        -f "key=$pubkey" \
        -F read_only=false \
        --jq '"    added key id \(.id), write access"'
fi

echo
echo "== second run: clone and verify =="
ssh "$HOST" 'bash ~/bootstrap_cluster.sh' 2>&1 | sed 's/^/    /'

echo
echo "Deploy keys for this repo: https://github.com/$REPO/settings/keys"
echo "Revoke there if the cluster is ever rebuilt again."
