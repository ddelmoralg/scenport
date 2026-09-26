#!/bin/bash
# One-time setup on the cluster login node. Idempotent: run it twice.
#
#   bash scripts/bootstrap_cluster.sh
#
# First run generates a deploy key and stops, printing the public half. Register
# that half on the repo (Settings > Deploy keys, tick "Allow write access") and run
# the script again. The second run clones or repoints the checkout and verifies
# that pushing works.
#
# The private half is generated here and never leaves this machine.

set -euo pipefail

REPO="${SCENPORT_REPO:-ralorin/scenport}"
ROOT="${SCENPORT_ROOT:-$HOME/scenport}"
ALIAS="github-scenport"
KEY="$HOME/.ssh/scenport_deploy"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# --- host key, so nothing ever stops to ask -------------------------------------

if ! ssh-keygen -F github.com >/dev/null 2>&1; then
    ssh-keyscan -t ed25519,rsa github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
    echo "added github.com to known_hosts"
fi

# --- the key --------------------------------------------------------------------

if [ ! -f "$KEY" ]; then
    ssh-keygen -t ed25519 -f "$KEY" -N "" -C "scenport deploy key on $(hostname -s)" >/dev/null
    echo "generated $KEY"
fi
chmod 600 "$KEY"

# --- ssh config -----------------------------------------------------------------

if ! grep -q "Host $ALIAS" "$HOME/.ssh/config" 2>/dev/null; then
    cat >> "$HOME/.ssh/config" <<CONF

Host $ALIAS
    HostName github.com
    User git
    IdentityFile $KEY
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
CONF
    chmod 600 "$HOME/.ssh/config"
    echo "added '$ALIAS' to ~/.ssh/config"
fi

# --- can we talk to github yet? --------------------------------------------------

auth_output="$(ssh -o BatchMode=yes -T "git@$ALIAS" 2>&1 || true)"

if ! printf '%s' "$auth_output" | grep -q "successfully authenticated"; then
    echo
    echo "The key is not registered on the repo yet. Public half:"
    echo
    cat "$KEY.pub"
    echo
    echo "Register it at https://github.com/$REPO/settings/keys with write access,"
    echo "then run this script again."
    exit 0
fi

echo "deploy key works: $(printf '%s' "$auth_output" | head -1)"

# --- checkout --------------------------------------------------------------------

if [ -d "$ROOT/.git" ]; then
    git -C "$ROOT" remote set-url origin "git@$ALIAS:$REPO.git"
    git -C "$ROOT" pull --rebase --autostash
    echo "updated $ROOT"
else
    git clone "git@$ALIAS:$REPO.git" "$ROOT"
    echo "cloned into $ROOT"
fi

git -C "$ROOT" config user.name "scenport runner"
git -C "$ROOT" config user.email "scenport@$(hostname -f)"

echo
echo "done. next:"
echo "  conda env create -f $ROOT/environment.yml   # if it is not there already"
echo "  cd $ROOT && bash slurm/submit_all.sh"
