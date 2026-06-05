#!/usr/bin/env bash
# scripts/run_remote.sh — run heavy MMM compute on a VM you control (multi-core PyMC, no time limit,
# and the place to run the REAL R Robyn). The web sandbox and GitHub Actions both cap cores/time;
# a VM does not. You provide the host; this syncs the repo over, runs your command in a venv, and
# pulls docs/ back.
#
# Env:  MMM_VM_HOST=user@host   (required)
#       MMM_VM_DIR=mmm_example  (remote path, default)
# Usage:
#   MMM_VM_HOST=ubuntu@1.2.3.4 bash scripts/run_remote.sh 'make all'
#   MMM_VM_HOST=ubuntu@1.2.3.4 bash scripts/run_remote.sh \
#       'bash scripts/run_conditional_sweep.sh "0.5 1.0 2.0" "11 22 33 44 55" 0.6'
set -euo pipefail
: "${MMM_VM_HOST:?set MMM_VM_HOST=user@host}"
DIR="${MMM_VM_DIR:-mmm_example}"
CMD="${1:?provide a remote command, e.g. 'make all'}"

echo "==> sync repo to $MMM_VM_HOST:$DIR"
rsync -az --delete --exclude .git --exclude .venv --exclude node_modules \
  --exclude 'artifacts/*.nc' --exclude 'artifacts/*.rds' ./ "$MMM_VM_HOST:$DIR/"

echo "==> run on VM (venv auto-created on first use; PyMC can use all cores there)"
ssh "$MMM_VM_HOST" "set -e; cd '$DIR'; \
  python3 -m venv .venv 2>/dev/null || true; . .venv/bin/activate; \
  pip -q install -U pip >/dev/null; pip -q install -e '.[fit]' >/dev/null; \
  export RETICULATE_PYTHON=\$(command -v python); $CMD"

echo "==> sync results back (docs/)"
rsync -az "$MMM_VM_HOST:$DIR/docs/" ./docs/
echo "done. Review docs/, then commit/push locally."
