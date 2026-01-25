#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
PROJECT_DIR="/Users/robschoen/Dropbox/CC/GLA"
REPO_DIR="/Users/robschoen/Dropbox/CC/NBA_Data"
SEASON="2025-26"
ENV_NAME="gla_admin"
# --------------------------

cd "$PROJECT_DIR"

# Make sure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH."
  echo "If you're using Miniconda/Anaconda, try:"
  echo "  source ~/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

# Initialize conda for scripts
eval "$(conda shell.bash hook)"

# Activate env
conda activate "$ENV_NAME"

echo "[run] Updating season data for $SEASON"
python -m backend.admin.cli --repo-dir "$REPO_DIR" update-data --season "$SEASON"

# Build a nice commit message with today's date
TODAY="$(date +%Y-%m-%d)"
MSG="Update ${SEASON} data (${TODAY})"

echo "[run] Commit + push if needed"
python -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$MSG"
