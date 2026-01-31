#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
PROJECT_DIR="/Users/robschoen/Dropbox/CC/GLA"
REPO_DIR="/Users/robschoen/Dropbox/CC/NBA_Data"
SEASON="2025-26"
ENV_NAME="gla_admin"
CONDA_PATH="/opt/miniconda3"
# --------------------------

cd "$PROJECT_DIR"

# Initialize conda (use full path for launchd compatibility)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate env
conda activate "$ENV_NAME"

echo "[run] Updating season data for $SEASON"
python -m backend.admin.cli --repo-dir "$REPO_DIR" update-data --season "$SEASON"

# Build a nice commit message with today's date
TODAY="$(date +%Y-%m-%d)"
MSG="Update ${SEASON} data (${TODAY})"

echo "[run] Commit + push if needed"
python -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$MSG"
