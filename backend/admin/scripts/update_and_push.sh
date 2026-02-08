#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
PROJECT_DIR="/Users/robschoen/Dropbox/CC/GLA"
REPO_DIR="/Users/robschoen/Dropbox/CC/NBA_Data"
SEASON="2025-26"
ENV_NAME="gla_admin"
CONDA_PATH="/opt/miniconda3"
ENV_PYTHON="$CONDA_PATH/envs/$ENV_NAME/bin/python"
# --------------------------

cd "$PROJECT_DIR"

if [ ! -x "$ENV_PYTHON" ]; then
    echo "[error] Python not found at $ENV_PYTHON"
    exit 1
fi

# Load API keys from .env (for LLM interpretation generation)
if [ -f "$PROJECT_DIR/backend/.env" ]; then
    source "$PROJECT_DIR/backend/.env"
fi

echo "[run] Updating season data for $SEASON"
"$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" update-data --season "$SEASON"

# Commit/push data updates immediately (before interpretations)
TODAY="$(date +%Y-%m-%d)"
MSG="Update ${SEASON} data (${TODAY})"
echo "[run] Commit + push data updates if needed"
"$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$MSG"

echo "[run] Regenerating contributions for $SEASON"
"$ENV_PYTHON" "$PROJECT_DIR/backend/admin/generate_contributions.py" --season "$SEASON" --repo-dir "$REPO_DIR"

echo "[run] Generating LLM interpretations for new games"
"$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" generate-interpretations --season "$SEASON" --current --incremental --max-new 20

echo "[run] Commit + push contributions + interpretations if needed"
"$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$MSG"
