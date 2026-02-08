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

echo "[run] Commit + push contribution updates if needed"
CONTRIB_MSG="Update ${SEASON} contributions (${TODAY})"
"$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$CONTRIB_MSG"

echo "[run] Generating LLM interpretations for new games"
INTERP_EXIT=0
if "$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" generate-interpretations --season "$SEASON" --current --incremental --max-new 20; then
    :
else
    INTERP_EXIT=$?
    echo "[warn] Interpretation generation failed with exit code $INTERP_EXIT; continuing so contributions are already pushed."
fi

echo "[run] Commit + push interpretation updates if needed"
INTERP_MSG="Update ${SEASON} interpretations (${TODAY})"
"$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$INTERP_MSG"

if [ "$INTERP_EXIT" -ne 0 ]; then
    echo "[warn] Exiting with interpretation failure code: $INTERP_EXIT"
    exit "$INTERP_EXIT"
fi
