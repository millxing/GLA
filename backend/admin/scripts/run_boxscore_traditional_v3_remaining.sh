#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/robschoen/Dropbox/CC/GLA}"
REPO_DIR="${REPO_DIR:-/Users/robschoen/Dropbox/CC/NBA_Data}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/gla_admin/bin/python}"
DOWNLOADER="$PROJECT_DIR/backend/admin/download_boxscore_traditional_v3.py"

START_YEAR="${START_YEAR:-2023}"
END_YEAR="${END_YEAR:-2024}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12}"
RETRIES="${RETRIES:-1}"
PAUSE_SECONDS="${PAUSE_SECONDS:-2}"
REST_EVERY_GAMES="${REST_EVERY_GAMES:-50}"
REST_SECONDS="${REST_SECONDS:-20}"
MAX_TIMEOUT_COOLDOWNS="${MAX_TIMEOUT_COOLDOWNS:-1}"
RESUME_SLEEP_SECONDS="${RESUME_SLEEP_SECONDS:-120}"
RESUME_OFFSET_STEP="${RESUME_OFFSET_STEP:-100}"
MAX_AUTO_RESUME_PASSES="${MAX_AUTO_RESUME_PASSES:-100}"

# Fresh-process gap between seasons.
SEASON_PROCESS_SLEEP_SECONDS="${SEASON_PROCESS_SLEEP_SECONDS:-120}"

season_label() {
    local start_year="$1"
    local end_suffix
    end_suffix=$(((start_year + 1) % 100))
    printf "%04d-%02d" "$start_year" "$end_suffix"
}

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[error] Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ ! -f "$DOWNLOADER" ]; then
    echo "[error] Downloader not found: $DOWNLOADER" >&2
    exit 1
fi

cd "$PROJECT_DIR"

failed_seasons=()

for ((year=START_YEAR; year<=END_YEAR; year++)); do
    season="$(season_label "$year")"
    echo
    echo "[run] Starting BoxScoreTraditionalV3 for $season"
    echo "[run] Repo dir: $REPO_DIR"
    echo "[run] Fresh process per season to avoid season-transition throttling"

    if ! "$PYTHON_BIN" "$DOWNLOADER" \
        --season "$season" \
        --repo-dir "$REPO_DIR" \
        --auto-resume \
        --timeout-seconds "$TIMEOUT_SECONDS" \
        --retries "$RETRIES" \
        --pause-seconds "$PAUSE_SECONDS" \
        --rest-every-games "$REST_EVERY_GAMES" \
        --rest-seconds "$REST_SECONDS" \
        --max-timeout-cooldowns "$MAX_TIMEOUT_COOLDOWNS" \
        --resume-sleep-seconds "$RESUME_SLEEP_SECONDS" \
        --resume-offset-step "$RESUME_OFFSET_STEP" \
        --max-auto-resume-passes "$MAX_AUTO_RESUME_PASSES"
    then
        echo "[warn] Season failed or stopped incomplete: $season"
        failed_seasons+=("$season")
    else
        echo "[ok] Season completed: $season"
    fi

    if [ "$year" -lt "$END_YEAR" ] && [ "$SEASON_PROCESS_SLEEP_SECONDS" -gt 0 ]; then
        echo "[run] Sleeping ${SEASON_PROCESS_SLEEP_SECONDS}s before next season"
        sleep "$SEASON_PROCESS_SLEEP_SECONDS"
    fi
done

echo
if [ "${#failed_seasons[@]}" -gt 0 ]; then
    echo "[done] Completed with failures: ${failed_seasons[*]}"
    exit 1
fi

echo "[done] All requested seasons completed successfully."
