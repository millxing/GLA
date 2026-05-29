#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/robschoen/Dropbox/CC/GLA"
REPORTS_DIR="$PROJECT_DIR/reports"
UPDATER="$PROJECT_DIR/backend/admin/scripts/update_and_push.sh"
LOG_FILE="/Users/robschoen/Library/Logs/nba-data-update.log"
RUN_HOUR="${RUN_HOUR:-6}"
LOCK_DIR="/tmp/gla-nba-data-update.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"

now_hour="$(date +%H)"
today="$(date +%Y%m%d)"
report_file="$REPORTS_DIR/update${today}.txt"

log_line() {
    (printf "%s | %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG_FILE") 2>/dev/null || true
}

mkdir -p "$REPORTS_DIR"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    existing_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        log_line "Skipping daily updater wrapper; another updater run is already active (pid=$existing_pid)"
        exit 0
    fi
    rm -rf "$LOCK_DIR"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        log_line "Skipping daily updater wrapper; could not acquire updater lock"
        exit 0
    fi
fi
printf "%s\n" "$$" > "$LOCK_PID_FILE" 2>/dev/null || true
cleanup_lock() {
    rm -f "$LOCK_PID_FILE" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [ "${FORCE_RUN:-0}" != "1" ]; then
    if [ "$now_hour" -lt "$RUN_HOUR" ]; then
        log_line "Skipping daily updater wrapper; before ${RUN_HOUR}:00 and RunAtLoad fired early"
        exit 0
    fi

    if [ -f "$report_file" ] && grep -q "Daily update finished successfully" "$report_file"; then
        log_line "Skipping daily updater wrapper; successful report already exists for $today"
        exit 0
    fi
fi

/bin/bash "$UPDATER"
