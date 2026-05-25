#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/robschoen/Dropbox/CC/GLA"
REPORTS_DIR="$PROJECT_DIR/reports"
UPDATER="$PROJECT_DIR/backend/admin/scripts/update_and_push.sh"
LOG_FILE="/Users/robschoen/Library/Logs/nba-data-update.log"
RUN_HOUR="${RUN_HOUR:-6}"

now_hour="$(date +%H)"
today="$(date +%Y%m%d)"
report_file="$REPORTS_DIR/update${today}.txt"

log_line() {
    (printf "%s | %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG_FILE") 2>/dev/null || true
}

mkdir -p "$REPORTS_DIR"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

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

exec /bin/bash "$UPDATER"
