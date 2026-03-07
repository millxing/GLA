#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
PROJECT_DIR="/Users/robschoen/Dropbox/CC/GLA"
REPO_DIR="/Users/robschoen/Dropbox/CC/NBA_Data"
SEASON="${SEASON:-}"
ENV_NAME="gla_admin"
CONDA_PATH="/opt/miniconda3"
ENV_PYTHON="$CONDA_PATH/envs/$ENV_NAME/bin/python"
PBP_ANALYZE_STATES_REL="PBPdata/game_states"
PBP_ANALYZE_STATES_ROOT="$REPO_DIR/$PBP_ANALYZE_STATES_REL"
PBP_INDEX_SCRIPT="$PROJECT_DIR/backend/admin/scripts/build_pbp_analyze_index.py"
DRY_RUN="${DRY_RUN:-0}"
REPORTS_DIR="$PROJECT_DIR/reports"
PBP_TARGET_SOURCE="nbastatsv3"
PBP_SOURCE_FOR_STATES="nbastatsv3"
ENABLE_INTERPRETATIONS="${ENABLE_INTERPRETATIONS:-1}"
# --------------------------

cd "$PROJECT_DIR"

mkdir -p "$REPORTS_DIR"
REPORT_DATE="$(date +%Y%m%d)"
REPORT_FILE="$REPORTS_DIR/update${REPORT_DATE}.txt"
CURRENT_STEP="startup"

report_line() {
    local status="$1"
    local details="$2"
    printf "%s | %s | %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$status" "$details" >> "$REPORT_FILE"
}

on_error() {
    local exit_code="$?"
    report_line "FAILED" "${CURRENT_STEP} failed (exit=${exit_code})"
    exit "$exit_code"
}

trap on_error ERR

run_step() {
    local label="$1"
    shift
    CURRENT_STEP="$label"
    report_line "START" "$label"
    "$@"
    report_line "SUCCESS" "$label"
}

if [ ! -x "$ENV_PYTHON" ]; then
    report_line "FAILED" "Python not found at $ENV_PYTHON"
    echo "[error] Python not found at $ENV_PYTHON"
    exit 1
fi

if [ -z "$SEASON" ]; then
    SEASON="$("$ENV_PYTHON" -c 'from backend.config import get_current_season; print(get_current_season())')"
fi
echo "[run] Target season: $SEASON"
if [ "$DRY_RUN" = "1" ]; then
    echo "[run] DRY RUN enabled (no writes/commits)"
fi
report_line "INFO" "Daily update started (season=$SEASON dry_run=$DRY_RUN)"
report_line "INFO" "PBP timeline states root: $PBP_ANALYZE_STATES_ROOT"

echo "[check] Verifying nba_api version"
NBA_API_VERSION="$("$ENV_PYTHON" -c 'from importlib.metadata import version; print(version("nba_api"))')"
if [ "$NBA_API_VERSION" != "1.11.4" ]; then
    report_line "FAILED" "nba_api version check failed (expected=1.11.4 found=$NBA_API_VERSION)"
    echo "[error] Expected nba_api==1.11.4, found $NBA_API_VERSION"
    exit 1
fi
report_line "SUCCESS" "nba_api version verified (1.11.4)"

echo "[check] Verifying scikit-learn version"
SKLEARN_VERSION="$("$ENV_PYTHON" -c 'from importlib.metadata import version; print(version("scikit-learn"))')"
if [ "$SKLEARN_VERSION" != "1.8.0" ]; then
    report_line "FAILED" "scikit-learn version check failed (expected=1.8.0 found=$SKLEARN_VERSION)"
    echo "[error] Expected scikit-learn==1.8.0, found $SKLEARN_VERSION"
    exit 1
fi
report_line "SUCCESS" "scikit-learn version verified (1.8.0)"

echo "[check] Verifying parquet engine (pyarrow/fastparquet)"
PARQUET_ENGINE="$("$ENV_PYTHON" -c 'import importlib.util as u; print("pyarrow" if u.find_spec("pyarrow") else ("fastparquet" if u.find_spec("fastparquet") else ""))')"
if [ -n "$PARQUET_ENGINE" ]; then
    report_line "SUCCESS" "Parquet engine available ($PARQUET_ENGINE)"
else
    report_line "FAILED" "Parquet engine missing (install pyarrow in $ENV_NAME)"
    echo "[error] No parquet engine in $ENV_NAME (pyarrow/fastparquet)."
    echo "[error] Game-state timeline now depends on packed parquet files; install pyarrow and rerun."
    exit 1
fi
report_line "INFO" "PBP source target for this run: $PBP_TARGET_SOURCE"

run_build_states() {
    local phase="$1"
    local rc=0
    local packed_parquet="$PBP_ANALYZE_STATES_ROOT/$phase/$SEASON/_states_${SEASON}_${phase}.parquet"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] Would build-pbp-game-states, pack parquet (with event WP column), verify schema, and refresh pbp_analyze index for phase=$phase"
        return 0
    fi

    set +e
    "$ENV_PYTHON" -m backend.admin.cli \
        --repo-dir "$REPO_DIR" \
        build-pbp-game-states \
        --season "$SEASON" \
        --phase "$phase" \
        --pbp-source "$PBP_SOURCE_FOR_STATES" \
        --output-root "$PBP_ANALYZE_STATES_ROOT"
    rc=$?
    set -e

    if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
        echo "[error] build-pbp-game-states failed for phase=$phase (exit $rc)"
        exit "$rc"
    fi
    if [ "$rc" -eq 2 ]; then
        echo "[warn] build-pbp-game-states completed with validation mismatches for phase=$phase"
    fi

    "$ENV_PYTHON" -m backend.admin.cli \
        --repo-dir "$REPO_DIR" \
        pack-pbp-game-states \
        --season "$SEASON" \
        --phase "$phase" \
        --input-root "$PBP_ANALYZE_STATES_ROOT" \
        --compression zstd \
        --overwrite \
        --delete-json

    # Daily guardrail: packed timeline parquet must include event-level WP cache column.
    "$ENV_PYTHON" - <<PY
from pathlib import Path
import pyarrow.parquet as pq

parquet_path = Path(r"$packed_parquet")
required_col = "home_win_prob_by_event_json"

if not parquet_path.exists():
    raise SystemExit(f"[error] Packed game-states parquet missing: {parquet_path}")

schema = pq.ParquetFile(parquet_path).schema_arrow
if required_col not in schema.names:
    raise SystemExit(f"[error] Missing required column '{required_col}' in {parquet_path}")

print(f"[ok] Verified {required_col} in {parquet_path}")
PY

    report_line "SUCCESS" "Verified packed game-states parquet includes home_win_prob_by_event_json (season=$SEASON phase=$phase)"

    "$ENV_PYTHON" "$PBP_INDEX_SCRIPT" \
        --states-root "$PBP_ANALYZE_STATES_ROOT" \
        --season "$SEASON" \
        --phase "$phase"
}

run_build_timeline_metrics() {
    local phase="$1"
    local metrics_json="$PBP_ANALYZE_STATES_ROOT/$phase/$SEASON/_timeline_metrics_${SEASON}_${phase}.json"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] Would build-pbp-timeline-metrics and verify output for phase=$phase"
        return 0
    fi

    "$ENV_PYTHON" -m backend.admin.cli \
        --repo-dir "$REPO_DIR" \
        build-pbp-timeline-metrics \
        --season "$SEASON" \
        --phase "$phase" \
        --input-root "$PBP_ANALYZE_STATES_ROOT" \
        --output-root "$PBP_ANALYZE_STATES_ROOT" \
        --overwrite

    "$ENV_PYTHON" - <<PY
from pathlib import Path
import json

metrics_path = Path(r"$metrics_json")

if not metrics_path.exists():
    raise SystemExit(f"[error] Timeline metrics output missing: {metrics_path}")

with metrics_path.open("r", encoding="utf-8") as src:
    payload = json.load(src)

games = payload.get("games")
if not isinstance(games, list):
    raise SystemExit(f"[error] Invalid timeline metrics payload (games missing/list) in {metrics_path}")

required_fields = {"game_id", "excitement_factor", "comeback_factor", "possession_changes"}
missing = required_fields.difference(payload["games"][0].keys()) if games else set()
if missing:
    raise SystemExit(
        f"[error] Timeline metrics payload missing required fields {sorted(missing)} in {metrics_path}"
    )

print(f"[ok] Verified timeline metrics: {metrics_path} games={len(games)}")
PY

    report_line "SUCCESS" "Verified timeline metrics JSON generated (season=$SEASON phase=$phase)"
}

verify_timeline_artifacts_synced() {
    local phase="$1"
    local season="$2"
    local states_rel="PBPdata/game_states/$phase/$season/_states_${season}_${phase}.parquet"
    local metrics_rel="PBPdata/game_states/$phase/$season/_timeline_metrics_${season}_${phase}.json"
    local index_rel="PBPdata/game_states/$phase/$season/_index_${season}_${phase}.json"
    local rel

    for rel in "$states_rel" "$metrics_rel" "$index_rel"; do
        if [ ! -f "$REPO_DIR/$rel" ]; then
            echo "[error] Expected timeline artifact missing: $REPO_DIR/$rel"
            return 1
        fi
    done

    local dirty
    dirty="$(git -C "$REPO_DIR" status --short -- "$states_rel" "$metrics_rel" "$index_rel")"
    if [ -n "$dirty" ]; then
        echo "[error] Timeline artifacts still have uncommitted changes after data commit:"
        echo "$dirty"
        return 1
    fi

    local upstream
    set +e
    upstream="$(git -C "$REPO_DIR" rev-parse --abbrev-ref --symbolic-full-name "@{upstream}" 2>/dev/null)"
    local upstream_rc=$?
    set -e

    if [ "$upstream_rc" -eq 0 ] && [ -n "$upstream" ]; then
        local ahead
        local behind
        ahead="$(git -C "$REPO_DIR" rev-list --count "${upstream}..HEAD")"
        behind="$(git -C "$REPO_DIR" rev-list --count "HEAD..${upstream}")"
        if [ "$ahead" -ne 0 ] || [ "$behind" -ne 0 ]; then
            echo "[error] NBA_Data is not synced with $upstream (ahead=$ahead behind=$behind)"
            return 1
        fi
    fi

    local last_touch
    set +e
    last_touch="$(git -C "$REPO_DIR" log -n 1 --pretty=format:'%h %ad %s' --date=iso -- "$states_rel" "$metrics_rel" "$index_rel")"
    set -e
    if [ -n "$last_touch" ]; then
        report_line "INFO" "Timeline artifacts verified clean/synced (season=$season phase=$phase last_touch=$last_touch)"
    else
        report_line "INFO" "Timeline artifacts verified clean/synced (season=$season phase=$phase)"
    fi
}

# Load API keys from .env (for LLM interpretation generation)
if [ -f "$PROJECT_DIR/backend/.env" ]; then
    set -a
    source "$PROJECT_DIR/backend/.env"
    set +a
fi

echo "[run] Updating season data for $SEASON"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] Skipping update-data (no CLI dry-run mode available)"
    report_line "SKIPPED" "Season data update skipped due to dry run"
else
    run_step \
        "Update season data (team_game_logs/linescores/box_score_advanced) for $SEASON" \
        "$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" update-data --season "$SEASON"
fi

echo "[run] Updating raw PBP for $SEASON (regular + playoffs)"
CURRENT_STEP="Update raw PBP (regular + playoffs) for $SEASON"
report_line "START" "$CURRENT_STEP"
PBP_EXIT=0
set +e
"$ENV_PYTHON" -m backend.admin.cli \
    --repo-dir "$REPO_DIR" \
    update-pbp-raw \
    --season "$SEASON" \
    --phase both \
    --target-source "$PBP_TARGET_SOURCE" \
    --no-backup \
    $([ "$DRY_RUN" = "1" ] && echo "--dry-run")
PBP_EXIT=$?
set -e
if [ "$PBP_EXIT" -eq 0 ]; then
    report_line "SUCCESS" "$CURRENT_STEP"
else
    report_line "FAILED" "${CURRENT_STEP} failed (exit=${PBP_EXIT})"
    report_line "WARN" "Continuing despite raw PBP update failure so downstream contribution steps can run"
    echo "[warn] raw PBP update failed (exit $PBP_EXIT); continuing so contributions can still be updated"
fi

echo "[run] Rebuilding PBP game states for PBP Analyze (regular)"
if [ "$PBP_EXIT" -eq 0 ]; then
    CURRENT_STEP="Build PBP Analyze game states (regular) for $SEASON"
    report_line "START" "$CURRENT_STEP"
    run_build_states regular
    report_line "SUCCESS" "$CURRENT_STEP"

    CURRENT_STEP="Build PBP timeline metrics (regular) for $SEASON"
    report_line "START" "$CURRENT_STEP"
    run_build_timeline_metrics regular
    report_line "SUCCESS" "$CURRENT_STEP"
else
    report_line "SKIPPED" "Build PBP Analyze game states (regular) skipped because raw PBP update failed"
    report_line "SKIPPED" "Build PBP timeline metrics (regular) skipped because raw PBP update failed"
fi

# Build playoff game_states only when postseason games exist in team_game_logs.
HAS_POSTSEASON="$(
"$ENV_PYTHON" - <<PY
import pandas as pd
from pathlib import Path

csv_path = Path(r"$REPO_DIR") / "team_game_logs_$SEASON.csv"
if not csv_path.exists():
    print("0")
    raise SystemExit(0)

df = pd.read_csv(csv_path, usecols=["game_type"], low_memory=False)
game_type = df["game_type"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
has_postseason = bool(game_type.isin(["playoffs", "play_in"]).any())
print("1" if has_postseason else "0")
PY
)"

if [ "$HAS_POSTSEASON" = "1" ] && [ "$PBP_EXIT" -eq 0 ]; then
    echo "[run] Rebuilding PBP game states for PBP Analyze (playoffs)"
    CURRENT_STEP="Build PBP Analyze game states (playoffs) for $SEASON"
    report_line "START" "$CURRENT_STEP"
    run_build_states playoffs
    report_line "SUCCESS" "$CURRENT_STEP"

    CURRENT_STEP="Build PBP timeline metrics (playoffs) for $SEASON"
    report_line "START" "$CURRENT_STEP"
    run_build_timeline_metrics playoffs
    report_line "SUCCESS" "$CURRENT_STEP"
elif [ "$HAS_POSTSEASON" = "1" ]; then
    report_line "SKIPPED" "Build PBP Analyze game states (playoffs) skipped because raw PBP update failed"
    report_line "SKIPPED" "Build PBP timeline metrics (playoffs) skipped because raw PBP update failed"
else
    echo "[run] No playoff/play-in games found for $SEASON yet; skipping playoff game-state build"
    report_line "SKIPPED" "Build PBP Analyze game states (playoffs) skipped; no playoff/play-in games for $SEASON"
    report_line "SKIPPED" "Build PBP timeline metrics (playoffs) skipped; no playoff/play-in games for $SEASON"
fi

if [ "$PBP_EXIT" -ne 0 ]; then
    report_line "SKIPPED" "PBP timeline commit via NBA_Data skipped because raw PBP update failed"
fi

echo "[run] Building situational game logs for $SEASON"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] Skipping situational game-log build"
    report_line "SKIPPED" "Situational game-log build skipped due to dry run"
else
    run_step \
        "Build situational game logs (garbage_filtered + clutch) for $SEASON" \
        "$ENV_PYTHON" "$PROJECT_DIR/backend/admin/build_situational_gamelogs.py" --season "$SEASON" --repo-dir "$REPO_DIR" --incremental --scope-state-mode pre
fi

# Commit/push data updates immediately (before interpretations)
TODAY="$(date +%Y-%m-%d)"
MSG="Update ${SEASON} data (${TODAY})"
echo "[run] Commit + push data updates if needed"
run_step \
    "Commit and push data updates to GitHub (message: $MSG)" \
    "$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$MSG" $([ "$DRY_RUN" = "1" ] && echo "--dry-run")

if [ "$DRY_RUN" != "1" ] && [ "$PBP_EXIT" -eq 0 ]; then
    run_step \
        "Verify timeline artifacts are committed and pushed (regular) for $SEASON" \
        verify_timeline_artifacts_synced regular "$SEASON"

    if [ "$HAS_POSTSEASON" = "1" ]; then
        run_step \
            "Verify timeline artifacts are committed and pushed (playoffs) for $SEASON" \
            verify_timeline_artifacts_synced playoffs "$SEASON"
    fi
fi

echo "[run] Regenerating contributions for $SEASON (all scopes)"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] Skipping contribution regeneration"
    report_line "SKIPPED" "Contribution regeneration skipped due to dry run"
else
    run_step \
        "Regenerate contributions (scope=all) for $SEASON" \
        "$ENV_PYTHON" "$PROJECT_DIR/backend/admin/generate_contributions.py" --season "$SEASON" --repo-dir "$REPO_DIR" --data-scope all --incremental
    run_step \
        "Regenerate contributions (scope=garbage_filtered) for $SEASON" \
        "$ENV_PYTHON" "$PROJECT_DIR/backend/admin/generate_contributions.py" --season "$SEASON" --repo-dir "$REPO_DIR" --data-scope garbage_filtered --incremental
    run_step \
        "Regenerate contributions (scope=clutch) for $SEASON" \
        "$ENV_PYTHON" "$PROJECT_DIR/backend/admin/generate_contributions.py" --season "$SEASON" --repo-dir "$REPO_DIR" --data-scope clutch --incremental
fi

echo "[run] Commit + push contribution updates if needed"
CONTRIB_MSG="Update ${SEASON} contributions (${TODAY})"
run_step \
    "Commit and push contribution updates to GitHub (message: $CONTRIB_MSG)" \
    "$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$CONTRIB_MSG" $([ "$DRY_RUN" = "1" ] && echo "--dry-run")

INTERP_EXIT=0
INTERP_COMMIT_EXIT=0
if [ "$ENABLE_INTERPRETATIONS" = "1" ]; then
    echo "[run] Generating LLM interpretations for new games"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] Skipping interpretation generation"
        report_line "SKIPPED" "Generate LLM interpretations skipped due to dry run"
    else
        CURRENT_STEP="Generate LLM interpretations for $SEASON"
        report_line "START" "$CURRENT_STEP"
        if "$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" generate-interpretations --season "$SEASON" --current --incremental; then
            report_line "SUCCESS" "$CURRENT_STEP"
        else
            INTERP_EXIT=$?
            report_line "FAILED" "${CURRENT_STEP} failed (exit=${INTERP_EXIT})"
            report_line "WARN" "Continuing despite interpretation generation failure so data/contribution updates remain published"
            echo "[warn] Interpretation generation failed with exit code $INTERP_EXIT; continuing."
        fi
    fi

    echo "[run] Commit + push interpretation updates if needed"
    INTERP_MSG="Update ${SEASON} interpretations (${TODAY})"
    CURRENT_STEP="Commit and push interpretation updates to GitHub (message: $INTERP_MSG)"
    report_line "START" "$CURRENT_STEP"
    set +e
    "$ENV_PYTHON" -m backend.admin.cli --repo-dir "$REPO_DIR" commit-and-push --message "$INTERP_MSG" $([ "$DRY_RUN" = "1" ] && echo "--dry-run")
    INTERP_COMMIT_EXIT=$?
    set -e
    if [ "$INTERP_COMMIT_EXIT" -eq 0 ]; then
        report_line "SUCCESS" "$CURRENT_STEP"
    else
        report_line "FAILED" "${CURRENT_STEP} failed (exit=${INTERP_COMMIT_EXIT})"
        report_line "WARN" "Continuing despite interpretation commit/push failure; core morning update already completed"
        echo "[warn] Interpretation commit/push failed with exit code $INTERP_COMMIT_EXIT; continuing."
    fi
else
    echo "[run] Interpretation generation disabled; skipping interpretation update steps"
    report_line "SKIPPED" "Generate LLM interpretations skipped (ENABLE_INTERPRETATIONS=$ENABLE_INTERPRETATIONS)"
    report_line "SKIPPED" "Commit and push interpretation updates skipped (ENABLE_INTERPRETATIONS=$ENABLE_INTERPRETATIONS)"
fi

if [ "$INTERP_EXIT" -ne 0 ] || [ "$INTERP_COMMIT_EXIT" -ne 0 ]; then
    report_line "WARN" "Daily update finished with interpretation warnings (generation_exit=${INTERP_EXIT} commit_exit=${INTERP_COMMIT_EXIT})"
else
    report_line "SUCCESS" "Daily update finished successfully"
fi
