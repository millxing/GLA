#!/usr/bin/env python3
"""Train and export season-specific win probability model (WPM) artifacts.

This module builds one artifact per season using rolling prior-season windows:
- First season (2000-01): in-sample training (special-case fallback)
- All later seasons: train on up to N prior seasons (default: 3)

Each artifact includes:
- Training window metadata
- Out-of-sample metrics on the target season
- Serialized fitted estimators for:
  1) Logistic Regression with interaction effects
  2) HistGradientBoosting
  3) RandomForest + isotonic calibration
- Soft-vote behavior (simple mean of model probabilities)
"""

from __future__ import annotations

import base64
import gzip
import json
import pickle
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import (
    PBP_GITHUB_RAW_BASE_URL,
    PBP_REMOTE_CACHE_DIR,
    PBP_ROOT_DIR,
    PBP_WINPROB_BASE_ROOT,
    PBP_WINPROB_MODELS_ROOT,
    get_available_seasons,
)


RANDOM_STATE = 42
LOOKBACK_SEASONS_DEFAULT = 3

WINPROB_REQUIRED_COLUMNS = [
    "home",
    "road",
    "quarter",
    "seconds_left",
    "differential",
    "possession",
    "final_score_diff",
]

FEATURE_COLUMNS = [
    "game_seconds_left",
    "current_differential",
    "possession_numeric",
]

LOGISTIC_NAME = "logistic_interactions"
HGB_NAME = "hist_gradient_boosting"
RF_NAME = "random_forest_isotonic"
SOFT_VOTE_NAME = "ensemble_soft_vote"
SYMMETRY_MODE = "mirror_average"
REQUIRED_SKLEARN_VERSION = "1.8.0"


DEFAULT_INPUT_ROOT = PBP_WINPROB_BASE_ROOT
DEFAULT_OUTPUT_ROOT = PBP_WINPROB_MODELS_ROOT


SMALL_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 50,
    "max_samples": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def _pbpdata_relative(path: Path) -> Optional[str]:
    try:
        rel = path.expanduser().resolve().relative_to(PBP_ROOT_DIR)
    except Exception:
        return None
    return rel.as_posix()


def _cache_path_for_relative(relative_path: str) -> Path:
    return PBP_REMOTE_CACHE_DIR / relative_path


def _remote_url(relative_path: str) -> str:
    return f"{PBP_GITHUB_RAW_BASE_URL}/{relative_path.lstrip('/')}"


def _download_remote_to_cache(relative_path: str) -> Optional[Path]:
    cache_path = _cache_path_for_relative(relative_path)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(_remote_url(relative_path), headers={"User-Agent": "GLA-winprob-fallback"})
    try:
        with urlopen(req, timeout=20) as resp:
            payload = resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return None

    if not payload:
        return None

    cache_path.write_bytes(payload)
    return cache_path


def _ensure_local_or_remote(path: Path) -> Optional[Path]:
    p = path.expanduser().resolve()
    if p.exists():
        return p

    relative = _pbpdata_relative(p)
    if not relative:
        return None
    return _download_remote_to_cache(relative)


@lru_cache(maxsize=1)
def _discover_remote_wpm_seasons() -> Tuple[str, ...]:
    seasons: List[str] = []
    for season in get_available_seasons():
        rel = f"winprob_models/wpm_{season}.json"
        req = Request(_remote_url(rel), method="HEAD", headers={"User-Agent": "GLA-winprob-fallback"})
        try:
            with urlopen(req, timeout=6):
                seasons.append(season)
        except HTTPError as exc:
            if exc.code != 404:
                continue
        except Exception:
            continue
    return tuple(seasons)


def _runtime_lib_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    packages = {
        "scikit_learn": "scikit-learn",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    for key, package_name in packages.items():
        try:
            versions[key] = pkg_version(package_name)
        except PackageNotFoundError:
            versions[key] = "unknown"
    return versions


def _assert_required_sklearn_runtime() -> str:
    runtime_sklearn = str(_runtime_lib_versions().get("scikit_learn") or "").strip() or "unknown"
    if runtime_sklearn != REQUIRED_SKLEARN_VERSION:
        raise RuntimeError(
            f"[wpm] scikit-learn=={REQUIRED_SKLEARN_VERSION} is required for winprob models; "
            f"found {runtime_sklearn}. Activate/install the required version and retry."
        )
    return runtime_sklearn


def _assert_artifact_sklearn_version(artifact_path: Path, artifact: Dict[str, Any]) -> str:
    artifact_versions = artifact.get("library_versions")
    if not isinstance(artifact_versions, dict):
        raise ValueError(
            f"[wpm] Artifact {artifact_path.name} is missing library_versions metadata. "
            f"Rebuild all winprob artifacts with scikit-learn=={REQUIRED_SKLEARN_VERSION}."
        )

    artifact_sklearn = str(artifact_versions.get("scikit_learn") or "").strip()
    if artifact_sklearn != REQUIRED_SKLEARN_VERSION:
        raise ValueError(
            f"[wpm] Artifact {artifact_path.name} was trained with scikit-learn={artifact_sklearn or 'unknown'}, "
            f"but required version is {REQUIRED_SKLEARN_VERSION}. Rebuild all winprob artifacts."
        )
    return artifact_sklearn


def _season_sort_key(season: str) -> int:
    try:
        return int(season.split("-")[0])
    except Exception:
        return -1


def _find_stacked_path(input_root: Path, season: str, phase: str) -> Optional[Path]:
    filename = f"stacked_{season}_winprob_base.csv"
    candidates = [
        input_root / filename,
        input_root / phase / filename,
    ]
    for path in candidates:
        resolved = _ensure_local_or_remote(path)
        if resolved is not None:
            return resolved
    return None


def discover_seasons(input_root: Path, phase: str) -> List[str]:
    seasons = set()
    patterns = [
        input_root.glob("stacked_*_winprob_base.csv"),
        (input_root / phase).glob("stacked_*_winprob_base.csv"),
    ]

    for iterator in patterns:
        for p in iterator:
            stem = p.stem  # stacked_2023-24_winprob_base
            prefix = "stacked_"
            suffix = "_winprob_base"
            if not (stem.startswith(prefix) and stem.endswith(suffix)):
                continue
            season = stem[len(prefix) : -len(suffix)]
            if season:
                seasons.add(season)

    return sorted(seasons, key=_season_sort_key)


def _engineer_features(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    model_df = raw_df.copy()
    model_df = model_df.dropna(subset=["quarter", "seconds_left", "differential", "final_score_diff"])

    model_df["target_home_win_sign"] = np.where(model_df["final_score_diff"] > 0, 1, -1).astype(int)
    model_df["target_home_win_bin"] = (model_df["target_home_win_sign"] == 1).astype(int)

    model_df["full_q_left"] = np.where(model_df["quarter"] <= 4, 4 - model_df["quarter"], 0)
    model_df["game_seconds_left"] = model_df["full_q_left"] * 720 + model_df["seconds_left"]
    model_df["current_differential"] = model_df["differential"]
    model_df["possession_numeric"] = np.select(
        [
            model_df["possession"] == model_df["home"],
            model_df["possession"] == model_df["road"],
        ],
        [1, -1],
        default=0,
    ).astype(int)

    x = model_df[FEATURE_COLUMNS].astype(float)
    y_sign = model_df["target_home_win_sign"].astype(int)
    y_bin = model_df["target_home_win_bin"].astype(int)
    return x, y_sign, y_bin


def load_season_matrix(input_root: Path, season: str, phase: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    csv_path = _find_stacked_path(input_root, season, phase)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find stacked winprob base for season={season} under {input_root}")

    raw = pd.read_csv(csv_path, usecols=WINPROB_REQUIRED_COLUMNS)
    x, y_sign, y_bin = _engineer_features(raw)
    return x, y_sign, y_bin, int(len(raw))


def _train_models(x_train: pd.DataFrame, y_train_sign: pd.Series) -> Dict[str, Any]:
    # Mirror-augment training rows so home/road labels are symmetric:
    # swap perspective by flipping differential + possession and flipping target sign.
    x_mirror = x_train.copy()
    x_mirror["current_differential"] = -x_mirror["current_differential"]
    x_mirror["possession_numeric"] = -x_mirror["possession_numeric"]
    y_mirror = -y_train_sign

    x_fit = pd.concat([x_train, x_mirror], ignore_index=True)
    y_fit = pd.concat([y_train_sign, y_mirror], ignore_index=True)

    logistic_model = make_pipeline(
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )
    logistic_model.fit(x_fit, y_fit)

    hgb_model = HistGradientBoostingClassifier(
        max_depth=6,
        random_state=RANDOM_STATE,
    )
    hgb_model.fit(x_fit, y_fit)

    # ensemble=False keeps one base estimator in the fitted artifact while still
    # calibrating via CV predictions on the training set.
    rf_calibrated = CalibratedClassifierCV(
        estimator=RandomForestClassifier(**SMALL_RF_PARAMS),
        method="isotonic",
        cv=5,
        ensemble=False,
    )
    rf_calibrated.fit(x_fit, y_fit)

    return {
        LOGISTIC_NAME: logistic_model,
        HGB_NAME: hgb_model,
        RF_NAME: rf_calibrated,
    }


def _mirror_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["current_differential"] = -out["current_differential"]
    out["possession_numeric"] = -out["possession_numeric"]
    return out


def _home_prob(model: Any, x: pd.DataFrame, symmetry_mode: Optional[str] = None) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model classes do not include +1 home-win class.")
    idx = classes.index(1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        p_home = model.predict_proba(x)[:, idx]

    if symmetry_mode == SYMMETRY_MODE:
        x_mirror = _mirror_features(x)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            p_mirror_home = model.predict_proba(x_mirror)[:, idx]
        # Symmetric home-win probability wrapper:
        # p*(s) = 0.5 * (p(s) + (1 - p(s_mirror)))
        p_home = 0.5 * (p_home + (1.0 - p_mirror_home))

    return p_home


def _metric_row(y_true_bin: pd.Series, p_home: np.ndarray) -> Dict[str, float]:
    y_true = y_true_bin.to_numpy()
    y_pred = (p_home >= 0.5).astype(int)

    out: Dict[str, float] = {
        "brier_home_win": float(brier_score_loss(y_true, p_home)),
        "log_loss_home_win": float(log_loss(y_true, p_home, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_home_win": float(f1_score(y_true, y_pred, pos_label=1)),
    }
    try:
        out["roc_auc_home_win"] = float(roc_auc_score(y_true, p_home))
    except ValueError:
        out["roc_auc_home_win"] = float("nan")
    return out


def _evaluate_models(models: Dict[str, Any], x_test: pd.DataFrame, y_test_bin: pd.Series) -> Dict[str, Dict[str, float]]:
    probs: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        p_home = _home_prob(model, x_test, symmetry_mode=SYMMETRY_MODE)
        probs[name] = p_home
        metrics[name] = _metric_row(y_test_bin, p_home)

    p_ensemble = np.mean(
        np.column_stack([probs[LOGISTIC_NAME], probs[HGB_NAME], probs[RF_NAME]]),
        axis=1,
    )
    metrics[SOFT_VOTE_NAME] = _metric_row(y_test_bin, p_ensemble)
    return metrics


def _encode_model(model: Any) -> Tuple[str, Dict[str, int]]:
    raw = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = gzip.compress(raw, compresslevel=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    return b64, {"pickle_bytes": len(raw), "gzip_bytes": len(compressed), "base64_chars": len(b64)}


def _decode_model(payload_b64: str) -> Any:
    compressed = base64.b64decode(payload_b64.encode("ascii"))
    raw = gzip.decompress(compressed)
    return pickle.loads(raw)


def _patch_model_compat(model: Any) -> int:
    """Patch known cross-version sklearn pickle incompatibilities."""
    patched = 0
    seen: set[int] = set()

    def _walk(obj: Any) -> None:
        nonlocal patched
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, LogisticRegression):
            # Older artifacts can miss this attribute after pickle load.
            if not hasattr(obj, "multi_class"):
                obj.multi_class = "auto"
                patched += 1
            return

        if isinstance(obj, dict):
            for value in obj.values():
                _walk(value)
            return

        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                _walk(value)
            return

        steps = getattr(obj, "steps", None)
        if isinstance(steps, list):
            for _, step in steps:
                _walk(step)

        for attr in ("estimator", "estimator_", "base_estimator"):
            child = getattr(obj, attr, None)
            if child is not None:
                _walk(child)

        estimators = getattr(obj, "estimators_", None)
        if isinstance(estimators, list):
            for child in estimators:
                _walk(child)

    _walk(model)
    return patched


def _normalize_game_key(game_id: Any) -> str:
    if game_id is None:
        return ""
    s = str(game_id).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s
    trimmed = digits.lstrip("0")
    return trimmed or "0"


def _training_window(seasons: List[str], season: str, lookback_seasons: int) -> List[str]:
    idx = seasons.index(season)
    if idx == 0:
        return [season]
    start = max(0, idx - lookback_seasons)
    return seasons[start:idx]


def _concat_training_data(
    season_cache: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, int]],
    train_seasons: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    x_parts = [season_cache[s][0] for s in train_seasons]
    y_parts = [season_cache[s][1] for s in train_seasons]
    x_train = pd.concat(x_parts, ignore_index=True)
    y_train = pd.concat(y_parts, ignore_index=True)
    return x_train, y_train


def build_winprob_models(
    input_root: str,
    output_root: str,
    phase: str = "regular",
    season: Optional[str] = None,
    lookback_seasons: int = LOOKBACK_SEASONS_DEFAULT,
    overwrite: bool = False,
) -> int:
    _assert_required_sklearn_runtime()

    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    seasons = discover_seasons(input_path, phase)
    if not seasons:
        print(f"[wpm] No stacked winprob CSVs found under: {input_path}")
        return 1

    if season is not None and season not in seasons:
        print(f"[wpm] Requested season not found in input files: {season}")
        return 1

    target_seasons = [season] if season else seasons

    cache: "OrderedDict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, int]]" = OrderedDict()
    max_cache_items = max(lookback_seasons + 2, 4)

    for target_season in target_seasons:
        train_seasons = _training_window(seasons, target_season, lookback_seasons)
        out_file = output_path / f"wpm_{target_season}.json"
        if out_file.exists() and not overwrite:
            print(f"[wpm] Skip existing artifact (use --overwrite): {out_file}")
            continue

        needed = train_seasons + [target_season]
        for s in needed:
            if s not in cache:
                cache[s] = load_season_matrix(input_path, s, phase)
                while len(cache) > max_cache_items:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(s)

        x_train, y_train_sign = _concat_training_data(cache, train_seasons)
        x_test, _, y_test_bin, test_rows_raw = cache[target_season]

        models = _train_models(x_train, y_train_sign)
        oos_metrics = _evaluate_models(models, x_test, y_test_bin)

        encoded_models: Dict[str, str] = {}
        model_sizes: Dict[str, Dict[str, int]] = {}
        for name, model in models.items():
            payload, sizes = _encode_model(model)
            encoded_models[name] = payload
            model_sizes[name] = sizes

        artifact = {
            "schema_version": 1,
            "season": target_season,
            "phase": phase,
            "library_versions": _runtime_lib_versions(),
            "trained_on_seasons": train_seasons,
            "lookback_seasons": lookback_seasons,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "test_rows_raw_before_feature_clean": test_rows_raw,
            "target_positive_rate_train": float((y_train_sign == 1).mean()),
            "target_positive_rate_test": float(y_test_bin.mean()),
            "feature_spec": {
                "required_input_columns": WINPROB_REQUIRED_COLUMNS,
                "engineered_features": FEATURE_COLUMNS,
                "notes": {
                    "game_seconds_left": "((4 - quarter) * 720 + seconds_left) for regulation; 0 prior quarters in OT",
                    "possession_numeric": "home=1, road=-1, unknown=0",
                    "target": "final_score_diff > 0 => home win",
                },
            },
            "model_hyperparameters": {
                LOGISTIC_NAME: {
                    "polynomial_degree": 2,
                    "interaction_only": True,
                    "include_bias": False,
                    "logistic_solver": "liblinear",
                    "logistic_max_iter": 1000,
                    "random_state": RANDOM_STATE,
                },
                HGB_NAME: {
                    "max_depth": 6,
                    "random_state": RANDOM_STATE,
                },
                RF_NAME: {
                    **SMALL_RF_PARAMS,
                    "calibration_method": "isotonic",
                    "calibration_cv": 5,
                    "calibration_ensemble": False,
                },
                SOFT_VOTE_NAME: {
                    "type": "unweighted_mean_probability",
                    "members": [LOGISTIC_NAME, HGB_NAME, RF_NAME],
                    "symmetry_mode": SYMMETRY_MODE,
                },
            },
            "oos_metrics": oos_metrics,
            "artifact_encoding": "base64+gzip+pickle",
            "artifact_sizes": model_sizes,
            "model_artifacts": encoded_models,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        with out_file.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=True)

        brier = oos_metrics[SOFT_VOTE_NAME]["brier_home_win"]
        print(
            f"[wpm] Wrote {out_file} "
            f"(train_seasons={train_seasons}, train_rows={len(x_train)}, "
            f"test_rows={len(x_test)}, ensemble_brier={brier:.6f})"
        )

    return 0


def _resolve_artifact_path(output_root: str, season: str, phase: str) -> Path:
    base = Path(output_root).expanduser().resolve()
    flat_path = base / f"wpm_{season}.json"
    resolved_flat = _ensure_local_or_remote(flat_path)
    if resolved_flat is not None:
        return resolved_flat
    # Backward compatibility for older artifacts written under <output_root>/<phase>/.
    phase_path = base / phase / f"wpm_{season}.json"
    resolved_phase = _ensure_local_or_remote(phase_path)
    if resolved_phase is not None:
        return resolved_phase
    raise FileNotFoundError(f"No WPM artifact found for season={season} under {base}")


def list_wpm_seasons(output_root: str, phase: str = "regular") -> List[str]:
    root = Path(output_root).expanduser().resolve()
    # Prefer flat files; include legacy <phase>/ files for compatibility.
    candidates = list(root.glob("wpm_*.json")) + list((root / phase).glob("wpm_*.json"))
    seasons = set()
    for p in candidates:
        stem = p.stem
        if not stem.startswith("wpm_"):
            continue
        seasons.add(stem[4:])
    if not seasons:
        seasons.update(_discover_remote_wpm_seasons())
    return sorted(seasons, key=_season_sort_key)


@lru_cache(maxsize=64)
def _load_artifact_with_models(output_root: str, phase: str, season: str) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    artifact_path = _resolve_artifact_path(output_root=output_root, season=season, phase=phase)
    with artifact_path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)

    _assert_required_sklearn_runtime()
    _assert_artifact_sklearn_version(artifact_path=artifact_path, artifact=artifact)

    model_payloads = artifact.get("model_artifacts", {})
    missing = [k for k in [LOGISTIC_NAME, HGB_NAME, RF_NAME] if k not in model_payloads]
    if missing:
        raise ValueError(f"Artifact missing model payloads: {missing}")

    models = {k: _decode_model(v) for k, v in model_payloads.items()}
    _ = sum(_patch_model_compat(model) for model in models.values())
    return artifact_path, artifact, models


@lru_cache(maxsize=32)
def _load_actual_frame(input_root: str, phase: str, season: str) -> pd.DataFrame:
    input_path = Path(input_root).expanduser().resolve()
    csv_path = _find_stacked_path(input_path, season, phase)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find stacked winprob base for season={season} under {input_root}")

    cols = [
        "gameid",
        "home",
        "road",
        "quarter",
        "seconds_left",
        "differential",
        "possession",
        "final_score_diff",
    ]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df.dropna(subset=["gameid", "quarter", "seconds_left", "differential", "final_score_diff"])
    df = df.copy()
    df["_game_key"] = df["gameid"].map(_normalize_game_key)
    df["_possession_numeric"] = np.select(
        [df["possession"] == df["home"], df["possession"] == df["road"]],
        [1, -1],
        default=0,
    ).astype(int)
    return df


def _lookup_actual_result(
    input_root: str,
    phase: str,
    season: str,
    game_id: str,
    quarter: int,
    seconds_left: float,
    differential: float,
    possession_numeric: int,
) -> Optional[Dict[str, Any]]:
    frame = _load_actual_frame(input_root=input_root, phase=phase, season=season)
    game_key = _normalize_game_key(game_id)
    game_rows = frame[frame["_game_key"] == game_key]
    if game_rows.empty:
        return None

    exact = game_rows[
        (game_rows["quarter"].astype(int) == int(quarter))
        & (np.isclose(game_rows["seconds_left"].astype(float), float(seconds_left), atol=0.5))
        & (game_rows["differential"].astype(float) == float(differential))
        & (game_rows["_possession_numeric"].astype(int) == int(possession_numeric))
    ]

    if exact.empty:
        tmp = game_rows.copy()
        tmp["_distance"] = (
            (tmp["quarter"].astype(float) - float(quarter)).abs() * 1000.0
            + (tmp["seconds_left"].astype(float) - float(seconds_left)).abs()
            + (tmp["differential"].astype(float) - float(differential)).abs() * 10.0
            + (tmp["_possession_numeric"].astype(float) - float(possession_numeric)).abs() * 5.0
        )
        matched = tmp.sort_values("_distance", ascending=True).iloc[0]
        match_type = "nearest"
        distance = float(matched["_distance"])
    else:
        matched = exact.iloc[0]
        match_type = "exact"
        distance = 0.0

    final_score_diff = float(matched["final_score_diff"])
    return {
        "game_id": str(matched["gameid"]),
        "home_team": str(matched["home"]),
        "road_team": str(matched["road"]),
        "final_score_diff": final_score_diff,
        "actual_home_win": bool(final_score_diff > 0),
        "actual_result_label": "HOME_WIN" if final_score_diff > 0 else "ROAD_WIN",
        "match_type": match_type,
        "match_distance": distance,
        "matched_state": {
            "quarter": int(matched["quarter"]),
            "seconds_left": float(matched["seconds_left"]),
            "differential": float(matched["differential"]),
            "possession_numeric": int(matched["_possession_numeric"]),
        },
    }


def _lookup_game_state_by_game_seconds_left(
    input_root: str,
    phase: str,
    season: str,
    game_id: str,
    game_seconds_left: float,
) -> Optional[Dict[str, Any]]:
    frame = _load_actual_frame(input_root=input_root, phase=phase, season=season)
    game_key = _normalize_game_key(game_id)
    game_rows = frame[frame["_game_key"] == game_key].copy()
    if game_rows.empty:
        return None

    # Keep the same feature definition used by model training.
    game_rows["_full_q_left"] = np.where(game_rows["quarter"].astype(int) <= 4, 4 - game_rows["quarter"].astype(int), 0)
    game_rows["_game_seconds_left"] = game_rows["_full_q_left"] * 720.0 + game_rows["seconds_left"].astype(float)
    game_rows["_distance"] = (game_rows["_game_seconds_left"] - float(game_seconds_left)).abs()
    game_rows["_is_terminal_non_tie"] = (
        (game_rows["quarter"].astype(int) >= 4)
        & np.isclose(game_rows["seconds_left"].astype(float), 0.0, atol=0.5)
        & (~np.isclose(game_rows["differential"].astype(float), 0.0, atol=1e-9))
    ).astype(int)
    matched = game_rows.sort_values(
        ["_distance", "_is_terminal_non_tie", "quarter"],
        ascending=[True, False, False],
    ).iloc[0]

    final_score_diff = float(matched["final_score_diff"])
    return {
        "requested_game_seconds_left": float(game_seconds_left),
        "matched_game_seconds_left": float(matched["_game_seconds_left"]),
        "distance_seconds": float(matched["_distance"]),
        "game_id": str(matched["gameid"]),
        "home_team": str(matched["home"]),
        "road_team": str(matched["road"]),
        "quarter": int(matched["quarter"]),
        "seconds_left": float(matched["seconds_left"]),
        "differential": float(matched["differential"]),
        "possession_numeric": int(matched["_possession_numeric"]),
        "final_score_diff": final_score_diff,
        "actual_home_win": bool(final_score_diff > 0),
        "actual_result_label": "HOME_WIN" if final_score_diff > 0 else "ROAD_WIN",
    }


def _feature_row(
    quarter: int,
    seconds_left: float,
    differential: float,
    possession_numeric: Optional[int] = None,
    possession: Optional[str] = None,
    home_team: Optional[str] = None,
    road_team: Optional[str] = None,
) -> pd.DataFrame:
    if possession_numeric is None:
        if possession and home_team and road_team:
            if possession == home_team:
                possession_numeric = 1
            elif possession == road_team:
                possession_numeric = -1
            else:
                possession_numeric = 0
        else:
            possession_numeric = 0

    full_q_left = 4 - int(quarter) if int(quarter) <= 4 else 0
    game_seconds_left = full_q_left * 720 + float(seconds_left)

    return pd.DataFrame(
        [
            {
                "game_seconds_left": float(game_seconds_left),
                "current_differential": float(differential),
                "possession_numeric": int(possession_numeric),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def _terminal_home_probability(quarter: int, seconds_left: float, differential: float) -> Optional[float]:
    if int(quarter) < 4:
        return None
    if float(seconds_left) > 0.0:
        return None
    if float(differential) > 0.0:
        return 1.0
    if float(differential) < 0.0:
        return 0.0
    # End of regulation tie can proceed to OT; keep it neutral.
    return 0.5


def _predict_state_probabilities(
    models: Dict[str, Any],
    quarter: int,
    seconds_left: float,
    differential: float,
    possession_numeric: int,
) -> Tuple[float, float, float, float]:
    forced = _terminal_home_probability(quarter=quarter, seconds_left=seconds_left, differential=differential)
    if forced is not None:
        return forced, forced, forced, forced

    x = _feature_row(
        quarter=quarter,
        seconds_left=seconds_left,
        differential=differential,
        possession_numeric=possession_numeric,
    )
    p_lr = float(_home_prob(models[LOGISTIC_NAME], x, symmetry_mode=SYMMETRY_MODE)[0])
    p_hgb = float(_home_prob(models[HGB_NAME], x, symmetry_mode=SYMMETRY_MODE)[0])
    p_rf = float(_home_prob(models[RF_NAME], x, symmetry_mode=SYMMETRY_MODE)[0])
    p_soft_vote = float(np.mean([p_lr, p_hgb, p_rf]))
    return (
        float(np.clip(p_lr, 0.0, 1.0)),
        float(np.clip(p_hgb, 0.0, 1.0)),
        float(np.clip(p_rf, 0.0, 1.0)),
        float(np.clip(p_soft_vote, 0.0, 1.0)),
    )


def predict_home_winprob_batch(
    season: str,
    output_root: str,
    phase: str,
    states: List[Dict[str, Any]],
) -> List[Optional[float]]:
    """Predict home win probability for multiple game states."""
    if not states:
        return []

    _, _, models = _load_artifact_with_models(
        output_root=output_root,
        phase=str(phase or "regular"),
        season=season,
    )

    rows: List[Dict[str, float]] = []
    row_indices: List[int] = []
    probs: List[Optional[float]] = [None] * len(states)

    for idx, state in enumerate(states):
        try:
            quarter = int(state.get("quarter"))
            seconds_left = float(state.get("seconds_left"))
            differential = float(state.get("differential"))
            possession_numeric = int(state.get("possession_numeric", 0))
        except (TypeError, ValueError):
            continue

        if quarter <= 0:
            continue

        max_seconds = 300.0 if quarter > 4 else 720.0
        seconds_left = min(max(seconds_left, 0.0), max_seconds)
        possession_numeric = max(-1, min(1, possession_numeric))
        full_q_left = 4 - quarter if quarter <= 4 else 0

        forced = _terminal_home_probability(quarter=quarter, seconds_left=seconds_left, differential=differential)
        if forced is not None:
            probs[idx] = forced
            continue

        rows.append(
            {
                "game_seconds_left": float(full_q_left * 720.0 + seconds_left),
                "current_differential": float(differential),
                "possession_numeric": float(possession_numeric),
            }
        )
        row_indices.append(idx)

    if not rows:
        return probs

    x = pd.DataFrame(rows, columns=FEATURE_COLUMNS).astype(float)
    p_lr = _home_prob(models[LOGISTIC_NAME], x, symmetry_mode=SYMMETRY_MODE)
    p_hgb = _home_prob(models[HGB_NAME], x, symmetry_mode=SYMMETRY_MODE)
    p_rf = _home_prob(models[RF_NAME], x, symmetry_mode=SYMMETRY_MODE)
    p_soft_vote = np.mean(np.column_stack([p_lr, p_hgb, p_rf]), axis=1)

    for idx, p_home in zip(row_indices, p_soft_vote):
        probs[idx] = float(np.clip(p_home, 0.0, 1.0))

    return probs


def predict_winprob(
    season: str,
    output_root: str,
    phase: str,
    quarter: int,
    seconds_left: float,
    differential: float,
    possession_numeric: Optional[int] = None,
    possession: Optional[str] = None,
    home_team: Optional[str] = None,
    road_team: Optional[str] = None,
) -> int:
    try:
        artifact_path, _, models = _load_artifact_with_models(output_root=output_root, phase=phase, season=season)
    except Exception as exc:
        print(f"[wpm] {exc}")
        return 1

    x = _feature_row(
        quarter=quarter,
        seconds_left=seconds_left,
        differential=differential,
        possession_numeric=possession_numeric,
        possession=possession,
        home_team=home_team,
        road_team=road_team,
    )
    p_lr, p_hgb, p_rf, p_soft_vote = _predict_state_probabilities(
        models=models,
        quarter=int(quarter),
        seconds_left=float(seconds_left),
        differential=float(differential),
        possession_numeric=int(x.iloc[0]["possession_numeric"]),
    )

    result = {
        "season": season,
        "artifact_path": str(artifact_path),
        "inputs": {
            "quarter": int(quarter),
            "seconds_left": float(seconds_left),
            "differential": float(differential),
            "possession_numeric": int(x.iloc[0]["possession_numeric"]),
            "home_team": home_team,
            "road_team": road_team,
            "possession": possession,
        },
        "probabilities": {
            LOGISTIC_NAME: p_lr,
            HGB_NAME: p_hgb,
            RF_NAME: p_rf,
            SOFT_VOTE_NAME: p_soft_vote,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


def forecast_with_actual(
    season: str,
    phase: str,
    output_root: str,
    input_root: str,
    game_id: str,
    quarter: int,
    seconds_left: float,
    differential: float,
    possession_numeric: int,
) -> Dict[str, Any]:
    artifact_path, artifact, models = _load_artifact_with_models(output_root=output_root, phase=phase, season=season)
    p_lr, p_hgb, p_rf, p_soft_vote = _predict_state_probabilities(
        models=models,
        quarter=int(quarter),
        seconds_left=float(seconds_left),
        differential=float(differential),
        possession_numeric=int(possession_numeric),
    )

    actual = _lookup_actual_result(
        input_root=input_root,
        phase=phase,
        season=season,
        game_id=game_id,
        quarter=quarter,
        seconds_left=seconds_left,
        differential=differential,
        possession_numeric=possession_numeric,
    )

    return {
        "season": season,
        "phase": phase,
        "game_id": str(game_id),
        "artifact_path": str(artifact_path),
        "trained_on_seasons": artifact.get("trained_on_seasons", []),
        "inputs": {
            "quarter": int(quarter),
            "seconds_left": float(seconds_left),
            "differential": float(differential),
            "possession_numeric": int(possession_numeric),
        },
        "forecast": {
            LOGISTIC_NAME: p_lr,
            HGB_NAME: p_hgb,
            RF_NAME: p_rf,
            SOFT_VOTE_NAME: p_soft_vote,
            "predicted_label": "HOME_WIN" if p_soft_vote >= 0.5 else "ROAD_WIN",
        },
        "actual": actual,
    }


def forecast_from_game_seconds_left(
    season: str,
    phase: str,
    output_root: str,
    input_root: str,
    game_id: str,
    game_seconds_left: float,
) -> Dict[str, Any]:
    artifact_path, artifact, models = _load_artifact_with_models(output_root=output_root, phase=phase, season=season)

    state = _lookup_game_state_by_game_seconds_left(
        input_root=input_root,
        phase=phase,
        season=season,
        game_id=game_id,
        game_seconds_left=game_seconds_left,
    )
    if state is None:
        raise ValueError(f"Game ID not found in season {season}: {game_id}")

    p_lr, p_hgb, p_rf, p_soft_vote = _predict_state_probabilities(
        models=models,
        quarter=int(state["quarter"]),
        seconds_left=float(state["seconds_left"]),
        differential=float(state["differential"]),
        possession_numeric=int(state["possession_numeric"]),
    )

    return {
        "season": season,
        "phase": phase,
        "game_id": str(game_id),
        "artifact_path": str(artifact_path),
        "trained_on_seasons": artifact.get("trained_on_seasons", []),
        "state_lookup": state,
        "forecast": {
            LOGISTIC_NAME: p_lr,
            HGB_NAME: p_hgb,
            RF_NAME: p_rf,
            SOFT_VOTE_NAME: p_soft_vote,
            "predicted_label": "HOME_WIN" if p_soft_vote >= 0.5 else "ROAD_WIN",
        },
    }


def forecast_hypothetical(
    season: str,
    phase: str,
    output_root: str,
    quarter: int,
    seconds_left: float,
    differential: float,
    possession_numeric: int,
) -> Dict[str, Any]:
    artifact_path, artifact, models = _load_artifact_with_models(output_root=output_root, phase=phase, season=season)
    p_lr, p_hgb, p_rf, p_soft_vote = _predict_state_probabilities(
        models=models,
        quarter=int(quarter),
        seconds_left=float(seconds_left),
        differential=float(differential),
        possession_numeric=int(possession_numeric),
    )

    return {
        "season": season,
        "phase": phase,
        "artifact_path": str(artifact_path),
        "trained_on_seasons": artifact.get("trained_on_seasons", []),
        "inputs": {
            "quarter": int(quarter),
            "seconds_left": float(seconds_left),
            "differential": float(differential),
            "possession_numeric": int(possession_numeric),
        },
        "forecast": {
            LOGISTIC_NAME: p_lr,
            HGB_NAME: p_hgb,
            RF_NAME: p_rf,
            SOFT_VOTE_NAME: p_soft_vote,
            "predicted_label": "HOME_WIN" if p_soft_vote >= 0.5 else "ROAD_WIN",
        },
    }
