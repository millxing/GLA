#!/usr/bin/env python3
"""
Generate polished PNG charts for the Lakers blog post.

Usage:
    python backend/admin/generate_lakers_blog_charts.py
    python backend/admin/generate_lakers_blog_charts.py --team LAL --season 2025-26
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BG = "#faf7f0"
TEXT = "#201a16"
MUTED = "#6c6258"
GRID = "#d9d0c3"
SOFT_GRID = "#ebe3d8"
PANEL = "#fffdf8"
LAKERS_GOLD = "#f3bf3b"
LAKERS_PURPLE = "#552583"
NEGATIVE = "#d46a4c"
POSITIVE = "#2b7a62"
NEUTRAL = "#9b8f82"
POINT_FILL = "#c8beb2"

GEORGIA = Path("/System/Library/Fonts/Supplemental/Georgia.ttf")
GEORGIA_BOLD = Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf")
MONO = Path("/System/Library/Fonts/Menlo.ttc")
DEFAULT_REPO_DIR = Path("/Users/robschoen/Dropbox/CC/NBA_Data").resolve()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(GEORGIA_BOLD if bold else GEORGIA), size=size)


def _mono_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    index = 1 if bold else 0
    return ImageFont.truetype(str(MONO), size=size, index=index)


def _build_team_rows(game_logs_path: Path, advanced_path: Path, linescores_path: Path | None = None) -> pd.DataFrame:
    game_logs = pd.read_csv(game_logs_path)
    advanced = pd.read_csv(advanced_path)

    rows: list[dict[str, object]] = []
    for _, row in game_logs.iterrows():
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "game_date": row["game_date"],
                "team": row["team_abbreviation_home"],
                "opponent": row["team_abbreviation_road"],
                "home_road": "home",
                "pts": float(row["pts_home"]),
                "opp_pts": float(row["pts_road"]),
                "win": 1 if row["wl_home"] == "W" else 0,
            }
        )
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "game_date": row["game_date"],
                "team": row["team_abbreviation_road"],
                "opponent": row["team_abbreviation_home"],
                "home_road": "road",
                "pts": float(row["pts_road"]),
                "opp_pts": float(row["pts_home"]),
                "win": 1 if row["wl_home"] == "L" else 0,
            }
        )
    team_rows = pd.DataFrame(rows)

    adv_rows: list[dict[str, object]] = []
    for _, row in advanced.iterrows():
        adv_rows.append(
            {
                "game_id": int(row["game_id"]),
                "team": row["team_abbreviation_home"],
                "poss": float(row["possessions_home"]),
                "opp_poss": float(row["possessions_road"]),
                "minutes": float(row["minutes_home"]),
            }
        )
        adv_rows.append(
            {
                "game_id": int(row["game_id"]),
                "team": row["team_abbreviation_road"],
                "poss": float(row["possessions_road"]),
                "opp_poss": float(row["possessions_home"]),
                "minutes": float(row["minutes_road"]),
            }
        )
    team_rows = team_rows.merge(pd.DataFrame(adv_rows), on=["game_id", "team"], how="left")

    if linescores_path is None:
        return team_rows

    linescores = pd.read_csv(linescores_path)
    line_rows: list[dict[str, object]] = []
    for _, row in linescores.iterrows():
        line_rows.append(
            {
                "game_id": int(row["game_id"]),
                "team": row["team_abbreviation_home"],
                "q1": float(row["pts_qtr1_home"]),
                "q2": float(row["pts_qtr2_home"]),
                "q3": float(row["pts_qtr3_home"]),
                "q4": float(row["pts_qtr4_home"]),
                "opp_q1": float(row["pts_qtr1_road"]),
                "opp_q2": float(row["pts_qtr2_road"]),
                "opp_q3": float(row["pts_qtr3_road"]),
                "opp_q4": float(row["pts_qtr4_road"]),
            }
        )
        line_rows.append(
            {
                "game_id": int(row["game_id"]),
                "team": row["team_abbreviation_road"],
                "q1": float(row["pts_qtr1_road"]),
                "q2": float(row["pts_qtr2_road"]),
                "q3": float(row["pts_qtr3_road"]),
                "q4": float(row["pts_qtr4_road"]),
                "opp_q1": float(row["pts_qtr1_home"]),
                "opp_q2": float(row["pts_qtr2_home"]),
                "opp_q3": float(row["pts_qtr3_home"]),
                "opp_q4": float(row["pts_qtr4_home"]),
            }
        )
    return team_rows.merge(pd.DataFrame(line_rows), on=["game_id", "team"], how="left")


def _season_summary(df: pd.DataFrame) -> dict[str, float]:
    games = float(len(df))
    wins = float(df["win"].sum()) if "win" in df else np.nan
    pts = float(df["pts"].sum())
    opp_pts = float(df["opp_pts"].sum())
    poss = float(df["poss"].sum())
    opp_poss = float(df["opp_poss"].sum())
    ortg = 100.0 * pts / poss if poss else np.nan
    drtg = 100.0 * opp_pts / opp_poss if opp_poss else np.nan
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins if np.isfinite(wins) else np.nan,
        "win_pct": wins / games if games and np.isfinite(wins) else np.nan,
        "pts": pts,
        "opp_pts": opp_pts,
        "poss": poss,
        "opp_poss": opp_poss,
        "minutes": float(df["minutes"].sum()) if "minutes" in df else np.nan,
        "ortg": ortg,
        "drtg": drtg,
        "net": ortg - drtg if np.isfinite(ortg) and np.isfinite(drtg) else np.nan,
        "margin_per_game": (pts - opp_pts) / games if games else np.nan,
    }


def _load_context(repo_dir: Path, team: str, season: str) -> dict[str, object]:
    full = _build_team_rows(
        repo_dir / f"team_game_logs_{season}.csv",
        repo_dir / f"box_score_advanced_{season}.csv",
        repo_dir / f"linescores_{season}.csv",
    )
    non_garbage = _build_team_rows(
        repo_dir / f"team_game_logs_garbage_filtered_{season}.csv",
        repo_dir / f"box_score_advanced_garbage_filtered_{season}.csv",
    )
    clutch = _build_team_rows(
        repo_dir / f"team_game_logs_clutch_{season}.csv",
        repo_dir / f"box_score_advanced_clutch_{season}.csv",
    )

    full_team = full[full["team"] == team].copy().sort_values("game_date")
    non_garbage_team = non_garbage[non_garbage["team"] == team].copy().sort_values("game_date")
    clutch_team = clutch[clutch["team"] == team].copy().sort_values("game_date")

    full_team = full_team.merge(
        non_garbage_team[["game_id", "pts", "opp_pts", "poss", "opp_poss", "minutes"]].rename(
            columns={
                "pts": "ng_pts",
                "opp_pts": "ng_opp_pts",
                "poss": "ng_poss",
                "opp_poss": "ng_opp_poss",
                "minutes": "ng_minutes",
            }
        ),
        on="game_id",
        how="left",
    )
    full_team = full_team.merge(
        clutch_team[["game_id", "pts", "opp_pts", "poss", "opp_poss", "minutes"]].rename(
            columns={
                "pts": "cl_pts",
                "opp_pts": "cl_opp_pts",
                "poss": "cl_poss",
                "opp_poss": "cl_opp_poss",
                "minutes": "cl_minutes",
            }
        ),
        on="game_id",
        how="left",
    ).fillna(0)

    full_team["margin"] = full_team["pts"] - full_team["opp_pts"]
    full_team["margin_after_3"] = (
        full_team["q1"]
        + full_team["q2"]
        + full_team["q3"]
        - full_team["opp_q1"]
        - full_team["opp_q2"]
        - full_team["opp_q3"]
    )
    full_team["q4_margin"] = full_team["q4"] - full_team["opp_q4"]
    full_team["garbage_pts"] = full_team["pts"] - full_team["ng_pts"]
    full_team["garbage_opp_pts"] = full_team["opp_pts"] - full_team["ng_opp_pts"]
    full_team["garbage_poss"] = full_team["poss"] - full_team["ng_poss"]
    full_team["garbage_opp_poss"] = full_team["opp_poss"] - full_team["ng_opp_poss"]
    full_team["garbage_minutes"] = full_team["minutes"] - full_team["ng_minutes"]

    team_table = pd.DataFrame(
        [{"team": team_name, **_season_summary(group)} for team_name, group in full.groupby("team")]
    )
    team_table["net_rank"] = team_table["net"].rank(method="min", ascending=False)
    team_table["win_rank"] = team_table["wins"].rank(method="min", ascending=False)

    q4_table = (
        full.assign(
            margin=lambda frame: frame["pts"] - frame["opp_pts"],
            margin_after_3=lambda frame: frame["q1"]
            + frame["q2"]
            + frame["q3"]
            - frame["opp_q1"]
            - frame["opp_q2"]
            - frame["opp_q3"],
            q4_margin=lambda frame: frame["q4"] - frame["opp_q4"],
        )
        .groupby("team")
        .agg(
            q4_margin=("q4_margin", "mean"),
            first3_margin=("margin_after_3", "mean"),
            final_margin=("margin", "mean"),
        )
        .reset_index()
    )
    q4_table["q4_rank"] = q4_table["q4_margin"].rank(method="min", ascending=False)

    full_summary = _season_summary(full_team)
    non_garbage_summary = _season_summary(non_garbage_team)
    clutch_summary = _season_summary(clutch_team)

    garbage_pts = float(full_team["garbage_pts"].sum())
    garbage_opp_pts = float(full_team["garbage_opp_pts"].sum())
    garbage_poss = float(full_team["garbage_poss"].sum())
    garbage_opp_poss = float(full_team["garbage_opp_poss"].sum())
    garbage_summary = {
        "poss": garbage_poss,
        "opp_poss": garbage_opp_poss,
        "minutes": float(full_team["garbage_minutes"].sum()),
        "ortg": 100.0 * garbage_pts / garbage_poss if garbage_poss else np.nan,
        "drtg": 100.0 * garbage_opp_pts / garbage_opp_poss if garbage_opp_poss else np.nan,
        "net": (100.0 * garbage_pts / garbage_poss - 100.0 * garbage_opp_pts / garbage_opp_poss)
        if garbage_poss and garbage_opp_poss
        else np.nan,
    }

    pyth_wins = (full_summary["pts"] ** 14) / ((full_summary["pts"] ** 14) + (full_summary["opp_pts"] ** 14)) * full_summary["games"]

    return {
        "team": team,
        "season": season,
        "full": full_team,
        "full_summary": full_summary,
        "non_garbage_summary": non_garbage_summary,
        "garbage_summary": garbage_summary,
        "clutch_summary": clutch_summary,
        "team_table": team_table,
        "q4_table": q4_table,
        "pyth_wins": pyth_wins,
        "close_record_5": (
            int(full_team[(full_team["margin"].abs() <= 5) & (full_team["win"] == 1)].shape[0]),
            int(full_team[(full_team["margin"].abs() <= 5) & (full_team["win"] == 0)].shape[0]),
        ),
        "comeback_wins": int(full_team[(full_team["margin_after_3"] < 0) & (full_team["win"] == 1)].shape[0]),
        "leading_after_3_record": (
            int(full_team[(full_team["margin_after_3"] > 0) & (full_team["win"] == 1)].shape[0]),
            int(full_team[(full_team["margin_after_3"] > 0) & (full_team["win"] == 0)].shape[0]),
        ),
    }


def _new_canvas(width: int = 1600, height: int = 1000) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BG)
    return image, ImageDraw.Draw(image)


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = [words[0]]
    for word in words[1:]:
        trial = f"{lines[-1]} {word}"
        if _measure(draw, trial, font)[0] <= max_width:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font: ImageFont.FreeTypeFont,
    fill: str = TEXT,
    line_gap: int = 7,
) -> None:
    x0, y0, x1, _ = box
    lines = _wrap_text(draw, text, font, x1 - x0)
    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill)
        y += _measure(draw, line, font)[1] + line_gap


def _draw_title_block(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    title_font = _font(42, bold=True)
    subtitle_font = _font(21)
    draw.text((90, 72), title, font=title_font, fill=TEXT)
    _draw_wrapped_text(draw, subtitle, (90, 130, 1460, 200), font=subtitle_font, fill=MUTED, line_gap=8)


def _draw_footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((90, 948), text, font=_font(15), fill=MUTED)


def _draw_rotated_text(
    image: Image.Image,
    text: str,
    position: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
    angle: int,
) -> None:
    scratch = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    left, top, right, bottom = scratch_draw.textbbox((0, 0), text, font=font)
    text_image = Image.new("RGBA", (right - left + 8, bottom - top + 8), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4, 4), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    image.alpha_composite(rotated, dest=position)


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=24, fill=PANEL, outline=SOFT_GRID, width=2)


def _draw_rich_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str]) -> None:
    _draw_panel(draw, box)
    x0, y0, _, _ = box
    y = y0 + 22
    for idx, line in enumerate(lines):
        font = _font(21 if idx == 0 else 18, bold=(idx == 0))
        draw.text((x0 + 22, y), line, font=font, fill=TEXT if idx == 0 else MUTED)
        y += _measure(draw, line, font)[1] + 10


def _linear_map(value: float, domain: tuple[float, float], target: tuple[float, float]) -> float:
    d0, d1 = domain
    t0, t1 = target
    if d1 == d0:
        return (t0 + t1) / 2.0
    return t0 + (value - d0) * (t1 - t0) / (d1 - d0)


def _signed(value: float) -> str:
    if abs(value) < 0.05:
        return "0.0"
    return f"{value:+.1f}"


def _plot_time_splits(context: dict[str, object], output_path: Path) -> None:
    image = Image.new("RGB", (1600, 840), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "The Lakers change as the game gets more meaningful", font=_mono_font(34, bold=True), fill="#111827")
    draw.text(
        (48, 78),
        "2025-26 net rating by context. Their full-season profile looks mediocre until the leverage rises.",
        font=_mono_font(20),
        fill="#6b7280",
    )

    values = [
        context["full_summary"]["net"],
        context["non_garbage_summary"]["net"],
        context["garbage_summary"]["net"],
        context["clutch_summary"]["net"],
    ]
    labels = ["Overall", "Non-garbage", "Garbage time", "Clutch"]
    shares = [
        100.0,
        100.0 * context["non_garbage_summary"]["minutes"] / context["full_summary"]["minutes"],
        100.0 * context["garbage_summary"]["minutes"] / context["full_summary"]["minutes"],
        100.0 * context["clutch_summary"]["minutes"] / context["full_summary"]["minutes"],
    ]
    colors = ["#93c5fd", "#60a5fa", "#fca5a5", "#1e40af"]

    chart_box = (285, 130, 1540, 700)
    label_x = 60
    x0, y0, x1, y1 = chart_box

    min_value = min(values) - 5.0
    max_value = max(values) + 5.0
    min_value = math.floor(min_value / 5.0) * 5.0
    max_value = math.ceil(max_value / 5.0) * 5.0

    tick_font = _mono_font(17)
    for tick in np.arange(min_value, max_value + 0.1, 5.0):
        x = _linear_map(float(tick), (min_value, max_value), (x0, x1))
        draw.line((x, y0, x, y1), fill="#dbe3ef" if tick != 0 else "#94a3b8", width=2 if tick == 0 else 1)
        tick_text = f"{int(tick):d}"
        tw, th = _measure(draw, tick_text, tick_font)
        draw.text((x - tw / 2, y1 + 12), tick_text, font=tick_font, fill="#6b7280")

    row_height = 124
    bar_height = 68
    label_font = _mono_font(24, bold=True)
    value_font = _mono_font(22, bold=True)
    meta_font = _mono_font(17)
    zero_x = _linear_map(0.0, (min_value, max_value), (x0, x1))

    for idx, (label, value, share, color) in enumerate(zip(labels, values, shares, colors, strict=True)):
        cy = y0 + 52 + idx * row_height
        draw.text((label_x, cy - 12), label, font=label_font, fill="#111827")
        share_text = "all minutes" if idx == 0 else f"{share:.1f}% of minutes"
        draw.text((label_x, cy + 26), share_text, font=meta_font, fill="#6b7280")
        value_x = _linear_map(float(value), (min_value, max_value), (x0, x1))
        left = min(zero_x, value_x)
        right = max(zero_x, value_x)
        draw.rounded_rectangle((left, cy, right, cy + bar_height), radius=22, fill=color)
        draw.rounded_rectangle((left, cy, right, cy + bar_height), radius=22, outline="#ffffff", width=2)
        draw.text(
            (right + 16 if value >= 0 else left - 16, cy + 18),
            _signed(value),
            font=value_font,
            fill="#111827",
            anchor="la" if value >= 0 else "ra",
        )

    draw.text((48, 815), "Source: NBA play-by-play data", font=_mono_font(15), fill="#6b7280")
    image.save(output_path)


def _plot_record_vs_net(context: dict[str, object], output_path: Path) -> None:
    image = Image.new("RGBA", (1600, 840), "#ffffff")
    draw = ImageDraw.Draw(image)
    team = context["team"]
    team_table = context["team_table"].copy()
    lakers = team_table[team_table["team"] == team].iloc[0]

    title_font = _mono_font(34, bold=True)
    subtitle_font = _mono_font(20)
    draw.text((48, 28), "Lakers' record better than expected given net rating", font=title_font, fill="#111827")
    draw.text((48, 78), "Through March 7, 2026.", font=subtitle_font, fill="#6b7280")

    x0, y0, x1, y1 = (140, 125, 1540, 700)
    x_min = math.floor((float(team_table['net'].min()) - 1.5) / 2.0) * 2.0
    x_max = math.ceil((float(team_table['net'].max()) + 1.5) / 2.0) * 2.0
    y_min, y_max = 0.20, 0.80

    axis_font = _mono_font(17)
    for tick in np.arange(x_min, x_max + 0.1, 2.0):
        x = _linear_map(float(tick), (x_min, x_max), (x0, x1))
        draw.line((x, y0, x, y1), fill="#dbe3ef" if tick != 0 else "#94a3b8", width=2 if tick == 0 else 1)
        label = f"{tick:.0f}"
        tw, _ = _measure(draw, label, axis_font)
        draw.text((x - tw / 2, y1 + 10), label, font=axis_font, fill="#6b7280")

    for tick in np.arange(y_min, y_max + 0.001, 0.1):
        y = _linear_map(float(tick), (y_min, y_max), (y1, y0))
        draw.line((x0, y, x1, y), fill="#dbe3ef" if abs(tick - 0.5) > 1e-6 else "#94a3b8", width=2 if abs(tick - 0.5) <= 1e-6 else 1)
        label = f"{tick:.1f}"
        tw, th = _measure(draw, label, axis_font)
        draw.text((x0 - tw - 16, y - th / 2), label, font=axis_font, fill="#6b7280")

    for _, row in team_table.iterrows():
        x = _linear_map(float(row["net"]), (x_min, x_max), (x0, x1))
        y = _linear_map(float(row["win_pct"]), (y_min, y_max), (y1, y0))
        radius = 11 if row["team"] == team else 8
        fill = "#1e40af" if row["team"] == team else "#cbd5e1"
        outline = "#93c5fd" if row["team"] == team else "#ffffff"
        width = 5 if row["team"] == team else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)

    lx = _linear_map(float(lakers["net"]), (x_min, x_max), (x0, x1))
    ly = _linear_map(float(lakers["win_pct"]), (y_min, y_max), (y1, y0))
    callout_x = 220
    callout_y = 170
    draw.line((callout_x + 260, callout_y + 86, lx - 16, ly - 10), fill="#1e40af", width=3)
    draw.rounded_rectangle((callout_x, callout_y, callout_x + 280, callout_y + 120), radius=20, fill="#eff6ff", outline="#bfdbfe", width=2)
    draw.text((callout_x + 20, callout_y + 20), "Lakers", font=_mono_font(24, bold=True), fill="#1e3a8a")
    draw.text(
        (callout_x + 20, callout_y + 60),
        f"{int(lakers['wins'])}-{int(lakers['losses'])}, {_signed(float(lakers['net']))} net rating",
        font=_mono_font(18),
        fill="#334155",
    )

    x_label_font = _mono_font(20, bold=True)
    x_label = "Net rating"
    xw, _ = _measure(draw, x_label, x_label_font)
    draw.text(((x0 + x1 - xw) / 2, 748), x_label, font=x_label_font, fill="#111827")
    _draw_rotated_text(image, "Winning percentage", (34, 355), font=_mono_font(20, bold=True), fill="#111827", angle=90)

    footer_font = _mono_font(15)
    draw.text((48, 815), "Source: NBA team game logs and box score advanced data.", font=footer_font, fill="#6b7280")
    image.convert("RGB").save(output_path)


def _plot_fourth_quarter(context: dict[str, object], output_path: Path) -> None:
    image = Image.new("RGB", (1600, 840), "#ffffff")
    draw = ImageDraw.Draw(image)
    full_team = context["full"]

    title_font = _mono_font(34, bold=True)
    subtitle_font = _mono_font(20)
    draw.text((48, 28), "The Lakers spend three quarters setting up the fourth", font=title_font, fill="#111827")
    draw.text((48, 78), "Average scoring margin by quarter", font=subtitle_font, fill="#6b7280")

    labels = ["1Q", "2Q", "3Q", "4Q"]
    values = [
        float((full_team["q1"] - full_team["opp_q1"]).mean()),
        float((full_team["q2"] - full_team["opp_q2"]).mean()),
        float((full_team["q3"] - full_team["opp_q3"]).mean()),
        float(full_team["q4_margin"].mean()),
    ]
    colors = ["#93c5fd" if value >= 0 else "#fca5a5" for value in values]
    colors[-1] = "#1e40af"

    x0, y0, x1, y1 = (110, 130, 1510, 690)
    y_min = math.floor((min(values) - 0.4) * 2.0) / 2.0
    y_max = math.ceil((max(values) + 0.6) * 2.0) / 2.0
    axis_font = _mono_font(17)

    for tick in np.arange(y_min, y_max + 0.01, 0.5):
        y = _linear_map(float(tick), (y_min, y_max), (y1, y0))
        draw.line((x0, y, x1, y), fill="#dbe3ef" if abs(tick) > 1e-6 else "#94a3b8", width=2 if abs(tick) <= 1e-6 else 1)
        label = f"{tick:.1f}"
        tw, th = _measure(draw, label, axis_font)
        draw.text((x0 - tw - 18, y - th / 2), label, font=axis_font, fill="#6b7280")

    slot_width = (x1 - x0) / len(labels)
    bar_width = 168
    zero_y = _linear_map(0.0, (y_min, y_max), (y1, y0))
    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        cx = x0 + slot_width * idx + slot_width / 2
        y_value = _linear_map(float(value), (y_min, y_max), (y1, y0))
        left = cx - bar_width / 2
        right = cx + bar_width / 2
        top = min(zero_y, y_value)
        bottom = max(zero_y, y_value)
        draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=color)
        draw.rounded_rectangle((left, top, right, bottom), radius=22, outline="#ffffff", width=2)
        label_font = _mono_font(24, bold=True)
        tw, _ = _measure(draw, label, label_font)
        draw.text((cx - tw / 2, y1 + 18), label, font=label_font, fill="#111827")
        value_label = _signed(value)
        value_font = _mono_font(22, bold=True)
        vw, vh = _measure(draw, value_label, value_font)
        text_y = top - vh - 14 if value >= 0 else bottom + 12
        draw.text((cx - vw / 2, text_y), value_label, font=value_font, fill="#111827")

    draw.text((48, 815), "Source: NBA linescore data.", font=_mono_font(15), fill="#6b7280")
    image.save(output_path)


def generate_charts(repo_dir: Path, output_dir: Path, team: str, season: str) -> list[Path]:
    context = _load_context(repo_dir=repo_dir, team=team, season=season)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{team.lower()}_{season}"
    outputs = [
        output_dir / f"{prefix}_blog_time_splits.png",
        output_dir / f"{prefix}_blog_record_vs_net.png",
        output_dir / f"{prefix}_blog_fourth_quarter.png",
    ]
    _plot_time_splits(context, outputs[0])
    _plot_record_vs_net(context, outputs[1])
    _plot_fourth_quarter(context, outputs[2])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished PNG charts for a blog post.")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR, help="Path to the NBA_Data repo.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory for exported images.")
    parser.add_argument("--team", default="LAL", help="Team abbreviation. Default: LAL")
    parser.add_argument("--season", default="2025-26", help="Season string. Default: 2025-26")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_charts(
        repo_dir=args.repo_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        team=args.team.upper(),
        season=args.season,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
