#!/usr/bin/env python3
# Revision 2: includes root RESULTS_* output, per-fold warm-up diagnostics, and CI-not-computed handling.
"""
Diagnostic benchmark: current model vs fair bookmaker odds vs model+odds.

This script is intentionally diagnostic-only. It does not change production
defaults, weights, UI, parsers, or model files.

Method:
- Load the project's historical feature dataset using backtest_report.load_data().
- Create expanding-window, date-safe walk-forward folds.
- In every fold train two separate models:
    A) current FEATURE_COLUMNS (no odds features)
    C) ALL_FEATURE_COLUMNS (includes has_odds + implied 1/X/2)
- Evaluate A, B (fair bookmaker implied probabilities), and C on the exact same
  test rows: only rows with valid historical odds.
- Aggregate out-of-fold predictions per league and per league+season.
- Report Accuracy, LogLoss, multiclass Brier, X precision, X recall, N, and
  paired deltas vs bookmaker odds with bootstrap confidence intervals.

Negative delta LogLoss/Brier vs odds means the candidate is better than odds.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backtest_report import load_data, train_model
from schema import (
    ALL_FEATURE_COLUMNS,
    CLASS_MAP,
    FEATURE_COLUMNS,
    LEAGUE_MAP,
    encode_league,
)

logger = logging.getLogger(__name__)

LEAGUES = ("E0", "E1", "E2", "E3")
VARIANTS = ("Model", "Odds", "Model+Odds")
EPS = 1e-15

_LEAGUE_CODE_BY_ID = {code: name for name, code in LEAGUE_MAP.items()}


def decode_league(value) -> str:
    """Return the football-data league code (E0..E3) for raw or encoded values."""
    if value is None:
        return ""
    text = str(value).strip()
    if text in LEAGUE_MAP:
        return text
    try:
        return _LEAGUE_CODE_BY_ID.get(int(float(text)), "")
    except (TypeError, ValueError):
        return ""


def prepare_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Build the model matrix without mutating grouping columns in df."""
    local = df.copy()
    if "League" in local.columns:
        local["League"] = local["League"].apply(encode_league)
    missing = [c for c in feature_columns if c not in local.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return local[list(feature_columns)]


def valid_odds_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows with usable implied bookmaker probabilities."""
    required = ("has_odds", "ImpliedHome", "ImpliedDraw", "ImpliedAway")
    if any(c not in df.columns for c in required):
        return np.zeros(len(df), dtype=bool)

    has_odds = pd.to_numeric(df["has_odds"], errors="coerce").fillna(0.0).to_numpy()
    probs = (
        df[["ImpliedHome", "ImpliedDraw", "ImpliedAway"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    finite = np.isfinite(probs).all(axis=1)
    positive = (probs > 0.0).all(axis=1)
    row_sum = probs.sum(axis=1)
    sane_sum = np.isfinite(row_sum) & (row_sum > 0.0)

    return (has_odds > 0.5) & finite & positive & sane_sum


def normalized_odds_probs(df: pd.DataFrame) -> np.ndarray:
    """Normalized [H, D, A] bookmaker probabilities for valid rows."""
    probs = (
        df[["ImpliedHome", "ImpliedDraw", "ImpliedAway"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    row_sum = probs.sum(axis=1, keepdims=True)
    if (row_sum <= 0).any() or not np.isfinite(row_sum).all():
        raise ValueError("Invalid implied odds probabilities in evaluation sample")
    return probs / row_sum


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    onehot = np.eye(3, dtype=float)[y_true]
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def metric_row(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float | int]:
    """Metrics for one variant on one fixed sample."""
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty sample")

    pred = np.argmax(y_proba, axis=1)
    return {
        "N": int(len(y_true)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "LogLoss": float(log_loss(y_true, y_proba, labels=[0, 1, 2])),
        "Brier": multiclass_brier(y_true, y_proba),
        "X_precision": float(
            precision_score(y_true, pred, labels=[1], average="macro", zero_division=0)
        ),
        "X_recall": float(
            recall_score(y_true, pred, labels=[1], average="macro", zero_division=0)
        ),
    }


def per_match_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    p_true = y_proba[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(p_true, EPS, 1.0))


def per_match_brier(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    onehot = np.eye(3, dtype=float)[y_true]
    return np.sum((y_proba - onehot) ** 2, axis=1)


def paired_bootstrap_delta_ci(
    y_true: np.ndarray,
    candidate_proba: np.ndarray,
    odds_proba: np.ndarray,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """95% paired bootstrap CI for candidate - odds."""
    if n_bootstrap <= 0 or len(y_true) < 2:
        return float("nan"), float("nan")

    if metric == "logloss":
        delta = per_match_logloss(y_true, candidate_proba) - per_match_logloss(
            y_true, odds_proba
        )
    elif metric == "brier":
        delta = per_match_brier(y_true, candidate_proba) - per_match_brier(
            y_true, odds_proba
        )
    else:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")

    rng = np.random.default_rng(seed)
    n = len(delta)
    means: list[np.ndarray] = []
    remaining = n_bootstrap
    chunk_size = max(1, min(250, n_bootstrap))

    while remaining:
        k = min(chunk_size, remaining)
        idx = rng.integers(0, n, size=(k, n))
        means.append(delta[idx].mean(axis=1))
        remaining -= k

    boot = np.concatenate(means)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def date_safe_walk_forward_folds(
    df: pd.DataFrame,
    *,
    n_segments: int = 4,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward folds without splitting a calendar date.

    Segment 0 is training-only. For segment i>=1:
      train = all earlier date segments
      test  = date segment i
    """
    if n_segments < 3:
        raise ValueError("n_segments must be >= 3")

    dates = pd.to_datetime(df["Date"], errors="coerce")
    unique_dates = np.array(sorted(pd.unique(dates.dropna())))
    if len(unique_dates) < n_segments:
        raise ValueError(
            f"Need at least {n_segments} unique dates, got {len(unique_dates)}"
        )

    segments = [seg for seg in np.array_split(unique_dates, n_segments) if len(seg)]
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    date_values = dates.to_numpy()

    for fold_idx in range(1, len(segments)):
        train_dates = np.concatenate(segments[:fold_idx])
        test_dates = segments[fold_idx]

        train_mask = np.isin(date_values, train_dates)
        test_mask = np.isin(date_values, test_dates)

        if train_mask.any() and test_mask.any():
            folds.append((fold_idx, train_mask, test_mask))

    return folds


def _prediction_frame_for_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    fold_idx: int,
) -> pd.DataFrame:
    """Train both variants and return paired predictions for valid-odds test rows."""
    base_model = train_model(df_train, feature_columns=list(FEATURE_COLUMNS))
    odds_model = train_model(df_train, feature_columns=list(ALL_FEATURE_COLUMNS))
    if base_model is None or odds_model is None:
        raise RuntimeError(f"Fold {fold_idx}: model training failed")

    mask = valid_odds_mask(df_test)
    eval_df = df_test.loc[mask].copy()
    if eval_df.empty:
        logger.warning("Fold %d: no valid-odds rows in test segment", fold_idx)
        return pd.DataFrame()

    y_true = eval_df["FTR"].map(CLASS_MAP)
    if y_true.isna().any():
        raise ValueError(f"Fold {fold_idx}: unknown FTR label in evaluation sample")
    y_true_arr = y_true.to_numpy(dtype=int)

    base_x = prepare_features(eval_df, FEATURE_COLUMNS)
    odds_x = prepare_features(eval_df, ALL_FEATURE_COLUMNS)

    p_model = np.asarray(base_model.predict_proba(base_x), dtype=float)
    p_model_odds = np.asarray(odds_model.predict_proba(odds_x), dtype=float)
    p_odds = normalized_odds_probs(eval_df)

    if not (
        p_model.shape == p_model_odds.shape == p_odds.shape == (len(eval_df), 3)
    ):
        raise ValueError(f"Fold {fold_idx}: prediction shape mismatch")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(eval_df["Date"]).to_numpy(),
            "League": eval_df["League"].astype(str).to_numpy(),
            "Season": eval_df.get(
                "Season", pd.Series(["UNK"] * len(eval_df), index=eval_df.index)
            ).astype(str).to_numpy(),
            "HomeTeam": eval_df.get(
                "HomeTeam", pd.Series([""] * len(eval_df), index=eval_df.index)
            ).to_numpy(),
            "AwayTeam": eval_df.get(
                "AwayTeam", pd.Series([""] * len(eval_df), index=eval_df.index)
            ).to_numpy(),
            "FTR": eval_df["FTR"].astype(str).to_numpy(),
            "y_true": y_true_arr,
            "fold": fold_idx,
            "train_N": len(df_train),
            "test_N": len(df_test),
            "paired_N": len(eval_df),
        }
    )

    for prefix, probs in (
        ("model", p_model),
        ("odds", p_odds),
        ("model_odds", p_model_odds),
    ):
        out[f"{prefix}_H"] = probs[:, 0]
        out[f"{prefix}_D"] = probs[:, 1]
        out[f"{prefix}_A"] = probs[:, 2]

    return out


def run_diagnostic(
    df: pd.DataFrame,
    *,
    n_segments: int = 4,
) -> pd.DataFrame:
    """Generate paired out-of-fold predictions."""
    if df.empty:
        raise ValueError("Empty dataset")

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date", "FTR", "League"]).sort_values(
        "Date"
    ).reset_index(drop=True)
    work["League"] = work["League"].apply(decode_league)
    work = work[work["League"].isin(LEAGUES)].reset_index(drop=True)

    folds = date_safe_walk_forward_folds(work, n_segments=n_segments)
    if not folds:
        raise ValueError("No walk-forward folds could be created")

    frames: list[pd.DataFrame] = []
    for fold_idx, train_mask, test_mask in folds:
        df_train = work.loc[train_mask].copy()
        df_test = work.loc[test_mask].copy()

        logger.info(
            "Fold %d: train=%d (%s..%s), test=%d (%s..%s)",
            fold_idx,
            len(df_train),
            df_train["Date"].min().date(),
            df_train["Date"].max().date(),
            len(df_test),
            df_test["Date"].min().date(),
            df_test["Date"].max().date(),
        )

        if len(df_train) < 100:
            logger.warning("Fold %d skipped: train sample too small", fold_idx)
            continue

        frame = _prediction_frame_for_fold(
            df_train,
            df_test,
            fold_idx=fold_idx,
        )
        if not frame.empty:
            logger.info(
                "Fold %d: paired evaluation rows with odds=%d", fold_idx, len(frame)
            )
            frames.append(frame)

    if not frames:
        raise ValueError("No paired out-of-fold predictions were produced")

    predictions = pd.concat(frames, ignore_index=True)

    required_prob_cols = [
        "model_H", "model_D", "model_A",
        "odds_H", "odds_D", "odds_A",
        "model_odds_H", "model_odds_D", "model_odds_A",
    ]
    if predictions[required_prob_cols].isna().any().any():
        raise ValueError("Paired prediction frame contains missing probabilities")

    return predictions


def _proba_for_variant(group: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == "Model":
        cols = ["model_H", "model_D", "model_A"]
    elif variant == "Odds":
        cols = ["odds_H", "odds_D", "odds_A"]
    elif variant == "Model+Odds":
        cols = ["model_odds_H", "model_odds_D", "model_odds_A"]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return group[cols].to_numpy(dtype=float)


def summarize_group(
    group: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    y_true = group["y_true"].to_numpy(dtype=int)
    p_odds = _proba_for_variant(group, "Odds")
    odds_metrics = metric_row(y_true, p_odds)

    rows: list[dict] = []
    for variant_idx, variant in enumerate(VARIANTS):
        p = _proba_for_variant(group, variant)
        m = metric_row(y_true, p)

        if variant == "Odds":
            delta_ll = 0.0
            delta_br = 0.0
            ll_lo = ll_hi = 0.0
            br_lo = br_hi = 0.0
        else:
            delta_ll = m["LogLoss"] - odds_metrics["LogLoss"]
            delta_br = m["Brier"] - odds_metrics["Brier"]
            ll_lo, ll_hi = paired_bootstrap_delta_ci(
                y_true,
                p,
                p_odds,
                metric="logloss",
                n_bootstrap=bootstrap,
                seed=seed + variant_idx * 1009,
            )
            br_lo, br_hi = paired_bootstrap_delta_ci(
                y_true,
                p,
                p_odds,
                metric="brier",
                n_bootstrap=bootstrap,
                seed=seed + variant_idx * 2003,
            )

        rows.append(
            {
                "Variant": variant,
                **m,
                "Delta_LogLoss_vs_Odds": float(delta_ll),
                "Delta_Brier_vs_Odds": float(delta_br),
                "Delta_LogLoss_CI95_L": float(ll_lo),
                "Delta_LogLoss_CI95_U": float(ll_hi),
                "Delta_Brier_CI95_L": float(br_lo),
                "Delta_Brier_CI95_U": float(br_hi),
            }
        )
    return rows


def build_summary_tables(
    predictions: pd.DataFrame,
    *,
    bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    league_rows: list[dict] = []
    season_rows: list[dict] = []

    for league, group in predictions.groupby("League", sort=True):
        for row in summarize_group(group, bootstrap=bootstrap, seed=seed):
            league_rows.append({"League": league, **row})

    for (league, season), group in predictions.groupby(
        ["League", "Season"], sort=True, dropna=False
    ):
        for row in summarize_group(
            group,
            bootstrap=bootstrap,
            seed=seed + sum(ord(ch) for ch in f"{league}:{season}"),
        ):
            season_rows.append(
                {"League": league, "Season": str(season), **row}
            )

    return pd.DataFrame(league_rows), pd.DataFrame(season_rows)


def build_fold_table(
    predictions: pd.DataFrame,
    *,
    bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize every walk-forward fold on the paired odds-valid sample."""
    rows: list[dict] = []

    for fold_idx, group in predictions.groupby("fold", sort=True):
        for col in ("train_N", "test_N", "paired_N"):
            if col not in group.columns:
                raise ValueError(f"Missing fold diagnostic column: {col}")
            if group[col].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Fold {fold_idx}: inconsistent {col} values in paired predictions"
                )

        train_n = int(group["train_N"].iloc[0])
        test_n = int(group["test_N"].iloc[0])
        paired_n = int(group["paired_N"].iloc[0])
        if paired_n != len(group):
            raise ValueError(
                f"Fold {fold_idx}: paired_N={paired_n} but prediction rows={len(group)}"
            )

        for row in summarize_group(
            group,
            bootstrap=bootstrap,
            seed=seed + int(fold_idx) * 7919,
        ):
            rows.append(
                {
                    "Fold": int(fold_idx),
                    "Train_N": train_n,
                    "Test_N": test_n,
                    "Paired_N": paired_n,
                    **row,
                }
            )

    return pd.DataFrame(rows)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(f):
        return "NA"
    return f"{f:.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        vals: list[str] = []
        for c in columns:
            if c in ("Fold", "Train_N", "Test_N", "Paired_N", "N"):
                vals.append(str(int(row[c])))
            elif c in ("League", "Season", "Variant"):
                vals.append(str(row[c]))
            else:
                vals.append(_fmt(row[c]))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _evidence_label(row: pd.Series) -> str:
    if row["Variant"] == "Odds":
        return "baseline"

    d_ll = float(row["Delta_LogLoss_vs_Odds"])
    d_br = float(row["Delta_Brier_vs_Odds"])
    ci_values = np.array(
        [
            row["Delta_LogLoss_CI95_L"],
            row["Delta_LogLoss_CI95_U"],
            row["Delta_Brier_CI95_L"],
            row["Delta_Brier_CI95_U"],
        ],
        dtype=float,
    )
    ci_computed = bool(np.isfinite(ci_values).all())

    if not ci_computed:
        if d_ll < 0 and d_br < 0:
            return "better on both point estimates; CI not computed"
        if d_ll > 0 and d_br > 0:
            return "worse on both point estimates; CI not computed"
        return "mixed probability-metric result; CI not computed"

    ll_lo = float(row["Delta_LogLoss_CI95_L"])
    ll_hi = float(row["Delta_LogLoss_CI95_U"])
    br_lo = float(row["Delta_Brier_CI95_L"])
    br_hi = float(row["Delta_Brier_CI95_U"])

    if d_ll < 0 and d_br < 0:
        if ll_hi < 0 and br_hi < 0:
            return "better on both; paired 95% CIs below 0"
        return "better on both point estimates; uncertainty overlaps 0"
    if d_ll > 0 and d_br > 0:
        if ll_lo > 0 and br_lo > 0:
            return "worse on both; paired 95% CIs above 0"
        return "worse on both point estimates; uncertainty overlaps 0"
    return "mixed probability-metric result"


def render_markdown_report(
    predictions: pd.DataFrame,
    fold_table: pd.DataFrame,
    league_table: pd.DataFrame,
    season_table: pd.DataFrame,
    *,
    n_segments: int,
    bootstrap: int,
    base_sha: str,
) -> str:
    date_min = pd.to_datetime(predictions["Date"]).min().date()
    date_max = pd.to_datetime(predictions["Date"]).max().date()
    folds = sorted(predictions["fold"].unique().tolist())

    fold_cols = [
        "Fold",
        "Train_N",
        "Test_N",
        "Paired_N",
        "Variant",
        "N",
        "Accuracy",
        "LogLoss",
        "Brier",
        "X_precision",
        "X_recall",
        "Delta_LogLoss_vs_Odds",
        "Delta_Brier_vs_Odds",
    ]
    display_cols = [
        "League",
        "Variant",
        "N",
        "Accuracy",
        "LogLoss",
        "Brier",
        "X_precision",
        "X_recall",
        "Delta_LogLoss_vs_Odds",
        "Delta_Brier_vs_Odds",
        "Delta_LogLoss_CI95_L",
        "Delta_LogLoss_CI95_U",
        "Delta_Brier_CI95_L",
        "Delta_Brier_CI95_U",
    ]
    season_cols = ["Season"] + display_cols

    lines = [
        "# Diagnostic: Model vs Bookmaker Odds",
        "",
        "## Scope",
        "",
        "- Diagnostic only. No production defaults, weights, UI, parser, Elo, Poisson, or streck logic changed.",
        f"- Requested base SHA: `{base_sha}`.",
        f"- Walk-forward date segments: {n_segments}; evaluated folds: {folds}.",
        f"- Paired out-of-fold evaluation rows: {len(predictions)}.",
        f"- Evaluation date range: {date_min} to {date_max}.",
        f"- Paired bootstrap resamples per delta: {bootstrap}.",
        "- All three variants are evaluated on exactly the same rows with valid historical odds.",
        "- `Odds` uses the fair implied probabilities already produced by FeatureBuilder "
        "(Bet365 when complete, otherwise Pinnacle).",
        "- `Model` uses current `FEATURE_COLUMNS` without odds.",
        "- `Model+Odds` is separately trained inside every fold with `ALL_FEATURE_COLUMNS`.",
        "- Negative Delta_LogLoss/Brier_vs_Odds means the candidate is better than odds.",
        "",
        "## Per-fold metrics and warm-up diagnostic",
        "",
        _markdown_table(fold_table[fold_cols], fold_cols),
        "",
        "Early folds have materially less historical training data because this is an expanding-window test. "
        "If a model deficit versus odds shrinks as fold index and Train_N increase, treat data starvation/warm-up "
        "as a plausible explanation rather than concluding from the aggregate alone that the model has no signal.",
        "",
        "## Per-league metrics",
        "",
        _markdown_table(league_table[display_cols], display_cols),
        "",
        "## Per-league evidence summary",
        "",
    ]

    for _, row in league_table[league_table["Variant"] != "Odds"].iterrows():
        lines.append(
            f"- **{row['League']} / {row['Variant']}**: {_evidence_label(row)} "
            f"(N={int(row['N'])}, "
            f"ΔLogLoss={_fmt(row['Delta_LogLoss_vs_Odds'])}, "
            f"ΔBrier={_fmt(row['Delta_Brier_vs_Odds'])})."
        )

    lines.extend(
        [
            "",
            "## Per-season metrics",
            "",
            _markdown_table(season_table[season_cols], season_cols),
            "",
            "## Interpretation guardrails",
            "",
            "- Accuracy is secondary; LogLoss and Brier are the primary probability-quality metrics.",
            "- X precision/recall are reported separately because draw performance can be hidden by overall accuracy.",
            "- A point estimate alone is not treated as proof. When bootstrap is enabled, paired intervals show uncertainty in the delta vs odds.",
            "- This report does not activate `USE_ODDS_FEATURES`, change `combined_probability.py`, or recommend production weights.",
            "",
        ]
    )
    return "\n".join(lines)


def save_outputs(
    report_text: str,
    league_table: pd.DataFrame,
    season_table: pd.DataFrame,
    *,
    report_path: Path,
) -> tuple[Path, Path, Path]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    league_csv = report_path.with_name(report_path.stem + "_LEAGUE.csv")
    season_csv = report_path.with_name(report_path.stem + "_SEASON.csv")
    league_table.to_csv(league_csv, index=False)
    season_table.to_csv(season_csv, index=False)
    return report_path, league_csv, season_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic walk-forward benchmark: model vs odds vs model+odds"
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh cached football-data.co.uk history before running",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=4,
        help="Number of date-safe walk-forward segments (default: 4)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Paired bootstrap resamples for delta CIs; 0 disables (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RESULTS_MODEL_VS_ODDS_DIAGNOSTIC.md"),
        help="Markdown report path",
    )
    parser.add_argument(
        "--base-sha",
        default="10b2d438090ae513de2dfb23c12c96bafb77ff1b",
        help="Expected base SHA recorded in the report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    refresh = args.refresh_data or os.environ.get(
        "BACKTEST_REFRESH_DATA", ""
    ).lower() in ("1", "true", "yes")

    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("No data loaded")
        return 1

    predictions = run_diagnostic(df, n_segments=args.segments)
    fold_table = build_fold_table(
        predictions,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    league_table, season_table = build_summary_tables(
        predictions,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report = render_markdown_report(
        predictions,
        fold_table,
        league_table,
        season_table,
        n_segments=args.segments,
        bootstrap=args.bootstrap,
        base_sha=args.base_sha,
    )
    paths = save_outputs(
        report,
        league_table,
        season_table,
        report_path=args.output,
    )

    logger.info("Diagnostic complete")
    for path in paths:
        logger.info("Wrote %s", path)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
