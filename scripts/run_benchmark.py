#!/usr/bin/env python3
"""
Benchmark: Main model vs simple baselines on historical football matches.

Compares the project's calibrated XGBoost model against:
  1. Home-favoring baseline (fixed home-advantage probabilities)
  2. Elo-only baseline (probabilities from Elo difference only)
  3. Most-frequent-class baseline (always predict training-set class distribution)
  4. Bookmaker implied-probability baseline (if odds data available)

Usage:
    python scripts/run_benchmark.py                 # use cached data
    python scripts/run_benchmark.py --refresh-data  # download fresh data first

All baselines and the main model are evaluated on the same held-out test set
using a temporal split (newest tertile of data).
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so we can import project modules
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from backtest_report import load_data, train_model
from schema import FEATURE_COLUMNS, ALL_FEATURE_COLUMNS, ODDS_FEATURE_COLUMNS, CLASS_MAP, encode_league

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray,
                           n_classes: int = 3) -> float:
    """Multiclass Brier score (lower is better)."""
    y_oh = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        if 0 <= label < n_classes:
            y_oh[i, label] = 1
    return float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))


def top_class_accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Accuracy of the argmax prediction (same as plain accuracy on argmax)."""
    preds = np.argmax(y_proba, axis=1)
    return float(accuracy_score(y_true, preds))


def calibration_summary(y_true: np.ndarray, y_proba: np.ndarray,
                        n_bins: int = 5) -> list:
    """Simple calibration table: for each confidence bin, mean predicted
    probability vs actual hit rate of the chosen class."""
    max_proba = y_proba.max(axis=1)
    pred_class = y_proba.argmax(axis=1)
    correct = (pred_class == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_proba >= lo) & (max_proba < hi)
        if hi == 1.0:
            mask = mask | (max_proba == 1.0)
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": f"{lo:.2f}-{hi:.2f}", "count": 0,
                "avg_confidence": 0.0, "avg_accuracy": 0.0,
            })
        else:
            rows.append({
                "bin": f"{lo:.2f}-{hi:.2f}",
                "count": count,
                "avg_confidence": round(float(max_proba[mask].mean()), 4),
                "avg_accuracy": round(float(correct[mask].mean()), 4),
            })
    return rows


def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute the full metric set for one model/baseline."""
    try:
        ll = float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
    except Exception:
        ll = float("nan")
    return {
        "log_loss": round(ll, 4),
        "brier": round(brier_score_multiclass(y_true, y_proba), 4),
        "accuracy": round(top_class_accuracy(y_true, y_proba), 4),
        "top_class_accuracy": round(top_class_accuracy(y_true, y_proba), 4),
        "n_matches": len(y_true),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def home_favoring_baseline(n: int) -> np.ndarray:
    """Fixed probability baseline with home advantage.

    Uses rough historical English football averages:
      Home win ~45%, Draw ~27%, Away win ~28%.
    Every match gets the same probabilities.
    """
    proba = np.tile([0.45, 0.27, 0.28], (n, 1))
    return proba


def most_frequent_class_baseline(y_train: np.ndarray, n: int,
                                 n_classes: int = 3) -> np.ndarray:
    """Predict the training-set class distribution for every match."""
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    freq = counts / counts.sum()
    return np.tile(freq, (n, 1))


def elo_only_baseline(df_test: pd.DataFrame) -> np.ndarray:
    """Probabilities derived solely from Elo ratings.

    Uses a logistic model:
        P(home win) = 1 / (1 + 10^(-(elo_diff + home_bonus) / 400))

    Draw probability is estimated by shrinking win/loss probs toward 0.5
    using a simple heuristic: draws are more likely when win prob is near 0.5.

    This is intentionally simple -- it's a *baseline*, not a production model.
    """
    elo_home = df_test["HomeElo"].values.astype(float)
    elo_away = df_test["AwayElo"].values.astype(float)
    diff = elo_home - elo_away

    # Home-field bonus of ~65 Elo points (≈ standard in chess/football Elo)
    HOME_BONUS = 65.0
    adj_diff = diff + HOME_BONUS

    # Logistic win probability (two-outcome model)
    p_home_raw = 1.0 / (1.0 + np.power(10.0, -adj_diff / 400.0))
    p_away_raw = 1.0 - p_home_raw

    # Draw probability heuristic: higher when match is close
    # Base draw rate ~26%, scaled by how close the match is
    DRAW_BASE = 0.26
    closeness = 1.0 - np.abs(p_home_raw - 0.5) * 2.0  # 1 when even, 0 when lopsided
    p_draw = DRAW_BASE * (0.5 + 0.5 * closeness)

    # Redistribute remaining probability
    remaining = 1.0 - p_draw
    total_raw = p_home_raw + p_away_raw
    p_home = remaining * (p_home_raw / total_raw)
    p_away = remaining * (p_away_raw / total_raw)

    proba = np.column_stack([p_home, p_draw, p_away])
    return proba


def bookmaker_baseline(df_test: pd.DataFrame) -> tuple:
    """Bookmaker implied probabilities baseline.

    Returns (proba_array, mask_array) where mask indicates rows with valid odds.
    Only includes matches that have has_odds == 1 and valid implied probs.
    """
    if "has_odds" not in df_test.columns:
        return None, None

    has_odds = df_test["has_odds"].values.astype(float)
    mask = has_odds > 0.5

    if mask.sum() == 0:
        return None, None

    implied_h = df_test["ImpliedHome"].values.astype(float)
    implied_d = df_test["ImpliedDraw"].values.astype(float)
    implied_a = df_test["ImpliedAway"].values.astype(float)

    proba = np.column_stack([implied_h, implied_d, implied_a])

    # Re-normalize to ensure they sum to 1
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    proba = proba / row_sums

    return proba, mask


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_temporal_split(df: pd.DataFrame, test_fraction: float = 0.333):
    """Split data by time: the newest `test_fraction` is the eval set.

    Returns (df_train, df_test) both sorted by date.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    cut = int(len(df) * (1.0 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def encode_target(df: pd.DataFrame) -> np.ndarray:
    return df["FTR"].map(CLASS_MAP).values


def prepare_features(df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    """Prepare feature matrix from a df that already has feature columns."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    df_local = df.copy()
    if "League" in df_local.columns:
        df_local["League"] = df_local["League"].apply(encode_league)
    for c in feature_columns:
        if c not in df_local.columns:
            df_local[c] = 0
    return df_local[feature_columns]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(df: pd.DataFrame, with_odds: bool = False) -> dict:
    """Run the full benchmark and return results dict.

    Args:
        df: Full dataset with features already computed.
        with_odds: If True, also train/eval a model variant that includes
                   odds-derived features (ImpliedHome/Draw/Away, has_odds).
    """
    logger.info("Total matches loaded: %d", len(df))

    df_train, df_test = prepare_temporal_split(df, test_fraction=0.333)
    logger.info(
        "Temporal split: train=%d matches, test=%d matches", len(df_train), len(df_test)
    )

    train_date_min = df_train["Date"].min()
    train_date_max = df_train["Date"].max()
    test_date_min = df_test["Date"].min()
    test_date_max = df_test["Date"].max()

    logger.info(
        "Train period: %s to %s", train_date_min.date(), train_date_max.date()
    )
    logger.info(
        "Test period:  %s to %s", test_date_min.date(), test_date_max.date()
    )

    y_train = encode_target(df_train)
    y_test = encode_target(df_test)
    n_test = len(y_test)

    results = {
        "meta": {
            "train_matches": len(df_train),
            "test_matches": n_test,
            "train_period": f"{train_date_min.date()} to {train_date_max.date()}",
            "test_period": f"{test_date_min.date()} to {test_date_max.date()}",
        },
        "models": {},
    }

    # --- 1. Main model WITHOUT odds (calibrated XGBoost) ---
    logger.info("Training main model WITHOUT odds (calibrated XGBoost)...")
    model = train_model(df_train, feature_columns=list(FEATURE_COLUMNS))
    if model is not None:
        X_test = prepare_features(df_test, feature_columns=list(FEATURE_COLUMNS))
        y_proba_main = model.predict_proba(X_test)
        results["models"]["main_model"] = compute_all_metrics(y_test, y_proba_main)
        logger.info("Main model metrics: %s", results["models"]["main_model"])
    else:
        logger.error("Main model training failed!")
        results["models"]["main_model"] = {"error": "training failed"}

    # --- 1b. Main model WITH odds (if requested) ---
    model_odds = None
    y_proba_odds = None
    if with_odds:
        logger.info("Training main model WITH odds features...")

        # Check how many test matches actually have odds
        if "has_odds" in df_test.columns:
            odds_available = (df_test["has_odds"].fillna(0).astype(float) > 0.5).sum()
        else:
            odds_available = 0
        logger.info(
            "Odds availability in test set: %d / %d matches (%.1f%%)",
            odds_available, n_test, 100.0 * odds_available / n_test if n_test else 0,
        )

        model_odds = train_model(df_train, feature_columns=list(ALL_FEATURE_COLUMNS))
        if model_odds is not None:
            X_test_odds = prepare_features(df_test, feature_columns=list(ALL_FEATURE_COLUMNS))
            y_proba_odds = model_odds.predict_proba(X_test_odds)
            results["models"]["main_model_with_odds"] = compute_all_metrics(
                y_test, y_proba_odds
            )
            results["models"]["main_model_with_odds"]["note"] = (
                f"Trained/evaluated with odds features. "
                f"{odds_available}/{n_test} test matches had real odds data."
            )
            logger.info(
                "Main model WITH odds metrics: %s",
                results["models"]["main_model_with_odds"],
            )
        else:
            logger.error("Main model WITH odds training failed!")
            results["models"]["main_model_with_odds"] = {"error": "training failed"}

    # --- 2. Home-favoring baseline ---
    logger.info("Computing home-favoring baseline...")
    y_proba_home = home_favoring_baseline(n_test)
    results["models"]["home_favoring"] = compute_all_metrics(y_test, y_proba_home)
    logger.info("Home baseline metrics: %s", results["models"]["home_favoring"])

    # --- 3. Most-frequent-class baseline ---
    logger.info("Computing most-frequent-class baseline...")
    y_proba_freq = most_frequent_class_baseline(y_train, n_test)
    results["models"]["most_frequent_class"] = compute_all_metrics(y_test, y_proba_freq)
    logger.info("Freq baseline metrics: %s", results["models"]["most_frequent_class"])

    # --- 4. Elo-only baseline ---
    logger.info("Computing Elo-only baseline...")
    y_proba_elo = elo_only_baseline(df_test)
    results["models"]["elo_only"] = compute_all_metrics(y_test, y_proba_elo)
    logger.info("Elo-only metrics: %s", results["models"]["elo_only"])

    # --- 5. Bookmaker baseline (if odds available) ---
    logger.info("Checking for bookmaker odds data...")
    bm_proba, bm_mask = bookmaker_baseline(df_test)
    if bm_proba is not None and bm_mask is not None and bm_mask.sum() > 0:
        # Evaluate only on matches with odds
        y_test_bm = y_test[bm_mask]
        y_proba_bm = bm_proba[bm_mask]
        results["models"]["bookmaker_implied"] = compute_all_metrics(
            y_test_bm, y_proba_bm
        )
        results["models"]["bookmaker_implied"]["note"] = (
            f"Evaluated on {int(bm_mask.sum())} of {n_test} test matches with odds data"
        )
        logger.info("Bookmaker metrics: %s", results["models"]["bookmaker_implied"])

        # Also re-evaluate main models on the same odds-only subset for fair comparison
        if model is not None:
            results["models"]["main_model_odds_subset"] = compute_all_metrics(
                y_test_bm, y_proba_main[bm_mask]
            )
            results["models"]["main_model_odds_subset"]["note"] = (
                "Main model (no odds) evaluated on same odds-only subset for fair comparison"
            )
        if with_odds and model_odds is not None and y_proba_odds is not None:
            results["models"]["main_model_with_odds_subset"] = compute_all_metrics(
                y_test_bm, y_proba_odds[bm_mask]
            )
            results["models"]["main_model_with_odds_subset"]["note"] = (
                "Main model (with odds) evaluated on same odds-only subset for fair comparison"
            )
    else:
        logger.info("No usable bookmaker odds data found in test set. Skipping.")
        results["models"]["bookmaker_implied"] = {
            "note": "Not included - no odds data available in test set"
        }

    # --- Calibration summaries ---
    if model is not None:
        results["calibration"] = calibration_summary(y_test, y_proba_main)
    if with_odds and model_odds is not None and y_proba_odds is not None:
        results["calibration_with_odds"] = calibration_summary(y_test, y_proba_odds)

    # --- A/B delta summary (with_odds mode) ---
    if with_odds:
        a_metrics = results["models"].get("main_model", {})
        b_metrics = results["models"].get("main_model_with_odds", {})
        if "log_loss" in a_metrics and "log_loss" in b_metrics:
            results["ab_delta"] = {
                "log_loss_delta": round(b_metrics["log_loss"] - a_metrics["log_loss"], 4),
                "brier_delta": round(b_metrics["brier"] - a_metrics["brier"], 4),
                "accuracy_delta": round(b_metrics["accuracy"] - a_metrics["accuracy"], 4),
                "note": "Negative log_loss/brier delta = odds variant is better. Positive accuracy delta = odds variant is better.",
            }

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    """Print a human-readable benchmark report."""
    meta = results["meta"]

    print("\n" + "=" * 72)
    print("  BENCHMARK REPORT: Main Model vs Simple Baselines")
    print("=" * 72)

    print(f"\n  Train set: {meta['train_matches']} matches ({meta['train_period']})")
    print(f"  Test set:  {meta['test_matches']} matches ({meta['test_period']})")

    # Results table
    print("\n" + "-" * 72)
    header = f"{'Model':<28} {'Log Loss':>10} {'Brier':>10} {'Accuracy':>10} {'N':>6}"
    print(header)
    print("-" * 72)

    # Determine ranking by log_loss (lower is better)
    model_items = []
    for name, metrics in results["models"].items():
        if "error" in metrics or "log_loss" not in metrics:
            continue
        if name == "main_model_odds_subset":
            continue  # show separately
        model_items.append((name, metrics))

    model_items.sort(key=lambda x: x[1]["log_loss"])

    for rank, (name, m) in enumerate(model_items, 1):
        label = f"{rank}. {name}"
        note = m.get("note", "")
        print(
            f"  {label:<26} {m['log_loss']:>10.4f} {m['brier']:>10.4f} "
            f"{m['accuracy']:>10.4f} {m['n_matches']:>6}"
        )
        if note:
            print(f"     {note}")

    # Odds subset comparison (if available)
    odds_subset_models = [
        ("main_model (odds subset)", "main_model_odds_subset"),
        ("main+odds (odds subset)", "main_model_with_odds_subset"),
        ("bookmaker_implied", "bookmaker_implied"),
    ]
    has_subset = any(
        k in results["models"] and "log_loss" in results["models"][k]
        for _, k in odds_subset_models
    )
    if has_subset:
        print("\n  --- Fair comparison on odds-only subset ---")
        for label, key in odds_subset_models:
            m = results["models"].get(key, {})
            if "log_loss" in m:
                print(
                    f"  {label:<28} {m['log_loss']:>10.4f} {m['brier']:>10.4f} "
                    f"{m['accuracy']:>10.4f} {m['n_matches']:>6}"
                )

    # Calibration
    for cal_key, cal_label in [
        ("calibration", "Main Model (no odds)"),
        ("calibration_with_odds", "Main Model (with odds)"),
    ]:
        if cal_key in results:
            print("\n" + "-" * 72)
            print(f"  Calibration Summary ({cal_label})")
            print("-" * 72)
            print(f"  {'Bin':<14} {'Count':>8} {'Avg Conf':>12} {'Avg Acc':>12}")
            for row in results[cal_key]:
                print(
                    f"  {row['bin']:<14} {row['count']:>8} "
                    f"{row['avg_confidence']:>12.4f} {row['avg_accuracy']:>12.4f}"
                )

    print("\n" + "=" * 72)
    print("  RANKING (by log loss, lower is better):")
    print("=" * 72)
    for rank, (name, m) in enumerate(model_items, 1):
        marker = " <-- MAIN MODEL" if name == "main_model" else ""
        print(f"  {rank}. {name}: log_loss={m['log_loss']:.4f}{marker}")

    # Verdict
    print("\n" + "=" * 72)
    if model_items and model_items[0][0] == "main_model":
        second = model_items[1] if len(model_items) > 1 else None
        gap = (second[1]["log_loss"] - model_items[0][1]["log_loss"]) if second else 0
        print(f"  VERDICT: Main model WINS (best log loss).")
        if second:
            print(f"  Gap to next best ({second[0]}): {gap:.4f} log loss points.")
    else:
        main_rank = next(
            (i for i, (n, _) in enumerate(model_items, 1) if n == "main_model"),
            None,
        )
        if main_rank:
            winner = model_items[0]
            gap = (
                results["models"].get("main_model", {}).get("log_loss", 0)
                - winner[1]["log_loss"]
            )
            print(
                f"  VERDICT: Main model LOSES. Ranked #{main_rank}."
            )
            print(
                f"  Best model: {winner[0]} "
                f"(log_loss={winner[1]['log_loss']:.4f}, "
                f"gap={gap:.4f})."
            )
        else:
            print("  VERDICT: Could not determine ranking (main model may have failed).")

    # --- A/B Delta Summary ---
    if "ab_delta" in results:
        d = results["ab_delta"]
        print("\n" + "=" * 72)
        print("  A/B COMPARISON: main_model (no odds) vs main_model_with_odds")
        print("=" * 72)
        print(f"  Log loss delta (B - A): {d['log_loss_delta']:+.4f}")
        print(f"  Brier delta   (B - A): {d['brier_delta']:+.4f}")
        print(f"  Accuracy delta(B - A): {d['accuracy_delta']:+.4f}")
        if d["log_loss_delta"] < -0.005:
            print("  => Odds features IMPROVE the model (lower log loss).")
        elif d["log_loss_delta"] > 0.005:
            print("  => Odds features HURT the model (higher log loss).")
        else:
            print("  => Odds features have NEGLIGIBLE effect on log loss.")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark main model vs simple baselines",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh data instead of using cache",
    )
    parser.add_argument(
        "--with-odds",
        action="store_true",
        help="Also benchmark a model variant trained WITH odds features (A/B comparison)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = args.refresh_data or os.environ.get(
        "BACKTEST_REFRESH_DATA", ""
    ).lower() in ("1", "true", "yes")
    with_odds = args.with_odds or os.environ.get(
        "BENCHMARK_WITH_ODDS", ""
    ).lower() in ("1", "true", "yes")

    logger.info("Loading data (refresh=%s, with_odds=%s)...", refresh, with_odds)
    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("No data loaded. Run with --refresh-data to download.")
        return 1

    results = run_benchmark(df, with_odds=with_odds)
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
