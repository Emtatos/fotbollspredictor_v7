#!/usr/bin/env python3
"""
Benchmark: Entropy vs Gain-based half-guard selection.

Runs the same walk-forward backtest twice -- once with the old entropy-based
half-guard selection and once with the new gain/top2-based selection -- and
prints a side-by-side comparison.

Usage:
    python scripts/benchmark_halfguards.py                # use cached data
    python scripts/benchmark_halfguards.py --refresh-data # download fresh data first

The script does NOT touch any production logic; it only calls
backtest_report helpers with different `selection_mode` values.
"""
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import logging
import numpy as np
import pandas as pd

from backtest_report import (
    load_data,
    train_model,
    predict_with_entropy,
    compute_block_metrics,
    select_halfguards_gain,
    select_halfguards_entropy,
    get_top2_predictions,
    N_HALF,
    LEAGUES,
)
from schema import encode_league

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core comparison runner
# ---------------------------------------------------------------------------

def run_comparison(df: pd.DataFrame, n_folds: int = 3):
    """Run both entropy and gain selection on identical folds.

    Returns a dict with per-fold and aggregate results for both modes,
    plus a diff analysis showing where selections diverge.
    """
    if df.empty:
        logger.error("Empty dataframe")
        return None

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["time_fold"] = pd.qcut(df["Date"], q=n_folds, labels=False, duplicates="drop")

    results = {"folds": [], "divergence": []}

    for fold_idx in range(1, n_folds):
        train_mask = df["time_fold"] < fold_idx
        test_mask = df["time_fold"] == fold_idx
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        if len(df_train) < 100 or len(df_test) < 20:
            logger.warning(
                "Fold %d: insufficient data (train=%d, test=%d)",
                fold_idx, len(df_train), len(df_test),
            )
            continue

        logger.info(
            "Fold %d: train=%d, test=%d", fold_idx, len(df_train), len(df_test)
        )

        model = train_model(df_train)
        if model is None:
            logger.warning("Fold %d: model training failed", fold_idx)
            continue

        y_true, y_proba, pred_top1, entropy_values = predict_with_entropy(
            model, df_test
        )

        league_codes = (
            df_test["League"].apply(encode_league).values
            if "League" in df_test.columns
            else None
        )

        n_half = min(N_HALF, len(y_true) // 4)

        # --- Run both modes ---
        metrics_gain = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=n_half, league_codes=league_codes,
            selection_mode="gain",
        )
        metrics_entropy = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=n_half, league_codes=league_codes,
            selection_mode="entropy",
        )

        # --- Divergence analysis ---
        if n_half > 0:
            idx_gain = set(select_halfguards_gain(y_proba, n_half).tolist())
            idx_entropy = set(
                select_halfguards_entropy(y_proba, entropy_values, n_half).tolist()
            )
            n_different = len(idx_gain.symmetric_difference(idx_entropy))

            # Per-selection stats
            sorted_desc = np.sort(y_proba, axis=1)[:, ::-1]
            gains_all = sorted_desc[:, 1]
            top2s_all = sorted_desc[:, 0] + sorted_desc[:, 1]

            def _stats(indices):
                idx_list = list(indices)
                return {
                    "avg_gain": float(np.mean(gains_all[idx_list])),
                    "avg_top2": float(np.mean(top2s_all[idx_list])),
                    "avg_entropy": float(np.mean(entropy_values[idx_list])),
                }

            stats_gain = _stats(idx_gain)
            stats_entropy = _stats(idx_entropy)
        else:
            n_different = 0
            stats_gain = {"avg_gain": 0, "avg_top2": 0, "avg_entropy": 0}
            stats_entropy = {"avg_gain": 0, "avg_top2": 0, "avg_entropy": 0}

        fold_result = {
            "fold": fold_idx,
            "n_matches": len(y_true),
            "n_half": n_half,
            "gain": metrics_gain,
            "entropy": metrics_entropy,
            "n_different_selections": n_different,
            "stats_gain": stats_gain,
            "stats_entropy": stats_entropy,
        }
        results["folds"].append(fold_result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(results: dict) -> str:
    """Print and return a markdown comparison report."""
    if not results or not results["folds"]:
        msg = "No results to report."
        print(msg)
        return msg

    lines = []

    lines.append("# Half-Guard Selection Benchmark: Entropy vs Gain")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Walk-forward folds: {len(results['folds'])}")
    total = sum(f["n_matches"] for f in results["folds"])
    lines.append(f"- Total test matches: {total}")
    lines.append(f"- Half-guards per fold (N_HALF): {N_HALF}")
    lines.append(f"- Leagues: {', '.join(LEAGUES)}")
    lines.append("")

    # --- Per-fold table ---
    lines.append("## Per-Fold Results")
    lines.append("")
    lines.append(
        "| Fold | N | Mode | Acc_Top1 | Acc_Top2_HG | Combined | LogLoss | Brier |"
    )
    lines.append(
        "|------|---|------|----------|-------------|----------|---------|-------|"
    )

    for f in results["folds"]:
        for mode_key, label in [("gain", "gain"), ("entropy", "entropy")]:
            m = f[mode_key]
            lines.append(
                f"| {f['fold']} | {m['n_matches']} | {label} "
                f"| {m['accuracy_top1']:.4f} "
                f"| {m['accuracy_top2_on_halfguards']:.4f} "
                f"| {m['combined_ticket_hit_rate']:.4f} "
                f"| {m['logloss']:.4f} "
                f"| {m['brier']:.4f} |"
            )

    # --- Aggregate ---
    lines.append("")
    lines.append("## Aggregate (Mean across folds)")
    lines.append("")
    lines.append(
        "| Metric | Entropy | Gain | Delta (gain - entropy) | Better |"
    )
    lines.append(
        "|--------|---------|------|------------------------|--------|"
    )

    metric_keys = [
        ("accuracy_top1", "Acc_Top1", True),
        ("accuracy_top2_on_halfguards", "Acc_Top2_HG", True),
        ("combined_ticket_hit_rate", "Combined", True),
        ("logloss", "LogLoss", False),
        ("brier", "Brier", False),
    ]

    for key, display, higher_is_better in metric_keys:
        vals_e = [f["entropy"][key] for f in results["folds"]]
        vals_g = [f["gain"][key] for f in results["folds"]]
        mean_e = np.mean(vals_e)
        mean_g = np.mean(vals_g)
        delta = mean_g - mean_e

        if higher_is_better:
            better = "gain" if delta > 0.0001 else ("entropy" if delta < -0.0001 else "tie")
        else:
            better = "gain" if delta < -0.0001 else ("entropy" if delta > 0.0001 else "tie")

        lines.append(
            f"| {display} | {mean_e:.4f} | {mean_g:.4f} | {delta:+.4f} | {better} |"
        )

    # --- Divergence ---
    lines.append("")
    lines.append("## Selection Divergence")
    lines.append("")
    lines.append(
        "| Fold | N_half | Selections that differ |"
    )
    lines.append(
        "|------|--------|-----------------------|"
    )
    total_diff = 0
    total_hg = 0
    for f in results["folds"]:
        lines.append(
            f"| {f['fold']} | {f['n_half']} | {f['n_different_selections']} |"
        )
        total_diff += f["n_different_selections"]
        total_hg += f["n_half"]

    lines.append(
        f"| **Total** | {total_hg} | {total_diff} |"
    )

    # --- Per-selection stats ---
    lines.append("")
    lines.append("## Average Stats for Selected Half-Guards")
    lines.append("")
    lines.append(
        "| Fold | Mode | Avg Gain | Avg Top2 | Avg Entropy |"
    )
    lines.append(
        "|------|------|----------|----------|-------------|"
    )
    for f in results["folds"]:
        sg = f["stats_gain"]
        se = f["stats_entropy"]
        lines.append(
            f"| {f['fold']} | gain | {sg['avg_gain']:.4f} | {sg['avg_top2']:.4f} | {sg['avg_entropy']:.4f} |"
        )
        lines.append(
            f"| {f['fold']} | entropy | {se['avg_gain']:.4f} | {se['avg_top2']:.4f} | {se['avg_entropy']:.4f} |"
        )

    # --- Conclusion ---
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")

    # Derive overall verdict
    acc_hg_e = np.mean([f["entropy"]["accuracy_top2_on_halfguards"] for f in results["folds"]])
    acc_hg_g = np.mean([f["gain"]["accuracy_top2_on_halfguards"] for f in results["folds"]])
    comb_e = np.mean([f["entropy"]["combined_ticket_hit_rate"] for f in results["folds"]])
    comb_g = np.mean([f["gain"]["combined_ticket_hit_rate"] for f in results["folds"]])

    hg_delta = acc_hg_g - acc_hg_e
    comb_delta = comb_g - comb_e

    if hg_delta > 0.01 and comb_delta > 0.005:
        verdict = "**Gain-based selection is better** than entropy-based on this dataset."
    elif hg_delta < -0.01 and comb_delta < -0.005:
        verdict = "**Entropy-based selection is better** than gain-based on this dataset."
    else:
        verdict = "**Mixed / no clear winner** -- the two methods perform similarly on this dataset."

    lines.append(verdict)
    lines.append("")
    lines.append(f"- Acc_Top2_HG delta (gain - entropy): {hg_delta:+.4f}")
    lines.append(f"- Combined hit rate delta (gain - entropy): {comb_delta:+.4f}")
    lines.append(f"- Selections that differed: {total_diff} / {total_hg}")

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark entropy vs gain half-guard selection",
    )
    parser.add_argument(
        "--refresh-data", action="store_true",
        help="Download fresh data instead of using cache",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write markdown report to this file (default: RESULTS_HALF_GUARD_COMPARISON.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    refresh = args.refresh_data or os.environ.get(
        "BACKTEST_REFRESH_DATA", ""
    ).lower() in ("1", "true", "yes")

    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("Failed to load data")
        return 1

    logger.info("Loaded %d matches", len(df))

    results = run_comparison(df, n_folds=3)
    if results is None:
        return 1

    report = print_comparison(results)

    out_path = args.output or str(_REPO_ROOT / "RESULTS_HALF_GUARD_COMPARISON.md")
    Path(out_path).write_text(report + "\n", encoding="utf-8")
    logger.info("Report written to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
