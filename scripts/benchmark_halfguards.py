#!/usr/bin/env python3
"""
Benchmark: Entropy vs Gain-based half-guard selection.

Runs the same walk-forward backtest twice -- once with the old entropy-based
half-guard selection and once with the new gain/top2-based selection -- and
prints a side-by-side comparison.

Expanded version: sweeps over multiple folds, leagues, and N_HALF values
to produce a robust comparison matrix.

Usage:
    python scripts/benchmark_halfguards.py                # use cached data
    python scripts/benchmark_halfguards.py --refresh-data # download fresh data first
    python scripts/benchmark_halfguards.py --expanded     # run expanded matrix benchmark

The script does NOT touch any production logic; it only calls
backtest_report helpers with different `selection_mode` values.
"""
import argparse
import csv
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

# Expanded benchmark defaults
N_HALF_VALUES = [2, 4, 6]
EXPANDED_N_FOLDS = 5
LEAGUE_SUBSETS = {
    "all": None,
    "E0": [0],
    "E1": [1],
    "E2": [2],
    "E3": [3],
}


# ---------------------------------------------------------------------------
# Core comparison runner (original, single N_HALF)
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
# Expanded comparison runner (matrix: folds x leagues x N_HALF)
# ---------------------------------------------------------------------------

def _run_single_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_half: int,
    fold_idx: int,
    league_label: str,
):
    """Run entropy vs gain for a single fold / N_HALF combination.

    Returns a result dict or None if data is insufficient.
    """
    if len(df_train) < 100 or len(df_test) < 20:
        logger.warning(
            "Fold %d, league=%s: insufficient data (train=%d, test=%d)",
            fold_idx, league_label, len(df_train), len(df_test),
        )
        return None

    model = train_model(df_train)
    if model is None:
        logger.warning(
            "Fold %d, league=%s: model training failed",
            fold_idx, league_label,
        )
        return None

    y_true, y_proba, pred_top1, entropy_values = predict_with_entropy(
        model, df_test
    )

    league_codes = (
        df_test["League"].apply(encode_league).values
        if "League" in df_test.columns
        else None
    )

    effective_n_half = min(n_half, len(y_true) // 4)
    if effective_n_half <= 0:
        logger.warning(
            "Fold %d, league=%s, n_half=%d: not enough matches",
            fold_idx, league_label, n_half,
        )
        return None

    metrics_gain = compute_block_metrics(
        y_true, y_proba, pred_top1, entropy_values,
        n_half=effective_n_half, league_codes=league_codes,
        selection_mode="gain",
    )
    metrics_entropy = compute_block_metrics(
        y_true, y_proba, pred_top1, entropy_values,
        n_half=effective_n_half, league_codes=league_codes,
        selection_mode="entropy",
    )

    idx_gain = set(
        select_halfguards_gain(y_proba, effective_n_half).tolist()
    )
    idx_entropy = set(
        select_halfguards_entropy(
            y_proba, entropy_values, effective_n_half
        ).tolist()
    )
    n_different = len(idx_gain.symmetric_difference(idx_entropy))

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

    return {
        "fold": fold_idx,
        "league": league_label,
        "n_half_requested": n_half,
        "n_half_effective": effective_n_half,
        "n_matches": len(y_true),
        "gain": metrics_gain,
        "entropy": metrics_entropy,
        "n_different_selections": n_different,
        "stats_gain": _stats(idx_gain),
        "stats_entropy": _stats(idx_entropy),
    }


def run_expanded_comparison(
    df: pd.DataFrame,
    n_folds: int = EXPANDED_N_FOLDS,
    n_half_values: list = None,
    league_subsets: dict = None,
):
    """Run the full expanded benchmark matrix.

    Iterates over folds x league subsets x N_HALF values.
    Returns a list of per-cell result dicts.
    """
    if n_half_values is None:
        n_half_values = N_HALF_VALUES
    if league_subsets is None:
        league_subsets = LEAGUE_SUBSETS

    if df.empty:
        logger.error("Empty dataframe")
        return []

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(
        drop=True
    )
    df["time_fold"] = pd.qcut(
        df["Date"], q=n_folds, labels=False, duplicates="drop"
    )

    results = []

    for fold_idx in range(1, n_folds):
        train_mask = df["time_fold"] < fold_idx
        test_mask = df["time_fold"] == fold_idx
        df_train_all = df[train_mask].copy()
        df_test_all = df[test_mask].copy()

        if len(df_train_all) < 100 or len(df_test_all) < 20:
            logger.warning(
                "Fold %d: insufficient data (train=%d, test=%d), skipping",
                fold_idx, len(df_train_all), len(df_test_all),
            )
            continue

        for league_label, league_filter in league_subsets.items():
            if league_filter is not None and "League" in df.columns:
                df_train = df_train_all[
                    df_train_all["League"].isin(league_filter)
                ].copy()
                df_test = df_test_all[
                    df_test_all["League"].isin(league_filter)
                ].copy()
            else:
                df_train = df_train_all.copy()
                df_test = df_test_all.copy()

            if len(df_train) < 50 or len(df_test) < 10:
                logger.info(
                    "Fold %d, league=%s: skipping (train=%d, test=%d)",
                    fold_idx, league_label,
                    len(df_train), len(df_test),
                )
                continue

            logger.info(
                "Fold %d, league=%s: train=%d, test=%d",
                fold_idx, league_label,
                len(df_train), len(df_test),
            )

            for n_half in n_half_values:
                result = _run_single_fold(
                    df_train, df_test, n_half,
                    fold_idx, league_label,
                )
                if result is not None:
                    results.append(result)

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_results(results: list) -> dict:
    """Aggregate expanded results into summary tables.

    Returns a dict with keys: total, by_league, by_fold,
    by_n_half, by_league_n_half.
    """
    if not results:
        return {}

    def _agg(subset):
        if not subset:
            return None

        n_total_hg = sum(r["n_half_effective"] for r in subset)
        n_matches = sum(r["n_matches"] for r in subset)
        n_divergent = sum(
            r["n_different_selections"] for r in subset
        )

        w_matches = [r["n_matches"] for r in subset]
        w_hg = [r["n_half_effective"] for r in subset]

        acc_top1_g = np.average(
            [r["gain"]["accuracy_top1"] for r in subset],
            weights=w_matches,
        )
        acc_top1_e = np.average(
            [r["entropy"]["accuracy_top1"] for r in subset],
            weights=w_matches,
        )
        acc_hg_g = np.average(
            [r["gain"]["accuracy_top2_on_halfguards"] for r in subset],
            weights=w_hg,
        )
        acc_hg_e = np.average(
            [r["entropy"]["accuracy_top2_on_halfguards"] for r in subset],
            weights=w_hg,
        )
        comb_g = np.average(
            [r["gain"]["combined_ticket_hit_rate"] for r in subset],
            weights=w_matches,
        )
        comb_e = np.average(
            [r["entropy"]["combined_ticket_hit_rate"] for r in subset],
            weights=w_matches,
        )

        avg_gain_g = np.mean(
            [r["stats_gain"]["avg_gain"] for r in subset]
        )
        avg_gain_e = np.mean(
            [r["stats_entropy"]["avg_gain"] for r in subset]
        )
        avg_top2_g = np.mean(
            [r["stats_gain"]["avg_top2"] for r in subset]
        )
        avg_top2_e = np.mean(
            [r["stats_entropy"]["avg_top2"] for r in subset]
        )

        return {
            "n_cells": len(subset),
            "n_matches": n_matches,
            "n_total_hg_decisions": n_total_hg,
            "n_divergent_selections": n_divergent,
            "gain_acc_top1": float(acc_top1_g),
            "entropy_acc_top1": float(acc_top1_e),
            "gain_acc_hg": float(acc_hg_g),
            "entropy_acc_hg": float(acc_hg_e),
            "gain_combined": float(comb_g),
            "entropy_combined": float(comb_e),
            "delta_acc_hg": float(acc_hg_g - acc_hg_e),
            "delta_combined": float(comb_g - comb_e),
            "gain_avg_gain": float(avg_gain_g),
            "entropy_avg_gain": float(avg_gain_e),
            "gain_avg_top2": float(avg_top2_g),
            "entropy_avg_top2": float(avg_top2_e),
        }

    agg = {}

    all_league = [r for r in results if r["league"] == "all"]
    agg["total"] = _agg(all_league)

    leagues_seen = sorted(
        set(r["league"] for r in results if r["league"] != "all")
    )
    agg["by_league"] = {}
    for league in leagues_seen:
        subset = [r for r in results if r["league"] == league]
        agg["by_league"][league] = _agg(subset)

    folds_seen = sorted(set(r["fold"] for r in all_league))
    agg["by_fold"] = {}
    for fold in folds_seen:
        subset = [r for r in all_league if r["fold"] == fold]
        agg["by_fold"][fold] = _agg(subset)

    nh_seen = sorted(
        set(r["n_half_requested"] for r in all_league)
    )
    agg["by_n_half"] = {}
    for nh in nh_seen:
        subset = [
            r for r in all_league if r["n_half_requested"] == nh
        ]
        agg["by_n_half"][nh] = _agg(subset)

    agg["by_league_n_half"] = {}
    for league in leagues_seen:
        for nh in nh_seen:
            subset = [
                r for r in results
                if r["league"] == league
                and r["n_half_requested"] == nh
            ]
            if subset:
                agg["by_league_n_half"][(league, nh)] = _agg(subset)

    return agg


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results: list, path: Path):
    """Export raw per-cell results to CSV."""
    if not results:
        return

    fieldnames = [
        "fold", "league", "n_half_requested", "n_half_effective",
        "n_matches",
        "gain_acc_top1", "gain_acc_hg", "gain_combined",
        "gain_logloss", "gain_brier",
        "entropy_acc_top1", "entropy_acc_hg", "entropy_combined",
        "entropy_logloss", "entropy_brier",
        "n_different_selections",
        "gain_avg_gain", "gain_avg_top2", "gain_avg_entropy",
        "entropy_avg_gain", "entropy_avg_top2", "entropy_avg_entropy",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "fold": r["fold"],
                "league": r["league"],
                "n_half_requested": r["n_half_requested"],
                "n_half_effective": r["n_half_effective"],
                "n_matches": r["n_matches"],
                "gain_acc_top1": round(
                    r["gain"]["accuracy_top1"], 4
                ),
                "gain_acc_hg": round(
                    r["gain"]["accuracy_top2_on_halfguards"], 4
                ),
                "gain_combined": round(
                    r["gain"]["combined_ticket_hit_rate"], 4
                ),
                "gain_logloss": round(r["gain"]["logloss"], 4),
                "gain_brier": round(r["gain"]["brier"], 4),
                "entropy_acc_top1": round(
                    r["entropy"]["accuracy_top1"], 4
                ),
                "entropy_acc_hg": round(
                    r["entropy"]["accuracy_top2_on_halfguards"], 4
                ),
                "entropy_combined": round(
                    r["entropy"]["combined_ticket_hit_rate"], 4
                ),
                "entropy_logloss": round(
                    r["entropy"]["logloss"], 4
                ),
                "entropy_brier": round(r["entropy"]["brier"], 4),
                "n_different_selections": (
                    r["n_different_selections"]
                ),
                "gain_avg_gain": round(
                    r["stats_gain"]["avg_gain"], 4
                ),
                "gain_avg_top2": round(
                    r["stats_gain"]["avg_top2"], 4
                ),
                "gain_avg_entropy": round(
                    r["stats_gain"]["avg_entropy"], 4
                ),
                "entropy_avg_gain": round(
                    r["stats_entropy"]["avg_gain"], 4
                ),
                "entropy_avg_top2": round(
                    r["stats_entropy"]["avg_top2"], 4
                ),
                "entropy_avg_entropy": round(
                    r["stats_entropy"]["avg_entropy"], 4
                ),
            })

    logger.info("CSV written to %s (%d rows)", path, len(results))


# ---------------------------------------------------------------------------
# Reporting (original simple report)
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

    if hg_delta > 0.01:
        verdict = (
            "**Gain-based selection is better** for half-guard accuracy "
            f"(Acc_Top2_HG: {acc_hg_g:.4f} vs {acc_hg_e:.4f}, delta {hg_delta:+.4f})."
        )
        if abs(comb_delta) < 0.005:
            verdict += (
                f" The combined ticket hit rate difference is small ({comb_delta:+.4f}) "
                f"because N_HALF={N_HALF} is a small fraction of total matches, "
                "but the gain-based method picks half-guards that are much more likely to hit."
            )
    elif hg_delta < -0.01:
        verdict = (
            "**Entropy-based selection is better** for half-guard accuracy "
            f"(Acc_Top2_HG: {acc_hg_e:.4f} vs {acc_hg_g:.4f}, delta {hg_delta:+.4f})."
        )
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
# Expanded report
# ---------------------------------------------------------------------------

def _verdict(delta_hg: float) -> str:
    if delta_hg > 0.01:
        return "gain"
    elif delta_hg < -0.01:
        return "entropy"
    return "tie"


def print_expanded_report(
    results: list,
    agg: dict,
    n_folds: int,
    n_half_values: list,
) -> str:
    """Print and return an expanded markdown comparison report."""
    if not results or not agg:
        msg = "No results to report."
        print(msg)
        return msg

    lines = []
    lines.append(
        "# Half-Guard Benchmark: Expanded Entropy vs Gain Comparison"
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- Walk-forward folds: {n_folds} "
        f"(using {n_folds - 1} test folds)"
    )
    lines.append(f"- Leagues: {', '.join(LEAGUES)}")
    lines.append(
        "- N_HALF values tested: "
        f"{', '.join(str(x) for x in n_half_values)}"
    )
    lines.append(
        "- League subsets: all (combined), E0, E1, E2, E3 (individual)"
    )
    total_cells = len(results)
    lines.append(f"- Total benchmark cells: {total_cells}")

    tot = agg.get("total")
    if tot:
        lines.append(
            "- Total test matches (all-league cells): "
            f"{tot['n_matches']}"
        )
        lines.append(
            "- Total half-guard decisions (all-league cells): "
            f"{tot['n_total_hg_decisions']}"
        )
    lines.append("")

    # --- Overall summary ---
    lines.append("## Overall Summary (all leagues combined)")
    lines.append("")
    if tot:
        lines.append(
            "| Metric | Entropy | Gain | Delta | Better |"
        )
        lines.append(
            "|--------|---------|------|-------|--------|"
        )
        lines.append(
            f"| Acc_Top2_HG | {tot['entropy_acc_hg']:.4f} "
            f"| {tot['gain_acc_hg']:.4f} "
            f"| {tot['delta_acc_hg']:+.4f} "
            f"| {_verdict(tot['delta_acc_hg'])} |"
        )
        lines.append(
            f"| Combined | {tot['entropy_combined']:.4f} "
            f"| {tot['gain_combined']:.4f} "
            f"| {tot['delta_combined']:+.4f} "
            f"| {_verdict(tot['delta_combined'])} |"
        )
        d1 = tot["gain_acc_top1"] - tot["entropy_acc_top1"]
        lines.append(
            f"| Acc_Top1 | {tot['entropy_acc_top1']:.4f} "
            f"| {tot['gain_acc_top1']:.4f} "
            f"| {d1:+.4f} | tie |"
        )
        lines.append("")
        lines.append(
            f"- Divergent selections: "
            f"{tot['n_divergent_selections']} / "
            f"{tot['n_total_hg_decisions']}"
        )
        lines.append(
            f"- Mean gain (gain-selected): "
            f"{tot['gain_avg_gain']:.4f}"
        )
        lines.append(
            f"- Mean gain (entropy-selected): "
            f"{tot['entropy_avg_gain']:.4f}"
        )
        lines.append(
            f"- Mean top2 (gain-selected): "
            f"{tot['gain_avg_top2']:.4f}"
        )
        lines.append(
            f"- Mean top2 (entropy-selected): "
            f"{tot['entropy_avg_top2']:.4f}"
        )
    lines.append("")

    # --- By N_HALF ---
    lines.append("## Results by N_HALF")
    lines.append("")
    lines.append(
        "| N_HALF | HG Decisions | Entropy Acc_HG "
        "| Gain Acc_HG | Delta | Better |"
    )
    lines.append(
        "|--------|-------------|----------------"
        "|-------------|-------|--------|"
    )
    for nh in sorted(agg.get("by_n_half", {}).keys()):
        a = agg["by_n_half"][nh]
        lines.append(
            f"| {nh} | {a['n_total_hg_decisions']} "
            f"| {a['entropy_acc_hg']:.4f} "
            f"| {a['gain_acc_hg']:.4f} "
            f"| {a['delta_acc_hg']:+.4f} "
            f"| {_verdict(a['delta_acc_hg'])} |"
        )
    lines.append("")

    # --- By League ---
    lines.append("## Results by League")
    lines.append("")
    lines.append(
        "| League | HG Decisions | Entropy Acc_HG "
        "| Gain Acc_HG | Delta | Combined Delta | Better |"
    )
    lines.append(
        "|--------|-------------|----------------"
        "|-------------|-------|----------------|--------|"
    )
    for league in sorted(agg.get("by_league", {}).keys()):
        a = agg["by_league"][league]
        lines.append(
            f"| {league} | {a['n_total_hg_decisions']} "
            f"| {a['entropy_acc_hg']:.4f} "
            f"| {a['gain_acc_hg']:.4f} "
            f"| {a['delta_acc_hg']:+.4f} "
            f"| {a['delta_combined']:+.4f} "
            f"| {_verdict(a['delta_acc_hg'])} |"
        )
    lines.append("")

    # --- By Fold ---
    lines.append("## Results by Fold (all leagues)")
    lines.append("")
    lines.append(
        "| Fold | N_matches | HG Decisions "
        "| Entropy Acc_HG | Gain Acc_HG | Delta | Better |"
    )
    lines.append(
        "|------|----------|-------------"
        "|----------------|-------------|-------|--------|"
    )
    for fold in sorted(agg.get("by_fold", {}).keys()):
        a = agg["by_fold"][fold]
        lines.append(
            f"| {fold} | {a['n_matches']} "
            f"| {a['n_total_hg_decisions']} "
            f"| {a['entropy_acc_hg']:.4f} "
            f"| {a['gain_acc_hg']:.4f} "
            f"| {a['delta_acc_hg']:+.4f} "
            f"| {_verdict(a['delta_acc_hg'])} |"
        )
    lines.append("")

    # --- By League x N_HALF ---
    lines.append("## Results by League x N_HALF")
    lines.append("")
    lines.append(
        "| League | N_HALF | HG Decisions "
        "| Entropy Acc_HG | Gain Acc_HG | Delta | Better |"
    )
    lines.append(
        "|--------|--------|-------------"
        "|----------------|-------------|-------|--------|"
    )
    for (league, nh) in sorted(
        agg.get("by_league_n_half", {}).keys()
    ):
        a = agg["by_league_n_half"][(league, nh)]
        lines.append(
            f"| {league} | {nh} "
            f"| {a['n_total_hg_decisions']} "
            f"| {a['entropy_acc_hg']:.4f} "
            f"| {a['gain_acc_hg']:.4f} "
            f"| {a['delta_acc_hg']:+.4f} "
            f"| {_verdict(a['delta_acc_hg'])} |"
        )
    lines.append("")

    # --- Conclusion ---
    lines.append("## Conclusion")
    lines.append("")

    if tot:
        hg_delta = tot["delta_acc_hg"]
        comb_delta = tot["delta_combined"]
        n_hg = tot["n_total_hg_decisions"]

        if hg_delta > 0.01:
            lines.append(
                "**Gain-based selection is better overall** for "
                "half-guard accuracy "
                f"(Acc_Top2_HG: {tot['gain_acc_hg']:.4f} vs "
                f"{tot['entropy_acc_hg']:.4f}, "
                f"delta {hg_delta:+.4f}) across "
                f"{n_hg} half-guard decisions."
            )
        elif hg_delta < -0.01:
            lines.append(
                "**Entropy-based selection is better overall** for "
                "half-guard accuracy "
                f"(Acc_Top2_HG: {tot['entropy_acc_hg']:.4f} vs "
                f"{tot['gain_acc_hg']:.4f}, "
                f"delta {hg_delta:+.4f}) across "
                f"{n_hg} half-guard decisions."
            )
        else:
            lines.append(
                "**Mixed / no clear winner overall** -- delta is "
                f"{hg_delta:+.4f} across "
                f"{n_hg} half-guard decisions."
            )

        lines.append("")

        seg_gain = 0
        seg_entropy = 0
        seg_tie = 0
        for league in sorted(agg.get("by_league", {}).keys()):
            v = _verdict(agg["by_league"][league]["delta_acc_hg"])
            if v == "gain":
                seg_gain += 1
            elif v == "entropy":
                seg_entropy += 1
            else:
                seg_tie += 1

        lines.append("### Per-segment breakdown")
        lines.append("")
        lines.append(f"- Leagues where gain wins: {seg_gain}")
        lines.append(f"- Leagues where entropy wins: {seg_entropy}")
        lines.append(f"- Leagues tied: {seg_tie}")

        nh_gain = sum(
            1 for v in agg.get("by_n_half", {}).values()
            if _verdict(v["delta_acc_hg"]) == "gain"
        )
        nh_entropy = sum(
            1 for v in agg.get("by_n_half", {}).values()
            if _verdict(v["delta_acc_hg"]) == "entropy"
        )
        nh_tie = sum(
            1 for v in agg.get("by_n_half", {}).values()
            if _verdict(v["delta_acc_hg"]) == "tie"
        )
        lines.append(
            f"- N_HALF settings where gain wins: {nh_gain}"
        )
        lines.append(
            f"- N_HALF settings where entropy wins: {nh_entropy}"
        )
        lines.append(f"- N_HALF settings tied: {nh_tie}")
        lines.append("")
        lines.append(
            "- Overall Acc_Top2_HG delta (gain - entropy): "
            f"{hg_delta:+.4f}"
        )
        lines.append(
            "- Overall Combined delta (gain - entropy): "
            f"{comb_delta:+.4f}"
        )
        lines.append(f"- Total half-guard decisions: {n_hg}")
        lines.append(
            "- Total divergent selections: "
            f"{tot['n_divergent_selections']}"
        )

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
        help="Write markdown report to this file",
    )
    parser.add_argument(
        "--expanded", action="store_true",
        help="Run expanded benchmark matrix "
             "(more folds, leagues, N_HALF values)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=EXPANDED_N_FOLDS,
        help=f"Number of time-based folds "
             f"(default: {EXPANDED_N_FOLDS})",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Write raw CSV results to this file",
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

    if args.expanded:
        n_folds = args.n_folds
        results = run_expanded_comparison(
            df, n_folds=n_folds, n_half_values=N_HALF_VALUES,
        )
        if not results:
            logger.error("No results from expanded benchmark")
            return 1

        agg = aggregate_results(results)
        report = print_expanded_report(
            results, agg, n_folds, N_HALF_VALUES
        )

        out_path = args.output or str(
            _REPO_ROOT / "RESULTS_HALF_GUARD_COMPARISON_EXPANDED.md"
        )
        Path(out_path).write_text(
            report + "\n", encoding="utf-8"
        )
        logger.info("Report written to %s", out_path)

        csv_path = args.csv or str(
            _REPO_ROOT / "results_halfguard_expanded.csv"
        )
        export_csv(results, Path(csv_path))
    else:
        results = run_comparison(df, n_folds=3)
        if results is None:
            return 1

        report = print_comparison(results)

        out_path = args.output or str(
            _REPO_ROOT / "RESULTS_HALF_GUARD_COMPARISON.md"
        )
        Path(out_path).write_text(
            report + "\n", encoding="utf-8"
        )
        logger.info("Report written to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
