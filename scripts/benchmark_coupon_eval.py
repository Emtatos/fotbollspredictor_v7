#!/usr/bin/env python3
"""
Coupon-style evaluation: Entropy vs Gain half-guard selection.

Groups test matches into realistic coupon/ticket blocks and measures
practical outcomes beyond isolated top-2 metrics:

  1. Ticket hit rate      – how often the entire coupon is correct
  2. Correct per coupon   – distribution (mean / median)
  3. Half-guard rescue    – coupons saved by HG that would fail as singles
  4. Comparison across N_HALF values (2, 4, 6)
  5. Divergence           – coupons where gain and entropy pick different HGs

Usage:
    python scripts/benchmark_coupon_eval.py                # use cached data
    python scripts/benchmark_coupon_eval.py --refresh-data # download fresh
    python scripts/benchmark_coupon_eval.py --coupon-size 10  # custom size

The script does NOT touch any production logic.
"""
import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backtest_report import (  # noqa: E402
    get_top2_predictions,
    load_data,
    predict_with_entropy,
    select_halfguards_entropy,
    select_halfguards_gain,
    train_model,
)
from schema import encode_league  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_COUPON_SIZE = 8
DEFAULT_N_FOLDS = 5
DEFAULT_N_HALF_VALUES = [2, 4, 6]


# ---------------------------------------------------------------------------
# Coupon helpers
# ---------------------------------------------------------------------------

def build_coupons(n_matches: int, coupon_size: int) -> List[np.ndarray]:
    """Split match indices into consecutive coupons of fixed size.

    The last coupon is dropped if it has fewer than coupon_size matches
    to keep all coupons comparable.
    """
    indices = np.arange(n_matches)
    n_full = n_matches // coupon_size
    coupons = []
    for i in range(n_full):
        start = i * coupon_size
        coupons.append(indices[start: start + coupon_size])
    return coupons


def evaluate_coupon(
    coupon_indices: np.ndarray,
    y_true: np.ndarray,
    pred_top1: np.ndarray,
    top2_preds: np.ndarray,
    half_guard_set: set,
) -> Dict:
    """Evaluate a single coupon.

    Returns dict with:
        hit: bool – all matches correct
        n_correct: int – number of correct matches in coupon
        n_total: int – coupon size
        hg_rescued: bool – coupon passes with HG but would fail without
    """
    n_correct = 0
    all_correct = True
    all_correct_singles = True  # as if no half-guards

    for idx in coupon_indices:
        true_label = y_true[idx]
        is_hg = idx in half_guard_set

        if is_hg:
            match_ok = true_label in top2_preds[idx]
        else:
            match_ok = true_label == pred_top1[idx]

        # Singles-only evaluation (for rescue calculation)
        single_ok = true_label == pred_top1[idx]

        if match_ok:
            n_correct += 1
        else:
            all_correct = False

        if not single_ok:
            all_correct_singles = False

    rescued = all_correct and not all_correct_singles

    return {
        "hit": all_correct,
        "n_correct": n_correct,
        "n_total": len(coupon_indices),
        "hg_rescued": rescued,
    }


def run_coupon_eval_on_fold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pred_top1: np.ndarray,
    entropy_values: np.ndarray,
    coupon_size: int,
    n_half: int,
) -> Dict:
    """Run coupon evaluation for both entropy and gain on one fold.

    For each coupon within the fold, the half-guard selection is done
    *within that coupon* (not globally), matching how a real user would
    pick half-guards from the matches on their ticket.

    Returns a dict with per-mode results and divergence info.
    """
    n_matches = len(y_true)
    top2_preds = get_top2_predictions(y_proba)
    coupons = build_coupons(n_matches, coupon_size)

    if not coupons:
        return None

    effective_n_half = min(n_half, coupon_size // 2)
    if effective_n_half <= 0:
        return None

    results = {
        "n_coupons": len(coupons),
        "coupon_size": coupon_size,
        "n_half_requested": n_half,
        "n_half_effective": effective_n_half,
        "n_matches_evaluated": len(coupons) * coupon_size,
    }

    for mode in ("gain", "entropy"):
        hits = 0
        correct_counts = []
        rescues = 0

        for coupon_idx in coupons:
            # Select half-guards within this coupon
            coupon_proba = y_proba[coupon_idx]
            coupon_entropy = entropy_values[coupon_idx]

            if mode == "gain":
                local_hg_indices = select_halfguards_gain(
                    coupon_proba, effective_n_half
                )
            else:
                local_hg_indices = select_halfguards_entropy(
                    coupon_proba, coupon_entropy, effective_n_half
                )

            # Map local indices back to global
            hg_global = set(coupon_idx[local_hg_indices].tolist())

            result = evaluate_coupon(
                coupon_idx, y_true, pred_top1, top2_preds, hg_global
            )

            if result["hit"]:
                hits += 1
            correct_counts.append(result["n_correct"])
            if result["hg_rescued"]:
                rescues += 1

        correct_arr = np.array(correct_counts)
        results[f"{mode}_ticket_hit_rate"] = hits / len(coupons)
        results[f"{mode}_ticket_hits"] = hits
        results[f"{mode}_mean_correct"] = float(np.mean(correct_arr))
        results[f"{mode}_median_correct"] = float(np.median(correct_arr))
        results[f"{mode}_rescues"] = rescues
        results[f"{mode}_rescue_rate"] = rescues / len(coupons)

    # Divergence: count coupons where gain and entropy pick different HGs
    n_divergent = 0
    for coupon_idx in coupons:
        coupon_proba = y_proba[coupon_idx]
        coupon_entropy = entropy_values[coupon_idx]

        gain_local = set(
            select_halfguards_gain(coupon_proba, effective_n_half).tolist()
        )
        entropy_local = set(
            select_halfguards_entropy(
                coupon_proba, coupon_entropy, effective_n_half
            ).tolist()
        )

        if gain_local != entropy_local:
            n_divergent += 1

    results["n_divergent_coupons"] = n_divergent
    results["divergence_rate"] = n_divergent / len(coupons)

    return results


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    df: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    coupon_size: int = DEFAULT_COUPON_SIZE,
    n_half_values: List[int] = None,
) -> List[Dict]:
    """Run coupon evaluation across folds and N_HALF values.

    Returns a list of per-cell result dicts.
    """
    if n_half_values is None:
        n_half_values = DEFAULT_N_HALF_VALUES

    if df.empty:
        logger.error("Empty dataframe")
        return []

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["time_fold"] = pd.qcut(
        df["Date"], q=n_folds, labels=False, duplicates="drop"
    )

    all_results = []

    for fold_idx in range(1, n_folds):
        train_mask = df["time_fold"] < fold_idx
        test_mask = df["time_fold"] == fold_idx
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        if len(df_train) < 100 or len(df_test) < 20:
            logger.warning(
                "Fold %d: insufficient data (train=%d, test=%d), skipping",
                fold_idx, len(df_train), len(df_test),
            )
            continue

        logger.info(
            "Fold %d: train=%d, test=%d",
            fold_idx, len(df_train), len(df_test),
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

        seen_effective = set()
        for n_half in n_half_values:
            effective = min(n_half, coupon_size // 2)
            if effective in seen_effective:
                logger.info(
                    "Fold %d: N_HALF=%d -> effective=%d already "
                    "evaluated, skipping",
                    fold_idx, n_half, effective,
                )
                continue
            seen_effective.add(effective)

            result = run_coupon_eval_on_fold(
                y_true, y_proba, pred_top1, entropy_values,
                coupon_size=coupon_size,
                n_half=n_half,
            )
            if result is not None:
                result["fold"] = fold_idx
                result["n_test_matches"] = len(y_true)

                # Per-league breakdown of top1 accuracy for context
                if league_codes is not None:
                    from sklearn.metrics import accuracy_score
                    result["overall_top1_acc"] = float(
                        accuracy_score(y_true, pred_top1)
                    )

                all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_coupon_results(results: List[Dict]) -> Dict:
    """Aggregate per-cell results into summary tables."""
    if not results:
        return {}

    def _agg(subset: List[Dict]) -> Optional[Dict]:
        if not subset:
            return None

        n_coupons = sum(r["n_coupons"] for r in subset)
        weights = [r["n_coupons"] for r in subset]

        gain_hit = np.average(
            [r["gain_ticket_hit_rate"] for r in subset], weights=weights
        )
        entropy_hit = np.average(
            [r["entropy_ticket_hit_rate"] for r in subset], weights=weights
        )

        gain_mean_corr = np.average(
            [r["gain_mean_correct"] for r in subset], weights=weights
        )
        entropy_mean_corr = np.average(
            [r["entropy_mean_correct"] for r in subset], weights=weights
        )

        gain_rescues = sum(r["gain_rescues"] for r in subset)
        entropy_rescues = sum(r["entropy_rescues"] for r in subset)

        gain_rescue_rate = gain_rescues / n_coupons if n_coupons > 0 else 0
        entropy_rescue_rate = (
            entropy_rescues / n_coupons if n_coupons > 0 else 0
        )

        n_divergent = sum(r["n_divergent_coupons"] for r in subset)

        return {
            "n_cells": len(subset),
            "n_coupons": n_coupons,
            "gain_ticket_hit_rate": float(gain_hit),
            "entropy_ticket_hit_rate": float(entropy_hit),
            "delta_ticket_hit_rate": float(gain_hit - entropy_hit),
            "gain_mean_correct": float(gain_mean_corr),
            "entropy_mean_correct": float(entropy_mean_corr),
            "delta_mean_correct": float(gain_mean_corr - entropy_mean_corr),
            "gain_rescues": gain_rescues,
            "entropy_rescues": entropy_rescues,
            "gain_rescue_rate": float(gain_rescue_rate),
            "entropy_rescue_rate": float(entropy_rescue_rate),
            "delta_rescue_rate": float(gain_rescue_rate - entropy_rescue_rate),
            "n_divergent_coupons": n_divergent,
            "divergence_rate": float(n_divergent / n_coupons)
            if n_coupons > 0
            else 0.0,
        }

    agg = {}
    agg["total"] = _agg(results)

    # By N_HALF (use effective value for grouping)
    nh_seen = sorted(set(r["n_half_effective"] for r in results))
    agg["by_n_half"] = {}
    for nh in nh_seen:
        subset = [r for r in results if r["n_half_effective"] == nh]
        agg["by_n_half"][nh] = _agg(subset)

    # By fold
    folds_seen = sorted(set(r["fold"] for r in results))
    agg["by_fold"] = {}
    for fold in folds_seen:
        subset = [r for r in results if r["fold"] == fold]
        agg["by_fold"][fold] = _agg(subset)

    return agg


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_coupon_csv(results: List[Dict], path: Path) -> None:
    """Export raw per-cell results to CSV."""
    if not results:
        return

    fieldnames = [
        "fold", "coupon_size", "n_half_requested", "n_half_effective",
        "n_coupons", "n_test_matches", "n_matches_evaluated",
        "gain_ticket_hit_rate", "gain_ticket_hits",
        "gain_mean_correct", "gain_median_correct",
        "gain_rescues", "gain_rescue_rate",
        "entropy_ticket_hit_rate", "entropy_ticket_hits",
        "entropy_mean_correct", "entropy_median_correct",
        "entropy_rescues", "entropy_rescue_rate",
        "n_divergent_coupons", "divergence_rate",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {}
            for k in fieldnames:
                val = r.get(k)
                if isinstance(val, float):
                    row[k] = round(val, 6)
                else:
                    row[k] = val
            writer.writerow(row)

    logger.info("CSV written to %s (%d rows)", path, len(results))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _verdict(delta: float, threshold: float = 0.001) -> str:
    if delta > threshold:
        return "gain"
    elif delta < -threshold:
        return "entropy"
    return "tie"


def generate_report(
    results: List[Dict],
    agg: Dict,
    coupon_size: int,
    n_half_values: List[int],
    n_folds: int,
) -> str:
    """Generate a markdown report."""
    if not results or not agg:
        return "No results to report."

    lines = []

    lines.append(
        "# Half-Guard Coupon Evaluation: Entropy vs Gain"
    )
    lines.append("")

    # --- Setup ---
    lines.append("## Setup")
    lines.append("")
    lines.append("### Coupon definition")
    lines.append("")
    lines.append(
        f"- **Coupon size**: {coupon_size} matches per coupon"
    )
    lines.append(
        "- Matches are grouped chronologically into consecutive "
        "coupons within each test fold"
    )
    lines.append(
        "- Incomplete coupons (< coupon_size) are dropped"
    )
    lines.append(
        "- Half-guard selection is done **per coupon** "
        "(not globally), matching real usage"
    )
    lines.append("")
    lines.append("### Evaluation rules")
    lines.append("")
    lines.append(
        "- **Single matches** (not half-guarded): correct if "
        "top-1 prediction matches actual result"
    )
    lines.append(
        "- **Half-guard matches**: correct if actual result is "
        "in top-2 predictions"
    )
    lines.append(
        "- **Coupon hit**: ALL matches in the coupon are correct"
    )
    lines.append(
        "- **HG rescue**: coupon passes with half-guards but "
        "would have failed with singles only"
    )
    lines.append("")

    lines.append("### Parameters")
    lines.append("")
    lines.append(f"- Walk-forward folds: {n_folds}")
    lines.append(
        f"- N_HALF values: "
        f"{', '.join(str(x) for x in n_half_values)}"
    )
    lines.append("- Leagues: E0, E1, E2, E3")

    tot = agg.get("total")
    if tot:
        lines.append(f"- Total coupons evaluated: {tot['n_coupons']}")
    lines.append("")

    # --- Overall Summary ---
    lines.append("## Overall Summary")
    lines.append("")

    if tot:
        lines.append(
            "| Metric | Entropy | Gain "
            "| Delta (gain-entropy) | Better |"
        )
        lines.append(
            "|--------|---------|------"
            "|----------------------|--------|"
        )

        lines.append(
            f"| Ticket hit rate | {tot['entropy_ticket_hit_rate']:.4f} "
            f"| {tot['gain_ticket_hit_rate']:.4f} "
            f"| {tot['delta_ticket_hit_rate']:+.4f} "
            f"| {_verdict(tot['delta_ticket_hit_rate'])} |"
        )
        lines.append(
            f"| Mean correct/coupon | "
            f"{tot['entropy_mean_correct']:.2f} "
            f"| {tot['gain_mean_correct']:.2f} "
            f"| {tot['delta_mean_correct']:+.4f} "
            f"| {_verdict(tot['delta_mean_correct'], 0.01)} |"
        )
        lines.append(
            f"| HG rescue rate | "
            f"{tot['entropy_rescue_rate']:.4f} "
            f"| {tot['gain_rescue_rate']:.4f} "
            f"| {tot['delta_rescue_rate']:+.4f} "
            f"| {_verdict(tot['delta_rescue_rate'])} |"
        )
        lines.append(
            f"| HG rescues (count) | "
            f"{tot['entropy_rescues']} "
            f"| {tot['gain_rescues']} "
            f"| {tot['gain_rescues'] - tot['entropy_rescues']:+d} "
            f"| {_verdict(tot['gain_rescues'] - tot['entropy_rescues'])} |"
        )
        lines.append("")

        lines.append(
            f"- Coupons where gain and entropy selected different "
            f"half-guards: **{tot['n_divergent_coupons']}** / "
            f"{tot['n_coupons']} "
            f"({tot['divergence_rate']:.1%})"
        )
        lines.append("")

    # --- By N_HALF ---
    lines.append("## Results by N_HALF")
    lines.append("")
    lines.append(
        "| N_HALF | Coupons | Entropy Hit Rate | Gain Hit Rate "
        "| Delta | Entropy Rescues | Gain Rescues | Divergent |"
    )
    lines.append(
        "|--------|---------|-----------------|-------------- "
        "|-------|-----------------|--------------|-----------|"
    )

    for nh in sorted(agg.get("by_n_half", {}).keys()):
        a = agg["by_n_half"][nh]
        lines.append(
            f"| {nh} | {a['n_coupons']} "
            f"| {a['entropy_ticket_hit_rate']:.4f} "
            f"| {a['gain_ticket_hit_rate']:.4f} "
            f"| {a['delta_ticket_hit_rate']:+.4f} "
            f"| {a['entropy_rescues']} "
            f"| {a['gain_rescues']} "
            f"| {a['n_divergent_coupons']} |"
        )
    lines.append("")

    # --- Mean correct by N_HALF ---
    lines.append("## Mean Correct per Coupon by N_HALF")
    lines.append("")
    lines.append(
        "| N_HALF | Entropy Mean | Gain Mean | Delta |"
    )
    lines.append(
        "|--------|-------------|-----------|-------|"
    )
    for nh in sorted(agg.get("by_n_half", {}).keys()):
        a = agg["by_n_half"][nh]
        lines.append(
            f"| {nh} | {a['entropy_mean_correct']:.2f} "
            f"| {a['gain_mean_correct']:.2f} "
            f"| {a['delta_mean_correct']:+.4f} |"
        )
    lines.append("")

    # --- By Fold ---
    lines.append("## Results by Fold")
    lines.append("")
    lines.append(
        "| Fold | Coupons | Entropy Hit Rate | Gain Hit Rate "
        "| Delta | Better |"
    )
    lines.append(
        "|------|---------|-----------------|-------------- "
        "|-------|--------|"
    )
    for fold in sorted(agg.get("by_fold", {}).keys()):
        a = agg["by_fold"][fold]
        lines.append(
            f"| {fold} | {a['n_coupons']} "
            f"| {a['entropy_ticket_hit_rate']:.4f} "
            f"| {a['gain_ticket_hit_rate']:.4f} "
            f"| {a['delta_ticket_hit_rate']:+.4f} "
            f"| {_verdict(a['delta_ticket_hit_rate'])} |"
        )
    lines.append("")

    # --- Per-cell detail ---
    lines.append("## Detailed Per-Cell Results")
    lines.append("")
    lines.append(
        "| Fold | N_HALF | Coupons | Entropy Hits | Gain Hits "
        "| E Rescues | G Rescues | Divergent |"
    )
    lines.append(
        "|------|--------|---------|-------------|---------- "
        "|-----------|-----------|-----------|"
    )
    for r in sorted(results, key=lambda x: (x["fold"], x["n_half_effective"])):
        lines.append(
            f"| {r['fold']} | {r['n_half_effective']} "
            f"| {r['n_coupons']} "
            f"| {r['entropy_ticket_hits']}/{r['n_coupons']} "
            f"| {r['gain_ticket_hits']}/{r['n_coupons']} "
            f"| {r['entropy_rescues']} "
            f"| {r['gain_rescues']} "
            f"| {r['n_divergent_coupons']} |"
        )
    lines.append("")

    # --- Conclusion ---
    lines.append("## Conclusion")
    lines.append("")

    if tot:
        delta_hit = tot["delta_ticket_hit_rate"]
        delta_rescue = tot["delta_rescue_rate"]
        delta_correct = tot["delta_mean_correct"]

        # Overall assessment
        signals = []
        if delta_hit > 0.001:
            signals.append("gain_better")
        elif delta_hit < -0.001:
            signals.append("entropy_better")
        else:
            signals.append("tie_hit")

        if delta_rescue > 0.001:
            signals.append("gain_rescues_more")
        elif delta_rescue < -0.001:
            signals.append("entropy_rescues_more")
        else:
            signals.append("tie_rescue")

        if delta_correct > 0.01:
            signals.append("gain_more_correct")
        elif delta_correct < -0.01:
            signals.append("entropy_more_correct")
        else:
            signals.append("tie_correct")

        gain_wins = sum(1 for s in signals if "gain" in s)
        entropy_wins = sum(1 for s in signals if "entropy" in s)

        if gain_wins > entropy_wins:
            overall = "better"
            verdict_text = (
                "**Gain-based selection produces better practical "
                "coupon outcomes** than entropy-based selection."
            )
        elif entropy_wins > gain_wins:
            overall = "better (entropy)"
            verdict_text = (
                "**Entropy-based selection produces better practical "
                "coupon outcomes** than gain-based selection."
            )
        else:
            if abs(delta_hit) < 0.001 and abs(delta_rescue) < 0.001:
                overall = "unchanged"
                verdict_text = (
                    "**No meaningful difference** between gain and "
                    "entropy in practical coupon outcomes."
                )
            else:
                overall = "mixed"
                verdict_text = (
                    "**Mixed results** -- gain and entropy each have "
                    "advantages in different metrics."
                )

        lines.append(f"### Verdict: {overall}")
        lines.append("")
        lines.append(verdict_text)
        lines.append("")

        lines.append("### Key numbers")
        lines.append("")
        lines.append(
            f"- Ticket hit rate delta (gain - entropy): "
            f"**{delta_hit:+.4f}**"
        )
        lines.append(
            f"- Mean correct/coupon delta: "
            f"**{delta_correct:+.4f}**"
        )
        lines.append(
            f"- HG rescue rate delta: "
            f"**{delta_rescue:+.4f}**"
        )
        lines.append(
            f"- HG rescue count: gain={tot['gain_rescues']}, "
            f"entropy={tot['entropy_rescues']}"
        )
        lines.append(
            f"- Divergent selections: "
            f"{tot['n_divergent_coupons']}/{tot['n_coupons']} "
            f"({tot['divergence_rate']:.1%})"
        )

        # Per N_HALF verdict
        lines.append("")
        lines.append("### Per-N_HALF verdict")
        lines.append("")
        for nh in sorted(agg.get("by_n_half", {}).keys()):
            a = agg["by_n_half"][nh]
            v = _verdict(a["delta_ticket_hit_rate"])
            lines.append(
                f"- N_HALF={nh}: ticket hit delta "
                f"{a['delta_ticket_hit_rate']:+.4f} "
                f"-> **{v}**"
            )

    lines.append("")

    report = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coupon-style evaluation of entropy vs gain "
            "half-guard selection"
        ),
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh data instead of using cache",
    )
    parser.add_argument(
        "--coupon-size",
        type=int,
        default=DEFAULT_COUPON_SIZE,
        help=f"Matches per coupon (default: {DEFAULT_COUPON_SIZE})",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Number of time folds (default: {DEFAULT_N_FOLDS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    refresh = (
        args.refresh_data
        or os.environ.get("BACKTEST_REFRESH_DATA", "").lower()
        in ("1", "true", "yes")
    )

    logger.info("Loading data (refresh=%s)...", refresh)
    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("Failed to load data")
        return 1

    logger.info("Loaded %d matches", len(df))

    coupon_size = args.coupon_size
    n_folds = args.n_folds
    n_half_values = DEFAULT_N_HALF_VALUES

    logger.info(
        "Running coupon evaluation: coupon_size=%d, n_folds=%d, "
        "n_half_values=%s",
        coupon_size, n_folds, n_half_values,
    )

    results = run_full_evaluation(
        df,
        n_folds=n_folds,
        coupon_size=coupon_size,
        n_half_values=n_half_values,
    )

    if not results:
        logger.error("No results produced")
        return 1

    agg = aggregate_coupon_results(results)

    # Export CSV
    csv_path = _REPO_ROOT / "results_coupon_eval.csv"
    export_coupon_csv(results, csv_path)

    # Generate and write report
    report = generate_report(
        results, agg, coupon_size, n_half_values, n_folds
    )
    report_path = _REPO_ROOT / "RESULTS_HALF_GUARD_COUPON_EVAL.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Also print
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
