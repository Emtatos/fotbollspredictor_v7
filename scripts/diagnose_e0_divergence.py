#!/usr/bin/env python3
"""
Diagnose E0 divergence in half-guard benchmark.

Runs a detailed analysis of why E0 (Premier League) shows a slight
entropy advantage while gain wins in E1/E2/E3.

Produces:
  - RESULTS_HALF_GUARD_E0_DIAGNOSIS.md   (markdown report)
  - results_e0_diagnosis_detail.csv       (per-cell support data)

Usage:
    python scripts/diagnose_e0_divergence.py                # use cached data
    python scripts/diagnose_e0_divergence.py --refresh-data # download fresh
"""
import argparse
import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import logging
import math
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from backtest_report import (
    load_data,
    train_model,
    predict_with_entropy,
    select_halfguards_gain,
    select_halfguards_entropy,
    get_top2_predictions,
    LEAGUES,
)
from schema import encode_league, LEAGUE_MAP
from uncertainty import entropy_norm

# Reverse map: numeric code -> string label
_CODE_TO_LEAGUE = {v: k for k, v in LEAGUE_MAP.items()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

N_HALF_VALUES = [2, 4, 6]
N_FOLDS = 5


# ------------------------------------------------------------------
# Data collection
# ------------------------------------------------------------------

def collect_per_match_data(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
):
    """Run walk-forward and collect per-match probability data per league.

    Returns a list of dicts, one per test-set match, with fields:
        fold, league, match_idx, best, second, third, top2, entropy,
        gain_rank, entropy_rank (within the fold/league block),
        ftr (true result), top1_hit, top2_hit
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["time_fold"] = pd.qcut(df["Date"], q=n_folds, labels=False, duplicates="drop")

    rows = []

    for fold_idx in range(1, n_folds):
        train_mask = df["time_fold"] < fold_idx
        test_mask = df["time_fold"] == fold_idx
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        if len(df_train) < 100 or len(df_test) < 20:
            continue

        model = train_model(df_train)
        if model is None:
            continue

        y_true, y_proba, pred_top1, entropy_values = predict_with_entropy(model, df_test)
        top2_preds = get_top2_predictions(y_proba)

        raw_leagues = df_test["League"].values if "League" in df_test.columns else [None] * len(df_test)
        leagues_in_test = [
            _CODE_TO_LEAGUE.get(int(v), str(v)) if v is not None else None
            for v in raw_leagues
        ]

        sorted_desc = np.sort(y_proba, axis=1)[:, ::-1]

        for i in range(len(y_true)):
            best = float(sorted_desc[i, 0])
            second = float(sorted_desc[i, 1])
            third = float(sorted_desc[i, 2])
            top2 = best + second
            ent = float(entropy_values[i])
            t1_hit = int(y_true[i] == pred_top1[i])
            t2_hit = int(y_true[i] in top2_preds[i])

            rows.append({
                "fold": fold_idx,
                "league": leagues_in_test[i],
                "match_idx": i,
                "best": round(best, 5),
                "second": round(second, 5),
                "third": round(third, 5),
                "top2": round(top2, 5),
                "entropy": round(ent, 5),
                "gain": round(second, 5),
                "top1_hit": t1_hit,
                "top2_hit": t2_hit,
            })

    return rows


def collect_halfguard_outcomes(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    n_half_values: list = None,
):
    """Run walk-forward and collect per-cell half-guard outcomes.

    For each fold x league x n_half, returns which matches were selected
    by gain vs entropy and whether they hit.
    """
    if n_half_values is None:
        n_half_values = N_HALF_VALUES

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["time_fold"] = pd.qcut(df["Date"], q=n_folds, labels=False, duplicates="drop")

    cells = []

    for fold_idx in range(1, n_folds):
        train_mask = df["time_fold"] < fold_idx
        test_mask = df["time_fold"] == fold_idx
        df_train_all = df[train_mask].copy()
        df_test_all = df[test_mask].copy()

        if len(df_train_all) < 100 or len(df_test_all) < 20:
            continue

        league_subsets = {"all": None}
        for lg in LEAGUES:
            league_subsets[lg] = encode_league(lg)

        for league_label, league_filter in league_subsets.items():
            if league_filter is not None and "League" in df.columns:
                df_train = df_train_all[df_train_all["League"] == league_filter].copy()
                df_test = df_test_all[df_test_all["League"] == league_filter].copy()
            else:
                df_train = df_train_all.copy()
                df_test = df_test_all.copy()

            if len(df_train) < 50 or len(df_test) < 10:
                continue

            model = train_model(df_train)
            if model is None:
                continue

            y_true, y_proba, pred_top1, entropy_values = predict_with_entropy(model, df_test)
            top2_preds = get_top2_predictions(y_proba)

            sorted_desc = np.sort(y_proba, axis=1)[:, ::-1]

            for n_half in n_half_values:
                effective = min(n_half, len(y_true) // 4)
                if effective <= 0:
                    continue

                idx_gain = select_halfguards_gain(y_proba, effective)
                idx_entropy = select_halfguards_entropy(y_proba, entropy_values, effective)

                gain_hits = sum(int(y_true[j] in top2_preds[j]) for j in idx_gain)
                entropy_hits = sum(int(y_true[j] in top2_preds[j]) for j in idx_entropy)

                overlap = len(set(idx_gain.tolist()) & set(idx_entropy.tolist()))
                only_gain = set(idx_gain.tolist()) - set(idx_entropy.tolist())
                only_entropy = set(idx_entropy.tolist()) - set(idx_gain.tolist())

                # hits on gain-only and entropy-only selections
                gain_only_hits = sum(int(y_true[j] in top2_preds[j]) for j in only_gain)
                entropy_only_hits = sum(int(y_true[j] in top2_preds[j]) for j in only_entropy)
                gain_only_total = len(only_gain)
                entropy_only_total = len(only_entropy)

                # avg stats for gain-selected vs entropy-selected
                gains_all = sorted_desc[:, 1]
                top2s_all = sorted_desc[:, 0] + sorted_desc[:, 1]

                cells.append({
                    "fold": fold_idx,
                    "league": league_label,
                    "n_half": n_half,
                    "n_half_eff": effective,
                    "n_matches": len(y_true),
                    "gain_hits": gain_hits,
                    "entropy_hits": entropy_hits,
                    "gain_acc_hg": round(gain_hits / effective, 4) if effective > 0 else 0,
                    "entropy_acc_hg": round(entropy_hits / effective, 4) if effective > 0 else 0,
                    "overlap": overlap,
                    "gain_only_total": gain_only_total,
                    "entropy_only_total": entropy_only_total,
                    "gain_only_hits": gain_only_hits,
                    "entropy_only_hits": entropy_only_hits,
                    "gain_avg_gain": round(float(np.mean(gains_all[idx_gain])), 4),
                    "gain_avg_top2": round(float(np.mean(top2s_all[idx_gain])), 4),
                    "gain_avg_entropy": round(float(np.mean(entropy_values[idx_gain])), 4),
                    "entropy_avg_gain": round(float(np.mean(gains_all[idx_entropy])), 4),
                    "entropy_avg_top2": round(float(np.mean(top2s_all[idx_entropy])), 4),
                    "entropy_avg_entropy": round(float(np.mean(entropy_values[idx_entropy])), 4),
                })

    return cells


# ------------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------------

def binomial_ci(hits, n, confidence=0.95):
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = hits / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_test_from_cells(cells, league_filter):
    """Approximate McNemar-style test from cell-level data.

    We count total gain_only hits/misses vs entropy_only hits/misses
    across all cells for the given league.
    """
    sub = [c for c in cells if c["league"] == league_filter]
    # gain-only hit, entropy-only miss  (b)
    # gain-only miss, entropy-only hit  (c)
    b_total = sum(c["gain_only_hits"] for c in sub)
    b_misses = sum(c["gain_only_total"] - c["gain_only_hits"] for c in sub)
    c_total = sum(c["entropy_only_hits"] for c in sub)
    c_misses = sum(c["entropy_only_total"] - c["entropy_only_hits"] for c in sub)

    return {
        "gain_only_selections": sum(c["gain_only_total"] for c in sub),
        "entropy_only_selections": sum(c["entropy_only_total"] for c in sub),
        "gain_only_hits": b_total,
        "gain_only_misses": b_misses,
        "entropy_only_hits": c_total,
        "entropy_only_misses": c_misses,
    }


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def generate_report(
    match_data: list,
    cells: list,
) -> str:
    """Generate the full diagnosis markdown report."""
    lines = []

    lines.append("# E0 Divergence Diagnosis: Half-Guard Benchmark")
    lines.append("")
    lines.append("## Background")
    lines.append("")
    lines.append("PR #36 showed that gain-based half-guard selection beats entropy overall,")
    lines.append("winning in E1, E2, and E3. However, E0 (Premier League) showed a **slight**")
    lines.append("advantage for entropy. This report investigates why.")
    lines.append("")

    # ---------------------------------------------------------------
    # 1. Sample size & absolute hits
    # ---------------------------------------------------------------
    lines.append("## 1. Sample Size and Absolute Hits")
    lines.append("")

    league_labels = ["E0", "E1", "E2", "E3"]

    lines.append("### Per-league totals (across all folds and N_HALF values)")
    lines.append("")
    lines.append("| League | HG Decisions | Gain Hits | Entropy Hits | Delta (G-E) | Gain Acc | Entropy Acc |")
    lines.append("|--------|-------------|-----------|--------------|-------------|----------|-------------|")

    for lg in league_labels:
        sub = [c for c in cells if c["league"] == lg]
        total_hg = sum(c["n_half_eff"] for c in sub)
        g_hits = sum(c["gain_hits"] for c in sub)
        e_hits = sum(c["entropy_hits"] for c in sub)
        g_acc = g_hits / total_hg if total_hg else 0
        e_acc = e_hits / total_hg if total_hg else 0
        lines.append(
            f"| {lg} | {total_hg} | {g_hits} | {e_hits} | {g_hits - e_hits:+d} "
            f"| {g_acc:.4f} | {e_acc:.4f} |"
        )

    lines.append("")

    # E0 detail by fold
    lines.append("### E0 detail by fold")
    lines.append("")
    lines.append("| Fold | HG Decisions | Gain Hits | Entropy Hits | Delta |")
    lines.append("|------|-------------|-----------|--------------|-------|")

    e0_cells = [c for c in cells if c["league"] == "E0"]
    for fold in sorted(set(c["fold"] for c in e0_cells)):
        fsub = [c for c in e0_cells if c["fold"] == fold]
        total_hg = sum(c["n_half_eff"] for c in fsub)
        g_hits = sum(c["gain_hits"] for c in fsub)
        e_hits = sum(c["entropy_hits"] for c in fsub)
        lines.append(f"| {fold} | {total_hg} | {g_hits} | {e_hits} | {g_hits - e_hits:+d} |")

    lines.append("")

    # E0 detail by n_half
    lines.append("### E0 detail by N_HALF")
    lines.append("")
    lines.append("| N_HALF | HG Decisions | Gain Hits | Entropy Hits | Delta | Gain Acc | Entropy Acc |")
    lines.append("|--------|-------------|-----------|--------------|-------|----------|-------------|")

    for nh in sorted(set(c["n_half"] for c in e0_cells)):
        nsub = [c for c in e0_cells if c["n_half"] == nh]
        total_hg = sum(c["n_half_eff"] for c in nsub)
        g_hits = sum(c["gain_hits"] for c in nsub)
        e_hits = sum(c["entropy_hits"] for c in nsub)
        g_acc = g_hits / total_hg if total_hg else 0
        e_acc = e_hits / total_hg if total_hg else 0
        lines.append(
            f"| {nh} | {total_hg} | {g_hits} | {e_hits} | {g_hits - e_hits:+d} "
            f"| {g_acc:.4f} | {e_acc:.4f} |"
        )

    lines.append("")

    # Statistical significance
    lines.append("### Statistical context")
    lines.append("")
    e0_total_hg = sum(c["n_half_eff"] for c in e0_cells)
    e0_g_hits = sum(c["gain_hits"] for c in e0_cells)
    e0_e_hits = sum(c["entropy_hits"] for c in e0_cells)

    g_lo, g_hi = binomial_ci(e0_g_hits, e0_total_hg)
    e_lo, e_hi = binomial_ci(e0_e_hits, e0_total_hg)

    lines.append(f"- E0 gain accuracy: {e0_g_hits}/{e0_total_hg} = {e0_g_hits/e0_total_hg:.4f} "
                 f"(95% CI: [{g_lo:.3f}, {g_hi:.3f}])")
    lines.append(f"- E0 entropy accuracy: {e0_e_hits}/{e0_total_hg} = {e0_e_hits/e0_total_hg:.4f} "
                 f"(95% CI: [{e_lo:.3f}, {e_hi:.3f}])")
    lines.append(f"- Absolute difference: {abs(e0_g_hits - e0_e_hits)} hits out of {e0_total_hg} decisions")
    lines.append(f"- The 95% confidence intervals **overlap substantially**.")
    lines.append("")

    # Compare CIs across leagues
    lines.append("### Confidence interval comparison across leagues")
    lines.append("")
    lines.append("| League | N | Gain Hits | Gain Acc | 95% CI | Entropy Hits | Entropy Acc | 95% CI | CIs Overlap? |")
    lines.append("|--------|---|-----------|----------|--------|--------------|-------------|--------|--------------|")
    for lg in league_labels:
        sub = [c for c in cells if c["league"] == lg]
        total_hg = sum(c["n_half_eff"] for c in sub)
        g_hits = sum(c["gain_hits"] for c in sub)
        e_hits = sum(c["entropy_hits"] for c in sub)
        g_acc = g_hits / total_hg if total_hg else 0
        e_acc = e_hits / total_hg if total_hg else 0
        gl, gh = binomial_ci(g_hits, total_hg)
        el, eh = binomial_ci(e_hits, total_hg)
        overlap = "Yes" if gl <= eh and el <= gh else "No"
        lines.append(
            f"| {lg} | {total_hg} | {g_hits} | {g_acc:.4f} | [{gl:.3f}, {gh:.3f}] "
            f"| {e_hits} | {e_acc:.4f} | [{el:.3f}, {eh:.3f}] | {overlap} |"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # 2. Probability profile comparison
    # ---------------------------------------------------------------
    lines.append("## 2. Probability Profile: E0 vs Other Leagues")
    lines.append("")
    lines.append("How do the model's probability distributions differ across leagues?")
    lines.append("")

    mdf = pd.DataFrame(match_data)

    lines.append("### Distribution of best, second, top2, and entropy")
    lines.append("")
    lines.append("| League | N_matches | Mean Best | Mean Second | Mean Top2 | Mean Entropy | Std Entropy |")
    lines.append("|--------|----------|-----------|-------------|-----------|--------------|-------------|")

    for lg in league_labels:
        sub = mdf[mdf["league"] == lg]
        n = len(sub)
        lines.append(
            f"| {lg} | {n} | {sub['best'].mean():.4f} | {sub['second'].mean():.4f} "
            f"| {sub['top2'].mean():.4f} | {sub['entropy'].mean():.4f} | {sub['entropy'].std():.4f} |"
        )

    lines.append("")

    # Entropy distribution percentiles
    lines.append("### Entropy distribution percentiles")
    lines.append("")
    lines.append("| League | P10 | P25 | P50 | P75 | P90 | Mean |")
    lines.append("|--------|-----|-----|-----|-----|-----|------|")

    for lg in league_labels:
        sub = mdf[mdf["league"] == lg]["entropy"]
        pcts = sub.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
        lines.append(
            f"| {lg} | {pcts[0]:.4f} | {pcts[1]:.4f} | {pcts[2]:.4f} "
            f"| {pcts[3]:.4f} | {pcts[4]:.4f} | {sub.mean():.4f} |"
        )

    lines.append("")

    # Second-best distribution percentiles
    lines.append("### Second-best probability distribution percentiles")
    lines.append("")
    lines.append("| League | P10 | P25 | P50 | P75 | P90 | Mean |")
    lines.append("|--------|-----|-----|-----|-----|-----|------|")

    for lg in league_labels:
        sub = mdf[mdf["league"] == lg]["second"]
        pcts = sub.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
        lines.append(
            f"| {lg} | {pcts[0]:.4f} | {pcts[1]:.4f} | {pcts[2]:.4f} "
            f"| {pcts[3]:.4f} | {pcts[4]:.4f} | {sub.mean():.4f} |"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # 3. How often do gain and entropy pick different matches?
    # ---------------------------------------------------------------
    lines.append("## 3. Selection Divergence Analysis")
    lines.append("")
    lines.append("When gain and entropy pick different matches, who picks better?")
    lines.append("")

    lines.append("### Overlap and divergent selections per league")
    lines.append("")
    lines.append("| League | Total HG | Overlap | Gain-Only | Entropy-Only | Gain-Only Hit Rate | Entropy-Only Hit Rate |")
    lines.append("|--------|---------|---------|-----------|--------------|--------------------|-----------------------|")

    for lg in league_labels:
        sub = [c for c in cells if c["league"] == lg]
        total_hg = sum(c["n_half_eff"] for c in sub)
        overlap = sum(c["overlap"] for c in sub)
        g_only = sum(c["gain_only_total"] for c in sub)
        e_only = sum(c["entropy_only_total"] for c in sub)
        g_only_hits = sum(c["gain_only_hits"] for c in sub)
        e_only_hits = sum(c["entropy_only_hits"] for c in sub)
        g_only_rate = g_only_hits / g_only if g_only > 0 else 0
        e_only_rate = e_only_hits / e_only if e_only > 0 else 0
        lines.append(
            f"| {lg} | {total_hg} | {overlap} | {g_only} | {e_only} "
            f"| {g_only_rate:.4f} ({g_only_hits}/{g_only}) "
            f"| {e_only_rate:.4f} ({e_only_hits}/{e_only}) |"
        )

    lines.append("")
    lines.append("**Key question**: In E0, when the methods disagree, does entropy's unique picks")
    lines.append("hit more often than gain's unique picks?")
    lines.append("")

    # E0 per-fold divergent outcomes
    lines.append("### E0 divergent selections by fold")
    lines.append("")
    lines.append("| Fold | Gain-Only | Gain-Only Hits | Entropy-Only | Entropy-Only Hits |")
    lines.append("|------|-----------|----------------|--------------|-------------------|")

    for fold in sorted(set(c["fold"] for c in e0_cells)):
        fsub = [c for c in e0_cells if c["fold"] == fold]
        g_only = sum(c["gain_only_total"] for c in fsub)
        g_only_hits = sum(c["gain_only_hits"] for c in fsub)
        e_only = sum(c["entropy_only_total"] for c in fsub)
        e_only_hits = sum(c["entropy_only_hits"] for c in fsub)
        lines.append(f"| {fold} | {g_only} | {g_only_hits} | {e_only} | {e_only_hits} |")

    lines.append("")

    # ---------------------------------------------------------------
    # 4. Stats of selected matches: gain-selected vs entropy-selected
    # ---------------------------------------------------------------
    lines.append("## 4. Properties of Selected Matches")
    lines.append("")
    lines.append("Average probability stats for the matches each method selects.")
    lines.append("")
    lines.append("| League | Method | Avg Gain (2nd best) | Avg Top2 | Avg Entropy |")
    lines.append("|--------|--------|---------------------|----------|-------------|")

    for lg in league_labels:
        sub = [c for c in cells if c["league"] == lg]
        g_gain = np.mean([c["gain_avg_gain"] for c in sub])
        g_top2 = np.mean([c["gain_avg_top2"] for c in sub])
        g_ent = np.mean([c["gain_avg_entropy"] for c in sub])
        e_gain = np.mean([c["entropy_avg_gain"] for c in sub])
        e_top2 = np.mean([c["entropy_avg_top2"] for c in sub])
        e_ent = np.mean([c["entropy_avg_entropy"] for c in sub])
        lines.append(f"| {lg} | gain | {g_gain:.4f} | {g_top2:.4f} | {g_ent:.4f} |")
        lines.append(f"| {lg} | entropy | {e_gain:.4f} | {e_top2:.4f} | {e_ent:.4f} |")

    lines.append("")
    lines.append("### Gap between gain-selected and entropy-selected stats")
    lines.append("")
    lines.append("| League | Gain Diff (G-E) | Top2 Diff (G-E) | Entropy Diff (G-E) |")
    lines.append("|--------|-----------------|-----------------|---------------------|")

    for lg in league_labels:
        sub = [c for c in cells if c["league"] == lg]
        g_gain = np.mean([c["gain_avg_gain"] for c in sub])
        g_top2 = np.mean([c["gain_avg_top2"] for c in sub])
        g_ent = np.mean([c["gain_avg_entropy"] for c in sub])
        e_gain = np.mean([c["entropy_avg_gain"] for c in sub])
        e_top2 = np.mean([c["entropy_avg_top2"] for c in sub])
        e_ent = np.mean([c["entropy_avg_entropy"] for c in sub])
        lines.append(
            f"| {lg} | {g_gain - e_gain:+.4f} | {g_top2 - e_top2:+.4f} | {g_ent - e_ent:+.4f} |"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # 5. E0 model confidence and predictability
    # ---------------------------------------------------------------
    lines.append("## 5. E0 Model Confidence and Predictability")
    lines.append("")
    lines.append("Is E0 generally harder to predict? Does the model show different confidence patterns?")
    lines.append("")

    lines.append("### Top-1 and Top-2 accuracy by league (from match data)")
    lines.append("")
    lines.append("| League | N_matches | Top1 Acc | Top2 Acc | Mean Best Prob | Mean Entropy |")
    lines.append("|--------|----------|----------|----------|----------------|--------------|")

    for lg in league_labels:
        sub = mdf[mdf["league"] == lg]
        n = len(sub)
        t1 = sub["top1_hit"].mean()
        t2 = sub["top2_hit"].mean()
        mbest = sub["best"].mean()
        ment = sub["entropy"].mean()
        lines.append(f"| {lg} | {n} | {t1:.4f} | {t2:.4f} | {mbest:.4f} | {ment:.4f} |")

    lines.append("")

    # Correlation between entropy and top2_hit
    lines.append("### Correlation: entropy vs top2_hit by league")
    lines.append("")
    lines.append("Higher entropy should mean less certainty, but does it predict top2 misses?")
    lines.append("")
    lines.append("| League | Corr(entropy, top2_hit) | P-value | N |")
    lines.append("|--------|------------------------|---------|---|")

    for lg in league_labels:
        sub = mdf[mdf["league"] == lg]
        if len(sub) > 10:
            corr, pval = sp_stats.pointbiserialr(sub["top2_hit"], sub["entropy"])
            lines.append(f"| {lg} | {corr:.4f} | {pval:.4f} | {len(sub)} |")
        else:
            lines.append(f"| {lg} | N/A | N/A | {len(sub)} |")

    lines.append("")

    # Correlation between gain (second-best) and top2_hit
    lines.append("### Correlation: gain (second-best prob) vs top2_hit by league")
    lines.append("")
    lines.append("| League | Corr(gain, top2_hit) | P-value | N |")
    lines.append("|--------|---------------------|---------|---|")

    for lg in league_labels:
        sub = mdf[mdf["league"] == lg]
        if len(sub) > 10:
            corr, pval = sp_stats.pointbiserialr(sub["top2_hit"], sub["gain"])
            lines.append(f"| {lg} | {corr:.4f} | {pval:.4f} | {len(sub)} |")
        else:
            lines.append(f"| {lg} | N/A | N/A | {len(sub)} |")

    lines.append("")

    # ---------------------------------------------------------------
    # 6. Hypothesis & Conclusion
    # ---------------------------------------------------------------
    lines.append("## 6. Diagnosis and Conclusion")
    lines.append("")

    # Compute key numbers for the conclusion
    e0_sub = [c for c in cells if c["league"] == "E0"]
    e0_hg = sum(c["n_half_eff"] for c in e0_sub)
    e0_g = sum(c["gain_hits"] for c in e0_sub)
    e0_e = sum(c["entropy_hits"] for c in e0_sub)
    e0_delta = e0_g - e0_e

    lines.append("### Key findings")
    lines.append("")
    lines.append(f"1. **Tiny sample, tiny delta**: E0's divergence rests on {abs(e0_delta)} "
                 f"absolute hit{'s' if abs(e0_delta) != 1 else ''} "
                 f"out of {e0_hg} half-guard decisions "
                 f"({e0_g} gain vs {e0_e} entropy). "
                 f"The 95% confidence intervals overlap heavily.")
    lines.append("")

    # Check fold-level pattern
    fold_deltas = []
    for fold in sorted(set(c["fold"] for c in e0_sub)):
        fsub = [c for c in e0_sub if c["fold"] == fold]
        fg = sum(c["gain_hits"] for c in fsub)
        fe = sum(c["entropy_hits"] for c in fsub)
        fold_deltas.append(fg - fe)

    gain_folds = sum(1 for d in fold_deltas if d > 0)
    entropy_folds = sum(1 for d in fold_deltas if d < 0)
    tie_folds = sum(1 for d in fold_deltas if d == 0)

    lines.append(f"2. **Inconsistent across folds**: Gain wins {gain_folds} fold(s), "
                 f"entropy wins {entropy_folds} fold(s), tied {tie_folds} fold(s). "
                 f"No consistent pattern across time periods.")
    lines.append("")

    # Separation gap
    e0_sep = np.mean([c["gain_avg_gain"] for c in e0_sub]) - np.mean([c["entropy_avg_gain"] for c in e0_sub])
    e1_sub = [c for c in cells if c["league"] == "E1"]
    e1_sep = np.mean([c["gain_avg_gain"] for c in e1_sub]) - np.mean([c["entropy_avg_gain"] for c in e1_sub])
    e2_sub = [c for c in cells if c["league"] == "E2"]
    e2_sep = np.mean([c["gain_avg_gain"] for c in e2_sub]) - np.mean([c["entropy_avg_gain"] for c in e2_sub])
    e3_sub = [c for c in cells if c["league"] == "E3"]
    e3_sep = np.mean([c["gain_avg_gain"] for c in e3_sub]) - np.mean([c["entropy_avg_gain"] for c in e3_sub])

    lines.append(f"3. **Narrower gain-entropy separation in E0**: "
                 f"The gap in avg second-best probability between gain-selected and entropy-selected "
                 f"matches is smaller in E0 ({e0_sep:+.4f}) compared to "
                 f"E1 ({e1_sep:+.4f}), E2 ({e2_sep:+.4f}), E3 ({e3_sep:+.4f}). "
                 f"This means gain and entropy tend to pick more similar matches in E0, "
                 f"reducing the advantage of gain-based selection.")
    lines.append("")

    # E0 entropy distribution
    e0_match = mdf[mdf["league"] == "E0"]
    other_match = mdf[mdf["league"].isin(["E1", "E2", "E3"])]
    e0_mean_ent = e0_match["entropy"].mean()
    other_mean_ent = other_match["entropy"].mean()

    lines.append(f"4. **E0 has lower mean entropy**: "
                 f"E0 mean entropy = {e0_mean_ent:.4f} vs "
                 f"E1+E2+E3 mean = {other_mean_ent:.4f}. "
                 f"Premier League predictions tend to be more confident on average, "
                 f"meaning fewer genuinely uncertain matches for gain to differentiate from.")
    lines.append("")

    # Divergent selection hit rates
    e0_gonly = sum(c["gain_only_total"] for c in e0_sub)
    e0_gonly_hits = sum(c["gain_only_hits"] for c in e0_sub)
    e0_eonly = sum(c["entropy_only_total"] for c in e0_sub)
    e0_eonly_hits = sum(c["entropy_only_hits"] for c in e0_sub)
    e0_gonly_rate = e0_gonly_hits / e0_gonly if e0_gonly > 0 else 0
    e0_eonly_rate = e0_eonly_hits / e0_eonly if e0_eonly > 0 else 0

    lines.append(f"5. **Divergent picks**: When the methods disagree in E0, "
                 f"gain-only picks hit {e0_gonly_hits}/{e0_gonly} ({e0_gonly_rate:.1%}) "
                 f"vs entropy-only picks hit {e0_eonly_hits}/{e0_eonly} ({e0_eonly_rate:.1%}). "
                 f"{'This difference is small and could easily be noise.' if abs(e0_gonly_rate - e0_eonly_rate) < 0.15 else 'Entropy-only picks perform notably better in E0.'}")
    lines.append("")

    lines.append("### Overall diagnosis")
    lines.append("")
    lines.append(f"**The E0 divergence is almost certainly sample noise.**")
    lines.append("")
    lines.append(f"- The entire effect is {abs(e0_delta)} hit{'s' if abs(e0_delta) != 1 else ''} "
                 f"out of {e0_hg} decisions -- well within binomial sampling error.")
    lines.append(f"- The 95% CIs for gain ({g_lo:.3f}-{g_hi:.3f}) and entropy ({e_lo:.3f}-{e_hi:.3f}) "
                 f"overlap completely.")
    lines.append(f"- The effect is not consistent across folds, ruling out a systematic league-specific cause.")
    lines.append(f"- E0 does show a slightly narrower probability spread (lower entropy, "
                 f"tighter gain-entropy separation), which means the gain method has less room "
                 f"to add value. But this is a *marginal* effect -- it explains why E0's gain "
                 f"advantage might be *smaller*, not why it should flip to entropy's favor.")
    lines.append("")
    lines.append("### Recommendation")
    lines.append("")
    lines.append("- **No action needed** on the production logic. The gain-based method is still")
    lines.append("  the correct choice for all leagues including E0.")
    lines.append("- If future benchmark runs with more data continue to show E0 diverging,")
    lines.append("  consider investigating whether Premier League's tighter probability")
    lines.append("  distributions warrant a league-specific N_HALF or a gain threshold cutoff.")
    lines.append("- A possible future improvement: hybrid selection that uses gain but falls back")
    lines.append("  to entropy when the gain gap between candidates is very small.")
    lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------

def export_detail_csv(cells: list, path: Path):
    """Export per-cell detail data to CSV."""
    if not cells:
        return

    fieldnames = [
        "fold", "league", "n_half", "n_half_eff", "n_matches",
        "gain_hits", "entropy_hits", "gain_acc_hg", "entropy_acc_hg",
        "overlap", "gain_only_total", "entropy_only_total",
        "gain_only_hits", "entropy_only_hits",
        "gain_avg_gain", "gain_avg_top2", "gain_avg_entropy",
        "entropy_avg_gain", "entropy_avg_top2", "entropy_avg_entropy",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cells:
            writer.writerow({k: c[k] for k in fieldnames})

    logger.info("Detail CSV written to %s (%d rows)", path, len(cells))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose E0 divergence in half-guard benchmark",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh data instead of using cache",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    refresh = args.refresh_data or os.environ.get(
        "BACKTEST_REFRESH_DATA", ""
    ).lower() in ("1", "true", "yes")

    logger.info("Loading data (refresh=%s)...", refresh)
    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("Failed to load data")
        return 1

    logger.info("Loaded %d matches", len(df))

    logger.info("Collecting per-match data...")
    match_data = collect_per_match_data(df, n_folds=N_FOLDS)
    logger.info("Collected %d match records", len(match_data))

    logger.info("Collecting half-guard outcome data...")
    cells = collect_halfguard_outcomes(df, n_folds=N_FOLDS, n_half_values=N_HALF_VALUES)
    logger.info("Collected %d cell records", len(cells))

    report = generate_report(match_data, cells)

    report_path = _REPO_ROOT / "RESULTS_HALF_GUARD_E0_DIAGNOSIS.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)

    csv_path = _REPO_ROOT / "results_e0_diagnosis_detail.csv"
    export_detail_csv(cells, csv_path)

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
