#!/usr/bin/env python3
"""
Diagnostic benchmark of the half-guard sign selection rule (X protection).

Production removes the least likely sign:

    least_likely = argmin(probs)

Because a fair draw probability rarely exceeds one third, X is the argmin on
almost every close match. This diagnostic measures one parameterised rule
family in which the current rule and "always keep X" are the endpoints:

    threshold_tau(p, tau):
        if p_X >= tau:  keep X, remove the less likely of {1, 2}
        else:           remove argmin(p)                # current behaviour

    tau = 0.00 -> X is never removed
    tau = 1.00 -> identical to the current argmin rule

Diagnostic only:
- no production rule, weight, default, flag, model artifact, or UI is changed
- streck is not used and the full production blend is not evaluated
- fold logic, frozen reference window, sample check and paired bootstrap are
  reused unchanged from scripts/diagnose_production_blend.py
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from combined_probability import CombinedMatchProbability
from scripts.diagnose_league_representation import validate_pr43_sample
from scripts.diagnose_production_blend import (
    DEFAULT_MAX_DATE,
    REFERENCE_CACHE_FILES,
    freeze_reference_window,
    load_reference_data,
    parse_max_date,
    probability_matrix,
    run_diagnostic,
)
from ui_utils import get_halfguard_sign_combined, pick_half_guards_combined

logger = logging.getLogger(__name__)

BASE_SHA = "6c89d944c8fe46c27b635bf84c3f09afa54332c7"
SIGNS = ("1", "X", "2")
SIGN_BY_REMOVED_INDEX = {0: "X2", 1: "12", 2: "1X"}
TAUS = (0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 1.00)
REFERENCE_TAU = 1.00
PROBABILITY_SOURCES = ("odds_only", "production_blend")
PRIMARY_SOURCE = "odds_only"
SELECTIONS = ("all", "top7of13")
PRIMARY_SELECTION = "all"
COUPON_SIZE = 13
COUPON_GUARDS = 7
BOOTSTRAP_QUANTILES = (0.025, 0.975)
MAX_REPORTED_MISMATCHES = 20
TEXT_COLUMNS = frozenset({"Source", "Selection", "League", "TauLabel"})
INTEGER_COLUMNS = frozenset(
    {
        "N",
        "N_X",
        "N_nonX",
        "GuardedMatches",
        "Matches_Checked",
        "Mismatches",
        "Coupons_N",
        "Correct",
        "Coupons_with_Correct",
    }
)

GUARDRAILS = (
    "Per-league confidence intervals are exploratory and are not adjusted "
    "for multiple comparisons. An isolated subgroup result must not override "
    "the overall paired comparison.",
    "The sweep evaluates multiple candidate thresholds on the same data. Its "
    "intervals are exploratory and are not adjusted for threshold selection. "
    "An optimal tau indicates direction, not a value that may be adopted "
    "without independent out-of-sample confirmation.",
    "Synthetic coupons are formed from 13 chronologically adjacent matches "
    "and do not reproduce the league mix or difficulty profile of real "
    "Stryktipset coupons.",
    "Streck is not used in this diagnostic and the full production blend is "
    "not evaluated. The question is limited to which sign the half-guard rule "
    "removes.",
)


def tau_label(tau: float) -> str:
    return f"{float(tau):.2f}"


def _fmt(value: object, digits: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def markdown_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    digits: int = 6,
) -> str:
    """Render a table with text, integer and float columns typed correctly."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in columns:
            if column in TEXT_COLUMNS:
                values.append(str(row[column]))
            elif column in INTEGER_COLUMNS:
                values.append(str(int(row[column])))
            else:
                values.append(_fmt(row[column], digits))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def removed_index_threshold(probs: Sequence[float], tau: float) -> int:
    """
    Index of the sign removed by the threshold rule.

    Ties resolve to the lowest index, matching numpy.argmin.
    """
    values = np.asarray(probs, dtype=float)
    if values.shape != (3,):
        raise ValueError("probs must contain exactly three probabilities")
    if not np.isfinite(values).all():
        raise ValueError("probs must be finite")

    if values[1] >= float(tau):
        return 0 if values[0] <= values[2] else 2
    return int(np.argmin(values))


def halfguard_sign_threshold(probs: Sequence[float], tau: float) -> str:
    """Two-sign half-guard string, e.g. `1X`, from the threshold rule."""
    removed = removed_index_threshold(probs, tau)
    return "".join(SIGNS[index] for index in range(3) if index != removed)


def removed_indices(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Per-row removed index for a full (n, 3) probability matrix."""
    probabilities = np.asarray(matrix, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("matrix must have shape (n, 3)")
    return np.array(
        [removed_index_threshold(row, tau) for row in probabilities],
        dtype=int,
    )


def combined_match(
    probs: Sequence[float],
    *,
    home_team: str = "",
    away_team: str = "",
) -> CombinedMatchProbability:
    """Wrap one probability triple in the production dataclass."""
    values = np.asarray(probs, dtype=float)
    return CombinedMatchProbability(
        home_team=home_team,
        away_team=away_team,
        prob_1=float(values[0]),
        prob_x=float(values[1]),
        prob_2=float(values[2]),
        entropy=0.0,
        sources={},
    )


def verify_reference_tau_identity(
    predictions: pd.DataFrame,
    *,
    sources: Sequence[str] = PROBABILITY_SOURCES,
) -> list[dict]:
    """
    Positionwise identity check of tau = 1.00 against production.

    The removed sign must be identical on every single match, not merely in
    aggregate. Any deviation aborts the run.
    """
    rows: list[dict] = []
    for source in sources:
        matrix = probability_matrix(predictions, source)
        mismatches: list[str] = []

        for position in range(len(matrix)):
            probs = matrix[position]
            expected = get_halfguard_sign_combined(
                combined_match(
                    probs,
                    home_team=str(predictions["HomeTeam"].iloc[position]),
                    away_team=str(predictions["AwayTeam"].iloc[position]),
                )
            )
            actual = halfguard_sign_threshold(probs, REFERENCE_TAU)
            if actual != expected:
                mismatches.append(
                    f"row {position} "
                    f"({predictions['Date'].iloc[position]} "
                    f"{predictions['HomeTeam'].iloc[position]}-"
                    f"{predictions['AwayTeam'].iloc[position]}, "
                    f"p=[{probs[0]:.6f}, {probs[1]:.6f}, {probs[2]:.6f}]): "
                    f"production={expected}, tau=1.00={actual}"
                )

        if mismatches:
            shown = mismatches[:MAX_REPORTED_MISMATCHES]
            suffix = (
                f"\n- ... and {len(mismatches) - len(shown)} more mismatches"
                if len(mismatches) > len(shown)
                else ""
            )
            raise ValueError(
                f"tau=1.00 is not identical to get_halfguard_sign_combined() "
                f"for source {source}; {len(mismatches)} mismatching matches:"
                "\n- " + "\n- ".join(shown) + suffix
            )

        rows.append(
            {
                "Source": source,
                "Matches_Checked": int(len(matrix)),
                "Mismatches": 0,
            }
        )
        logger.info(
            "Identity check passed for %s: %d matches, 0 mismatches",
            source,
            len(matrix),
        )
    return rows


def assign_coupons(predictions: pd.DataFrame) -> pd.Series:
    """
    Label complete synthetic coupons of 13 chronologically adjacent matches.

    Rows in a trailing incomplete block are labelled with an empty string and
    excluded from coupon aggregation. This is an approximation of a real
    Stryktipset coupon.
    """
    labels = pd.Series([""] * len(predictions), index=predictions.index)
    for fold_idx, group in predictions.groupby("fold", sort=True):
        positions = np.arange(len(group))
        blocks = positions // COUPON_SIZE
        complete = np.bincount(blocks) == COUPON_SIZE
        for position, block in zip(group.index, blocks):
            if complete[block]:
                labels.loc[position] = f"f{int(fold_idx)}c{int(block):03d}"
    return labels


def guard_selection_mask(
    predictions: pd.DataFrame,
    matrix: np.ndarray,
    selection: str,
    coupons: pd.Series,
) -> np.ndarray:
    """
    Which matches the half-guard rule is applied to.

    `all` isolates the rule question from the selection question. `top7of13`
    reuses the production selection heuristic
    `ui_utils.pick_half_guards_combined()` inside each synthetic coupon.
    """
    if selection == "all":
        return np.ones(len(predictions), dtype=bool)
    if selection != "top7of13":
        raise ValueError(f"Unknown selection: {selection}")

    mask = np.zeros(len(predictions), dtype=bool)
    coupon_values = coupons.to_numpy()
    for coupon in pd.unique(coupon_values):
        if coupon == "":
            continue
        positions = np.flatnonzero(coupon_values == coupon)
        matches = [combined_match(matrix[position]) for position in positions]
        chosen = pick_half_guards_combined(matches, COUPON_GUARDS)
        mask[positions[np.array(sorted(chosen), dtype=int)]] = True
    return mask


def hit_vector(
    y_true: np.ndarray,
    matrix: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Whether the realised outcome survived the removal, per match."""
    return removed_indices(matrix, tau) != np.asarray(y_true, dtype=int)


def _safe_rate(hits: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    return float(hits[mask].mean())


def rule_metrics(
    y_true: np.ndarray,
    matrix: np.ndarray,
    tau: float,
) -> dict[str, float | int]:
    """All requested descriptive metrics for one tau on one group."""
    outcomes = np.asarray(y_true, dtype=int)
    removed = removed_indices(matrix, tau)
    hits = removed != outcomes
    is_x = outcomes == 1

    signs = np.array(
        [SIGN_BY_REMOVED_INDEX[int(index)] for index in removed],
        dtype=object,
    )
    n = int(len(outcomes))
    hit_rate = float(hits.mean()) if n else float("nan")

    return {
        "Tau": float(tau),
        "N": n,
        "HitRate": hit_rate,
        "N_X": int(is_x.sum()),
        "HitRate_X": _safe_rate(hits, is_x),
        "N_nonX": int((~is_x).sum()),
        "HitRate_nonX": _safe_rate(hits, ~is_x),
        "Share_1X": float(np.mean(signs == "1X")) if n else float("nan"),
        "Share_12": float(np.mean(signs == "12")) if n else float("nan"),
        "Share_X2": float(np.mean(signs == "X2")) if n else float("nan"),
        "X_removed_rate": float(np.mean(signs == "12")) if n else float("nan"),
        "X_actual_rate": float(is_x.mean()) if n else float("nan"),
        "ExpectedCorrect_per13": hit_rate * COUPON_SIZE,
    }


def paired_ci_lookup(
    y_true: np.ndarray,
    matrix: np.ndarray,
    *,
    taus: Sequence[float] = TAUS,
    reference_tau: float = REFERENCE_TAU,
    n_bootstrap: int,
    seed: int,
) -> dict[tuple[str, float], tuple[float, float]]:
    """
    Paired bootstrap over matches, shared indices across every tau.

    Returns 95% intervals for delta hit rate versus the reference tau, for the
    overall, outcome-was-X and outcome-was-not-X subsets.
    """
    outcomes = np.asarray(y_true, dtype=int)
    taus = tuple(taus)
    reference_index = taus.index(float(reference_tau))
    metrics = ("HitRate", "HitRate_X", "HitRate_nonX")

    lookup: dict[tuple[str, float], tuple[float, float]] = {}
    if n_bootstrap <= 0 or len(outcomes) < 2:
        for metric in metrics:
            for tau in taus:
                lookup[(metric, float(tau))] = (
                    (0.0, 0.0)
                    if float(tau) == float(reference_tau)
                    else (float("nan"), float("nan"))
                )
        return lookup

    hits = np.column_stack(
        [hit_vector(outcomes, matrix, tau).astype(float) for tau in taus]
    )
    is_x = (outcomes == 1).astype(float)

    rng = np.random.default_rng(seed)
    n = len(outcomes)
    remaining = int(n_bootstrap)
    chunk_size = max(1, min(100, remaining))
    collected: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}

    while remaining:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(count, n))
        sampled = hits[indices]
        sampled_x = is_x[indices]

        collected["HitRate"].append(sampled.mean(axis=1))

        for metric, weights in (
            ("HitRate_X", sampled_x),
            ("HitRate_nonX", 1.0 - sampled_x),
        ):
            denominator = weights.sum(axis=1)
            numerator = np.einsum("ijk,ij->ik", sampled, weights)
            with np.errstate(invalid="ignore", divide="ignore"):
                rates = numerator / denominator[:, None]
            rates[denominator == 0, :] = np.nan
            collected[metric].append(rates)

        remaining -= count

    for metric in metrics:
        means = np.concatenate(collected[metric], axis=0)
        for tau_index, tau in enumerate(taus):
            if float(tau) == float(reference_tau):
                lookup[(metric, float(tau))] = (0.0, 0.0)
                continue
            delta = means[:, tau_index] - means[:, reference_index]
            delta = delta[np.isfinite(delta)]
            if delta.size == 0:
                lookup[(metric, float(tau))] = (float("nan"), float("nan"))
                continue
            lookup[(metric, float(tau))] = tuple(
                float(value)
                for value in np.quantile(delta, BOOTSTRAP_QUANTILES)
            )

    return lookup


def summarize_group(
    y_true: np.ndarray,
    matrix: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    """Metrics plus paired deltas and intervals versus the reference tau."""
    reference = rule_metrics(y_true, matrix, REFERENCE_TAU)
    ci_lookup = paired_ci_lookup(
        y_true,
        matrix,
        n_bootstrap=bootstrap,
        seed=seed,
    )

    rows: list[dict] = []
    for tau in TAUS:
        metrics = rule_metrics(y_true, matrix, tau)
        row = dict(metrics)
        row["TauLabel"] = tau_label(tau)
        row["IsCurrentRule"] = bool(float(tau) == REFERENCE_TAU)
        for metric in ("HitRate", "HitRate_X", "HitRate_nonX"):
            lower, upper = ci_lookup[(metric, float(tau))]
            row[f"Delta_{metric}"] = float(metrics[metric] - reference[metric])
            row[f"Delta_{metric}_CI95_L"] = lower
            row[f"Delta_{metric}_CI95_U"] = upper
        rows.append(row)
    return rows


def build_metric_tables(
    predictions: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Overall, per-league and per-coupon tables for every source/selection."""
    coupons = assign_coupons(predictions)
    outcomes = predictions["y_true"].to_numpy(dtype=int)
    leagues = predictions["League"].astype(str).to_numpy()

    overall_rows: list[dict] = []
    league_rows: list[dict] = []
    coupon_rows: list[dict] = []

    for source_index, source in enumerate(PROBABILITY_SOURCES):
        matrix = probability_matrix(predictions, source)
        for selection_index, selection in enumerate(SELECTIONS):
            guarded = guard_selection_mask(
                predictions,
                matrix,
                selection,
                coupons,
            )
            offset = (source_index * 10 + selection_index) * 1_000_003
            metadata = {
                "Source": source,
                "Selection": selection,
                "GuardedMatches": int(guarded.sum()),
            }

            for row in summarize_group(
                outcomes[guarded],
                matrix[guarded],
                bootstrap=bootstrap,
                seed=seed + offset,
            ):
                overall_rows.append({**metadata, "League": "ALL", **row})

            for league_index, league in enumerate(sorted(set(leagues))):
                league_mask = guarded & (leagues == league)
                if not league_mask.any():
                    continue
                for row in summarize_group(
                    outcomes[league_mask],
                    matrix[league_mask],
                    bootstrap=bootstrap,
                    seed=seed + offset + (league_index + 1) * 7_001,
                ):
                    league_rows.append(
                        {
                            **metadata,
                            "GuardedMatches": int(league_mask.sum()),
                            "League": league,
                            **row,
                        }
                    )

            coupon_rows.extend(
                coupon_summary_rows(
                    predictions,
                    matrix,
                    outcomes,
                    coupons,
                    guarded,
                    source=source,
                    selection=selection,
                )
            )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(league_rows),
        pd.DataFrame(coupon_rows),
    )


def coupon_correct_counts(
    matrix: np.ndarray,
    outcomes: np.ndarray,
    coupons: pd.Series,
    guarded: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Correct signs per synthetic 13-match coupon.

    Guarded matches count as correct when the outcome survived the removal.
    Unguarded matches are scored as single signs on argmax(p), which is
    independent of tau.
    """
    hits = hit_vector(outcomes, matrix, tau)
    singles = np.argmax(matrix, axis=1) == outcomes
    correct = np.where(guarded, hits, singles)

    coupon_values = coupons.to_numpy()
    counts: list[int] = []
    for coupon in pd.unique(coupon_values):
        if coupon == "":
            continue
        positions = coupon_values == coupon
        counts.append(int(correct[positions].sum()))
    return np.array(counts, dtype=int)


def coupon_summary_rows(
    predictions: pd.DataFrame,
    matrix: np.ndarray,
    outcomes: np.ndarray,
    coupons: pd.Series,
    guarded: np.ndarray,
    *,
    source: str,
    selection: str,
) -> list[dict]:
    """Mean correct plus the full distribution per tau."""
    rows: list[dict] = []
    for tau in TAUS:
        counts = coupon_correct_counts(
            matrix,
            outcomes,
            coupons,
            guarded,
            tau,
        )
        total = int(len(counts))
        mean_correct = float(counts.mean()) if total else float("nan")
        for value in range(COUPON_SIZE + 1):
            occurrences = int((counts == value).sum())
            rows.append(
                {
                    "Source": source,
                    "Selection": selection,
                    "Tau": float(tau),
                    "TauLabel": tau_label(tau),
                    "IsCurrentRule": bool(float(tau) == REFERENCE_TAU),
                    "Coupons_N": total,
                    "MeanCorrect": mean_correct,
                    "MedianCorrect": (
                        float(np.median(counts)) if total else float("nan")
                    ),
                    "Correct": value,
                    "Coupons_with_Correct": occurrences,
                    "Share": (
                        float(occurrences / total) if total else float("nan")
                    ),
                }
            )
    return rows


OVERALL_COLUMNS = (
    "Source",
    "Selection",
    "League",
    "TauLabel",
    "N",
    "HitRate",
    "HitRate_X",
    "HitRate_nonX",
    "N_X",
    "N_nonX",
    "Share_1X",
    "Share_12",
    "Share_X2",
    "X_removed_rate",
    "X_actual_rate",
    "ExpectedCorrect_per13",
    "Delta_HitRate",
    "Delta_HitRate_CI95_L",
    "Delta_HitRate_CI95_U",
    "Delta_HitRate_X",
    "Delta_HitRate_X_CI95_L",
    "Delta_HitRate_X_CI95_U",
    "Delta_HitRate_nonX",
    "Delta_HitRate_nonX_CI95_L",
    "Delta_HitRate_nonX_CI95_U",
)

DISTRIBUTION_COLUMNS = (
    "Source",
    "Selection",
    "TauLabel",
    "Share_1X",
    "Share_12",
    "Share_X2",
    "X_removed_rate",
)

COUPON_COLUMNS = (
    "Source",
    "Selection",
    "TauLabel",
    "Coupons_N",
    "MeanCorrect",
    "MedianCorrect",
)


def evidence_summary(overall: pd.DataFrame) -> list[str]:
    """Neutral, direction-explicit reading of the overall paired comparison."""
    lines: list[str] = []
    for source in PROBABILITY_SOURCES:
        for selection in SELECTIONS:
            subset = overall[
                (overall["Source"] == source)
                & (overall["Selection"] == selection)
                & (overall["League"] == "ALL")
            ]
            if subset.empty:
                continue
            candidates = subset[~subset["IsCurrentRule"]]
            supported = candidates[
                (candidates["Delta_HitRate"] > 0)
                & (candidates["Delta_HitRate_CI95_L"] > 0)
            ]
            best = candidates.sort_values(
                ["Delta_HitRate", "Tau"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]

            if supported.empty:
                verdict = (
                    "no tau raises the total hit rate with a paired 95% CI "
                    "excluding 0; the current rule is not beaten"
                )
            else:
                winner = supported.sort_values(
                    ["Delta_HitRate", "Tau"],
                    ascending=[False, True],
                    kind="mergesort",
                ).iloc[0]
                verdict = (
                    f"tau={tau_label(winner['Tau'])} raises the total hit "
                    f"rate by {winner['Delta_HitRate']:.6f} with a paired "
                    "95% CI above 0"
                )

            lines.append(
                f"- **{source} / {selection}**: {verdict}. Best point "
                f"estimate is tau={tau_label(best['Tau'])} "
                f"(ΔHitRate={best['Delta_HitRate']:.6f}, CI95 "
                f"[{best['Delta_HitRate_CI95_L']:.6f}, "
                f"{best['Delta_HitRate_CI95_U']:.6f}], "
                f"ΔHitRate_X={best['Delta_HitRate_X']:.6f}, "
                f"ΔHitRate_nonX={best['Delta_HitRate_nonX']:.6f})."
            )
    return lines


def render_report(
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    league: pd.DataFrame,
    coupon: pd.DataFrame,
    identity: list[dict],
    *,
    bootstrap: int,
    strict_sample: bool,
) -> str:
    date_min = pd.to_datetime(predictions["Date"]).min().date()
    date_max = pd.to_datetime(predictions["Date"]).max().date()
    max_date = str(predictions["max_date"].iloc[0])
    source_max_date = str(predictions["source_max_date"].iloc[0])
    source_row_n = int(predictions["source_row_N"].iloc[0])
    frozen_row_n = int(predictions["frozen_row_N"].iloc[0])
    excluded_n = int(predictions["excluded_post_max_date_N"].iloc[0])

    coupon_summary = (
        coupon.drop_duplicates(subset=["Source", "Selection", "TauLabel"])
        if not coupon.empty
        else coupon
    )

    lines = [
        "# Diagnostic: half-guard sign selection rule (X protection)",
        "",
        "## Question",
        "",
        "> Can the half-guard rule be improved by protecting X from being "
        "removed too easily, without lowering the total hit rate?",
        "",
        "## Scope",
        "",
        f"- Base SHA: `{BASE_SHA}`.",
        "- This is a pure rule comparison. Streck is not simulated and no "
        "full production blend is built.",
        "- The rule family is "
        "`threshold_tau(p, tau)`: keep X and remove the less likely of "
        "{1, 2} when `p_X >= tau`, otherwise remove `argmin(p)`.",
        "- `tau` values evaluated: "
        + ", ".join(f"`{tau_label(tau)}`" for tau in TAUS)
        + ".",
        "- `tau = 0.00` never removes X. `tau = 1.00` is the current "
        "`argmin` rule and is the paired reference for every delta.",
        "- Negative delta means worse than the current rule.",
        f"- Primary probability source: `{PRIMARY_SOURCE}` (fair implied "
        "probabilities). Robustness source: `production_blend` "
        "(odds+model, no streck, as in PR #45).",
        f"- Primary selection: `{PRIMARY_SELECTION}` (the rule is applied to "
        "every match, isolating the rule question from the selection "
        f"question). Secondary selection: `top7of13` "
        f"({COUPON_GUARDS} of {COUPON_SIZE} most uncertain matches per "
        "synthetic coupon via `ui_utils.pick_half_guards_combined()`).",
        "- No production rule, weight, default, flag or UI is changed, and no "
        "tau is activated.",
        "",
        "## Sample",
        "",
        "- Fold logic, frozen reference window, sample check and paired "
        "bootstrap are reused unchanged from "
        "`scripts/diagnose_production_blend.py`.",
        "- Frozen reference input, in the exact order passed to "
        "`normalize_csv_data()`: "
        + ", ".join(f"`{name}`" for name in REFERENCE_CACHE_FILES)
        + ".",
        f"- `--max-date {max_date}` is the frozen evaluation window.",
        f"- Refreshed source rows before freeze: {source_row_n}.",
        f"- Refreshed source max date before freeze: {source_max_date}.",
        f"- Frozen rows before fold construction: {frozen_row_n}.",
        f"- Rows after the cutoff excluded before fold construction: "
        f"{excluded_n}.",
        f"- Paired rows: {len(predictions)}.",
        f"- Evaluation date range: {date_min} to {date_max}.",
        f"- Strict PR #43 sample parity enforced: {strict_sample}.",
        f"- Bootstrap resamples: {bootstrap}.",
        "- `League` is kept at constant `-1` in the model, matching current "
        "production.",
        "",
        "## Identity check for tau = 1.00",
        "",
        "The check is positionwise: the removed sign must be identical to "
        "`ui_utils.get_halfguard_sign_combined()` on every single match. Any "
        "deviation aborts the run.",
        "",
        markdown_table(
            pd.DataFrame(identity),
            ["Source", "Matches_Checked", "Mismatches"],
        ),
        "",
        "## Overall by tau",
        "",
        markdown_table(
            overall,
            list(OVERALL_COLUMNS),
            digits=6,
        ),
        "",
        "## Half-guard type distribution by tau",
        "",
        markdown_table(
            overall[overall["League"] == "ALL"],
            list(DISTRIBUTION_COLUMNS),
            digits=6,
        ),
        "",
        "## Per league by tau",
        "",
        markdown_table(
            league,
            list(OVERALL_COLUMNS),
            digits=6,
        ),
        "",
        "## Synthetic coupons",
        "",
        markdown_table(
            coupon_summary,
            list(COUPON_COLUMNS),
            digits=6,
        ),
        "",
        "The full coupon distribution is persisted in "
        "`RESULTS_HALFGUARD_RULE_COUPON.csv`.",
        "",
        "## Evidence summary",
        "",
        *evidence_summary(overall),
        "",
        "- No tau is declared a winner if the total hit rate falls or if the "
        "relevant confidence interval overlaps 0. Trading X misses for 1/2 "
        "misses is not an improvement.",
        "",
        "## Interpretation guardrails",
        "",
        *[f"- {text}" for text in GUARDRAILS],
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    report: str,
    overall: pd.DataFrame,
    league: pd.DataFrame,
    coupon: pd.DataFrame,
    *,
    report_path: Path,
) -> tuple[Path, ...]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    outputs = (
        report_path,
        report_path.with_name(report_path.stem + "_OVERALL.csv"),
        report_path.with_name(report_path.stem + "_LEAGUE.csv"),
        report_path.with_name(report_path.stem + "_COUPON.csv"),
    )
    overall.to_csv(outputs[1], index=False)
    league.to_csv(outputs[2], index=False)
    coupon.to_csv(outputs[3], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark half-guard sign selection thresholds"
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--allow-sample-drift",
        action="store_true",
        help="Allow a sample different from the committed PR #43 sample",
    )
    parser.add_argument(
        "--max-date",
        type=parse_max_date,
        default=parse_max_date(DEFAULT_MAX_DATE),
    )
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RESULTS_HALFGUARD_RULE.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    refresh = args.refresh_data or os.environ.get(
        "BACKTEST_REFRESH_DATA",
        "",
    ).lower() in ("1", "true", "yes")

    df = load_reference_data(refresh=refresh)
    if df.empty:
        logger.error("No data loaded")
        return 1

    frozen_df, freeze_metadata = freeze_reference_window(
        df,
        max_date=args.max_date,
    )
    predictions = run_diagnostic(
        frozen_df,
        n_segments=args.segments,
        freeze_metadata=freeze_metadata,
    )
    if not args.allow_sample_drift:
        validate_pr43_sample(predictions)

    identity = verify_reference_tau_identity(predictions)
    overall, league, coupon = build_metric_tables(
        predictions,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report = render_report(
        predictions,
        overall,
        league,
        coupon,
        identity,
        bootstrap=args.bootstrap,
        strict_sample=not args.allow_sample_drift,
    )
    outputs = save_outputs(
        report,
        overall,
        league,
        coupon,
        report_path=args.output,
    )

    logger.info("Half-guard rule diagnostic complete")
    for path in outputs:
        logger.info("Wrote %s", path)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
