#!/usr/bin/env python3
"""
Diagnostic benchmark of the production odds/model blend versus odds-only.

Historical streck data is unavailable. Therefore this diagnostic measures the
real production combination path when streck is absent:

    default weights 0.50 odds / 0.35 model / 0.15 streck
    -> streck omitted by combine_probabilities()
    -> effective 0.50 / 0.85 odds and 0.35 / 0.85 model

The script also sweeps the effective model weight from 0.00 to 1.00 in steps
of 0.05, plus the exact no-streck production weight.

Diagnostic only:
- no production default, weight, flag, model artifact, or UI is changed
- streck values are never fabricated or simulated
- the real combined_probability.combine_probabilities() function is used
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backtest_report import load_data, train_model
from combined_probability import DEFAULT_WEIGHTS, combine_probabilities
from schema import CLASS_MAP, FEATURE_COLUMNS
from scripts.diagnose_league_representation import (
    X_CALIBRATION_BINS,
    apply_league_representation,
    canonicalize_history,
    metric_row,
    validate_pr43_sample,
)
from scripts.diagnose_model_vs_odds import (
    date_safe_walk_forward_folds,
    normalized_odds_probs,
    prepare_features,
    valid_odds_mask,
)

logger = logging.getLogger(__name__)

BASE_SHA = "75e3c3a011c6294f6727b49fd70b63d54750dc46"
DEFAULT_MAX_DATE = "2026-05-24"
CORE_VARIANTS = ("odds_only", "model_only", "production_blend")
EPS = 1e-15
BOOTSTRAP_QUANTILES = (0.025, 0.975)


def effective_production_weights() -> tuple[float, float]:
    """Return effective (odds, model) weights when streck is unavailable."""
    odds_weight = float(DEFAULT_WEIGHTS["odds"])
    model_weight = float(DEFAULT_WEIGHTS["model"])
    total = odds_weight + model_weight
    if total <= 0:
        raise ValueError("Odds/model production weights must have positive total")
    return odds_weight / total, model_weight / total


PRODUCTION_ODDS_WEIGHT, PRODUCTION_MODEL_WEIGHT = effective_production_weights()


def sweep_weights() -> tuple[float, ...]:
    """0.00..1.00 by 0.05 plus the exact production model weight."""
    values = {round(index / 20.0, 12) for index in range(21)}
    values.add(round(PRODUCTION_MODEL_WEIGHT, 12))
    return tuple(sorted(values))


SWEEP_WEIGHTS = sweep_weights()


def weight_slug(model_weight: float) -> str:
    return f"{float(model_weight):.12f}".replace(".", "p")


def sweep_prefix(model_weight: float) -> str:
    return f"sweep_w_{weight_slug(model_weight)}"


def sweep_probability_columns(model_weight: float) -> list[str]:
    prefix = sweep_prefix(model_weight)
    return [f"{prefix}_{sign}" for sign in ("H", "D", "A")]


def core_probability_columns(label: str) -> list[str]:
    mapping = {
        "odds_only": "odds",
        "model_only": "model",
        "production_blend": "production",
    }
    if label not in mapping:
        raise ValueError(f"Unknown core variant: {label}")
    return [f"{mapping[label]}_{sign}" for sign in ("H", "D", "A")]


def parse_max_date(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse the frozen benchmark window end date."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise argparse.ArgumentTypeError(
            f"Invalid --max-date value: {value!r}; expected YYYY-MM-DD"
        )
    return pd.Timestamp(parsed).normalize()


def freeze_reference_window(
    df: pd.DataFrame,
    *,
    max_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Freeze the entire refreshed dataset before any fold construction.

    Fold boundaries are derived from the set of unique dates. Therefore late
    rows must be removed before date_safe_walk_forward_folds() sees the data;
    filtering only evaluation rows after folds are built would change every
    boundary and break comparability with PR #43/#44.
    """
    if df.empty:
        raise ValueError("Cannot freeze an empty benchmark dataset")

    cutoff = parse_max_date(max_date)
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    if work["Date"].isna().any():
        raise ValueError("Benchmark dataset contains invalid dates")

    source_max_date = pd.Timestamp(work["Date"].max())
    source_rows = int(len(work))
    frozen = (
        work.loc[work["Date"] <= cutoff]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if frozen.empty:
        raise ValueError(
            f"No benchmark rows remain at or before max-date {cutoff.date()}"
        )
    if pd.Timestamp(frozen["Date"].max()) > cutoff:
        raise ValueError("Reference-window cutoff was not applied correctly")

    metadata = {
        "max_date": str(cutoff.date()),
        "source_max_date": str(source_max_date.date()),
        "source_row_N": source_rows,
        "frozen_row_N": int(len(frozen)),
        "excluded_post_max_date_N": int(source_rows - len(frozen)),
    }
    return frozen, metadata

def combine_rows(
    odds_probs: np.ndarray,
    model_probs: np.ndarray,
    *,
    weights: dict[str, float] | None,
) -> np.ndarray:
    """
    Combine paired rows through the real production function.

    streck_pcts is deliberately always None. This is both the historical-data
    limitation and the exact production path for a match without streck.
    """
    odds = np.asarray(odds_probs, dtype=float)
    model = np.asarray(model_probs, dtype=float)

    if odds.shape != model.shape or odds.ndim != 2 or odds.shape[1] != 3:
        raise ValueError(
            "odds_probs and model_probs must have identical shape (n, 3)"
        )

    combined = np.empty_like(odds, dtype=float)
    for index in range(len(odds)):
        combined[index] = combine_probabilities(
            odds_probs=odds[index],
            model_probs=model[index],
            streck_pcts=None,
            weights=weights,
        )

    if not np.isfinite(combined).all() or (combined < 0).any():
        raise ValueError("Combined probabilities contain invalid values")
    if not np.allclose(combined.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Combined probabilities do not sum to 1")
    return combined


def production_blend_rows(
    odds_probs: np.ndarray,
    model_probs: np.ndarray,
) -> np.ndarray:
    """Use DEFAULT_WEIGHTS and let production renormalise missing streck."""
    return combine_rows(odds_probs, model_probs, weights=None)


def sweep_blend_rows(
    odds_probs: np.ndarray,
    model_probs: np.ndarray,
    model_weight: float,
) -> np.ndarray:
    """Use the real combiner with effective two-source sweep weights."""
    weight = float(model_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("model_weight must be between 0 and 1")

    return combine_rows(
        odds_probs,
        model_probs,
        weights={
            "odds": 1.0 - weight,
            "model": weight,
            "streck": 0.0,
        },
    )


def _fit_base_model_probabilities(
    df_train: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> np.ndarray:
    """
    Train the actual production feature set.

    League is kept as a constant -1 column, matching current production.
    """
    train_rep = apply_league_representation(df_train, "league_none")
    eval_rep = apply_league_representation(eval_df, "league_none")

    model = train_model(
        train_rep,
        feature_columns=list(FEATURE_COLUMNS),
    )
    if model is None:
        raise RuntimeError("Base model training failed")

    x_eval = prepare_features(
        eval_rep,
        feature_columns=list(FEATURE_COLUMNS),
    )
    probabilities = np.asarray(model.predict_proba(x_eval), dtype=float)

    if probabilities.shape != (len(eval_df), 3):
        raise ValueError(
            f"Expected model probability shape {(len(eval_df), 3)}, "
            f"got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model probabilities contain invalid values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Model probabilities do not sum to 1")
    return probabilities


def prediction_frame_for_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    fold_idx: int,
) -> pd.DataFrame:
    """Build all core and sweep probability sources on one paired test fold."""
    mask = valid_odds_mask(df_test)
    eval_df = df_test.loc[mask].copy()
    if eval_df.empty:
        logger.warning("Fold %d has no valid-odds evaluation rows", fold_idx)
        return pd.DataFrame()

    y_true = eval_df["FTR"].map(CLASS_MAP)
    if y_true.isna().any():
        raise ValueError(f"Fold {fold_idx}: unknown result label")

    odds_probs = normalized_odds_probs(eval_df)
    model_probs = _fit_base_model_probabilities(df_train, eval_df)
    production_probs = production_blend_rows(odds_probs, model_probs)

    output = pd.DataFrame(
        {
            "Date": pd.to_datetime(eval_df["Date"]).to_numpy(),
            "League": eval_df["League"].astype(str).to_numpy(),
            "Season": eval_df.get(
                "Season",
                pd.Series(["UNK"] * len(eval_df), index=eval_df.index),
            )
            .astype(str)
            .to_numpy(),
            "HomeTeam": eval_df.get(
                "HomeTeam",
                pd.Series([""] * len(eval_df), index=eval_df.index),
            ).to_numpy(),
            "AwayTeam": eval_df.get(
                "AwayTeam",
                pd.Series([""] * len(eval_df), index=eval_df.index),
            ).to_numpy(),
            "FTR": eval_df["FTR"].astype(str).to_numpy(),
            "y_true": y_true.to_numpy(dtype=int),
            "fold": int(fold_idx),
            "train_N": int(len(df_train)),
            "test_N": int(len(df_test)),
            "paired_N": int(len(eval_df)),
        }
    )

    for prefix, probabilities in (
        ("odds", odds_probs),
        ("model", model_probs),
        ("production", production_probs),
    ):
        for class_index, sign in enumerate(("H", "D", "A")):
            output[f"{prefix}_{sign}"] = probabilities[:, class_index]

    for model_weight in SWEEP_WEIGHTS:
        probabilities = sweep_blend_rows(
            odds_probs,
            model_probs,
            model_weight,
        )
        prefix = sweep_prefix(model_weight)
        for class_index, sign in enumerate(("H", "D", "A")):
            output[f"{prefix}_{sign}"] = probabilities[:, class_index]

    return output


def run_diagnostic(
    df: pd.DataFrame,
    *,
    n_segments: int = 4,
    freeze_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run date-safe expanding-window evaluation on an already-frozen dataset."""
    work = canonicalize_history(df)

    if freeze_metadata is None:
        raise ValueError(
            "freeze_metadata is required; freeze_reference_window() must run "
            "immediately after load_data() before run_diagnostic()"
        )

    cutoff = parse_max_date(str(freeze_metadata["max_date"]))
    if pd.to_datetime(work["Date"]).max() > cutoff:
        raise ValueError(
            "run_diagnostic received rows after max-date; "
            "freeze_reference_window() must run before fold construction"
        )

    logger.info(
        "Frozen reference window ending %s: source_rows=%d, frozen_rows=%d, "
        "excluded_post_max_date=%d, source_max_date=%s",
        freeze_metadata["max_date"],
        freeze_metadata["source_row_N"],
        freeze_metadata["frozen_row_N"],
        freeze_metadata["excluded_post_max_date_N"],
        freeze_metadata["source_max_date"],
    )
    folds = date_safe_walk_forward_folds(work, n_segments=n_segments)
    if not folds:
        raise ValueError("No walk-forward folds could be created")

    frames: list[pd.DataFrame] = []
    for fold_idx, train_mask, test_mask in folds:
        df_train = work.loc[train_mask].copy()
        df_test = work.loc[test_mask].copy()

        if len(df_train) < 100:
            logger.warning(
                "Fold %d skipped because train_N=%d is too small",
                fold_idx,
                len(df_train),
            )
            continue

        if df_train["Date"].max() >= df_test["Date"].min():
            raise ValueError(f"Fold {fold_idx}: temporal leakage detected")

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

        frame = prediction_frame_for_fold(
            df_train,
            df_test,
            fold_idx=fold_idx,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No paired predictions were produced")

    predictions = pd.concat(frames, ignore_index=True)
    for key, value in freeze_metadata.items():
        predictions[key] = value
    validate_prediction_frame(predictions)
    return predictions


def probability_matrix(group: pd.DataFrame, label: str) -> np.ndarray:
    if label in CORE_VARIANTS:
        columns = core_probability_columns(label)
    elif label.startswith("sweep:"):
        model_weight = float(label.split(":", 1)[1])
        columns = sweep_probability_columns(model_weight)
    else:
        raise ValueError(f"Unknown probability source: {label}")
    return group[columns].to_numpy(dtype=float)


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    """Validate complete pairing and the two critical blend identity anchors."""
    required: list[str] = []
    for label in CORE_VARIANTS:
        required.extend(core_probability_columns(label))
    for model_weight in SWEEP_WEIGHTS:
        required.extend(sweep_probability_columns(model_weight))

    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction frame missing columns: {missing}")
    if predictions[required].isna().any().any():
        raise ValueError("Prediction frame has missing probabilities")

    freeze_columns = (
        "max_date",
        "source_max_date",
        "source_row_N",
        "frozen_row_N",
        "excluded_post_max_date_N",
    )
    missing_freeze = [
        column for column in freeze_columns
        if column not in predictions.columns
    ]
    if missing_freeze:
        raise ValueError(
            f"Prediction frame missing sample-freeze metadata: {missing_freeze}"
        )
    for column in freeze_columns:
        if predictions[column].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent sample-freeze metadata: {column}")

    cutoff = parse_max_date(str(predictions["max_date"].iloc[0]))
    if pd.to_datetime(predictions["Date"]).max() > cutoff:
        raise ValueError("Predictions contain rows after the frozen max-date")

    for fold_idx, group in predictions.groupby("fold", sort=True):
        for column in ("train_N", "test_N", "paired_N"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Fold {fold_idx}: inconsistent {column}")
        if int(group["paired_N"].iloc[0]) != len(group):
            raise ValueError(f"Fold {fold_idx}: paired_N does not match row count")

    odds = probability_matrix(predictions, "odds_only")
    sweep_zero = probability_matrix(predictions, "sweep:0.0")
    if not np.allclose(odds, sweep_zero, atol=1e-12, rtol=0.0):
        maximum = float(np.max(np.abs(odds - sweep_zero)))
        raise ValueError(
            f"w=0 sweep is not identical to odds_only; max_abs_diff={maximum}"
        )

    production = probability_matrix(predictions, "production_blend")
    production_sweep = probability_matrix(
        predictions,
        f"sweep:{PRODUCTION_MODEL_WEIGHT}",
    )
    if not np.allclose(production, production_sweep, atol=1e-12, rtol=0.0):
        maximum = float(np.max(np.abs(production - production_sweep)))
        raise ValueError(
            "Exact production-weight sweep differs from production_blend; "
            f"max_abs_diff={maximum}"
        )


def per_match_logloss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    true_probability = y_proba[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(true_probability, EPS, 1.0))


def per_match_brier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    onehot = np.eye(3, dtype=float)[y_true]
    return np.sum((y_proba - onehot) ** 2, axis=1)


def paired_ci_lookup(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    comparisons: Iterable[tuple[str, str]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Shared paired bootstrap indices for every comparison in one group."""
    comparisons = tuple(dict.fromkeys(comparisons))
    labels = tuple(probabilities)
    label_index = {label: index for index, label in enumerate(labels)}

    for candidate, reference in comparisons:
        if candidate not in probabilities or reference not in probabilities:
            raise ValueError(
                f"Unknown bootstrap comparison: {candidate} vs {reference}"
            )

    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}

    if n_bootstrap <= 0 or len(y_true) < 2:
        for candidate, reference in comparisons:
            value = (0.0, 0.0) if candidate == reference else (
                float("nan"),
                float("nan"),
            )
            lookup[("logloss", candidate, reference)] = value
            lookup[("brier", candidate, reference)] = value
        return lookup

    logloss_matrix = np.column_stack(
        [per_match_logloss(y_true, probabilities[label]) for label in labels]
    )
    brier_matrix = np.column_stack(
        [per_match_brier(y_true, probabilities[label]) for label in labels]
    )

    rng = np.random.default_rng(seed)
    n = len(y_true)
    remaining = int(n_bootstrap)
    chunk_size = max(1, min(100, remaining))
    bootstrap_logloss: list[np.ndarray] = []
    bootstrap_brier: list[np.ndarray] = []

    while remaining:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(count, n))
        bootstrap_logloss.append(logloss_matrix[indices].mean(axis=1))
        bootstrap_brier.append(brier_matrix[indices].mean(axis=1))
        remaining -= count

    logloss_means = np.concatenate(bootstrap_logloss, axis=0)
    brier_means = np.concatenate(bootstrap_brier, axis=0)

    for candidate, reference in comparisons:
        if candidate == reference:
            lookup[("logloss", candidate, reference)] = (0.0, 0.0)
            lookup[("brier", candidate, reference)] = (0.0, 0.0)
            continue

        candidate_index = label_index[candidate]
        reference_index = label_index[reference]

        logloss_delta = (
            logloss_means[:, candidate_index]
            - logloss_means[:, reference_index]
        )
        brier_delta = (
            brier_means[:, candidate_index]
            - brier_means[:, reference_index]
        )

        lookup[("logloss", candidate, reference)] = tuple(
            float(value)
            for value in np.quantile(logloss_delta, BOOTSTRAP_QUANTILES)
        )
        lookup[("brier", candidate, reference)] = tuple(
            float(value)
            for value in np.quantile(brier_delta, BOOTSTRAP_QUANTILES)
        )

    return lookup


def summarize_core_group(
    group: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    y_true = group["y_true"].to_numpy(dtype=int)
    probabilities = {
        label: probability_matrix(group, label)
        for label in CORE_VARIANTS
    }
    metrics = {
        label: metric_row(y_true, probabilities[label])
        for label in CORE_VARIANTS
    }
    comparisons = (
        ("model_only", "odds_only"),
        ("production_blend", "odds_only"),
    )
    ci_lookup = paired_ci_lookup(
        y_true,
        probabilities,
        comparisons,
        n_bootstrap=bootstrap,
        seed=seed,
    )

    weights = {
        "odds_only": (1.0, 0.0),
        "model_only": (0.0, 1.0),
        "production_blend": (
            PRODUCTION_ODDS_WEIGHT,
            PRODUCTION_MODEL_WEIGHT,
        ),
    }

    rows: list[dict] = []
    for label in CORE_VARIANTS:
        odds_weight, model_weight = weights[label]
        if label == "odds_only":
            delta_logloss = 0.0
            delta_brier = 0.0
            logloss_ci = (0.0, 0.0)
            brier_ci = (0.0, 0.0)
        else:
            delta_logloss = float(
                metrics[label]["LogLoss"] - metrics["odds_only"]["LogLoss"]
            )
            delta_brier = float(
                metrics[label]["Brier"] - metrics["odds_only"]["Brier"]
            )
            logloss_ci = ci_lookup[("logloss", label, "odds_only")]
            brier_ci = ci_lookup[("brier", label, "odds_only")]

        rows.append(
            {
                "Variant": label,
                "N": int(metrics[label]["N"]),
                "OddsWeight": float(odds_weight),
                "ModelWeight": float(model_weight),
                "StreckMeasured": False,
                **metrics[label],
                "Delta_LogLoss_vs_Odds": delta_logloss,
                "Delta_LogLoss_vs_Odds_CI95_L": logloss_ci[0],
                "Delta_LogLoss_vs_Odds_CI95_U": logloss_ci[1],
                "Delta_Brier_vs_Odds": delta_brier,
                "Delta_Brier_vs_Odds_CI95_L": brier_ci[0],
                "Delta_Brier_vs_Odds_CI95_U": brier_ci[1],
            }
        )

    return rows


def build_core_group_table(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    if group_columns:
        iterator = predictions.groupby(
            list(group_columns),
            sort=True,
            dropna=False,
        )
    else:
        iterator = [((), predictions)]

    for group_index, (group_key, group) in enumerate(iterator):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = dict(zip(group_columns, group_key))

        for row in summarize_core_group(
            group,
            bootstrap=bootstrap,
            seed=seed + group_index * 100_003,
        ):
            rows.append({**metadata, **row})

    return pd.DataFrame(rows)


def build_fold_table(
    predictions: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold_idx, group in predictions.groupby("fold", sort=True):
        metadata = {
            "Fold": int(fold_idx),
            "Train_N": int(group["train_N"].iloc[0]),
            "Test_N": int(group["test_N"].iloc[0]),
            "Paired_N": int(group["paired_N"].iloc[0]),
        }
        for row in summarize_core_group(
            group,
            bootstrap=bootstrap,
            seed=seed + int(fold_idx) * 70_001,
        ):
            rows.append({**metadata, **row})
    return pd.DataFrame(rows)


def sweep_metrics_for_group(
    group: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    y_true = group["y_true"].to_numpy(dtype=int)
    probabilities = {
        "odds_only": probability_matrix(group, "odds_only"),
    }
    for model_weight in SWEEP_WEIGHTS:
        label = f"sweep:{model_weight}"
        probabilities[label] = probability_matrix(group, label)

    metrics = {
        label: metric_row(y_true, proba)
        for label, proba in probabilities.items()
    }
    comparisons = tuple(
        (f"sweep:{model_weight}", "odds_only")
        for model_weight in SWEEP_WEIGHTS
    )
    ci_lookup = paired_ci_lookup(
        y_true,
        probabilities,
        comparisons,
        n_bootstrap=bootstrap,
        seed=seed,
    )

    rows: list[dict] = []
    for model_weight in SWEEP_WEIGHTS:
        label = f"sweep:{model_weight}"
        logloss_ci = ci_lookup[("logloss", label, "odds_only")]
        brier_ci = ci_lookup[("brier", label, "odds_only")]
        rows.append(
            {
                "ModelWeight": float(model_weight),
                "OddsWeight": float(1.0 - model_weight),
                "IsGridWeight": bool(
                    any(
                        abs(model_weight - index / 20.0) <= 1e-12
                        for index in range(21)
                    )
                ),
                "IsProductionWeight": bool(
                    abs(model_weight - PRODUCTION_MODEL_WEIGHT) <= 1e-12
                ),
                **metrics[label],
                "Delta_LogLoss_vs_Odds": float(
                    metrics[label]["LogLoss"]
                    - metrics["odds_only"]["LogLoss"]
                ),
                "Delta_LogLoss_vs_Odds_CI95_L": logloss_ci[0],
                "Delta_LogLoss_vs_Odds_CI95_U": logloss_ci[1],
                "Delta_Brier_vs_Odds": float(
                    metrics[label]["Brier"]
                    - metrics["odds_only"]["Brier"]
                ),
                "Delta_Brier_vs_Odds_CI95_L": brier_ci[0],
                "Delta_Brier_vs_Odds_CI95_U": brier_ci[1],
            }
        )
    return rows


def select_optimal_weight(overall_sweep: pd.DataFrame) -> float:
    """Minimum overall LogLoss; ties resolve to the lower model weight."""
    if overall_sweep.empty:
        raise ValueError("Cannot select w* from an empty sweep")
    ordered = overall_sweep.sort_values(
        ["LogLoss", "ModelWeight"],
        ascending=[True, True],
        kind="mergesort",
    )
    return float(ordered.iloc[0]["ModelWeight"])


def build_weight_sweep_table(
    predictions: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    overall_rows = sweep_metrics_for_group(
        predictions,
        bootstrap=bootstrap,
        seed=seed,
    )
    overall = pd.DataFrame(overall_rows)
    optimum_weight = select_optimal_weight(overall)

    rows: list[dict] = []
    for row in overall_rows:
        rows.append(
            {
                "Scope": "overall",
                "League": "ALL",
                **row,
                "IsOverallOptimum": bool(
                    abs(row["ModelWeight"] - optimum_weight) <= 1e-12
                ),
            }
        )

    for league_index, (league, group) in enumerate(
        predictions.groupby("League", sort=True)
    ):
        league_rows = sweep_metrics_for_group(
            group,
            bootstrap=bootstrap,
            seed=seed + (league_index + 1) * 200_003,
        )
        for row in league_rows:
            rows.append(
                {
                    "Scope": "league",
                    "League": str(league),
                    **row,
                    "IsOverallOptimum": bool(
                        abs(row["ModelWeight"] - optimum_weight) <= 1e-12
                    ),
                }
            )

    table = pd.DataFrame(rows)
    return table, optimum_weight


def build_overall_decision_table(
    core_overall: pd.DataFrame,
    weight_sweep: pd.DataFrame,
    optimum_weight: float,
) -> pd.DataFrame:
    """Persist the three core variants plus the data-selected w* row."""
    rows = core_overall.to_dict(orient="records")
    optimum = weight_sweep[
        (weight_sweep["Scope"] == "overall")
        & np.isclose(
            weight_sweep["ModelWeight"],
            optimum_weight,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    if len(optimum) != 1:
        raise ValueError("Expected exactly one overall w* row")

    source = optimum.iloc[0]
    rows.append(
        {
            "Variant": "sweep_w_star",
            "N": int(source["N"]),
            "OddsWeight": float(source["OddsWeight"]),
            "ModelWeight": float(source["ModelWeight"]),
            "StreckMeasured": False,
            "Accuracy": float(source["Accuracy"]),
            "LogLoss": float(source["LogLoss"]),
            "Brier": float(source["Brier"]),
            "X_top2_rate": float(source["X_top2_rate"]),
            "X_mean_prob": float(source["X_mean_prob"]),
            "X_actual_rate": float(source["X_actual_rate"]),
            "X_brier": float(source["X_brier"]),
            "Delta_LogLoss_vs_Odds": float(
                source["Delta_LogLoss_vs_Odds"]
            ),
            "Delta_LogLoss_vs_Odds_CI95_L": float(
                source["Delta_LogLoss_vs_Odds_CI95_L"]
            ),
            "Delta_LogLoss_vs_Odds_CI95_U": float(
                source["Delta_LogLoss_vs_Odds_CI95_U"]
            ),
            "Delta_Brier_vs_Odds": float(
                source["Delta_Brier_vs_Odds"]
            ),
            "Delta_Brier_vs_Odds_CI95_L": float(
                source["Delta_Brier_vs_Odds_CI95_L"]
            ),
            "Delta_Brier_vs_Odds_CI95_U": float(
                source["Delta_Brier_vs_Odds_CI95_U"]
            ),
        }
    )
    return pd.DataFrame(rows)


def selected_calibration_sources(
    predictions: pd.DataFrame,
    optimum_weight: float,
) -> dict[str, tuple[np.ndarray, float]]:
    """Core variants plus sweep anchors requested by the diagnostic."""
    return {
        "odds_only": (
            probability_matrix(predictions, "odds_only"),
            0.0,
        ),
        "model_only": (
            probability_matrix(predictions, "model_only"),
            1.0,
        ),
        "production_blend": (
            probability_matrix(predictions, "production_blend"),
            PRODUCTION_MODEL_WEIGHT,
        ),
        "sweep_w_0": (
            probability_matrix(predictions, "sweep:0.0"),
            0.0,
        ),
        "sweep_w_production": (
            probability_matrix(
                predictions,
                f"sweep:{PRODUCTION_MODEL_WEIGHT}",
            ),
            PRODUCTION_MODEL_WEIGHT,
        ),
        "sweep_w_star": (
            probability_matrix(predictions, f"sweep:{optimum_weight}"),
            optimum_weight,
        ),
    }


def build_x_calibration_table(
    predictions: pd.DataFrame,
    optimum_weight: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = [("overall", "ALL", predictions)]
    scopes.extend(
        ("league", str(league), group)
        for league, group in predictions.groupby("League", sort=True)
    )

    for scope, league, group in scopes:
        observed = (group["y_true"].to_numpy(dtype=int) == 1).astype(float)

        source_specs = {
            "odds_only": (probability_matrix(group, "odds_only"), 0.0),
            "model_only": (probability_matrix(group, "model_only"), 1.0),
            "production_blend": (
                probability_matrix(group, "production_blend"),
                PRODUCTION_MODEL_WEIGHT,
            ),
            "sweep_w_0": (probability_matrix(group, "sweep:0.0"), 0.0),
            "sweep_w_production": (
                probability_matrix(
                    group,
                    f"sweep:{PRODUCTION_MODEL_WEIGHT}",
                ),
                PRODUCTION_MODEL_WEIGHT,
            ),
            "sweep_w_star": (
                probability_matrix(group, f"sweep:{optimum_weight}"),
                optimum_weight,
            ),
        }

        for source, (probabilities, model_weight) in source_specs.items():
            x_probability = probabilities[:, 1]
            for lower, upper, bin_label in X_CALIBRATION_BINS:
                if upper > 1.0:
                    mask = (
                        (x_probability >= lower)
                        & (x_probability <= 1.0)
                    )
                else:
                    mask = (
                        (x_probability >= lower)
                        & (x_probability < upper)
                    )

                count = int(mask.sum())
                rows.append(
                    {
                        "Scope": scope,
                        "League": league,
                        "Source": source,
                        "ModelWeight": float(model_weight),
                        "Bin": bin_label,
                        "N": count,
                        "MeanPredictedX": (
                            float(x_probability[mask].mean())
                            if count
                            else float("nan")
                        ),
                        "ObservedXRate": (
                            float(observed[mask].mean())
                            if count
                            else float("nan")
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
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
    digits: int = 4,
) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]

    integer_columns = {"Fold", "Train_N", "Test_N", "Paired_N", "N"}
    text_columns = {
        "Variant",
        "Scope",
        "League",
        "Season",
        "Source",
        "Bin",
    }

    for _, row in df.iterrows():
        values: list[str] = []
        for column in columns:
            if column in integer_columns:
                values.append(str(int(row[column])))
            elif column in text_columns:
                values.append(str(row[column]))
            else:
                values.append(_fmt(row[column], digits))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def evidence_text(row: pd.Series) -> str:
    delta_ll = float(row["Delta_LogLoss_vs_Odds"])
    ll_lo = float(row["Delta_LogLoss_vs_Odds_CI95_L"])
    ll_hi = float(row["Delta_LogLoss_vs_Odds_CI95_U"])
    delta_br = float(row["Delta_Brier_vs_Odds"])
    br_lo = float(row["Delta_Brier_vs_Odds_CI95_L"])
    br_hi = float(row["Delta_Brier_vs_Odds_CI95_U"])

    if not np.isfinite([delta_ll, ll_lo, ll_hi, delta_br, br_lo, br_hi]).all():
        return "confidence interval not computed"

    if delta_ll < 0 and delta_br < 0:
        if ll_hi < 0 and br_hi < 0:
            return "better on both metrics; paired 95% CIs below 0"
        return "better point estimates; uncertainty overlaps 0"

    if delta_ll > 0 and delta_br > 0:
        if ll_lo > 0 and br_lo > 0:
            return "worse on both metrics; paired 95% CIs above 0"
        return "worse point estimates; uncertainty overlaps 0"

    return "mixed LogLoss/Brier result"


def render_report(
    predictions: pd.DataFrame,
    overall_decision: pd.DataFrame,
    fold_table: pd.DataFrame,
    league_table: pd.DataFrame,
    season_table: pd.DataFrame,
    weight_sweep: pd.DataFrame,
    calibration_table: pd.DataFrame,
    *,
    optimum_weight: float,
    bootstrap: int,
    strict_sample: bool,
) -> str:
    date_min = pd.to_datetime(predictions["Date"]).min().date()
    date_max = pd.to_datetime(predictions["Date"]).max().date()
    max_date = str(predictions["max_date"].iloc[0])
    source_max_date = str(predictions["source_max_date"].iloc[0])
    source_row_n = int(predictions["source_row_N"].iloc[0])
    frozen_row_n = int(predictions["frozen_row_N"].iloc[0])
    excluded_post_max_date_n = int(
        predictions["excluded_post_max_date_N"].iloc[0]
    )

    decision_columns = [
        "Variant",
        "N",
        "OddsWeight",
        "ModelWeight",
        "Accuracy",
        "LogLoss",
        "Brier",
        "X_top2_rate",
        "X_mean_prob",
        "X_actual_rate",
        "X_brier",
        "Delta_LogLoss_vs_Odds",
        "Delta_LogLoss_vs_Odds_CI95_L",
        "Delta_LogLoss_vs_Odds_CI95_U",
        "Delta_Brier_vs_Odds",
        "Delta_Brier_vs_Odds_CI95_L",
        "Delta_Brier_vs_Odds_CI95_U",
    ]
    group_columns = decision_columns[:-6] + decision_columns[-6:]
    fold_columns = ["Fold", "Train_N", "Test_N", "Paired_N"] + group_columns
    league_columns = ["League"] + group_columns
    season_columns = ["Season", "League"] + group_columns
    sweep_columns = [
        "Scope",
        "League",
        "ModelWeight",
        "OddsWeight",
        "IsGridWeight",
        "IsProductionWeight",
        "IsOverallOptimum",
        "N",
        "LogLoss",
        "Brier",
        "X_top2_rate",
        "X_mean_prob",
        "X_actual_rate",
        "X_brier",
        "Delta_LogLoss_vs_Odds",
        "Delta_LogLoss_vs_Odds_CI95_L",
        "Delta_LogLoss_vs_Odds_CI95_U",
        "Delta_Brier_vs_Odds",
        "Delta_Brier_vs_Odds_CI95_L",
        "Delta_Brier_vs_Odds_CI95_U",
    ]
    calibration_columns = [
        "Source",
        "ModelWeight",
        "Bin",
        "N",
        "MeanPredictedX",
        "ObservedXRate",
    ]

    production_row = overall_decision[
        overall_decision["Variant"] == "production_blend"
    ].iloc[0]
    model_row = overall_decision[
        overall_decision["Variant"] == "model_only"
    ].iloc[0]
    optimum_row = overall_decision[
        overall_decision["Variant"] == "sweep_w_star"
    ].iloc[0]

    overall_sweep = weight_sweep[weight_sweep["Scope"] == "overall"]
    league_sweep = weight_sweep[weight_sweep["Scope"] == "league"]
    overall_calibration = calibration_table[
        calibration_table["Scope"] == "overall"
    ]

    lines = [
        "# Diagnostic: production odds/model blend versus odds-only",
        "",
        "## Scope and historical-data limitation",
        "",
        f"- Base SHA: `{BASE_SHA}`.",
        "- Historical streck data is unavailable in the model/backtest dataset.",
        "- No streck values are fabricated or simulated.",
        "- The full 50/35/15 three-source production blend is therefore not "
        "backtested here.",
        "- This diagnostic measures the real production path when streck is absent: "
        f"effective odds weight `{PRODUCTION_ODDS_WEIGHT:.12f}` and model weight "
        f"`{PRODUCTION_MODEL_WEIGHT:.12f}`.",
        "- `production_blend` calls the real "
        "`combined_probability.combine_probabilities()` with odds and model "
        "present and streck absent.",
        "- The no-streck result does not establish how real streck would affect "
        "the full three-source blend or a comparison against odds+streck.",
        "- No production weight, default, model artifact, or flag is changed.",
        "",
        "## Sample",
        "",
        f"- Refreshed source rows before freeze: {source_row_n}.",
        f"- Refreshed source max date before freeze: {source_max_date}.",
        f"- Frozen reference-window max-date: {max_date}.",
        f"- Frozen rows before fold construction: {frozen_row_n}.",
        f"- Rows after the cutoff excluded before fold construction: "
        f"{excluded_post_max_date_n}.",
        "- The cutoff is applied to the entire canonicalised history before "
        "date-safe fold boundaries are calculated.",
        f"- Paired rows: {len(predictions)}.",
        f"- Evaluation date range: {date_min} to {date_max}.",
        f"- Strict PR #43 sample parity enforced: {strict_sample}.",
        f"- Bootstrap resamples: {bootstrap}.",
        "- `League` is kept at constant `-1`, matching current production.",
        "- Negative delta means the candidate is better than odds-only.",
        "",
        "## Overall decision table",
        "",
        markdown_table(
            overall_decision[decision_columns],
            decision_columns,
            digits=6,
        ),
        "",
        "## Overall evidence summary",
        "",
        f"- **production_blend vs odds_only**: {evidence_text(production_row)} "
        f"(ΔLogLoss={_fmt(production_row['Delta_LogLoss_vs_Odds'], 6)}, "
        f"ΔBrier={_fmt(production_row['Delta_Brier_vs_Odds'], 6)}).",
        f"- **model_only vs odds_only**: {evidence_text(model_row)} "
        f"(ΔLogLoss={_fmt(model_row['Delta_LogLoss_vs_Odds'], 6)}, "
        f"ΔBrier={_fmt(model_row['Delta_Brier_vs_Odds'], 6)}).",
        f"- **w*={optimum_weight:.12f} vs odds_only**: "
        f"{evidence_text(optimum_row)} "
        f"(ΔLogLoss={_fmt(optimum_row['Delta_LogLoss_vs_Odds'], 6)}, "
        f"ΔBrier={_fmt(optimum_row['Delta_Brier_vs_Odds'], 6)}).",
        "",
        "## Per-fold core variants",
        "",
        markdown_table(fold_table[fold_columns], fold_columns, digits=6),
        "",
        "## Per-league core variants",
        "",
        markdown_table(
            league_table[league_columns],
            league_columns,
            digits=6,
        ),
        "",
        "## Per-league and season core variants",
        "",
        markdown_table(
            season_table[season_columns],
            season_columns,
            digits=6,
        ),
        "",
        "## Weight sweep — overall",
        "",
        markdown_table(
            overall_sweep[sweep_columns],
            sweep_columns,
            digits=6,
        ),
        "",
        "## Weight sweep — per league",
        "",
        markdown_table(
            league_sweep[sweep_columns],
            sweep_columns,
            digits=6,
        ),
        "",
        "## Overall p_X calibration",
        "",
        markdown_table(
            overall_calibration[calibration_columns],
            calibration_columns,
            digits=6,
        ),
        "",
        "Per-league p_X calibration is persisted in "
        "`RESULTS_PRODUCTION_BLEND_X_CALIBRATION.csv`.",
        "",
        "## Interpretation guardrails",
        "",
        "- This is a frozen parity benchmark whose default max-date is "
        f"`{DEFAULT_MAX_DATE}`; it is not an evaluation of the newly started 2026/27 season.",
        "- LogLoss and multiclass Brier are the primary probability-quality "
        "metrics.",
        "- `w=0` is numerically checked against `odds_only`; the exact production "
        "weight is numerically checked against `production_blend`.",
        "- Per-league and per-season confidence intervals are exploratory and "
        "are not adjusted for multiple comparisons. An isolated subgroup "
        "result must not override the overall paired comparison.",
        "- The streck component of the production blend is not measured here; "
        "streck is unavailable in the historical dataset. Results describe the "
        "odds/model part of the blend with weights renormalised over the two "
        "available sources.",
        "- The absence of a benefit without streck cannot by itself prove the "
        "behaviour of the full three-source blend, because the realised streck "
        "distribution is unobserved.",
        "- w* is selected on the same data it is evaluated on and is therefore "
        "optimistic. It indicates the direction of a possible improvement, not "
        "a value that can be adopted as a production weight without independent "
        "out-of-sample confirmation.",
        "- The sweep evaluates many candidate weights. Its confidence intervals "
        "are exploratory and are not adjusted for selection or multiple "
        "comparisons.",
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    report: str,
    overall_decision: pd.DataFrame,
    fold_table: pd.DataFrame,
    league_table: pd.DataFrame,
    season_table: pd.DataFrame,
    weight_sweep: pd.DataFrame,
    calibration_table: pd.DataFrame,
    *,
    report_path: Path,
) -> tuple[Path, ...]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    outputs = (
        report_path,
        report_path.with_name(report_path.stem + "_OVERALL.csv"),
        report_path.with_name(report_path.stem + "_FOLD.csv"),
        report_path.with_name(report_path.stem + "_LEAGUE.csv"),
        report_path.with_name(report_path.stem + "_SEASON.csv"),
        report_path.with_name(report_path.stem + "_WEIGHT_SWEEP.csv"),
        report_path.with_name(report_path.stem + "_X_CALIBRATION.csv"),
    )
    overall_decision.to_csv(outputs[1], index=False)
    fold_table.to_csv(outputs[2], index=False)
    league_table.to_csv(outputs[3], index=False)
    season_table.to_csv(outputs[4], index=False)
    weight_sweep.to_csv(outputs[5], index=False)
    calibration_table.to_csv(outputs[6], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark no-streck production blend versus odds-only"
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help=(
            "Download/cache historical data before running. Exact PR #43 "
            "sample parity remains enforced unless --allow-sample-drift is set."
        ),
    )
    parser.add_argument(
        "--allow-sample-drift",
        action="store_true",
        help="Allow a sample different from the committed PR #43 sample",
    )
    parser.add_argument(
        "--max-date",
        type=parse_max_date,
        default=parse_max_date(DEFAULT_MAX_DATE),
        help=(
            "Freeze the full refreshed dataset at this date before fold "
            "construction (default: 2026-05-24)"
        ),
    )
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RESULTS_PRODUCTION_BLEND.md"),
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

    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("No data loaded")
        return 1

    # IMPORTANT: fold boundaries depend on the set of unique dates. Freeze the
    # entire refreshed dataset immediately after loading and before any fold
    # construction so later-season matches cannot shift PR #43/#44 boundaries.
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

    core_overall = build_core_group_table(
        predictions,
        [],
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    fold_table = build_fold_table(
        predictions,
        bootstrap=args.bootstrap,
        seed=args.seed + 10_000,
    )
    league_table = build_core_group_table(
        predictions,
        ["League"],
        bootstrap=args.bootstrap,
        seed=args.seed + 20_000,
    )
    season_table = build_core_group_table(
        predictions,
        ["League", "Season"],
        bootstrap=args.bootstrap,
        seed=args.seed + 30_000,
    )
    weight_sweep, optimum_weight = build_weight_sweep_table(
        predictions,
        bootstrap=args.bootstrap,
        seed=args.seed + 40_000,
    )
    overall_decision = build_overall_decision_table(
        core_overall,
        weight_sweep,
        optimum_weight,
    )
    calibration_table = build_x_calibration_table(
        predictions,
        optimum_weight,
    )

    report = render_report(
        predictions,
        overall_decision,
        fold_table,
        league_table,
        season_table,
        weight_sweep,
        calibration_table,
        optimum_weight=optimum_weight,
        bootstrap=args.bootstrap,
        strict_sample=not args.allow_sample_drift,
    )
    outputs = save_outputs(
        report,
        overall_decision,
        fold_table,
        league_table,
        season_table,
        weight_sweep,
        calibration_table,
        report_path=args.output,
    )

    logger.info("Production-blend diagnostic complete")
    for path in outputs:
        logger.info("Wrote %s", path)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
