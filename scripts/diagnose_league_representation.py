#!/usr/bin/env python3
"""
Diagnostic benchmark of League-feature representations.

Compares three League representations on the exact PR #43 walk-forward sample:

- league_none:    League is the constant -1 (current production behaviour)
- league_ordinal: League is encoded E0->0, E1->1, E2->2, E3->3
- league_onehot:  League is replaced by four fixed binary columns

Each representation is trained separately for:
- base features (FEATURE_COLUMNS)
- base + historical odds features (ALL_FEATURE_COLUMNS)

The historical bookmaker probabilities remain the common reference baseline.

Diagnostic only:
- no production defaults are changed
- encode_league() is not changed
- no model artifact is promoted
- no UI, weighting, streck, scanner, Elo, Poisson, or GPT logic is touched
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backtest_report import load_data, train_model
from schema import ALL_FEATURE_COLUMNS, CLASS_MAP, FEATURE_COLUMNS
from scripts.diagnose_model_vs_odds import (
    date_safe_walk_forward_folds,
    decode_league,
    normalized_odds_probs,
    prepare_features,
    valid_odds_mask,
)

logger = logging.getLogger(__name__)

BASE_SHA = "63bd1c471156a789ae7f7c3a3775d765485c65e7"
LEAGUES = ("E0", "E1", "E2", "E3")
REPRESENTATIONS = ("league_none", "league_ordinal", "league_onehot")
FEATURE_SETS = ("base", "with_odds")
ONEHOT_COLUMNS = tuple(f"League_{league}" for league in LEAGUES)
EPS = 1e-15

# Exact sample produced by PR #43 on the committed cache and base SHA above.
EXPECTED_PR43_SAMPLE = {
    "paired_rows": 2978,
    "date_min": "2025-01-16",
    "date_max": "2026-05-24",
    "folds": {
        1: {"train_N": 1094, "test_N": 942, "paired_N": 942},
        2: {"train_N": 2036, "test_N": 1055, "paired_N": 1055},
        3: {"train_N": 3091, "test_N": 981, "paired_N": 981},
    },
}

X_CALIBRATION_BINS = (
    (0.00, 0.20, "0.00-0.20"),
    (0.20, 0.25, "0.20-0.25"),
    (0.25, 0.30, "0.25-0.30"),
    (0.30, 1.01, "0.30+"),
)


@dataclass(frozen=True)
class VariantSpec:
    feature_set: str
    representation: str

    @property
    def label(self) -> str:
        return f"{self.feature_set}/{self.representation}"

    @property
    def prefix(self) -> str:
        return self.label.replace("/", "__")


MODEL_VARIANTS = tuple(
    VariantSpec(feature_set, representation)
    for feature_set in FEATURE_SETS
    for representation in REPRESENTATIONS
)
VARIANT_BY_LABEL = {spec.label: spec for spec in MODEL_VARIANTS}
ALL_VARIANT_LABELS = ("Odds",) + tuple(spec.label for spec in MODEL_VARIANTS)


def canonicalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the shared historical frame without choosing a representation."""
    if df.empty:
        raise ValueError("Empty dataset")

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = (
        work.dropna(subset=["Date", "FTR", "League"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    work["League"] = work["League"].apply(decode_league)
    work = work[work["League"].isin(LEAGUES)].reset_index(drop=True)

    if work.empty:
        raise ValueError("No E0-E3 rows remain after League normalization")
    return work


def feature_columns_for(feature_set: str, representation: str) -> list[str]:
    """Return the exact ordered model feature columns for one variant."""
    if feature_set == "base":
        columns = list(FEATURE_COLUMNS)
    elif feature_set == "with_odds":
        columns = list(ALL_FEATURE_COLUMNS)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown League representation: {representation}")

    if representation != "league_onehot":
        return columns

    if "League" not in columns:
        raise ValueError("Expected League in project feature columns")

    league_index = columns.index("League")
    return columns[:league_index] + list(ONEHOT_COLUMNS) + columns[league_index + 1 :]


def apply_league_representation(
    df: pd.DataFrame,
    representation: str,
) -> pd.DataFrame:
    """
    Transform a model-input copy while preserving the caller's metadata frame.

    league_none deliberately keeps the League column and sets it to -1.
    It is not dropped, because the purpose is exact current-production parity.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown League representation: {representation}")

    out = df.copy()
    canonical = out["League"].apply(decode_league)
    invalid = ~canonical.isin(LEAGUES)
    if invalid.any():
        bad = sorted(set(out.loc[invalid, "League"].astype(str)))
        raise ValueError(f"Invalid League values for representation: {bad}")

    if representation == "league_none":
        out["League"] = -1.0

    elif representation == "league_ordinal":
        # train_model()/prepare_features() encode these strings to 0..3.
        out["League"] = canonical

    else:
        # Keep canonical League only as metadata; it is excluded from X.
        out["League"] = canonical
        for league, column in zip(LEAGUES, ONEHOT_COLUMNS):
            out[column] = (canonical == league).astype(float)

    validate_representation_frame(out, representation)
    return out


def validate_representation_frame(
    df: pd.DataFrame,
    representation: str,
) -> None:
    """Fail fast if a representation is not exactly what the benchmark claims."""
    if representation == "league_none":
        values = pd.to_numeric(df["League"], errors="coerce")
        if values.isna().any() or not np.all(values.to_numpy(dtype=float) == -1.0):
            raise ValueError("league_none must keep League as a constant -1 column")
        return

    if representation == "league_ordinal":
        if not df["League"].isin(LEAGUES).all():
            raise ValueError("league_ordinal must contain canonical E0-E3 values")
        return

    if representation == "league_onehot":
        missing = [column for column in ONEHOT_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"league_onehot missing fixed columns: {missing}")

        values = df[list(ONEHOT_COLUMNS)].to_numpy(dtype=float)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError("league_onehot columns must be binary")
        if not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError("league_onehot rows must contain exactly one active league")
        return

    raise ValueError(f"Unknown League representation: {representation}")


def _fit_predict_variant(
    df_train: pd.DataFrame,
    eval_df: pd.DataFrame,
    spec: VariantSpec,
) -> np.ndarray:
    """Train one independently fitted fold model and predict the paired sample."""
    train_rep = apply_league_representation(df_train, spec.representation)
    eval_rep = apply_league_representation(eval_df, spec.representation)
    feature_columns = feature_columns_for(spec.feature_set, spec.representation)

    model = train_model(train_rep, feature_columns=feature_columns)
    if model is None:
        raise RuntimeError(f"Training failed for {spec.label}")

    x_eval = prepare_features(eval_rep, feature_columns=feature_columns)
    proba = np.asarray(model.predict_proba(x_eval), dtype=float)

    if proba.shape != (len(eval_df), 3):
        raise ValueError(
            f"{spec.label}: expected probability shape {(len(eval_df), 3)}, "
            f"got {proba.shape}"
        )
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError(f"{spec.label}: invalid probabilities")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{spec.label}: probabilities do not sum to 1")

    return proba


def prediction_frame_for_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    fold_idx: int,
) -> pd.DataFrame:
    """Create all seven paired probability sources for one test fold."""
    odds_mask = valid_odds_mask(df_test)
    eval_df = df_test.loc[odds_mask].copy()
    if eval_df.empty:
        logger.warning("Fold %d has no valid-odds evaluation rows", fold_idx)
        return pd.DataFrame()

    y_true = eval_df["FTR"].map(CLASS_MAP)
    if y_true.isna().any():
        raise ValueError(f"Fold {fold_idx}: unknown result label")

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

    odds_proba = normalized_odds_probs(eval_df)
    for index, sign in enumerate(("H", "D", "A")):
        output[f"odds_{sign}"] = odds_proba[:, index]

    for spec in MODEL_VARIANTS:
        logger.info("Fold %d: training %s", fold_idx, spec.label)
        proba = _fit_predict_variant(df_train, eval_df, spec)
        for index, sign in enumerate(("H", "D", "A")):
            output[f"{spec.prefix}_{sign}"] = proba[:, index]

    return output


def run_diagnostic(
    df: pd.DataFrame,
    *,
    n_segments: int = 4,
) -> pd.DataFrame:
    """Run the six-model date-safe expanding-window diagnostic."""
    work = canonicalize_history(df)
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
    validate_prediction_frame(predictions)
    return predictions


def probability_columns(label: str) -> list[str]:
    if label == "Odds":
        prefix = "odds"
    else:
        prefix = VARIANT_BY_LABEL[label].prefix
    return [f"{prefix}_H", f"{prefix}_D", f"{prefix}_A"]


def probability_matrix(group: pd.DataFrame, label: str) -> np.ndarray:
    return group[probability_columns(label)].to_numpy(dtype=float)


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    """Verify paired N and complete predictions for every variant."""
    required = []
    for label in ALL_VARIANT_LABELS:
        required.extend(probability_columns(label))

    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction frame missing columns: {missing}")
    if predictions[required].isna().any().any():
        raise ValueError("Prediction frame has missing variant probabilities")

    for fold_idx, group in predictions.groupby("fold", sort=True):
        for column in ("train_N", "test_N", "paired_N"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Fold {fold_idx}: inconsistent {column}")
        if int(group["paired_N"].iloc[0]) != len(group):
            raise ValueError(f"Fold {fold_idx}: paired_N does not match row count")


def validate_pr43_sample(predictions: pd.DataFrame) -> None:
    """
    Enforce exact reuse of PR #43's committed evaluation sample.

    Use --allow-sample-drift only for exploratory runs, never for the committed
    benchmark requested by this task.
    """
    expected = EXPECTED_PR43_SAMPLE
    errors: list[str] = []

    if len(predictions) != expected["paired_rows"]:
        errors.append(
            f"paired_rows expected {expected['paired_rows']}, got {len(predictions)}"
        )

    date_min = str(pd.to_datetime(predictions["Date"]).min().date())
    date_max = str(pd.to_datetime(predictions["Date"]).max().date())
    if date_min != expected["date_min"]:
        errors.append(f"date_min expected {expected['date_min']}, got {date_min}")
    if date_max != expected["date_max"]:
        errors.append(f"date_max expected {expected['date_max']}, got {date_max}")

    actual_folds = set(int(value) for value in predictions["fold"].unique())
    expected_folds = set(expected["folds"])
    if actual_folds != expected_folds:
        errors.append(
            f"fold ids expected {sorted(expected_folds)}, got {sorted(actual_folds)}"
        )

    for fold_idx, fold_expected in expected["folds"].items():
        group = predictions[predictions["fold"] == fold_idx]
        if group.empty:
            continue
        for key in ("train_N", "test_N", "paired_N"):
            actual = int(group[key].iloc[0])
            if actual != fold_expected[key]:
                errors.append(
                    f"fold {fold_idx} {key} expected {fold_expected[key]}, got {actual}"
                )

    if errors:
        raise ValueError(
            "PR #43 sample parity failed:\n- " + "\n- ".join(errors)
        )


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    onehot = np.eye(3, dtype=float)[y_true]
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def x_top2_mask(y_proba: np.ndarray) -> np.ndarray:
    """
    Whether X is not strictly below both alternatives.

    Ties at the second-highest probability count as top-2.
    """
    return y_proba[:, 1] >= np.minimum(y_proba[:, 0], y_proba[:, 2])


def metric_row(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty sample")

    pred = np.argmax(y_proba, axis=1)
    x_actual = (y_true == 1).astype(float)
    x_prob = y_proba[:, 1]

    return {
        "N": int(len(y_true)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "LogLoss": float(log_loss(y_true, y_proba, labels=[0, 1, 2])),
        "Brier": multiclass_brier(y_true, y_proba),
        "X_top2_rate": float(x_top2_mask(y_proba).mean()),
        "X_mean_prob": float(x_prob.mean()),
        "X_actual_rate": float(x_actual.mean()),
        "X_brier": float(np.mean((x_prob - x_actual) ** 2)),
    }


def per_match_logloss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    true_prob = y_proba[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(true_prob, EPS, 1.0))


def per_match_brier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    onehot = np.eye(3, dtype=float)[y_true]
    return np.sum((y_proba - onehot) ** 2, axis=1)


def bootstrap_pairwise_ci_lookup(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """
    Compute all required paired CIs from one shared set of resamples.

    Reusing the same bootstrap indices across variants is both faster and a
    cleaner paired comparison than launching a separate bootstrap for every
    table cell.
    """
    labels = list(probabilities)
    label_index = {label: index for index, label in enumerate(labels)}

    comparisons: set[tuple[str, str]] = set()
    for spec in MODEL_VARIANTS:
        comparisons.add((spec.label, "Odds"))
        comparisons.add((spec.label, f"{spec.feature_set}/league_none"))
        comparisons.add((spec.label, f"{spec.feature_set}/league_ordinal"))

    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for candidate, reference in comparisons:
        if candidate == reference:
            lookup[("logloss", candidate, reference)] = (0.0, 0.0)
            lookup[("brier", candidate, reference)] = (0.0, 0.0)

    if n_bootstrap <= 0 or len(y_true) < 2:
        for candidate, reference in comparisons:
            if candidate != reference:
                lookup[("logloss", candidate, reference)] = (
                    float("nan"),
                    float("nan"),
                )
                lookup[("brier", candidate, reference)] = (
                    float("nan"),
                    float("nan"),
                )
        return lookup

    logloss_matrix = np.column_stack(
        [per_match_logloss(y_true, probabilities[label]) for label in labels]
    )
    brier_matrix = np.column_stack(
        [per_match_brier(y_true, probabilities[label]) for label in labels]
    )

    rng = np.random.default_rng(seed)
    n = len(y_true)
    chunk_size = max(1, min(100, n_bootstrap))
    remaining = n_bootstrap
    boot_logloss: list[np.ndarray] = []
    boot_brier: list[np.ndarray] = []

    while remaining:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(count, n))
        boot_logloss.append(logloss_matrix[indices].mean(axis=1))
        boot_brier.append(brier_matrix[indices].mean(axis=1))
        remaining -= count

    logloss_means = np.concatenate(boot_logloss, axis=0)
    brier_means = np.concatenate(boot_brier, axis=0)

    for candidate, reference in comparisons:
        if candidate == reference:
            continue

        candidate_index = label_index[candidate]
        reference_index = label_index[reference]

        ll_delta = (
            logloss_means[:, candidate_index]
            - logloss_means[:, reference_index]
        )
        br_delta = (
            brier_means[:, candidate_index]
            - brier_means[:, reference_index]
        )

        lookup[("logloss", candidate, reference)] = tuple(
            float(value) for value in np.quantile(ll_delta, [0.025, 0.975])
        )
        lookup[("brier", candidate, reference)] = tuple(
            float(value) for value in np.quantile(br_delta, [0.025, 0.975])
        )

    return lookup

def summarize_group(
    group: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    """Metrics plus paired deltas vs odds, league_none, and league_ordinal."""
    y_true = group["y_true"].to_numpy(dtype=int)
    probabilities = {
        label: probability_matrix(group, label)
        for label in ALL_VARIANT_LABELS
    }
    metrics_by_label = {
        label: metric_row(y_true, proba)
        for label, proba in probabilities.items()
    }
    ci_lookup = bootstrap_pairwise_ci_lookup(
        y_true,
        probabilities,
        n_bootstrap=bootstrap,
        seed=seed,
    )

    def comparison(
        candidate: str,
        reference: str,
        metric: str,
    ) -> tuple[float, float, float]:
        metric_key = "LogLoss" if metric == "logloss" else "Brier"
        delta = float(
            metrics_by_label[candidate][metric_key]
            - metrics_by_label[reference][metric_key]
        )
        lo, hi = ci_lookup[(metric, candidate, reference)]
        return delta, lo, hi

    rows: list[dict] = []
    for label in ALL_VARIANT_LABELS:
        metrics = metrics_by_label[label]

        if label == "Odds":
            feature_set = "reference"
            representation = "odds"
            odds_delta = (0.0, 0.0, 0.0)
            odds_brier_delta = (0.0, 0.0, 0.0)
            none_delta = (float("nan"),) * 3
            none_brier_delta = (float("nan"),) * 3
            ordinal_delta = (float("nan"),) * 3
            ordinal_brier_delta = (float("nan"),) * 3
        else:
            spec = VARIANT_BY_LABEL[label]
            feature_set = spec.feature_set
            representation = spec.representation
            none_label = f"{spec.feature_set}/league_none"
            ordinal_label = f"{spec.feature_set}/league_ordinal"

            odds_delta = comparison(label, "Odds", "logloss")
            odds_brier_delta = comparison(label, "Odds", "brier")
            none_delta = comparison(label, none_label, "logloss")
            none_brier_delta = comparison(label, none_label, "brier")
            ordinal_delta = comparison(label, ordinal_label, "logloss")
            ordinal_brier_delta = comparison(label, ordinal_label, "brier")

        rows.append(
            {
                "Variant": label,
                "FeatureSet": feature_set,
                "LeagueRepresentation": representation,
                **metrics,
                "Delta_LogLoss_vs_Odds": odds_delta[0],
                "Delta_LogLoss_vs_Odds_CI95_L": odds_delta[1],
                "Delta_LogLoss_vs_Odds_CI95_U": odds_delta[2],
                "Delta_Brier_vs_Odds": odds_brier_delta[0],
                "Delta_Brier_vs_Odds_CI95_L": odds_brier_delta[1],
                "Delta_Brier_vs_Odds_CI95_U": odds_brier_delta[2],
                "Delta_LogLoss_vs_None": none_delta[0],
                "Delta_LogLoss_vs_None_CI95_L": none_delta[1],
                "Delta_LogLoss_vs_None_CI95_U": none_delta[2],
                "Delta_Brier_vs_None": none_brier_delta[0],
                "Delta_Brier_vs_None_CI95_L": none_brier_delta[1],
                "Delta_Brier_vs_None_CI95_U": none_brier_delta[2],
                "Delta_LogLoss_vs_Ordinal": ordinal_delta[0],
                "Delta_LogLoss_vs_Ordinal_CI95_L": ordinal_delta[1],
                "Delta_LogLoss_vs_Ordinal_CI95_U": ordinal_delta[2],
                "Delta_Brier_vs_Ordinal": ordinal_brier_delta[0],
                "Delta_Brier_vs_Ordinal_CI95_L": ordinal_brier_delta[1],
                "Delta_Brier_vs_Ordinal_CI95_U": ordinal_brier_delta[2],
            }
        )

    return rows

def build_group_table(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    if group_columns:
        grouped = predictions.groupby(
            list(group_columns),
            sort=True,
            dropna=False,
        )
        iterator = grouped
    else:
        iterator = [((), predictions)]

    for group_index, (group_key, group) in enumerate(iterator):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = dict(zip(group_columns, group_key))

        for row in summarize_group(
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
        train_n = int(group["train_N"].iloc[0])
        test_n = int(group["test_N"].iloc[0])
        paired_n = int(group["paired_N"].iloc[0])

        for row in summarize_group(
            group,
            bootstrap=bootstrap,
            seed=seed + int(fold_idx) * 70_001,
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


def build_x_calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Overall and per-league p_X calibration for all seven sources."""
    rows: list[dict] = []
    scopes = [("overall", "ALL", predictions)]
    scopes.extend(
        ("league", str(league), group)
        for league, group in predictions.groupby("League", sort=True)
    )

    for scope, league, group in scopes:
        y_true = group["y_true"].to_numpy(dtype=int)
        observed = (y_true == 1).astype(float)

        for label in ALL_VARIANT_LABELS:
            x_prob = probability_matrix(group, label)[:, 1]
            for lower, upper, bin_label in X_CALIBRATION_BINS:
                if upper > 1.0:
                    mask = (x_prob >= lower) & (x_prob <= 1.0)
                else:
                    mask = (x_prob >= lower) & (x_prob < upper)

                count = int(mask.sum())
                rows.append(
                    {
                        "Scope": scope,
                        "League": league,
                        "Variant": label,
                        "Bin": bin_label,
                        "N": count,
                        "MeanPredictedX": (
                            float(x_prob[mask].mean()) if count else float("nan")
                        ),
                        "ObservedXRate": (
                            float(observed[mask].mean()) if count else float("nan")
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]

    integer_columns = {"Fold", "Train_N", "Test_N", "Paired_N", "N"}
    text_columns = {
        "League",
        "Season",
        "Variant",
        "FeatureSet",
        "LeagueRepresentation",
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
                values.append(_fmt(row[column]))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def evidence_text(row: pd.Series, reference: str) -> str:
    """Neutral evidence wording for paired LogLoss and Brier comparisons."""
    if reference == "None":
        ll = float(row["Delta_LogLoss_vs_None"])
        ll_lo = float(row["Delta_LogLoss_vs_None_CI95_L"])
        ll_hi = float(row["Delta_LogLoss_vs_None_CI95_U"])
        br = float(row["Delta_Brier_vs_None"])
        br_lo = float(row["Delta_Brier_vs_None_CI95_L"])
        br_hi = float(row["Delta_Brier_vs_None_CI95_U"])
    elif reference == "Ordinal":
        ll = float(row["Delta_LogLoss_vs_Ordinal"])
        ll_lo = float(row["Delta_LogLoss_vs_Ordinal_CI95_L"])
        ll_hi = float(row["Delta_LogLoss_vs_Ordinal_CI95_U"])
        br = float(row["Delta_Brier_vs_Ordinal"])
        br_lo = float(row["Delta_Brier_vs_Ordinal_CI95_L"])
        br_hi = float(row["Delta_Brier_vs_Ordinal_CI95_U"])
    else:
        raise ValueError(reference)

    if not np.isfinite([ll, ll_lo, ll_hi, br, br_lo, br_hi]).all():
        return "comparison not available"

    if ll < 0 and br < 0:
        if ll_hi < 0 and br_hi < 0:
            return "better on both metrics; paired 95% CIs below 0"
        return "better point estimates; uncertainty overlaps 0"
    if ll > 0 and br > 0:
        if ll_lo > 0 and br_lo > 0:
            return "worse on both metrics; paired 95% CIs above 0"
        return "worse point estimates; uncertainty overlaps 0"
    return "mixed LogLoss/Brier result"


def render_report(
    predictions: pd.DataFrame,
    overall_table: pd.DataFrame,
    fold_table: pd.DataFrame,
    league_table: pd.DataFrame,
    season_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    *,
    bootstrap: int,
    strict_sample: bool,
) -> str:
    date_min = pd.to_datetime(predictions["Date"]).min().date()
    date_max = pd.to_datetime(predictions["Date"]).max().date()

    metric_columns = [
        "Variant",
        "N",
        "Accuracy",
        "LogLoss",
        "Brier",
        "X_top2_rate",
        "X_mean_prob",
        "X_actual_rate",
        "X_brier",
        "Delta_LogLoss_vs_Odds",
        "Delta_Brier_vs_Odds",
    ]
    fold_columns = ["Fold", "Train_N", "Test_N", "Paired_N"] + metric_columns
    league_columns = ["League"] + metric_columns
    season_columns = ["Season", "League"] + metric_columns

    direct_columns = [
        "League",
        "FeatureSet",
        "LeagueRepresentation",
        "N",
        "Delta_LogLoss_vs_None",
        "Delta_LogLoss_vs_None_CI95_L",
        "Delta_LogLoss_vs_None_CI95_U",
        "Delta_Brier_vs_None",
        "Delta_Brier_vs_None_CI95_L",
        "Delta_Brier_vs_None_CI95_U",
        "Delta_LogLoss_vs_Ordinal",
        "Delta_LogLoss_vs_Ordinal_CI95_L",
        "Delta_LogLoss_vs_Ordinal_CI95_U",
        "Delta_Brier_vs_Ordinal",
        "Delta_Brier_vs_Ordinal_CI95_L",
        "Delta_Brier_vs_Ordinal_CI95_U",
    ]
    direct = league_table[
        league_table["LeagueRepresentation"].isin(
            ["league_ordinal", "league_onehot"]
        )
    ]

    overall_calibration = calibration_table[
        calibration_table["Scope"] == "overall"
    ]
    calibration_columns = [
        "Variant",
        "Bin",
        "N",
        "MeanPredictedX",
        "ObservedXRate",
    ]

    lines = [
        "# Diagnostic: League representation",
        "",
        "## Scope",
        "",
        f"- Base SHA: `{BASE_SHA}`.",
        "- Diagnostic only; no production setting or model artifact is changed.",
        "- Six independently trained model variants: three League representations "
        "for base features and the same three for base+odds features.",
        "- `league_none` keeps the League column and sets every value to `-1`, "
        "matching current production behaviour exactly.",
        "- `league_ordinal` uses E0-E3 encoded as 0-3.",
        "- `league_onehot` removes League from X and uses four fixed binary columns.",
        "- All variants and the bookmaker reference use exactly the same valid-odds "
        "test rows in every fold.",
        f"- Paired evaluation rows: {len(predictions)}.",
        f"- Evaluation date range: {date_min} to {date_max}.",
        f"- Paired bootstrap resamples: {bootstrap}.",
        f"- Strict PR #43 sample parity enforced: {strict_sample}.",
        "- Negative delta means the candidate is better than the named reference.",
        "",
        "## Overall metrics",
        "",
        markdown_table(overall_table[metric_columns], metric_columns),
        "",
        "## Per-fold metrics",
        "",
        markdown_table(fold_table[fold_columns], fold_columns),
        "",
        "## Per-league metrics",
        "",
        markdown_table(league_table[league_columns], league_columns),
        "",
        "## Direct League-representation comparisons",
        "",
        "These are the decision-bearing paired comparisons. Deltas against odds "
        "alone cannot establish whether League itself helps.",
        "",
        markdown_table(direct[direct_columns], direct_columns),
        "",
        "## Per-league evidence summary",
        "",
    ]

    for _, row in direct.iterrows():
        representation = row["LeagueRepresentation"]
        reference = "None" if representation == "league_ordinal" else "Ordinal"
        lines.append(
            f"- **{row['League']} / {row['FeatureSet']} / {representation} "
            f"vs {reference.lower()}**: {evidence_text(row, reference)} "
            f"(ΔLogLoss={_fmt(row[f'Delta_LogLoss_vs_{reference}'])}, "
            f"ΔBrier={_fmt(row[f'Delta_Brier_vs_{reference}'])})."
        )

    lines.extend(
        [
            "",
            "## Per-league and season metrics",
            "",
            markdown_table(season_table[season_columns], season_columns),
            "",
            "## Overall p_X calibration",
            "",
            markdown_table(
                overall_calibration[calibration_columns],
                calibration_columns,
            ),
            "",
            "Per-league p_X calibration is saved in the generated "
            "`_X_CALIBRATION.csv` artifact.",
            "",
            "## Interpretation guardrails",
            "",
            "- LogLoss and Brier are the primary model-selection metrics.",
            "- Representation decisions require the direct paired deltas against "
            "`league_none`; comparison with odds is contextual only.",
            "- Do not declare a representation winner when LogLoss/Brier conflict "
            "or paired confidence intervals overlap 0.",
            "- `X_top2_rate` counts ties for second place as top-2.",
            "- Small third-decimal differences are expected and must not trigger "
            "a production change without paired evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def save_outputs(
    report: str,
    fold_table: pd.DataFrame,
    league_table: pd.DataFrame,
    season_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    *,
    report_path: Path,
) -> tuple[Path, ...]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    outputs = (
        report_path,
        report_path.with_name(report_path.stem + "_FOLD.csv"),
        report_path.with_name(report_path.stem + "_LEAGUE.csv"),
        report_path.with_name(report_path.stem + "_SEASON.csv"),
        report_path.with_name(report_path.stem + "_X_CALIBRATION.csv"),
    )
    fold_table.to_csv(outputs[1], index=False)
    league_table.to_csv(outputs[2], index=False)
    season_table.to_csv(outputs[3], index=False)
    calibration_table.to_csv(outputs[4], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark League feature representations"
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help=(
            "Download/cache historical data before running. Exact PR #43 "
            "sample parity is still validated unless --allow-sample-drift is set."
        ),
    )
    parser.add_argument(
        "--allow-sample-drift",
        action="store_true",
        help="Allow a sample different from the committed PR #43 sample",
    )
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RESULTS_LEAGUE_REPRESENTATION.md"),
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

    predictions = run_diagnostic(df, n_segments=args.segments)
    if not args.allow_sample_drift:
        validate_pr43_sample(predictions)

    overall_table = build_group_table(
        predictions,
        [],
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    fold_table = build_fold_table(
        predictions,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    league_table = build_group_table(
        predictions,
        ["League"],
        bootstrap=args.bootstrap,
        seed=args.seed + 10_000,
    )
    season_table = build_group_table(
        predictions,
        ["League", "Season"],
        bootstrap=args.bootstrap,
        seed=args.seed + 20_000,
    )
    calibration_table = build_x_calibration_table(predictions)

    report = render_report(
        predictions,
        overall_table,
        fold_table,
        league_table,
        season_table,
        calibration_table,
        bootstrap=args.bootstrap,
        strict_sample=not args.allow_sample_drift,
    )
    outputs = save_outputs(
        report,
        fold_table,
        league_table,
        season_table,
        calibration_table,
        report_path=args.output,
    )

    logger.info("League-representation diagnostic complete")
    for path in outputs:
        logger.info("Wrote %s", path)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
