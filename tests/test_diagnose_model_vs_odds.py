# Revision 2 tests: per-fold diagnostics and CI-not-computed coverage included.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.diagnose_model_vs_odds import (
    build_fold_table,
    build_summary_tables,
    date_safe_walk_forward_folds,
    decode_league,
    metric_row,
    paired_bootstrap_delta_ci,
    render_markdown_report,
    valid_odds_mask,
)


def test_decode_league_accepts_raw_and_encoded_values():
    assert decode_league("E0") == "E0"
    assert decode_league(3) == "E3"
    assert decode_league(2.0) == "E2"
    assert decode_league("1.0") == "E1"
    assert decode_league(None) == ""
    assert decode_league("SP1") == ""


def test_valid_odds_mask_rejects_missing_and_invalid_rows():
    df = pd.DataFrame(
        {
            "has_odds": [1.0, 0.0, 1.0, 1.0],
            "ImpliedHome": [0.50, 0.50, np.nan, 0.0],
            "ImpliedDraw": [0.25, 0.25, 0.30, 0.50],
            "ImpliedAway": [0.25, 0.25, 0.30, 0.50],
        }
    )
    mask = valid_odds_mask(df)
    assert mask.tolist() == [True, False, False, False]


def test_metric_row_reports_draw_precision_and_recall():
    y_true = np.array([0, 1, 1, 2, 2])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    m = metric_row(y_true, proba)
    assert m["N"] == 5
    assert m["Accuracy"] == pytest.approx(3 / 5)
    assert m["X_precision"] == pytest.approx(1 / 2)
    assert m["X_recall"] == pytest.approx(1 / 2)


def test_date_safe_walk_forward_never_leaks_same_or_future_date():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                ]
            )
        }
    )
    folds = date_safe_walk_forward_folds(df, n_segments=4)
    assert folds

    for _, train_mask, test_mask in folds:
        train_dates = df.loc[train_mask, "Date"]
        test_dates = df.loc[test_mask, "Date"]

        assert set(train_dates).isdisjoint(set(test_dates))
        assert train_dates.max() < test_dates.min()


def test_paired_bootstrap_delta_ci_has_expected_sign_for_better_candidate():
    y_true = np.array([0, 1, 2] * 20)
    odds = np.tile(
        np.array(
            [
                [0.45, 0.30, 0.25],
                [0.35, 0.35, 0.30],
                [0.35, 0.30, 0.35],
            ]
        ),
        (20, 1),
    )
    candidate = np.tile(
        np.array(
            [
                [0.65, 0.20, 0.15],
                [0.20, 0.60, 0.20],
                [0.15, 0.20, 0.65],
            ]
        ),
        (20, 1),
    )

    ll_lo, ll_hi = paired_bootstrap_delta_ci(
        y_true,
        candidate,
        odds,
        metric="logloss",
        n_bootstrap=300,
        seed=42,
    )
    br_lo, br_hi = paired_bootstrap_delta_ci(
        y_true,
        candidate,
        odds,
        metric="brier",
        n_bootstrap=300,
        seed=42,
    )

    assert ll_hi < 0
    assert br_hi < 0
    assert ll_lo <= ll_hi
    assert br_lo <= br_hi


def _paired_predictions_fixture() -> pd.DataFrame:
    rows = []
    outcomes = [0, 1, 2, 0, 1, 2]
    for i, y in enumerate(outcomes):
        league = "E0" if i < 3 else "E1"
        season = "2425" if i % 2 == 0 else "2526"

        odds = np.array([0.45, 0.30, 0.25])
        if y == 1:
            odds = np.array([0.35, 0.35, 0.30])
        elif y == 2:
            odds = np.array([0.35, 0.30, 0.35])

        improved = odds.copy()
        improved[y] += 0.15
        improved /= improved.sum()

        model = np.array([0.40, 0.30, 0.30])

        rows.append(
            {
                "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "League": league,
                "Season": season,
                "fold": 1 + (i // 3),
                "train_N": 100 if i < 3 else 200,
                "test_N": 50,
                "paired_N": 3,
                "y_true": y,
                "model_H": model[0],
                "model_D": model[1],
                "model_A": model[2],
                "odds_H": odds[0],
                "odds_D": odds[1],
                "odds_A": odds[2],
                "model_odds_H": improved[0],
                "model_odds_D": improved[1],
                "model_odds_A": improved[2],
            }
        )
    return pd.DataFrame(rows)


def test_summary_tables_keep_identical_n_for_all_variants():
    predictions = _paired_predictions_fixture()
    league, season = build_summary_tables(
        predictions, bootstrap=50, seed=7
    )

    for _, group in league.groupby("League"):
        assert group["N"].nunique() == 1
        assert set(group["Variant"]) == {"Model", "Odds", "Model+Odds"}

    for _, group in season.groupby(["League", "Season"]):
        assert group["N"].nunique() == 1
        assert set(group["Variant"]) == {"Model", "Odds", "Model+Odds"}


def test_model_plus_odds_delta_is_negative_when_fixture_is_better():
    predictions = _paired_predictions_fixture()
    league, _ = build_summary_tables(predictions, bootstrap=50, seed=7)
    candidate = league[league["Variant"] == "Model+Odds"]
    assert (candidate["Delta_LogLoss_vs_Odds"] < 0).all()
    assert (candidate["Delta_Brier_vs_Odds"] < 0).all()


def test_report_contains_required_diagnostic_columns():
    predictions = _paired_predictions_fixture()
    league, season = build_summary_tables(predictions, bootstrap=0, seed=7)
    fold = build_fold_table(predictions, bootstrap=0, seed=7)
    report = render_markdown_report(
        predictions,
        fold,
        league,
        season,
        n_segments=4,
        bootstrap=0,
        base_sha="10b2d43",
    )

    for expected in [
        "Accuracy",
        "LogLoss",
        "Brier",
        "X_precision",
        "X_recall",
        "Delta_LogLoss_vs_Odds",
        "Delta_Brier_vs_Odds",
        "Per-season metrics",
        "No production defaults",
    ]:
        assert expected in report

def test_fold_table_contains_train_test_and_paired_n():
    predictions = _paired_predictions_fixture()
    fold = build_fold_table(predictions, bootstrap=0, seed=7)

    assert set(["Fold", "Train_N", "Test_N", "Paired_N", "Variant"]).issubset(
        fold.columns
    )
    for _, group in fold.groupby("Fold"):
        assert group["Train_N"].nunique() == 1
        assert group["Test_N"].nunique() == 1
        assert group["Paired_N"].nunique() == 1
        assert group["N"].nunique() == 1
        assert int(group["Paired_N"].iloc[0]) == int(group["N"].iloc[0])
        assert set(group["Variant"]) == {"Model", "Odds", "Model+Odds"}


def test_report_says_ci_not_computed_when_bootstrap_disabled():
    predictions = _paired_predictions_fixture()
    fold = build_fold_table(predictions, bootstrap=0, seed=7)
    league, season = build_summary_tables(predictions, bootstrap=0, seed=7)

    report = render_markdown_report(
        predictions,
        fold,
        league,
        season,
        n_segments=4,
        bootstrap=0,
        base_sha="10b2d43",
    )

    assert "CI not computed" in report
    assert "Per-fold metrics and warm-up diagnostic" in report
    assert "Train_N" in report
    assert "Test_N" in report

