import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scripts.diagnose_production_blend as module_under_test
from combined_probability import DEFAULT_WEIGHTS, combine_probabilities
from scripts.diagnose_production_blend import (
    CORE_VARIANTS,
    DEFAULT_MAX_DATE,
    PRODUCTION_MODEL_WEIGHT,
    PRODUCTION_ODDS_WEIGHT,
    SWEEP_WEIGHTS,
    build_core_group_table,
    build_fold_table,
    build_overall_decision_table,
    build_weight_sweep_table,
    build_x_calibration_table,
    effective_production_weights,
    freeze_reference_window,
    parse_max_date,
    probability_matrix,
    production_blend_rows,
    render_report,
    save_outputs,
    select_optimal_weight,
    sweep_blend_rows,
    sweep_probability_columns,
    validate_prediction_frame,
)


def _probability_sources():
    odds = np.array(
        [
            [0.55, 0.25, 0.20],
            [0.38, 0.34, 0.28],
            [0.30, 0.28, 0.42],
            [0.46, 0.30, 0.24],
            [0.35, 0.31, 0.34],
            [0.28, 0.27, 0.45],
            [0.50, 0.29, 0.21],
            [0.33, 0.35, 0.32],
        ],
        dtype=float,
    )
    model = np.array(
        [
            [0.48, 0.30, 0.22],
            [0.40, 0.32, 0.28],
            [0.35, 0.30, 0.35],
            [0.44, 0.33, 0.23],
            [0.38, 0.34, 0.28],
            [0.31, 0.31, 0.38],
            [0.47, 0.32, 0.21],
            [0.36, 0.34, 0.30],
        ],
        dtype=float,
    )
    return odds, model


def _prediction_fixture() -> pd.DataFrame:
    odds, model = _probability_sources()
    production = production_blend_rows(odds, model)
    outcomes = np.array([0, 1, 2, 0, 2, 2, 0, 1], dtype=int)

    rows = []
    for index, y_true in enumerate(outcomes):
        fold = 1 if index < 4 else 2
        league = "E0" if index < 4 else "E1"
        row = {
            "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
            "League": league,
            "Season": "2526",
            "FTR": ("H", "D", "A")[y_true],
            "y_true": y_true,
            "fold": fold,
            "train_N": 100 if fold == 1 else 200,
            "test_N": 4,
            "paired_N": 4,
            "max_date": "2026-05-24",
            "source_max_date": "2026-08-08",
            "source_row_N": 10,
            "frozen_row_N": 8,
            "excluded_post_max_date_N": 2,
        }
        for prefix, probabilities in (
            ("odds", odds),
            ("model", model),
            ("production", production),
        ):
            row[f"{prefix}_H"] = probabilities[index, 0]
            row[f"{prefix}_D"] = probabilities[index, 1]
            row[f"{prefix}_A"] = probabilities[index, 2]

        for weight in SWEEP_WEIGHTS:
            probabilities = sweep_blend_rows(odds, model, weight)
            columns = sweep_probability_columns(weight)
            row[columns[0]] = probabilities[index, 0]
            row[columns[1]] = probabilities[index, 1]
            row[columns[2]] = probabilities[index, 2]

        rows.append(row)

    return pd.DataFrame(rows)


def _all_tables(bootstrap=40, seed=7):
    predictions = _prediction_fixture()
    core_overall = build_core_group_table(
        predictions,
        [],
        bootstrap=bootstrap,
        seed=seed,
    )
    fold = build_fold_table(
        predictions,
        bootstrap=bootstrap,
        seed=seed + 100,
    )
    league = build_core_group_table(
        predictions,
        ["League"],
        bootstrap=bootstrap,
        seed=seed + 200,
    )
    season = build_core_group_table(
        predictions,
        ["League", "Season"],
        bootstrap=bootstrap,
        seed=seed + 300,
    )
    sweep, optimum = build_weight_sweep_table(
        predictions,
        bootstrap=bootstrap,
        seed=seed + 400,
    )
    overall = build_overall_decision_table(
        core_overall,
        sweep,
        optimum,
    )
    calibration = build_x_calibration_table(predictions, optimum)
    return predictions, overall, fold, league, season, sweep, calibration, optimum


def test_parse_max_date_default_and_invalid_value():
    assert DEFAULT_MAX_DATE == "2026-05-24"
    assert parse_max_date(DEFAULT_MAX_DATE) == pd.Timestamp("2026-05-24")

    with pytest.raises(Exception):
        parse_max_date("not-a-date")


def test_freeze_reference_window_excludes_future_rows():
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2026-05-23", "2026-05-24", "2026-08-08"]
            ),
            "League": ["E0", "E1", "E3"],
            "FTR": ["H", "D", "A"],
        }
    )

    frozen, metadata = freeze_reference_window(
        frame,
        max_date="2026-05-24",
    )

    assert frozen["Date"].max() == pd.Timestamp("2026-05-24")
    assert len(frozen) == 2
    assert metadata == {
        "max_date": "2026-05-24",
        "source_max_date": "2026-08-08",
        "source_row_N": 3,
        "frozen_row_N": 2,
        "excluded_post_max_date_N": 1,
    }


def test_max_date_filter_keeps_fold_boundaries_identical_to_reference_window():
    # 120 in-window dates, then five later dates that would otherwise change
    # the unique-date segmentation.
    reference = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=120, freq="D"),
            "League": ["E0"] * 120,
            "FTR": ["H"] * 120,
        }
    )
    extended = pd.concat(
        [
            reference,
            pd.DataFrame(
                {
                    "Date": pd.date_range("2026-08-08", periods=5, freq="D"),
                    "League": ["E0"] * 5,
                    "FTR": ["H"] * 5,
                }
            ),
        ],
        ignore_index=True,
    )

    frozen, _ = freeze_reference_window(
        extended,
        max_date=reference["Date"].max(),
    )

    reference_folds = module_under_test.date_safe_walk_forward_folds(
        reference,
        n_segments=4,
    )
    frozen_folds = module_under_test.date_safe_walk_forward_folds(
        frozen,
        n_segments=4,
    )

    assert len(reference_folds) == len(frozen_folds)
    for (_, reference_train, reference_test), (
        _,
        frozen_train,
        frozen_test,
    ) in zip(reference_folds, frozen_folds):
        assert reference_train.tolist() == frozen_train.tolist()
        assert reference_test.tolist() == frozen_test.tolist()


def test_run_diagnostic_requires_pre_frozen_metadata():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=120, freq="D"),
            "League": ["E0"] * 120,
            "FTR": ["H"] * 120,
        }
    )

    with pytest.raises(ValueError, match="freeze_metadata is required"):
        module_under_test.run_diagnostic(frame)


def test_effective_production_weights_are_50_over_85_and_35_over_85():
    odds_weight, model_weight = effective_production_weights()

    assert DEFAULT_WEIGHTS["odds"] == pytest.approx(0.50)
    assert DEFAULT_WEIGHTS["model"] == pytest.approx(0.35)
    assert odds_weight == pytest.approx(50.0 / 85.0)
    assert model_weight == pytest.approx(35.0 / 85.0)
    assert odds_weight + model_weight == pytest.approx(1.0)


def test_production_blend_calls_real_combiner_with_no_streck(monkeypatch):
    odds, model = _probability_sources()
    calls = []
    real_function = combine_probabilities

    def spy(*, odds_probs, model_probs, streck_pcts, weights):
        calls.append(
            {
                "odds": odds_probs.copy(),
                "model": model_probs.copy(),
                "streck": streck_pcts,
                "weights": weights,
            }
        )
        return real_function(
            odds_probs=odds_probs,
            model_probs=model_probs,
            streck_pcts=streck_pcts,
            weights=weights,
        )

    monkeypatch.setattr(module_under_test, "combine_probabilities", spy)
    result = production_blend_rows(odds[:3], model[:3])

    assert result.shape == (3, 3)
    assert len(calls) == 3
    assert all(call["streck"] is None for call in calls)
    assert all(call["weights"] is None for call in calls)


def test_production_blend_matches_default_weight_renormalisation():
    odds, model = _probability_sources()
    actual = production_blend_rows(odds, model)
    expected = (
        DEFAULT_WEIGHTS["odds"] * odds
        + DEFAULT_WEIGHTS["model"] * model
    ) / (
        DEFAULT_WEIGHTS["odds"] + DEFAULT_WEIGHTS["model"]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_weight_sweep_contains_grid_endpoints_and_exact_production_weight():
    expected_grid = {round(index / 20.0, 12) for index in range(21)}

    assert expected_grid.issubset(set(SWEEP_WEIGHTS))
    assert min(SWEEP_WEIGHTS) == 0.0
    assert max(SWEEP_WEIGHTS) == 1.0
    assert any(
        abs(weight - PRODUCTION_MODEL_WEIGHT) <= 1e-12
        for weight in SWEEP_WEIGHTS
    )


def test_w_zero_is_numerically_identical_to_odds_only():
    odds, model = _probability_sources()
    sweep_zero = sweep_blend_rows(odds, model, 0.0)

    np.testing.assert_allclose(sweep_zero, odds, atol=1e-12, rtol=0.0)


def test_exact_production_weight_matches_production_blend():
    odds, model = _probability_sources()

    actual = sweep_blend_rows(odds, model, PRODUCTION_MODEL_WEIGHT)
    expected = production_blend_rows(odds, model)

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_prediction_frame_validates_all_sources_and_identity_anchors():
    predictions = _prediction_fixture()
    validate_prediction_frame(predictions)

    for label in CORE_VARIANTS:
        assert probability_matrix(predictions, label).shape == (8, 3)


def test_prediction_frame_rejects_a_nonidentical_w_zero_anchor():
    predictions = _prediction_fixture()
    zero_column = sweep_probability_columns(0.0)[0]
    predictions.loc[0, zero_column] += 0.001

    with pytest.raises(ValueError, match="w=0 sweep is not identical"):
        validate_prediction_frame(predictions)


def test_core_tables_keep_identical_n_and_required_deltas():
    predictions = _prediction_fixture()
    table = build_core_group_table(
        predictions,
        ["League"],
        bootstrap=30,
        seed=3,
    )

    for _, group in table.groupby("League"):
        assert set(group["Variant"]) == set(CORE_VARIANTS)
        assert group["N"].nunique() == 1

    for column in (
        "Delta_LogLoss_vs_Odds",
        "Delta_LogLoss_vs_Odds_CI95_L",
        "Delta_LogLoss_vs_Odds_CI95_U",
        "Delta_Brier_vs_Odds",
        "Delta_Brier_vs_Odds_CI95_L",
        "Delta_Brier_vs_Odds_CI95_U",
    ):
        assert column in table.columns


def test_select_optimal_weight_breaks_logloss_tie_toward_lower_weight():
    table = pd.DataFrame(
        {
            "ModelWeight": [0.20, 0.10, 0.30],
            "LogLoss": [1.0, 1.0, 1.1],
        }
    )

    assert select_optimal_weight(table) == pytest.approx(0.10)


def test_weight_sweep_has_overall_and_per_league_rows_and_flags():
    predictions = _prediction_fixture()
    table, optimum = build_weight_sweep_table(
        predictions,
        bootstrap=20,
        seed=11,
    )

    assert set(table["Scope"]) == {"overall", "league"}
    assert set(table[table["Scope"] == "league"]["League"]) == {"E0", "E1"}

    overall = table[table["Scope"] == "overall"]
    assert len(overall) == len(SWEEP_WEIGHTS)
    assert overall["IsOverallOptimum"].sum() == 1
    assert overall["IsProductionWeight"].sum() == 1
    assert any(
        abs(weight - optimum) <= 1e-12
        for weight in overall["ModelWeight"]
    )

    zero = overall[np.isclose(overall["ModelWeight"], 0.0)].iloc[0]
    assert zero["Delta_LogLoss_vs_Odds"] == pytest.approx(0.0, abs=1e-12)
    assert zero["Delta_Brier_vs_Odds"] == pytest.approx(0.0, abs=1e-12)


def test_overall_decision_table_contains_core_variants_and_w_star():
    _, overall, _, _, _, _, _, _ = _all_tables()

    assert set(overall["Variant"]) == {
        "odds_only",
        "model_only",
        "production_blend",
        "sweep_w_star",
    }
    assert overall["N"].nunique() == 1


def test_x_calibration_contains_requested_sources_and_fixed_bins():
    predictions = _prediction_fixture()
    _, optimum = build_weight_sweep_table(
        predictions,
        bootstrap=0,
        seed=9,
    )
    calibration = build_x_calibration_table(predictions, optimum)

    overall = calibration[calibration["Scope"] == "overall"]
    assert set(overall["Source"]) == {
        "odds_only",
        "model_only",
        "production_blend",
        "sweep_w_0",
        "sweep_w_production",
        "sweep_w_star",
    }
    assert set(overall["Bin"]) == {
        "0.00-0.20",
        "0.20-0.25",
        "0.25-0.30",
        "0.30+",
    }
    counts = overall.groupby("Source")["N"].sum()
    assert (counts == len(predictions)).all()


def test_report_contains_streck_limitation_weight_sweep_and_guardrails():
    (
        predictions,
        overall,
        fold,
        league,
        season,
        sweep,
        calibration,
        optimum,
    ) = _all_tables(bootstrap=0)

    report = render_report(
        predictions,
        overall,
        fold,
        league,
        season,
        sweep,
        calibration,
        optimum_weight=optimum,
        bootstrap=0,
        strict_sample=True,
    )

    for expected in (
        "Historical streck data is unavailable",
        "Frozen reference-window max-date: 2026-05-24",
        "Rows after the cutoff excluded before fold construction",
        "frozen parity benchmark whose default max-date is",
        "full 50/35/15 three-source production blend is therefore not",
        "production_blend",
        "Weight sweep — overall",
        "Weight sweep — per league",
        "w* is selected on the same data it is evaluated on",
        "The no-streck result does not establish how real streck would affect",
        "w=0",
    ):
        assert expected in report


def test_save_outputs_creates_all_requested_artifacts(tmp_path):
    (
        _,
        overall,
        fold,
        league,
        season,
        sweep,
        calibration,
        _,
    ) = _all_tables(bootstrap=0)

    outputs = save_outputs(
        "report",
        overall,
        fold,
        league,
        season,
        sweep,
        calibration,
        report_path=tmp_path / "RESULTS_PRODUCTION_BLEND.md",
    )

    expected = {
        "RESULTS_PRODUCTION_BLEND.md",
        "RESULTS_PRODUCTION_BLEND_OVERALL.csv",
        "RESULTS_PRODUCTION_BLEND_FOLD.csv",
        "RESULTS_PRODUCTION_BLEND_LEAGUE.csv",
        "RESULTS_PRODUCTION_BLEND_SEASON.csv",
        "RESULTS_PRODUCTION_BLEND_WEIGHT_SWEEP.csv",
        "RESULTS_PRODUCTION_BLEND_X_CALIBRATION.csv",
    }
    assert {path.name for path in outputs} == expected
    assert all(path.exists() for path in outputs)

    saved_overall = pd.read_csv(
        tmp_path / "RESULTS_PRODUCTION_BLEND_OVERALL.csv"
    )
    assert set(saved_overall["Variant"]) == {
        "odds_only",
        "model_only",
        "production_blend",
        "sweep_w_star",
    }


def test_main_allows_refresh_and_retains_strict_sample_validation(
    monkeypatch,
    tmp_path,
):
    args = type(
        "Args",
        (),
        {
            "refresh_data": True,
            "allow_sample_drift": False,
            "max_date": pd.Timestamp("2026-05-24"),
            "segments": 4,
            "bootstrap": 0,
            "seed": 42,
            "output": tmp_path / "RESULTS_PRODUCTION_BLEND.md",
        },
    )()

    loaded = pd.DataFrame({"sentinel": [1]})
    predictions = _prediction_fixture()
    validation_calls = []
    call_order = []

    monkeypatch.setattr(module_under_test, "parse_args", lambda: args)
    def fake_load_data(refresh=False):
        call_order.append("load")
        return loaded if refresh else pd.DataFrame()

    monkeypatch.setattr(
        module_under_test,
        "load_data",
        fake_load_data,
    )
    monkeypatch.setattr(
        module_under_test,
        "freeze_reference_window",
        lambda df, max_date: (
            call_order.append("freeze") or df,
            {
                "max_date": "2026-05-24",
                "source_max_date": "2026-08-08",
                "source_row_N": len(df),
                "frozen_row_N": len(df),
                "excluded_post_max_date_N": 0,
            },
        ),
    )
    def fake_run_diagnostic(df, n_segments=4, freeze_metadata=None):
        call_order.append("run")
        assert freeze_metadata["max_date"] == "2026-05-24"
        return predictions

    monkeypatch.setattr(
        module_under_test,
        "run_diagnostic",
        fake_run_diagnostic,
    )
    monkeypatch.setattr(
        module_under_test,
        "validate_pr43_sample",
        lambda frame: validation_calls.append(len(frame)),
    )

    simple_core = pd.DataFrame(
        [
            {
                "Variant": label,
                "N": len(predictions),
                "OddsWeight": 1.0 if label == "odds_only" else 0.0,
                "ModelWeight": 1.0 if label == "model_only" else 0.5,
                "StreckMeasured": False,
                "Accuracy": 0.5,
                "LogLoss": 1.0,
                "Brier": 0.6,
                "X_top2_rate": 0.5,
                "X_mean_prob": 0.25,
                "X_actual_rate": 0.25,
                "X_brier": 0.18,
                "Delta_LogLoss_vs_Odds": 0.0,
                "Delta_LogLoss_vs_Odds_CI95_L": 0.0,
                "Delta_LogLoss_vs_Odds_CI95_U": 0.0,
                "Delta_Brier_vs_Odds": 0.0,
                "Delta_Brier_vs_Odds_CI95_L": 0.0,
                "Delta_Brier_vs_Odds_CI95_U": 0.0,
            }
            for label in CORE_VARIANTS
        ]
    )
    simple_sweep = pd.DataFrame(
        [
            {
                "Scope": "overall",
                "League": "ALL",
                "ModelWeight": 0.0,
                "OddsWeight": 1.0,
                "IsGridWeight": True,
                "IsProductionWeight": False,
                "IsOverallOptimum": True,
                "N": len(predictions),
                "Accuracy": 0.5,
                "LogLoss": 1.0,
                "Brier": 0.6,
                "X_top2_rate": 0.5,
                "X_mean_prob": 0.25,
                "X_actual_rate": 0.25,
                "X_brier": 0.18,
                "Delta_LogLoss_vs_Odds": 0.0,
                "Delta_LogLoss_vs_Odds_CI95_L": 0.0,
                "Delta_LogLoss_vs_Odds_CI95_U": 0.0,
                "Delta_Brier_vs_Odds": 0.0,
                "Delta_Brier_vs_Odds_CI95_L": 0.0,
                "Delta_Brier_vs_Odds_CI95_U": 0.0,
            }
        ]
    )

    monkeypatch.setattr(
        module_under_test,
        "build_core_group_table",
        lambda *args, **kwargs: simple_core.copy(),
    )
    monkeypatch.setattr(
        module_under_test,
        "build_fold_table",
        lambda *args, **kwargs: simple_core.copy(),
    )
    monkeypatch.setattr(
        module_under_test,
        "build_weight_sweep_table",
        lambda *args, **kwargs: (simple_sweep.copy(), 0.0),
    )
    monkeypatch.setattr(
        module_under_test,
        "build_overall_decision_table",
        lambda core, sweep, optimum: pd.concat(
            [
                core,
                core.iloc[[0]].assign(Variant="sweep_w_star"),
            ],
            ignore_index=True,
        ),
    )
    monkeypatch.setattr(
        module_under_test,
        "build_x_calibration_table",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        module_under_test,
        "render_report",
        lambda *args, **kwargs: "report",
    )
    saved = []
    monkeypatch.setattr(
        module_under_test,
        "save_outputs",
        lambda *args, **kwargs: saved.append(True) or tuple(),
    )

    assert module_under_test.main() == 0
    assert validation_calls == [len(predictions)]
    assert saved == [True]
    assert call_order == ["load", "freeze", "run"]
