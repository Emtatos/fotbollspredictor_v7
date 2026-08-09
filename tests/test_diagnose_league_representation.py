import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scripts.diagnose_league_representation as module_under_test
from scripts.diagnose_league_representation import (
    ALL_VARIANT_LABELS,
    ONEHOT_COLUMNS,
    OVERALL_DIRECT_COLUMNS,
    VariantSpec,
    apply_league_representation,
    build_fold_table,
    build_group_table,
    build_overall_direct_table,
    build_x_calibration_table,
    feature_columns_for,
    metric_row,
    probability_columns,
    render_report,
    save_outputs,
    validate_prediction_frame,
    x_top2_mask,
)
from schema import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS


def _history_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=4, freq="D"),
            "League": ["E0", "E1", "E2", "E3"],
            "FTR": ["H", "D", "A", "H"],
        }
    )


def test_league_none_keeps_constant_minus_one_column():
    transformed = apply_league_representation(
        _history_fixture(),
        "league_none",
    )
    assert "League" in transformed.columns
    assert transformed["League"].tolist() == [-1.0, -1.0, -1.0, -1.0]


def test_league_ordinal_keeps_canonical_codes_for_project_encoder():
    transformed = apply_league_representation(
        _history_fixture(),
        "league_ordinal",
    )
    assert transformed["League"].tolist() == ["E0", "E1", "E2", "E3"]


def test_league_onehot_always_creates_four_fixed_binary_columns():
    transformed = apply_league_representation(
        _history_fixture().iloc[:2],
        "league_onehot",
    )
    assert all(column in transformed.columns for column in ONEHOT_COLUMNS)
    values = transformed[list(ONEHOT_COLUMNS)].to_numpy()
    assert np.isin(values, [0.0, 1.0]).all()
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    assert transformed["League_E2"].sum() == 0.0
    assert transformed["League_E3"].sum() == 0.0


@pytest.mark.parametrize("feature_set,source", [
    ("base", FEATURE_COLUMNS),
    ("with_odds", ALL_FEATURE_COLUMNS),
])
def test_onehot_feature_columns_replace_not_supplement_league(feature_set, source):
    columns = feature_columns_for(feature_set, "league_onehot")
    assert "League" not in columns
    assert all(column in columns for column in ONEHOT_COLUMNS)
    assert len(columns) == len(source) - 1 + len(ONEHOT_COLUMNS)


def test_non_onehot_feature_columns_are_unchanged():
    assert feature_columns_for("base", "league_none") == list(FEATURE_COLUMNS)
    assert feature_columns_for("base", "league_ordinal") == list(FEATURE_COLUMNS)
    assert feature_columns_for("with_odds", "league_none") == list(
        ALL_FEATURE_COLUMNS
    )


def test_x_metrics_use_top2_mean_actual_and_class_brier():
    y_true = np.array([1, 0, 2, 1])
    proba = np.array(
        [
            [0.32, 0.36, 0.32],  # X rank 1
            [0.60, 0.25, 0.15],  # X rank 2
            [0.45, 0.15, 0.40],  # X rank 3
            [0.40, 0.30, 0.30],  # X tied rank 2
        ]
    )

    metrics = metric_row(y_true, proba)
    assert metrics["X_top2_rate"] == pytest.approx(0.75)
    assert metrics["X_mean_prob"] == pytest.approx(0.265)
    assert metrics["X_actual_rate"] == pytest.approx(0.5)

    expected_brier = np.mean((proba[:, 1] - np.array([1, 0, 0, 1])) ** 2)
    assert metrics["X_brier"] == pytest.approx(expected_brier)


def test_x_top2_counts_tie_for_second_place():
    proba = np.array(
        [
            [0.40, 0.30, 0.30],
            [0.50, 0.20, 0.30],
        ]
    )
    assert x_top2_mask(proba).tolist() == [True, False]


def _prediction_fixture() -> pd.DataFrame:
    rows = []
    outcomes = [0, 1, 2, 0, 1, 2, 0, 1]

    for index, y_true in enumerate(outcomes):
        league = "E0" if index < 4 else "E1"
        odds = np.array([0.45, 0.30, 0.25])
        if y_true == 1:
            odds = np.array([0.35, 0.35, 0.30])
        elif y_true == 2:
            odds = np.array([0.35, 0.30, 0.35])

        row = {
            "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
            "League": league,
            "Season": "2526",
            "y_true": y_true,
            "fold": 1,
            "train_N": 100,
            "test_N": len(outcomes),
            "paired_N": len(outcomes),
            "odds_H": odds[0],
            "odds_D": odds[1],
            "odds_A": odds[2],
        }

        for label in ALL_VARIANT_LABELS:
            if label == "Odds":
                continue
            spec = label.split("/")
            representation = spec[1]

            if representation == "league_none":
                proba = np.array([0.40, 0.30, 0.30])
            elif representation == "league_ordinal":
                proba = odds.copy()
                proba[y_true] += 0.05
                proba /= proba.sum()
            else:
                proba = odds.copy()
                proba[y_true] += 0.10
                proba /= proba.sum()

            prefix = VariantSpec(spec[0], representation).prefix
            row[f"{prefix}_H"] = proba[0]
            row[f"{prefix}_D"] = proba[1]
            row[f"{prefix}_A"] = proba[2]

        rows.append(row)

    return pd.DataFrame(rows)


def test_prediction_frame_has_complete_paired_variants():
    predictions = _prediction_fixture()
    validate_prediction_frame(predictions)

    for label in ALL_VARIANT_LABELS:
        assert all(column in predictions.columns for column in probability_columns(label))


def test_summary_has_identical_n_and_direct_representation_deltas():
    predictions = _prediction_fixture()
    table = build_group_table(
        predictions,
        ["League"],
        bootstrap=50,
        seed=7,
    )

    for _, group in table.groupby("League"):
        assert group["N"].nunique() == 1
        assert set(group["Variant"]) == set(ALL_VARIANT_LABELS)

    ordinal = table[
        table["Variant"] == "base/league_ordinal"
    ]
    onehot = table[
        table["Variant"] == "base/league_onehot"
    ]

    assert (ordinal["Delta_LogLoss_vs_None"] < 0).all()
    assert (ordinal["Delta_Brier_vs_None"] < 0).all()
    assert (onehot["Delta_LogLoss_vs_Ordinal"] < 0).all()
    assert (onehot["Delta_Brier_vs_Ordinal"] < 0).all()


def test_x_calibration_has_fixed_bins_for_every_variant_and_scope():
    predictions = _prediction_fixture()
    calibration = build_x_calibration_table(predictions)

    overall = calibration[calibration["Scope"] == "overall"]
    assert set(overall["Variant"]) == set(ALL_VARIANT_LABELS)
    assert set(overall["Bin"]) == {
        "0.00-0.20",
        "0.20-0.25",
        "0.25-0.30",
        "0.30+",
    }

    counts = overall.groupby("Variant")["N"].sum()
    assert (counts == len(predictions)).all()


def _report_tables(bootstrap: int = 50, seed: int = 7):
    predictions = _prediction_fixture()
    return predictions, {
        "overall_table": build_group_table(
            predictions, [], bootstrap=bootstrap, seed=seed
        ),
        "fold_table": build_fold_table(
            predictions, bootstrap=bootstrap, seed=seed
        ),
        "league_table": build_group_table(
            predictions, ["League"], bootstrap=bootstrap, seed=seed
        ),
        "season_table": build_group_table(
            predictions, ["League", "Season"], bootstrap=bootstrap, seed=seed
        ),
        "calibration_table": build_x_calibration_table(predictions),
    }


def test_overall_direct_table_covers_both_feature_sets_and_comparisons():
    _, tables = _report_tables()
    direct = build_overall_direct_table(tables["overall_table"])

    assert set(direct["FeatureSet"]) == {"base", "with_odds"}
    for feature_set in ("base", "with_odds"):
        rows = direct[direct["FeatureSet"] == feature_set]
        pairs = set(zip(rows["Candidate"], rows["Reference"]))
        assert pairs == {
            ("league_ordinal", "league_none"),
            ("league_onehot", "league_none"),
            ("league_onehot", "league_ordinal"),
        }

    assert list(direct.columns) == list(OVERALL_DIRECT_COLUMNS)
    assert direct[list(OVERALL_DIRECT_COLUMNS)].notna().all().all()


def test_overall_direct_values_come_from_the_overall_table():
    _, tables = _report_tables()
    overall = tables["overall_table"].set_index("Variant")
    direct = build_overall_direct_table(tables["overall_table"])

    row = direct[
        (direct["FeatureSet"] == "base")
        & (direct["Candidate"] == "league_onehot")
        & (direct["Reference"] == "league_ordinal")
    ].iloc[0]
    source = overall.loc["base/league_onehot"]

    assert row["Delta_LogLoss"] == pytest.approx(
        source["Delta_LogLoss_vs_Ordinal"]
    )
    assert row["Delta_LogLoss_CI95_L"] == pytest.approx(
        source["Delta_LogLoss_vs_Ordinal_CI95_L"]
    )
    assert row["Delta_LogLoss_CI95_U"] == pytest.approx(
        source["Delta_LogLoss_vs_Ordinal_CI95_U"]
    )
    assert row["Delta_Brier"] == pytest.approx(source["Delta_Brier_vs_Ordinal"])
    assert row["Delta_Brier_CI95_L"] == pytest.approx(
        source["Delta_Brier_vs_Ordinal_CI95_L"]
    )
    assert row["Delta_Brier_CI95_U"] == pytest.approx(
        source["Delta_Brier_vs_Ordinal_CI95_U"]
    )


def test_report_contains_overall_direct_section_conclusion_and_guardrail():
    predictions, tables = _report_tables()
    report = render_report(
        predictions,
        **tables,
        bootstrap=50,
        strict_sample=True,
    )

    assert "Overall direct League-representation comparisons" in report
    for column in OVERALL_DIRECT_COLUMNS:
        assert column in report

    section = report.split(
        "## Overall direct League-representation comparisons"
    )[1].split("## Per-fold metrics")[0]
    for feature_set in ("base", "with_odds"):
        for candidate, reference in (
            ("league_ordinal", "league_none"),
            ("league_onehot", "league_none"),
            ("league_onehot", "league_ordinal"),
        ):
            assert (
                f"| {feature_set} | {candidate} | {reference} |" in section
            )

    assert (
        "No active League representation significantly improves on "
        "`league_none` overall." in section
    )
    assert (
        "One-hot is not significantly different from ordinal overall."
        in section
    )
    assert (
        "Every relevant overall paired 95% confidence interval overlaps 0."
        in section
    )
    assert "No League representation can be declared a winner." in section
    assert (
        "No production change is recommended based on this diagnostic."
        in section
    )

    assert (
        "Per-league and per-season confidence intervals are exploratory and "
        "are not adjusted for multiple comparisons. An isolated subgroup "
        "result must not override the overall paired comparison." in report
    )


def test_save_outputs_writes_overall_csv_with_all_sources_and_deltas(tmp_path):
    _, tables = _report_tables()
    outputs = save_outputs(
        "report",
        tables["overall_table"],
        tables["fold_table"],
        tables["league_table"],
        tables["season_table"],
        tables["calibration_table"],
        report_path=tmp_path / "RESULTS_LEAGUE_REPRESENTATION.md",
    )

    overall_path = tmp_path / "RESULTS_LEAGUE_REPRESENTATION_OVERALL.csv"
    assert overall_path in outputs
    assert overall_path.exists()

    saved = pd.read_csv(overall_path)
    assert set(saved["Variant"]) == set(ALL_VARIANT_LABELS)
    for column in (
        "FeatureSet",
        "LeagueRepresentation",
        "N",
        "Accuracy",
        "LogLoss",
        "Brier",
        "X_top2_rate",
        "X_mean_prob",
        "X_actual_rate",
        "X_brier",
    ):
        assert column in saved.columns
    for reference in ("Odds", "None", "Ordinal"):
        for metric in ("LogLoss", "Brier"):
            assert f"Delta_{metric}_vs_{reference}" in saved.columns
            assert f"Delta_{metric}_vs_{reference}_CI95_L" in saved.columns
            assert f"Delta_{metric}_vs_{reference}_CI95_U" in saved.columns


def test_main_allows_refresh_while_retaining_strict_sample_validation(
    monkeypatch, tmp_path
):
    args = type(
        "Args",
        (),
        {
            "refresh_data": True,
            "allow_sample_drift": False,
            "segments": 4,
            "bootstrap": 0,
            "seed": 42,
            "output": tmp_path / "RESULTS_LEAGUE_REPRESENTATION.md",
        },
    )()

    loaded = pd.DataFrame({"sentinel": [1]})
    predictions = _prediction_fixture()
    validation_calls = []

    monkeypatch.setattr(module_under_test, "parse_args", lambda: args)
    monkeypatch.setattr(
        module_under_test,
        "load_data",
        lambda refresh=False: loaded if refresh else pd.DataFrame(),
    )
    monkeypatch.setattr(
        module_under_test,
        "run_diagnostic",
        lambda df, n_segments=4: predictions,
    )
    monkeypatch.setattr(
        module_under_test,
        "validate_pr43_sample",
        lambda frame: validation_calls.append(len(frame)),
    )

    simple_table = pd.DataFrame({"Variant": ["Odds"], "N": [len(predictions)]})
    calibration = pd.DataFrame(
        {
            "Scope": ["overall"],
            "League": ["ALL"],
            "Variant": ["Odds"],
            "Bin": ["0.30+"],
            "N": [len(predictions)],
            "MeanPredictedX": [0.3],
            "ObservedXRate": [0.25],
        }
    )
    monkeypatch.setattr(
        module_under_test,
        "build_group_table",
        lambda *args, **kwargs: simple_table.copy(),
    )
    monkeypatch.setattr(
        module_under_test,
        "build_fold_table",
        lambda *args, **kwargs: simple_table.copy(),
    )
    monkeypatch.setattr(
        module_under_test,
        "build_x_calibration_table",
        lambda *args, **kwargs: calibration.copy(),
    )
    monkeypatch.setattr(module_under_test, "render_report", lambda *a, **k: "ok")
    monkeypatch.setattr(
        module_under_test,
        "save_outputs",
        lambda *a, **k: (args.output,),
    )

    assert module_under_test.main() == 0
    assert validation_calls == [len(predictions)]

