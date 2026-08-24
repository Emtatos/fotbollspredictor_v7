import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scripts.diagnose_halfguard_rule as module_under_test
import scripts.diagnose_league_representation as league_module
import scripts.diagnose_production_blend as blend_module
from scripts.diagnose_halfguard_rule import (
    COUPON_GUARDS,
    COUPON_SIZE,
    PROBABILITY_SOURCES,
    REFERENCE_TAU,
    TAUS,
    assign_coupons,
    build_metric_tables,
    combined_match,
    coupon_correct_counts,
    guard_selection_mask,
    halfguard_sign_threshold,
    removed_index_threshold,
    render_report,
    rule_metrics,
    save_outputs,
    verify_reference_tau_identity,
)
from scripts.diagnose_production_blend import production_blend_rows
from ui_utils import get_halfguard_sign_combined


def _probability_grid(n: int = 60, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.array([4.0, 3.0, 3.5]), size=n)
    return np.asarray(raw, dtype=float)


def _prediction_fixture(n: int = 60, seed: int = 5) -> pd.DataFrame:
    odds = _probability_grid(n, seed)
    model = _probability_grid(n, seed + 1)
    production = production_blend_rows(odds, model)

    rng = np.random.default_rng(seed + 2)
    outcomes = rng.integers(0, 3, size=n)

    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-16", periods=n, freq="D"),
            "League": ["E0", "E1"] * (n // 2),
            "Season": "2526",
            "HomeTeam": [f"H{index}" for index in range(n)],
            "AwayTeam": [f"A{index}" for index in range(n)],
            "FTR": [("H", "D", "A")[value] for value in outcomes],
            "y_true": outcomes,
            "fold": [1] * (n // 2) + [2] * (n - n // 2),
            "train_N": 500,
            "test_N": n // 2,
            "paired_N": n // 2,
            "max_date": "2026-05-24",
            "source_max_date": "2026-05-24",
            "source_row_N": 4072,
            "frozen_row_N": 4072,
            "excluded_post_max_date_N": 0,
        }
    )
    for prefix, matrix in (
        ("odds", odds),
        ("model", model),
        ("production", production),
    ):
        for index, sign in enumerate(("H", "D", "A")):
            frame[f"{prefix}_{sign}"] = matrix[:, index]
    return frame


def test_tau_one_is_identical_to_production_halfguard_rule():
    matrix = _probability_grid(400, seed=17)

    for probs in matrix:
        expected = get_halfguard_sign_combined(combined_match(probs))
        assert halfguard_sign_threshold(probs, REFERENCE_TAU) == expected


def test_identity_check_is_positionwise_and_aborts_on_any_deviation(
    monkeypatch,
):
    predictions = _prediction_fixture()
    identity = verify_reference_tau_identity(predictions)

    assert [row["Source"] for row in identity] == list(PROBABILITY_SOURCES)
    assert all(row["Matches_Checked"] == len(predictions) for row in identity)
    assert all(row["Mismatches"] == 0 for row in identity)

    def broken_sign(probs, tau):
        return "1X" if tau == REFERENCE_TAU else halfguard_sign_threshold(
            probs,
            tau,
        )

    monkeypatch.setattr(
        module_under_test,
        "halfguard_sign_threshold",
        broken_sign,
    )
    with pytest.raises(ValueError, match="not identical"):
        verify_reference_tau_identity(predictions)


def test_tau_zero_never_removes_x():
    matrix = _probability_grid(400, seed=23)

    for probs in matrix:
        assert "X" in halfguard_sign_threshold(probs, 0.0)
        assert removed_index_threshold(probs, 0.0) != 1


def test_rule_always_returns_exactly_two_signs():
    matrix = _probability_grid(200, seed=29)
    allowed = {"1X", "12", "X2"}

    for tau in TAUS:
        for probs in matrix:
            sign = halfguard_sign_threshold(probs, tau)
            assert len(sign) == 2
            assert sign in allowed


def test_x_and_non_x_hit_rates_sum_to_the_total_hit_rate():
    predictions = _prediction_fixture()
    outcomes = predictions["y_true"].to_numpy(dtype=int)
    matrix = blend_module.probability_matrix(predictions, "odds_only")

    for tau in TAUS:
        metrics = rule_metrics(outcomes, matrix, tau)
        assert metrics["N_X"] + metrics["N_nonX"] == metrics["N"]
        combined = (
            metrics["N_X"] * metrics["HitRate_X"]
            + metrics["N_nonX"] * metrics["HitRate_nonX"]
        )
        assert combined == pytest.approx(metrics["N"] * metrics["HitRate"])


def test_sample_control_is_unchanged_from_pr45():
    assert (
        module_under_test.validate_pr43_sample
        is league_module.validate_pr43_sample
    )
    assert (
        module_under_test.REFERENCE_CACHE_FILES
        is blend_module.REFERENCE_CACHE_FILES
    )
    assert module_under_test.DEFAULT_MAX_DATE == blend_module.DEFAULT_MAX_DATE
    assert module_under_test.run_diagnostic is blend_module.run_diagnostic
    assert (
        module_under_test.freeze_reference_window
        is blend_module.freeze_reference_window
    )


def test_taus_cover_the_requested_grid_with_current_rule_endpoint():
    assert TAUS == (0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 1.00)
    assert REFERENCE_TAU == 1.00


def test_coupons_are_complete_blocks_of_thirteen_within_a_fold():
    predictions = _prediction_fixture(n=60)
    coupons = assign_coupons(predictions)

    labelled = coupons[coupons != ""]
    assert set(labelled.value_counts().unique()) == {COUPON_SIZE}
    for coupon, group in predictions.loc[labelled.index].groupby(
        coupons[labelled.index]
    ):
        assert group["fold"].nunique() == 1
        assert group["Date"].is_monotonic_increasing, coupon


def test_secondary_selection_guards_seven_matches_per_coupon():
    predictions = _prediction_fixture(n=60)
    coupons = assign_coupons(predictions)
    matrix = blend_module.probability_matrix(predictions, "odds_only")

    mask = guard_selection_mask(predictions, matrix, "top7of13", coupons)
    guarded = pd.Series(mask, index=predictions.index)

    complete = coupons != ""
    assert not guarded[~complete].any()
    for _, group in guarded[complete].groupby(coupons[complete]):
        assert int(group.sum()) == COUPON_GUARDS


def test_coupon_correct_counts_are_bounded_by_the_coupon_size():
    predictions = _prediction_fixture(n=60)
    coupons = assign_coupons(predictions)
    outcomes = predictions["y_true"].to_numpy(dtype=int)
    matrix = blend_module.probability_matrix(predictions, "odds_only")
    guarded = guard_selection_mask(predictions, matrix, "all", coupons)

    counts = coupon_correct_counts(matrix, outcomes, coupons, guarded, 0.30)

    assert len(counts) == int((coupons != "").sum()) // COUPON_SIZE
    assert counts.min() >= 0
    assert counts.max() <= COUPON_SIZE


def test_tables_expose_every_source_selection_and_tau_with_paired_deltas():
    predictions = _prediction_fixture()
    overall, league, coupon = build_metric_tables(
        predictions,
        bootstrap=40,
        seed=3,
    )

    assert set(overall["Source"]) == set(PROBABILITY_SOURCES)
    assert set(overall["Selection"]) == {"all", "top7of13"}
    for _, group in overall.groupby(["Source", "Selection"]):
        assert [float(value) for value in group["Tau"]] == list(TAUS)
        reference = group[group["IsCurrentRule"]]
        assert len(reference) == 1
        assert float(reference["Delta_HitRate"].iloc[0]) == 0.0

    for column in (
        "Delta_HitRate_CI95_L",
        "Delta_HitRate_X_CI95_U",
        "Delta_HitRate_nonX_CI95_L",
    ):
        assert column in overall.columns
        assert column in league.columns

    assert set(league["League"]) == {"E0", "E1"}
    assert set(coupon["Correct"]) == set(range(COUPON_SIZE + 1))


def test_save_outputs_creates_all_four_requested_files(tmp_path):
    predictions = _prediction_fixture()
    overall, league, coupon = build_metric_tables(
        predictions,
        bootstrap=0,
        seed=3,
    )
    identity = verify_reference_tau_identity(predictions)
    report = render_report(
        predictions,
        overall,
        league,
        coupon,
        identity,
        bootstrap=0,
        strict_sample=True,
    )

    outputs = save_outputs(
        report,
        overall,
        league,
        coupon,
        report_path=tmp_path / "RESULTS_HALFGUARD_RULE.md",
    )

    assert {path.name for path in outputs} == {
        "RESULTS_HALFGUARD_RULE.md",
        "RESULTS_HALFGUARD_RULE_OVERALL.csv",
        "RESULTS_HALFGUARD_RULE_LEAGUE.csv",
        "RESULTS_HALFGUARD_RULE_COUPON.csv",
    }
    assert all(path.exists() for path in outputs)

    saved = pd.read_csv(tmp_path / "RESULTS_HALFGUARD_RULE_OVERALL.csv")
    assert len(saved) == len(TAUS) * len(PROBABILITY_SOURCES) * 2


def test_report_contains_method_sample_identity_and_guardrails():
    predictions = _prediction_fixture()
    overall, league, coupon = build_metric_tables(
        predictions,
        bootstrap=0,
        seed=3,
    )
    identity = verify_reference_tau_identity(predictions)

    report = render_report(
        predictions,
        overall,
        league,
        coupon,
        identity,
        bootstrap=0,
        strict_sample=True,
    )

    for expected in (
        "Identity check for tau = 1.00",
        "Negative delta means worse than the current rule",
        "Strict PR #43 sample parity enforced: True",
        "Per-league confidence intervals are exploratory",
        "not adjusted for threshold selection",
        "Synthetic coupons are formed from 13 chronologically adjacent",
        "Streck is not used in this diagnostic",
        "No tau is declared a winner if the total hit rate falls",
    ):
        assert expected in report


def test_main_keeps_strict_sample_validation_and_identity_check(
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
            "output": tmp_path / "RESULTS_HALFGUARD_RULE.md",
        },
    )()

    predictions = _prediction_fixture()
    call_order: list[str] = []

    monkeypatch.setattr(module_under_test, "parse_args", lambda: args)
    monkeypatch.setattr(
        module_under_test,
        "load_reference_data",
        lambda *, refresh=False: (
            call_order.append("load") or pd.DataFrame({"sentinel": [1]})
        ),
    )
    monkeypatch.setattr(
        module_under_test,
        "freeze_reference_window",
        lambda df, max_date: (
            call_order.append("freeze") or df,
            {
                "max_date": "2026-05-24",
                "source_max_date": "2026-05-24",
                "source_row_N": len(df),
                "frozen_row_N": len(df),
                "excluded_post_max_date_N": 0,
            },
        ),
    )
    monkeypatch.setattr(
        module_under_test,
        "run_diagnostic",
        lambda df, n_segments=4, freeze_metadata=None: (
            call_order.append("run") or predictions
        ),
    )
    monkeypatch.setattr(
        module_under_test,
        "validate_pr43_sample",
        lambda frame: call_order.append("sample"),
    )

    assert module_under_test.main() == 0
    assert call_order == ["load", "freeze", "run", "sample"]
    assert (tmp_path / "RESULTS_HALFGUARD_RULE.md").exists()
    assert (tmp_path / "RESULTS_HALFGUARD_RULE_OVERALL.csv").exists()
    assert (tmp_path / "RESULTS_HALFGUARD_RULE_LEAGUE.csv").exists()
    assert (tmp_path / "RESULTS_HALFGUARD_RULE_COUPON.csv").exists()
