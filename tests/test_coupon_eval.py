"""
Tests for coupon-style half-guard evaluation helpers.

Covers:
- build_coupons produces correct groups
- evaluate_coupon computes hit/rescue correctly
- run_coupon_eval_on_fold integrates per-coupon logic
- aggregate_coupon_results produces expected structure
- export_coupon_csv writes valid CSV
- generate_report produces markdown with expected sections
- _verdict helper returns correct labels
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.benchmark_coupon_eval import (  # noqa: E402
    build_coupons,
    evaluate_coupon,
    run_coupon_eval_on_fold,
    aggregate_coupon_results,
    export_coupon_csv,
    generate_report,
    _verdict,
)
from uncertainty import entropy_norm  # noqa: E402


class TestBuildCoupons:
    """Tests for build_coupons."""

    def test_exact_division(self):
        coupons = build_coupons(16, coupon_size=8)
        assert len(coupons) == 2
        assert len(coupons[0]) == 8
        assert len(coupons[1]) == 8
        np.testing.assert_array_equal(coupons[0], np.arange(8))
        np.testing.assert_array_equal(coupons[1], np.arange(8, 16))

    def test_drops_incomplete_tail(self):
        coupons = build_coupons(19, coupon_size=8)
        assert len(coupons) == 2  # 19 // 8 = 2, last 3 dropped
        assert len(coupons[0]) == 8
        assert len(coupons[1]) == 8

    def test_too_few_matches(self):
        coupons = build_coupons(5, coupon_size=8)
        assert len(coupons) == 0

    def test_single_coupon(self):
        coupons = build_coupons(8, coupon_size=8)
        assert len(coupons) == 1
        np.testing.assert_array_equal(coupons[0], np.arange(8))

    def test_zero_matches(self):
        coupons = build_coupons(0, coupon_size=8)
        assert len(coupons) == 0


class TestEvaluateCoupon:
    """Tests for evaluate_coupon."""

    def _make_data(self):
        """Create simple test data: 4 matches."""
        y_true = np.array([0, 1, 2, 0])  # H, D, A, H
        pred_top1 = np.array([0, 0, 2, 0])  # correct: 0, 2, 3
        # top2 for each match (argsort desc, take top 2)
        y_proba = np.array([
            [0.60, 0.25, 0.15],  # top2={0,1}
            [0.50, 0.30, 0.20],  # top2={0,1}
            [0.20, 0.30, 0.50],  # top2={2,1}
            [0.55, 0.30, 0.15],  # top2={0,1}
        ])
        top2_preds = np.argsort(y_proba, axis=1)[:, -2:]
        return y_true, pred_top1, top2_preds

    def test_all_singles_all_correct(self):
        """Coupon with no HG, all singles correct."""
        y_true = np.array([0, 2, 0])
        pred_top1 = np.array([0, 2, 0])
        top2_preds = np.array([[0, 1], [1, 2], [0, 1]])
        coupon = np.array([0, 1, 2])

        result = evaluate_coupon(coupon, y_true, pred_top1, top2_preds, set())
        assert result["hit"] is True
        assert result["n_correct"] == 3
        assert result["hg_rescued"] is False

    def test_single_miss_coupon_fails(self):
        """One wrong single -> coupon fails."""
        y_true, pred_top1, top2_preds = self._make_data()
        coupon = np.array([0, 1, 2, 3])
        # Match 1: true=1, pred=0 -> miss
        result = evaluate_coupon(
            coupon, y_true, pred_top1, top2_preds, set()
        )
        assert result["hit"] is False
        assert result["n_correct"] == 3

    def test_halfguard_saves_match(self):
        """HG on match 1 -> top2 includes true label -> coupon hits."""
        y_true, pred_top1, top2_preds = self._make_data()
        coupon = np.array([0, 1, 2, 3])
        # Match 1: true=1, top2={0,1} -> hit with HG
        result = evaluate_coupon(
            coupon, y_true, pred_top1, top2_preds, {1}
        )
        assert result["hit"] is True
        assert result["n_correct"] == 4
        assert result["hg_rescued"] is True

    def test_rescue_flag_false_when_singles_also_pass(self):
        """If singles would also pass, rescue=False."""
        y_true = np.array([0, 2])
        pred_top1 = np.array([0, 2])
        top2_preds = np.array([[0, 1], [1, 2]])
        coupon = np.array([0, 1])

        result = evaluate_coupon(
            coupon, y_true, pred_top1, top2_preds, {1}
        )
        assert result["hit"] is True
        assert result["hg_rescued"] is False  # singles would also pass

    def test_halfguard_miss_coupon_fails(self):
        """HG match misses -> coupon fails."""
        y_true = np.array([2, 0])  # true: A, H
        pred_top1 = np.array([0, 0])  # pred: H, H
        # top2 for match 0: {0,1} -> true=2 NOT in {0,1} -> miss
        top2_preds = np.array([[0, 1], [0, 1]])
        coupon = np.array([0, 1])

        result = evaluate_coupon(
            coupon, y_true, pred_top1, top2_preds, {0}
        )
        assert result["hit"] is False


class TestRunCouponEvalOnFold:
    """Tests for run_coupon_eval_on_fold with synthetic data."""

    def _make_fold_data(self, n=24):
        """Create synthetic fold data."""
        np.random.seed(42)
        y_proba = np.random.dirichlet([2, 1, 1], size=n)
        y_true = np.array([
            np.random.choice([0, 1, 2], p=p) for p in y_proba
        ])
        pred_top1 = np.argmax(y_proba, axis=1)
        entropy_values = np.array([
            entropy_norm(p[0], p[1], p[2]) for p in y_proba
        ])
        return y_true, y_proba, pred_top1, entropy_values

    def test_returns_dict_with_expected_keys(self):
        y_true, y_proba, pred_top1, entropy_values = self._make_fold_data()
        result = run_coupon_eval_on_fold(
            y_true, y_proba, pred_top1, entropy_values,
            coupon_size=8, n_half=2,
        )
        assert result is not None
        assert "gain_ticket_hit_rate" in result
        assert "entropy_ticket_hit_rate" in result
        assert "gain_rescues" in result
        assert "entropy_rescues" in result
        assert "n_divergent_coupons" in result
        assert "n_coupons" in result

    def test_n_coupons_matches_expected(self):
        y_true, y_proba, pred_top1, entropy_values = self._make_fold_data(24)
        result = run_coupon_eval_on_fold(
            y_true, y_proba, pred_top1, entropy_values,
            coupon_size=8, n_half=2,
        )
        assert result["n_coupons"] == 3  # 24 // 8

    def test_rates_between_0_and_1(self):
        y_true, y_proba, pred_top1, entropy_values = self._make_fold_data(40)
        result = run_coupon_eval_on_fold(
            y_true, y_proba, pred_top1, entropy_values,
            coupon_size=8, n_half=2,
        )
        assert 0 <= result["gain_ticket_hit_rate"] <= 1
        assert 0 <= result["entropy_ticket_hit_rate"] <= 1
        assert 0 <= result["divergence_rate"] <= 1

    def test_returns_none_for_too_few_matches(self):
        y_true, y_proba, pred_top1, entropy_values = self._make_fold_data(4)
        result = run_coupon_eval_on_fold(
            y_true, y_proba, pred_top1, entropy_values,
            coupon_size=8, n_half=2,
        )
        assert result is None

    def test_n_half_capped_by_coupon_size(self):
        y_true, y_proba, pred_top1, entropy_values = self._make_fold_data(16)
        result = run_coupon_eval_on_fold(
            y_true, y_proba, pred_top1, entropy_values,
            coupon_size=8, n_half=6,
        )
        assert result is not None
        assert result["n_half_effective"] == 4  # min(6, 8//2)


class TestAggregateCouponResults:
    """Tests for aggregate_coupon_results."""

    def _make_fake(self, fold=1, n_half=2, n_coupons=10,
                   gain_hit=0.3, entropy_hit=0.2):
        return {
            "fold": fold,
            "n_coupons": n_coupons,
            "coupon_size": 8,
            "n_half_requested": n_half,
            "n_half_effective": n_half,
            "n_test_matches": 80,
            "n_matches_evaluated": 80,
            "gain_ticket_hit_rate": gain_hit,
            "gain_ticket_hits": int(gain_hit * n_coupons),
            "gain_mean_correct": 5.5,
            "gain_median_correct": 5.0,
            "gain_rescues": 2,
            "gain_rescue_rate": 0.2,
            "entropy_ticket_hit_rate": entropy_hit,
            "entropy_ticket_hits": int(entropy_hit * n_coupons),
            "entropy_mean_correct": 5.3,
            "entropy_median_correct": 5.0,
            "entropy_rescues": 1,
            "entropy_rescue_rate": 0.1,
            "n_divergent_coupons": 3,
            "divergence_rate": 0.3,
        }

    def test_returns_expected_keys(self):
        results = [
            self._make_fake(fold=1, n_half=2),
            self._make_fake(fold=1, n_half=4),
            self._make_fake(fold=2, n_half=2),
        ]
        agg = aggregate_coupon_results(results)
        assert "total" in agg
        assert "by_n_half" in agg
        assert "by_fold" in agg

    def test_total_counts(self):
        results = [
            self._make_fake(fold=1, n_half=2, n_coupons=10),
            self._make_fake(fold=2, n_half=2, n_coupons=15),
        ]
        agg = aggregate_coupon_results(results)
        assert agg["total"]["n_coupons"] == 25

    def test_delta_sign(self):
        results = [
            self._make_fake(gain_hit=0.4, entropy_hit=0.2),
        ]
        agg = aggregate_coupon_results(results)
        assert agg["total"]["delta_ticket_hit_rate"] > 0

    def test_empty_input(self):
        assert aggregate_coupon_results([]) == {}


class TestVerdict:
    """Tests for _verdict helper."""

    def test_gain_wins(self):
        assert _verdict(0.05) == "gain"

    def test_entropy_wins(self):
        assert _verdict(-0.05) == "entropy"

    def test_tie(self):
        assert _verdict(0.0005) == "tie"
        assert _verdict(-0.0005) == "tie"
        assert _verdict(0.0) == "tie"


class TestExportCouponCsv:
    """Tests for export_coupon_csv."""

    def test_writes_valid_csv(self):
        results = [
            {
                "fold": 1, "coupon_size": 8,
                "n_half_requested": 2, "n_half_effective": 2,
                "n_coupons": 10, "n_test_matches": 80,
                "n_matches_evaluated": 80,
                "gain_ticket_hit_rate": 0.3, "gain_ticket_hits": 3,
                "gain_mean_correct": 5.5, "gain_median_correct": 5.0,
                "gain_rescues": 1, "gain_rescue_rate": 0.1,
                "entropy_ticket_hit_rate": 0.2, "entropy_ticket_hits": 2,
                "entropy_mean_correct": 5.3, "entropy_median_correct": 5.0,
                "entropy_rescues": 0, "entropy_rescue_rate": 0.0,
                "n_divergent_coupons": 3, "divergence_rate": 0.3,
            }
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="r"
        ) as tmp:
            path = Path(tmp.name)

        export_coupon_csv(results, path)

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert "gain_ticket_hit_rate" in reader.fieldnames
        assert "entropy_rescues" in reader.fieldnames
        assert "n_divergent_coupons" in reader.fieldnames
        path.unlink()

    def test_empty_input(self):
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as tmp:
            path = Path(tmp.name)
        export_coupon_csv([], path)
        path.unlink(missing_ok=True)


class TestGenerateReport:
    """Tests for generate_report."""

    def test_report_contains_sections(self):
        results = [
            {
                "fold": 1, "coupon_size": 8,
                "n_half_requested": 2, "n_half_effective": 2,
                "n_coupons": 10, "n_test_matches": 80,
                "n_matches_evaluated": 80,
                "gain_ticket_hit_rate": 0.3, "gain_ticket_hits": 3,
                "gain_mean_correct": 5.5, "gain_median_correct": 5.0,
                "gain_rescues": 2, "gain_rescue_rate": 0.2,
                "entropy_ticket_hit_rate": 0.2, "entropy_ticket_hits": 2,
                "entropy_mean_correct": 5.3, "entropy_median_correct": 5.0,
                "entropy_rescues": 1, "entropy_rescue_rate": 0.1,
                "n_divergent_coupons": 3, "divergence_rate": 0.3,
            }
        ]
        agg = aggregate_coupon_results(results)
        report = generate_report(results, agg, 8, [2], 3)

        assert "Coupon Evaluation" in report
        assert "Setup" in report
        assert "Overall Summary" in report
        assert "Ticket hit rate" in report
        assert "rescue" in report.lower()
        assert "Conclusion" in report
        assert "Verdict" in report

    def test_empty_report(self):
        report = generate_report([], {}, 8, [2], 3)
        assert "No results" in report


class TestImports:
    """Test that the coupon eval script is importable."""

    def test_importable(self):
        from scripts.benchmark_coupon_eval import (
            build_coupons,
            evaluate_coupon,
            run_coupon_eval_on_fold,
            run_full_evaluation,
            aggregate_coupon_results,
            export_coupon_csv,
            generate_report,
        )
        assert callable(build_coupons)
        assert callable(evaluate_coupon)
        assert callable(run_coupon_eval_on_fold)
        assert callable(run_full_evaluation)
        assert callable(aggregate_coupon_results)
        assert callable(export_coupon_csv)
        assert callable(generate_report)
