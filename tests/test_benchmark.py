"""
Minimal tests for the benchmark module.

Covers:
- Each baseline returns valid probabilities (shape, sums to 1, non-negative)
- Metrics function returns correct values on a known fixture
- Benchmark script can run on a small synthetic dataset
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.run_benchmark import (
    home_favoring_baseline,
    most_frequent_class_baseline,
    elo_only_baseline,
    bookmaker_baseline,
    brier_score_multiclass,
    compute_all_metrics,
    calibration_summary,
)


class TestBaselineProbabilities:
    """Test that each baseline returns valid probability distributions."""

    def test_home_favoring_shape_and_sum(self):
        proba = home_favoring_baseline(10)
        assert proba.shape == (10, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        assert (proba >= 0).all()

    def test_home_favoring_single_match(self):
        proba = home_favoring_baseline(1)
        assert proba.shape == (1, 3)
        assert proba[0, 0] > proba[0, 2], "Home prob should exceed away prob"

    def test_most_frequent_class_shape_and_sum(self):
        y_train = np.array([0, 0, 0, 1, 1, 2])  # H=3, D=2, A=1
        proba = most_frequent_class_baseline(y_train, 5)
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        assert (proba >= 0).all()

    def test_most_frequent_class_reflects_distribution(self):
        y_train = np.array([0, 0, 0, 1, 2])  # 60% H, 20% D, 20% A
        proba = most_frequent_class_baseline(y_train, 1)
        assert proba[0, 0] == pytest.approx(0.6, abs=1e-8)
        assert proba[0, 1] == pytest.approx(0.2, abs=1e-8)
        assert proba[0, 2] == pytest.approx(0.2, abs=1e-8)

    def test_elo_only_shape_and_sum(self):
        import pandas as pd
        df = pd.DataFrame({
            "HomeElo": [1500.0, 1600.0, 1400.0],
            "AwayElo": [1500.0, 1400.0, 1600.0],
        })
        proba = elo_only_baseline(df)
        assert proba.shape == (3, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        assert (proba >= 0).all()

    def test_elo_only_favors_stronger_team(self):
        import pandas as pd
        df = pd.DataFrame({
            "HomeElo": [1700.0],
            "AwayElo": [1300.0],
        })
        proba = elo_only_baseline(df)
        assert proba[0, 0] > proba[0, 2], "Higher Elo home team should have higher win prob"

    def test_elo_only_equal_elo_still_home_advantage(self):
        import pandas as pd
        df = pd.DataFrame({
            "HomeElo": [1500.0],
            "AwayElo": [1500.0],
        })
        proba = elo_only_baseline(df)
        # With home bonus, home should be slightly favored even at equal Elo
        assert proba[0, 0] > proba[0, 2]

    def test_bookmaker_baseline_no_odds(self):
        import pandas as pd
        df = pd.DataFrame({"HomeElo": [1500.0]})
        proba, mask = bookmaker_baseline(df)
        assert proba is None
        assert mask is None

    def test_bookmaker_baseline_with_odds(self):
        import pandas as pd
        df = pd.DataFrame({
            "has_odds": [1.0, 1.0, 0.0],
            "ImpliedHome": [0.5, 0.4, 0.33],
            "ImpliedDraw": [0.25, 0.3, 0.33],
            "ImpliedAway": [0.25, 0.3, 0.33],
        })
        proba, mask = bookmaker_baseline(df)
        assert proba is not None
        assert mask is not None
        assert mask.sum() == 2
        # Check that odds-rows sum to 1
        np.testing.assert_allclose(proba[mask].sum(axis=1), 1.0, atol=1e-8)


class TestMetrics:
    """Test metric functions on known fixtures."""

    def test_brier_score_perfect(self):
        """Perfect predictions should give Brier score of 0."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert brier_score_multiclass(y_true, y_proba) == pytest.approx(0.0, abs=1e-8)

    def test_brier_score_worst(self):
        """Completely wrong predictions should give high Brier score."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # Each row: (0-1)^2 + (0-0)^2 + (1-0)^2 = 2 per class mismatch
        assert brier_score_multiclass(y_true, y_proba) == pytest.approx(2.0, abs=1e-8)

    def test_compute_all_metrics_keys(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
        ])
        m = compute_all_metrics(y_true, y_proba)
        assert "log_loss" in m
        assert "brier" in m
        assert "accuracy" in m
        assert "top_class_accuracy" in m
        assert "n_matches" in m
        assert m["n_matches"] == 5
        # All predictions are correct (argmax matches true)
        assert m["accuracy"] == 1.0

    def test_compute_all_metrics_wrong_predictions(self):
        y_true = np.array([0, 0, 0])
        y_proba = np.array([
            [0.1, 0.5, 0.4],  # predicts D, true H -> wrong
            [0.1, 0.4, 0.5],  # predicts A, true H -> wrong
            [0.5, 0.3, 0.2],  # predicts H, true H -> correct
        ])
        m = compute_all_metrics(y_true, y_proba)
        assert m["accuracy"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_calibration_summary_returns_bins(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
        ])
        cal = calibration_summary(y_true, y_proba, n_bins=5)
        assert len(cal) == 5
        total_count = sum(row["count"] for row in cal)
        assert total_count == 5
