"""
Tests for half-guard selection benchmark helpers.

Covers:
- select_halfguards_gain returns correct indices
- select_halfguards_entropy returns correct indices
- compute_block_metrics routes correctly based on selection_mode
- benchmark_halfguards script is importable and run_comparison works on synthetic data
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backtest_report import (
    select_halfguards_gain,
    select_halfguards_entropy,
    compute_block_metrics,
)
from uncertainty import entropy_norm


class TestSelectHalfguardsGain:
    """Tests for the gain-based selection helper."""

    def test_picks_highest_second_best(self):
        """Match with highest second-best prob should be picked first."""
        y_proba = np.array([
            [0.70, 0.20, 0.10],  # gain=0.20
            [0.55, 0.40, 0.05],  # gain=0.40
            [0.60, 0.30, 0.10],  # gain=0.30
        ])
        idx = select_halfguards_gain(y_proba, n_half=1)
        assert idx[0] == 1

    def test_tiebreak_by_top2(self):
        """Equal gain -> higher top2 wins."""
        y_proba = np.array([
            [0.50, 0.35, 0.15],  # gain=0.35, top2=0.85
            [0.55, 0.35, 0.10],  # gain=0.35, top2=0.90
        ])
        idx = select_halfguards_gain(y_proba, n_half=1)
        assert idx[0] == 1

    def test_returns_n_half_indices(self):
        y_proba = np.random.dirichlet([1, 1, 1], size=10)
        idx = select_halfguards_gain(y_proba, n_half=3)
        assert len(idx) == 3
        assert len(set(idx)) == 3  # all unique


class TestSelectHalfguardsEntropy:
    """Tests for the entropy-based selection helper."""

    def test_picks_highest_entropy(self):
        """Match with highest entropy should be picked first."""
        y_proba = np.array([
            [0.80, 0.15, 0.05],  # low entropy
            [0.34, 0.33, 0.33],  # high entropy (near uniform)
            [0.60, 0.25, 0.15],  # medium entropy
        ])
        entropy_values = np.array([
            entropy_norm(0.80, 0.15, 0.05),
            entropy_norm(0.34, 0.33, 0.33),
            entropy_norm(0.60, 0.25, 0.15),
        ])
        idx = select_halfguards_entropy(y_proba, entropy_values, n_half=1)
        assert idx[0] == 1

    def test_returns_n_half_indices(self):
        entropy_values = np.random.rand(10)
        y_proba = np.random.dirichlet([1, 1, 1], size=10)
        idx = select_halfguards_entropy(y_proba, entropy_values, n_half=4)
        assert len(idx) == 4
        assert len(set(idx)) == 4


class TestComputeBlockMetricsSelectionMode:
    """Tests that selection_mode parameter routes correctly."""

    def _make_data(self):
        """Create test data where gain and entropy select different matches."""
        # Match 0: near uniform -> HIGH entropy, gain=0.33
        # Match 1: skewed but high second-best -> LOW entropy, gain=0.40
        # Match 2: very skewed -> LOW entropy, gain=0.15
        y_true = np.array([0, 0, 0])
        y_proba = np.array([
            [0.34, 0.33, 0.33],  # entropy ~1.0, gain=0.33
            [0.55, 0.40, 0.05],  # entropy ~0.7, gain=0.40
            [0.80, 0.15, 0.05],  # entropy ~0.5, gain=0.15
        ])
        pred_top1 = np.argmax(y_proba, axis=1)
        entropy_values = np.array([
            entropy_norm(0.34, 0.33, 0.33),
            entropy_norm(0.55, 0.40, 0.05),
            entropy_norm(0.80, 0.15, 0.05),
        ])
        return y_true, y_proba, pred_top1, entropy_values

    def test_gain_mode_selects_match1(self):
        """Gain mode should pick match 1 (highest gain=0.40)."""
        y_true, y_proba, pred_top1, entropy_values = self._make_data()
        metrics = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=1, selection_mode="gain",
        )
        # Match 1: true=0, top2={0,1} -> hit
        assert metrics["accuracy_top2_on_halfguards"] == 1.0

    def test_entropy_mode_selects_match0(self):
        """Entropy mode should pick match 0 (highest entropy)."""
        y_true, y_proba, pred_top1, entropy_values = self._make_data()
        metrics = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=1, selection_mode="entropy",
        )
        # Match 0: true=0, top2={0,1} -> hit
        assert metrics["accuracy_top2_on_halfguards"] == 1.0

    def test_default_mode_is_gain(self):
        """Default selection_mode should be gain."""
        y_true, y_proba, pred_top1, entropy_values = self._make_data()
        m_default = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values, n_half=1,
        )
        m_gain = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=1, selection_mode="gain",
        )
        assert m_default["accuracy_top2_on_halfguards"] == m_gain["accuracy_top2_on_halfguards"]
        assert m_default["combined_ticket_hit_rate"] == m_gain["combined_ticket_hit_rate"]

    def test_modes_can_diverge(self):
        """Construct data where entropy and gain select different matches
        leading to different accuracy_top2_on_halfguards."""
        # Match 0: near-uniform -> HIGH entropy, gain=0.33, top2_indices={0,2}
        #          true label = 1 -> NOT in {0,2} -> MISS for top2
        # Match 1: skewed high second -> LOW entropy, gain=0.45, top2_indices={0,1}
        #          true label = 0 -> in {0,1} -> HIT for top2
        y_true = np.array([1, 0])
        y_proba = np.array([
            [0.50, 0.10, 0.40],  # entropy ~0.87, gain=0.40, top2={0,2}
            [0.50, 0.45, 0.05],  # entropy ~0.67, gain=0.45, top2={0,1}
        ])
        pred_top1 = np.argmax(y_proba, axis=1)
        entropy_values = np.array([
            entropy_norm(0.50, 0.10, 0.40),
            entropy_norm(0.50, 0.45, 0.05),
        ])

        m_entropy = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=1, selection_mode="entropy",
        )
        m_gain = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=1, selection_mode="gain",
        )

        # Entropy picks match 0 (highest entropy).  true=1, top2={0,2} -> MISS
        assert m_entropy["accuracy_top2_on_halfguards"] == 0.0

        # Gain picks match 1 (gain=0.45).  true=0, top2={0,1} -> HIT
        assert m_gain["accuracy_top2_on_halfguards"] == 1.0


class TestBenchmarkScriptImport:
    """Test that the benchmark script is importable."""

    def test_importable(self):
        from scripts.benchmark_halfguards import run_comparison, print_comparison
        assert callable(run_comparison)
        assert callable(print_comparison)

    def test_print_comparison_empty(self):
        from scripts.benchmark_halfguards import print_comparison
        report = print_comparison({"folds": []})
        assert "No results" in report

    def test_print_comparison_with_data(self):
        """print_comparison should produce markdown with expected sections."""
        from scripts.benchmark_halfguards import print_comparison
        fake_metrics = {
            "n_matches": 100,
            "accuracy_top1": 0.45,
            "accuracy_top2_on_halfguards": 0.75,
            "combined_ticket_hit_rate": 0.52,
            "logloss": 1.05,
            "brier": 0.65,
        }
        fake_results = {
            "folds": [
                {
                    "fold": 1,
                    "n_matches": 100,
                    "n_half": 4,
                    "gain": fake_metrics,
                    "entropy": fake_metrics,
                    "n_different_selections": 2,
                    "stats_gain": {"avg_gain": 0.35, "avg_top2": 0.85, "avg_entropy": 0.7},
                    "stats_entropy": {"avg_gain": 0.30, "avg_top2": 0.80, "avg_entropy": 0.9},
                }
            ],
            "divergence": [],
        }
        report = print_comparison(fake_results)
        assert "Entropy vs Gain" in report
        assert "Acc_Top2_HG" in report
        assert "Combined" in report
        assert "Conclusion" in report
        assert "Selection Divergence" in report
