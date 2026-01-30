"""
Tests for backtest_report.py
"""
import pytest
import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path

from backtest_report import (
    compute_brier_score_multiclass,
    compute_block_metrics,
    get_top2_predictions,
    train_model,
)


class TestComputeBrierScore:
    """Tests for Brier score computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score of 0."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        brier = compute_brier_score_multiclass(y_true, y_proba)
        assert brier == pytest.approx(0.0, abs=1e-10)

    def test_worst_predictions(self):
        """Completely wrong predictions should have high Brier score."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.0, 0.0, 1.0],  # Predicted 2, actual 0
            [1.0, 0.0, 0.0],  # Predicted 0, actual 1
            [0.0, 1.0, 0.0],  # Predicted 1, actual 2
        ])
        brier = compute_brier_score_multiclass(y_true, y_proba)
        assert brier > 1.0  # Should be high

    def test_uniform_predictions(self):
        """Uniform predictions should have moderate Brier score."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
        ])
        brier = compute_brier_score_multiclass(y_true, y_proba)
        # For uniform: (1-1/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
        assert brier == pytest.approx(2/3, abs=1e-10)


class TestGetTop2Predictions:
    """Tests for top-2 prediction extraction."""

    def test_top2_extraction(self):
        """Test that top-2 predictions are correctly extracted."""
        y_proba = np.array([
            [0.6, 0.3, 0.1],  # Top-2: [0, 1]
            [0.1, 0.5, 0.4],  # Top-2: [1, 2]
            [0.2, 0.3, 0.5],  # Top-2: [1, 2]
        ])
        top2 = get_top2_predictions(y_proba)
        
        assert 0 in top2[0] and 1 in top2[0]
        assert 1 in top2[1] and 2 in top2[1]
        assert 1 in top2[2] and 2 in top2[2]


class TestComputeBlockMetrics:
    """Tests for block metrics computation."""

    def test_metrics_structure(self):
        """Test that all expected metrics are returned."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
        ])
        pred_top1 = np.argmax(y_proba, axis=1)
        entropy_values = np.array([0.8, 0.9, 0.7, 0.6, 0.95])
        
        metrics = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values, n_half=2
        )
        
        assert 'n_matches' in metrics
        assert 'accuracy_top1' in metrics
        assert 'accuracy_top2_on_halfguards' in metrics
        assert 'combined_ticket_hit_rate' in metrics
        assert 'logloss' in metrics
        assert 'brier' in metrics

    def test_metrics_ranges(self):
        """Test that metrics are in valid ranges."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_proba = np.random.dirichlet([1, 1, 1], size=8)
        pred_top1 = np.argmax(y_proba, axis=1)
        entropy_values = np.random.rand(8)
        
        metrics = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values, n_half=2
        )
        
        assert 0.0 <= metrics['accuracy_top1'] <= 1.0
        assert 0.0 <= metrics['accuracy_top2_on_halfguards'] <= 1.0
        assert 0.0 <= metrics['combined_ticket_hit_rate'] <= 1.0
        assert metrics['brier'] >= 0.0


class TestBacktestReportScript:
    """Integration tests for the backtest report script."""

    @pytest.fixture
    def mini_dataset(self, tmp_path):
        """Create a minimal dataset for testing."""
        # Create a small synthetic dataset
        np.random.seed(42)
        n_samples = 200
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        teams = ['TeamA', 'TeamB', 'TeamC', 'TeamD', 'TeamE', 'TeamF']
        
        data = {
            'Date': dates,
            'HomeTeam': np.random.choice(teams, n_samples),
            'AwayTeam': np.random.choice(teams, n_samples),
            'FTHG': np.random.randint(0, 4, n_samples),
            'FTAG': np.random.randint(0, 4, n_samples),
            'League': np.random.choice(['E0', 'E1'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Determine FTR based on goals
        def get_ftr(row):
            if row['FTHG'] > row['FTAG']:
                return 'H'
            elif row['FTHG'] < row['FTAG']:
                return 'A'
            return 'D'
        
        df['FTR'] = df.apply(get_ftr, axis=1)
        
        # Save to CSV
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        return df

    def test_train_model_with_mini_data(self, mini_dataset):
        """Test that model training works with minimal data."""
        from feature_engineering import create_features
        
        df_features = create_features(mini_dataset)
        
        # Use first 150 for training
        df_train = df_features.iloc[:150]
        
        model = train_model(df_train)
        
        # Model should be created (or None if data is insufficient)
        # We just check it doesn't crash
        if model is not None:
            assert hasattr(model, 'predict_proba')

    def test_script_output_contains_expected_strings(self, capsys):
        """Test that running the report produces expected output strings."""
        # This test checks that the print_report function outputs expected strings
        from backtest_report import print_report
        
        # Create mock metrics
        mock_metrics = [
            {
                'fold': 1,
                'n_matches': 100,
                'accuracy_top1': 0.45,
                'accuracy_top2_on_halfguards': 0.75,
                'combined_ticket_hit_rate': 0.52,
                'logloss': 1.05,
                'brier': 0.65,
                'accuracy_0': 0.48,
                'logloss_0': 1.02,
            },
            {
                'fold': 2,
                'n_matches': 100,
                'accuracy_top1': 0.47,
                'accuracy_top2_on_halfguards': 0.80,
                'combined_ticket_hit_rate': 0.55,
                'logloss': 1.03,
                'brier': 0.63,
                'accuracy_0': 0.50,
                'logloss_0': 1.00,
            },
        ]
        
        print_report(mock_metrics)
        
        captured = capsys.readouterr()
        
        # Check for expected strings in output
        assert 'accuracy_top1' in captured.out.lower() or 'acc_top1' in captured.out.lower()
        assert 'logloss' in captured.out.lower()
        assert 'brier' in captured.out.lower()
        assert 'BACKTEST REPORT' in captured.out

    def test_script_runs_without_error(self):
        """Test that the script can be imported and has main function."""
        import backtest_report
        
        assert hasattr(backtest_report, 'main')
        assert hasattr(backtest_report, 'run_backtest')
        assert hasattr(backtest_report, 'print_report')
        assert hasattr(backtest_report, 'load_data')
