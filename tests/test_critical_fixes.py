"""
Tests for the four critical stabilization fixes:
  A. Strict model loading
  B. Time-correct calibration
  C. Walk-forward without data leakage
  D. Inference consistency (FeatureBuilder.features_for_match)
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from model_handler import (
    _make_base_xgb,
    _walk_forward,
    load_model,
    train_and_save_model,
    MODEL_BASE_FILENAME,
    MODEL_CALIBRATED_FILENAME,
)
from feature_builder import FeatureBuilder
from schema import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_model() -> XGBClassifier:
    """Train a tiny XGB model for serialization tests."""
    rng = np.random.RandomState(42)
    X = rng.rand(60, 3)
    y = rng.choice([0, 1, 2], size=60)
    m = XGBClassifier(
        n_estimators=5, max_depth=2, eval_metric="mlogloss",
        random_state=42, use_label_encoder=False,
    )
    m.fit(X, y)
    return m


def _make_history_df(n: int = 120) -> pd.DataFrame:
    """Create a minimal history DataFrame for FeatureBuilder."""
    rng = np.random.RandomState(0)
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham", "Man Utd"]
    rows = []
    for i in range(n):
        ht = teams[i % len(teams)]
        at = teams[(i + 3) % len(teams)]
        fthg = rng.randint(0, 4)
        ftag = rng.randint(0, 4)
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"
        rows.append({
            "Date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
            "HomeTeam": ht,
            "AwayTeam": at,
            "FTHG": fthg,
            "FTAG": ftag,
            "FTR": ftr,
            "League": "E0",
            "Season": "2023",
        })
    return pd.DataFrame(rows)


# ============================================================================
# A. Strict model loading (3 tests)
# ============================================================================

class TestStrictModelLoading:
    """load_model must load exactly what is requested when a specific file is given."""

    def test_specific_file_loads_exactly(self, tmp_path: Path):
        """Specific .joblib file requested -> exactly that file is loaded."""
        model = _make_dummy_model()
        specific = tmp_path / "xgboost_model_v7_2425.joblib"
        joblib.dump(model, specific)

        loaded = load_model(specific)

        assert loaded is not None
        assert isinstance(loaded, XGBClassifier)

    def test_specific_file_missing_returns_none(self, tmp_path: Path):
        """Specific file does not exist -> None, no fallback."""
        missing = tmp_path / "xgboost_model_v7_9999.joblib"

        # Even if generic files exist, they must NOT be used
        joblib.dump(_make_dummy_model(), tmp_path / MODEL_CALIBRATED_FILENAME)
        joblib.dump(_make_dummy_model(), tmp_path / MODEL_BASE_FILENAME)

        loaded = load_model(missing)
        assert loaded is None

    def test_directory_path_uses_legacy_fallback(self, tmp_path: Path):
        """Directory path -> legacy fallback chain (calibrated > base)."""
        from sklearn.frozen import FrozenEstimator

        model = _make_dummy_model()
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model))
        # Fit calibrated on tiny data so it's valid
        rng = np.random.RandomState(1)
        calibrated.fit(rng.rand(30, 3), rng.choice([0, 1, 2], size=30))
        joblib.dump(calibrated, tmp_path / MODEL_CALIBRATED_FILENAME)

        loaded = load_model(tmp_path)  # directory, no suffix

        assert loaded is not None
        assert isinstance(loaded, CalibratedClassifierCV)


# ============================================================================
# B. Time-correct calibration (2 tests)
# ============================================================================

class TestCalibrationFix:
    """Calibration must use the already-trained base_model on the cal-split."""

    @patch("model_handler.CalibratedClassifierCV")
    @patch("model_handler._walk_forward", return_value={
        "logloss_mean": 1.0, "logloss_std": 0.1,
        "brier_mean": 0.7, "brier_std": 0.1,
        "accuracy_mean": 0.4, "accuracy_std": 0.1,
        "f1_macro_mean": 0.3, "f1_macro_std": 0.1,
        "n_folds": 3,
    })
    def test_calibration_uses_trained_model(self, mock_wf, mock_ccv, tmp_path):
        """CalibratedClassifierCV receives the already-trained base_model, not a new one."""
        from model_handler import _time_split, _prepare_df, get_feature_columns
        from schema import CLASS_MAP

        # Build a real feature df via FeatureBuilder so columns are correct
        hist = _make_history_df(200)
        builder = FeatureBuilder()
        df = builder.fit(hist)

        feat_cols = get_feature_columns()
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df["FTR"] = hist["FTR"].values[:len(df)]

        # Mock the CalibratedClassifierCV so we can inspect what it receives
        mock_cal_instance = MagicMock()
        mock_cal_instance.predict_proba.return_value = np.full((50, 3), 1 / 3)
        mock_cal_instance.predict.return_value = np.zeros(50, dtype=int)
        mock_ccv.return_value = mock_cal_instance

        model_path = tmp_path / "test_model.joblib"
        train_and_save_model(
            df, model_path,
            run_hyperparam_search=False,
        )

        # CalibratedClassifierCV should have been called with a FrozenEstimator wrapping the trained model
        call_kwargs = mock_ccv.call_args
        assert call_kwargs is not None

        # The estimator passed should be a FrozenEstimator wrapping an XGBClassifier
        from sklearn.frozen import FrozenEstimator
        estimator_arg = call_kwargs.kwargs.get("estimator") or call_kwargs.args[0]
        assert isinstance(estimator_arg, FrozenEstimator)

    @patch("model_handler.CalibratedClassifierCV")
    @patch("model_handler._walk_forward", return_value={
        "logloss_mean": 1.0, "logloss_std": 0.1,
        "brier_mean": 0.7, "brier_std": 0.1,
        "accuracy_mean": 0.4, "accuracy_std": 0.1,
        "f1_macro_mean": 0.3, "f1_macro_std": 0.1,
        "n_folds": 3,
    })
    def test_calibration_fit_on_cal_split(self, mock_wf, mock_ccv, tmp_path):
        """Calibration .fit() is called with cal-split data, not train+cal."""
        from model_handler import _time_split, get_feature_columns
        from schema import CLASS_MAP

        hist = _make_history_df(200)
        builder = FeatureBuilder()
        df = builder.fit(hist)

        feat_cols = get_feature_columns()
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df["FTR"] = hist["FTR"].values[:len(df)]

        mock_cal_instance = MagicMock()
        mock_cal_instance.predict_proba.return_value = np.full((50, 3), 1 / 3)
        mock_cal_instance.predict.return_value = np.zeros(50, dtype=int)
        mock_ccv.return_value = mock_cal_instance

        model_path = tmp_path / "test_model.joblib"
        train_and_save_model(df, model_path, run_hyperparam_search=False)

        # .fit() should have been called with cal-split size
        fit_call = mock_cal_instance.fit.call_args
        assert fit_call is not None
        X_fit = fit_call.args[0] if fit_call.args else fit_call.kwargs.get("X")
        n_total = len(df)
        n_cal_expected = int(n_total * 0.15)  # _time_split default cal_frac=0.15
        # cal size should be roughly 15% of total, not 85% (train+cal)
        assert len(X_fit) < n_total * 0.5, (
            f"Calibration fit got {len(X_fit)} rows — looks like train+cal, not just cal"
        )


# ============================================================================
# C. Walk-forward without data leakage (2 tests)
# ============================================================================

class TestWalkForwardFix:
    """_walk_forward must run on train+cal only and use best_params."""

    @patch("model_handler._walk_forward", return_value={
        "logloss_mean": 1.0, "logloss_std": 0.1,
        "brier_mean": 0.7, "brier_std": 0.1,
        "accuracy_mean": 0.4, "accuracy_std": 0.1,
        "f1_macro_mean": 0.3, "f1_macro_std": 0.1,
        "n_folds": 3,
    })
    def test_walk_forward_called_with_train_cal_not_full(self, mock_wf, tmp_path):
        """_walk_forward is called with train+cal data, not X_full."""
        hist = _make_history_df(200)
        builder = FeatureBuilder()
        df = builder.fit(hist)

        from model_handler import get_feature_columns
        feat_cols = get_feature_columns()
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df["FTR"] = hist["FTR"].values[:len(df)]

        model_path = tmp_path / "test_model.joblib"
        train_and_save_model(df, model_path, run_hyperparam_search=False)

        assert mock_wf.called
        call_args = mock_wf.call_args
        X_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("X")

        n_total = len(df)
        n_test = n_total - int(n_total * 0.70) - int(n_total * 0.15)
        n_train_cal = n_total - n_test  # should be ~85% of total

        # Walk-forward input should be train+cal (not full)
        assert len(X_arg) < n_total, (
            f"Walk-forward got {len(X_arg)} rows = full dataset ({n_total}), expected train+cal"
        )

    @patch("model_handler._walk_forward", return_value={
        "logloss_mean": 1.0, "logloss_std": 0.1,
        "brier_mean": 0.7, "brier_std": 0.1,
        "accuracy_mean": 0.4, "accuracy_std": 0.1,
        "f1_macro_mean": 0.3, "f1_macro_std": 0.1,
        "n_folds": 3,
    })
    def test_walk_forward_receives_xgb_params(self, mock_wf, tmp_path):
        """_walk_forward receives xgb_params kwarg."""
        hist = _make_history_df(200)
        builder = FeatureBuilder()
        df = builder.fit(hist)

        from model_handler import get_feature_columns
        feat_cols = get_feature_columns()
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df["FTR"] = hist["FTR"].values[:len(df)]

        model_path = tmp_path / "test_model.joblib"
        train_and_save_model(df, model_path, run_hyperparam_search=False)

        call_kwargs = mock_wf.call_args.kwargs if mock_wf.call_args.kwargs else {}
        # xgb_params should be passed (may be empty dict when search is off)
        assert "xgb_params" in call_kwargs, "xgb_params not passed to _walk_forward"

    def test_walk_forward_uses_params(self):
        """_walk_forward creates models with provided xgb_params."""
        rng = np.random.RandomState(7)
        X = pd.DataFrame(rng.rand(200, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.choice([0, 1, 2], size=200))

        custom_params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}

        with patch("model_handler._make_base_xgb", wraps=_make_base_xgb) as spy:
            _walk_forward(X, y, n_folds=2, xgb_params=custom_params)
            # Every call to _make_base_xgb should have received the custom params
            for call in spy.call_args_list:
                for key, val in custom_params.items():
                    assert call.kwargs.get(key) == val or (
                        key in (call.kwargs or {})
                    ), f"_make_base_xgb not called with {key}={val}"


# ============================================================================
# D. Inference consistency (2 tests)
# ============================================================================

class TestInferenceConsistency:
    """predict_match must use FeatureBuilder.features_for_match()."""

    def test_predict_match_uses_features_for_match(self):
        """predict_match() calls FeatureBuilder.features_for_match()."""
        import importlib
        import streamlit as st

        # Patch st.secrets before importing app (module-level code reads secrets)
        mock_secrets = MagicMock()
        mock_secrets.get.return_value = None
        with patch.object(st, "secrets", mock_secrets, create=True):
            # Force re-import so module-level code sees the mock
            import app as app_mod
            importlib.reload(app_mod)
            predict_match_fn = app_mod.predict_match

            # Set up mock builder
            mock_builder = MagicMock(spec=FeatureBuilder)
            mock_builder.features_for_match.return_value = {
                c: 0.0 for c in FEATURE_COLUMNS
            }
            mock_builder.get_team_state.return_value = {
                "MatchesPlayed": 10, "FormPts": 2.0, "FormGD": 0.5,
                "FormHome": 2.0, "FormAway": 1.5, "GoalsFor": 1.5,
                "GoalsAgainst": 1.0, "Streak": 1, "Elo": 1500.0,
                "Position": 5, "League": 0,
            }

            with patch.object(app_mod, "_get_feature_builder", return_value=mock_builder):
                # Mock model
                mock_model = MagicMock()
                mock_model.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])
                mock_model.classes_ = np.array([0, 1, 2])

                dummy_df = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"]})

                result = predict_match_fn(mock_model, "Arsenal", "Chelsea", dummy_df)

                # features_for_match must have been called
                mock_builder.features_for_match.assert_called_once()

    def test_features_for_match_keys_match_feature_columns(self):
        """FeatureBuilder.features_for_match() returns keys compatible with FEATURE_COLUMNS."""
        hist = _make_history_df(100)
        builder = FeatureBuilder()
        builder.fit(hist)

        features = builder.features_for_match("Arsenal", "Chelsea")
        assert features is not None

        # All FEATURE_COLUMNS should be present in the returned dict
        missing = [c for c in FEATURE_COLUMNS if c not in features]
        assert not missing, f"Missing feature columns from features_for_match: {missing}"
