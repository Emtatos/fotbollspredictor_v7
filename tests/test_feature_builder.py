import pytest
import pandas as pd
import numpy as np
from feature_builder import FeatureBuilder
from schema import FEATURE_COLUMNS, STATS_FEATURE_COLUMNS


def _make_history():
    dates = pd.date_range("2024-01-01", periods=10, freq="7D")
    data = {
        "Date": dates,
        "HomeTeam": [
            "Arsenal", "Chelsea", "Liverpool", "Arsenal", "Chelsea",
            "Liverpool", "Arsenal", "Chelsea", "Liverpool", "Arsenal",
        ],
        "AwayTeam": [
            "Chelsea", "Liverpool", "Arsenal", "Liverpool", "Arsenal",
            "Chelsea", "Chelsea", "Liverpool", "Arsenal", "Chelsea",
        ],
        "FTHG": [2, 1, 0, 3, 1, 2, 1, 0, 1, 2],
        "FTAG": [1, 1, 2, 0, 1, 1, 0, 2, 1, 0],
        "FTR": ["H", "D", "A", "H", "D", "H", "H", "A", "D", "H"],
        "League": ["E0"] * 10,
    }
    return pd.DataFrame(data)


class TestFeatureBuilderTrainInferConsistency:

    def test_last_match_features_identical_in_train_and_infer(self):
        history = _make_history()
        builder_train = FeatureBuilder()
        df_train = builder_train.fit(history)

        history_before_last = history.iloc[:-1].copy()
        builder_infer = FeatureBuilder()
        builder_infer.fit(history_before_last)

        last_row = history.iloc[-1]
        infer_features = builder_infer.features_for_match(
            last_row["HomeTeam"], last_row["AwayTeam"], league=last_row["League"]
        )

        train_row = df_train.iloc[-1]
        feature_keys = [
            "HomeFormPts", "HomeFormGD", "AwayFormPts", "AwayFormGD",
            "HomeFormHome", "AwayFormAway",
            "HomeGoalsFor", "HomeGoalsAgainst", "AwayGoalsFor", "AwayGoalsAgainst",
            "HomeStreak", "AwayStreak",
            "H2H_HomeWins", "H2H_Draws", "H2H_AwayWins", "H2H_HomeGoalDiff",
            "HomePosition", "AwayPosition", "PositionDiff",
            "HomeElo", "AwayElo", "League",
        ]

        for key in feature_keys:
            train_val = float(train_row[key])
            infer_val = float(infer_features[key])
            assert abs(train_val - infer_val) < 1e-6, (
                f"{key}: train={train_val}, infer={infer_val}"
            )

    def test_fit_produces_all_schema_columns(self):
        history = _make_history()
        builder = FeatureBuilder()
        df = builder.fit(history)
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_lookahead_first_match_has_defaults(self):
        history = _make_history()
        builder = FeatureBuilder()
        df = builder.fit(history)
        first = df.iloc[0]
        assert first["HomeFormPts"] == 0.0
        assert first["HomeElo"] == 1500.0
        assert first["AwayElo"] == 1500.0
        assert first["HomeStreak"] == 0.0

    def test_elo_updates_after_result(self):
        history = _make_history()
        builder = FeatureBuilder()
        df = builder.fit(history)
        assert df.iloc[1]["HomeElo"] != 1500.0 or df.iloc[1]["AwayElo"] != 1500.0

    def test_get_team_state_matches_features_for_match(self):
        history = _make_history()
        builder = FeatureBuilder()
        builder.fit(history)

        state = builder.get_team_state("Arsenal")
        assert state is not None
        assert state["MatchesPlayed"] > 0
        assert "Elo" in state
        assert "FormPts" in state

    def test_empty_history(self):
        builder = FeatureBuilder()
        result = builder.fit(pd.DataFrame())
        assert result.empty

    def test_features_for_unknown_team_returns_none(self):
        history = _make_history()
        builder = FeatureBuilder()
        builder.fit(history)
        assert builder.features_for_match("Unknown FC", "Arsenal") is None

    def test_date_sorting_enforced(self):
        history = _make_history()
        shuffled = history.sample(frac=1, random_state=42).reset_index(drop=True)
        builder = FeatureBuilder()
        df = builder.fit(shuffled)
        assert df["Date"].is_monotonic_increasing


def _make_history_with_stats():
    dates = pd.date_range("2024-01-01", periods=6, freq="7D")
    data = {
        "Date": dates,
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
        "AwayTeam": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
        "FTHG": [2, 1, 3, 0, 1, 2],
        "FTAG": [1, 0, 1, 2, 1, 0],
        "FTR": ["H", "H", "H", "A", "D", "H"],
        "League": ["E0"] * 6,
        "HS": [12, 8, 15, 6, 10, 14],
        "AS": [7, 10, 9, 12, 11, 5],
        "HST": [5, 3, 7, 2, 4, 6],
        "AST": [3, 5, 4, 6, 5, 2],
        "HC": [6, 4, 8, 3, 5, 7],
        "AC": [3, 6, 4, 7, 6, 2],
        "HY": [2, 1, 3, 0, 1, 2],
        "AY": [1, 2, 1, 3, 2, 1],
    }
    return pd.DataFrame(data)


class TestMatchstatsFeatures:

    def test_has_matchstats_flag_with_stats(self):
        history = _make_history_with_stats()
        builder = FeatureBuilder()
        df = builder.fit(history)
        assert df.iloc[0]["has_matchstats"] == 0.0
        assert df.iloc[2]["has_matchstats"] == 1.0

    def test_has_matchstats_flag_without_stats(self):
        history = _make_history()
        builder = FeatureBuilder()
        df = builder.fit(history)
        assert all(df["has_matchstats"] == 0.0)

    def test_stats_columns_present(self):
        history = _make_history_with_stats()
        builder = FeatureBuilder()
        df = builder.fit(history)
        for col in STATS_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing stats column: {col}"

    def test_no_nan_or_inf_in_stats_features(self):
        history = _make_history_with_stats()
        builder = FeatureBuilder()
        df = builder.fit(history)
        for col in STATS_FEATURE_COLUMNS:
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_conversion_laplace_smoothing(self):
        history = _make_history_with_stats()
        builder = FeatureBuilder()
        df = builder.fit(history)
        for val in df["HomeConversion"]:
            assert 0.0 < val < 1.0

    def test_stats_features_identical_train_infer(self):
        history = _make_history_with_stats()
        builder_train = FeatureBuilder()
        df_train = builder_train.fit(history)

        builder_infer = FeatureBuilder()
        builder_infer.fit(history.iloc[:-1].copy())
        last = history.iloc[-1]
        infer_f = builder_infer.features_for_match(
            last["HomeTeam"], last["AwayTeam"], league=last["League"]
        )

        for col in STATS_FEATURE_COLUMNS:
            train_val = float(df_train.iloc[-1][col])
            infer_val = float(infer_f[col])
            assert abs(train_val - infer_val) < 1e-6, f"{col}: train={train_val}, infer={infer_val}"

    def test_pipeline_works_without_stats(self):
        history = _make_history()
        builder = FeatureBuilder()
        df = builder.fit(history)
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"
