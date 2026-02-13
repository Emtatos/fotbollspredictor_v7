import pytest
import pandas as pd
import numpy as np
from feature_builder import FeatureBuilder
from schema import FEATURE_COLUMNS


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
