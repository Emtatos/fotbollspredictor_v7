import numpy as np
import pandas as pd
import pytest

from feature_builder import FeatureBuilder
from schema import FEATURE_COLUMNS, ALL_FEATURE_COLUMNS, CLASS_MAP


def _make_sample_dataset(n: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="3D")
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    rows = []
    for i in range(n):
        ht = teams[i % len(teams)]
        at = teams[(i + 1) % len(teams)]
        fthg = int(rng.randint(0, 4))
        ftag = int(rng.randint(0, 4))
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"
        rows.append({
            "Date": dates[i],
            "HomeTeam": ht,
            "AwayTeam": at,
            "FTHG": fthg,
            "FTAG": ftag,
            "FTR": ftr,
            "League": "E0",
            "Season": "2425",
            "HS": int(rng.randint(5, 20)),
            "AS": int(rng.randint(5, 20)),
            "HST": int(rng.randint(1, 10)),
            "AST": int(rng.randint(1, 10)),
            "HC": int(rng.randint(2, 10)),
            "AC": int(rng.randint(2, 10)),
            "HY": int(rng.randint(0, 5)),
            "AY": int(rng.randint(0, 5)),
        })
    return pd.DataFrame(rows)


class TestSmokePipeline:

    def test_feature_builder_produces_valid_features(self):
        df = _make_sample_dataset(30)
        builder = FeatureBuilder()
        result = builder.fit(df)

        assert len(result) == 30
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing: {col}"
            assert not result[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_feature_builder_produces_odds_columns(self):
        df = _make_sample_dataset(30)
        builder = FeatureBuilder()
        result = builder.fit(df)

        for col in ALL_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing: {col}"

    def test_train_predict_roundtrip(self):
        from xgboost import XGBClassifier

        df = _make_sample_dataset(30)
        builder = FeatureBuilder()
        result = builder.fit(df)

        X = result[FEATURE_COLUMNS].copy()
        y = result["FTR"].map(CLASS_MAP)

        model = XGBClassifier(
            n_estimators=10,
            max_depth=2,
            random_state=42,
            eval_metric="mlogloss",
        )
        try:
            model = XGBClassifier(
                use_label_encoder=False,
                n_estimators=10,
                max_depth=2,
                random_state=42,
                eval_metric="mlogloss",
            )
        except TypeError:
            pass
        model.fit(X, y)

        proba = model.predict_proba(X.iloc[:1])
        assert proba.shape == (1, 3)
        assert abs(proba.sum() - 1.0) < 1e-6
        assert not np.isnan(proba).any()

    def test_inference_via_features_for_match(self):
        df = _make_sample_dataset(30)
        builder = FeatureBuilder()
        builder.fit(df)

        features = builder.features_for_match("TeamA", "TeamB", league="E0")
        assert features is not None

        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"
            assert not np.isnan(features[col]), f"NaN in {col}"

    def test_probs_sum_to_one(self):
        from xgboost import XGBClassifier
        from schema import proba_to_1x2

        df = _make_sample_dataset(30)
        builder = FeatureBuilder()
        result = builder.fit(df)

        X = result[FEATURE_COLUMNS].copy()
        y = result["FTR"].map(CLASS_MAP)

        model = XGBClassifier(
            n_estimators=10,
            max_depth=2,
            random_state=42,
            eval_metric="mlogloss",
        )
        try:
            model = XGBClassifier(
                use_label_encoder=False,
                n_estimators=10,
                max_depth=2,
                random_state=42,
                eval_metric="mlogloss",
            )
        except TypeError:
            pass
        model.fit(X, y)

        for i in range(min(5, len(X))):
            proba = model.predict_proba(X.iloc[i : i + 1])[0]
            assert abs(sum(proba) - 1.0) < 1e-6, f"Probs don't sum to 1 for row {i}"
            result_map = proba_to_1x2(proba, classes=model.classes_)
            total = result_map["1"] + result_map["X"] + result_map["2"]
            assert abs(total - 1.0) < 1e-6

    def test_metadata_generation(self):
        from pathlib import Path
        import tempfile
        from metadata import generate_metadata, load_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_metadata(
                model_dir=Path(tmpdir),
                features_list=list(FEATURE_COLUMNS),
                calibration_method="sigmoid",
                train_size=100,
            )
            assert out.exists()
            meta = load_metadata(Path(tmpdir))
            assert meta is not None
            assert meta["model_version"] == "v8.0"
            assert meta["splits"]["train"] == 100
            assert len(meta["features"]) == len(FEATURE_COLUMNS)

    def test_pipeline_without_stats_or_odds(self):
        rng = np.random.RandomState(99)
        dates = pd.date_range("2024-06-01", periods=20, freq="4D")
        teams = ["Alpha", "Beta", "Gamma", "Delta"]
        rows = []
        for i in range(20):
            ht = teams[i % len(teams)]
            at = teams[(i + 2) % len(teams)]
            fthg = int(rng.randint(0, 3))
            ftag = int(rng.randint(0, 3))
            if fthg > ftag:
                ftr = "H"
            elif fthg < ftag:
                ftr = "A"
            else:
                ftr = "D"
            rows.append({
                "Date": dates[i],
                "HomeTeam": ht,
                "AwayTeam": at,
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "League": "E1",
            })
        df = pd.DataFrame(rows)
        builder = FeatureBuilder()
        result = builder.fit(df)

        assert len(result) == 20
        for col in FEATURE_COLUMNS:
            assert col in result.columns
        assert all(result["has_matchstats"] == 0.0)
        assert all(result["has_odds"] == 0.0)
