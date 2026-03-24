# tests/test_e2e_smoke.py
"""
End-to-end smoke test: genererar syntetisk matchdata,
kör feature engineering och tränar modell.
Ingen nätverksåtkomst behövs.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from data_processing import normalize_csv_data
from feature_engineering import create_features
from feature_builder import FeatureBuilder
from schema import FEATURE_COLUMNS


@pytest.fixture
def synthetic_matches(tmp_path):
    """Skapar en CSV med 60 syntetiska matcher för E0."""
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham", "Everton"]
    rows = []
    base_date = datetime(2024, 8, 17)
    for i in range(60):
        ht = teams[i % len(teams)]
        at = teams[(i + 1) % len(teams)]
        fthg = np.random.randint(0, 4)
        ftag = np.random.randint(0, 4)
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"
        rows.append({
            "Date": (base_date + timedelta(days=i * 3)).strftime("%d/%m/%Y"),
            "HomeTeam": ht,
            "AwayTeam": at,
            "FTHG": fthg,
            "FTAG": ftag,
            "FTR": ftr,
            "HS": np.random.randint(5, 20),
            "AS": np.random.randint(5, 20),
            "HST": np.random.randint(2, 10),
            "AST": np.random.randint(2, 10),
            "HF": np.random.randint(5, 18),
            "AF": np.random.randint(5, 18),
            "HC": np.random.randint(2, 12),
            "AC": np.random.randint(2, 12),
            "HY": np.random.randint(0, 5),
            "AY": np.random.randint(0, 5),
            "HR": np.random.randint(0, 1),
            "AR": np.random.randint(0, 1),
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "E0_2425.csv"
    df.to_csv(csv_path, index=False)
    return [csv_path]


def test_full_pipeline_smoke(synthetic_matches):
    """Hela pipelinen: CSV → normalize → features → kontrollera schema."""
    # Steg 1: Normalisera
    df_clean = normalize_csv_data(synthetic_matches)
    assert len(df_clean) >= 50, f"Bara {len(df_clean)} rader efter normalisering"
    assert "FTR" in df_clean.columns

    # Steg 2: Feature engineering
    df_features = create_features(df_clean)
    assert len(df_features) >= 50

    # Steg 3: Alla schemakolumner finns
    missing = [c for c in FEATURE_COLUMNS if c not in df_features.columns]
    assert missing == [], f"Saknade feature-kolumner: {missing}"

    # Steg 4: Inga NaN i features (efter first ~5 matcher som behövs för form)
    df_later = df_features.iloc[10:]
    for col in FEATURE_COLUMNS:
        nan_count = df_later[col].isna().sum()
        assert nan_count == 0, f"Kolumn {col} har {nan_count} NaN efter rad 10"
