"""CI smoke pipeline: generates synthetic data, runs training, metadata, and features export.

This avoids network calls to football-data.co.uk and runs in ~10s.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing import normalize_csv_data
from feature_engineering import create_features
from model_handler import train_and_save_model, load_model, get_feature_columns, use_odds_features
from metadata import generate_metadata
from schema import ABLATION_GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEAGUES = ["E0", "E1"]
SEASONS = ["2425"]


def _generate_synthetic_csv(league: str, season: str, n: int = 80) -> Path:
    rng = np.random.RandomState(hash(f"{league}{season}") % 2**31)
    teams = {
        "E0": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham", "Man United", "Everton", "Newcastle"],
        "E1": ["Leeds", "Burnley", "Sunderland", "Norwich", "Coventry", "Watford", "Bristol City", "Middlesbrough"],
    }
    team_list = teams.get(league, teams["E0"])
    dates = pd.date_range("2024-08-10", periods=n, freq="3D")
    rows = []
    for i in range(n):
        ht = team_list[i % len(team_list)]
        at = team_list[(i + 3) % len(team_list)]
        fthg = int(rng.randint(0, 5))
        ftag = int(rng.randint(0, 5))
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"
        rows.append({
            "Date": dates[i].strftime("%d/%m/%Y"),
            "HomeTeam": ht,
            "AwayTeam": at,
            "FTHG": fthg,
            "FTAG": ftag,
            "FTR": ftr,
            "HS": int(rng.randint(5, 20)),
            "AS": int(rng.randint(5, 20)),
            "HST": int(rng.randint(1, 10)),
            "AST": int(rng.randint(1, 10)),
            "HF": int(rng.randint(5, 18)),
            "AF": int(rng.randint(5, 18)),
            "HC": int(rng.randint(2, 12)),
            "AC": int(rng.randint(2, 12)),
            "HY": int(rng.randint(0, 5)),
            "AY": int(rng.randint(0, 5)),
            "HR": int(rng.randint(0, 2)),
            "AR": int(rng.randint(0, 2)),
            "B365H": round(float(rng.uniform(1.2, 6.0)), 2),
            "B365D": round(float(rng.uniform(2.5, 5.0)), 2),
            "B365A": round(float(rng.uniform(1.2, 6.0)), 2),
            "PSH": round(float(rng.uniform(1.2, 6.0)), 2),
            "PSD": round(float(rng.uniform(2.5, 5.0)), 2),
            "PSA": round(float(rng.uniform(1.2, 6.0)), 2),
        })
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{league}_{season}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Generated synthetic CSV: %s (%d rows)", out, n)
    return out


def main() -> None:
    logger.info("=== CI Smoke Pipeline Start ===")

    csv_files = []
    for league in LEAGUES:
        for season in SEASONS:
            csv_files.append(_generate_synthetic_csv(league, season))
    logger.info("Generated %d CSV files", len(csv_files))

    df_clean = normalize_csv_data(file_paths=csv_files)
    if df_clean.empty:
        logger.error("normalize_csv_data returned empty DataFrame")
        sys.exit(1)
    logger.info("Normalized %d matches", len(df_clean))

    df_features = create_features(df=df_clean)
    if df_features.empty:
        logger.error("create_features returned empty DataFrame")
        sys.exit(1)
    logger.info("Created features: %d rows, %d columns", len(df_features), len(df_features.columns))

    features_path = Path("data") / "features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(features_path)
    logger.info("Saved features to %s", features_path)

    model_path = Path("models") / "xgboost_model_v7_ci.joblib"
    trained = train_and_save_model(df_features=df_features, model_path=model_path)
    if trained is None:
        logger.error("train_and_save_model returned None")
        sys.exit(1)
    logger.info("Model trained and saved to %s", model_path)

    date_col = df_features["Date"] if "Date" in df_features.columns else None
    date_range = None
    if date_col is not None:
        date_range = (str(date_col.min()), str(date_col.max()))

    with_odds = use_odds_features()
    feat_cols = get_feature_columns(with_odds=with_odds)
    meta_path = generate_metadata(
        model_dir=model_path.parent,
        features_list=feat_cols,
        calibration_method="sigmoid",
        train_size=len(df_features),
        dataset_leagues=LEAGUES,
        dataset_seasons=SEASONS,
        dataset_date_range=date_range,
        extra={
            "use_odds_features": with_odds,
            "ablation_groups": list(ABLATION_GROUPS.keys()),
        },
    )
    logger.info("Metadata saved to %s", meta_path)

    loaded = load_model(model_path=model_path)
    if loaded is None:
        logger.error("Could not reload model from disk")
        sys.exit(1)
    logger.info("Model reload verification OK")

    report_path = Path("reports") / "backtest_report.md"
    if report_path.exists():
        logger.info("Backtest report found: %s (%d bytes)", report_path, report_path.stat().st_size)
    else:
        logger.warning("No backtest report at %s", report_path)

    logger.info("=== CI Smoke Pipeline PASSED ===")


if __name__ == "__main__":
    main()
