import pandas as pd
import logging

from feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    logger.info("--- STARTAR FEATURE ENGINEERING (unified builder) ---")
    builder = FeatureBuilder()
    result = builder.fit(df)
    added_cols = [c for c in result.columns if c not in df.columns]
    logger.info("Feature engineering slutförd. Nya kolumner (%d): %s", len(added_cols), added_cols)
    return result
