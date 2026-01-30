# inference.py
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from schema import FEATURE_COLUMNS, get_expected_feature_columns, proba_to_1x2, encode_league


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # League kan vara str -> int
    if "League" in df.columns:
        df["League"] = df["League"].apply(encode_league).astype(float)
    # övriga -> numeric
    for c in df.columns:
        if c == "League":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def build_feature_row(feature_dict: Dict[str, Any], expected_cols: Optional[list[str]] = None) -> pd.DataFrame:
    cols = expected_cols or list(FEATURE_COLUMNS)
    row = {c: feature_dict.get(c, 0) for c in cols}
    df = pd.DataFrame([row], columns=cols)
    return _coerce_numeric(df)


def predict_match(model, feature_row_df: pd.DataFrame) -> Dict[str, float]:
    expected_cols = get_expected_feature_columns(model)

    # Fylla saknade + ordna
    X = feature_row_df.copy()
    for c in expected_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[expected_cols]
    X = _coerce_numeric(X)

    proba = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", None)
    return proba_to_1x2(proba, classes=classes)
