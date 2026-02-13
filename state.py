from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from feature_builder import FeatureBuilder


def build_current_team_states(df_history: pd.DataFrame, k_factor: int = 20) -> Dict[str, Dict[str, Any]]:
    if df_history is None or df_history.empty:
        return {}
    builder = FeatureBuilder(k_factor=k_factor)
    builder.fit(df_history)
    return builder.get_all_team_states()
