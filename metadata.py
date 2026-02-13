import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

MODEL_VERSION = "v8.0"
METADATA_FILENAME = "metadata.json"


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_lib_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for lib_name in ["xgboost", "sklearn", "pandas", "numpy", "joblib", "streamlit"]:
        try:
            mod = __import__(lib_name)
            versions[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib_name] = "not installed"
    return versions


def generate_metadata(
    model_dir: Path,
    features_list: List[str],
    model_params: Optional[Dict[str, Any]] = None,
    calibration_method: str = "sigmoid",
    train_size: int = 0,
    calibrate_size: int = 0,
    test_size: int = 0,
    dataset_leagues: Optional[List[str]] = None,
    dataset_seasons: Optional[List[str]] = None,
    dataset_date_range: Optional[tuple] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "train_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "lib_versions": _get_lib_versions(),
        "git_commit": _get_git_commit(),
        "features": features_list,
        "model_params": model_params or {},
        "calibration_method": calibration_method,
        "splits": {
            "train": train_size,
            "calibrate": calibrate_size,
            "test": test_size,
        },
        "dataset_coverage": {
            "leagues": dataset_leagues or [],
            "seasons": dataset_seasons or [],
            "date_range": {
                "min": str(dataset_date_range[0]) if dataset_date_range else None,
                "max": str(dataset_date_range[1]) if dataset_date_range else None,
            },
        },
    }
    if extra:
        meta.update(extra)

    out_path = model_dir / METADATA_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return out_path


def load_metadata(model_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(model_dir) / METADATA_FILENAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
