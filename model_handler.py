import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from schema import ALL_FEATURE_COLUMNS, CLASS_MAP, FEATURE_COLUMNS, ODDS_FEATURE_COLUMNS, encode_league


def use_odds_features() -> bool:
    return os.environ.get("USE_ODDS_FEATURES", "0") == "1"


def get_feature_columns(with_odds: bool = False) -> List[str]:
    if with_odds:
        return list(ALL_FEATURE_COLUMNS)
    return list(FEATURE_COLUMNS)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_BASE_FILENAME = "model_base.pkl"
MODEL_CALIBRATED_FILENAME = "model_calibrated.pkl"
MODEL_LEGACY_FILENAME = "model.joblib"

_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.005, 0.01, 0.02, 0.05],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [1.0, 1.5, 2.0],
}


def _brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> float:
    y_oh = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        if 0 <= label < n_classes:
            y_oh[i, label] = 1
    return float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))


def _reliability_bins(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> List[Dict]:
    bins: List[Dict] = []
    edges = np.linspace(0, 1, n_bins + 1)
    max_proba = y_proba.max(axis=1)
    pred_class = y_proba.argmax(axis=1)
    correct = (pred_class == y_true).astype(float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_proba >= lo) & (max_proba < hi)
        if hi == 1.0:
            mask = mask | (max_proba == 1.0)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "count": 0, "avg_conf": 0.0, "avg_acc": 0.0})
        else:
            bins.append({
                "bin_lo": round(lo, 2),
                "bin_hi": round(hi, 2),
                "count": count,
                "avg_conf": round(float(max_proba[mask].mean()), 4),
                "avg_acc": round(float(correct[mask].mean()), 4),
            })
    return bins


def _time_split(df: pd.DataFrame, train_frac: float = 0.70, cal_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + cal_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def _prepare_df(df_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df_features.copy()
    if "League" in df.columns:
        df["League"] = df["League"].apply(encode_league)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date", ascending=True).reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return df


def _make_base_xgb(**overrides) -> XGBClassifier:
    kw: Dict = dict(
        objective="multi:softprob",
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    kw.update(overrides)
    try:
        return XGBClassifier(use_label_encoder=False, **kw)
    except TypeError:
        return XGBClassifier(**kw)


def _walk_forward(X: pd.DataFrame, y: pd.Series, n_folds: int = 3) -> Dict:
    n = len(X)
    fold_size = n // (n_folds + 1)
    results: Dict[str, List[float]] = {"logloss": [], "brier": [], "accuracy": [], "f1_macro": []}

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)
        if test_start >= n or train_end < 30:
            continue

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        if len(X_te) < 10:
            continue

        model = _make_base_xgb()
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        proba = model.predict_proba(X_te)
        preds = model.predict(X_te)
        y_arr = np.array(y_te)

        results["logloss"].append(log_loss(y_arr, proba, labels=[0, 1, 2]))
        results["brier"].append(_brier_multiclass(y_arr, proba))
        results["accuracy"].append(accuracy_score(y_arr, preds))
        results["f1_macro"].append(f1_score(y_arr, preds, average="macro", zero_division=0))

    summary: Dict = {}
    for k, vals in results.items():
        if vals:
            summary[k + "_mean"] = round(float(np.mean(vals)), 4)
            summary[k + "_std"] = round(float(np.std(vals)), 4)
        else:
            summary[k + "_mean"] = float("nan")
            summary[k + "_std"] = float("nan")
    summary["n_folds"] = len(results["logloss"])
    return summary


def _hyperparam_search(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 20) -> Dict:
    base = _make_base_xgb()
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base,
        _PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return dict(search.best_params_)


def _per_league_season_metrics(df_test: pd.DataFrame, y_true: np.ndarray, y_proba: np.ndarray, preds: np.ndarray) -> List[Dict]:
    rows: List[Dict] = []
    league_col = "League" if "League" in df_test.columns else None
    season_col = "Season" if "Season" in df_test.columns else None

    if league_col is None:
        return rows

    group_cols = [league_col] + ([season_col] if season_col else [])
    groups = df_test.groupby(group_cols)
    for group_key, group_df in groups:
        idx = group_df.index
        mask = df_test.index.isin(idx)
        pos = np.where(mask)[0]
        if len(pos) < 5:
            continue
        yt = y_true[pos]
        yp = y_proba[pos]
        pr = preds[pos]
        row: Dict = {}
        if season_col:
            row["League"] = group_key[0]
            row["Season"] = group_key[1]
        else:
            row["League"] = group_key
        row["n"] = len(pos)
        row["accuracy"] = round(float(accuracy_score(yt, pr)), 4)
        row["f1_macro"] = round(float(f1_score(yt, pr, average="macro", zero_division=0)), 4)
        try:
            row["logloss"] = round(float(log_loss(yt, yp, labels=[0, 1, 2])), 4)
        except Exception:
            row["logloss"] = float("nan")
        row["brier"] = round(float(_brier_multiclass(yt, yp)), 4)
        rows.append(row)
    return rows


def _top_features_by_gain(model: XGBClassifier, feature_names: List[str], top_k: int = 15) -> List[Tuple[str, float]]:
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    pairs: List[Tuple[str, float]] = []
    for fname, gain in importance.items():
        idx = int(fname.replace("f", "")) if fname.startswith("f") and fname[1:].isdigit() else -1
        name = feature_names[idx] if 0 <= idx < len(feature_names) else fname
        pairs.append((name, float(gain)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def _quick_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    label: str,
) -> Dict:
    train_df, cal_df, test_df = _time_split(df)
    X_tr = train_df[feature_cols]
    y_tr = train_df["FTR"].map(CLASS_MAP)
    X_cal = cal_df[feature_cols]
    y_cal = cal_df["FTR"].map(CLASS_MAP)
    X_te = test_df[feature_cols]
    y_te = test_df["FTR"].map(CLASS_MAP)

    model = _make_base_xgb()
    model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)
    proba = model.predict_proba(X_te)
    preds = model.predict(X_te)
    y_arr = np.array(y_te)

    result: Dict = {"variant": label, "n_test": len(y_arr)}
    result["accuracy"] = round(float(accuracy_score(y_arr, preds)), 4)
    result["f1_macro"] = round(float(f1_score(y_arr, preds, average="macro", zero_division=0)), 4)
    result["brier"] = round(float(_brier_multiclass(y_arr, proba)), 4)
    try:
        result["logloss"] = round(float(log_loss(y_arr, proba, labels=[0, 1, 2])), 4)
    except Exception:
        result["logloss"] = float("nan")
    return result


def generate_backtest_report(
    metrics: Dict,
    walk_forward: Dict,
    per_league: List[Dict],
    reliability: List[Dict],
    top_features: List[Tuple[str, float]],
    conf_matrix: np.ndarray,
    best_params: Dict,
    train_size: int,
    cal_size: int,
    test_size: int,
    variant_comparison: Optional[List[Dict]] = None,
) -> str:
    lines = ["# Backtest Report", ""]
    lines.append("## Split Summary")
    lines.append(f"- Train: {train_size} matches")
    lines.append(f"- Calibration: {cal_size} matches")
    lines.append(f"- Test: {test_size} matches")
    lines.append("")

    lines.append("## Test Set Metrics (held-out)")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k in ["accuracy", "f1_macro", "logloss", "brier"]:
        lines.append(f"| {k} | {metrics.get(k, 'N/A')} |")
    lines.append("")

    lines.append("## Walk-Forward Cross-Validation")
    lines.append(f"Folds: {walk_forward.get('n_folds', 0)}")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|--------|------|-----|")
    for k in ["logloss", "brier", "accuracy", "f1_macro"]:
        m = walk_forward.get(k + "_mean", "N/A")
        s = walk_forward.get(k + "_std", "N/A")
        lines.append(f"| {k} | {m} | {s} |")
    lines.append("")

    lines.append("## Best Hyperparameters")
    for k, v in best_params.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Top Features (by gain)")
    lines.append("| Rank | Feature | Gain |")
    lines.append("|------|---------|------|")
    for i, (name, gain) in enumerate(top_features, 1):
        lines.append(f"| {i} | {name} | {round(gain, 2)} |")
    lines.append("")

    lines.append("## Confusion Matrix (Test Set)")
    labels = ["H", "D", "A"]
    lines.append("| Predicted -> | H | D | A |")
    lines.append("|-------------|---|---|---|")
    for i, label in enumerate(labels):
        row_vals = " | ".join(str(int(conf_matrix[i][j])) for j in range(3))
        lines.append(f"| **Actual {label}** | {row_vals} |")
    lines.append("")

    if per_league:
        lines.append("## Per-League/Season Breakdown")
        header_keys = list(per_league[0].keys())
        lines.append("| " + " | ".join(header_keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_keys)) + " |")
        for row in per_league:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header_keys) + " |")
        lines.append("")

    lines.append("## Calibration (Reliability Bins)")
    lines.append("| Bin | Count | Avg Confidence | Avg Accuracy |")
    lines.append("|-----|-------|----------------|--------------|")
    for b in reliability:
        lines.append(f"| {b['bin_lo']}-{b['bin_hi']} | {b['count']} | {b['avg_conf']} | {b['avg_acc']} |")
    lines.append("")

    if variant_comparison:
        lines.append("## Model Variant Comparison (base vs +odds)")
        header = ["variant", "n_test", "logloss", "brier", "accuracy", "f1_macro"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for v in variant_comparison:
            lines.append("| " + " | ".join(str(v.get(k, "")) for k in header) + " |")
        lines.append("")

    return "\n".join(lines)


def train_and_save_model(
    df_features: pd.DataFrame,
    model_path: Path,
    calibration_method: str = "sigmoid",
    run_hyperparam_search: bool = True,
    n_search_iter: int = 20,
) -> Optional[CalibratedClassifierCV]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = model_path.parent if model_path.parent != Path(".") else MODEL_DIR

    with_odds = use_odds_features()
    feat_cols = get_feature_columns(with_odds=with_odds)
    logger.info("USE_ODDS_FEATURES=%s  feature_count=%d", with_odds, len(feat_cols))

    try:
        df = _prepare_df(df_features, feat_cols)
        X_full = df[feat_cols]
        y_full = df["FTR"].map(CLASS_MAP)

        train_df, cal_df, test_df = _time_split(df)
        X_train, y_train = train_df[feat_cols], train_df["FTR"].map(CLASS_MAP)
        X_cal, y_cal = cal_df[feat_cols], cal_df["FTR"].map(CLASS_MAP)
        X_test, y_test = test_df[feat_cols], test_df["FTR"].map(CLASS_MAP)

        logger.info("Split: train=%d, calibrate=%d, test=%d", len(X_train), len(X_cal), len(X_test))

        best_params: Dict = {}
        if run_hyperparam_search and len(X_train) >= 100:
            logger.info("Running hyperparameter search (%d iterations)...", n_search_iter)
            best_params = _hyperparam_search(X_train, y_train, n_iter=n_search_iter)
            logger.info("Best params: %s", best_params)

        base_model = _make_base_xgb(**best_params)
        base_model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
        logger.info("Base model trained.")

        cal_method = calibration_method
        if cal_method == "isotonic" and len(X_cal) < 100:
            cal_method = "sigmoid"
            logger.info("Too few calibration samples for isotonic; using sigmoid.")

        X_train_cal = pd.concat([X_train, X_cal], ignore_index=True)
        y_train_cal = pd.concat([y_train, y_cal], ignore_index=True)
        calibrated_model = CalibratedClassifierCV(
            estimator=_make_base_xgb(**best_params),
            method=cal_method,
            cv=3,
            ensemble=False,
        )
        calibrated_model.fit(X_train_cal, y_train_cal)
        logger.info("Calibration done (method=%s).", cal_method)

        y_test_arr = np.array(y_test)
        test_proba = calibrated_model.predict_proba(X_test)
        test_preds = calibrated_model.predict(X_test)

        test_metrics: Dict = {
            "accuracy": round(float(accuracy_score(y_test_arr, test_preds)), 4),
            "f1_macro": round(float(f1_score(y_test_arr, test_preds, average="macro", zero_division=0)), 4),
            "brier": round(float(_brier_multiclass(y_test_arr, test_proba)), 4),
        }
        try:
            test_metrics["logloss"] = round(float(log_loss(y_test_arr, test_proba, labels=[0, 1, 2])), 4)
        except Exception:
            test_metrics["logloss"] = float("nan")

        logger.info("TEST accuracy=%.4f  logloss=%.4f  brier=%.4f  f1=%.4f",
                     test_metrics["accuracy"], test_metrics["logloss"],
                     test_metrics["brier"], test_metrics["f1_macro"])

        wf = _walk_forward(X_full, y_full, n_folds=3)
        logger.info("Walk-forward: logloss_mean=%.4f+/-%.4f", wf["logloss_mean"], wf["logloss_std"])

        conf_mat = confusion_matrix(y_test_arr, test_preds, labels=[0, 1, 2])
        reliability = _reliability_bins(y_test_arr, test_proba)
        top_feat = _top_features_by_gain(base_model, list(feat_cols))
        per_league = _per_league_season_metrics(test_df, y_test_arr, test_proba, test_preds)

        variant_comparison: Optional[List[Dict]] = None
        has_odds_col = all(c in df.columns for c in ODDS_FEATURE_COLUMNS)
        if has_odds_col:
            logger.info("Running variant comparison (base vs +odds)...")
            base_eval = _quick_eval(df, list(FEATURE_COLUMNS), "base")
            odds_eval = _quick_eval(df, list(ALL_FEATURE_COLUMNS), "+odds")
            variant_comparison = [base_eval, odds_eval]
            logger.info("base logloss=%.4f  +odds logloss=%.4f",
                        base_eval["logloss"], odds_eval["logloss"])

        report = generate_backtest_report(
            metrics=test_metrics,
            walk_forward=wf,
            per_league=per_league,
            reliability=reliability,
            top_features=top_feat,
            conf_matrix=conf_mat,
            best_params=best_params,
            train_size=len(X_train),
            cal_size=len(X_cal),
            test_size=len(X_test),
            variant_comparison=variant_comparison,
        )
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "backtest_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Backtest report saved to %s", report_path)

        base_model_path = model_dir / MODEL_BASE_FILENAME
        calibrated_model_path = model_dir / MODEL_CALIBRATED_FILENAME
        joblib.dump(base_model, base_model_path)
        joblib.dump(calibrated_model, calibrated_model_path)
        joblib.dump(calibrated_model, model_path)
        logger.info("Models saved: %s, %s, %s", base_model_path, calibrated_model_path, model_path)

        return calibrated_model

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return None


def load_model(model_path: Path) -> Optional[Union[CalibratedClassifierCV, XGBClassifier]]:
    model_dir = model_path.parent if model_path.is_file() or not model_path.exists() else model_path

    calibrated_path = model_dir / MODEL_CALIBRATED_FILENAME
    base_path = model_dir / MODEL_BASE_FILENAME

    if calibrated_path.exists():
        try:
            model = joblib.load(calibrated_path)
            logger.info("Loaded calibrated model from %s", calibrated_path)
            if isinstance(model, CalibratedClassifierCV):
                return model
        except Exception as e:
            logger.warning("Failed to load calibrated model: %s", e)

    if base_path.exists():
        try:
            model = joblib.load(base_path)
            logger.info("Loaded base model from %s", base_path)
            if isinstance(model, XGBClassifier):
                return model
        except Exception as e:
            logger.warning("Failed to load base model: %s", e)

    if not model_path.exists():
        logger.warning("Model file %s not found.", model_path)
        return None

    try:
        model = joblib.load(model_path)
        logger.info("Loaded model from %s (legacy)", model_path)
        if isinstance(model, (CalibratedClassifierCV, XGBClassifier)):
            return model
        logger.error("Invalid model type in %s", model_path)
        return None
    except Exception as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        return None
