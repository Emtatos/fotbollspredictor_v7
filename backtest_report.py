#!/usr/bin/env python3
"""
Backtest Report - Walk-forward light backtest for fotbollspredictor.

Performs a time-based walk-forward backtest using tertiles (3 time segments)
and reports metrics including entropy-based half-guard hit rates.

Usage:
    python backtest_report.py              # Use cached data (default)
    python backtest_report.py --refresh-data  # Download fresh data

Environment:
    BACKTEST_REFRESH_DATA=1  # Alternative to --refresh-data flag
"""
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from data_processing import normalize_csv_data
from feature_engineering import create_features
from model_handler import _init_xgb_classifier, _fit_with_optional_early_stopping
from schema import FEATURE_COLUMNS, CLASS_MAP, encode_league
from uncertainty import entropy_norm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

N_HALF = 4  # Number of half-guards per test block
CACHE_DIR = Path("data/cache")
LEAGUES = ["E0", "E1", "E2", "E3"]


def compute_brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> float:
    """Compute multiclass Brier score."""
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        if 0 <= label < n_classes:
            y_onehot[i, label] = 1
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def train_model(df_train: pd.DataFrame) -> Optional[CalibratedClassifierCV]:
    """
    Train a calibrated XGBoost model on the given training data.
    
    Returns a model with predict_proba method.
    """
    df_local = df_train.copy()
    
    if "League" in df_local.columns:
        df_local["League"] = df_local["League"].apply(encode_league)
    
    missing = [c for c in FEATURE_COLUMNS if c not in df_local.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return None
    
    X = df_local[FEATURE_COLUMNS]
    y = df_local["FTR"].map(CLASS_MAP)
    
    if len(X) < 50:
        logger.warning("Too few samples for training: %d", len(X))
        return None
    
    # Time-based split for validation (80/20)
    cut = int(len(df_local) * 0.8)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_val, y_val = X.iloc[cut:], y.iloc[cut:]
    
    if len(X_val) < 10:
        X_val, y_val = X_train[-50:], y_train[-50:]
    
    base_model = _init_xgb_classifier()
    _fit_with_optional_early_stopping(base_model, X_train, y_train, X_val, y_val)
    
    # Calibrate with cross-validation
    calibrated_model = CalibratedClassifierCV(
        estimator=_init_xgb_classifier(),
        method="sigmoid",
        cv=3,
        ensemble=False
    )
    calibrated_model.fit(X, y)
    
    return calibrated_model


def predict_with_entropy(
    model: CalibratedClassifierCV,
    df_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict on test data and compute entropy for each match.
    
    Returns:
        y_true: True labels (0=H, 1=D, 2=A)
        y_proba: Probability matrix (n_samples, 3)
        pred_top1: Top-1 predictions
        entropy_values: Entropy for each match
    """
    df_local = df_test.copy()
    
    if "League" in df_local.columns:
        df_local["League"] = df_local["League"].apply(encode_league)
    
    for c in FEATURE_COLUMNS:
        if c not in df_local.columns:
            df_local[c] = 0
    
    X = df_local[FEATURE_COLUMNS]
    y_true = df_local["FTR"].map(CLASS_MAP).values
    
    y_proba = model.predict_proba(X)
    pred_top1 = np.argmax(y_proba, axis=1)
    
    entropy_values = np.array([
        entropy_norm(p[0], p[1], p[2]) for p in y_proba
    ])
    
    return y_true, y_proba, pred_top1, entropy_values


def get_top2_predictions(y_proba: np.ndarray) -> np.ndarray:
    """Get top-2 predictions for each sample."""
    return np.argsort(y_proba, axis=1)[:, -2:]


def compute_block_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pred_top1: np.ndarray,
    entropy_values: np.ndarray,
    n_half: int,
    league_codes: Optional[np.ndarray] = None
) -> Dict:
    """Compute all metrics for a test block."""
    n_matches = len(y_true)
    
    # Top-1 accuracy
    accuracy_top1 = accuracy_score(y_true, pred_top1)
    
    # Select half-guards (highest entropy)
    half_guard_indices = np.argsort(entropy_values)[-n_half:] if n_half > 0 else np.array([])
    
    # Top-2 predictions
    top2_preds = get_top2_predictions(y_proba)
    
    # Accuracy on half-guards (top-2 hit)
    if len(half_guard_indices) > 0:
        half_guard_hits = sum(
            y_true[i] in top2_preds[i] for i in half_guard_indices
        )
        accuracy_top2_on_halfguards = half_guard_hits / len(half_guard_indices)
    else:
        accuracy_top2_on_halfguards = 0.0
    
    # Combined ticket hit rate
    non_half_indices = np.setdiff1d(np.arange(n_matches), half_guard_indices)
    
    top1_hits = sum(y_true[i] == pred_top1[i] for i in non_half_indices)
    top2_hits = sum(y_true[i] in top2_preds[i] for i in half_guard_indices) if len(half_guard_indices) > 0 else 0
    
    combined_hits = top1_hits + top2_hits
    combined_ticket_hit_rate = combined_hits / n_matches if n_matches > 0 else 0.0
    
    # Log loss
    try:
        logloss = log_loss(y_true, y_proba, labels=[0, 1, 2])
    except Exception:
        logloss = float('nan')
    
    # Brier score
    brier = compute_brier_score_multiclass(y_true, y_proba)
    
    metrics = {
        'n_matches': n_matches,
        'accuracy_top1': accuracy_top1,
        'accuracy_top2_on_halfguards': accuracy_top2_on_halfguards,
        'combined_ticket_hit_rate': combined_ticket_hit_rate,
        'logloss': logloss,
        'brier': brier,
    }
    
    # Per-league metrics
    if league_codes is not None:
        unique_leagues = np.unique(league_codes)
        for league in unique_leagues:
            mask = league_codes == league
            if mask.sum() > 0:
                league_acc = accuracy_score(y_true[mask], pred_top1[mask])
                try:
                    league_logloss = log_loss(y_true[mask], y_proba[mask], labels=[0, 1, 2])
                except Exception:
                    league_logloss = float('nan')
                metrics[f'accuracy_{league}'] = league_acc
                metrics[f'logloss_{league}'] = league_logloss
    
    return metrics


def get_seasons() -> List[str]:
    """Get the list of seasons to use for backtest."""
    from main import get_current_season_code
    CURRENT_SEASON = get_current_season_code()
    return [
        str(int(CURRENT_SEASON) - 202),
        str(int(CURRENT_SEASON) - 101),
        CURRENT_SEASON
    ]


def get_cache_filename(league: str, season: str) -> Path:
    """Get the cache file path for a league/season combination."""
    return CACHE_DIR / f"{league}_{season}.csv"


def check_cache_exists() -> Tuple[bool, List[Path], List[Tuple[str, str]]]:
    """
    Check if all required cache files exist.
    
    Returns:
        Tuple of (all_exist, existing_files, missing_combinations)
    """
    seasons = get_seasons()
    existing_files = []
    missing = []
    
    for season in seasons:
        for league in LEAGUES:
            cache_file = get_cache_filename(league, season)
            if cache_file.exists():
                existing_files.append(cache_file)
            else:
                missing.append((league, season))
    
    return len(missing) == 0, existing_files, missing


def download_and_cache_data() -> List[Path]:
    """
    Download data from football-data.co.uk and save to cache.
    
    Returns list of cached file paths.
    """
    from data_loader import download_season_data
    import shutil
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seasons = get_seasons()
    
    logger.info("Downloading data for seasons: %s", seasons)
    cached_files = []
    
    for season in seasons:
        try:
            downloaded_files = download_season_data(season_code=season, leagues=LEAGUES)
            
            for src_file in downloaded_files:
                league = src_file.stem.split("_")[0]
                cache_file = get_cache_filename(league, season)
                
                shutil.copy(src_file, cache_file)
                cached_files.append(cache_file)
                logger.info("Cached: %s -> %s", src_file, cache_file)
                
        except Exception as e:
            logger.error(
                "Failed to download data for season %s. "
                "URL: https://www.football-data.co.uk/mmz4281/%s/<LEAGUE>.csv "
                "Cache path: %s. Error: %s",
                season, season, CACHE_DIR, e
            )
            raise RuntimeError(
                f"Download failed for season {season}. "
                f"Check network connection and try again with --refresh-data"
            ) from e
    
    return cached_files


def load_data_from_cache() -> pd.DataFrame:
    """Load data from cache files."""
    all_exists, existing_files, missing = check_cache_exists()
    
    if not existing_files:
        return pd.DataFrame()
    
    logger.info("Loading %d cached files", len(existing_files))
    df_clean = normalize_csv_data(file_paths=existing_files)
    
    if df_clean.empty:
        logger.error("No data could be normalized from cache")
        return pd.DataFrame()
    
    df_features = create_features(df=df_clean)
    return df_features


def load_data(refresh: bool = False) -> pd.DataFrame:
    """
    Load and prepare data for backtest.
    
    Args:
        refresh: If True, download fresh data. If False, use cache only.
    
    Returns:
        DataFrame with features, or empty DataFrame on failure.
    """
    all_exists, existing_files, missing = check_cache_exists()
    
    if refresh:
        logger.info("Refresh requested - downloading fresh data...")
        try:
            download_and_cache_data()
            return load_data_from_cache()
        except Exception as e:
            logger.error("Download failed: %s", e)
            return pd.DataFrame()
    
    if not all_exists:
        logger.error("=" * 60)
        logger.error("CACHE MISSING - Cannot run backtest without data")
        logger.error("=" * 60)
        logger.error("")
        logger.error("Missing cache files for:")
        for league, season in missing:
            cache_path = get_cache_filename(league, season)
            logger.error("  - %s (season %s): %s", league, season, cache_path)
        logger.error("")
        logger.error("To download data, run one of:")
        logger.error("  python backtest_report.py --refresh-data")
        logger.error("  BACKTEST_REFRESH_DATA=1 python backtest_report.py")
        logger.error("")
        logger.error("Cache directory: %s", CACHE_DIR.absolute())
        logger.error("=" * 60)
        return pd.DataFrame()
    
    logger.info("Using cached data from %s", CACHE_DIR)
    return load_data_from_cache()


def run_backtest(df: pd.DataFrame, n_folds: int = 3) -> List[Dict]:
    """
    Run walk-forward backtest with time-based splits.
    
    Uses tertiles (3 segments) by default for faster CI.
    """
    if df.empty:
        logger.error("Empty dataframe for backtest")
        return []
    
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    # Create time-based folds using quantiles
    df["time_fold"] = pd.qcut(df["Date"], q=n_folds, labels=False, duplicates='drop')
    
    all_metrics = []
    
    for fold_idx in range(1, n_folds):
        train_mask = df["time_fold"] < fold_idx
        test_mask = df["time_fold"] == fold_idx
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        if len(df_train) < 100 or len(df_test) < 20:
            logger.warning("Fold %d: insufficient data (train=%d, test=%d)", 
                         fold_idx, len(df_train), len(df_test))
            continue
        
        logger.info("Fold %d: training on %d matches, testing on %d matches",
                   fold_idx, len(df_train), len(df_test))
        
        model = train_model(df_train)
        if model is None:
            logger.warning("Fold %d: model training failed", fold_idx)
            continue
        
        y_true, y_proba, pred_top1, entropy_values = predict_with_entropy(model, df_test)
        
        # Get league codes for per-league metrics
        league_codes = df_test["League"].apply(encode_league).values if "League" in df_test.columns else None
        
        metrics = compute_block_metrics(
            y_true, y_proba, pred_top1, entropy_values,
            n_half=min(N_HALF, len(y_true) // 4),
            league_codes=league_codes
        )
        metrics['fold'] = fold_idx
        all_metrics.append(metrics)
    
    return all_metrics


def print_report(all_metrics: List[Dict]) -> None:
    """Print a formatted backtest report."""
    if not all_metrics:
        print("\n" + "=" * 60)
        print("BACKTEST REPORT - NO RESULTS")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("BACKTEST REPORT - Walk-Forward (Tertiles)")
    print("=" * 60)
    
    # Per-fold results
    print("\n--- Per-Fold Results ---\n")
    print(f"{'Fold':<6} {'N':<6} {'Acc_Top1':<10} {'Acc_Top2_HG':<12} {'Combined':<10} {'LogLoss':<10} {'Brier':<10}")
    print("-" * 70)
    
    for m in all_metrics:
        print(f"{m['fold']:<6} {m['n_matches']:<6} {m['accuracy_top1']:<10.4f} "
              f"{m['accuracy_top2_on_halfguards']:<12.4f} {m['combined_ticket_hit_rate']:<10.4f} "
              f"{m['logloss']:<10.4f} {m['brier']:<10.4f}")
    
    # Aggregate metrics
    total_matches = sum(m['n_matches'] for m in all_metrics)
    avg_acc_top1 = np.mean([m['accuracy_top1'] for m in all_metrics])
    avg_acc_top2_hg = np.mean([m['accuracy_top2_on_halfguards'] for m in all_metrics])
    avg_combined = np.mean([m['combined_ticket_hit_rate'] for m in all_metrics])
    avg_logloss = np.mean([m['logloss'] for m in all_metrics if not np.isnan(m['logloss'])])
    avg_brier = np.mean([m['brier'] for m in all_metrics])
    
    print("-" * 70)
    print(f"{'TOTAL':<6} {total_matches:<6} {avg_acc_top1:<10.4f} "
          f"{avg_acc_top2_hg:<12.4f} {avg_combined:<10.4f} "
          f"{avg_logloss:<10.4f} {avg_brier:<10.4f}")
    
    # Per-league breakdown
    print("\n--- Per-League Breakdown (Average across folds) ---\n")
    
    league_names = {0: "E0 (PL)", 1: "E1 (Champ)", 2: "E2 (L1)", 3: "E3 (L2)"}
    
    print(f"{'League':<15} {'Accuracy':<12} {'LogLoss':<12}")
    print("-" * 40)
    
    for league_code, league_name in league_names.items():
        acc_key = f'accuracy_{league_code}'
        ll_key = f'logloss_{league_code}'
        
        accs = [m.get(acc_key) for m in all_metrics if m.get(acc_key) is not None]
        lls = [m.get(ll_key) for m in all_metrics if m.get(ll_key) is not None and not np.isnan(m.get(ll_key, float('nan')))]
        
        if accs:
            avg_acc = np.mean(accs)
            avg_ll = np.mean(lls) if lls else float('nan')
            print(f"{league_name:<15} {avg_acc:<12.4f} {avg_ll:<12.4f}")
    
    print("\n" + "=" * 60)
    print("LEGEND:")
    print("  Acc_Top1      = Top-1 accuracy (argmax prediction)")
    print("  Acc_Top2_HG   = Top-2 accuracy on half-guards (entropy-selected)")
    print("  Combined      = Combined ticket hit rate (top1 + top2 for HG)")
    print("  LogLoss       = Multiclass log loss")
    print("  Brier         = Multiclass Brier score")
    print(f"  N_HALF        = {N_HALF} matches per fold selected as half-guards")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest for fotbollspredictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_report.py              # Use cached data (default)
  python backtest_report.py --refresh-data  # Download fresh data

Environment variables:
  BACKTEST_REFRESH_DATA=1  # Alternative to --refresh-data flag

Cache location: data/cache/
        """
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh data instead of using cache"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    refresh = args.refresh_data or os.environ.get("BACKTEST_REFRESH_DATA", "").lower() in ("1", "true", "yes")
    
    if refresh:
        logger.info("Starting backtest report with data refresh...")
    else:
        logger.info("Starting backtest report using cached data...")
    
    df = load_data(refresh=refresh)
    if df.empty:
        logger.error("Failed to load data")
        return 1
    
    logger.info("Loaded %d matches", len(df))
    
    all_metrics = run_backtest(df, n_folds=3)
    
    print_report(all_metrics)
    
    if not all_metrics:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
