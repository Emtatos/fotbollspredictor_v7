# model_handler.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional, Tuple, Union

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

from schema import CLASS_MAP, FEATURE_COLUMNS, encode_league
import inspect

# Konfigurera logger för modulen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = Path("models")

# Model file names
MODEL_BASE_FILENAME = "model_base.pkl"
MODEL_CALIBRATED_FILENAME = "model_calibrated.pkl"
# Legacy filename for backward compatibility
MODEL_LEGACY_FILENAME = "model.joblib"


def _init_xgb_classifier() -> XGBClassifier:
    """
    Initiera XGBClassifier med fallback om vissa parametrar inte stöds
    (t.ex. 'use_label_encoder' borttagen i nyare versioner).
    """
    # Optimerade hyperparametrar från tuning
    base_kwargs = dict(
        objective="multi:softprob",
        n_estimators=100,  # Mindre estimators för snabbare träning
        learning_rate=0.01,  # Långsammare inlärning för bättre generalisering
        max_depth=3,  # Mindre djup för att undvika överanpassning
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    try:
        # Prova med use_label_encoder (finns i äldre xgboost)
        return XGBClassifier(use_label_encoder=False, **base_kwargs)
    except TypeError:
        # Nyare xgboost (>2.0) stöder inte use_label_encoder
        logging.info("XGBoost utan 'use_label_encoder' (nyare version) används.")
        return XGBClassifier(**base_kwargs)


def _fit_with_optional_early_stopping(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """
    Träna modellen och använd 'early_stopping_rounds' endast om metoden stöder det.
    Faller tillbaka till vanlig fit() annars.
    """
    fit_sig = inspect.signature(model.fit)
    kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}

    if "early_stopping_rounds" in fit_sig.parameters:
        kwargs["early_stopping_rounds"] = 20

    try:
        model.fit(X_train, y_train, **kwargs)
    except TypeError as e:
        # Om vi ändå råkar skicka ett ogiltigt arg, kör utan det.
        if "early_stopping_rounds" in kwargs:
            logging.info("Tidigare XGBoost saknar early_stopping_rounds i fit(); provar utan.")
            kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **kwargs)
        else:
            raise


def _compute_brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> float:
    """
    Compute multiclass Brier score.
    
    Brier score = mean(sum((p - y_onehot)^2)) per sample
    Lower is better. Range: [0, 2] for 3 classes.
    """
    # One-hot encode y_true
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        if 0 <= label < n_classes:
            y_onehot[i, label] = 1
    
    # Compute Brier score
    brier = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
    return brier


def train_and_save_model(
    df_features: pd.DataFrame,
    model_path: Path,
    calibration_method: str = "sigmoid"
) -> Optional[CalibratedClassifierCV]:
    """
    Tränar en XGBoost-klassificeringsmodell, kalibrerar den med Platt scaling (sigmoid)
    eller isotonic regression, och sparar båda modellerna till disk.
    
    Args:
        df_features: DataFrame med features och FTR-kolumn
        model_path: Sökväg för att spara modellen (används för att bestämma katalog)
        calibration_method: "sigmoid" (Platt scaling) eller "isotonic"
    
    Returns:
        Den kalibrerade modellen, eller None vid fel
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine model directory from model_path
    model_dir = model_path.parent if model_path.parent != Path(".") else MODEL_DIR

    try:
        # 1) Features & mål (single source of truth via schema.py)
        df_local = df_features.copy()

        # League: säkerställ numeriskt
        if "League" in df_local.columns:
            df_local["League"] = df_local["League"].apply(encode_league)

        # Säkerställ att alla features finns (fyll annars med 0)
        missing = [c for c in FEATURE_COLUMNS if c not in df_local.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = df_local[FEATURE_COLUMNS]
        y = df_local["FTR"].map(CLASS_MAP)

        # 2) Tidsbaserad train/val-split (undvik dataläckage)
        if "Date" in df_local.columns:
            df_local["Date"] = pd.to_datetime(df_local["Date"], errors="coerce")
            df_local = df_local.dropna(subset=["Date"]).sort_values("Date", ascending=True)
            split_point = df_local["Date"].quantile(0.8)
            train_df = df_local[df_local["Date"] <= split_point]
            val_df = df_local[df_local["Date"] > split_point]
            if len(val_df) == 0 or len(train_df) == 0:
                # Fallback om quantile ger tomt
                cut = int(len(df_local) * 0.8)
                train_df = df_local.iloc[:cut]
                val_df = df_local.iloc[cut:]
            X_train, y_train = train_df[FEATURE_COLUMNS], train_df["FTR"].map(CLASS_MAP)
            X_val, y_val = val_df[FEATURE_COLUMNS], val_df["FTR"].map(CLASS_MAP)
        else:
            # Fallback: enkel split på index
            cut = int(len(df_local) * 0.8)
            X_train, y_train = X.iloc[:cut], y.iloc[:cut]
            X_val, y_val = X.iloc[cut:], y.iloc[cut:]

        logging.info("Träningsdata: %d rader. Valideringsdata: %d rader.", len(X_train), len(X_val))

        # 3) Initiera och träna basmodell
        base_model = _init_xgb_classifier()
        logging.info("Startar basmodellträning...")
        _fit_with_optional_early_stopping(base_model, X_train, y_train, X_val, y_val)
        logging.info("Basmodellträning slutförd.")

        # 4) Utvärdera basmodell på val-set
        y_val_arr = np.array(y_val)
        base_proba = base_model.predict_proba(X_val)
        
        try:
            logloss_base = log_loss(y_val, base_proba, labels=[0, 1, 2])
        except Exception:
            logloss_base = float('nan')
        
        brier_base = _compute_brier_score_multiclass(y_val_arr, base_proba)
        
        preds_base = base_model.predict(X_val)
        accuracy_base = accuracy_score(y_val, preds_base)
        
        logging.info("=== BASMODELL METRICS (före kalibrering) ===")
        logging.info("VAL_ACCURACY_BASE=%.4f", accuracy_base)
        logging.info("VAL_LOGLOSS_BASE=%.4f", logloss_base if logloss_base == logloss_base else float('nan'))
        logging.info("VAL_BRIER_BASE=%.4f", brier_base)

        # 5) Kalibrera modellen med CalibratedClassifierCV
        # Använd cv=2 för att kalibrera på val-set (sklearn 1.8+ stöder inte cv="prefit")
        logging.info("Startar kalibrering med metod: %s", calibration_method)
        
        # Skapa en ny kalibrerad modell som tränas med cross-validation
        # Vi använder ensemble=False för att få en enda kalibrerad modell
        calibrated_model = CalibratedClassifierCV(
            estimator=_init_xgb_classifier(),
            method=calibration_method,
            cv=3,
            ensemble=False
        )
        # Träna kalibrerad modell på hela datasetet (train + val)
        X_all = pd.concat([X_train, X_val], ignore_index=True)
        y_all = pd.concat([y_train, y_val], ignore_index=True)
        calibrated_model.fit(X_all, y_all)
        logging.info("Kalibrering slutförd.")

        # 6) Utvärdera kalibrerad modell på val-set
        cal_proba = calibrated_model.predict_proba(X_val)
        
        try:
            logloss_cal = log_loss(y_val, cal_proba, labels=[0, 1, 2])
        except Exception:
            logloss_cal = float('nan')
        
        brier_cal = _compute_brier_score_multiclass(y_val_arr, cal_proba)
        
        preds_cal = calibrated_model.predict(X_val)
        accuracy_cal = accuracy_score(y_val, preds_cal)
        
        logging.info("=== KALIBRERAD MODELL METRICS (efter kalibrering) ===")
        logging.info("VAL_ACCURACY_CAL=%.4f", accuracy_cal)
        logging.info("VAL_LOGLOSS_CAL=%.4f", logloss_cal if logloss_cal == logloss_cal else float('nan'))
        logging.info("VAL_BRIER_CAL=%.4f", brier_cal)
        
        # Log improvement
        logloss_improvement = logloss_base - logloss_cal if (logloss_base == logloss_base and logloss_cal == logloss_cal) else 0
        brier_improvement = brier_base - brier_cal
        logging.info("=== FÖRBÄTTRING ===")
        logging.info("LOGLOSS_IMPROVEMENT=%.4f (lägre är bättre)", logloss_improvement)
        logging.info("BRIER_IMPROVEMENT=%.4f (lägre är bättre)", brier_improvement)

        # 7) Spara båda modellerna
        base_model_path = model_dir / MODEL_BASE_FILENAME
        calibrated_model_path = model_dir / MODEL_CALIBRATED_FILENAME
        
        joblib.dump(base_model, base_model_path)
        logging.info("Basmodellen har sparats till: %s", base_model_path)
        
        joblib.dump(calibrated_model, calibrated_model_path)
        logging.info("Kalibrerad modell har sparats till: %s", calibrated_model_path)
        
        # Also save to the legacy path for backward compatibility
        joblib.dump(calibrated_model, model_path)
        logging.info("Kalibrerad modell har också sparats till: %s (legacy path)", model_path)
        
        return calibrated_model

    except Exception as e:
        logging.error("Ett fel inträffade under träning eller lagring av modellen: %s", e, exc_info=True)
        return None


def load_model(model_path: Path) -> Optional[Union[CalibratedClassifierCV, XGBClassifier]]:
    """
    Laddar en tränad modell från disk.
    
    Prioriterar kalibrerad modell (model_calibrated.pkl) om den finns i samma katalog.
    Faller tillbaka till basmodell (model_base.pkl) eller angiven sökväg.
    
    Args:
        model_path: Sökväg till modellen (eller katalog att söka i)
    
    Returns:
        Kalibrerad modell om tillgänglig, annars basmodell, eller None vid fel
    """
    model_dir = model_path.parent if model_path.is_file() or not model_path.exists() else model_path
    
    # Priority order: calibrated > base > legacy path
    calibrated_path = model_dir / MODEL_CALIBRATED_FILENAME
    base_path = model_dir / MODEL_BASE_FILENAME
    
    # Try calibrated model first
    if calibrated_path.exists():
        try:
            model = joblib.load(calibrated_path)
            logging.info("Kalibrerad modell har laddats från: %s", calibrated_path)
            if isinstance(model, CalibratedClassifierCV):
                return model
            logging.warning("Filen %s är inte en giltig CalibratedClassifierCV.", calibrated_path)
        except Exception as e:
            logging.warning("Kunde inte ladda kalibrerad modell från %s: %s", calibrated_path, e)
    
    # Try base model
    if base_path.exists():
        try:
            model = joblib.load(base_path)
            logging.info("Basmodell har laddats från: %s", base_path)
            if isinstance(model, XGBClassifier):
                return model
            logging.warning("Filen %s är inte en giltig XGBClassifier.", base_path)
        except Exception as e:
            logging.warning("Kunde inte ladda basmodell från %s: %s", base_path, e)
    
    # Fallback to legacy path
    if not model_path.exists():
        logging.warning("Modellfilen %s hittades inte.", model_path)
        return None

    try:
        model = joblib.load(model_path)
        logging.info("Modellen har laddats från: %s (legacy path)", model_path)
        if isinstance(model, (CalibratedClassifierCV, XGBClassifier)):
            return model
        logging.error("Filen %s är inte en giltig modell.", model_path)
        return None
    except Exception as e:
        logging.error("Kunde inte ladda modell från %s: %s", model_path, e)
        return None
