# model_handler.py
from pathlib import Path
import pandas as pd
import joblib
import logging
from typing import Optional, Tuple

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import inspect

# Konfigurera logger för modulen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = Path("models")


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


def train_and_save_model(
    df_features: pd.DataFrame,
    model_path: Path
) -> Optional[XGBClassifier]:
    """
    Tränar en XGBoost-klassificeringsmodell och sparar den till disk.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Features & mål (inkludera alla nya features)
        feature_cols = [
            'HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD',
            'HomeFormHome', 'AwayFormAway',
            'HomeGoalsFor', 'HomeGoalsAgainst', 'AwayGoalsFor', 'AwayGoalsAgainst',
            'HomeStreak', 'AwayStreak',
            'HomeElo', 'AwayElo'
        ]
        X = df_features[feature_cols]
        y = df_features["FTR"].map({"H": 0, "D": 1, "A": 2})

        # 2) Train/val-split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info("Träningsdata: %d rader. Valideringsdata: %d rader.", len(X_train), len(X_val))

        # 3) Initiera och träna modell (kompatibel)
        model = _init_xgb_classifier()
        logging.info("Startar modellträning...")
        _fit_with_optional_early_stopping(model, X_train, y_train, X_val, y_val)
        logging.info("Modellträning slutförd.")

        # 4) Utvärdera
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        logging.info("Modellens träffsäkerhet på valideringsdata: %.2f%%", accuracy * 100)

        # 5) Spara
        joblib.dump(model, model_path)
        logging.info("Modellen har sparats till: %s", model_path)
        return model

    except Exception as e:
        logging.error("Ett fel inträffade under träning eller lagring av modellen: %s", e, exc_info=True)
        return None


def load_model(model_path: Path) -> Optional[XGBClassifier]:
    """
    Laddar en tränad modell från disk.
    """
    if not model_path.exists():
        logging.warning("Modellfilen %s hittades inte.", model_path)
        return None

    try:
        model = joblib.load(model_path)
        logging.info("Modellen har laddats från: %s", model_path)
        if isinstance(model, XGBClassifier):
            return model
        logging.error("Filen %s är inte en giltig XGBClassifier-modell.", model_path)
        return None
    except Exception as e:
        logging.error("Kunde inte ladda modell från %s: %s", model_path, e)
        return None
