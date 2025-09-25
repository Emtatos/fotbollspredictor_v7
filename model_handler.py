# model_handler.py
from pathlib import Path
import pandas as pd
import joblib
import logging
from typing import Optional, Tuple

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Konfigurera logger för modulen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = Path("models")

def train_and_save_model(
    df_features: pd.DataFrame, 
    model_path: Path
) -> Optional[XGBClassifier]:
    """
    Tränar en XGBoost-klassificeringsmodell och sparar den till disk.

    Parametrar
    ----------
    df_features : pd.DataFrame
        En DataFrame som innehåller features (form, ELO, etc.) och målvariabeln ('FTR').
    model_path : Path
        Fullständig sökväg där den tränade modellen ska sparas.

    Returnerar
    -------
    Optional[XGBClassifier]
        Den tränade modell-objektet om träningen lyckades, annars None.
    """
    # Se till att målmappen för modellen finns
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Definiera features (X) och target (y)
        feature_cols = [
            'HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD', 
            'HomeElo', 'AwayElo'
        ]
        X = df_features[feature_cols]
        y = df_features['FTR'].map({'H': 0, 'D': 1, 'A': 2})

        # 2. Dela upp data i träning- och validerings-set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info("Träningsdata: %d rader. Valideringsdata: %d rader.", len(X_train), len(X_val))

        # 3. Initiera och träna XGBoost-modellen
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=250,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,  # Använd alla tillgängliga CPU-kärnor
            random_state=42
        )

        logging.info("Startar modellträning...")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        logging.info("Modellträning slutförd.")

        # 4. Utvärdera modellen på valideringsdatan
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        logging.info("Modellens träffsäkerhet på valideringsdata: %.2f%%", accuracy * 100)

        # 5. Spara den tränade modellen
        joblib.dump(model, model_path)
        logging.info("Modellen har sparats till: %s", model_path)
        
        return model

    except Exception as e:
        logging.error("Ett fel inträffade under träning eller lagring av modellen: %s", e, exc_info=True)
        return None


def load_model(model_path: Path) -> Optional[XGBClassifier]:
    """
    Laddar en tränad modell från disk.

    Parametrar
    ----------
    model_path : Path
        Sökväg till den sparade modellfilen.

    Returnerar
    -------
    Optional[XGBClassifier]
        Det laddade modell-objektet om det lyckades, annars None.
    """
    if not model_path.exists():
        logging.warning("Modellfilen %s hittades inte.", model_path)
        return None
    
    try:
        model = joblib.load(model_path)
        logging.info("Modellen har laddats från: %s", model_path)
        if isinstance(model, XGBClassifier):
            return model
        else:
            logging.error("Filen %s är inte en giltig XGBClassifier-modell.", model_path)
            return None
    except Exception as e:
        logging.error("Kunde inte ladda modell från %s: %s", model_path, e)
        return None
