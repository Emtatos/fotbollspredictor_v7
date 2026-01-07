"""
Ensemble-modell som kombinerar XGBoost och Random Forest
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from typing import Optional

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def train_ensemble_model(df_features: pd.DataFrame, model_path: Path) -> Optional[VotingClassifier]:
    """
    Tränar en ensemble-modell som kombinerar XGBoost och Random Forest
    
    Parametrar:
    -----------
    df_features : pd.DataFrame
        DataFrame med features
    model_path : Path
        Sökväg där modellen ska sparas
    
    Returnerar:
    -----------
    VotingClassifier eller None om träningen misslyckades
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1) Features & mål
        feature_cols = [
            'HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD',
            'HomeFormHome', 'AwayFormAway',
            'HomeGoalsFor', 'HomeGoalsAgainst', 'AwayGoalsFor', 'AwayGoalsAgainst',
            'HomeStreak', 'AwayStreak',
            'H2H_HomeWins', 'H2H_Draws', 'H2H_AwayWins', 'H2H_HomeGoalDiff',
            'HomePosition', 'AwayPosition', 'PositionDiff',
            'HomeElo', 'AwayElo'
        ]
        
        X = df_features[feature_cols]
        y = df_features["FTR"].map({"H": 0, "D": 1, "A": 2})
        
        # 2) Train/val-split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("Träningsdata: %d rader. Valideringsdata: %d rader.", len(X_train), len(X_val))
        
        # 3) Skapa individuella modeller
        logger.info("Skapar XGBoost-modell...")
        xgb_model = XGBClassifier(
            objective='multi:softprob',
            n_estimators=150,  # Fler estimators
            learning_rate=0.05,  # Lite snabbare inlärning
            max_depth=4,  # Lite mer komplexitet
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False
        )
        
        logger.info("Skapar Random Forest-modell...")
        rf_model = RandomForestClassifier(
            n_estimators=300,  # Fler träd
            max_depth=15,  # Mer djup
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',  # Hantera klassobalans
            n_jobs=-1,
            random_state=42
        )
        
        # 4) Skapa ensemble med soft voting (använder sannolikheter)
        logger.info("Skapar ensemble-modell med soft voting...")
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model)
            ],
            voting='soft',  # Använd sannolikheter istället för hårda röster
            weights=[1.2, 0.8]  # Ge XGBoost lite mer vikt (kan justeras)
        )
        
        # 5) Träna ensemble
        logger.info("Tränar ensemble-modell...")
        ensemble.fit(X_train, y_train)
        logger.info("Ensemble-träning slutförd.")
        
        # 6) Utvärdera
        preds = ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        logger.info("Ensemble träffsäkerhet på valideringsdata: %.2f%%", accuracy * 100)
        
        # Visa detaljerad rapport
        logger.info("\n=== CLASSIFICATION REPORT ===")
        target_names = ['Hemmavinst', 'Oavgjort', 'Bortavinst']
        report = classification_report(y_val, preds, target_names=target_names)
        logger.info("\n" + report)
        
        # 7) Utvärdera individuella modeller för jämförelse
        logger.info("\n=== INDIVIDUELLA MODELLER ===")
        # Hämta modeller från ensemble
        xgb_fitted = ensemble.named_estimators_['xgb']
        rf_fitted = ensemble.named_estimators_['rf']
        
        xgb_preds = xgb_fitted.predict(X_val)
        rf_preds = rf_fitted.predict(X_val)
        
        xgb_acc = accuracy_score(y_val, xgb_preds)
        rf_acc = accuracy_score(y_val, rf_preds)
        
        logger.info("XGBoost träffsäkerhet: %.2f%%", xgb_acc * 100)
        logger.info("Random Forest träffsäkerhet: %.2f%%", rf_acc * 100)
        logger.info("Ensemble träffsäkerhet: %.2f%%", accuracy * 100)
        
        # 8) Spara
        joblib.dump(ensemble, model_path)
        logger.info("Ensemble-modellen har sparats till: %s", model_path)
        
        return ensemble
        
    except Exception as e:
        logger.error("Ett fel inträffade under träning av ensemble: %s", e, exc_info=True)
        return None


def load_ensemble_model(model_path: Path) -> Optional[VotingClassifier]:
    """Laddar en tränad ensemble-modell från disk"""
    if not model_path.exists():
        logger.warning("Modellfilen %s hittades inte.", model_path)
        return None
    
    try:
        model = joblib.load(model_path)
        logger.info("Ensemble-modellen har laddats från: %s", model_path)
        return model
    except Exception as e:
        logger.error("Kunde inte ladda ensemble-modell från %s: %s", model_path, e)
        return None


if __name__ == "__main__":
    # Testa ensemble-modellen
    import pandas as pd
    
    logger.info("=== TRÄNAR ENSEMBLE-MODELL ===")
    
    # Ladda features
    features_path = Path("data/features.parquet")
    if not features_path.exists():
        logger.error("Features-fil saknas. Kör main.py först.")
        exit(1)
    
    df_features = pd.read_parquet(features_path)
    logger.info("Laddade %d matcher från features.parquet", len(df_features))
    
    # Träna ensemble
    model_path = Path("models") / "ensemble_model_v7_2526.joblib"
    ensemble = train_ensemble_model(df_features, model_path)
    
    if ensemble:
        logger.info("\n✅ Ensemble-modell tränad och sparad framgångsrikt!")
    else:
        logger.error("\n❌ Ensemble-träning misslyckades")
