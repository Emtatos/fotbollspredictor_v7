"""
Hyperparameter tuning för XGBoost-modellen
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Ladda features och förbered data"""
    df = pd.read_parquet('data/features.parquet')
    
    # Ta bort rader med NaN i target
    df = df.dropna(subset=['FTR'])
    
    # Features
    feature_cols = [
        'HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD',
        'HomeFormHome', 'AwayFormAway',
        'HomeGoalsFor', 'HomeGoalsAgainst', 'AwayGoalsFor', 'AwayGoalsAgainst',
        'HomeStreak', 'AwayStreak',
        'HomeElo', 'AwayElo'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    
    return X, y, df

def tune_hyperparameters():
    """Kör hyperparameter tuning med GridSearchCV"""
    logger.info("=== HYPERPARAMETER TUNING ===")
    
    X, y, df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Träningsdata: {len(X_train)} rader")
    logger.info(f"Testdata: {len(X_test)} rader")
    
    # Beräkna class weights för obalans
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in class_counts.items()}
    
    logger.info(f"Class weights: {class_weights}")
    
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 2, 3]  # För klassobalans
    }
    
    logger.info(f"Testing {np.prod([len(v) for v in param_grid.values()])} kombinationer...")
    
    # GridSearchCV
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Startar grid search...")
    grid_search.fit(X_train, y_train)
    
    # Bästa parametrar
    logger.info(f"\\n=== BÄSTA PARAMETRAR ===")
    logger.info(f"{grid_search.best_params_}")
    
    # Träna final modell med bästa parametrar
    best_model = grid_search.best_estimator_
    
    # Utvärdera
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    logger.info(f"\\n=== RESULTAT ===")
    logger.info(f"Träningsnoggrannhet: {train_acc:.2%}")
    logger.info(f"Testnoggrannhet: {test_acc:.2%}")
    
    logger.info(f"\\n=== CLASSIFICATION REPORT ===")
    logger.info(f"\\n{classification_report(y_test, y_pred_test, target_names=['Hemmavinst', 'Oavgjort', 'Bortavinst'])}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\\n=== FEATURE IMPORTANCE ===")
    logger.info(f"\\n{feature_importance.to_string()}")
    
    # Spara modellen
    model_path = Path("models") / "xgboost_model_v7_2526_tuned.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"\\nTunad modell sparad till: {model_path}")
    
    return best_model, test_acc

if __name__ == "__main__":
    model, accuracy = tune_hyperparameters()
    print(f"\\n🎯 Final träffsäkerhet: {accuracy:.2%}")
