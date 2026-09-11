import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from preprocess import load_and_preprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / 'models'
DATA_DIR  = BASE / 'data'
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def train_and_save():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess(path=DATA_DIR / 'synthetic_data.csv')
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Infected in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    # --- Model 1: Random Forest (tuned) ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # --- Model 2: XGBoost (2024+ best practice) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # handles class imbalance
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # --- Model 3: Gradient Boosting ---
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    # Evaluate each individually with cross-validation
    print("\nCross-validating models (5-fold ROC-AUC)...")
    for name, clf in [("RandomForest", rf), ("XGBoost", xgb), ("GradBoost", gb)]:
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # --- Ensemble: Voting Classifier (soft voting) ---
    print("\nTraining ensemble model...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)

    # Evaluate on test set
    p = ensemble.predict_proba(X_test)[:, 1]
    preds = (p >= 0.5).astype(int)
    auc = roc_auc_score(y_test, p)

    print(f"\n=== FINAL ENSEMBLE TEST RESULTS ===")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(ensemble, MODEL_DIR / 'rf_model.pkl')
    print(f"Model saved to {MODEL_DIR / 'rf_model.pkl'}")

if __name__ == '__main__':
    train_and_save()
