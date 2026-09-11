
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL = joblib.load(MODEL_DIR / 'rf_model.pkl')

def predict_from_dict(d):
    # d is a dict with same schema as training features (without id/infected)
    df = pd.DataFrame([d])
    # Apply one-hot encoding to match training
    df = pd.get_dummies(df)
    # Align columns with model's expected features
    df = df.reindex(columns=MODEL.feature_names_in_, fill_value=0)
    proba = MODEL.predict_proba(df)[:,1][0]
    return float(proba)
