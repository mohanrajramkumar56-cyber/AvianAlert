import pandas as pd
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_ORDER = [
    "WBC_count", "RBC_count", "Hemoglobin", "Platelet_count",
    "CRP", "ALT", "AST", "Urine_pH", "Fever", "Cough", "Sore_throat"
]

def predict_from_dict(data_dict):
    """
    Returns predicted class: 0=No Flu, 1=Possible Flu, 2=Possible Bird Flu
    Uses DataFrame so feature names match exactly what the model was trained on.
    """
    df = pd.DataFrame([{k: data_dict[k] for k in FEATURE_ORDER}])
    prediction = model.predict(df)[0]
    return int(prediction)

def predict_proba_from_dict(data_dict):
    """Returns class probabilities as a list [p_no_flu, p_flu, p_bird_flu]"""
    df = pd.DataFrame([{k: data_dict[k] for k in FEATURE_ORDER}])
    proba = model.predict_proba(df)[0]
    return [float(p) for p in proba]
