import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle
import os

def generate_sample_data(n=10000, seed=42):
    """
    Generate realistic synthetic patient data.
    Labels are based on actual symptom/lab logic - NOT random.
    0 = No Flu, 1 = Possible Flu, 2 = Possible Bird Flu
    """
    np.random.seed(seed)

    # Generate features
    wbc          = np.random.normal(7000, 1500, n)
    rbc          = np.random.normal(4.5, 0.5, n)
    hemoglobin   = np.random.normal(13.5, 1.5, n)
    platelet     = np.random.normal(250000, 50000, n)
    crp          = np.abs(np.random.normal(5, 4, n))
    alt          = np.abs(np.random.normal(30, 10, n))
    ast          = np.abs(np.random.normal(28, 10, n))
    urine_ph     = np.clip(np.random.normal(6.0, 0.5, n), 4.0, 8.5)
    fever        = np.random.randint(0, 2, n)
    cough        = np.random.randint(0, 2, n)
    sore_throat  = np.random.randint(0, 2, n)

    df = pd.DataFrame({
        "WBC_count":    np.clip(wbc, 2000, 20000),
        "RBC_count":    np.clip(rbc, 2.5, 6.0),
        "Hemoglobin":   np.clip(hemoglobin, 7.0, 18.0),
        "Platelet_count": np.clip(platelet, 50000, 500000),
        "CRP":          np.clip(crp, 0.0, 50.0),
        "ALT":          np.clip(alt, 0.0, 100.0),
        "AST":          np.clip(ast, 0.0, 100.0),
        "Urine_pH":     urine_ph,
        "Fever":        fever,
        "Cough":        cough,
        "Sore_throat":  sore_throat,
    })

    # --- Label logic (realistic clinical rules) ---
    # Bird flu indicators: low WBC, high CRP, high liver enzymes, fever + cough together
    # Key insight: LABS ALONE can indicate bird flu even without classic symptoms
    labels = np.zeros(n, dtype=int)  # default: No Flu

    # Score-based classification
    score = np.zeros(n)

    # Symptoms (contribute to flu/bird flu)
    score += (df["Fever"] == 1).astype(float) * 2.0
    score += (df["Cough"] == 1).astype(float) * 1.5
    score += (df["Sore_throat"] == 1).astype(float) * 1.0

    # Lab markers - CRITICAL: these alone can indicate bird flu
    score += (df["WBC_count"] < 5000).astype(float) * 3.0    # leukopenia = strong bird flu sign
    score += (df["WBC_count"] < 4000).astype(float) * 1.5    # severe leukopenia = extra weight
    score += (df["CRP"] > 15).astype(float) * 2.5            # high inflammation
    score += (df["CRP"] > 25).astype(float) * 1.5            # very high CRP = extra weight
    score += (df["ALT"] > 60).astype(float) * 2.0            # liver stress (bird flu damages liver)
    score += (df["AST"] > 55).astype(float) * 2.0            # liver stress
    score += (df["Platelet_count"] < 150000).astype(float) * 2.0   # thrombocytopenia = bird flu sign
    score += (df["Platelet_count"] < 120000).astype(float) * 1.0   # severe thrombocytopenia

    # Combined lab pattern: low WBC + low platelets + high liver = strong bird flu signal
    # even without fever/cough (silent/early bird flu)
    lab_pattern = (
        (df["WBC_count"] < 5000) &
        (df["Platelet_count"] < 160000) &
        (df["CRP"] > 12)
    ).astype(float)
    score += lab_pattern * 3.0   # bonus for the combined dangerous pattern

    # Add small noise so boundary isn't perfectly sharp
    score += np.random.normal(0, 0.4, n)

    labels[score >= 3.0] = 1   # Possible Flu
    labels[score >= 7.0] = 2   # Possible Bird Flu

    df["Target"] = labels
    return df

df = generate_sample_data(10000)
print(f"Generated {len(df)} rows")
print(f"Class distribution: {dict(zip(*np.unique(df['Target'], return_counts=True)))}")

X = df.drop("Target", axis=1)
y = df["Target"]

# --- Ensemble model ---
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight='balanced',
    random_state=42, n_jobs=-1
)
xgb = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0
)
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)
print("Training ensemble model on 10,000 rows...")
ensemble.fit(X, y)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(ensemble, f)

print(f"Model saved to {model_path}")
