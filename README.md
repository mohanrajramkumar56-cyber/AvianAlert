# 🦠 AvianAlert — AI-Powered Bird Flu (H5N1) Screening System

> **Prototype project using synthetic data. Not for real medical use.**

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution](#-solution)
3. [How It Works](#-how-it-works)
4. [Project Structure](#-project-structure)
5. [Tech Stack](#-tech-stack)
6. [ML Model Details](#-ml-model-details)
7. [Accuracy & Test Results](#-accuracy--test-results)
8. [Setup & Installation](#-setup--installation)
9. [Running the App](#-running-the-app)
10. [Running Tests](#-running-tests)
11. [Screenshots](#-screenshots)
12. [Known Limitations](#-known-limitations)
13. [Future Improvements](#-future-improvements)

---

## 🚨 Problem Statement

**Bird Flu (H5N1 Avian Influenza)** is a highly dangerous virus that spreads from birds to humans. It has a **~60% fatality rate** in confirmed human cases — far higher than seasonal flu.

The key challenges in detecting it early are:

- **Symptoms look like regular flu** — fever, cough, sore throat. Doctors can easily miss it.
- **Lab tests are expensive and slow** — PCR testing takes 24–48 hours and isn't always available.
- **Rural and low-resource settings** have no rapid screening tool.
- **Early cases go undetected** — patients who work on poultry farms or had bird contact may not even mention it.
- **Silent bird flu is dangerous** — some patients have critically abnormal blood values (low WBC, low platelets, high liver enzymes) but no classic symptoms yet.

> **There was no fast, accessible, first-line screening tool that combines symptoms + blood test results + exposure history to flag high-risk patients immediately.**

---

## ✅ Solution

**AvianAlert** is an AI-powered bird flu screening web application that:

- Takes a patient's **symptoms**, **blood test results**, and **exposure history** as input
- Runs them through a trained **ensemble ML model** (Random Forest + XGBoost + Gradient Boosting)
- Outputs a **risk classification** in seconds:
  - 🟢 No Flu Detected
  - 🟡 Possible Flu
  - 🔴 Possible Bird Flu
- Shows **confidence percentages** for each class
- Helps doctors and healthcare workers make faster, better-informed triage decisions

The system has **two versions**:
- `birdflu_prototype/` — Clinical prototype (includes exposure history: bird contact, travel, PCR, chest X-ray)
- `birdflu_webapp/` — Web application version (blood test + symptoms, for use at any clinic)

---

## ⚙️ How It Works

```
Patient Data Input
       │
       ▼
  Feature Engineering
  (symptoms + blood labs + exposure)
       │
       ▼
  Ensemble ML Model
  ┌────────────────────────────────────┐
  │  RandomForest  +  XGBoost  +  GBM │
  │      (Voting Classifier - soft)    │
  └────────────────────────────────────┘
       │
       ▼
  Risk Classification
  0 = No Flu  |  1 = Possible Flu  |  2 = Possible Bird Flu
       │
       ▼
  Streamlit Web UI
  (Risk label + confidence % + clinical advice)
```

### Key Clinical Signals the Model Uses

| Signal | Why It Matters |
|--------|---------------|
| Low WBC (< 5000) | Leukopenia — hallmark of H5N1 infection |
| Low Platelets (< 150k) | Thrombocytopenia — common in bird flu |
| High CRP (> 15) | Severe systemic inflammation |
| Elevated ALT/AST | Liver damage — bird flu attacks the liver |
| Fever + Cough + Dyspnea | Classic triad of H5N1 respiratory symptoms |
| Bird/Poultry Contact | Strongest exposure risk factor |
| PCR Positive | Confirms viral presence |

---

## 📁 Project Structure

```
AvianAlert/
│
├── birdflu_prototype/              # Core ML prototype
│   ├── data/
│   │   └── synthetic_data.csv      # 10,000 synthetic patient records
│   ├── src/
│   │   ├── data_generator.py       # Generates realistic synthetic data
│   │   ├── preprocess.py           # Feature engineering & train/test split
│   │   ├── train.py                # Trains ensemble model (RF+XGBoost+GBM)
│   │   └── predict.py              # Prediction function (returns probability)
│   ├── models/
│   │   └── rf_model.pkl            # Trained ensemble model (~12.7 MB)
│   ├── app/
│   │   └── streamlit_app.py        # Prototype Streamlit UI
│   └── requirements.txt
│
├── birdflu_webapp/                 # Production-ready web app
│   ├── src/
│   │   ├── train_and_save_model.py # Trains 3-class webapp model
│   │   ├── predict.py              # Prediction + probability functions
│   │   └── model.pkl               # Trained webapp model (~22.9 MB)
│   └── streamlit_app.py            # Main web app UI
│
├── full_ml_test_report.py          # 64-test professional QA suite
├── test_edge_cases.py              # Edge case & risk detection tests
├── run_tests.py                    # Quick end-to-end smoke test
├── README.md                       # This file
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.13.5 | Core language |
| Web UI | Streamlit | 1.48.0 | Interactive web interface |
| ML Models | scikit-learn | 1.7.1 | RandomForest, VotingClassifier |
| ML Models | XGBoost | 3.0.3 | Gradient boosted trees |
| Data | pandas | 2.3.1 | Data manipulation |
| Data | NumPy | 2.2.6 | Numerical operations |
| Model Storage | joblib / pickle | built-in | Saving/loading trained models |
| Testing | Python unittest + custom | — | 64-test QA suite |

### Why These Technologies?

- **XGBoost 3.0** — State-of-the-art gradient boosting. Better than plain Random Forest on tabular clinical data.
- **VotingClassifier (Ensemble)** — Combines 3 models. Reduces variance, avoids single-model blind spots.
- **Streamlit** — Fastest way to build a clean medical web UI in pure Python. No frontend code needed.
- **scikit-learn 1.7** — Latest stable release (2025/2026), best class imbalance handling via `class_weight='balanced'`.

---

## 🤖 ML Model Details

### Prototype Model (`birdflu_prototype`)
- **Type:** VotingClassifier (soft voting)
- **Estimators:** RandomForest (300 trees) + XGBoost (400 trees) + GradientBoosting (300 trees)
- **Task:** Binary classification — Infected (1) vs Healthy (0)
- **Features (24):** age, travel, contact_birds, poultry_visit, undercooked, contact_person, fever, cough, sore_throat, muscle_ache, headache, malaise, runny_nose, diarrhea, dyspnea, wbc, rt_pcr, chest_xray, gender (one-hot), occupation (one-hot)
- **Class imbalance:** Handled via `class_weight='balanced'` + `scale_pos_weight` in XGBoost
- **Training data:** 8,000 rows | Test data: 2,000 rows

### Webapp Model (`birdflu_webapp`)
- **Type:** VotingClassifier (soft voting)
- **Task:** 3-class classification — No Flu (0) / Possible Flu (1) / Possible Bird Flu (2)
- **Features (11):** WBC_count, RBC_count, Hemoglobin, Platelet_count, CRP, ALT, AST, Urine_pH, Fever, Cough, Sore_throat
- **Training data:** 10,000 rows with clinically-scored labels

### Label Generation Logic (Webapp)
Labels are NOT random. They are assigned using a clinical scoring system:

```
score += Fever × 2.0
score += Cough × 1.5
score += Sore_throat × 1.0
score += (WBC < 5000) × 3.0        # leukopenia
score += (WBC < 4000) × 1.5        # severe leukopenia
score += (CRP > 15) × 2.5          # high inflammation
score += (CRP > 25) × 1.5          # very high CRP
score += (ALT > 60) × 2.0          # liver stress
score += (AST > 55) × 2.0          # liver stress
score += (Platelets < 150k) × 2.0  # thrombocytopenia
score += combined lab pattern × 3.0 # bonus for combined danger

score >= 3.0  → Possible Flu (class 1)
score >= 7.0  → Possible Bird Flu (class 2)
```

---

## 📊 Accuracy & Test Results

### Final Model Performance (on 2,000-row holdout test set — never seen during training)

| Metric | Score | What It Means |
|--------|-------|---------------|
| **ROC-AUC** | **0.9995** | Near-perfect ability to separate sick from healthy |
| **Accuracy** | **98.95%** | 1,989 correct out of 2,000 patients |
| **Precision** | **97.93%** | Only 9 false alarms out of 2,000 |
| **Recall** | **97.25%** | Only 12 sick patients missed out of 437 |
| **F1 Score** | **97.59%** | Best balance of precision and recall |
| **False Negative Rate** | **2.75%** | 97.25% of sick patients correctly caught |
| **False Positive Rate** | **0.58%** | Very low unnecessary alarms |

### Confusion Matrix

```
                 Predicted
                 Healthy   Infected
Actual Healthy    1554        9      ← Only 9 false alarms
       Infected     12      425      ← Only 12 missed cases
```

### Improvement Over Original (2024 Baseline)

| | 2024 Baseline | 2026 AvianAlert | Improvement |
|--|---|---|---|
| Training data | 2,000 rows | **10,000 rows** | +5× more data |
| Model | Single RandomForest | **RF + XGBoost + GBM Ensemble** | 3-model ensemble |
| ROC-AUC | 0.9722 | **0.9995** | **+2.73%** |
| Accuracy | 90.00% | **98.95%** | **+8.95%** |
| Bug (random labels) | ❌ Present | ✅ Fixed | Critical fix |

### QA Test Suite Results

**64 tests across 12 categories — 64/64 PASSED**

| Category | Tests | Status |
|----------|-------|--------|
| T1 — Model File & Load | 6/6 | ✅ |
| T2 — Input Validation & Types | 6/6 | ✅ |
| T3 — Prediction Correctness | 7/7 | ✅ |
| T4 — Real Accuracy Metrics | 8/8 | ✅ |
| T5 — Class Imbalance Handling | 4/4 | ✅ |
| T6 — Boundary & Extreme Values | 6/6 | ✅ |
| T7 — Consistency & Determinism | 4/4 | ✅ |
| T8 — Performance & Speed | 4/4 | ✅ |
| T9 — Data Pipeline | 7/7 | ✅ |
| T10 — Regression vs Baseline | 4/4 | ✅ |
| T11 — Stress Tests | 3/3 | ✅ |
| T12 — Clinical Logic Validation | 5/5 | ✅ |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/avianalert.git
cd avianalert
```

### Step 2 — Create virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r birdflu_prototype/requirements.txt
```

**requirements.txt contents:**
```
pandas
numpy
scikit-learn
xgboost
streamlit
joblib
```

### Step 4 — Generate training data (10,000 rows)

```bash
python birdflu_prototype/src/data_generator.py
```

### Step 5 — Train the models

```bash
# Train prototype model
cd birdflu_prototype
python src/train.py
cd ..

# Train webapp model
python birdflu_webapp/src/train_and_save_model.py
```

> **Note:** Model `.pkl` files are included in the repo so you can skip steps 4–5 and run directly.

---

## ▶️ Running the App

### Webapp (Recommended)

```bash
cd birdflu_webapp
streamlit run streamlit_app.py
```

Then open your browser at: **http://localhost:8501**

### Prototype App

```bash
cd birdflu_prototype
streamlit run app/streamlit_app.py
```

---

## 🧪 Running Tests

### Quick smoke test (end-to-end)

```bash
python run_tests.py
```

### Edge case & risk detection tests (13 tests)

```bash
python test_edge_cases.py
```

### Full professional QA suite (64 tests)

```bash
python full_ml_test_report.py
```

Expected output:
```
TOTAL: 64/64 PASSED   0 FAILED   0 ERRORS
ALL TESTS PASSED - Model is production-quality!
```

---

## 🖥️ Screenshots

### Web App — Input Form
The user fills in blood test results and symptoms:

- **Blood values:** WBC count, RBC, Hemoglobin, Platelet count, CRP, ALT, AST, Urine pH
- **Symptoms:** Fever, Cough, Sore Throat

### Web App — Results
After clicking **Predict**, the app shows:
- Risk classification (No Flu / Possible Flu / Possible Bird Flu)
- Confidence breakdown (% for each class)
- Progress bar showing model confidence
- Clinical advice (seek testing / monitor / consult doctor)

---

## ⚠️ Known Limitations

1. **Synthetic data only** — The model is trained on computer-generated data, not real patient records. It must be retrained on real, de-identified clinical data before any medical use.

2. **Webapp doesn't ask about exposure** — The webapp only uses blood test values and symptoms. It cannot factor in bird contact, travel history, or PCR results. The prototype version includes these.

3. **Silent bird flu edge case** — Patients with critically abnormal labs (low WBC + low platelets + high CRP) but no symptoms are classified as "Possible Flu" rather than "Possible Bird Flu" by the webapp, because without exposure data the model can't fully distinguish it. The prototype handles this correctly.

4. **Not a diagnostic tool** — This is a screening aid only. A positive prediction must be followed up with clinical testing (RT-PCR).

5. **3-class labels are synthetic** — The No Flu / Flu / Bird Flu labels are generated by a clinical scoring heuristic, not real diagnosed cases.

---

## 🔮 Future Improvements

- [ ] Add bird/poultry exposure fields to the webapp (closes the silent bird flu gap)
- [ ] Train on real de-identified H5N1 patient datasets (WHO, CDC open data)
- [ ] Add SHAP explainability — show which features drove the prediction
- [ ] Add patient history tracking (multiple visits over time)
- [ ] REST API endpoint (FastAPI) for integration with hospital systems
- [ ] Mobile-friendly UI for field use by healthcare workers
- [ ] Multi-language support (Hindi, Tamil, etc.) for India rollout
- [ ] Deploy to cloud (AWS/GCP) with authentication

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

Built as a prototype AI health screening system.  
For questions or contributions, open an issue or pull request.

---

> **Disclaimer:** This project is a prototype built with synthetic data for demonstration and research purposes only.  
> It is **NOT** a certified medical device and must **NOT** be used for actual clinical diagnosis or treatment decisions.  
> Always consult a qualified healthcare professional.
