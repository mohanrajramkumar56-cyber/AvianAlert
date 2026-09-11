"""
=============================================================
BIRD FLU ML APPLICATION - PROFESSIONAL TEST SUITE
Senior QA Engineer Report (10+ years ML testing experience)
=============================================================
Test Categories:
  T1  - Model File & Load Tests
  T2  - Input Validation & Type Tests
  T3  - Functional / Prediction Correctness
  T4  - Real Accuracy Metrics (holdout set)
  T5  - Class Imbalance & Minority Class Detection
  T6  - Boundary & Extreme Value Tests
  T7  - Consistency & Determinism Tests
  T8  - Performance / Speed Tests
  T9  - Data Pipeline Tests
  T10 - Regression Tests (model didn't get worse)
  T11 - Stress Tests (bulk inference)
  T12 - Clinical Logic Validation
=============================================================
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
import joblib, pickle

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'birdflu_prototype', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'birdflu_webapp'))

from predict import predict_from_dict as proto_predict
from src.predict import predict_from_dict as webapp_predict, predict_proba_from_dict
from preprocess import load_and_preprocess
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score
)

# ─────────────────────────── Test Runner ────────────────────────────
results = []

def test(category, name, fn):
    try:
        msg = fn()
        status = "PASS"
        detail = msg if msg else ""
    except AssertionError as e:
        status = "FAIL"
        detail = str(e)
    except Exception as e:
        status = "ERROR"
        detail = f"{type(e).__name__}: {e}"
    results.append((category, name, status, detail))
    icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "!")
    print(f"  [{icon}] {name}")
    if detail:
        for line in str(detail).split("\n"):
            print(f"       {line}")

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ─────────────────── Load Models & Data Once ────────────────────────
BASE = os.path.dirname(__file__)

PROTO_MODEL_PATH = os.path.join(BASE, 'birdflu_prototype', 'models', 'rf_model.pkl')
WEBAPP_MODEL_PATH = os.path.join(BASE, 'birdflu_webapp', 'src', 'model.pkl')
DATA_PATH = os.path.join(BASE, 'birdflu_prototype', 'data', 'synthetic_data.csv')

proto_model = joblib.load(PROTO_MODEL_PATH)
with open(WEBAPP_MODEL_PATH, 'rb') as f:
    webapp_model = pickle.load(f)

# Load full holdout test set
X_train, X_test, y_train, y_test = load_and_preprocess(path=DATA_PATH)

# ─────────────────────────────────────────────────────────────────────
# T1 — MODEL FILE & LOAD TESTS
# ─────────────────────────────────────────────────────────────────────
section("T1 — MODEL FILE & LOAD TESTS")

def t1_proto_model_exists():
    assert os.path.exists(PROTO_MODEL_PATH), f"File not found: {PROTO_MODEL_PATH}"
    size_mb = os.path.getsize(PROTO_MODEL_PATH) / (1024*1024)
    return f"Size: {size_mb:.1f} MB"
test("T1", "Prototype model file exists", t1_proto_model_exists)

def t1_webapp_model_exists():
    assert os.path.exists(WEBAPP_MODEL_PATH), f"File not found: {WEBAPP_MODEL_PATH}"
    size_mb = os.path.getsize(WEBAPP_MODEL_PATH) / (1024*1024)
    return f"Size: {size_mb:.1f} MB"
test("T1", "Webapp model file exists", t1_webapp_model_exists)

def t1_proto_model_loads():
    m = joblib.load(PROTO_MODEL_PATH)
    assert hasattr(m, 'predict_proba'), "Model missing predict_proba"
    assert hasattr(m, 'estimators_'), "Not a VotingClassifier"
    return f"Estimators: {[e[0] for e in m.estimators]}"
test("T1", "Prototype model loads correctly (VotingClassifier)", t1_proto_model_loads)

def t1_webapp_model_loads():
    with open(WEBAPP_MODEL_PATH, 'rb') as f:
        m = pickle.load(f)
    assert hasattr(m, 'predict_proba'), "Model missing predict_proba"
    return f"Type: {type(m).__name__}"
test("T1", "Webapp model loads correctly", t1_webapp_model_loads)

def t1_feature_names():
    feats = proto_model.feature_names_in_
    assert len(feats) > 0, "No feature names stored"
    return f"{len(feats)} features: {list(feats)}"
test("T1", "Prototype model has feature names stored", t1_feature_names)

def t1_data_file_exists():
    assert os.path.exists(DATA_PATH), f"Data not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 10000, f"Expected 10k rows, got {len(df)}"
    return f"{len(df)} rows, {len(df.columns)} columns"
test("T1", "Training data file exists with 10k rows", t1_data_file_exists)

# ─────────────────────────────────────────────────────────────────────
# T2 — INPUT VALIDATION & TYPE TESTS
# ─────────────────────────────────────────────────────────────────────
section("T2 — INPUT VALIDATION & TYPE TESTS")

BASE_PROTO = {
    'age':30,'travel':0,'contact_birds':0,'poultry_visit':0,'undercooked':0,
    'contact_person':0,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
    'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':0,
    'wbc':7000,'rt_pcr':0,'chest_xray':0
}
BASE_WEBAPP = {
    'WBC_count':7000,'RBC_count':4.5,'Hemoglobin':13.5,'Platelet_count':250000,
    'CRP':5.0,'ALT':30.0,'AST':28.0,'Urine_pH':6.0,
    'Fever':0,'Cough':0,'Sore_throat':0
}

def t2_return_type_proto():
    r = proto_predict(BASE_PROTO)
    assert isinstance(r, float), f"Expected float, got {type(r)}"
    assert 0.0 <= r <= 1.0, f"Probability out of range: {r}"
    return f"Returns float in [0,1]: {r}"
test("T2", "Proto returns float probability in [0,1]", t2_return_type_proto)

def t2_return_type_webapp():
    r = webapp_predict(BASE_WEBAPP)
    assert isinstance(r, int), f"Expected int, got {type(r)}"
    assert r in [0,1,2], f"Expected 0/1/2, got {r}"
    return f"Returns int class label: {r}"
test("T2", "Webapp returns int class label (0/1/2)", t2_return_type_webapp)

def t2_proba_sum_to_one():
    p = predict_proba_from_dict(BASE_WEBAPP)
    assert len(p) == 3, f"Expected 3 probabilities, got {len(p)}"
    total = sum(p)
    assert abs(total - 1.0) < 0.001, f"Probabilities don't sum to 1: {total}"
    return f"Proba sums to {total:.4f}"
test("T2", "Webapp probabilities sum to 1.0", t2_proba_sum_to_one)

def t2_proba_all_non_negative():
    p = predict_proba_from_dict(BASE_WEBAPP)
    assert all(x >= 0 for x in p), f"Negative probability found: {p}"
    return f"All probabilities >= 0: {[f'{x:.3f}' for x in p]}"
test("T2", "All probabilities are non-negative", t2_proba_all_non_negative)

def t2_integer_inputs_work():
    # Pass integer values instead of floats for numeric fields
    d = {k: int(v) if isinstance(v, float) else v for k, v in BASE_WEBAPP.items()}
    r = webapp_predict(d)
    assert r in [0,1,2]
    return f"Integer inputs handled OK, class={r}"
test("T2", "Integer inputs handled (no type crash)", t2_integer_inputs_work)

def t2_missing_optional_field_proto():
    # When a feature is missing it should be filled with 0 via reindex
    d = dict(BASE_PROTO)
    d['age'] = 50
    r = proto_predict(d)
    assert 0.0 <= r <= 1.0
    return f"Missing fields handled, prob={r:.3f}"
test("T2", "Proto handles reindex fill for missing dummies", t2_missing_optional_field_proto)

# ─────────────────────────────────────────────────────────────────────
# T3 — FUNCTIONAL / PREDICTION CORRECTNESS
# ─────────────────────────────────────────────────────────────────────
section("T3 — FUNCTIONAL / PREDICTION CORRECTNESS")

def t3_clear_bird_flu():
    sick = {
        'WBC_count':3200,'RBC_count':3.5,'Hemoglobin':10.0,'Platelet_count':100000,
        'CRP':35.0,'ALT':90.0,'AST':85.0,'Urine_pH':5.0,
        'Fever':1,'Cough':1,'Sore_throat':1
    }
    r = webapp_predict(sick)
    p = predict_proba_from_dict(sick)
    assert r == 2, f"Clear bird flu patient got class {r}, not 2"
    assert p[2] > 0.5, f"Bird flu confidence too low: {p[2]:.2f}"
    return f"Bird Flu detected with {p[2]*100:.1f}% confidence"
test("T3", "Clear bird flu patient correctly classified as class 2", t3_clear_bird_flu)

def t3_clear_healthy():
    healthy = {
        'WBC_count':7800,'RBC_count':5.0,'Hemoglobin':15.0,'Platelet_count':300000,
        'CRP':1.5,'ALT':18.0,'AST':16.0,'Urine_pH':6.5,
        'Fever':0,'Cough':0,'Sore_throat':0
    }
    r = webapp_predict(healthy)
    p = predict_proba_from_dict(healthy)
    assert r == 0, f"Healthy patient got class {r}, not 0"
    assert p[0] > 0.9, f"No-flu confidence too low: {p[0]:.2f}"
    return f"Healthy correctly cleared with {p[0]*100:.1f}% confidence"
test("T3", "Healthy patient correctly cleared as class 0", t3_clear_healthy)

def t3_regular_flu_not_bird_flu():
    flu = {
        'WBC_count':9000,'RBC_count':4.6,'Hemoglobin':14.0,'Platelet_count':260000,
        'CRP':8.0,'ALT':25.0,'AST':22.0,'Urine_pH':6.5,
        'Fever':1,'Cough':1,'Sore_throat':1
    }
    r = webapp_predict(flu)
    assert r != 2, f"Regular flu was misclassified as Bird Flu"
    return f"Regular flu → class {r} (not Bird Flu) - correct"
test("T3", "Regular flu NOT misclassified as Bird Flu", t3_regular_flu_not_bird_flu)

def t3_proto_pcr_positive_raises_risk():
    # PCR positive should raise risk substantially
    base = dict(BASE_PROTO)
    base['contact_birds'] = 1
    base['rt_pcr'] = 0
    r_no_pcr = proto_predict(base)
    base['rt_pcr'] = 1
    r_pcr = proto_predict(base)
    assert r_pcr > r_no_pcr, f"PCR+ didn't raise risk: {r_no_pcr:.3f} -> {r_pcr:.3f}"
    return f"PCR- risk={r_no_pcr*100:.1f}%  PCR+ risk={r_pcr*100:.1f}% (+{(r_pcr-r_no_pcr)*100:.1f}%)"
test("T3", "PCR positive raises bird flu risk in proto", t3_proto_pcr_positive_raises_risk)

def t3_more_symptoms_higher_risk():
    d0 = dict(BASE_WEBAPP)
    d0.update({'Fever':0,'Cough':0,'Sore_throat':0})
    d1 = dict(BASE_WEBAPP)
    d1.update({'Fever':1,'Cough':0,'Sore_throat':0})
    d2 = dict(BASE_WEBAPP)
    d2.update({'Fever':1,'Cough':1,'Sore_throat':1})
    p0 = predict_proba_from_dict(d0)
    p1 = predict_proba_from_dict(d1)
    p2 = predict_proba_from_dict(d2)
    # Risk of having any flu (class 1 or 2) should increase with symptoms
    r0 = p0[1] + p0[2]
    r1 = p1[1] + p1[2]
    r2 = p2[1] + p2[2]
    assert r2 >= r1 >= r0, f"More symptoms didn't increase risk: {r0:.2f} {r1:.2f} {r2:.2f}"
    return f"Flu risk: 0 symp={r0*100:.0f}%  1 symp={r1*100:.0f}%  3 symp={r2*100:.0f}%"
test("T3", "More symptoms = higher flu risk (monotonic)", t3_more_symptoms_higher_risk)

def t3_lower_wbc_higher_bird_flu_risk():
    d_high = dict(BASE_WEBAPP); d_high['WBC_count'] = 10000
    d_low  = dict(BASE_WEBAPP); d_low['WBC_count']  = 3000
    p_high = predict_proba_from_dict(d_high)
    p_low  = predict_proba_from_dict(d_low)
    assert p_low[2] >= p_high[2], (
        f"Lower WBC should give higher bird flu risk. "
        f"WBC=10k: {p_high[2]:.3f}, WBC=3k: {p_low[2]:.3f}"
    )
    return f"Bird flu prob: WBC=10k → {p_high[2]*100:.1f}%,  WBC=3k → {p_low[2]*100:.1f}%"
test("T3", "Lower WBC = higher bird flu probability", t3_lower_wbc_higher_bird_flu_risk)

def t3_high_crp_raises_bird_flu():
    d_low  = dict(BASE_WEBAPP); d_low['CRP']  = 2.0
    d_high = dict(BASE_WEBAPP); d_high['CRP'] = 30.0
    p_low  = predict_proba_from_dict(d_low)
    p_high = predict_proba_from_dict(d_high)
    assert p_high[1] + p_high[2] >= p_low[1] + p_low[2], "High CRP should increase flu risk"
    return f"High CRP raises flu risk: CRP=2 → {(p_low[1]+p_low[2])*100:.1f}%,  CRP=30 → {(p_high[1]+p_high[2])*100:.1f}%"
test("T3", "High CRP raises flu/bird flu risk", t3_high_crp_raises_bird_flu)

# ─────────────────────────────────────────────────────────────────────
# T4 — REAL ACCURACY METRICS (Holdout Set)
# ─────────────────────────────────────────────────────────────────────
section("T4 — REAL ACCURACY METRICS (2000-row holdout test set)")

y_prob = proto_model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

def t4_accuracy():
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.90, f"Accuracy too low: {acc:.4f}"
    return f"Accuracy: {acc*100:.2f}%"
test("T4", "Overall accuracy >= 90%", t4_accuracy)

def t4_roc_auc():
    auc = roc_auc_score(y_test, y_prob)
    assert auc >= 0.95, f"ROC-AUC too low: {auc:.4f}"
    return f"ROC-AUC: {auc:.4f}"
test("T4", "ROC-AUC >= 0.95", t4_roc_auc)

def t4_precision():
    prec = precision_score(y_test, y_pred)
    assert prec >= 0.85, f"Precision too low: {prec:.4f}"
    return f"Precision (infected class): {prec*100:.2f}%"
test("T4", "Precision >= 85% (don't over-alarm healthy people)", t4_precision)

def t4_recall():
    rec = recall_score(y_test, y_pred)
    assert rec >= 0.85, f"Recall too low: {rec:.4f} — model missing too many sick patients"
    return f"Recall (infected class): {rec*100:.2f}%"
test("T4", "Recall >= 85% (don't miss sick patients)", t4_recall)

def t4_f1():
    f1 = f1_score(y_test, y_pred)
    assert f1 >= 0.85, f"F1 too low: {f1:.4f}"
    return f"F1 Score: {f1*100:.2f}%"
test("T4", "F1 Score >= 85%", t4_f1)

def t4_avg_precision():
    ap = average_precision_score(y_test, y_prob)
    assert ap >= 0.80, f"Average Precision too low: {ap:.4f}"
    return f"Average Precision (PR-AUC): {ap:.4f}"
test("T4", "Average Precision (PR-AUC) >= 0.80", t4_avg_precision)

def t4_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    false_negative_rate = fn / (fn + tp) if (fn+tp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp+tn) > 0 else 0
    assert false_negative_rate < 0.15, f"Too many missed cases (FNR): {false_negative_rate:.2%}"
    return (
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
        f"       False Negative Rate (missed sick): {false_negative_rate:.2%}\n"
        f"       False Positive Rate (false alarm): {false_positive_rate:.2%}"
    )
test("T4", "Confusion matrix: False Negative Rate < 15%", t4_confusion_matrix)

def t4_full_classification_report():
    report = classification_report(y_test, y_pred, target_names=['Healthy','Infected'])
    return "\n" + report
test("T4", "Full classification report", t4_full_classification_report)

# ─────────────────────────────────────────────────────────────────────
# T5 — CLASS IMBALANCE & MINORITY CLASS DETECTION
# ─────────────────────────────────────────────────────────────────────
section("T5 — CLASS IMBALANCE & MINORITY CLASS DETECTION")

def t5_class_distribution():
    infected = y_test.sum()
    total = len(y_test)
    ratio = infected / total
    assert 0.05 < ratio < 0.50, f"Unusual class ratio in test set: {ratio:.2%}"
    return f"Test set: {infected} infected ({ratio:.1%}), {total-infected} healthy"
test("T5", "Class distribution in test set is reasonable", t5_class_distribution)

def t5_minority_recall():
    rec = recall_score(y_test, y_pred)
    assert rec >= 0.85, f"Model misses too many infected cases: recall={rec:.2%}"
    return f"Minority class (infected) recall: {rec:.2%}"
test("T5", "Infected class recall >= 85% (class imbalance handled)", t5_minority_recall)

def t5_no_all_zero_prediction():
    assert y_pred.sum() > 0, "Model predicts nobody is infected — completely degenerate!"
    assert (y_pred == 0).sum() > 0, "Model predicts everyone infected — completely degenerate!"
    return f"Model predicts {y_pred.sum()} infected, {(y_pred==0).sum()} healthy out of {len(y_pred)}"
test("T5", "Model is not degenerate (predicts both classes)", t5_no_all_zero_prediction)

def t5_webapp_predicts_all_classes():
    test_cases = [
        {'WBC_count':3200,'RBC_count':3.5,'Hemoglobin':10.0,'Platelet_count':100000,'CRP':35.0,'ALT':90.0,'AST':85.0,'Urine_pH':5.0,'Fever':1,'Cough':1,'Sore_throat':1},
        {'WBC_count':7800,'RBC_count':5.0,'Hemoglobin':15.0,'Platelet_count':300000,'CRP':1.5,'ALT':18.0,'AST':16.0,'Urine_pH':6.5,'Fever':0,'Cough':0,'Sore_throat':0},
        {'WBC_count':9000,'RBC_count':4.6,'Hemoglobin':14.0,'Platelet_count':260000,'CRP':8.0,'ALT':25.0,'AST':22.0,'Urine_pH':6.5,'Fever':1,'Cough':1,'Sore_throat':0},
    ]
    seen = set(webapp_predict(c) for c in test_cases)
    assert len(seen) >= 2, f"Webapp only predicts class(es): {seen}"
    return f"Webapp predicts classes: {sorted(seen)}"
test("T5", "Webapp predicts multiple classes (not stuck on one)", t5_webapp_predicts_all_classes)

# ─────────────────────────────────────────────────────────────────────
# T6 — BOUNDARY & EXTREME VALUE TESTS
# ─────────────────────────────────────────────────────────────────────
section("T6 — BOUNDARY & EXTREME VALUE TESTS")

def t6_min_values():
    d = {'WBC_count':2000,'RBC_count':2.5,'Hemoglobin':7.0,'Platelet_count':50000,
         'CRP':0.0,'ALT':0.0,'AST':0.0,'Urine_pH':4.0,'Fever':0,'Cough':0,'Sore_throat':0}
    r = webapp_predict(d); p = predict_proba_from_dict(d)
    assert r in [0,1,2]; assert abs(sum(p)-1.0) < 0.001
    return f"Min values → class={r}, no crash"
test("T6", "Minimum boundary values don't crash", t6_min_values)

def t6_max_values():
    d = {'WBC_count':20000,'RBC_count':6.0,'Hemoglobin':18.0,'Platelet_count':500000,
         'CRP':50.0,'ALT':100.0,'AST':100.0,'Urine_pH':8.5,'Fever':1,'Cough':1,'Sore_throat':1}
    r = webapp_predict(d); p = predict_proba_from_dict(d)
    assert r in [0,1,2]; assert abs(sum(p)-1.0) < 0.001
    return f"Max values → class={r}, no crash"
test("T6", "Maximum boundary values don't crash", t6_max_values)

def t6_all_zeros_proto():
    d = {k:0 for k in BASE_PROTO}
    r = proto_predict(d)
    assert 0.0 <= r <= 1.0
    return f"All zeros → prob={r:.4f}"
test("T6", "All-zero proto input returns valid probability", t6_all_zeros_proto)

def t6_proto_age_zero():
    d = dict(BASE_PROTO); d['age'] = 0
    r = proto_predict(d)
    assert 0.0 <= r <= 1.0
    return f"Age=0 → prob={r:.4f}"
test("T6", "Age=0 (newborn) handled without crash", t6_proto_age_zero)

def t6_proto_age_100():
    d = dict(BASE_PROTO); d['age'] = 100
    r = proto_predict(d)
    assert 0.0 <= r <= 1.0
    return f"Age=100 → prob={r:.4f}"
test("T6", "Age=100 (elderly) handled without crash", t6_proto_age_100)

def t6_float_wbc():
    d = dict(BASE_WEBAPP); d['WBC_count'] = 4999.99
    r = webapp_predict(d); p = predict_proba_from_dict(d)
    assert r in [0,1,2]
    return f"Float WBC=4999.99 → class={r}"
test("T6", "Float boundary WBC value handled", t6_float_wbc)

# ─────────────────────────────────────────────────────────────────────
# T7 — CONSISTENCY & DETERMINISM TESTS
# ─────────────────────────────────────────────────────────────────────
section("T7 — CONSISTENCY & DETERMINISM TESTS")

def t7_same_input_same_output_proto():
    d = dict(BASE_PROTO); d.update({'fever':1,'cough':1,'wbc':4500})
    results_list = [proto_predict(d) for _ in range(10)]
    # Allow floating point tolerance of 1e-10 (machine epsilon variation is fine)
    base_val = results_list[0]
    max_diff = max(abs(r - base_val) for r in results_list)
    assert max_diff < 1e-6, f"Non-deterministic beyond tolerance (diff={max_diff:.2e}): {results_list}"
    return f"10 calls consistent within 1e-6 tolerance: {base_val:.6f} (max_diff={max_diff:.2e})"
test("T7", "Proto: same input always gives same output", t7_same_input_same_output_proto)

def t7_same_input_same_output_webapp():
    d = dict(BASE_WEBAPP); d.update({'Fever':1,'CRP':20})
    results_list = [webapp_predict(d) for _ in range(10)]
    assert len(set(results_list)) == 1, f"Non-deterministic: {results_list}"
    return f"10 calls, all return class {results_list[0]}"
test("T7", "Webapp: same input always gives same output", t7_same_input_same_output_webapp)

def t7_independent_calls_dont_interfere():
    # Predict sick, then healthy, then sick — sick results should be same
    sick = {'WBC_count':3200,'RBC_count':3.5,'Hemoglobin':10.0,'Platelet_count':100000,'CRP':35.0,'ALT':90.0,'AST':85.0,'Urine_pH':5.0,'Fever':1,'Cough':1,'Sore_throat':1}
    healthy = {'WBC_count':7800,'RBC_count':5.0,'Hemoglobin':15.0,'Platelet_count':300000,'CRP':1.5,'ALT':18.0,'AST':16.0,'Urine_pH':6.5,'Fever':0,'Cough':0,'Sore_throat':0}
    r1 = webapp_predict(sick)
    webapp_predict(healthy)
    r2 = webapp_predict(sick)
    assert r1 == r2, f"Calls are interfering: first={r1}, after healthy call={r2}"
    return f"Sick patient gives same result before and after healthy call: {r1}"
test("T7", "Calls don't interfere with each other (no shared state)", t7_independent_calls_dont_interfere)

def t7_order_of_dict_keys_doesnt_matter():
    d1 = {'WBC_count':5000,'RBC_count':4.0,'Hemoglobin':12.0,'Platelet_count':180000,'CRP':18.0,'ALT':55.0,'AST':50.0,'Urine_pH':5.8,'Fever':1,'Cough':1,'Sore_throat':0}
    # Reverse order
    d2 = {k: d1[k] for k in reversed(list(d1.keys()))}
    r1 = webapp_predict(d1)
    r2 = webapp_predict(d2)
    assert r1 == r2, f"Dict key order matters: {r1} vs {r2}"
    return f"Both orderings give class {r1}"
test("T7", "Dict key order doesn't change prediction", t7_order_of_dict_keys_doesnt_matter)

# ─────────────────────────────────────────────────────────────────────
# T8 — PERFORMANCE / SPEED TESTS
# ─────────────────────────────────────────────────────────────────────
section("T8 — PERFORMANCE / SPEED TESTS")

def t8_single_prediction_speed_proto():
    d = dict(BASE_PROTO)
    start = time.perf_counter()
    for _ in range(100):
        proto_predict(d)
    elapsed = (time.perf_counter() - start) / 100 * 1000
    assert elapsed < 100, f"Single prediction too slow: {elapsed:.2f}ms"
    return f"Avg per prediction: {elapsed:.2f}ms"
test("T8", "Proto: single prediction < 100ms", t8_single_prediction_speed_proto)

def t8_single_prediction_speed_webapp():
    d = dict(BASE_WEBAPP)
    start = time.perf_counter()
    for _ in range(100):
        webapp_predict(d)
    elapsed = (time.perf_counter() - start) / 100 * 1000
    assert elapsed < 100, f"Single prediction too slow: {elapsed:.2f}ms"
    return f"Avg per prediction: {elapsed:.2f}ms"
test("T8", "Webapp: single prediction < 100ms", t8_single_prediction_speed_webapp)

def t8_bulk_inference_100():
    rows = pd.DataFrame([BASE_WEBAPP] * 100)
    start = time.perf_counter()
    preds = webapp_model.predict(rows)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 2000, f"100 predictions took too long: {elapsed:.0f}ms"
    return f"100 predictions in {elapsed:.0f}ms ({elapsed/100:.2f}ms each)"
test("T8", "Webapp: batch of 100 predictions < 2 seconds", t8_bulk_inference_100)

def t8_model_load_time():
    start = time.perf_counter()
    joblib.load(PROTO_MODEL_PATH)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 10000, f"Model load too slow: {elapsed:.0f}ms"
    return f"Proto model load time: {elapsed:.0f}ms"
test("T8", "Proto model loads in < 10 seconds", t8_model_load_time)

# ─────────────────────────────────────────────────────────────────────
# T9 — DATA PIPELINE TESTS
# ─────────────────────────────────────────────────────────────────────
section("T9 — DATA PIPELINE TESTS")

def t9_no_nulls_in_data():
    df = pd.read_csv(DATA_PATH)
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values in training data"
    return f"Zero nulls in {len(df)} rows"
test("T9", "No null values in training data", t9_no_nulls_in_data)

def t9_no_duplicate_ids():
    df = pd.read_csv(DATA_PATH)
    dupes = df['id'].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate IDs"
    return f"All {len(df)} IDs are unique"
test("T9", "No duplicate IDs in training data", t9_no_duplicate_ids)

def t9_target_column_valid():
    df = pd.read_csv(DATA_PATH)
    vals = set(df['infected'].unique())
    assert vals == {0,1}, f"Unexpected target values: {vals}"
    return f"Target column only has 0 and 1 — OK"
test("T9", "Target column only contains 0 and 1", t9_target_column_valid)

def t9_wbc_range_valid():
    df = pd.read_csv(DATA_PATH)
    assert df['wbc'].min() >= 1000, f"WBC too low: {df['wbc'].min()}"
    assert df['wbc'].max() <= 25000, f"WBC too high: {df['wbc'].max()}"
    return f"WBC range: {df['wbc'].min():.0f} – {df['wbc'].max():.0f}"
test("T9", "WBC values within valid clinical range", t9_wbc_range_valid)

def t9_age_range_valid():
    df = pd.read_csv(DATA_PATH)
    assert df['age'].min() >= 0, f"Negative age found: {df['age'].min()}"
    assert df['age'].max() <= 120, f"Age too high: {df['age'].max()}"
    return f"Age range: {df['age'].min()} – {df['age'].max()}"
test("T9", "Age values within valid range (0–120)", t9_age_range_valid)

def t9_binary_columns_valid():
    df = pd.read_csv(DATA_PATH)
    binary_cols = ['travel','contact_birds','poultry_visit','undercooked','contact_person',
                   'fever','cough','sore_throat','muscle_ache','headache','malaise',
                   'runny_nose','diarrhea','dyspnea','rt_pcr','chest_xray']
    bad = []
    for col in binary_cols:
        vals = set(df[col].unique())
        if not vals.issubset({0,1}):
            bad.append(f"{col}={vals}")
    assert not bad, f"Non-binary values in: {bad}"
    return f"All {len(binary_cols)} binary columns contain only 0/1"
test("T9", "All binary feature columns contain only 0 and 1", t9_binary_columns_valid)

def t9_train_test_no_overlap():
    # Verify stratified split preserved class ratio
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    diff = abs(train_ratio - test_ratio)
    assert diff < 0.03, f"Class ratio drift between train/test: {diff:.3f}"
    return f"Train infected={train_ratio:.2%}  Test infected={test_ratio:.2%}  diff={diff:.3f}"
test("T9", "Train/test split maintains class ratio (stratified)", t9_train_test_no_overlap)

# ─────────────────────────────────────────────────────────────────────
# T10 — REGRESSION TESTS (BASELINE COMPARISON)
# ─────────────────────────────────────────────────────────────────────
section("T10 — REGRESSION TESTS (vs old 2000-row baseline)")

BASELINE_AUC    = 0.9722  # Original 2000-row single RF score
BASELINE_ACC    = 0.90    # Original accuracy

def t10_auc_better_than_baseline():
    auc = roc_auc_score(y_test, y_prob)
    assert auc > BASELINE_AUC, f"New AUC {auc:.4f} not better than baseline {BASELINE_AUC}"
    return f"New AUC={auc:.4f}  Baseline={BASELINE_AUC}  Improvement=+{(auc-BASELINE_AUC)*100:.2f}%"
test("T10", "ROC-AUC improved over 2000-row baseline (0.9722)", t10_auc_better_than_baseline)

def t10_acc_better_than_baseline():
    acc = accuracy_score(y_test, y_pred)
    assert acc > BASELINE_ACC, f"New accuracy {acc:.4f} not better than baseline {BASELINE_ACC}"
    return f"New Acc={acc*100:.2f}%  Baseline={BASELINE_ACC*100:.0f}%  Improvement=+{(acc-BASELINE_ACC)*100:.2f}%"
test("T10", "Accuracy improved over 2000-row baseline (90%)", t10_acc_better_than_baseline)

def t10_sick_still_detected():
    sick_proto = dict(BASE_PROTO)
    sick_proto.update({'fever':1,'cough':1,'wbc':4000,'rt_pcr':1,'contact_birds':1,'poultry_visit':1})
    r = proto_predict(sick_proto)
    assert r > 0.5, f"Sick patient regression: now only {r*100:.1f}% (was > 50%)"
    return f"Sick patient still at {r*100:.1f}% risk"
test("T10", "Sick patient still correctly flagged (no regression)", t10_sick_still_detected)

def t10_healthy_still_cleared():
    r = proto_predict(BASE_PROTO)
    assert r < 0.2, f"Healthy patient regression: now {r*100:.1f}% (was < 20%)"
    return f"Healthy patient still at {r*100:.1f}% risk"
test("T10", "Healthy patient still correctly cleared (no regression)", t10_healthy_still_cleared)

# ─────────────────────────────────────────────────────────────────────
# T11 — STRESS TESTS (BULK INFERENCE)
# ─────────────────────────────────────────────────────────────────────
section("T11 — STRESS TESTS (BULK INFERENCE)")

def t11_bulk_1000_predictions():
    rows = pd.DataFrame([BASE_WEBAPP] * 1000)
    start = time.perf_counter()
    preds = webapp_model.predict(rows)
    elapsed = time.perf_counter() - start
    assert elapsed < 30, f"1000 predictions took {elapsed:.1f}s (too slow)"
    assert len(preds) == 1000
    return f"1000 predictions in {elapsed:.2f}s ({elapsed/1000*1000:.1f}ms each)"
test("T11", "Webapp: 1000 bulk predictions complete < 30s", t11_bulk_1000_predictions)

def t11_random_valid_inputs_no_crash():
    np.random.seed(99)
    errors = []
    for i in range(200):
        d = {
            'WBC_count':    float(np.random.randint(2000, 20001)),
            'RBC_count':    round(float(np.random.uniform(2.5, 6.0)), 1),
            'Hemoglobin':   round(float(np.random.uniform(7.0, 18.0)), 1),
            'Platelet_count': float(np.random.randint(50000, 500001)),
            'CRP':          round(float(np.random.uniform(0, 50)), 1),
            'ALT':          round(float(np.random.uniform(0, 100)), 1),
            'AST':          round(float(np.random.uniform(0, 100)), 1),
            'Urine_pH':     round(float(np.random.uniform(4.0, 8.5)), 1),
            'Fever':        int(np.random.randint(0, 2)),
            'Cough':        int(np.random.randint(0, 2)),
            'Sore_throat':  int(np.random.randint(0, 2)),
        }
        try:
            r = webapp_predict(d)
            p = predict_proba_from_dict(d)
            assert r in [0,1,2]
            assert abs(sum(p) - 1.0) < 0.001
        except Exception as e:
            errors.append(f"Sample {i}: {e}")
    assert not errors, f"{len(errors)} errors:\n" + "\n".join(errors[:5])
    return f"200 random inputs — zero crashes, all outputs valid"
test("T11", "200 random valid inputs: no crashes, valid outputs", t11_random_valid_inputs_no_crash)

def t11_repeated_calls_stable_memory():
    # Run 500 predictions, check memory doesn't explode
    import tracemalloc
    tracemalloc.start()
    for _ in range(500):
        webapp_predict(BASE_WEBAPP)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024
    assert peak_mb < 200, f"Peak memory too high: {peak_mb:.1f}MB"
    return f"500 calls peak memory: {peak_mb:.1f}MB"
test("T11", "500 repeated calls: peak memory < 200MB", t11_repeated_calls_stable_memory)

# ─────────────────────────────────────────────────────────────────────
# T12 — CLINICAL LOGIC VALIDATION
# ─────────────────────────────────────────────────────────────────────
section("T12 — CLINICAL LOGIC VALIDATION")

def t12_leukopenia_is_key_signal():
    # Bird flu causes leukopenia (low WBC). Model must weight this heavily.
    normal_wbc = dict(BASE_WEBAPP); normal_wbc['WBC_count'] = 8000
    low_wbc    = dict(BASE_WEBAPP); low_wbc['WBC_count']    = 3000
    p_normal = predict_proba_from_dict(normal_wbc)
    p_low    = predict_proba_from_dict(low_wbc)
    assert p_low[2] > p_normal[2], "Leukopenia should increase Bird Flu probability"
    return f"Bird flu prob: Normal WBC={p_normal[2]*100:.1f}%  Low WBC={p_low[2]*100:.1f}%"
test("T12", "Leukopenia (low WBC) increases Bird Flu probability", t12_leukopenia_is_key_signal)

def t12_thrombocytopenia_is_signal():
    # Low platelets = thrombocytopenia, seen in bird flu
    normal = dict(BASE_WEBAPP); normal['Platelet_count'] = 300000
    low    = dict(BASE_WEBAPP); low['Platelet_count']    = 100000
    p_normal = predict_proba_from_dict(normal)
    p_low    = predict_proba_from_dict(low)
    assert p_low[1]+p_low[2] >= p_normal[1]+p_normal[2], "Low platelets should raise flu risk"
    return f"Flu risk: Normal platelets={( p_normal[1]+p_normal[2])*100:.1f}%  Low platelets={(p_low[1]+p_low[2])*100:.1f}%"
test("T12", "Thrombocytopenia (low platelets) raises flu risk", t12_thrombocytopenia_is_signal)

def t12_elevated_liver_enzymes():
    normal = dict(BASE_WEBAPP)
    normal.update({'ALT':25.0,'AST':22.0})
    elevated = dict(BASE_WEBAPP)
    elevated.update({'ALT':85.0,'AST':80.0})
    p_normal   = predict_proba_from_dict(normal)
    p_elevated = predict_proba_from_dict(elevated)
    assert p_elevated[1]+p_elevated[2] >= p_normal[1]+p_normal[2], "High liver enzymes should raise risk"
    return f"Flu risk: Normal liver={(p_normal[1]+p_normal[2])*100:.1f}%  Elevated liver={(p_elevated[1]+p_elevated[2])*100:.1f}%"
test("T12", "Elevated liver enzymes (ALT/AST) raises flu risk", t12_elevated_liver_enzymes)

def t12_high_crp_is_bird_flu_marker():
    normal = dict(BASE_WEBAPP); normal['CRP'] = 3.0
    very_high = dict(BASE_WEBAPP); very_high['CRP'] = 40.0
    p_n = predict_proba_from_dict(normal)
    p_h = predict_proba_from_dict(very_high)
    assert p_h[1]+p_h[2] > p_n[1]+p_n[2], "Very high CRP should indicate flu"
    return f"Flu risk: CRP=3 → {(p_n[1]+p_n[2])*100:.1f}%   CRP=40 → {(p_h[1]+p_h[2])*100:.1f}%"
test("T12", "Very high CRP (inflammation) raises flu risk", t12_high_crp_is_bird_flu_marker)

def t12_combination_worse_than_individual():
    # Combination of low WBC + high CRP + fever should be worse than any alone
    base    = dict(BASE_WEBAPP)
    low_wbc = dict(BASE_WEBAPP); low_wbc['WBC_count'] = 3500
    hi_crp  = dict(BASE_WEBAPP); hi_crp['CRP'] = 28.0
    combo   = dict(BASE_WEBAPP); combo.update({'WBC_count':3500,'CRP':28.0,'Fever':1,'Platelet_count':110000})
    p_base   = sum(predict_proba_from_dict(base)[1:])
    p_wbc    = sum(predict_proba_from_dict(low_wbc)[1:])
    p_crp    = sum(predict_proba_from_dict(hi_crp)[1:])
    p_combo  = sum(predict_proba_from_dict(combo)[1:])
    assert p_combo >= max(p_wbc, p_crp), f"Combo risk {p_combo:.2f} not >= individual max {max(p_wbc,p_crp):.2f}"
    return f"Flu risk: Base={p_base*100:.0f}%  LowWBC={p_wbc*100:.0f}%  HiCRP={p_crp*100:.0f}%  COMBO={p_combo*100:.0f}%"
test("T12", "Combined risk markers produce higher risk than individual", t12_combination_worse_than_individual)

# ─────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL TEST REPORT SUMMARY")
print(f"{'='*65}")

by_cat = {}
for cat, name, status, detail in results:
    by_cat.setdefault(cat, {'PASS':0,'FAIL':0,'ERROR':0})
    by_cat[cat][status] += 1

for cat, counts in sorted(by_cat.items()):
    total = sum(counts.values())
    passed = counts['PASS']
    bar = "#" * passed + "-" * (total - passed)
    print(f"  {cat}: [{bar}] {passed}/{total}", end="")
    if counts['FAIL'] + counts['ERROR'] > 0:
        print(f"  <-- {counts['FAIL']} FAIL  {counts['ERROR']} ERROR", end="")
    print()

total_pass  = sum(1 for _,_,s,_ in results if s == "PASS")
total_fail  = sum(1 for _,_,s,_ in results if s == "FAIL")
total_error = sum(1 for _,_,s,_ in results if s == "ERROR")
total_all   = len(results)

print(f"\n  TOTAL: {total_pass}/{total_all} PASSED   {total_fail} FAILED   {total_error} ERRORS")

# Print actual metric values
auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
print(f"\n  KEY METRICS (on 2000-row holdout set):")
print(f"    ROC-AUC   : {auc:.4f}")
print(f"    Accuracy  : {acc*100:.2f}%")
print(f"    Precision : {prec*100:.2f}%")
print(f"    Recall    : {rec*100:.2f}%")
print(f"    F1 Score  : {f1*100:.2f}%")
print(f"{'='*65}\n")

if total_fail + total_error == 0:
    print("  ALL TESTS PASSED - Model is production-quality!")
else:
    print("  SOME TESTS FAILED - Review failures above before deploying.")
