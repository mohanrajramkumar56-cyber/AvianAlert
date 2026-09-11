import sys
import os

print('='*55)
print('FULL END-TO-END TEST')
print('='*55)

# ---- PROTOTYPE TEST ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'birdflu_prototype', 'src'))

print()
print('[1] PROTOTYPE - Prediction Test')
from predict import predict_from_dict

sick = {
    'age': 45, 'travel': 1, 'contact_birds': 1, 'poultry_visit': 1, 'undercooked': 0,
    'contact_person': 0, 'fever': 1, 'cough': 1, 'sore_throat': 1, 'muscle_ache': 1,
    'headache': 1, 'malaise': 1, 'runny_nose': 0, 'diarrhea': 1, 'dyspnea': 1,
    'wbc': 4000, 'rt_pcr': 1, 'chest_xray': 1
}
p = predict_from_dict(sick)
print(f'  Sick patient    -> {p*100:.1f}% bird flu risk  [expect HIGH > 50%]')

healthy = {
    'age': 25, 'travel': 0, 'contact_birds': 0, 'poultry_visit': 0, 'undercooked': 0,
    'contact_person': 0, 'fever': 0, 'cough': 0, 'sore_throat': 0, 'muscle_ache': 0,
    'headache': 0, 'malaise': 0, 'runny_nose': 0, 'diarrhea': 0, 'dyspnea': 0,
    'wbc': 8000, 'rt_pcr': 0, 'chest_xray': 0
}
p2 = predict_from_dict(healthy)
print(f'  Healthy patient -> {p2*100:.1f}% bird flu risk  [expect LOW < 20%]')

assert p > 0.5,  f'FAIL: sick patient should be HIGH risk, got {p:.2f}'
assert p2 < 0.3, f'FAIL: healthy patient should be LOW risk, got {p2:.2f}'
print('  PASSED ✓')

# ---- WEBAPP TEST ----
print()
print('[2] WEBAPP - Prediction Test')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'birdflu_webapp'))
from src.predict import predict_from_dict as webapp_predict, predict_proba_from_dict

label_map = {0: 'No Flu Detected', 1: 'Possible Flu', 2: 'Possible Bird Flu'}

# Bird flu patient (low WBC, high CRP, high liver enzymes, fever + cough + sore throat)
bird_flu_patient = {
    'WBC_count': 3500, 'RBC_count': 3.9, 'Hemoglobin': 10.5,
    'Platelet_count': 120000, 'CRP': 28.0, 'ALT': 75.0, 'AST': 70.0,
    'Urine_pH': 5.5, 'Fever': 1, 'Cough': 1, 'Sore_throat': 1
}
r1 = webapp_predict(bird_flu_patient)
pb1 = predict_proba_from_dict(bird_flu_patient)
print(f'  Bird flu patient -> {label_map[r1]}')
print(f'    Probas: No Flu={pb1[0]*100:.1f}%  Flu={pb1[1]*100:.1f}%  Bird Flu={pb1[2]*100:.1f}%')

# Healthy patient
healthy_patient = {
    'WBC_count': 7500, 'RBC_count': 4.9, 'Hemoglobin': 14.5,
    'Platelet_count': 280000, 'CRP': 2.0, 'ALT': 22.0, 'AST': 18.0,
    'Urine_pH': 6.5, 'Fever': 0, 'Cough': 0, 'Sore_throat': 0
}
r2 = webapp_predict(healthy_patient)
pb2 = predict_proba_from_dict(healthy_patient)
print(f'  Healthy patient  -> {label_map[r2]}')
print(f'    Probas: No Flu={pb2[0]*100:.1f}%  Flu={pb2[1]*100:.1f}%  Bird Flu={pb2[2]*100:.1f}%')

# Flu patient
flu_patient = {
    'WBC_count': 6000, 'RBC_count': 4.3, 'Hemoglobin': 13.0,
    'Platelet_count': 200000, 'CRP': 12.0, 'ALT': 35.0, 'AST': 32.0,
    'Urine_pH': 6.0, 'Fever': 1, 'Cough': 1, 'Sore_throat': 0
}
r3 = webapp_predict(flu_patient)
pb3 = predict_proba_from_dict(flu_patient)
print(f'  Flu patient      -> {label_map[r3]}')
print(f'    Probas: No Flu={pb3[0]*100:.1f}%  Flu={pb3[1]*100:.1f}%  Bird Flu={pb3[2]*100:.1f}%')

assert r1 == 2, f'FAIL: bird flu patient should be class 2, got {r1}'
assert r2 == 0, f'FAIL: healthy patient should be class 0, got {r2}'
print('  PASSED ✓')

print()
print('='*55)
print('ALL TESTS PASSED ✓')
print('='*55)
print()
print('To run the webapp, use:')
print('  cd birdflu_webapp')
print('  streamlit run streamlit_app.py')
