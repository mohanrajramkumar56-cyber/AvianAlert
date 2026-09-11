"""
Edge case & tricky data tests for Bird Flu Predictor
Tests: wrong values, boundary cases, contradictory data, missing risk detection
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'birdflu_prototype', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'birdflu_webapp'))

from predict import predict_from_dict as proto_predict
from src.predict import predict_from_dict as webapp_predict, predict_proba_from_dict

label_map = {0: 'No Flu Detected', 1: 'Possible Flu', 2: 'Possible Bird Flu'}

passed = 0
failed = 0

def check(name, result, expected_label=None, expected_proto_min=None, expected_proto_max=None, note=""):
    global passed, failed
    ok = True
    reasons = []

    if expected_label is not None and result['webapp'] != expected_label:
        ok = False
        reasons.append(f"webapp expected={label_map[expected_label]} got={label_map[result['webapp']]}")
    if expected_proto_min is not None and result['proto'] < expected_proto_min:
        ok = False
        reasons.append(f"proto risk={result['proto']*100:.1f}% expected >= {expected_proto_min*100:.0f}%")
    if expected_proto_max is not None and result['proto'] > expected_proto_max:
        ok = False
        reasons.append(f"proto risk={result['proto']*100:.1f}% expected <= {expected_proto_max*100:.0f}%")

    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    proba = result['proba']
    print(f"  [{status}] {name}")
    print(f"         Webapp  : {label_map[result['webapp']]}  "
          f"(No Flu={proba[0]*100:.0f}%  Flu={proba[1]*100:.0f}%  BirdFlu={proba[2]*100:.0f}%)")
    print(f"         Proto   : {result['proto']*100:.1f}% bird flu risk")
    if note:
        print(f"         Note    : {note}")
    if not ok:
        for r in reasons:
            print(f"         !! {r}")
    print()

def run(proto_data, webapp_data):
    proto_risk = proto_predict(proto_data)
    webapp_result = webapp_predict(webapp_data)
    proba = predict_proba_from_dict(webapp_data)
    return {'proto': proto_risk, 'webapp': webapp_result, 'proba': proba}


print("=" * 65)
print("BIRD FLU PREDICTOR - EDGE CASE & RISK DETECTION TESTS")
print("=" * 65)

# ----------------------------------------------------------------
print("\n--- GROUP 1: OBVIOUS CASES (should be easy to get right) ---\n")

# 1. Classic bird flu patient
r = run(
    proto_data={'age':50,'travel':1,'contact_birds':1,'poultry_visit':1,'undercooked':1,
                'contact_person':0,'fever':1,'cough':1,'sore_throat':1,'muscle_ache':1,
                'headache':1,'malaise':1,'runny_nose':0,'diarrhea':1,'dyspnea':1,
                'wbc':3200,'rt_pcr':1,'chest_xray':1},
    webapp_data={'WBC_count':3200,'RBC_count':3.5,'Hemoglobin':10.0,'Platelet_count':100000,
                 'CRP':35.0,'ALT':90.0,'AST':85.0,'Urine_pH':5.0,
                 'Fever':1,'Cough':1,'Sore_throat':1}
)
check("Classic Bird Flu patient (all red flags)", r,
      expected_label=2, expected_proto_min=0.7,
      note="Low WBC, high CRP, high liver, fever+cough+sore throat, PCR+")

# 2. Completely healthy person
r = run(
    proto_data={'age':28,'travel':0,'contact_birds':0,'poultry_visit':0,'undercooked':0,
                'contact_person':0,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':7800,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':7800,'RBC_count':5.0,'Hemoglobin':15.0,'Platelet_count':300000,
                 'CRP':1.5,'ALT':18.0,'AST':16.0,'Urine_pH':6.5,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("Completely healthy person", r,
      expected_label=0, expected_proto_max=0.15,
      note="Normal labs, no symptoms, no exposure")

# ----------------------------------------------------------------
print("--- GROUP 2: TRICKY / MISLEADING CASES ---\n")

# 3. High risk exposure but NO symptoms yet (early stage)
r = run(
    proto_data={'age':35,'travel':1,'contact_birds':1,'poultry_visit':1,'undercooked':1,
                'contact_person':1,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':6500,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':6500,'RBC_count':4.5,'Hemoglobin':13.5,'Platelet_count':220000,
                 'CRP':4.0,'ALT':28.0,'AST':25.0,'Urine_pH':6.0,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("High exposure, NO symptoms yet (incubation?)", r,
      note="Exposed to birds+travel+contact but labs still normal - tricky early case")

# 4. Severe symptoms but NO bird contact (regular flu)
r = run(
    proto_data={'age':40,'travel':0,'contact_birds':0,'poultry_visit':0,'undercooked':0,
                'contact_person':0,'fever':1,'cough':1,'sore_throat':1,'muscle_ache':1,
                'headache':1,'malaise':1,'runny_nose':1,'diarrhea':0,'dyspnea':0,
                'wbc':9000,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':9000,'RBC_count':4.6,'Hemoglobin':14.0,'Platelet_count':260000,
                 'CRP':8.0,'ALT':25.0,'AST':22.0,'Urine_pH':6.5,
                 'Fever':1,'Cough':1,'Sore_throat':1}
)
check("Severe symptoms, NO bird contact (regular flu)", r,
      expected_label=1,
      note="Many symptoms but no exposure + normal WBC/liver = likely regular flu not bird flu")

# 5. Only breathing difficulty + low WBC (silent bird flu)
r = run(
    proto_data={'age':60,'travel':1,'contact_birds':1,'poultry_visit':0,'undercooked':0,
                'contact_person':0,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':1,
                'wbc':3800,'rt_pcr':0,'chest_xray':1},
    webapp_data={'WBC_count':3800,'RBC_count':4.0,'Hemoglobin':12.0,'Platelet_count':130000,
                 'CRP':22.0,'ALT':68.0,'AST':62.0,'Urine_pH':5.5,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("Only dyspnea + abnormal labs (silent bird flu)", r,
      expected_label=2, expected_proto_min=0.4,
      note="No classic symptoms but dangerous lab values - should NOT be dismissed")

# 6. Elderly patient, mild symptoms, slight lab changes
r = run(
    proto_data={'age':75,'travel':0,'contact_birds':1,'poultry_visit':0,'undercooked':0,
                'contact_person':0,'fever':1,'cough':1,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':1,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':5500,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':5500,'RBC_count':4.2,'Hemoglobin':13.0,'Platelet_count':190000,
                 'CRP':9.0,'ALT':40.0,'AST':38.0,'Urine_pH':6.0,
                 'Fever':1,'Cough':1,'Sore_throat':0}
)
check("Elderly patient, mild symptoms + bird contact", r,
      note="Age 75, bird contact, mild fever/cough - borderline case")

# ----------------------------------------------------------------
print("--- GROUP 3: CONTRADICTORY / WRONG DATA ---\n")

# 7. Extremely high WBC (opposite of bird flu) + all symptoms
r = run(
    proto_data={'age':33,'travel':0,'contact_birds':0,'poultry_visit':0,'undercooked':0,
                'contact_person':0,'fever':1,'cough':1,'sore_throat':1,'muscle_ache':1,
                'headache':1,'malaise':1,'runny_nose':1,'diarrhea':0,'dyspnea':0,
                'wbc':18000,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':18000,'RBC_count':5.5,'Hemoglobin':16.0,'Platelet_count':450000,
                 'CRP':6.0,'ALT':20.0,'AST':18.0,'Urine_pH':7.0,
                 'Fever':1,'Cough':1,'Sore_throat':1}
)
check("All symptoms BUT very high WBC (bacterial infection?)", r,
      expected_label=1,
      note="High WBC = fighting bacteria, not bird flu. Symptoms + elevated WBC = bacterial flu more likely")

# 8. Perfect labs + PCR positive (contradictory)
r = run(
    proto_data={'age':30,'travel':1,'contact_birds':1,'poultry_visit':1,'undercooked':0,
                'contact_person':0,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':7500,'rt_pcr':1,'chest_xray':0},
    webapp_data={'WBC_count':7500,'RBC_count':4.8,'Hemoglobin':14.5,'Platelet_count':280000,
                 'CRP':3.0,'ALT':22.0,'AST':20.0,'Urine_pH':6.5,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("PCR positive but perfect labs & no symptoms (asymptomatic?)", r,
      expected_proto_min=0.4,
      note="PCR+ with bird exposure but no symptoms yet - model should flag risk")

# 9. Impossible/boundary lab values (stress test)
r = run(
    proto_data={'age':0,'travel':0,'contact_birds':0,'poultry_visit':0,'undercooked':0,
                'contact_person':0,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':2000,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':2000,'RBC_count':2.5,'Hemoglobin':7.0,'Platelet_count':50000,
                 'CRP':0.0,'ALT':0.0,'AST':0.0,'Urine_pH':4.0,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("Minimum boundary values (no crash test)", r,
      note="All at minimum allowed values - model should not crash")

# 10. Maximum boundary values (no crash test)
r = run(
    proto_data={'age':100,'travel':1,'contact_birds':1,'poultry_visit':1,'undercooked':1,
                'contact_person':1,'fever':1,'cough':1,'sore_throat':1,'muscle_ache':1,
                'headache':1,'malaise':1,'runny_nose':1,'diarrhea':1,'dyspnea':1,
                'wbc':20000,'rt_pcr':1,'chest_xray':1},
    webapp_data={'WBC_count':20000,'RBC_count':6.0,'Hemoglobin':18.0,'Platelet_count':500000,
                 'CRP':50.0,'ALT':100.0,'AST':100.0,'Urine_pH':8.5,
                 'Fever':1,'Cough':1,'Sore_throat':1}
)
check("Maximum boundary values (all maxed out)", r,
      note="All at maximum allowed values - model should not crash")

# ----------------------------------------------------------------
print("--- GROUP 4: REAL-WORLD RISK SCENARIOS ---\n")

# 11. Poultry farm worker with just fatigue (no classic symptoms)
r = run(
    proto_data={'age':42,'travel':0,'contact_birds':1,'poultry_visit':1,'undercooked':1,
                'contact_person':0,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':1,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':5200,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':5200,'RBC_count':4.3,'Hemoglobin':13.5,'Platelet_count':175000,
                 'CRP':11.0,'ALT':45.0,'AST':42.0,'Urine_pH':6.0,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("Poultry farm worker, only fatigue + borderline labs", r,
      note="High-risk job, slight lab changes, no dramatic symptoms - should show at least Flu risk")

# 12. Child (age 8) with bird contact and fever
r = run(
    proto_data={'age':8,'travel':0,'contact_birds':1,'poultry_visit':1,'undercooked':0,
                'contact_person':0,'fever':1,'cough':1,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':1,'diarrhea':0,'dyspnea':0,
                'wbc':6000,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':6000,'RBC_count':4.3,'Hemoglobin':12.5,'Platelet_count':230000,
                 'CRP':7.0,'ALT':25.0,'AST':22.0,'Urine_pH':6.5,
                 'Fever':1,'Cough':1,'Sore_throat':0}
)
check("Child (age 8) with bird contact + fever/cough", r,
      note="Children can be high risk - should not be dismissed")

# 13. Traveler returned from high-risk zone, no symptoms yet
r = run(
    proto_data={'age':38,'travel':1,'contact_birds':1,'poultry_visit':0,'undercooked':1,
                'contact_person':1,'fever':0,'cough':0,'sore_throat':0,'muscle_ache':0,
                'headache':0,'malaise':0,'runny_nose':0,'diarrhea':0,'dyspnea':0,
                'wbc':7200,'rt_pcr':0,'chest_xray':0},
    webapp_data={'WBC_count':7200,'RBC_count':4.7,'Hemoglobin':14.0,'Platelet_count':255000,
                 'CRP':3.5,'ALT':27.0,'AST':24.0,'Urine_pH':6.5,
                 'Fever':0,'Cough':0,'Sore_throat':0}
)
check("Traveler from high-risk zone, no symptoms (incubation period)", r,
      note="Just returned, labs still clean - should still show some elevated risk due to exposure")

# ----------------------------------------------------------------
print("=" * 65)
print(f"RESULTS: {passed} PASSED  |  {failed} FAILED  |  {passed+failed} TOTAL")
print("=" * 65)

if failed == 0:
    print("All tests passed - model detects risk correctly!")
else:
    print(f"{failed} test(s) did not match expected behaviour - review above.")
