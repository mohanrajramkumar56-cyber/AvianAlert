import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Train model once and cache it (survives reruns, trains only once per session) ──
@st.cache_resource(show_spinner="Training model on first run... please wait ~60 seconds")
def load_or_train_model():
    model_path = Path(BASE_DIR) / 'models' / 'rf_model.pkl'
    data_path  = Path(BASE_DIR) / 'data'   / 'synthetic_data.csv'

    # If model file already exists on disk, load it
    if model_path.exists():
        return joblib.load(model_path)

    # Otherwise train from scratch
    from data_generator import generate
    from preprocess import load_and_preprocess
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from xgboost import XGBClassifier

    # Generate data if missing
    if not data_path.exists():
        generate(10000)

    X_train, X_test, y_train, y_test = load_and_preprocess(path=data_path)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    scale_pos = float((y_train == 0).sum()) / float(max((y_train == 1).sum(), 1))
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric='auc',
        random_state=42, n_jobs=-1, verbosity=0
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8, random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
        voting='soft', n_jobs=-1
    )
    ensemble.fit(X_train, y_train)

    # Save for future runs
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, model_path)

    return ensemble

MODEL = load_or_train_model()

def predict(d):
    df = pd.DataFrame([d])
    df = pd.get_dummies(df)
    df = df.reindex(columns=MODEL.feature_names_in_, fill_value=0)
    prob = MODEL.predict_proba(df)[:, 1][0]
    return float(prob)

# ── UI ──
st.set_page_config(page_title='AvianAlert — Bird Flu Screener', page_icon='🦠', layout='centered')

st.title('🦠 AvianAlert')
st.subheader('AI-Powered Bird Flu (H5N1) Screening')
st.caption('Prototype with synthetic data only. Not for real medical use.')
st.divider()

with st.form('patient_form'):
    st.subheader('👤 Patient Info')
    col1, col2 = st.columns(2)
    with col1:
        age        = st.number_input('Age', 0, 120, 30)
        gender     = st.selectbox('Gender', ['M', 'F', 'Other'])
    with col2:
        occupation = st.selectbox('Occupation', ['Farmer', 'Healthcare', 'Student', 'Office', 'Other'])
        wbc        = st.number_input('WBC Count (cells/uL)', 2000, 20000, value=7000, step=100,
                                     help='Normal: 4500-11000. Low WBC is a key bird flu sign.')

    st.subheader('🌍 Exposure History')
    col3, col4 = st.columns(2)
    with col3:
        travel         = st.checkbox('Traveled recently?')
        contact_birds  = st.checkbox('Contact with birds?')
        poultry_visit  = st.checkbox('Visited poultry farm?')
    with col4:
        undercooked    = st.checkbox('Ate undercooked poultry/eggs?')
        contact_person = st.checkbox('Contact with infected person?')
        rt_pcr         = st.checkbox('PCR test positive?')

    st.subheader('🤒 Symptoms')
    col5, col6, col7 = st.columns(3)
    with col5:
        fever       = st.checkbox('Fever')
        cough       = st.checkbox('Cough')
        sore_throat = st.checkbox('Sore throat')
    with col6:
        muscle_ache = st.checkbox('Muscle ache')
        headache    = st.checkbox('Headache')
        malaise     = st.checkbox('Malaise / Fatigue')
    with col7:
        runny_nose  = st.checkbox('Runny nose')
        diarrhea    = st.checkbox('Diarrhea')
        dyspnea     = st.checkbox('Difficulty breathing')

    chest_xray = st.checkbox('Chest X-ray abnormal?')
    submitted  = st.form_submit_button('🔍 Predict Risk', use_container_width=True, type='primary')

if submitted:
    d = {
        'age':            age,
        'travel':         1 if travel else 0,
        'contact_birds':  1 if contact_birds else 0,
        'poultry_visit':  1 if poultry_visit else 0,
        'undercooked':    1 if undercooked else 0,
        'contact_person': 1 if contact_person else 0,
        'fever':          1 if fever else 0,
        'cough':          1 if cough else 0,
        'sore_throat':    1 if sore_throat else 0,
        'muscle_ache':    1 if muscle_ache else 0,
        'headache':       1 if headache else 0,
        'malaise':        1 if malaise else 0,
        'runny_nose':     1 if runny_nose else 0,
        'diarrhea':       1 if diarrhea else 0,
        'dyspnea':        1 if dyspnea else 0,
        'wbc':            wbc,
        'rt_pcr':         1 if rt_pcr else 0,
        'chest_xray':     1 if chest_xray else 0,
    }

    prob = predict(d)

    st.divider()
    st.subheader('📊 Prediction Result')

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.metric('Bird Flu Risk', f'{prob * 100:.1f}%')
    with col_r2:
        st.progress(int(prob * 100), text=f'Model confidence: {prob*100:.1f}%')

    if prob >= 0.6:
        st.error('🚨 HIGH RISK — Seek clinical testing and isolate immediately.')
        st.markdown("""
        **Recommended actions:**
        - Isolate the patient immediately
        - Order RT-PCR test for H5N1
        - Notify public health authorities
        - Consider antiviral (Oseltamivir)
        """)
    elif prob >= 0.4:
        st.warning('⚠️ MEDIUM RISK — Consider testing and consult a doctor.')
        st.markdown("""
        **Recommended actions:**
        - Monitor closely for 24–48 hours
        - Consider RT-PCR if symptoms worsen
        - Review exposure history carefully
        """)
    else:
        st.success('✅ LOW RISK — Monitor symptoms. Consult if worsening.')
        st.markdown("""
        **Recommended actions:**
        - Continue monitoring symptoms
        - Return if fever persists > 3 days
        - Standard supportive care
        """)

    st.caption('This is a screening prototype only. Do not use for actual medical diagnosis.')
