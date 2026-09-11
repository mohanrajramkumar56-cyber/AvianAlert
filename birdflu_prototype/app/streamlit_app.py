
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from src.predict import predict_from_dict

st.set_page_config(page_title='Bird Flu Screener', layout='centered')
st.title('Bird Flu Screening Prototype (Demo)')
st.write('Prototype with synthetic data. Not for medical use.')

with st.form('patient'):
    age = st.number_input('Age', 0, 120, 30)
    gender = st.selectbox('Gender', ['M','F','Other'])
    occupation = st.selectbox('Occupation', ['Farmer','Healthcare','Student','Office','Other'])
    travel = st.checkbox('Traveled recently?')
    contact_birds = st.checkbox('Contact with birds?')
    poultry_visit = st.checkbox('Visited poultry farm?')
    undercooked = st.checkbox('Ate undercooked poultry/eggs?')
    contact_person = st.checkbox('Contact with infected person?')
    st.write('Symptoms:')
    fever = st.checkbox('Fever')
    cough = st.checkbox('Cough')
    sore_throat = st.checkbox('Sore throat')
    muscle_ache = st.checkbox('Muscle ache')
    headache = st.checkbox('Headache')
    malaise = st.checkbox('Malaise')
    runny_nose = st.checkbox('Runny nose')
    diarrhea = st.checkbox('Diarrhea')
    dyspnea = st.checkbox('Difficulty breathing')
    submitted = st.form_submit_button('Predict')

if submitted:
    # Build dict matching synthetic preprocessing expectation (simple demo)
    d = {
        'age': age,
        'travel': 1 if travel else 0,
        'contact_birds': 1 if contact_birds else 0,
        'poultry_visit': 1 if poultry_visit else 0,
        'undercooked': 1 if undercooked else 0,
        'contact_person': 1 if contact_person else 0,
        'fever': 1 if fever else 0,
        'cough': 1 if cough else 0,
        'sore_throat': 1 if sore_throat else 0,
        'muscle_ache': 1 if muscle_ache else 0,
        'headache': 1 if headache else 0,
        'malaise': 1 if malaise else 0,
        'runny_nose': 1 if runny_nose else 0,
        'diarrhea': 1 if diarrhea else 0,
        'dyspnea': 1 if dyspnea else 0,
        'wbc': 7000,
        'rt_pcr': 0,
        'chest_xray': 0
    }
    prob = predict_from_dict(d)
    st.metric('Predicted probability of bird flu', f'{prob*100:.1f}%')
    if prob >= 0.6:
        st.error('HIGH RISK — Seek clinical testing immediately.')
    elif prob >= 0.4:
        st.warning('MEDIUM RISK — Consider testing and consult a doctor.')
    else:
        st.success('LOW RISK — Monitor symptoms and consult if worsening.')
