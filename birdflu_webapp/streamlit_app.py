import streamlit as st
from src.predict import predict_from_dict, predict_proba_from_dict

st.set_page_config(page_title="Bird Flu Predictor", layout="centered")

st.title("🦠 Bird Flu Predictor")
st.caption("Prototype — synthetic data only. Not for real medical use.")
st.divider()

st.subheader("🩸 Blood Test Results")
col1, col2 = st.columns(2)
with col1:
    wbc       = st.number_input("WBC Count (cells/μL)", 2000, 20000, value=7000, step=100,
                                help="Normal: 4500–11000. Low WBC can indicate Bird Flu.")
    rbc       = st.number_input("RBC Count (million/μL)", 2.5, 6.0, value=4.5, step=0.1)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", 7.0, 18.0, value=13.5, step=0.1)
    platelet  = st.number_input("Platelet Count (/μL)", 50000, 500000, value=250000, step=1000,
                                help="Low platelets (<150k) can indicate Bird Flu.")
with col2:
    crp       = st.number_input("CRP (mg/L)", 0.0, 50.0, value=5.0, step=0.5,
                                help="Normal: <5. High CRP = inflammation/infection.")
    alt       = st.number_input("ALT (U/L)", 0.0, 100.0, value=30.0, step=1.0,
                                help="Normal: 7–56. High ALT = liver stress.")
    ast       = st.number_input("AST (U/L)", 0.0, 100.0, value=28.0, step=1.0,
                                help="Normal: 10–40. High AST = liver stress.")
    urine_ph  = st.number_input("Urine pH", 4.0, 8.5, value=6.0, step=0.1)

st.subheader("🤒 Symptoms")
col3, col4, col5 = st.columns(3)
with col3:
    fever      = st.radio("Fever", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col4:
    cough      = st.radio("Cough", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col5:
    sore_throat = st.radio("Sore Throat", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    fields = {
        "WBC_count":     wbc,
        "RBC_count":     rbc,
        "Hemoglobin":    hemoglobin,
        "Platelet_count": platelet,
        "CRP":           crp,
        "ALT":           alt,
        "AST":           ast,
        "Urine_pH":      urine_ph,
        "Fever":         fever,
        "Cough":         cough,
        "Sore_throat":   sore_throat,
    }

    result = predict_from_dict(fields)
    proba  = predict_proba_from_dict(fields)

    label_map = {
        0: ("✅ No Flu Detected",     "success"),
        1: ("⚠️ Possible Flu",        "warning"),
        2: ("🚨 Possible Bird Flu",   "error"),
    }
    label, msg_type = label_map[result]

    st.subheader("📊 Prediction Result")
    if msg_type == "success":
        st.success(f"**{label}**")
    elif msg_type == "warning":
        st.warning(f"**{label}**")
    else:
        st.error(f"**{label}**")

    st.subheader("Confidence Breakdown")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("No Flu",         f"{proba[0]*100:.1f}%")
    col_b.metric("Possible Flu",   f"{proba[1]*100:.1f}%")
    col_c.metric("Possible Bird Flu", f"{proba[2]*100:.1f}%")

    st.progress(int(proba[result] * 100), text=f"Model confidence: {proba[result]*100:.1f}%")

    if result == 2:
        st.error("⚠️ HIGH RISK — Seek clinical testing and isolate immediately.")
    elif result == 1:
        st.warning("🔶 MEDIUM RISK — Consult a doctor and consider testing.")
    else:
        st.info("🟢 LOW RISK — Monitor symptoms. See a doctor if they worsen.")

    st.caption("This is a prototype for demonstration only. Do not use for actual medical decisions.")
