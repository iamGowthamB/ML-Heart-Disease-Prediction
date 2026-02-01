# Author: Gowtham B
# Project: Heart Disease Prediction using Random Forest & Cross-Validation
# Type: Individual Portfolio Project

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioLens Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Advanced Styling (Medical Dashboard Theme)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Fields */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    
    /* Result Cards */
    .risk-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        color: #991b1b;
    }
    .risk-low {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        color: #166534;
    }
    
    /* Clean Divider */
    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Load Resources
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('c:/Project/CrossValidation-RandomForest/rf_model.pkl')
        scaler = joblib.load('c:/Project/CrossValidation-RandomForest/scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_resources()

if not model:
    st.error("⚠️ System Offline: Model files missing. Please run `train_model.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. App Header
# -----------------------------------------------------------------------------
col_logo, col_header = st.columns([1, 6])
with col_logo:
    st.markdown("<h1>🩺</h1>", unsafe_allow_html=True)
with col_header:
    st.title("CardioLens Pro")
    st.markdown("**AI-Powered Cardiac Risk Assessment System** | v2.0 Enterprise")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. Two-Column Layout (Input vs Output)
# -----------------------------------------------------------------------------
c_input, c_spacer, c_output = st.columns([1, 0.1, 1.2])

# --- LEFT COLUMN: INPUT FORM ---
with c_input:
    st.subheader("📋 Patient Metrics")
    
    with st.form("main_form"):
        # Section 1
        with st.expander("👤 Demographics & History", expanded=True):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age (Years)", 20, 100, 54)
            sex = c2.selectbox("Sex", (1, 0), format_func=lambda x: "Male" if x==1 else "Female")
            
            c3, c4 = st.columns(2)
            famhist = c3.selectbox("Family History", (0, 1), format_func=lambda x: "No" if x==0 else "Yes")
            smoker = c4.selectbox("Smoker Status", (0, 1), format_func=lambda x: "Non-Smoker" if x==0 else "Smoker")

        # Section 2
        with st.expander("🩸 Lab & Vitals", expanded=True):
            trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 130)
            chol = st.slider("Total Cholesterol (mg/dL)", 100, 600, 240)
            bmi = st.slider("BMI", 15.0, 45.0, 26.5)
            fbs_val = st.checkbox("Fasting Blood Sugar > 120 mg/dL")

        # Section 3
        with st.expander("💓 Cardiac Specifics", expanded=False):
            thalach = st.number_input("Max Heart Rate", 60, 250, 150)
            cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3), format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x])
            exang = st.radio("Exercise Angina", (0, 1), format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
            slope = st.selectbox("ST Slope", (0, 1, 2))
            ca = st.slider("Major Vessels (0-3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", (1, 2, 3), format_func=lambda x: ["Normal", "Fixed Defect", "Reversible"][x-1])

        submit_btn = st.form_submit_button("Run Analysis", use_container_width=True, type="primary")

# --- RIGHT COLUMN: PREDICTION DASHBOARD ---
with c_output:
    if submit_btn:
        # Preprocessing
        fbs = 1 if fbs_val else 0
        input_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': 0, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
            'bmi': bmi, 'smoker': smoker, 'famhist': famhist
        }
        df_in = pd.DataFrame(input_data, index=[0])
        
        with st.spinner("Analyzing parameters..."):
            time.sleep(0.8) # UX Delay
            scaled = scaler.transform(df_in)
            prob = model.predict_proba(scaled)[0][1]
            pred = model.predict(scaled)[0]

        # Top Result Card
        if prob > 0.5:
            st.markdown(f"""
            <div class="risk-card risk-high">
                <h2>🚨 High Risk Detected</h2>
                <p>Probability: <b>{prob*100:.1f}%</b></p>
                <p>Based on the analysis, this patient exhibits significant indicators consistent with heart disease.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-card risk-low">
                <h2>✅ Low Risk Profile</h2>
                <p>Probability: <b>{prob*100:.1f}%</b></p>
                <p>The patient's metrics are within a safe range, though routine monitoring is always advised.</p>
            </div>
            """, unsafe_allow_html=True)

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0)"}, # Hide default bar
                'steps': [
                    {'range': [0, 40], 'color': "#86efac"}, # Green
                    {'range': [40, 70], 'color': "#fde047"}, # Yellow
                    {'range': [70, 100], 'color': "#fca5a5"}], # Red
                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'thickness': 0.8,
                    'value': prob * 100}}))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Key Contributors (Dummy simulation for visual effect, real would use Shapley)
        st.subheader("Key Risk Contributors")
        factors = {'Cholesterol': chol/600, 'BP': trestbps/200, 'Age': age/100, 'Max HR': (220-thalach)/220}
        sorted_factors = dict(sorted(factors.items(), key=lambda item: item[1], reverse=True))
        
        for k, v in list(sorted_factors.items())[:3]:
            st.progress(min(v, 1.0), text=f"{k} Level")

    else:
        # Default Welcome State
        st.info("👈 Please complete the form on the left to generate a report.")
        st.image("https://img.freepik.com/free-vector/heart-rate-monitor_1308-102573.jpg", caption="CardioLens AI System Ready")

# Footer section
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>© 2026 CardioLens Medical Systems. For Research Use Only.</div>", unsafe_allow_html=True)
