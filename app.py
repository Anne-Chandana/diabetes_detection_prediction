# ============================================
# DIABETES RISK PREDICTION SYSTEM
# A Machine Learning Tool for Early Diabetes Detection
# Model: Decision Tree Classifier | Accuracy: 89.80 percent
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import os


import streamlit as st
import os


# ✅ set_page_config MUST be first — nothing st.* before this
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="auto"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f0f4f8;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(120deg, #0f4c5c 0%, #1e6f5c 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .title-container h1 {
        color: white;
        font-size: 42px;
        margin: 0;
        letter-spacing: 1px;
    }
    .title-container p {
        color: rgba(255,255,255,0.9);
        font-size: 16px;
        margin-top: 10px;
    }
    
    /* Card styling for input sections */
    .card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e7e9;
    }
    .card h3 {
        color: #0f4c5c;
        border-left: 4px solid #1e6f5c;
        padding-left: 15px;
        margin-top: 0;
        margin-bottom: 20px;
    }
    
    /* Result container styling */
    .result-container {
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
    }
    .result-high {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
    }
    .result-low {
        background: linear-gradient(135deg, #1e6f5c 0%, #2ecc71 100%);
        color: white;
    }
    .result-container h2 {
        margin: 0;
        font-size: 28px;
    }
    .result-container p {
        font-size: 18px;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Risk factors box */
    .risk-box {
        background-color: #fff8e7;
        border-left: 4px solid #f39c12;
        border-radius: 12px;
        padding: 15px;
        margin: 20px 0;
    }
    .risk-box h4 {
        color: #e67e22;
        margin: 0 0 10px 0;
    }
    .risk-box ul {
        margin: 0;
        padding-left: 20px;
    }
    .risk-box li {
        margin: 5px 0;
        color: #2c3e50;
    }
    
    /* Recommendations box */
    .rec-box {
        background-color: #e8f8f5;
        border-left: 4px solid #1e6f5c;
        border-radius: 12px;
        padding: 15px;
        margin: 20px 0;
    }
    .rec-box h4 {
        color: #1e6f5c;
        margin: 0 0 10px 0;
    }
    .rec-box ul {
        margin: 0;
        padding-left: 20px;
    }
    .rec-box li {
        margin: 5px 0;
        color: #2c3e50;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background-color: #e8f0fe;
        border-radius: 15px;
        padding: 15px;
        margin-top: 20px;
    }
    .sidebar-info h4 {
        color: #0f4c5c;
        margin: 0 0 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(120deg, #0f4c5c 0%, #1e6f5c 100%);
        color: white;
        font-size: 18px;
        padding: 12px 28px;
        border-radius: 40px;
        width: 100%;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Input field styling */
    .stSlider label, .stSelectbox label, .stRadio label {
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        color: #7f8c8d;
        font-size: 12px;
        border-top: 1px solid #e0e7e9;
        margin-top: 40px;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1e6f5c, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# FIX: Removed st.error() / st.info() from inside @st.cache_resource.
# Streamlit UI calls inside cached functions cause the app to hang on startup.
# Instead, we return the error message as a string and display it outside.
# ============================================

@st.cache_resource
def load_models():
    """Load the trained Decision Tree model and preprocessors"""
    try:
        # Load the main model
        model = joblib.load('models/diabetes_model.pkl')
        
        # Load feature engineering info
        feature_info = joblib.load('models/feature_engineering_info.pkl')
        
        # Extract components from feature_info
        if isinstance(feature_info, dict):
            scaler = feature_info.get('scaler', None)
            feature_columns = feature_info.get('feature_columns', None)
            numerical_features = feature_info.get('numerical_features', None)
            gender_mapping = feature_info.get('gender_mapping', {'Female': 0, 'Male': 1, 'Other': 2})
            smoking_mapping = feature_info.get('smoking_mapping', {
                'never': 0, 'former': 1, 'current': 2,
                'ever': 3, 'not current': 4, 'No Info': 5
            })
        else:
            # Try loading individual files if feature_info is not a dict
            scaler = joblib.load('models/scaler.pkl')
            feature_columns = joblib.load('models/feature_columns.pkl')
            gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
            smoking_mapping = {
                'never': 0, 'former': 1, 'current': 2,
                'ever': 3, 'not current': 4, 'No Info': 5
            }
            numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                                   'bmi_category', 'age_group', 'glucose_hba1c_ratio',
                                   'age_bmi_interaction', 'risk_score']

        # If scaler wasn't in feature_info dict, try to load it separately
        if scaler is None:
            try:
                scaler = joblib.load('models/scaler.pkl')
            except FileNotFoundError:
                raise FileNotFoundError("Could not find scaler.pkl")

        # Return None as the error string to indicate success
        return model, scaler, feature_columns, numerical_features, gender_mapping, smoking_mapping, None

    except Exception as e:
        # FIX: Return the error as a plain string — never call st.* inside cache_resource
        return None, None, None, None, None, None, str(e)


# Load all model components
model, scaler, feature_columns, numerical_features, gender_mapping, smoking_mapping, load_error = load_models()

# FIX: Display any load errors here, outside the cached function
if load_error is not None:
    model_loaded = False
    st.error(f"Unable to load the prediction model. Error: {load_error}")
    st.info("Please ensure all model files are present in the 'models/' folder.")
else:
    model_loaded = True

# ============================================
# HEADER SECTION
# ============================================

st.markdown("""
<div class="title-container">
    <h1>Diabetes Risk Assessment Tool</h1>
    <p>Evidence-based diabetes risk prediction using machine learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR INFORMATION
# ============================================

with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown("""
    This tool uses a machine learning model trained on health data to assess diabetes risk.
    """)

    st.markdown("---")

    st.markdown("### Key Risk Factors")
    st.markdown("""
    - High HbA1c level
    - Elevated blood glucose
    - Age over 45 years
    - High BMI
    - Hypertension
    - Heart disease
    """)

    st.markdown("---")

    st.markdown("### Clinical Thresholds")
    st.markdown("""
    **HbA1c:** 6.5% or higher indicates diabetes  
    **Blood Glucose:** 126 mg/dL or higher indicates diabetes  
    **BMI:** 30 or higher indicates obesity  
    **Age:** Risk increases significantly after 45 years
    """)

    st.markdown("---")

    if model_loaded:
        st.markdown("""
        <div class="sidebar-info">
            <h4>Model Performance</h4>
            <p><strong>Accuracy:</strong> 89.80%</p>
            <p><strong>Recall:</strong> 91.18%</p>
            <p><strong>AUC-ROC:</strong> 97.30%</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PREPROCESSING FUNCTION
# FIX: Moved below load_models() so feature_columns is defined before use,
# and accepts feature_columns as a parameter to avoid relying on global state.
# ============================================

def preprocess_patient(df_patient, gender_mapping, smoking_mapping, feature_columns):
    """Apply preprocessing as done during model training"""
    df_processed = df_patient.copy()

    # Encode gender
    df_processed['gender'] = df_processed['gender'].map(gender_mapping)

    # Encode smoking history
    df_processed['smoking_history'] = df_processed['smoking_history'].map(smoking_mapping)

    # Create engineered features
    def bmi_category(bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        else:
            return 3

    def age_group(age):
        if age < 30:
            return 0
        elif age < 45:
            return 1
        elif age < 60:
            return 2
        else:
            return 3

    df_processed['bmi_category'] = df_processed['bmi'].apply(bmi_category)
    df_processed['age_group'] = df_processed['age'].apply(age_group)
    df_processed['glucose_hba1c_ratio'] = df_processed['blood_glucose_level'] / (df_processed['HbA1c_level'] + 0.1)
    df_processed['age_bmi_interaction'] = (df_processed['age'] * df_processed['bmi']) / 100
    df_processed['risk_score'] = (
        (df_processed['HbA1c_level'] >= 6.5).astype(int) * 3 +
        (df_processed['blood_glucose_level'] >= 140).astype(int) * 2 +
        (df_processed['bmi'] >= 30).astype(int) * 1 +
        (df_processed['hypertension'] == 1).astype(int) * 1 +
        (df_processed['heart_disease'] == 1).astype(int) * 1
    )

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed[feature_columns]

# ============================================
# MAIN PREDICTION FORM
# ============================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3>Patient Health Profile</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Personal Information")

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male", "Other"],
        help="Select the patient's gender"
    )

    age = st.slider(
        "Age",
        min_value=18, max_value=100, value=45, step=1,
        help="Patient's age in years"
    )

    smoking_history = st.selectbox(
        "Smoking History",
        options=["never", "former", "current", "ever", "not current", "No Info"],
        help="Patient's smoking history"
    )

    st.markdown("#### Medical History")

    hypertension = st.radio(
        "History of Hypertension",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Has the patient been diagnosed with high blood pressure?"
    )

    heart_disease = st.radio(
        "History of Heart Disease",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Has the patient been diagnosed with any heart condition?"
    )

with col2:
    st.markdown("#### Clinical Measurements")

    bmi = st.slider(
        "Body Mass Index (BMI)",
        min_value=15.0, max_value=45.0, value=25.0, step=0.5,
        help="Weight in kilograms divided by height in meters squared"
    )

    # BMI interpretation
    if bmi < 18.5:
        st.caption("Underweight range")
    elif bmi < 25:
        st.caption("Normal weight range")
    elif bmi < 30:
        st.caption("Overweight range")
    else:
        st.caption("Obese range")

    hba1c = st.slider(
        "HbA1c Level",
        min_value=4.0, max_value=9.0, value=5.5, step=0.1,
        help="Average blood sugar level over the past 2 to 3 months"
    )

    # HbA1c interpretation
    if hba1c < 5.7:
        st.caption("Normal range")
    elif hba1c < 6.5:
        st.caption("Prediabetes range")
    else:
        st.caption("Diabetes range")

    blood_glucose = st.slider(
        "Fasting Blood Glucose",
        min_value=70, max_value=300, value=100, step=5,
        help="Blood sugar level after at least 8 hours of fasting"
    )

    # Blood glucose interpretation
    if blood_glucose < 100:
        st.caption("Normal range")
    elif blood_glucose < 126:
        st.caption("Prediabetes range")
    else:
        st.caption("Diabetes range")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION BUTTON
# ============================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Analyze Diabetes Risk", type="primary", use_container_width=True)

# ============================================
# PREDICTION RESULTS
# ============================================

if predict_button and model_loaded:
    # Prepare input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [blood_glucose]
    })

    # FIX: Pass feature_columns explicitly as a parameter
    input_processed = preprocess_patient(input_data, gender_mapping, smoking_mapping, feature_columns)

    # Scale numerical features
    input_processed[numerical_features] = scaler.transform(input_processed[numerical_features])

    # Make prediction
    prediction = model.predict(input_processed)[0]
    probabilities = model.predict_proba(input_processed)[0]

    risk_level = "High" if prediction == 1 else "Low"
    risk_probability = probabilities[1] if prediction == 1 else probabilities[0]

    # Display result immediately
    if risk_level == "High":
        st.markdown(f"""
        <div class="result-container result-high">
            <h2>High Diabetes Risk Detected</h2>
            <p>The analysis indicates a potential risk of diabetes with {risk_probability*100:.1f}% confidence.</p>
            <p>Medical consultation is strongly recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-container result-low">
            <h2>Low Diabetes Risk Detected</h2>
            <p>The analysis indicates low risk of diabetes with {risk_probability*100:.1f}% confidence.</p>
            <p>Continue maintaining a healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Probability gauge chart
    with st.spinner("Generating analysis chart..."):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_probability * 100,
            title={'text': "Diabetes Risk Probability", 'font': {'size': 16}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#e74c3c" if risk_level == "High" else "#2ecc71"},
                'steps': [
                    {'range': [0, 30], 'color': "#d5f5e3"},
                    {'range': [30, 70], 'color': "#fef9e7"},
                    {'range': [70, 100], 'color': "#fadbd8"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_probability * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Risk factors analysis
    st.markdown('<div class="risk-box">', unsafe_allow_html=True)
    st.markdown('<h4>Identified Risk Factors</h4>', unsafe_allow_html=True)

    risk_factors = []
    if age > 45:
        risk_factors.append("Age over 45 years")
    if bmi >= 30:
        risk_factors.append("Obese range - BMI 30 or above")
    elif bmi >= 25:
        risk_factors.append("Overweight range - BMI 25 to 30")
    if hba1c >= 6.5:
        risk_factors.append("HbA1c in diabetic range - 6.5 percent or above")
    elif hba1c >= 5.7:
        risk_factors.append("HbA1c in prediabetic range - 5.7 to 6.4 percent")
    if blood_glucose >= 126:
        risk_factors.append("Fasting glucose in diabetic range - 126 mg/dL or above")
    elif blood_glucose >= 100:
        risk_factors.append("Fasting glucose in prediabetic range - 100 to 125 mg/dL")
    if hypertension == 1:
        risk_factors.append("History of hypertension")
    if heart_disease == 1:
        risk_factors.append("History of heart disease")
    if smoking_history in ['current', 'former']:
        risk_factors.append("History of smoking")

    if risk_factors:
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    else:
        st.markdown("No significant risk factors identified")

    st.markdown('</div>', unsafe_allow_html=True)

    # Personalized recommendations
    st.markdown('<div class="rec-box">', unsafe_allow_html=True)
    st.markdown('<h4>Personalized Recommendations</h4>', unsafe_allow_html=True)

    if risk_level == "High":
        st.markdown("""
        **Immediate Steps to Consider:**
        - Schedule an appointment with a healthcare provider
        - Get a confirmatory fasting blood glucose test
        - Discuss medication options with your doctor
        
        **Lifestyle Modifications:**
        - Adopt a balanced diet with reduced sugar intake
        - Engage in regular physical activity for 30 minutes daily
        - Monitor blood sugar levels regularly
        - Work toward maintaining a healthy weight
        """)
    else:
        st.markdown("""
        **Preventive Health Measures:**
        - Maintain a balanced diet rich in vegetables and whole grains
        - Exercise regularly for at least 150 minutes per week
        - Schedule annual blood sugar screening
        - Maintain a healthy body weight
        - Avoid tobacco use and limit alcohol consumption
        """)

    st.markdown('</div>', unsafe_allow_html=True)

elif predict_button and not model_loaded:
    st.error("The prediction model could not be loaded. Please check the models folder and restart the application.")

# ============================================
# EDUCATIONAL FOOTER
# ============================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>Diabetes Risk Assessment Tool | Developed for Early Detection and Prevention</p>
    <p>This tool is for screening purposes only. Always consult a healthcare provider for medical decisions.</p>
</div>
""", unsafe_allow_html=True)