import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page title
st.title('💓 Heart Failure Prediction App')
st.write('Fill out the form below to check your risk of heart disease.')

# User input form
with st.form(key='heart_form'):
    age = st.number_input('Age', min_value=1, max_value=120, value=40)
    resting_bp = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', [0, 1])
    max_hr = st.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150)
    oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, step=0.1, value=1.0)

    sex = st.selectbox('Sex', ['Male', 'Female'])
    chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
    ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    exercise_angina = st.selectbox('Exercise Induced Angina?', ['Yes', 'No'])
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    submit = st.form_submit_button('Predict')

if submit:
    # One-hot encoding for categorical inputs
    sex_m = 1 if sex == 'Male' else 0
    cp_ata = 1 if chest_pain == 'ATA' else 0
    cp_nap = 1 if chest_pain == 'NAP' else 0
    cp_ta = 1 if chest_pain == 'TA' else 0
    ecg_normal = 1 if ecg == 'Normal' else 0
    ecg_st = 1 if ecg == 'ST' else 0
    angina_y = 1 if exercise_angina == 'Yes' else 0
    slope_flat = 1 if st_slope == 'Flat' else 0
    slope_up = 1 if st_slope == 'Up' else 0

    # Final input order (matches model training)
    input_data = [
        age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak,
        sex_m, cp_ata, cp_nap, cp_ta,
        ecg_normal, ecg_st, angina_y,
        slope_flat, slope_up
    ]

    # Column names (exact match from training set)
    columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]

    # Convert to DataFrame (fixes scaler warning)
    input_df = pd.DataFrame([input_data], columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 1:
        st.error('⚠️ High risk of heart disease!')
    else:
        st.success('✅ Low risk of heart disease.')

