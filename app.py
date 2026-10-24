import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="University Student Stress Prediction", page_icon="🎓", layout="centered")
MODEL_PATH = Path(__file__).parent / "university_student_stress_model.pkl"
ENCODER_PATH = Path(__file__).parent / "stress_label_encoder.pkl"

st.title("🎓 University Student Stress Prediction")
st.write("Enter the student's details to estimate the stress category predicted by the trained machine-learning model.")
st.info("This is an educational prediction tool, not a medical or mental-health diagnosis.")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    st.error("Model files are missing. Follow the README to run the notebook's model-saving cell, then place both .pkl files beside app.py.")
    st.stop()

try:
    model, encoder = load_artifacts()
except Exception as exc:
    st.error("The model could not be loaded. Check that the .pkl files were created by the same Python/scikit-learn environment and that both files are in this folder.")
    st.exception(exc)
    st.stop()

with st.form("stress_form"):
    st.subheader("Student details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 19, 24, 21)
        gender = st.selectbox("Gender", ["Female", "Male"])
        study_hours = st.slider("Study hours per day", 0, 9, 4)
        attendance = st.slider("Class attendance (%)", 40, 99, 69)
        tuition = st.selectbox("Tuition", ["No", "Yes"])
        exam_frequency = st.slider("Exam frequency", 1, 9, 5)
        assignment_load = st.slider("Assignment load", 1, 9, 5)
        sleep_hours = st.slider("Sleep hours per day", 4, 9, 7)
    with c2:
        exercise = st.selectbox("Physical exercise", ["No", "Yes"])
        social_media = st.slider("Social media use (hours/day)", 0, 7, 3)
        screen_time = st.slider("Screen time (hours/day)", 1, 11, 6)
        family_income = st.selectbox("Family income level", ["Low", "Medium", "High"])
        peer_pressure = st.slider("Peer pressure (1–9)", 1, 9, 5)
        family_support = st.slider("Family support (1–9)", 1, 9, 5)
        anxiety = st.slider("Anxiety level (1–9)", 1, 9, 5)
        university_type = st.selectbox("University type", ["National University", "Private University", "Public University"])
    submitted = st.form_submit_button("Predict Stress Level", type="primary", use_container_width=True)

if submitted:
    features = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Study_Hours": study_hours,
        "Class_Attendance": attendance,
        "Tuition": tuition,
        "Exam_Frequency": exam_frequency,
        "Assignment_Load": assignment_load,
        "Sleep_Hours": sleep_hours,
        "Physical_Exercise": exercise,
        "Social_Media_Use": social_media,
        "Screen_Time": screen_time,
        "Family_Income_Level": family_income,
        "Peer_Pressure": peer_pressure,
        "Family_Support": family_support,
        "Anxiety_Level": anxiety,
        "University_Type": university_type,
    }])
    try:
        prediction_encoded = model.predict(features)
        prediction = encoder.inverse_transform(prediction_encoded.astype(int))[0]
        st.subheader("Prediction result")
        if str(prediction).lower() == "low":
            st.success(f"Predicted stress level: {prediction}")
        elif str(prediction).lower() == "high":
            st.warning(f"Predicted stress level: {prediction}")
        else:
            st.info(f"Predicted stress level: {prediction}")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            st.write("Model class scores")
            st.bar_chart(pd.DataFrame({"Estimated score": probabilities}, index=encoder.classes_))
        st.caption("Predictions depend on the training dataset and may be wrong. This result should not be used as a diagnosis.")
    except Exception as exc:
        st.error("Prediction failed. Ensure the model was trained with the 16 input features listed in your notebook.")
        st.exception(exc)
