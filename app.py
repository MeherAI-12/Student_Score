import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------- Load Model & Data ----------

@st.cache_resource
def load_model():
    # Make sure this file is in the same folder as app.py
    with open("Student_score (1).pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    # Make sure this file is in the same folder as app.py
    df = pd.read_csv("student_scores (1).csv")
    return df

model = load_model()

# ---------- App UI ----------

st.title("📘 Student Score Prediction App")
st.write(
    """
Predict a student's exam **Score** based on:
- Hours Studied  
- Attendance (%)  
- Assignments Submitted  
"""
)

# Show dataset (optional)
with st.expander("📊 View Training Dataset"):
    df = load_data()
    st.dataframe(df)
    st.write("Summary statistics:")
    st.write(df.describe())

# ---------- Sidebar Inputs ----------

st.sidebar.header("Enter Student Details")

hours_studied = st.sidebar.number_input(
    "Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)

attendance = st.sidebar.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=85,
    step=1,
)

assignments_submitted = st.sidebar.number_input(
    "Assignments Submitted",
    min_value=0,
    max_value=20,
    value=8,
    step=1,
)

# ---------- Prediction ----------

if st.button("🔮 Predict Score"):
    # Create input in the same format as training data
    input_df = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Assignments_Submitted": [assignments_submitted]
    })

    try:
        prediction = model.predict(input_df)[0]
    except Exception:
        # Fallback: use numpy array if model expects only array
        features = np.array([[hours_studied, attendance, assignments_submitted]])
        prediction = model.predict(features)[0]

    st.subheader("Predicted Score")
    st.success(f"🎯 Estimated Exam Score: **{prediction:.2f}**")

st.caption("Model trained on student_scores dataset (Hours_Studied, Attendance, Assignments_Submitted → Score).")
  
