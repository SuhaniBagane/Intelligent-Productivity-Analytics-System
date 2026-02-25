import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("productivity_model.pkl")

st.title("📊 Productivity Prediction System")

st.write("Enter your study details to predict task performance")

# Inputs
study_hours = st.slider("Study Hours", 0, 12, 5)
prev_day = st.slider("Previous Day Study Hours", 0, 12, 4)
rolling_avg = st.slider("Rolling Average (last 3 days)", 0.0, 12.0, 4.5)

# Predict
if st.button("Predict"):
    data = pd.DataFrame([[study_hours, prev_day, rolling_avg]],
                        columns=["Study_Hours", "Prev_Day_Hours", "Rolling_Avg"])

    result = model.predict(data)

    st.success(f"📈 Predicted Tasks Completed: {round(result[0],2)}")