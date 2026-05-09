import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("salary_model.pkl")
model_columns = joblib.load("salary_columns.pkl")

# Page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Title
st.title("Salary Prediction")
st.write("Estimate a candidate's expected salary based on experience and background.")

st.divider()

# --- INPUT SECTION ---
st.subheader("Enter Candidate Details")

experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=2)
skills = st.number_input("Number of Skills", min_value=1, max_value=50, value=5)
certifications = st.number_input("Certifications", min_value=0, max_value=20, value=1)

education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
industry = st.selectbox("Industry", ["IT", "Finance", "Healthcare", "Education", "Retail"])
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
remote = st.selectbox("Remote Work", ["Yes", "No"])

st.divider()

# --- PREDICTION ---
if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        "experience_years": experience,
        "skills_count": skills,
        "certifications": certifications,
        "job_title_freq": 1,
        "education_level": education,
        "industry": industry,
        "company_size": company_size,
        "remote_work": remote
    }])

    # Encode
    input_data = pd.get_dummies(input_data)

    # Align columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]

    # Output (clean & human)
    st.subheader("Estimated Salary")
    st.write(f"${prediction:,.0f} per year")

    # Simple interpretation
    if prediction < 40000:
        st.caption("This is considered an entry-level salary range.")
    elif prediction < 90000:
        st.caption("This falls within a mid-level salary range.")
    else:
        st.caption("This is a high salary range, typically for experienced professionals.")