import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SA Salary Intelligence System",
    layout="wide"
)

# -----------------------------
# LOAD MODEL (FIXED)
# -----------------------------
try:
    model = joblib.load("salary_model.pkl")
    columns = joblib.load("salary_columns.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("job_salary_prediction_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

exp_col = "experience_years"
salary_col = "salary"

df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")
df = df.dropna()

# -----------------------------
# TITLE
# -----------------------------
st.title("South African Salary Intelligence System")

st.markdown("AI-powered salary prediction and market analysis for South Africa.")

st.divider()

# -----------------------------
# INPUT
# -----------------------------
st.sidebar.header("Candidate Profile")

experience = st.sidebar.number_input("Experience (Years)", 0, 40, 3)
skills = st.sidebar.number_input("Skills Count", 1, 50, 5)
certifications = st.sidebar.number_input("Certifications", 0, 20, 1)

education = st.sidebar.selectbox(
    "Education",
    ["Matric", "Diploma", "Bachelor", "Honours", "Masters", "PhD"]
)

industry = st.sidebar.selectbox(
    "Industry",
    ["IT", "Finance", "Healthcare", "Engineering", "Education"]
)

company_size = st.sidebar.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"]
)

remote = st.sidebar.selectbox(
    "Remote Work",
    ["Yes", "No"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Salary"):

    input_df = pd.DataFrame([{
        "experience_years": experience,
        "skills_count": skills,
        "certifications": certifications,
        "education_level": education,
        "industry": industry,
        "company_size": company_size,
        "remote_work": remote
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    # -----------------------------
    # SOUTH AFRICAN SALARY BANDS
    # -----------------------------
    def band(s):
        if s < 150000:
            return "Entry Level (R0–150k)"
        elif s < 350000:
            return "Junior (R150k–350k)"
        elif s < 700000:
            return "Mid-Level (R350k–700k)"
        elif s < 1200000:
            return "Senior (R700k–1.2M)"
        else:
            return "Executive (R1.2M+)"

    salary_band = band(prediction)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Predicted Salary (ZAR)", f"R {prediction:,.0f}")

    with c2:
        st.info(salary_band)

    # -----------------------------
    # MARKET COMPARISON
    # -----------------------------
    market_avg = df[salary_col].mean()

    if prediction > market_avg:
        st.success("Above South African market average")
    else:
        st.warning("Below South African market average")

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    st.subheader("AI Insights")

    st.write("""
    - Experience is the strongest salary driver in South Africa  
    - IT and Finance pay the highest salaries  
    - Certifications increase salary potential  
    - Larger companies offer better compensation  
    """)

    # -----------------------------
    # GRAPH
    # -----------------------------
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    ax.set_title("South African Salary Distribution")
    st.pyplot(fig)

    # -----------------------------
    # PDF REPORT
    # -----------------------------
    def create_pdf():

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "SA Salary Intelligence Report", ln=True, align="C")

        pdf.ln(5)

        pdf.set_font("Arial", size=11)

        pdf.multi_cell(0, 8, f"""
Experience: {experience}
Skills: {skills}
Certifications: {certifications}
Education: {education}
Industry: {industry}
Company Size: {company_size}
Remote Work: {remote}

Predicted Salary: R {prediction:,.0f}
Salary Band: {salary_band}
Market Status: {"Above Average" if prediction > market_avg else "Below Average"}

Generated: {datetime.now()}
""")

        return pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "Download Report",
        data=create_pdf(),
        file_name="SA_salary_report.pdf",
        mime="application/pdf"
    )

# -----------------------------
# DATA VISUALIZATION
# -----------------------------
st.divider()

st.subheader("Market Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    ax.set_title("Salary Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[exp_col], y=df[salary_col], ax=ax)
    ax.set_title("Experience vs Salary")
    st.pyplot(fig)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("Model Performance")

try:
    X = df.select_dtypes(include=[np.number]).drop(columns=[salary_col])
    X = X.reindex(columns=columns, fill_value=0)

    y = df[salary_col]

    y_pred = model.predict(X)

    r2 = np.corrcoef(y, y_pred)[0, 1]

    st.metric("Model Correlation Score", f"{r2:.2f}")

except Exception as e:
    st.warning(f"Evaluation skipped: {e}")