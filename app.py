import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Salary Intelligence System",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("salary_model.pkl")
columns = joblib.load("salary_columns.pkl")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("job_salary_prediction_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

exp_col = "experience_years"
salary_col = "salary"

df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")
df = df.dropna()

# =========================
# TITLE
# =========================
st.title("Salary Intelligence System (South Africa)")
st.write("Machine learning system for salary prediction and market analysis")

st.divider()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input Profile")

experience = st.sidebar.number_input("Experience Years", 0, 40, 3)
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

# =========================
# PREDICTION
# =========================
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

    # =========================
    # SA SALARY BANDS
    # =========================
    def get_band(s):
        if s < 150000:
            return "Entry Level"
        elif s < 350000:
            return "Junior"
        elif s < 700000:
            return "Mid-Level"
        elif s < 1200000:
            return "Senior"
        else:
            return "Executive"

    band = get_band(prediction)

    # =========================
    # OUTPUT
    # =========================
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Predicted Salary (ZAR)", f"R {prediction:,.0f}")

    with c2:
        st.info(band)

    # =========================
    # MARKET COMPARISON
    # =========================
    market_avg = df[salary_col].mean()

    if prediction > market_avg:
        st.success("Above South African market average")
    else:
        st.warning("Below South African market average")

    # =========================
    # INSIGHTS
    # =========================
    st.subheader("Insights")

    st.write(
        "Experience and industry are the strongest salary drivers in South Africa."
    )

    # =========================
    # CHART
    # =========================
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    ax.set_title("Salary Distribution (South Africa)")
    st.pyplot(fig)

    # =========================
    # PDF REPORT (PROFESSIONAL)
    # =========================
    def clean(text):
        return str(text).encode("ascii", "ignore").decode()

    def create_pdf():

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Salary Intelligence Report", ln=True, align="C")

        pdf.ln(5)

        pdf.set_font("Arial", size=11)

        report = f"""
Generated: {datetime.now()}

PROFILE
Experience: {experience} years
Skills: {skills}
Certifications: {certifications}
Education: {education}
Industry: {industry}
Company Size: {company_size}
Remote Work: {remote}

PREDICTION
Salary: R {prediction:,.0f}
Band: {band}

MARKET
Average Salary: R {market_avg:,.0f}
Status: {"Above Market" if prediction > market_avg else "Below Market"}
"""

        pdf.multi_cell(0, 8, clean(report))

        # Save chart
        chart_path = "chart.png"
        plt.figure()
        sns.histplot(df[salary_col], kde=True)
        plt.savefig(chart_path)
        plt.close()

        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Salary Distribution Chart", ln=True)

        pdf.image(chart_path, w=180)

        return pdf.output(dest="S").encode("latin-1", "ignore")

    st.download_button(
        "Download Professional Report",
        data=create_pdf(),
        file_name="salary_report.pdf",
        mime="application/pdf"
    )

# =========================
# DATA VISUALS
# =========================
st.divider()

st.subheader("Market Analytics")

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

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("Model Performance")

try:
    X = df.select_dtypes(include=[np.number]).drop(columns=[salary_col])
    X = X.reindex(columns=columns, fill_value=0)

    y = df[salary_col]
    y_pred = model.predict(X)

    st.metric("Correlation Score", f"{np.corrcoef(y, y_pred)[0,1]:.2f}")

except Exception as e:
    st.warning("Model evaluation skipped")