import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from fpdf import FPDF

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = joblib.load("salary_model.pkl")
model_columns = joblib.load("salary_columns.pkl")

# ------------------ LOAD DATA ------------------
DATA_PATH = "job_salary_prediction_dataset.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame({
        "experience_years": np.random.randint(1, 20, 100),
        "salary": np.random.randint(30000, 150000, 100)
    })

# ------------------ CLEAN COLUMN NAMES (CRITICAL FIX) ------------------
df.columns = df.columns.str.lower().str.replace(" ", "_")

# FORCE STANDARD COLUMNS
exp_col = "experience_years"
salary_col = "salary"

# ------------------ HEADER ------------------
st.title("Salary Prediction Dashboard")
st.markdown("Machine learning system with prediction, analytics, and reporting.")
st.divider()

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("Input Features")

experience = st.sidebar.number_input("Years of Experience", 0, 40, 2)
skills = st.sidebar.number_input("Number of Skills", 1, 50, 5)
certifications = st.sidebar.number_input("Certifications", 0, 20, 1)

education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
industry = st.sidebar.selectbox("Industry", ["IT", "Finance", "Healthcare", "Education", "Retail"])
company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])
remote = st.sidebar.selectbox("Remote Work", ["Yes", "No"])

st.divider()

# ------------------ PREDICTION ------------------
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

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Estimated Salary: R {prediction:,.0f} per year")

    # ------------------ INSIGHTS ------------------
    insights_text = f"""
Salary Prediction Report

Experience: {experience} years
Skills: {skills}
Certifications: {certifications}
Education: {education}
Industry: {industry}
Company Size: {company_size}
Remote Work: {remote}

Predicted Salary: R {prediction:,.0f}

Key Insights:
- Salary increases with experience
- Skills and certifications improve earning potential
- Industry strongly affects salary
- Education level influences career growth
"""

    # ------------------ GRAPH FOR PDF ------------------
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    ax.set_title("Salary Distribution")
    graph_path = "salary_graph.png"
    plt.savefig(graph_path)
    plt.close()

    # ------------------ PDF GENERATION ------------------
    def create_pdf(text, image_path):

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, text)

        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Salary Distribution Graph", ln=True)

        pdf.image(image_path, x=10, w=180)

        return pdf.output(dest="S").encode("latin-1")

    pdf_data = create_pdf(insights_text, graph_path)

    st.download_button(
        "Download PDF Report",
        data=pdf_data,
        file_name="salary_report.pdf",
        mime="application/pdf"
    )

st.divider()

# ------------------ GRAPHS ------------------
st.subheader("Data Insights Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("Experience vs Salary Trend")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[exp_col], y=df[salary_col], ax=ax)
    sns.regplot(x=df[exp_col], y=df[salary_col], ax=ax, scatter=False)
    st.pyplot(fig)

st.divider()

# ------------------ MODEL EVALUATION ------------------
st.subheader("Model Performance Dashboard")

try:
    X = df[[exp_col]]
    y = df[salary_col]

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    c1, c2 = st.columns(2)
    c1.metric("R2 Score", f"{r2:.2f}")
    c2.metric("MSE", f"{mse:,.0f}")

except:
    st.warning("Model evaluation skipped due to mismatch.")
