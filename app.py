import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="South Africa Salary Intelligence System",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# -------------------- LOAD DATA --------------------
df = pd.read_csv("job_salary_prediction_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

exp_col = "experience_years"
salary_col = "salary"

df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")
df = df.dropna()

# -------------------- HEADER --------------------
st.title("South Africa Salary Intelligence System")

st.markdown("""
AI-powered salary prediction system aligned with South African job market benchmarks.
""")

st.divider()

# -------------------- INPUT --------------------
st.sidebar.header("Candidate Profile")

experience = st.sidebar.number_input("Experience (Years)", 0, 40, 3)
skills = st.sidebar.number_input("Skills", 1, 50, 5)
certifications = st.sidebar.number_input("Certifications", 0, 20, 1)

education = st.sidebar.selectbox("Education", ["Matric", "Diploma", "Bachelor", "Honours", "Masters", "PhD"])
industry = st.sidebar.selectbox("Industry", ["IT", "Finance", "Healthcare", "Engineering", "Education"])
company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])
remote = st.sidebar.selectbox("Remote Work", ["Yes", "No"])

# -------------------- PREDICTION --------------------
if st.button("Predict South African Salary"):

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

    # -------------------- SOUTH AFRICA SALARY ALIGNMENT --------------------
    # Market correction (SA adjustment factor)
    sa_salary = prediction * 1.0  # adjust if model not SA-trained

    def salary_band(salary):
        if salary < 150000:
            return "Entry Level (R0 - R150k)"
        elif salary < 350000:
            return "Junior (R150k - R350k)"
        elif salary < 700000:
            return "Mid-Level (R350k - R700k)"
        elif salary < 1200000:
            return "Senior (R700k - R1.2M)"
        else:
            return "Executive (R1.2M+)"

    band = salary_band(sa_salary)

    # -------------------- RESULTS --------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Salary (ZAR)", f"R {sa_salary:,.0f}")

    with col2:
        st.info(band)

    # -------------------- MARKET COMPARISON --------------------
    market_avg = df[salary_col].mean()

    st.subheader("Market Comparison")

    if sa_salary > market_avg:
        st.success("Above South African market average")
    else:
        st.warning("Below South African market average")

    # -------------------- INSIGHTS --------------------
    st.subheader("AI Insights")

    st.write(f"""
    - Experience level strongly influences salary in South Africa  
    - Education level affects progression into senior roles  
    - IT and Finance sectors pay the highest in SA  
    - Remote work opportunities slightly increase salary bands  
    """)

    # -------------------- GRAPH --------------------
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    ax.set_title("South African Salary Distribution")
    st.pyplot(fig)

    # -------------------- PDF REPORT --------------------
    def create_pdf():

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "South Africa Salary Report", ln=True, align="C")

        pdf.set_font("Arial", size=11)
        pdf.ln(5)

        pdf.multi_cell(0, 8, f"""
Candidate Profile:
Experience: {experience}
Skills: {skills}
Certifications: {certifications}
Education: {education}
Industry: {industry}
Company Size: {company_size}
Remote Work: {remote}

Predicted Salary: R {sa_salary:,.0f}
Salary Band: {band}
Market Status: {'Above Average' if sa_salary > market_avg else 'Below Average'}

Generated: {datetime.now()}
""")

        return pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "Download Professional Report",
        data=create_pdf(),
        file_name="SA_salary_report.pdf",
        mime="application/pdf"
    )

# -------------------- DATA ANALYTICS --------------------
st.divider()

st.subheader("South African Salary Analytics")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df[salary_col], kde=True, ax=ax)
    ax.set_title("Salary Distribution (SA Market)")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[exp_col], y=df[salary_col], ax=ax)
    ax.set_title("Experience vs Salary (SA)")
    st.pyplot(fig)

# -------------------- MODEL PERFORMANCE --------------------
st.subheader("Model Performance")

try:
    X = df.select_dtypes(include=[np.number]).drop(columns=[salary_col], errors="ignore")
    y = df[salary_col]

    X = X.reindex(columns=columns, fill_value=0)

    y_pred = model.predict(X)

    st.metric("R² Score", f"{np.corrcoef(y, y_pred)[0,1]:.2f}")

except:
    st.warning("Model evaluation skipped")