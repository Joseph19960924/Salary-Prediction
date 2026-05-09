import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from fpdf import FPDF

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------
try:
    with open("salary_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("salary_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
DATA_PATH = "job_salary_prediction_dataset.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.warning("Dataset not found. Using sample data.")
    df = pd.DataFrame({
        "experience_years": np.random.randint(1, 20, 100),
        "salary": np.random.randint(30000, 150000, 100)
    })

# ---------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------
df.columns = df.columns.str.lower().str.replace(" ", "_")

# ---------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------
exp_col = "experience_years"
salary_col = "salary"

# ---------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------
if exp_col not in df.columns:
    st.error(f"Missing column: {exp_col}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

if salary_col not in df.columns:
    st.error(f"Missing column: {salary_col}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# ---------------------------------------------------
# CONVERT TO NUMERIC
# ---------------------------------------------------
df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")

df = df.dropna(subset=[exp_col, salary_col])

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("Salary Prediction Dashboard")

st.markdown("""
Machine learning dashboard for salary prediction,
analytics, visualization, and PDF reporting.
""")

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Input Features")

experience = st.sidebar.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=2
)

skills = st.sidebar.number_input(
    "Number of Skills",
    min_value=1,
    max_value=50,
    value=5
)

certifications = st.sidebar.number_input(
    "Certifications",
    min_value=0,
    max_value=20,
    value=1
)

education = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Bachelor", "Master", "PhD"]
)

industry = st.sidebar.selectbox(
    "Industry",
    ["IT", "Finance", "Healthcare", "Education", "Retail"]
)

company_size = st.sidebar.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"]
)

remote = st.sidebar.selectbox(
    "Remote Work",
    ["Yes", "No"]
)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric("Experience", f"{experience} yrs")
m2.metric("Skills", skills)
m3.metric("Certifications", certifications)
m4.metric("Remote Work", remote)

st.divider()

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
if st.button("Predict Salary"):

    try:

        # INPUT DATA
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

        # ENCODE
        input_data = pd.get_dummies(input_data)

        # ALIGN COLUMNS
        input_data = input_data.reindex(
            columns=model_columns,
            fill_value=0
        )

        # PREDICT
        prediction = model.predict(input_data)[0]

        # ---------------------------------------------------
        # RESULT
        # ---------------------------------------------------
        st.subheader("Prediction Result")

        st.success(
            f"Estimated Salary: R {prediction:,.0f} per year"
        )

        # ---------------------------------------------------
        # SALARY CATEGORY
        # ---------------------------------------------------
        if prediction < 40000:
            st.info("Entry-Level Salary Range")

        elif prediction < 90000:
            st.warning("Mid-Level Salary Range")

        else:
            st.success("Senior-Level Salary Range")

        # ---------------------------------------------------
        # CREATE GRAPH FOR PDF
        # ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(
            df[salary_col],
            kde=True,
            ax=ax
        )

        ax.set_title("Salary Distribution")

        graph_path = "salary_distribution.png"

        plt.savefig(
            graph_path,
            bbox_inches="tight"
        )

        plt.close()

        # ---------------------------------------------------
        # CREATE PDF
        # ---------------------------------------------------
        def create_pdf():

            pdf = FPDF()

            pdf.add_page()

            # TITLE
            pdf.set_font("Arial", "B", 16)

            pdf.cell(
                200,
                10,
                txt="Salary Prediction Report",
                ln=True,
                align="C"
            )

            pdf.ln(10)

            # BODY
            pdf.set_font("Arial", size=12)

            report_text = f"""
Experience: {experience} years
Skills: {skills}
Certifications: {certifications}
Education Level: {education}
Industry: {industry}
Company Size: {company_size}
Remote Work: {remote}

Predicted Salary:
R {prediction:,.0f} per year

Insights:
- Salary generally increases with experience
- Skills improve earning potential
- Certifications positively impact salary
- Industry and company size influence compensation
- Education level contributes to career growth
"""

            pdf.multi_cell(
                0,
                8,
                txt=report_text
            )

            pdf.ln(5)

            # GRAPH TITLE
            pdf.set_font("Arial", "B", 12)

            pdf.cell(
                0,
                10,
                txt="Salary Distribution Graph",
                ln=True
            )

            # ADD IMAGE
            pdf.image(
                graph_path,
                x=10,
                w=180
            )

            return pdf.output(
                dest="S"
            ).encode("latin-1")

        pdf_data = create_pdf()

        # ---------------------------------------------------
        # DOWNLOAD PDF
        # ---------------------------------------------------
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="salary_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------
st.subheader("Data Insights Dashboard")

col1, col2 = st.columns(2)

# ---------------------------------------------------
# SALARY DISTRIBUTION
# ---------------------------------------------------
with col1:

    st.markdown("### Salary Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        df[salary_col],
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Salary")

    st.pyplot(fig)

# ---------------------------------------------------
# EXPERIENCE VS SALARY
# ---------------------------------------------------
with col2:

    st.markdown("### Experience vs Salary Trend")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.scatterplot(
        x=df[exp_col],
        y=df[salary_col],
        ax=ax
    )

    sns.regplot(
        x=df[exp_col],
        y=df[salary_col],
        scatter=False,
        ax=ax
    )

    ax.set_xlabel("Experience Years")
    ax.set_ylabel("Salary")

    st.pyplot(fig)

st.divider()

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
st.subheader("Model Performance Dashboard")

try:

    # USE NUMERIC FEATURES
    X = df.select_dtypes(include=[np.number]).copy()

    # REMOVE TARGET
    if salary_col in X.columns:
        X = X.drop(columns=[salary_col])

    # ALIGN MODEL FEATURES
    X = X.reindex(
        columns=model_columns,
        fill_value=0
    )

    y = df[salary_col]

    # PREDICT
    y_pred = model.predict(X)

    # SCORES
    r2 = r2_score(y, y_pred)

    mse = mean_squared_error(y, y_pred)

    p1, p2 = st.columns(2)

    p1.metric(
        "R2 Score",
        f"{r2:.2f}"
    )

    p2.metric(
        "Mean Squared Error",
        f"{mse:,.0f}"
    )

except Exception as e:

    st.warning(
        f"Model evaluation skipped: {e}"
    )

# ---------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------
st.subheader("Business Insights")

st.write("""
- Employees with more experience generally earn higher salaries
- Certifications and skills improve earning potential
- Industry type significantly affects salary levels
- Larger companies may offer higher compensation
- Education level contributes to salary growth
""")