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
    page_title="Salary Intelligence System - South Africa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3C72;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2A5298;
    }
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# REALISTIC SOUTH AFRICAN SALARY CONSTANTS
# =========================
USD_TO_ZAR = 18.50

# Education multipliers (based on South African market)
EDUCATION_MULTIPLIERS = {
    "Matric": 0.70,      # 30% below baseline
    "Diploma": 0.85,     # 15% below baseline
    "Bachelor": 1.00,    # Baseline
    "Honours": 1.12,     # 12% above baseline
    "Masters": 1.25,     # 25% above baseline
    "PhD": 1.40          # 40% above baseline
}

# Career framework with education adjustments
CAREER_FRAMEWORK = {
    "Entry Level": {
        "years": (0, 2), 
        "min": 80000,
        "max": 180000,
        "typical": 130000,
        "color": "#FF6B6B"
    },
    "Junior": {
        "years": (3, 5), 
        "min": 180000,
        "max": 350000,
        "typical": 260000,
        "color": "#FFA07A"
    },
    "Mid-Level": {
        "years": (6, 9), 
        "min": 350000,
        "max": 600000,
        "typical": 480000,
        "color": "#4ECDC4"
    },
    "Senior": {
        "years": (10, 15), 
        "min": 600000,
        "max": 900000,
        "typical": 750000,
        "color": "#45B7D1"
    },
    "Executive": {
        "years": (16, 40), 
        "min": 900000,
        "max": 1800000,
        "typical": 1200000,
        "color": "#96CEB4"
    }
}

# Industry adjustment factors for South Africa
INDUSTRY_FACTORS = {
    "IT": 1.15,
    "Finance": 1.20,
    "Healthcare": 1.05,
    "Engineering": 1.10,
    "Education": 0.90
}

# Company size adjustments
COMPANY_SIZE_FACTORS = {
    "Small": 0.85,
    "Medium": 1.00,
    "Large": 1.15
}

# =========================
# LOAD RESOURCES
# =========================
@st.cache_resource
def load_model():
    return joblib.load("salary_model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("salary_columns.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("job_salary_prediction_dataset.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    exp_col = "experience_years"
    salary_col = "salary"
    df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
    df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")
    df = df.dropna()
    df[salary_col] = df[salary_col] * USD_TO_ZAR
    df.loc[df[exp_col] <= 15, salary_col] = df.loc[df[exp_col] <= 15, salary_col].clip(upper=1200000)
    return df

try:
    model = load_model()
    columns = load_columns()
    df = load_data()
    exp_col = "experience_years"
    salary_col = "salary"
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# =========================
# HELPER FUNCTIONS
# =========================
def get_career_level(years_exp):
    """Determine career level based ONLY on experience"""
    if years_exp <= 2:
        return "Entry Level"
    elif years_exp <= 5:
        return "Junior"
    elif years_exp <= 9:
        return "Mid-Level"
    elif years_exp <= 15:
        return "Senior"
    else:
        return "Executive"

def calculate_education_impact(education_level, base_salary):
    """Calculate salary adjustment based on education"""
    multiplier = EDUCATION_MULTIPLIERS.get(education_level, 1.00)
    return base_salary * multiplier

def get_education_premium(education_level):
    """Get the premium percentage for education level"""
    multiplier = EDUCATION_MULTIPLIERS.get(education_level, 1.00)
    return (multiplier - 1) * 100

def validate_salary_by_level(salary, level, experience, education):
    """Check if salary is realistic for career level and education"""
    level_data = CAREER_FRAMEWORK[level]
    edu_multiplier = EDUCATION_MULTIPLIERS.get(education, 1.00)
    
    # Adjust expected range based on education
    adjusted_min = level_data["min"] * edu_multiplier
    adjusted_max = level_data["max"] * edu_multiplier
    adjusted_typical = level_data["typical"] * edu_multiplier
    
    if salary < adjusted_min:
        return "below", f"Below range for {level} with {education} (R {adjusted_min:,.0f} - R {adjusted_max:,.0f})", adjusted_typical
    elif salary > adjusted_max:
        return "above", f"Above range for {level} with {education} - Excellent!", adjusted_typical
    else:
        return "within", f"Within range for {level} with {education}", adjusted_typical

def get_salary_percentile(salary, level, education):
    """Calculate percentile within career level and education"""
    edu_multiplier = EDUCATION_MULTIPLIERS.get(education, 1.00)
    # Adjust comparison salaries by education
    adjusted_salaries = df[df['career_level_temp'] == level][salary_col].values * edu_multiplier
    if len(adjusted_salaries) > 5:
        return (adjusted_salaries < salary).mean() * 100
    return 50

# Add career level to dataframe
df['career_level_temp'] = df[exp_col].apply(get_career_level)

# =========================
# UI HEADER
# =========================
st.markdown('<p class="main-header">Salary Intelligence System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Salary Predictions for the South African Market</p>', unsafe_allow_html=True)

col_info, col_rate = st.columns([3, 1])
with col_rate:
    st.metric("Exchange Rate", f"1 USD = {USD_TO_ZAR:.2f} ZAR")

st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### Your Profile")
    st.markdown("---")
    
    experience = st.slider("Years of Experience", 0, 40, 3, 
                           help="Total years of professional experience")
    skills = st.slider("Skills Count", 1, 50, 5,
                       help="Number of relevant technical skills")
    certifications = st.number_input("Certifications", 0, 20, 1,
                                     help="Professional certifications earned")
    
    st.markdown("---")
    st.markdown("### Qualifications")
    
    education = st.selectbox(
        "Highest Education Level",
        ["Matric", "Diploma", "Bachelor", "Honours", "Masters", "PhD"],
        help="Your highest completed qualification"
    )
    
    # Show education premium
    edu_premium = get_education_premium(education)
    if edu_premium > 0:
        st.info(f"📊 {education} typically earns {edu_premium:.0f}% above baseline")
    elif edu_premium < 0:
        st.warning(f"📊 {education} typically earns {abs(edu_premium):.0f}% below baseline")
    else:
        st.info(f"📊 {education} is the baseline qualification")
    
    st.markdown("---")
    st.markdown("### Work Context")
    
    industry = st.selectbox(
        "Industry",
        list(INDUSTRY_FACTORS.keys())
    )
    
    company_size = st.selectbox(
        "Company Size",
        ["Small", "Medium", "Large"]
    )
    
    remote = st.radio(
        "Remote Work",
        ["Yes", "No"],
        horizontal=True
    )
    
    st.markdown("---")
    st.caption("Powered by Machine Learning")
    st.caption(f"Trained on {len(df):,} South African salary records")

# =========================
# MAIN CONTENT
# =========================
if st.button("Predict Salary", type="primary", use_container_width=True):
    
    # Prepare input
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
    
    # Make prediction and convert to ZAR
    prediction_usd = model.predict(input_df)[0]
    prediction_zar = prediction_usd * USD_TO_ZAR
    
    # Apply adjustments
    industry_factor = INDUSTRY_FACTORS[industry]
    size_factor = COMPANY_SIZE_FACTORS[company_size]
    education_factor = EDUCATION_MULTIPLIERS[education]
    
    # Calculate final salary with all factors
    final_salary = prediction_zar * industry_factor * size_factor * education_factor
    
    # Get career level based on experience
    career_level = get_career_level(experience)
    level_data = CAREER_FRAMEWORK[career_level]
    
    # Enforce salary caps based on experience level and education
    edu_adjusted_max = level_data["max"] * education_factor
    edu_adjusted_min = level_data["min"] * education_factor
    
    if final_salary > edu_adjusted_max:
        final_salary = edu_adjusted_max * 0.95
    if final_salary < edu_adjusted_min and experience > 2:
        final_salary = edu_adjusted_min * 1.05
    
    # Validate salary
    validation_status, validation_message, adjusted_typical = validate_salary_by_level(
        final_salary, career_level, experience, education
    )
    
    # Calculate percentile
    percentile = get_salary_percentile(final_salary, career_level, education)
    
    # Calculate education premium in actual salary
    base_without_education = final_salary / education_factor
    education_value = final_salary - base_without_education
    
    # =========================
    # RESULTS DASHBOARD
    # =========================
    st.markdown("---")
    st.markdown("## Salary Intelligence Report")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Salary", f"R {final_salary:,.0f}")
        st.caption(f"R {final_salary/12:,.0f} per month")
    
    with col2:
        st.metric("Career Level", career_level)
        st.caption(f"{level_data['years'][0]}-{level_data['years'][1]} years")
    
    with col3:
        st.metric("Education Level", education)
        st.caption(f"{edu_premium:+.0f}% vs baseline")
    
    with col4:
        st.metric("Market Position", f"Top {100-percentile:.0f}%")
        st.caption(f"Higher than {percentile:.0f}% of peers")
    
    # Education value breakdown
    st.markdown("---")
    st.markdown("### Education Impact Analysis")
    
    col_edu1, col_edu2, col_edu3 = st.columns(3)
    
    with col_edu1:
        st.metric("Base Salary (without education)", f"R {base_without_education:,.0f}")
    
    with col_edu2:
        st.metric("Education Premium", f"R {education_value:,.0f}")
        st.caption(f"{edu_premium:+.0f}% increase")
    
    with col_edu3:
        st.metric("Value per Year of Study", f"R {education_value/4:,.0f}")
        st.caption("Based on 4-year degree equivalent")
    
    # Salary validation message
    if validation_status == "below":
        st.markdown(f"""
        <div class="warning-box">
        <strong>Salary Below Expected Range</strong><br>
        {validation_message}<br>
        Typical for {career_level} with {education}: R {adjusted_typical:,.0f}<br>
        Consider negotiating or pursuing further qualifications.
        </div>
        """, unsafe_allow_html=True)
    elif validation_status == "above":
        st.markdown(f"""
        <div class="success-box">
        <strong>Excellent Compensation</strong><br>
        {validation_message}<br>
        Your qualifications and experience are highly valued.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
        <strong>Competitive Salary</strong><br>
        {validation_message}<br>
        Your education level is appropriately compensated.
        </div>
        """, unsafe_allow_html=True)
    
    # =========================
    # EDUCATION COMPARISON CHART
    # =========================
    st.markdown("---")
    st.markdown("## Education Level Comparison")
    
    # Generate comparison data for all education levels
    edu_levels = ["Matric", "Diploma", "Bachelor", "Honours", "Masters", "PhD"]
    edu_salaries = []
    
    for edu in edu_levels:
        edu_factor = EDUCATION_MULTIPLIERS[edu]
        # Calculate what salary would be with same base but different education
        salary_with_edu = base_without_education * edu_factor
        edu_salaries.append(salary_with_edu)
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.bar(edu_levels, edu_salaries, color=['#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1', '#96CEB4', '#B5EAD7'])
    
    # Highlight current education
    current_idx = edu_levels.index(education)
    bars[current_idx].set_color('#FF4444')
    bars[current_idx].set_edgecolor('darkred')
    bars[current_idx].set_linewidth(2)
    
    ax.axhline(y=final_salary, color='red', linestyle='--', linewidth=2, label=f'Your Salary: R {final_salary:,.0f}')
    
    ax.set_title('Salary by Education Level (holding other factors constant)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Annual Salary (ZAR)')
    ax.legend()
    
    # Add value labels on bars
    for bar, salary in zip(bars, edu_salaries):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'R {salary:,.0f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption("This chart shows how different education levels impact salary, assuming all other factors remain constant.")
    
    # =========================
    # MARKET ANALYSIS
    # =========================
    st.markdown("---")
    st.markdown("## Market Analysis")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Salary distribution by career level and education
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Filter for same career level
        peer_data = df[df['career_level_temp'] == career_level]
        
        if len(peer_data) > 10:
            sns.histplot(peer_data[salary_col], kde=True, ax=ax, 
                        color=level_data["color"], alpha=0.5, label=f'All {career_level}')
        
        # Markers
        ax.axvline(final_salary, color='red', linestyle='--', linewidth=2, 
                  label=f'Your Salary: R {final_salary:,.0f}')
        ax.axvline(adjusted_typical, color='blue', linestyle=':', linewidth=1.5, 
                  label=f'Typical for {education}: R {adjusted_typical:,.0f}')
        
        ax.set_title(f'Salary Distribution - {career_level} Level', fontsize=12, fontweight='bold')
        ax.set_xlabel('Annual Salary (ZAR)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_viz2:
        # Education vs Salary by experience
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Group by education and show average salaries
        edu_avg = df.groupby('education_level')[salary_col].mean().sort_values()
        
        # Add current position
        current_edu_avg = edu_avg.get(education, final_salary)
        
        edu_avg.plot(kind='barh', ax=ax, color='lightblue', alpha=0.7)
        ax.axvline(x=final_salary, color='red', linestyle='--', linewidth=2, 
                  label=f'Your Salary: R {final_salary:,.0f}')
        
        ax.set_title('Average Salary by Education Level (Market Data)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Annual Salary (ZAR)')
        ax.set_ylabel('Education Level')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    # =========================
    # RETURN ON EDUCATION INVESTMENT
    # =========================
    st.markdown("---")
    st.markdown("## Return on Education Investment")
    
    # Calculate ROI for higher education
    if education in ["Bachelor", "Honours", "Masters", "PhD"]:
        years_to_degree = {
            "Bachelor": 3,
            "Honours": 4,
            "Masters": 5,
            "PhD": 8
        }
        
        years_studied = years_to_degree.get(education, 3)
        avg_degree_cost = years_studied * 50000  # R50k per year average
        
        # Calculate lifetime benefit
        working_years_remaining = max(0, 65 - (25 + experience))  # Assuming started at 25
        annual_benefit = education_value
        lifetime_benefit = annual_benefit * working_years_remaining
        roi = ((lifetime_benefit - avg_degree_cost) / avg_degree_cost) * 100
        
        col_roi1, col_roi2, col_roi3 = st.columns(3)
        
        with col_roi1:
            st.metric("Degree Cost Estimate", f"R {avg_degree_cost:,.0f}")
        with col_roi2:
            st.metric("Annual Benefit", f"R {education_value:,.0f}")
        with col_roi3:
            st.metric("Lifetime ROI", f"{roi:.0f}%")
        
        if roi > 200:
            st.success(f"Excellent return on investment! Your {education} is projected to earn {roi:.0f}% more over your career compared to not having it.")
        elif roi > 100:
            st.info(f"Good return on investment. Your {education} provides solid lifetime value.")
        else:
            st.warning(f"Consider additional certifications or specializations to maximize your education investment.")
    
    # =========================
    # PEER COMPARISON BY EDUCATION
    # =========================
    st.markdown("---")
    st.markdown("## Peer Comparison (Same Education Level)")
    
    peer_edu_data = df[df['education_level'] == education]
    
    if len(peer_edu_data) > 10:
        peer_avg = peer_edu_data[salary_col].mean()
        peer_median = peer_edu_data[salary_col].median()
        
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            diff_vs_avg = ((final_salary/peer_avg - 1)*100)
            st.metric(f"Average for {education}", f"R {peer_avg:,.0f}",
                     delta=f"{diff_vs_avg:+.0f}% vs you")
        
        with col_comp2:
            diff_vs_median = ((final_salary/peer_median - 1)*100)
            st.metric(f"Median for {education}", f"R {peer_median:,.0f}",
                     delta=f"{diff_vs_median:+.0f}% vs you")
        
        with col_comp3:
            same_edu_percentile = (peer_edu_data[salary_col] < final_salary).mean() * 100
            st.metric("Your Percentile", f"{same_edu_percentile:.0f}th",
                     delta=f"Top {100-same_edu_percentile:.0f}% among {education}")
    
    # =========================
    # RECOMMENDATIONS
    # =========================
    st.markdown("---")
    st.markdown("## Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("**Education & Career Development**")
        
        if education == "Matric":
            st.write("- Complete a diploma or degree to increase earnings by 30-40%")
            st.write("- Consider part-time or online studies while working")
            st.write("- Focus on vocational training and certifications")
            st.write("- Target R180k-R250k with Diploma within 2-3 years")
        
        elif education == "Diploma":
            st.write("- Upgrade to Bachelor's degree for 15-20% increase")
            st.write("- Specialized certifications can bridge the gap")
            st.write("- Consider BTech or advanced diploma")
            st.write("- Target R250k-R350k with degree within 2-3 years")
        
        elif education == "Bachelor":
            st.write("- Honours degree adds 12% to salary potential")
            st.write("- Professional certifications in your field")
            st.write("- Consider MBA or Masters for senior roles")
            st.write("- Target R450k-R600k with Honours in 2-3 years")
        
        elif education == "Honours":
            st.write("- Masters degree adds 13% to earning potential")
            st.write("- Professional registration (where applicable)")
            st.write("- Specialist certifications for your industry")
            st.write("- Target R600k-R800k with Masters in 3-4 years")
        
        elif education == "Masters":
            st.write("- PhD adds 15% for research/academic roles")
            st.write("- Executive education for leadership positions")
            st.write("- Industry thought leadership")
            st.write("- Target R900k-R1.2M with PhD in 3-5 years")
        
        else:  # PhD
            st.write("- Focus on industry application of research")
            st.write("- Consultancy and advisory roles")
            st.write("- Board and committee positions")
            st.write("- Leverage expertise for premium rates")
    
    with rec_col2:
        st.markdown("**Salary Optimization**")
        
        if final_salary < adjusted_typical:
            st.write("- Document qualifications and achievements")
            st.write("- Research market rates for your education level")
            st.write("- Highlight unique skills and specializations")
            st.write("- Consider companies that value education")
        
        elif final_salary > adjusted_typical:
            st.write("- Negotiate for professional development budget")
            st.write("- Request tuition reimbursement for further studies")
            st.write("- Ask for research or conference funding")
            st.write("- Leverage education for mentorship roles")
        
        else:
            st.write("- Request clear promotion criteria based on education")
            st.write("- Ask about education assistance programs")
            st.write("- Build business case for advanced degree ROI")
            st.write("- Network with peers at higher education levels")
    
    # =========================
    # PDF EXPORT
    # =========================
    st.markdown("---")
    
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 15, "Salary Intelligence Report - South Africa", ln=True, align="C")
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(10)
        
        # Profile Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Professional Profile", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, f"""
Experience: {experience} years
Career Level: {career_level}
Education: {education} ({edu_premium:+.0f}% vs baseline)
Skills: {skills} | Certifications: {certifications}
Industry: {industry}
Company Size: {company_size}
Remote Work: {remote}
        """)
        
        # Salary Analysis
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Salary Analysis", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, f"""
Predicted Salary: R {final_salary:,.0f} (R {final_salary/12:,.0f} per month)
Education Premium: R {education_value:,.0f} ({edu_premium:+.0f}%)
Typical Range for {career_level} with {education}: R {level_data['min'] * EDUCATION_MULTIPLIERS[education]:,.0f} - R {level_data['max'] * EDUCATION_MULTIPLIERS[education]:,.0f}
Market Percentile: Top {100-percentile:.0f}%
Status: {validation_message}
        """)
        
        return pdf.output(dest="S").encode("latin-1")
    
    st.download_button(
        "Download Full Report (PDF)",
        data=generate_pdf(),
        file_name=f"salary_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# =========================
# EDUCATION SALARY GUIDE
# =========================
with st.expander("Education & Salary Guide 2024", expanded=False):
    st.markdown("""
    ### Impact of Education on South African Salaries
    
    | Education Level | Premium vs Matric | Typical Entry Salary | Typical Senior Salary |
    |----------------|------------------|---------------------|----------------------|
    | Matric | Baseline (0%) | R60k - R90k | R120k - R180k |
    | Diploma | +15-20% | R80k - R120k | R180k - R300k |
    | Bachelor's Degree | +30-40% | R120k - R180k | R350k - R600k |
    | Honours Degree | +40-50% | R150k - R220k | R450k - R750k |
    | Master's Degree | +50-70% | R200k - R300k | R600k - R1M |
    | PhD | +70-100% | R300k - R450k | R800k - R1.5M |
    
    ### Return on Education Investment (20-year career)
    
    - **Bachelor's Degree**: R2M - R3M lifetime premium
    - **Honours Degree**: R3M - R4M lifetime premium  
    - **Master's Degree**: R4M - R6M lifetime premium
    - **PhD**: R5M - R8M lifetime premium (academic/research roles)
    
    ### Industry-Specific Education Value
    
    | Industry | Most Valued Qualification | Premium |
    |----------|-------------------------|---------|
    | Finance | CA(SA), CFA, Masters | +40-60% |
    | IT | Certifications + Degree | +30-50% |
    | Engineering | BEng, PrEng, Masters | +35-55% |
    | Healthcare | Specialization, Masters | +45-70% |
    | Education | Masters, PhD | +25-40% |
    
    *Note: Experience and skills can offset education differences over time.*
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Your data is private and not stored | Powered by Machine Learning | Built for South African professionals</p>
    <p style="font-size: 0.8rem;">Salary predictions consider experience, education, industry, and company size.</p>
</div>
""", unsafe_allow_html=True)