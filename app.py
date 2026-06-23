import streamlit as st
import pandas as pd
import joblib
import json, os, datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ── Custom transformer (MUST match the class used when the pipeline ───
#    was saved in sprint4_pipeline.ipynb, or joblib.load() will fail) ──
class SalaryPreprocessor(BaseEstimator, TransformerMixin):
    """
    Replicates Sprint 1 preprocessing:
      1. Date feature extraction
      2. skills_count from required_skills
      3. Drop raw columns
      4. Ordinal encoding (experience_level, company_size, education_required)
      5. Frequency encoding (job_title, salary_currency, employment_type,
                             company_location, employee_residence, industry)
      6. Select & order final feature columns
    """
    def __init__(self, encoding_maps, feature_columns, best_features):
        self.encoding_maps = encoding_maps
        self.feature_columns = feature_columns
        self.best_features = best_features

    def fit(self, X, y=None):
        return self   # maps already learnt from Sprint 1 training data

    def transform(self, X):
        df = X.copy()

        # ── Date features ───────────────────────────────────────────
        for col in ['posting_date', 'application_deadline']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'posting_date' in df.columns:
            df['posting_month'] = df['posting_date'].dt.month
            df['posting_year'] = df['posting_date'].dt.year
            df['posting_day'] = df['posting_date'].dt.day
            df['day_of_week'] = df['posting_date'].dt.dayofweek

        if 'posting_date' in df.columns and 'application_deadline' in df.columns:
            df['application_duration'] = (
                df['application_deadline'] - df['posting_date']
            ).dt.days
            df['days_to_apply'] = df['application_duration']

        # ── skills_count ────────────────────────────────────────────
        if 'required_skills' in df.columns:
            df['skills_count'] = df['required_skills'].apply(
                lambda x: len(str(x).split(',')))

        # ── Drop raw / id columns ───────────────────────────────────
        drop_cols = ['job_id', 'posting_date', 'application_deadline',
                     'required_skills', 'company_name', 'application_duration']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # ── Ordinal encoding ────────────────────────────────────────
        maps = self.encoding_maps
        if 'experience_level' in df.columns:
            df['experience_level'] = df['experience_level'].map(maps['exp_map'])
        if 'company_size' in df.columns:
            df['company_size'] = df['company_size'].map(maps['size_map'])
        if 'education_required' in df.columns:
            df['education_required'] = df['education_required'].map(maps['edu_map'])

        # ── Frequency encoding ───────────────────────────────────────
        for col in maps['freq_cols']:
            if col in df.columns:
                freq_map = maps['freq_maps'].get(col, {})
                df[col] = df[col].map(freq_map).fillna(0)

        # ── Select columns in exact Sprint 1 order ───────────────────
        available = [c for c in self.feature_columns if c in df.columns]
        df = df[available]

        # ── Subset to best_features (Sprint 3 RFE selection) ─────────
        best_avail = [c for c in self.best_features if c in df.columns]
        if best_avail:
            df = df[best_avail]

        return df

# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# ── Load pipeline ─────────────────────────────────────────────────────
# Place salary_pipeline_sprint4.joblib in the SAME folder as this app.py
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "salary_pipeline_sprint4.joblib")

@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)

pipeline = load_pipeline()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center;color:#1a5276;'>💼 AI Job Salary Predictor</h1>
    <p style='text-align:center;color:gray;'>Sprint 4 · Innomatics Research Labs · AI Jobs Dataset</p>
    <hr>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Model not found!")
    st.info(f"Copy `salary_pipeline_sprint4.joblib` into the same folder as app.py:\n\n`{os.path.dirname(os.path.abspath(__file__))}`")
    st.stop()

# ── Inputs ────────────────────────────────────────────────────────────
st.subheader("📝 Enter Job Details")
col1, col2, col3 = st.columns(3)

with col1:
    job_title = st.selectbox("Job Title", [
        "Data Scientist", "Machine Learning Engineer", "Data Analyst",
        "AI Engineer", "Data Engineer", "Research Scientist", "MLOps Engineer"
    ])

    experience_level = st.selectbox(
        "Experience Level",
        ["EN", "MI", "SE", "EX"],
        format_func=lambda x: {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}[x]
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["FT", "PT", "CT", "FL"],
        format_func=lambda x: {"FT": "Full Time", "PT": "Part Time", "CT": "Contract", "FL": "Freelance"}[x]
    )

    years_experience = st.slider("Years of Experience", 0, 20, 3)

with col2:
    company_size = st.selectbox(
        "Company Size",
        ["S", "M", "L"],
        format_func=lambda x: {"S": "Small", "M": "Medium", "L": "Large"}[x]
    )

    education_required = st.selectbox(
        "Education Required",
        ["Associate", "Bachelor", "Master", "PhD"]
    )

    remote_ratio = st.selectbox(
        "Remote Ratio",
        [0, 50, 100],
        format_func=lambda x: {0: "On-site", 50: "Hybrid", 100: "Fully Remote"}[x]
    )

    benefits_score = st.slider("Benefits Score", 0, 100, 70)

with col3:
    company_location = st.selectbox(
        "Company Location",
        ["US", "GB", "CA", "DE", "IN", "AU", "FR", "SG"]
    )

    employee_residence = st.selectbox(
        "Employee Residence",
        ["US", "GB", "CA", "DE", "IN", "AU", "FR", "SG"]
    )

    industry = st.selectbox(
        "Industry",
        ["Technology", "Finance", "Healthcare", "Education", "Retail", "Manufacturing"]
    )

    required_skills = st.text_input(
        "Required Skills (comma-separated)",
        value="Python, SQL, Machine Learning"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────
if st.button("🔮  Predict Salary", use_container_width=True, type="primary"):

    today = pd.Timestamp.today()
    deadline = today + pd.Timedelta(days=30)

    input_df = pd.DataFrame([{
        "job_title"              : job_title,
        "experience_level"       : experience_level,
        "employment_type"        : employment_type,
        "company_size"           : company_size,
        "education_required"     : education_required,
        "remote_ratio"           : remote_ratio,
        "benefits_score"         : benefits_score,
        "years_experience"       : years_experience,
        "job_description_length" : 800,
        "posting_date"           : str(today.date()),
        "application_deadline"   : str(deadline.date()),
        "required_skills"        : required_skills,
        "company_location"       : company_location,
        "employee_residence"     : employee_residence,
        "salary_currency"        : "USD",
        "industry"               : industry,
        "company_name"           : "Company"
    }])

    prediction = pipeline.predict(input_df)[0]

    # ── Result display ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a5276,#2980b9);
                border-radius:15px; padding:28px; text-align:center;'>
        <h2 style='color:white; margin:0;'>💰 Predicted Annual Salary</h2>
        <h1 style='font-size:3.2rem; color:#f9e79f; margin:10px 0;'>${prediction:,.0f}</h1>
        <p style='color:#d6eaf8; font-size:1.1rem; margin:0;'>≈ ${prediction/12:,.0f} / month</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("💵 Annual",  f"${prediction:,.0f}")
    c2.metric("📅 Monthly", f"${prediction/12:,.0f}")
    c3.metric("📆 Weekly",  f"${prediction/52:,.0f}")

    # ── Log prediction ────────────────────────────────────────────────
    log_path = os.path.join(os.path.dirname(__file__), "prediction_logs.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp"    : str(datetime.datetime.now()),
            "job_title"    : job_title,
            "experience"   : experience_level,
            "years_exp"    : years_experience,
            "company_size" : company_size,
            "remote_ratio" : remote_ratio,
            "prediction"   : round(float(prediction), 2)
        }) + "\n")
    st.caption("✅ Prediction logged to prediction_logs.jsonl")

# ── Sidebar: Prediction History ───────────────────────────────────────
st.sidebar.title("📜 Prediction History")
log_path = os.path.join(os.path.dirname(__file__), "prediction_logs.jsonl")

if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()[-10:]   # last 10 predictions

    if lines:
        for line in reversed(lines):
            r = json.loads(line)
            st.sidebar.markdown(
                f"**{r['job_title']}** ({r['experience']} · {r['years_exp']}yr)  \n"
                f"💰 **${r['prediction']:,.0f}**  \n"
                f"<small style='color:gray'>{r['timestamp'][:16]}</small>",
                unsafe_allow_html=True
            )
            st.sidebar.divider()

        if st.sidebar.button("🗑️ Clear History"):
            os.remove(log_path)
            st.rerun()
    else:
        st.sidebar.info("No predictions yet.")
else:
    st.sidebar.info("No predictions yet.")

# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>"
    "Sprint 4 · Innomatics Research Labs</p>",
    unsafe_allow_html=True
)
