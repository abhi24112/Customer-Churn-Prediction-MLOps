import streamlit as st
import requests
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="ChurnGuard — Prediction System",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# PREMIUM CSS REDESIGN
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0B0F1A !important;
    color: #E2E8F0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1300px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0F1525 0%, #131B2E 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.12) !important;
    padding-top: 2rem;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #63B3ED !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stSidebar"] .stAlert {
    background: rgba(99,179,237,0.08) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
}

/* ── Hero Title ── */
.hero-wrapper {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1A56DB, #63B3ED);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 8px 32px rgba(26,86,219,0.4);
    flex-shrink: 0;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #F0F6FF !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 0.88rem;
    color: #64748B;
    margin-top: 0.25rem;
    font-weight: 300;
}

/* ── Section Headers ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #63B3ED;
    margin: 1.6rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.3) 0%, transparent 100%);
}

/* ── Form Card ── */
.form-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 0.5rem;
}

/* ── Inputs & Selects ── */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] textarea {
    background-color: #0D1423 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s;
}
div[data-baseweb="input"] input:focus,
div[data-baseweb="select"] > div:focus-within {
    border-color: #63B3ED !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.12) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #1A56DB !important;
    border: 2px solid #63B3ED !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #63B3ED !important;
    font-size: 0.78rem !important;
}

/* ── Labels ── */
label, [data-testid="stWidgetLabel"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 4px !important;
}

/* ── Submit Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1A56DB 0%, #2563EB 100%) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 10px !important;
    height: 3.2em !important;
    margin-top: 1.2rem !important;
    box-shadow: 0 4px 24px rgba(26,86,219,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(26,86,219,0.55) !important;
    background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
}

/* ── Prediction Results ── */
.result-card {
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.result-card.churn {
    background: linear-gradient(135deg, rgba(220,38,38,0.13) 0%, rgba(15,12,25,0.9) 100%);
    border: 1px solid rgba(239,68,68,0.28);
    box-shadow: 0 0 40px rgba(239,68,68,0.08);
}
.result-card.stay {
    background: linear-gradient(135deg, rgba(16,185,129,0.13) 0%, rgba(12,20,25,0.9) 100%);
    border: 1px solid rgba(16,185,129,0.28);
    box-shadow: 0 0 40px rgba(16,185,129,0.08);
}

/* Big percentage dial */
.result-pct-wrap {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
}
.result-pct {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
}
.result-pct.churn { color: #FC8181; }
.result-pct.stay  { color: #6EE7B7; }
.result-pct-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-pct-label.churn { color: rgba(252,129,129,0.7); }
.result-pct-label.stay  { color: rgba(110,231,183,0.7); }

/* Divider between pct and text */
.result-divider {
    width: 1px;
    height: 80px;
    flex-shrink: 0;
    align-self: center;
}
.result-divider.churn { background: rgba(239,68,68,0.25); }
.result-divider.stay  { background: rgba(16,185,129,0.25); }

/* Right side text */
.result-body { flex: 1; min-width: 200px; }
.result-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.badge-churn { background: rgba(239,68,68,0.18); color: #FCA5A5; }
.badge-stay  { background: rgba(16,185,129,0.18); color: #6EE7B7; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-bottom: 0.45rem;
    line-height: 1.2;
}
.result-title.churn { color: #FCA5A5; }
.result-title.stay  { color: #6EE7B7; }
.result-desc {
    font-size: 0.86rem;
    color: #94A3B8;
    line-height: 1.65;
}

/* Progress bar */
.result-bar-wrap {
    margin-top: 1rem;
    width: 100%;
}
.result-bar-track {
    height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
    width: 100%;
}
.result-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
}
.result-bar-fill.churn { background: linear-gradient(90deg, #EF4444, #FCA5A5); }
.result-bar-fill.stay  { background: linear-gradient(90deg, #10B981, #6EE7B7); }

/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 2rem 0 !important; }

/* ── Footer links ── */
.footer-links {
    display: flex;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.footer-links a {
    color: #4B6487 !important;
    text-decoration: none;
    font-size: 0.82rem;
    transition: color 0.2s;
}
.footer-links a:hover { color: #63B3ED !important; }
.footer-copy {
    text-align: center;
    color: #2D3748;
    font-size: 0.75rem;
    margin-top: 0.75rem;
    letter-spacing: 0.04em;
}

/* ── Dropdown menus ── */
[data-baseweb="popover"] ul {
    background: #1A2235 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
[data-baseweb="popover"] li:hover { background: rgba(99,179,237,0.1) !important; }

/* ── Number input arrows ── */
button[kind="stepUp"], button[kind="stepDown"] {
    color: #63B3ED !important;
}

/* ── Spinner ── */
.stSpinner > div > div { border-top-color: #63B3ED !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-icon">🛡️</div>
    <div>
        <div class="hero-title">ChurnGuard</div>
        <div class="hero-subtitle">ML-powered customer retention intelligence · Powered by CatBoost</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("About")
st.sidebar.info("""
ChurnGuard helps businesses identify at-risk customers so they can take proactive retention measures.

The backend uses a **FastAPI** server and a **PostgreSQL** database for real-time inference and data logging.
""")

# ─────────────────────────────────────────────
# API Config
# ─────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# ─────────────────────────────────────────────
# MAIN FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Information</div>', unsafe_allow_html=True)

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="section-label" style="font-size:0.65rem; margin-top:0">Personal Details</div>', unsafe_allow_html=True)
        customerID      = st.text_input("Customer ID", value="7590-VHVEG")
        gender          = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen   = st.selectbox("Senior Citizen", [0, 1], help="0 = No, 1 = Yes")
        Partner         = st.selectbox("Has Partner?", ["Yes", "No"])
        Dependents      = st.selectbox("Has Dependents?", ["No", "Yes"])

    with col2:
        st.markdown('<div class="section-label" style="font-size:0.65rem; margin-top:0">Service Details</div>', unsafe_allow_html=True)
        tenure           = st.slider("Tenure (Months)", 0, 72, 1)
        PhoneService     = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines    = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown('<div class="section-label" style="font-size:0.65rem; margin-top:0">Contract & Billing</div>', unsafe_allow_html=True)
        StreamingTV      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        Contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod    = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges   = st.number_input("Monthly Charges ($)", min_value=0.0, value=29.85, step=0.01)
        TotalCharges     = st.number_input("Total Charges ($)", min_value=0.0, value=29.85, step=0.01)

    submit_button = st.form_submit_button("⚡  Run Prediction")

# ─────────────────────────────────────────────
# PREDICTION RESULT
# ─────────────────────────────────────────────
if submit_button:
    payload = {
        "customerID": customerID, "gender": gender,
        "SeniorCitizen": int(SeniorCitizen), "Partner": Partner,
        "Dependents": Dependents, "tenure": int(tenure),
        "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract,
        "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges), "TotalCharges": float(TotalCharges)
    }

    try:
        with st.spinner("Analyzing customer profile…"):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result      = response.json()
            prediction  = result.get("prediction")
            probability = result.get("probability", 0.5)
            confidence  = probability if prediction == 1 else (1 - probability)
            pct         = f"{confidence:.1%}"
            bar_width   = f"{confidence * 100:.1f}%"

            st.markdown("---")

            if prediction == 1:
                st.markdown(f"""
                <div class="result-card churn">
                    <div class="result-pct-wrap">
                        <div class="result-pct churn">{pct}</div>
                        <div class="result-pct-label churn">Churn Probability</div>
                    </div>
                    <div class="result-divider churn"></div>
                    <div class="result-body">
                        <span class="result-badge badge-churn">⚠&nbsp; High Risk</span>
                        <div class="result-title churn">🚨 Likely to Churn</div>
                        <div class="result-desc">
                            There is a <strong style="color:#FCA5A5">{pct} chance</strong> this customer will leave the service.
                            Consider targeted retention outreach — a personalised offer or a dedicated account manager call may prevent churn.
                        </div>
                        <div class="result-bar-wrap">
                            <div class="result-bar-track">
                                <div class="result-bar-fill churn" style="width:{bar_width}"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card stay">
                    <div class="result-pct-wrap">
                        <div class="result-pct stay">{pct}</div>
                        <div class="result-pct-label stay">Retention Probability</div>
                    </div>
                    <div class="result-divider stay"></div>
                    <div class="result-body">
                        <span class="result-badge badge-stay">✓&nbsp; Low Risk</span>
                        <div class="result-title stay">✅ Likely to Stay</div>
                        <div class="result-desc">
                            There is a <strong style="color:#6EE7B7">{pct} chance</strong> this customer will remain with the service.
                            No immediate retention action required — continue monitoring engagement metrics regularly.
                        </div>
                        <div class="result-bar-wrap">
                            <div class="result-bar-track">
                                <div class="result-bar-fill stay" style="width:{bar_width}"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"API Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Could not connect to the API: {e}")
        st.info(f"Ensure the FastAPI server is running at `{API_URL}`")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="footer-links">
    <a href="https://github.com/abhi24112/Customer-Churn-Prediction-MLOps">📂 Project Repository</a>
    <a href="https://github.com/abhi24112">👤 GitHub Profile</a>
    <a href="https://www.linkedin.com/in/abhipraj/">🔗 LinkedIn</a>
</div>
<div class="footer-copy">Built with Streamlit &amp; FastAPI &nbsp;·&nbsp; © 2026 MLOps Pipeline</div>
""", unsafe_allow_html=True)