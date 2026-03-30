import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ──────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Global ── */
    html, body, .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* ── Header banner ── */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, .35);
    }
    .hero h1 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #ffffff;
        margin: 0 0 .3rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,.8);
        font-size: 1.05rem;
        margin: 0;
    }

    /* ── Section cards ── */
    .section-card {
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 1.6rem 1.4rem 1.2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(6px);
    }
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: .85rem;
        font-weight: 600;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    /* ── Result boxes ── */
    .result-box {
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-churn {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        box-shadow: 0 6px 24px rgba(255, 65, 108, .4);
    }
    .result-stay {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 6px 24px rgba(56, 239, 125, .35);
    }
    .result-box h2 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.6rem;
        color: #fff;
        margin: 0 0 .3rem;
    }
    .result-box p {
        color: rgba(255,255,255,.85);
        font-size: .95rem;
        margin: 0;
    }

    /* ── Streamlit overrides ── */
    .stSelectbox label, .stNumberInput label {
        color: rgba(255,255,255,.75) !important;
        font-weight: 500 !important;
    }
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,.06) !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        color: #fff !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,.06) !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        color: #fff !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: .75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: .3px;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, .5);
    }

    /* ── Confidence meter ── */
    .conf-bar-bg {
        background: rgba(255,255,255,.12);
        border-radius: 8px;
        height: 10px;
        margin-top: .6rem;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width .6s ease;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Predictor</h1>
    <p>Enter customer details below to predict whether they will churn or stay.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model & columns
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("churn_model.pkl", "rb"))
    columns = pickle.load(open("model_columns.pkl", "rb"))
    return model, columns

try:
    model, model_columns = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "⚠️ **Model files not found.** Place `churn_model.pkl` and "
        "`model_columns.pkl` in the same directory as this app, then reload."
    )

# ──────────────────────────────────────────────
# Categorical options (match typical Telco dataset)
# ──────────────────────────────────────────────
CATEGORICAL_FIELDS = {
    "gender":           ["Female", "Male"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["Yes", "No"],
    "PhoneService":     ["Yes", "No"],
    "MultipleLines":    ["No phone service", "No", "Yes"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "Yes", "No internet service"],
    "OnlineBackup":     ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport":      ["No", "Yes", "No internet service"],
    "StreamingTV":      ["No", "Yes", "No internet service"],
    "StreamingMovies":  ["No", "Yes", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod":    ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)",
                         "Credit card (automatic)"],
}

NUMERIC_FIELDS = {
    "SeniorCitizen": {"min": 0, "max": 1, "default": 0, "step": 1,
                      "help": "0 = No, 1 = Yes"},
    "tenure":        {"min": 0, "max": 72, "default": 12, "step": 1,
                      "help": "Months with the company"},
    "MonthlyCharges": {"min": 0.0, "max": 200.0, "default": 50.0,
                       "step": 0.5, "help": "Monthly charge in $"},
    "TotalCharges":   {"min": 0.0, "max": 10000.0, "default": 600.0,
                       "step": 10.0, "help": "Cumulative charges in $"},
}

# ──────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────

# — Numeric section —
st.markdown('<div class="section-card"><div class="section-title">⸻ Account & Charges</div>', unsafe_allow_html=True)
num_cols = st.columns(2)
numeric_inputs = {}
for idx, (field, cfg) in enumerate(NUMERIC_FIELDS.items()):
    with num_cols[idx % 2]:
        numeric_inputs[field] = st.number_input(
            field,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg["step"],
            help=cfg["help"],
        )
st.markdown('</div>', unsafe_allow_html=True)

# — Categorical section —
st.markdown('<div class="section-card"><div class="section-title">⸻ Service Details</div>', unsafe_allow_html=True)
cat_cols = st.columns(2)
categorical_inputs = {}
for idx, (field, options) in enumerate(CATEGORICAL_FIELDS.items()):
    with cat_cols[idx % 2]:
        categorical_inputs[field] = st.selectbox(field, options)
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Prediction logic
# ──────────────────────────────────────────────
def align_input_to_model(raw: dict, model_cols: list) -> pd.DataFrame:
    """
    Convert raw user inputs into a DataFrame whose columns match
    exactly what the trained model expects.

    Steps:
      1. Build a single-row DataFrame from the raw inputs.
      2. Apply pd.get_dummies() — same function used during training.
      3. Use DataFrame.reindex(columns=model_cols, fill_value=0)
         → adds any dummy column the model expects but the input
           didn't produce (fill with 0).
         → drops any extra dummy column the input produced but the
           model doesn't expect.
      4. Column order is guaranteed to match model_cols.

    This prevents shape-mismatch or column-order errors at predict time.
    """
    df = pd.DataFrame([raw])
    df_encoded = pd.get_dummies(df)
    df_aligned = df_encoded.reindex(columns=model_cols, fill_value=0)
    return df_aligned


st.markdown("---")

if st.button("🔮  Predict Churn"):
    if not model_loaded:
        st.error("Cannot predict — model files are missing.")
    else:
        # Merge numeric + categorical into one dict
        raw_input = {**numeric_inputs, **categorical_inputs}

        # Align with training columns
        input_df = align_input_to_model(raw_input, model_columns)

        # Predict
        prediction = model.predict(input_df)[0]

        # Probability (Logistic Regression supports predict_proba)
        proba = model.predict_proba(input_df)[0]
        churn_prob = proba[1] * 100
        stay_prob = proba[0] * 100

        # Display result
        if prediction == 1:
            bar_color = "#ff416c"
            st.markdown(f"""
            <div class="result-box result-churn">
                <h2>⚠️ Customer Will Churn</h2>
                <p>Churn probability: <strong>{churn_prob:.1f}%</strong></p>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{churn_prob}%;background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bar_color = "#38ef7d"
            st.markdown(f"""
            <div class="result-box result-stay">
                <h2>✅ Customer Will Stay</h2>
                <p>Retention probability: <strong>{stay_prob:.1f}%</strong></p>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{stay_prob}%;background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Expandable detail
        with st.expander("🔍 View encoded input sent to model"):
            st.dataframe(input_df, use_container_width=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:rgba(255,255,255,.35);font-size:.8rem;'>"
    "Built with Streamlit · Logistic Regression model via scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
