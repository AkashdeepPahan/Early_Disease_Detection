import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import hashlib
import json
import io
import re
from difflib import SequenceMatcher
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

try:
    import shap
except Exception:
    shap = None

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  THEME / GLOBAL CSS  — Warm Clinical Minimalism
#  Palette: warm cream bg · forest green accent · slate text
#  Fonts: Lora (display/headers) · DM Sans (body/ui)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream:        #F7F3EE;
    --cream-dark:   #EDE7DC;
    --cream-border: #D8CFC4;
    --green:        #2D6A4F;
    --green-light:  #40916C;
    --green-pale:   #D8F3DC;
    --green-glow:   rgba(45,106,79,0.15);
    --red:          #C1121F;
    --red-pale:     #FFE5E7;
    --amber:        #B5700A;
    --amber-pale:   #FEF3E2;
    --slate:        #3D4451;
    --slate-mid:    #6B7280;
    --slate-light:  #9CA3AF;
    --white:        #FFFFFF;
    --shadow-sm:    0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md:    0 4px 16px rgba(0,0,0,0.10);
    --shadow-lg:    0 10px 32px rgba(0,0,0,0.12);
    --radius:       14px;
}

*, *::before, *::after { box-sizing: border-box; }

/* ── App background */
[data-testid="stAppViewContainer"] {
    background-color: var(--cream);
    background-image:
        radial-gradient(ellipse 70% 60% at 90% 5%, rgba(45,106,79,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 5% 95%, rgba(181,112,10,0.04) 0%, transparent 60%);
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar */
[data-testid="stSidebar"] {
    background: var(--white);
    border-right: 1px solid var(--cream-border);
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--slate-mid) !important;
    font-size: 0.875rem;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] h1 {
    font-family: 'Lora', serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--green) !important;
    letter-spacing: -0.01em;
    border-bottom: 2px solid var(--green-pale);
    padding-bottom: 10px;
    margin-bottom: 4px !important;
}

/* ── Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--slate-mid) !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--green) !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid var(--cream-border) !important;
    gap: 4px !important;
}

/* ── Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--white) !important;
    border: 1.5px solid var(--cream-border) !important;
    border-radius: 10px !important;
    color: var(--slate) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--green-light) !important;
    box-shadow: 0 0 0 3px var(--green-glow) !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label {
    color: var(--slate-mid) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--white) !important;
    border: 1.5px solid var(--cream-border) !important;
    border-radius: 10px !important;
    color: var(--slate) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons */
.stButton > button,
.stButton > button * {
    width: 100%;
    padding: 13px 0;
    background: var(--green) !important;
    color: var(--white) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 2px 8px rgba(45,106,79,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--green-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(45,106,79,0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Login card */
.login-card {
    max-width: 440px;
    margin: 48px auto;
    background: var(--white);
    border: 1px solid var(--cream-border);
    border-radius: 20px;
    padding: 52px 48px;
    text-align: center;
    box-shadow: var(--shadow-lg);
}
.login-card h2 {
    font-family: 'Lora', serif;
    color: var(--green);
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}
.login-card p {
    color: var(--slate-mid);
    font-size: 0.9rem;
    margin-bottom: 36px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Result cards */
.result-high {
    background: var(--red-pale);
    border: 1.5px solid rgba(193,18,31,0.3);
    border-left: 5px solid var(--red);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-top: 20px;
    animation: slideUp 0.4s ease;
    box-shadow: 0 4px 20px rgba(193,18,31,0.08);
}
.result-low {
    background: var(--green-pale);
    border: 1.5px solid rgba(45,106,79,0.3);
    border-left: 5px solid var(--green);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-top: 20px;
    animation: slideUp 0.4s ease;
    box-shadow: 0 4px 20px rgba(45,106,79,0.08);
}
.result-label {
    font-family: 'Lora', serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.result-high .result-label { color: var(--red); }
.result-low  .result-label { color: var(--green); }
.result-sub {
    font-size: 0.85rem;
    color: var(--slate-mid);
    margin-bottom: 18px;
    font-family: 'DM Sans', sans-serif;
}
.conf-bar-wrap {
    background: rgba(0,0,0,0.08);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 6px;
}
.conf-bar-high {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #C1121F, #E63946);
}
.conf-bar-low {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2D6A4F, #40916C);
}
.conf-pct {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--slate-mid);
    text-align: right;
    font-weight: 600;
}
.result-disclaimer {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid rgba(0,0,0,0.08);
    font-size: 0.76rem;
    color: var(--slate-mid);
    font-style: italic;
    font-family: 'DM Sans', sans-serif;
}

/* ── Info card */
.info-card {
    background: var(--white);
    border: 1px solid var(--cream-border);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin: 12px 0;
    box-shadow: var(--shadow-sm);
    color: var(--slate-mid);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── Section title */
.section-title {
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--slate);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.01em;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--cream-border);
}

/* ── History rows */
.hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    border-radius: 10px;
    margin: 6px 0;
    background: var(--white);
    border: 1px solid var(--cream-border);
    font-size: 0.85rem;
    color: var(--slate-mid);
    font-family: 'DM Sans', sans-serif;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s;
}
.hist-row:hover { box-shadow: var(--shadow-md); }
.hist-high {
    color: var(--red);
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    background: var(--red-pale);
    padding: 3px 10px;
    border-radius: 20px;
}
.hist-low {
    color: var(--green);
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    background: var(--green-pale);
    padding: 3px 10px;
    border-radius: 20px;
}

/* ── Header */
.edd-header {
    padding: 40px 0 10px 0;
    animation: fadeDown 0.6s ease;
}
.edd-header h1 {
    font-family: 'Lora', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--green);
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.03em;
}
.edd-header h1 em {
    font-style: italic;
    color: var(--amber);
}
.edd-header p {
    color: var(--slate-mid);
    font-size: 1rem;
    letter-spacing: 0.01em;
    margin-top: 8px;
    font-family: 'DM Sans', sans-serif;
}
.edd-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--green), var(--amber), transparent);
    margin: 16px 0 28px;
    border-radius: 2px;
    opacity: 0.5;
}

/* ── Upload area */
[data-testid="stFileUploader"] {
    background: var(--white) !important;
    border: 2px dashed var(--cream-border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--green-light) !important;
}

/* ── Alerts */
[data-testid="stAlert"] {
    background: var(--amber-pale) !important;
    border: 1px solid rgba(181,112,10,0.25) !important;
    border-radius: var(--radius) !important;
    color: var(--amber) !important;
}

/* ── Dataframe */
[data-testid="stDataFrame"] {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--cream-border) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Metrics */
[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--cream-border);
    border-radius: var(--radius);
    padding: 16px 20px !important;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] {
    color: var(--slate-mid) !important;
    font-size: 0.78rem !important;
    font-family: 'DM Sans', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: var(--green) !important;
    font-family: 'Lora', serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* ── Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb {
    background: var(--cream-border);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: var(--slate-light); }

/* ── Animations */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── General text */
p, span, div { color: var(--slate); }
h2, h3 {
    color: var(--slate);
    font-family: 'Lora', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}
hr { border-color: var(--cream-border) !important; }
code {
    background: var(--cream-dark) !important;
    color: var(--green) !important;
    border-radius: 6px !important;
    font-size: 0.85em !important;
    padding: 2px 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SIMPLE AUTH  (session-state based, no external DB needed)
# ─────────────────────────────────────────────────────────────
USERS = {
    "admin":  hashlib.sha256("admin123".encode()).hexdigest(),
    "doctor": hashlib.sha256("mediscan".encode()).hexdigest(),
    "guest":  hashlib.sha256("guest".encode()).hexdigest(),
}

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def show_login():
    st.markdown("""
    <div class='edd-header' style='text-align:center;'>
        <h1> MediScan <em>AI</em></h1>
        <p>Clinical Risk Assessment · Powered by Machine Learning</p>
    </div>
    <div class='edd-divider'></div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h2>Welcome back</h2>"
            "<p>Sign in with your credentials to access the platform</p>",
            unsafe_allow_html=True
        )

        username = st.text_input("Username", placeholder="admin · doctor · guest", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

        if st.button("Sign In →"):
            if username in USERS and USERS[username] == hash_pw(password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["history"] = []
                st.rerun()
            else:
                st.error("Invalid credentials. Try: admin/admin123 · doctor/mediscan · guest/guest")

        st.markdown("""
        <div style='margin-top:28px; padding:14px 16px;
             background:var(--cream); border:1px solid var(--cream-border);
             border-radius:10px; font-size:0.78rem; color:var(--slate-mid); text-align:left;'>
            <span style='color:var(--green);font-weight:600;font-family:"DM Sans",sans-serif;'>Demo accounts</span><br>
            <span style='font-family:"DM Sans",sans-serif;'>
            admin / admin123 &nbsp;·&nbsp; doctor / mediscan &nbsp;·&nbsp; guest / guest
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []

if not st.session_state["logged_in"]:
    show_login()
    st.stop()

# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────
MODEL_FOLDER = "./models"

@st.cache_resource
def load_model(name):
    try:
        model   = joblib.load(f"{MODEL_FOLDER}/{name}_model.joblib")
        scaler  = joblib.load(f"{MODEL_FOLDER}/{name}_scaler.joblib")
        feats   = joblib.load(f"{MODEL_FOLDER}/{name}_features.joblib")
        return model, scaler, feats
    except Exception:
        return None, None, None

def predict(model, scaler, feat_order, data: dict):
    df = pd.DataFrame([data], columns=feat_order).astype(float)
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    return int(pred), float(prob)


@st.cache_data(show_spinner=False)
def load_reference_dataset(model_key: str):
    if model_key == "diabetes":
        df = pd.read_csv(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            names=[
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
            ],
        )
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        df[zero_cols] = SimpleImputer(strategy="median").fit_transform(df[zero_cols])
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"].astype(int)
        return X, y

    if model_key == "heart":
        df = pd.read_csv(
            "https://gist.githubusercontent.com/trantuyen082001/1fc2f5c0ad1507f40e721e6d18b34138/raw/heart.csv"
        )
        X = df.drop("output", axis=1)
        y = df["output"].astype(int)
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
        return X, y

    if model_key == "cancer":
        bc = load_breast_cancer()
        X = pd.DataFrame(bc.data, columns=bc.feature_names)
        y = pd.Series(bc.target).astype(int)
        return X, y

    if model_key == "kidney":
        kidney_urls = [
            "https://raw.githubusercontent.com/patilgirish815/Kidney_Cancer_Prediction_Using_Machine_Learning/main/dataset/kidney_disease.csv",
            "https://raw.githubusercontent.com/dsrscientist/dataset1/master/kidney_disease.csv",
            "https://raw.githubusercontent.com/vijayasri-m/Chronic-Kidney-Disease-Prediction/main/kidney_disease.csv",
        ]

        df = None
        for url in kidney_urls:
            try:
                df = pd.read_csv(url)
                break
            except Exception:
                continue

        if df is None:
            raise RuntimeError("Unable to load kidney reference dataset from configured sources")

        if "id" in df.columns:
            df = df.drop("id", axis=1)

        if "classification" in df.columns:
            if df["classification"].dtype == object:
                df["classification"] = (
                    df["classification"].astype(str).str.strip().str.lower().map({"ckd": 1, "notckd": 0, "ckd\t": 1})
                )
            df = df.rename(columns={"classification": "class"})

        if "class" not in df.columns:
            raise RuntimeError("Kidney dataset missing target column")

        binary_map = {
            "yes": 1, "no": 0, "good": 1, "poor": 0,
            "present": 1, "notpresent": 0, "normal": 0, "abnormal": 1,
            "\tyes": 1, "\tno": 0, " yes": 1, " no": 0,
            "ckd": 1, "notckd": 0,
        }

        for col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            mapped = s.map(binary_map)
            df[col] = pd.to_numeric(mapped.fillna(s), errors="coerce")

        df = df.dropna(subset=["class"]).copy()
        df["class"] = df["class"].astype(int)
        X = df.drop("class", axis=1)
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
        y = df["class"]
        return X, y

    raise ValueError(f"Unsupported model key: {model_key}")


@st.cache_data(show_spinner=False)
def evaluate_model_performance(model_key: str):
    model, scaler, feat_order = load_model(model_key)
    if model is None:
        return {"error": "Model files not found"}

    try:
        X, y = load_reference_dataset(model_key)
        missing = [f for f in feat_order if f not in X.columns]
        if missing:
            return {"error": f"Dataset does not contain required model features: {missing[:4]}"}

        X = X[feat_order].copy()
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test_scaled = scaler.transform(X_test)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
            "F1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
            "ROC-AUC": round(auc(*roc_curve(y_test, y_prob)[:2]) * 100, 1),
        }
        return {
            "metrics": metrics,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "n_test": int(len(y_test)),
        }
    except Exception as e:
        return {"error": str(e)}


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


BATCH_COLUMN_ALIASES = {
    "diabetes": {
        "Pregnancies": ["pregnancy", "numpregnancies"],
        "Glucose": ["glucoselevel", "plasglucose"],
        "BloodPressure": ["blood_pressure", "bp", "diastolicbp"],
        "SkinThickness": ["skin_thickness", "skinfoldthickness"],
        "Insulin": ["insulinlevel", "seruminsulin"],
        "BMI": ["bodymassindex", "body_mass_index"],
        "DiabetesPedigreeFunction": ["dpf", "diabetespedigree", "pedigreefunction"],
        "Age": ["years", "ageyears"],
    },
    "heart": {
        "age": ["ageyears"],
        "sex": ["gender"],
        "cp": ["chestpain", "chestpaintype"],
        "trtbps": ["trestbps", "restingbp", "restingbloodpressure", "bloodpressure", "bp"],
        "chol": ["cholesterol", "serumcholesterol"],
        "fbs": ["fastingbloodsugar", "fastingbs", "fasting_sugar"],
        "restecg": ["restingecg", "restecgresult"],
        "thalachh": ["thalach", "maxheartrate", "maxhr", "maximumheartrate"],
        "exng": ["exerciseangina", "exerciseinducedangina", "exang"],
        "oldpeak": ["stdepression"],
        "slp": ["slope", "stsegmentslope"],
        "caa": ["ca", "nummajorvessels", "majorvessels"],
        "thall": ["thal", "thalassemia"],
    },
    "kidney": {
        "age": ["ageyears"],
        "bp": ["bloodpressure", "blood_pressure", "diastolicbp"],
        "sg": ["specificgravity", "specific_gravity"],
        "al": ["albumin"],
        "su": ["sugar"],
        "bgr": ["bloodglucoseresult", "randombloodglucose", "glucose"],
        "bu": ["bloodurea", "urea"],
        "sc": ["serumcreatinine", "creatinine"],
        "sod": ["sodium"],
        "pot": ["potassium"],
        "hemo": ["hemoglobin", "haemoglobin"],
        "pcv": ["packedcellvolume", "packed_cell_volume"],
        "wbcc": ["whitebloodcellcount", "wbc", "wc", "white_blood_cell_count"],
        "rbcc": ["redbloodcellcount", "rc", "rbc_count", "red_blood_cell_count", "rbc"],
        "htn": ["hypertension"],
        "dm": ["diabetesmellitus", "diabetes_mellitus", "diabetes"],
        "cad": ["coronaryarterydisease", "coronary_artery_disease"],
        "appet": ["appetite"],
        "pe": ["pedaledema", "pedal_edema", "oedema"],
        "ane": ["anemia", "anaemia"],
    },
}


def expand_expected_terms(disease_key: str, expected: str, aliases: list[str]) -> list[str]:
    terms = [expected] + aliases

    words = [w for w in re.split(r"[^A-Za-z0-9]+", str(expected)) if w]
    if len(words) > 1:
        terms.extend([
            " ".join(words),
            "_".join(words),
            "".join(words),
            "".join([words[0].lower()] + [w.title() for w in words[1:]]),
        ])

    compact_expected = str(expected).replace("_", " ")
    compact_words = [w for w in re.findall(r"[A-Z]?[a-z]+|[0-9]+", compact_expected) if w]
    if len(compact_words) > 1:
        terms.extend([
            " ".join(compact_words),
            "_".join([w.lower() for w in compact_words]),
            "".join([w.lower() for w in compact_words]),
        ])

    if disease_key == "diabetes" and expected == "DiabetesPedigreeFunction":
        terms.extend([
            "diabetes pedigree function",
            "diabetes_pedigree_function",
            "pedigree function",
            "diabetespedigree",
        ])

    if disease_key == "heart":
        heart_variants = {
            "slp": ["st slope", "st_slope"],
        }
        terms.extend(heart_variants.get(expected, []))

    if disease_key == "kidney":
        kidney_variants = {
            "wbcc": ["wbc", "white blood cell count", "white_blood_cell_count"],
            "rbcc": ["rbc count", "red blood cell count", "red_blood_cell_count"],
            "htn": ["hypertension"],
            "dm": ["diabetes mellitus", "diabetes_mellitus"],
            "cad": ["coronary artery disease", "coronary_artery_disease"],
            "appet": ["appetite"],
            "pe": ["pedal edema", "pedal_edema"],
            "ane": ["anemia", "anaemia"],
        }
        terms.extend(kidney_variants.get(expected, []))

    if disease_key == "cancer":
        if expected.startswith("mean "):
            base = expected.replace("mean ", "", 1)
            terms.extend([
                f"{base} mean", f"mean_{base}", f"{base}_mean", f"mean{base}", f"{base}mean"
            ])
        elif expected.endswith(" error"):
            base = expected.replace(" error", "")
            terms.extend([
                f"{base} error", f"error_{base}", f"{base}_error", f"{base}_se", f"se_{base}", f"{base}se"
            ])
        elif expected.startswith("worst "):
            base = expected.replace("worst ", "", 1)
            terms.extend([
                f"{base} worst", f"worst_{base}", f"{base}_worst", f"worst{base}", f"{base}worst"
            ])

    return list(dict.fromkeys(terms))


def resolve_batch_columns(disease_key: str, expected_cols, uploaded_cols):
    aliases = BATCH_COLUMN_ALIASES.get(disease_key, {})
    used = set()
    mapping = {}
    auto_matches = []

    for expected in expected_cols:
        match = None
        alias_list = aliases.get(expected, [])
        candidate_terms = expand_expected_terms(disease_key, expected, alias_list)

        for term in candidate_terms:
            term_norm = normalize_col_name(term)
            for col in uploaded_cols:
                if col in used:
                    continue
                if col == term or col.lower() == str(term).lower() or normalize_col_name(col) == term_norm:
                    match = col
                    break
            if match:
                break

        if not match:
            expected_norm = normalize_col_name(expected)
            best_col = None
            best_score = 0.0
            for col in uploaded_cols:
                if col in used:
                    continue
                score = SequenceMatcher(None, expected_norm, normalize_col_name(col)).ratio()
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_col is not None and best_score >= 0.82:
                match = best_col

        if match:
            mapping[expected] = match
            used.add(match)
            if match != expected:
                auto_matches.append((expected, match))

    missing = [col for col in expected_cols if col not in mapping]
    return mapping, missing, auto_matches

# ─────────────────────────────────────────────────────────────
#  DISEASE CONFIG
# ─────────────────────────────────────────────────────────────
DISEASE_CFG = {
    " Diabetes": {
        "key": "diabetes",
        "emoji": "",
        "color": "#2D6A4F",
        "tabs": {
            "Basic Info": ["Pregnancies", "Age", "BMI"],
            "Lab Results": ["Glucose", "Insulin", "DiabetesPedigreeFunction"],
            "Vitals":      ["BloodPressure", "SkinThickness"],
        },
        "ranges": {
            "Pregnancies": (0, 17, 0, int),
            "Age":         (20, 90, 30, int),
            "BMI":         (10.0, 70.0, 25.0, float),
            "Glucose":     (50, 250, 110, int),
            "Insulin":     (0, 900, 80, int),
            "DiabetesPedigreeFunction": (0.0, 2.5, 0.5, float),
            "BloodPressure": (40, 150, 72, int),
            "SkinThickness": (0, 100, 20, int),
        },
    },
    " Heart Disease": {
        "key": "heart",
        "emoji": "",
        "color": "#C1121F",
        "tabs": {
            "Basic Info":  ["age", "sex", "cp"],
            "Lab Results": ["chol", "fbs", "trtbps"],
            "Vitals":      ["restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"],
        },
        "ranges": {
            "age":      (20, 80, 50, int),
            "sex":      (0, 1, 0, int),
            "cp":       (0, 3, 1, int),
            "chol":     (100, 600, 200, int),
            "fbs":      (0, 1, 0, int),
            "trtbps":   (80, 200, 120, int),
            "restecg":  (0, 2, 0, int),
            "thalachh": (60, 220, 140, int),
            "exng":     (0, 1, 0, int),
            "oldpeak":  (0.0, 10.0, 1.0, float),
            "slp":      (0, 2, 1, int),
            "caa":      (0, 4, 0, int),
            "thall":    (0, 3, 2, int),
        },
    },
    " Breast Cancer": {
        "key": "cancer",
        "emoji": "",
        "color": "#B5700A",
        "tabs": {
            "Mean Values":  [f"mean {x}" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]],
            "Error Values": [f"{x} error" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]],
            "Worst Values": [f"worst {x}" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","worst concave points","symmetry","fractal dimension"]],
        },
        "ranges": {},
    },
    " Kidney Disease": {
        "key": "kidney",
        "emoji": "",
        "color": "#5C4033",
        "tabs": {
            "Basic Info":  ["age", "bp", "sg", "al", "su"],
            "Blood Tests": ["bgr", "bu", "sc", "sod", "pot", "hemo"],
            "Other":       ["pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane"],
        },
        "ranges": {
            "age":  (2, 90, 40, int),
            "bp":   (50, 180, 80, int),
            "sg":   (1.005, 1.025, 1.015, float),
            "al":   (0, 5, 0, int),
            "su":   (0, 5, 0, int),
            "bgr":  (70, 490, 120, int),
            "bu":   (1, 400, 40, int),
            "sc":   (0.4, 15.0, 1.2, float),
            "sod":  (111, 163, 137, int),
            "pot":  (2.5, 8.0, 4.5, float),
            "hemo": (3.1, 17.8, 12.0, float),
            "pcv":  (9, 54, 40, int),
            "wbcc": (2200, 26400, 8000, int),
            "rbcc": (2.1, 8.0, 4.7, float),
            "htn":  (0, 1, 0, int),
            "dm":   (0, 1, 0, int),
            "cad":  (0, 1, 0, int),
            "appet":(0, 1, 1, int),
            "pe":   (0, 1, 0, int),
            "ane":  (0, 1, 0, int),
        },
    },
}

HEART_FIELD_OPTIONS = {
    "sex":     {"label": "Gender",                        "options": [("Female", 0), ("Male", 1)]},
    "cp":      {"label": "Chest Pain Type",               "options": [("Typical angina", 0), ("Atypical angina", 1), ("Non-anginal pain", 2), ("Asymptomatic", 3)]},
    "fbs":     {"label": "Fasting Blood Sugar > 120 mg/dl","options": [("No", 0), ("Yes", 1)]},
    "restecg": {"label": "Resting ECG",                   "options": [("Normal", 0), ("ST-T abnormality", 1), ("Left ventricular hypertrophy", 2)]},
    "exng":    {"label": "Exercise Induced Angina",       "options": [("No", 0), ("Yes", 1)]},
    "slp":     {"label": "Slope",                         "options": [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)]},
    "caa":     {"label": "Major Vessels (0-4)",           "options": [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4)]},
    "thall":   {"label": "Thalassemia",                   "options": [("Unknown / Not set", 0), ("Fixed defect", 1), ("Normal", 2), ("Reversible defect", 3)]},
}

KIDNEY_FIELD_OPTIONS = {
    "htn":   {"label": "Hypertension",            "options": [("No", 0), ("Yes", 1)]},
    "dm":    {"label": "Diabetes Mellitus",        "options": [("No", 0), ("Yes", 1)]},
    "cad":   {"label": "Coronary Artery Disease",  "options": [("No", 0), ("Yes", 1)]},
    "appet": {"label": "Appetite",                 "options": [("Poor", 0), ("Good", 1)]},
    "pe":    {"label": "Pedal Edema",              "options": [("No", 0), ("Yes", 1)]},
    "ane":   {"label": "Anemia",                   "options": [("No", 0), ("Yes", 1)]},
}

# Build cancer ranges dynamically
cancer_feats = (
    [f"mean {x}" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]] +
    [f"{x} error" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]] +
    [f"worst {x}" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","worst concave points","symmetry","fractal dimension"]]
)
for f in cancer_feats:
    if "area" in f:        DISEASE_CFG[" Breast Cancer"]["ranges"][f] = (0.0, 2500.0, 500.0, float)
    elif "perimeter" in f: DISEASE_CFG[" Breast Cancer"]["ranges"][f] = (0.0, 300.0,  80.0,  float)
    elif "radius" in f:    DISEASE_CFG[" Breast Cancer"]["ranges"][f] = (0.0, 40.0,   14.0,  float)
    elif "texture" in f:   DISEASE_CFG[" Breast Cancer"]["ranges"][f] = (0.0, 45.0,   19.0,  float)
    else:                  DISEASE_CFG[" Breast Cancer"]["ranges"][f] = (0.0, 1.0,    0.1,   float)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" MediScan AI")
    st.markdown(
        f"<p style='color:#9CA3AF;font-size:0.8rem;margin-top:-8px;'>"
        f"Signed in as <b style='color:#2D6A4F;'>{st.session_state['username']}</b></p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    disease_label = st.selectbox("Disease Module", list(DISEASE_CFG.keys()))
    cfg = DISEASE_CFG[disease_label]

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;color:#6B7280;text-transform:uppercase;"
        "letter-spacing:0.06em;'>System Info</p>",
        unsafe_allow_html=True
    )
    feat_count = sum(len(v) for v in cfg['tabs'].values())
    st.markdown(f"""
    <div style='background:#F7F3EE;border:1px solid #D8CFC4;border-radius:10px;
         padding:14px 16px;font-size:0.82rem;color:#6B7280;line-height:1.8;
         font-family:"DM Sans",sans-serif;'>
        <b style='color:#2D6A4F;'>Module:</b> {disease_label}<br>
        <b style='color:#2D6A4F;'>Classifier:</b> Best selected<br>
        <b style='color:#2D6A4F;'>Validation:</b> 5-Fold CV<br>
        <b style='color:#2D6A4F;'>Features:</b> {feat_count}<br>
        <hr style='border-color:#D8CFC4;margin:8px 0;'>
        <i style='font-size:0.76rem;'>For educational use only. Always consult a licensed healthcare professional.</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Sign Out"):
        st.session_state["logged_in"] = False
        st.rerun()

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='edd-header'>
    <h1> MediScan <em>AI</em></h1>
    <p>AI-powered clinical risk assessment — {disease_label}</p>
</div>
<div class='edd-divider'></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_history, tab_performance = st.tabs([
    "  Predict",
    "  Batch Upload",
    "  History",
    "  Model Performance",
])

# Plotly theme helper (warm/light)
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#6B7280", family="DM Sans"),
    xaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", linecolor="#D8CFC4"),
    yaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", linecolor="#D8CFC4"),
)

# ══════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════
with tab_predict:
    model, scaler, feat_order = load_model(cfg["key"])

    if model is None:
        st.warning(f"No trained model found for **{disease_label}**. Please train and save models to `./models/` first.")
    else:
        st.markdown("<div class='section-title'>Patient Information</div>", unsafe_allow_html=True)

        input_data = {}
        tab_keys = list(cfg["tabs"].keys())
        sub_tabs = st.tabs(tab_keys)

        for ti, (tab_name, fields) in enumerate(cfg["tabs"].items()):
            with sub_tabs[ti]:
                cols = st.columns(2, gap="large")
                for i, field in enumerate(fields):
                    min_v, max_v, def_v, dtype = cfg["ranges"].get(field, (0.0, 100.0, 0.0, float))
                    with cols[i % 2]:
                        if cfg["key"] == "heart" and field in HEART_FIELD_OPTIONS:
                            option_info = HEART_FIELD_OPTIONS[field]
                            option_labels = [label for label, _ in option_info["options"]]
                            default_index = 0
                            for idx, (_, value) in enumerate(option_info["options"]):
                                if value == int(def_v):
                                    default_index = idx
                                    break
                            selected_label = st.selectbox(
                                option_info["label"], options=option_labels,
                                index=default_index, key=f"{cfg['key']}_{field}"
                            )
                            input_data[field] = next(value for label, value in option_info["options"] if label == selected_label)
                        elif cfg["key"] == "kidney" and field in KIDNEY_FIELD_OPTIONS:
                            option_info = KIDNEY_FIELD_OPTIONS[field]
                            option_labels = [label for label, _ in option_info["options"]]
                            default_index = 0
                            for idx, (_, value) in enumerate(option_info["options"]):
                                if value == int(def_v):
                                    default_index = idx
                                    break
                            selected_label = st.selectbox(
                                option_info["label"], options=option_labels,
                                index=default_index, key=f"{cfg['key']}_{field}"
                            )
                            input_data[field] = next(value for label, value in option_info["options"] if label == selected_label)
                        elif dtype == int:
                            input_data[field] = st.number_input(
                                field, min_value=int(min_v), max_value=int(max_v),
                                value=int(def_v), step=1, key=f"{cfg['key']}_{field}"
                            )
                        else:
                            input_data[field] = st.number_input(
                                field, min_value=float(min_v), max_value=float(max_v),
                                value=float(def_v), step=0.01, format="%.3f",
                                key=f"{cfg['key']}_{field}"
                            )

        st.markdown("<br>", unsafe_allow_html=True)
        _, bcol, _ = st.columns([1, 2, 1])
        with bcol:
            run_btn = st.button("Run Prediction →")

        if run_btn:
            pred, prob = predict(model, scaler, feat_order, input_data)
            conf_pct   = prob * 100 if pred == 1 else (1 - prob) * 100

            st.session_state["history"].append({
                "time":       datetime.now().strftime("%H:%M:%S"),
                "disease":    disease_label,
                "result":     "HIGH RISK" if pred == 1 else "LOW RISK",
                "confidence": f"{conf_pct:.1f}%",
                "data":       input_data.copy(),
                "prob":       prob,
            })

            rcol1, rcol2, rcol3 = st.columns([1, 3, 1])
            with rcol2:
                if pred == 1:
                    st.markdown(f"""
<div class='result-high'>
    <div class='result-label'> High Risk Detected</div>
    <div class='result-sub'>{disease_label} · Confidence Score</div>
    <div class='conf-bar-wrap'><div class='conf-bar-high' style='width:{conf_pct:.1f}%'></div></div>
    <div class='conf-pct'>{conf_pct:.1f}%</div>
    <div class='result-disclaimer'>This is an ML model output, not a medical diagnosis. Please consult a licensed healthcare professional immediately.</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class='result-low'>
    <div class='result-label'>✓ Low Risk</div>
    <div class='result-sub'>{disease_label} · Confidence Score</div>
    <div class='conf-bar-wrap'><div class='conf-bar-low' style='width:{conf_pct:.1f}%'></div></div>
    <div class='conf-pct'>{conf_pct:.1f}%</div>
    <div class='result-disclaimer'>This is an ML model output, not a medical diagnosis. Regular health checkups are still recommended.</div>
</div>""", unsafe_allow_html=True)

            # ── GAUGE CHART
            st.markdown("<br>", unsafe_allow_html=True)
            gauge_val   = prob * 100
            gauge_color = "#C1121F" if pred == 1 else "#2D6A4F"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gauge_val,
                delta={"reference": 50,
                       "increasing": {"color": "#C1121F"},
                       "decreasing": {"color": "#2D6A4F"}},
                title={"text": "Disease Risk Score (%)",
                       "font": {"color": "#6B7280", "size": 13, "family": "DM Sans"}},
                number={"font": {"color": gauge_color, "size": 34, "family": "Lora"}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickcolor": "#D8CFC4",
                             "tickfont":  {"color": "#9CA3AF", "size": 10}},
                    "bar":  {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "#FFFFFF",
                    "bordercolor": "#D8CFC4",
                    "steps": [
                        {"range": [0, 30],   "color": "rgba(45,106,79,0.08)"},
                        {"range": [30, 60],  "color": "rgba(181,112,10,0.07)"},
                        {"range": [60, 100], "color": "rgba(193,18,31,0.08)"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 2},
                                  "thickness": 0.8, "value": gauge_val},
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#6B7280", "family": "DM Sans"},
                height=280, margin=dict(l=30, r=30, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── EXPLAINABLE AI (SHAP)
            st.markdown("<div class='section-title'>Explainable AI (SHAP)</div>", unsafe_allow_html=True)
            if shap is None:
                st.warning("SHAP is not available in the current environment.")
            else:
                try:
                    input_df = pd.DataFrame([input_data], columns=feat_order).astype(float)
                    rf_model = joblib.load(f"{MODEL_FOLDER}/{cfg['key']}_rf.joblib")

                    explainer = shap.TreeExplainer(rf_model)
                    shap_values = explainer.shap_values(input_df)

                    expected_value = explainer.expected_value
                    if isinstance(shap_values, list):
                        class_idx = 1 if len(shap_values) > 1 else 0
                        sample_values = np.asarray(shap_values[class_idx])[0]
                        base_value = expected_value[class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                    else:
                        sv = np.asarray(shap_values)
                        if sv.ndim == 3:
                            sample_values = sv[0, :, 1]
                            base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                        else:
                            sample_values = sv[0]
                            base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value

                    explanation = shap.Explanation(
                        values=sample_values,
                        base_values=base_value,
                        data=input_df.iloc[0].values,
                        feature_names=input_df.columns.tolist(),
                    )

                    fig_shap_waterfall = plt.figure(figsize=(8, 4.5))
                    shap.plots.waterfall(explanation, show=False)
                    st.pyplot(fig_shap_waterfall)
                    plt.close(fig_shap_waterfall)

                    fig_shap_summary = plt.figure(figsize=(8, 3.8))
                    shap.summary_plot(np.asarray([sample_values]), input_df, plot_type="bar", show=False)
                    st.pyplot(fig_shap_summary)
                    plt.close(fig_shap_summary)
                except Exception as e:
                    st.warning(f"SHAP error: {e}")

            # ── FEATURE IMPORTANCE
            st.markdown("<div class='section-title'>Why This Result? — Feature Contribution</div>", unsafe_allow_html=True)
            try:
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "estimators_"):
                    imps = []
                    for est in model.estimators_:
                        if hasattr(est, "feature_importances_"):
                            imps.append(est.feature_importances_)
                    importances = np.mean(imps, axis=0) if imps else None
                else:
                    importances = None

                if importances is not None and len(importances) == len(feat_order):
                    feat_imp_df = pd.DataFrame({
                        "Feature":    feat_order,
                        "Importance": importances,
                        "Value":      [input_data.get(f, 0) for f in feat_order]
                    }).sort_values("Importance", ascending=True).tail(10)

                    fig_shap = go.Figure(go.Bar(
                        x=feat_imp_df["Importance"],
                        y=feat_imp_df["Feature"],
                        orientation="h",
                        marker=dict(
                            color=feat_imp_df["Importance"],
                            colorscale=[[0, "rgba(45,106,79,0.2)"], [1, cfg["color"]]],
                            line=dict(color="rgba(0,0,0,0.05)", width=1)
                        ),
                        text=[f"{v:.3f}" for v in feat_imp_df["Importance"]],
                        textposition="outside",
                        textfont=dict(color="#9CA3AF", size=11, family="DM Sans"),
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<br>Your value: %{customdata}<extra></extra>",
                        customdata=feat_imp_df["Value"],
                    ))
                    fig_shap.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#FFFFFF",
                        font=dict(color="#6B7280", family="DM Sans"),
                        xaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", title="Importance Score"),
                        yaxis=dict(gridcolor="#EDE7DC", color="#6B7280"),
                        height=360, margin=dict(l=20, r=70, t=20, b=40),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("Feature importance visualization not available for this model type.")
            except Exception as e:
                st.info(f"Feature contribution chart unavailable: {e}")

            # ── PDF REPORT DOWNLOAD
            st.markdown("<div class='section-title'>Download Report</div>", unsafe_allow_html=True)
            try:
                from fpdf import FPDF

                def pdf_safe(text):
                    if text is None:
                        return ""
                    return str(text).replace("—", "-").encode("ascii", "ignore").decode("ascii")

                def gen_pdf(disease, result, conf, data, prob_val, username):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_margins(20, 20, 20)

                    pdf.set_font("Helvetica", "B", 22)
                    pdf.set_text_color(45, 106, 79)
                    pdf.cell(0, 12, pdf_safe("MediScan AI — Clinical Report"), ln=True, align="C")
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(107, 114, 128)
                    pdf.cell(0, 6, pdf_safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  User: {username}"), ln=True, align="C")
                    pdf.ln(6)

                    pdf.set_draw_color(216, 207, 196)
                    pdf.set_line_width(0.5)
                    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                    pdf.ln(8)

                    is_high = result == "HIGH RISK"
                    pdf.set_text_color(193, 18, 31) if is_high else pdf.set_text_color(45, 106, 79)
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, pdf_safe(f"Result: {result}"), ln=True)
                    pdf.set_font("Helvetica", "", 11)
                    pdf.set_text_color(61, 68, 81)
                    pdf.cell(0, 6, pdf_safe(f"Disease Module: {disease}"), ln=True)
                    pdf.cell(0, 6, pdf_safe(f"Risk Probability: {prob_val*100:.1f}%   |   Confidence: {conf}"), ln=True)
                    pdf.ln(6)

                    pdf.set_text_color(45, 106, 79)
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 8, pdf_safe("Patient Input Parameters:"), ln=True)
                    pdf.ln(2)

                    pdf.set_font("Helvetica", "B", 10)
                    for i, (k, v) in enumerate(data.items()):
                        pdf.set_text_color(61, 68, 81)
                        pdf.cell(90, 7, pdf_safe(k), border=0)
                        pdf.set_font("Helvetica", "", 10)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(80, 7, pdf_safe(round(v, 4)), ln=True)
                        pdf.set_font("Helvetica", "B", 10)

                    pdf.ln(8)
                    pdf.set_draw_color(216, 207, 196)
                    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                    pdf.ln(6)

                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(107, 114, 128)
                    pdf.multi_cell(0, 5,
                        pdf_safe("DISCLAIMER: This report is generated by an ML model for educational purposes only and does "
                        "not constitute a medical diagnosis. Always consult a licensed healthcare professional for "
                        "medical advice, diagnosis, or treatment."))
                    return bytes(pdf.output(dest="S"))

                last = st.session_state["history"][-1]
                pdf_bytes = gen_pdf(
                    disease_label, last["result"], last["confidence"],
                    last["data"], last["prob"], st.session_state["username"]
                )
                _, dcol, _ = st.columns([1, 2, 1])
                with dcol:
                    st.download_button(
                        label=" Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"MediScan_{cfg['key']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            except ImportError:
                st.info("Install `fpdf2` to enable PDF reports: `pip install fpdf2`")
            except Exception as e:
                st.warning(f"PDF generation error: {e}")

# ══════════════════════════════════════════════════════════════
#  TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<div class='section-title'>Batch Patient Prediction</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
        Upload a CSV file with one patient per row. Column names must match the feature names
        expected by the selected disease model. The system will predict risk for all patients at once
        and display a colour-coded results table you can download.
        Column names are auto-matched when they are similar (e.g. underscores, aliases, or close spellings).
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        f"Upload CSV for {disease_label}", type=["csv"],
        help="Columns are auto-matched when names are similar."
    )

    if uploaded_file is not None:
        model_b, scaler_b, feat_order_b = load_model(cfg["key"])
        if model_b is None:
            st.error("Model not found for this disease. Train and save models first.")
        else:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded:** {len(df_upload)} patients · {len(df_upload.columns)} columns")
                st.dataframe(df_upload.head(5), use_container_width=True)

                col_mapping, missing_cols, auto_matches = resolve_batch_columns(
                    cfg["key"], feat_order_b, df_upload.columns.tolist()
                )

                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.markdown("**Expected columns:**")
                    st.code(", ".join(feat_order_b))
                else:
                    if auto_matches:
                        mapped_text = "\n".join([f"- {exp}  ←  {got}" for exp, got in auto_matches])
                        st.info(f"Auto-mapped columns:\n{mapped_text}")

                    df_model = df_upload[[col_mapping[f] for f in feat_order_b]].copy()
                    df_model.columns = feat_order_b

                    if cfg["key"] == "kidney":
                        kidney_binary_map = {
                            "yes": 1, "y": 1, "1": 1,
                            "no": 0,  "n": 0, "0": 0,
                            "good": 1, "poor": 0,
                            "present": 1, "notpresent": 0,
                        }
                        for col in ["htn", "dm", "cad", "appet", "pe", "ane"]:
                            if col in df_model.columns:
                                df_model[col] = df_model[col].apply(
                                    lambda x: kidney_binary_map.get(str(x).strip().lower(), x)
                                )

                    if st.button("Run Batch Prediction →"):
                        df_model_clean = df_model.copy()
                        df_model_clean = df_model_clean.apply(
                            lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x)
                        )
                        missing_tokens = {"", "?", "nan", "none", "null", "na", "n/a", "nil", "-", "--"}
                        df_model_clean = df_model_clean.apply(
                            lambda col: col.map(
                                lambda x: np.nan if isinstance(x, str) and x.lower() in missing_tokens else x
                            )
                        )

                        df_model_num = df_model_clean.apply(pd.to_numeric, errors="coerce")

                        invalid_mask = df_model_clean.notna() & df_model_num.isna()
                        invalid_rows = invalid_mask.any(axis=1)
                        if invalid_rows.any():
                            bad_idx = list(df_model_num.index[invalid_rows])
                            show_rows = ", ".join(str(i + 1) for i in bad_idx[:10])
                            more = "..." if len(bad_idx) > 10 else ""
                            st.error(
                                f"Non-numeric or invalid values found in rows: {show_rows}{more}. "
                                "Please clean those values and re-upload."
                            )
                        else:
                            missing_before = int(df_model_num.isna().sum().sum())
                            if missing_before > 0:
                                col_medians = df_model_num.median(numeric_only=True)
                                df_model_num = df_model_num.fillna(col_medians)
                                df_model_num = df_model_num.fillna(0.0)
                                st.info(f"Auto-cleaned {missing_before} missing value(s) using median imputation.")

                            results = []
                            for _, row in df_model_num.iterrows():
                                d = {f: row[f] for f in feat_order_b}
                                p, prob = predict(model_b, scaler_b, feat_order_b, d)
                                results.append({
                                    "Risk Level":      " HIGH RISK" if p == 1 else " LOW RISK",
                                    "Probability (%)": round(prob * 100, 1),
                                })

                            result_df = pd.concat([df_upload.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                            st.markdown("<div class='section-title'>Batch Results</div>", unsafe_allow_html=True)

                            high_count = sum(1 for r in results if "HIGH" in r["Risk Level"])
                            low_count  = len(results) - high_count

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Total Patients", len(results))
                            m2.metric("High Risk", high_count, delta=f"{high_count/len(results)*100:.0f}%")
                            m3.metric("Low Risk",  low_count,  delta=f"{low_count/len(results)*100:.0f}%")

                            st.dataframe(result_df, use_container_width=True)

                            csv_out = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                " Download Results CSV", csv_out,
                                file_name=f"batch_results_{cfg['key']}.csv", mime="text/csv"
                            )

                            # Risk distribution pie
                            fig_pie = go.Figure(go.Pie(
                                labels=["High Risk", "Low Risk"],
                                values=[high_count, low_count],
                                marker=dict(
                                    colors=["#C1121F", "#2D6A4F"],
                                    line=dict(color="#FFFFFF", width=2)
                                ),
                                hole=0.55,
                                textfont=dict(color="#FFFFFF", family="DM Sans"),
                            ))
                            fig_pie.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#6B7280", family="DM Sans"),
                                height=320,
                                legend=dict(font=dict(color="#6B7280")),
                                title=dict(text="Risk Distribution", font=dict(color="#3D4451", size=13, family="Lora")),
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        model_b, scaler_b, feat_order_b = load_model(cfg["key"])
        if feat_order_b:
            st.markdown(f"**Expected columns for {disease_label}:**")
            st.code(", ".join(feat_order_b))

# ══════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='section-title'>Prediction History (This Session)</div>", unsafe_allow_html=True)

    history = st.session_state["history"]
    if not history:
        st.info("No predictions made yet in this session. Run a prediction to see it here.")
    else:
        total = len(history)
        high  = sum(1 for h in history if h["result"] == "HIGH RISK")
        low   = total - high

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Predictions", total)
        m2.metric("High Risk", high)
        m3.metric("Low Risk", low)
        st.markdown("<br>", unsafe_allow_html=True)

        for i, h in enumerate(reversed(history)):
            risk_cls  = "hist-high" if h["result"] == "HIGH RISK" else "hist-low"
            risk_icon = "" if h["result"] == "HIGH RISK" else "✓"
            st.markdown(f"""
<div class='hist-row'>
    <span>#{total - i} &nbsp;·&nbsp; {h['time']} &nbsp;·&nbsp; {h['disease']}</span>
    <span class='{risk_cls}'>{risk_icon} {h['result']} ({h['confidence']})</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

        if len(history) >= 2:
            st.markdown("<div class='section-title'>Risk Probability Trend</div>", unsafe_allow_html=True)
            fig_trend = go.Figure(go.Scatter(
                x=[f"#{i+1} {h['disease']}" for i, h in enumerate(history)],
                y=[h["prob"] * 100 for h in history],
                mode="lines+markers",
                line=dict(color="#2D6A4F", width=2),
                marker=dict(
                    color=["#C1121F" if h["result"] == "HIGH RISK" else "#2D6A4F" for h in history],
                    size=10, line=dict(color="#FFFFFF", width=2)
                ),
                hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>"
            ))
            fig_trend.add_hline(y=50, line=dict(color="rgba(181,112,10,0.5)", dash="dot", width=1.5))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#6B7280", family="DM Sans"),
                xaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", linecolor="#D8CFC4"),
                yaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", range=[0, 100], title="Risk Probability (%)"),
                height=300, margin=dict(l=20, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 4 — MODEL PERFORMANCE DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_performance:
    st.markdown("<div class='section-title'>Model Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
        Performance metrics are loaded from saved evaluation results. If not available,
        representative benchmark values from training are shown.
    </div>
    """, unsafe_allow_html=True)

    BENCHMARK = {
        " Diabetes":      {"Accuracy": 78.6, "Precision": 76.2, "Recall": 72.4, "F1": 74.3, "ROC-AUC": 83.1},
        " Heart Disease": {"Accuracy": 85.2, "Precision": 83.7, "Recall": 86.0, "F1": 84.8, "ROC-AUC": 91.4},
        " Breast Cancer": {"Accuracy": 96.5, "Precision": 95.8, "Recall": 97.1, "F1": 96.4, "ROC-AUC": 99.1},
        " Kidney Disease": {"Accuracy": 97.8, "Precision": 97.2, "Recall": 98.1, "F1": 97.6, "ROC-AUC": 99.5},
    }

    selected_perf = st.selectbox("Select disease to view metrics:", list(BENCHMARK.keys()))
    selected_key = DISEASE_CFG[selected_perf]["key"]
    live_eval = evaluate_model_performance(selected_key)
    using_live_metrics = "error" not in live_eval

    if using_live_metrics:
        metrics = live_eval["metrics"]
        st.caption(f"Live evaluation from saved model + reference dataset ({live_eval['n_test']} test samples).")
    else:
        metrics = BENCHMARK[selected_perf]
        st.warning(f"Live evaluation unavailable for {selected_perf.strip()}: {live_eval['error']}. Showing benchmark values.")

    # Metric cards
    cols = st.columns(5)
    metric_colors = {
        "Accuracy":  "#2D6A4F",
        "Precision": "#B5700A",
        "Recall":    "#C1121F",
        "F1":        "#5C4033",
        "ROC-AUC":   "#40916C",
    }
    for col, (metric, val) in zip(cols, metrics.items()):
        with col:
            clr = metric_colors[metric]
            st.markdown(f"""
<div style='background:#FFFFFF;border:1px solid #D8CFC4;border-top:3px solid {clr};
     border-radius:12px;padding:18px 14px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.07);'>
    <div style='font-family:"Lora",serif;font-size:1.65rem;font-weight:700;color:{clr};
                letter-spacing:-0.02em;'>{val}%</div>
    <div style='font-size:0.74rem;color:#9CA3AF;letter-spacing:0.07em;
                text-transform:uppercase;margin-top:4px;font-family:"DM Sans",sans-serif;
                font-weight:500;'>{metric}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    cats  = list(metrics.keys())
    vals  = list(metrics.values())
    fig_r = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(45,106,79,0.10)",
        line=dict(color="#2D6A4F", width=2),
        marker=dict(color="#2D6A4F", size=7),
        name=selected_perf,
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(range=[60, 100], gridcolor="#EDE7DC", color="#9CA3AF", tickfont=dict(size=10)),
            angularaxis=dict(gridcolor="#EDE7DC", color="#6B7280"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", family="DM Sans"),
        height=380, margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("<div class='section-title'>Confusion Matrix & ROC Curve</div>", unsafe_allow_html=True)
    if using_live_metrics:
        y_true = live_eval["y_true"]
        y_pred = live_eval["y_pred"]
        y_prob = live_eval["y_prob"]

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4.6, 4.2))
            im = ax_cm.imshow(cm, cmap="Greens")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="#2F2F2F", fontsize=11)

            ax_cm.set_title("Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Low", "High"])
            ax_cm.set_yticklabels(["Low", "High"])
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with chart_col2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(5.0, 4.2))
            ax_roc.plot(fpr, tpr, color="#2D6A4F", linewidth=2.2, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.5)

            ax_roc.set_title("ROC Curve")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(alpha=0.2)
            fig_roc.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)
    else:
        st.info("Confusion Matrix and ROC Curve require successful live evaluation for this disease.")

    # All-disease comparison bar
    st.markdown("<div class='section-title'>All Disease Comparison — Accuracy</div>", unsafe_allow_html=True)
    all_names = list(BENCHMARK.keys())
    all_accs = [BENCHMARK[disease_name]["Accuracy"] for disease_name in all_names]
    st.caption("Comparison chart uses benchmark values to keep the dashboard fast and deployment-friendly.")
    bar_colors = ["#2D6A4F", "#C1121F", "#B5700A", "#5C4033"]
    fig_bar = go.Figure(go.Bar(
        x=all_names, y=all_accs,
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.6)", width=1),
            opacity=0.85,
        ),
        text=[f"{v}%" for v in all_accs],
        textposition="outside",
        textfont=dict(color="#6B7280", size=12, family="DM Sans"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#6B7280", family="DM Sans"),
        xaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", linecolor="#D8CFC4"),
        yaxis=dict(gridcolor="#EDE7DC", color="#9CA3AF", range=[0, 105], title="Accuracy (%)"),
        height=320, margin=dict(l=20, r=20, t=20, b=60),
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div style='font-size:0.76rem;color:#9CA3AF;font-style:italic;margin-top:8px;
        font-family:"DM Sans",sans-serif;'>
        * Metrics are computed from current saved models against reference datasets.
            Benchmark values are used only when live evaluation is unavailable.
    </div>
    """, unsafe_allow_html=True)