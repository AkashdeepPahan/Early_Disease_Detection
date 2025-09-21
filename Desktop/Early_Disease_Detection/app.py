import streamlit as st
import joblib
import os

# ================================
# Page Config & Styling
# ================================
st.set_page_config(page_title="AI Health Report", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f9fafb;
        color: #111827;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 1.1em;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        transform: scale(1.02);
    }
    .report-box {
        padding: 1.5em;
        border-radius: 12px;
        margin-top: 1.5em;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .positive {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 2px solid #b91c1c;
    }
    .negative {
        background-color: #dcfce7;
        color: #166534;
        border: 2px solid #166534;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# Feature Orders
# ================================
FEATURE_ORDERS = {
    "diabetes": [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"
    ],
    "heart": [
        "age","sex","cp","trtbps","chol","fbs","restecg",
        "thalachh","exng","oldpeak","slp","caa","thall"
    ],
    "cancer": [
        "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
        "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
        "radius error","texture error","perimeter error","area error","smoothness error",
        "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
        "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
        "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
    ]
}

def reorder_features(disease_key, user_inputs_dict):
    order = FEATURE_ORDERS[disease_key]
    return [user_inputs_dict[f] for f in order]

# ================================
# Model Loader & Prediction
# ================================
PROJECT_DIR = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(PROJECT_DIR, "models")

def load_model(model_name):
    model = joblib.load(os.path.join(MODEL_FOLDER, f"{model_name}_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_FOLDER, f"{model_name}_scaler.joblib"))
    return model, scaler

def predict_patient(model_name, features):
    model, scaler = load_model(model_name)
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    return pred, prob

# ================================
# Sidebar
# ================================
st.sidebar.title("⚙️ Settings")
disease = st.sidebar.selectbox(
    "Choose Disease:",
    ["Diabetes 🩸", "Heart ❤️", "Breast Cancer 🎗️"]
)

# Mapping display name → model name
disease_map = {
    "Diabetes 🩸": "diabetes",
    "Heart ❤️": "heart",
    "Breast Cancer 🎗️": "cancer"
}
model_key = disease_map[disease]

# ================================
# Main Form
# ================================
st.title("🩺 AI-Powered Health Report")
st.write(f"Provide patient data for **{disease}** prediction.")

with st.form("patient_form"):
    user_inputs = {}

    if model_key == "diabetes":
        user_inputs["Pregnancies"] = st.number_input("Pregnancies", min_value=0, step=1)
        user_inputs["Glucose"] = st.number_input("Glucose", min_value=0, step=1)
        user_inputs["BloodPressure"] = st.number_input("Blood Pressure", min_value=0, step=1)
        user_inputs["SkinThickness"] = st.number_input("Skin Thickness", min_value=0, step=1)
        user_inputs["Insulin"] = st.number_input("Insulin", min_value=0, step=1)
        user_inputs["BMI"] = st.number_input("BMI", min_value=0.0, step=0.1)
        user_inputs["DiabetesPedigreeFunction"] = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
        user_inputs["Age"] = st.number_input("Age", min_value=0, step=1)

    elif model_key == "heart":
        user_inputs["age"] = st.number_input("Age", min_value=0, step=1)
        user_inputs["sex"] = st.selectbox("Sex (0=female,1=male)", [0,1])  # 0=female,1=male
        user_inputs["cp"] = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
        user_inputs["trtbps"] = st.number_input("Resting Blood Pressure", min_value=0, step=1)
        user_inputs["chol"] = st.number_input("Cholesterol", min_value=0, step=1)
        user_inputs["fbs"] = st.selectbox("Fasting Blood Sugar >120 (1=yes,0=no)", [0,1])
        user_inputs["restecg"] = st.selectbox("Resting ECG", [0,1,2])
        user_inputs["thalachh"] = st.number_input("Max Heart Rate Achieved", min_value=0, step=1)
        user_inputs["exng"] = st.selectbox("Exercise Induced Angina (1=yes,0=no)", [0,1])
        user_inputs["oldpeak"] = st.number_input("Oldpeak", value=0.0, step=0.1)
        user_inputs["slp"] = st.selectbox("Slope (slp)", [0,1,2])
        user_inputs["caa"] = st.number_input("Number of Major Vessels (caa)", min_value=0, max_value=4, step=1)
        user_inputs["thall"] = st.selectbox("Thalassemia (thall)", [0,1,2,3])

    elif model_key == "cancer":
        st.info("📝 Please enter values for 30 features of Breast Cancer dataset.")
        for f in FEATURE_ORDERS["cancer"]:
            user_inputs[f] = st.number_input(f, value=1.0, step=0.1)

    submitted = st.form_submit_button("🔍 Generate Report")

# ================================
# Prediction & Output
# ================================
if submitted:
    features = reorder_features(model_key, user_inputs)
    pred, prob = predict_patient(model_key, features)

    if model_key == "diabetes":
        diagnosis = "Diabetes" if pred == 1 else "No Diabetes"
    elif model_key == "heart":
        diagnosis = "Heart Disease" if pred == 1 else "Healthy"
    else:
        diagnosis = "Malignant Cancer" if pred == 1 else "Benign"

    st.markdown("---")
    if pred == 1:
        st.markdown(f"<div class='report-box positive'>⚠️ {diagnosis}<br>Probability: {prob:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='report-box negative'>✅ {diagnosis}<br>Probability: {prob:.2f}</div>", unsafe_allow_html=True)
