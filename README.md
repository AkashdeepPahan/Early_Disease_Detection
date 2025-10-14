🩺 Early Disease Detection using Machine Learning

This project predicts Diabetes, Heart Disease, and Breast Cancer using machine learning models and provides an interactive Streamlit UI for users.

🚀 Features

🔍 Disease Prediction for Diabetes, Heart Disease, and Breast Cancer

📊 Multiple ML models trained (Logistic Regression, Random Forest, SVM, XGBoost, Neural Network, etc.)

🎨 Stylish UI built with Streamlit

✅ Probability-based prediction (confidence score shown)

💾 Saved models with joblib

📂 Project Structure
Early_Disease_Detection/
├── app.py                  # Streamlit UI
├── models/                 # Saved ML models & scalers
│   ├── diabetes_model.joblib
│   ├── diabetes_scaler.joblib
│   ├── heart_model.joblib
│   ├── heart_scaler.joblib
│   ├── cancer_model.joblib
│   ├── cancer_scaler.joblib
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/AkashdeepPahan/Early_Disease_Detection.git
cd Early_Disease_Detection


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

🌐 Deployment

You can deploy this project on Streamlit Cloud for free:

Push this repo to GitHub

Go to Streamlit Cloud

Select your repo → choose app.py as the entry point

Done ✅ your app will be live with a public link!

📊 Models Used

Logistic Regression

Random Forest

Support Vector Machine (SVM)

Gradient Boosting

K-Nearest Neighbors (KNN)

Neural Network (MLP)

XGBoost

🧪 Sample Inputs

You can use the following values to test the app:

🩸 Diabetes (8 features)
Feature	Example (Normal)
Pregnancies	2
Glucose	110
BloodPressure	70
SkinThickness	25
Insulin	80
BMI	24.5
DiabetesPedigreeFunction	0.35
Age	35
❤️ Heart Disease (13 features)
Feature	Example (Normal)
age	40
sex	1 (Male), 0 (Female)
cp	0 (typical angina)
trtbps	120
chol	200
fbs	0 (false)
restecg	1 (normal)
thalachh	170
exng	0 (no exercise-induced angina)
oldpeak	0.0
slp	2
caa	0
thall	2
🎗️ Breast Cancer (30 features – shortened example)
Feature	Example (Normal)
mean radius	13.5
mean texture	18.0
mean perimeter	85.0
mean area	550
mean smoothness	0.095
mean compactness	0.08
mean concavity	0.05
mean concave points	0.03
mean symmetry	0.18
mean fractal dimension	0.06
radius error	0.4
texture error	1.0
...	...
worst fractal dimension	0.08

👉 These values represent a normal/healthy patient and should usually give a low probability of disease.

📝 Disclaimer

⚠️ This project is for educational purposes only.
It should not be used for actual medical diagnosis.
Always consult with a medical professional for health concerns.

