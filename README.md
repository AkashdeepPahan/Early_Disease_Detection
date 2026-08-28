 Early Disease Detection using Machine Learning
 Live Demo

 Try it here: ["Early Disease Detection App" on my streamlit profile  ](https://share.streamlit.io/user/akashdeeppahan)


 Overview

The Early Disease Detection system uses machine learning models to predict the likelihood of four major diseases — Diabetes, Heart Disease, Breast Cancer, and Kidney Disease — based on key medical parameters.
This app empowers users and healthcare professionals with early insights to support preventive diagnosis and timely medical care.

 Features

Predicts the risk of:

Diabetes

Heart Disease

Breast Cancer

Kidney Disease

 Clean, aesthetic, and responsive Streamlit UI
 Real-time predictions from trained ML models
 Ensemble approach for higher accuracy
 Ready for both local and cloud deployment

 Technologies Used

Python 3.11+

Streamlit – User interface and hosting

Scikit-learn – Model training and evaluation

XGBoost – Gradient boosting for performance

Pandas, NumPy – Data preprocessing

Joblib – Model persistence

 Models Used

Each disease uses multiple ML models including Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, and XGBoost.
The best-performing models are stored in the /models/ directory.

The project also includes:

generate_report.py for creating the project report

retrain_models.py for rebuilding the saved model artifacts

models/
├── diabetes_model.joblib
├── diabetes_scaler.joblib
├── heart_model.joblib
├── heart_scaler.joblib
├── cancer_model.joblib
├── cancer_scaler.joblib
├── kidney_model.joblib
├── kidney_scaler.joblib

 How to Run Locally

Clone the repository

git clone https://github.com/AkashdeepPahan/Early_Disease_Detection.git
cd Early_Disease_Detection


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

 Interface Preview

 A minimal and elegant interface that allows users to:

Input their medical parameters

Select disease type

Instantly receive prediction results with probability scores


 Links

 GitHub Repository: https://github.com/AkashdeepPahan/Early_Disease_Detection


 Contact

 Akashdeep Pahan
 GitHub: https://github.com/AkashdeepPahan

 Early Disease Detection Web App

 If you found this project useful, please consider starring the repository on GitHub — it helps a lot!

 Deployment Notes

 This project is ready to deploy as a Streamlit app.

 Recommended setup:

 Use [app.py](app.py) as the Streamlit entrypoint.

 Keep the full /models folder in the deployment repository.

 Keep [requirements.txt](requirements.txt) at the project root.

 Keep [.streamlit/config.toml](.streamlit/config.toml) for consistent theme and server settings.

 Demo login credentials:

 admin / admin123

 doctor / mediscan

 guest / guest

 For Streamlit Cloud:

 Set the main file to app.py.

 Make sure all .joblib model files are included.

 If a network call fails in the performance tab, the app will fall back to benchmark values.