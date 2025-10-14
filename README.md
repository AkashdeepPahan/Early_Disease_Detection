🩺 Early Disease Detection using Machine Learning
🌐 Live Demo

👉 Try it here: ["Early Disease Detection App" on my streamlit profile  ](https://share.streamlit.io/user/akashdeeppahan)


📖 Overview

The Early Disease Detection system uses machine learning models to predict the likelihood of three major diseases — Diabetes, Heart Disease, and Breast Cancer — based on key medical parameters.
This app empowers users and healthcare professionals with early insights to support preventive diagnosis and timely medical care.

⚙️ Features

✅ Predicts the risk of:

🩸 Diabetes

❤️ Heart Disease

🎗️ Breast Cancer

✅ Clean, aesthetic, and responsive Streamlit UI
✅ Real-time predictions from trained ML models
✅ Ensemble approach for higher accuracy
✅ Ready for both local and cloud deployment

🧠 Technologies Used

Python 3.11+

Streamlit – User interface and hosting

Scikit-learn – Model training and evaluation

XGBoost – Gradient boosting for performance

Pandas, NumPy – Data preprocessing

Joblib – Model persistence

🧩 Models Used

Each disease uses multiple ML models including Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, and XGBoost.
The best-performing models are stored in the /models/ directory.

models/
├── diabetes_model.joblib
├── diabetes_scaler.joblib
├── heart_model.joblib
├── heart_scaler.joblib
├── cancer_model.joblib
├── cancer_scaler.joblib

🚀 How to Run Locally

Clone the repository

git clone https://github.com/AkashdeepPahan/Early_Disease_Detection.git
cd Early_Disease_Detection


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

🖥️ Interface Preview

✨ A minimal and elegant interface that allows users to:

Input their medical parameters

Select disease type

Instantly receive prediction results with probability scores


🔗 Links

💻 GitHub Repository: https://github.com/AkashdeepPahan/Early_Disease_Detection


📬 Contact

👤 Akashdeep Pahan
🌐 GitHub: https://github.com/AkashdeepPahan

🌐 Early Disease Detection Web App

⭐ If you found this project useful, please consider starring the repository on GitHub — it helps a lot!