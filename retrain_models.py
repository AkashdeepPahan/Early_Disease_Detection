"""
retrain_models.py
─────────────────────────────────────────────────────────────────────────────
Re-trains and saves all four MediScan AI models (diabetes, heart, cancer,
kidney) using your CURRENT installed scikit-learn version, fixing the
  AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
error that appears when models saved with an older sklearn are loaded by a
newer one.

Usage
-----
    python retrain_models.py

The script downloads the same datasets used in your notebook and saves the
.joblib files to  ./models/  (same folder app.py expects).

Requirements
------------
    pip install pandas numpy scikit-learn joblib xgboost requests
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import requests
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore")
np.random.seed(42)

MODEL_FOLDER = "./models"
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def fetch_csv(url, **kwargs):
    """Download a CSV from a URL and return a DataFrame."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), **kwargs)


def train_and_save(df, target_column, model_name):
    """
    Full pipeline: split → scale → tune → ensemble → save.
    Mirrors the notebook's train_and_evaluate() exactly.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = np.asarray(scaler.fit_transform(X_train).astype(np.float32))
    X_test_scaled  = np.asarray(scaler.transform(X_test).astype(np.float32))

    # ── Base estimators ──────────────────────────────────────
    log_reg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
        # NOTE: 'multi_class' parameter was removed in scikit-learn ≥1.5
        #       Do NOT pass it here — that's exactly what caused the error.
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42
    )
    svm = SVC(probability=True, class_weight="balanced", random_state=42)
    gb  = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25), max_iter=1000,
        early_stopping=True, validation_fraction=0.1, random_state=42
    )

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=1, eval_metric="logloss",
            random_state=42, use_label_encoder=False
        )
    except ImportError:
        xgb = None
        print("  XGBoost not installed — skipping.")

    # ── Grid-search tuning ───────────────────────────────────
    print("  Tuning Random Forest ...")
    rf_grid = GridSearchCV(
        rf,
        {"n_estimators": [150, 250, 350],
         "max_depth": [10, 15, 20],
         "min_samples_split": [2, 4, 6],
         "min_samples_leaf": [1, 2, 3],
         "max_features": ["sqrt", "log2"]},
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    print(f"    Best RF params: {rf_grid.best_params_}")

    print("  Tuning SVM ...")
    svm_grid = GridSearchCV(
        svm,
        {"C": [0.01, 0.1, 1, 10, 100],
         "kernel": ["linear", "rbf", "poly"],
         "gamma": ["scale", "auto"]},
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    svm_grid.fit(X_train_scaled, y_train)
    svm_best = svm_grid.best_estimator_
    print(f"    Best SVM params: {svm_grid.best_params_}")

    if xgb is not None:
        print("  Tuning XGBoost ...")
        xgb_grid = GridSearchCV(
            xgb,
            {"n_estimators": [100, 200, 300],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.05, 0.1],
             "subsample": [0.7, 0.9]},
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        xgb_best = xgb_grid.best_estimator_
        print(f"    Best XGB params: {xgb_grid.best_params_}")
    else:
        xgb_best = None

    # ── Train all individual models ──────────────────────────
    models_to_train = [
        ("Logistic Regression",  log_reg,  X_train_scaled, X_test_scaled),
        ("Random Forest (Tuned)", rf_best,  X_train,        X_test),
        ("SVM (Tuned)",           svm_best, X_train_scaled, X_test_scaled),
        ("Gradient Boosting",     gb,       X_train,        X_test),
        ("KNN",                   knn,      X_train_scaled, X_test_scaled),
        ("Neural Network",        mlp,      X_train_scaled, X_test_scaled),
    ]
    if xgb_best is not None:
        models_to_train.append(("XGBoost (Tuned)", xgb_best, X_train, X_test))

    trained = {}
    for name, m, Xtr, Xte in models_to_train:
        m.fit(Xtr, y_train)
        y_pred  = m.predict(Xte)
        y_proba = m.predict_proba(Xte)[:, 1]
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_proba)
        print(f"  {name:<28}  Acc={acc:.4f}  AUC={auc:.4f}")
        trained[name] = m

    # ── Voting ensemble ──────────────────────────────────────
    print("\n  Building Voting Ensemble ...")
    estimators = list(trained.items())
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train_scaled, y_train)

    y_pred  = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    print(f"  Ensemble                       Acc={acc:.4f}  AUC={auc:.4f}")

    # ── Save ─────────────────────────────────────────────────
    joblib.dump(ensemble, f"{MODEL_FOLDER}/{model_name}_model.joblib")
    joblib.dump(rf_best,  f"{MODEL_FOLDER}/{model_name}_rf.joblib")
    joblib.dump(scaler,   f"{MODEL_FOLDER}/{model_name}_scaler.joblib")
    joblib.dump(X.columns.tolist(), f"{MODEL_FOLDER}/{model_name}_features.joblib")
    print(f"\n  Done! Saved  ./models/{model_name}_model.joblib")
    print(f"  Done! Saved  ./models/{model_name}_scaler.joblib")
    print(f"  Done! Saved  ./models/{model_name}_features.joblib")


# ─────────────────────────────────────────────────────────────
#  1. DIABETES
# ─────────────────────────────────────────────────────────────
print("\n>>> Loading Diabetes dataset ...")
df_diabetes = fetch_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    names=["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
)
# Replace biologically impossible 0s with NaN then impute
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df_diabetes[zero_cols] = df_diabetes[zero_cols].replace(0, np.nan)
imp = SimpleImputer(strategy="median")
df_diabetes[zero_cols] = imp.fit_transform(df_diabetes[zero_cols])
train_and_save(df_diabetes, "Outcome", "diabetes")

# ─────────────────────────────────────────────────────────────
#  2. HEART DISEASE
# ─────────────────────────────────────────────────────────────
print("\n>>> Loading Heart Disease dataset ...")
df_heart = fetch_csv(
    "https://gist.githubusercontent.com/trantuyen082001/1fc2f5c0ad1507f40e721e6d18b34138/raw/heart.csv"
)
X_h = df_heart.drop("output", axis=1)
y_h = df_heart["output"]
X_h = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_h),
    columns=X_h.columns
)
df_heart = pd.concat([X_h, y_h.reset_index(drop=True)], axis=1)
train_and_save(df_heart, "output", "heart")

# ─────────────────────────────────────────────────────────────
#  3. BREAST CANCER
# ─────────────────────────────────────────────────────────────
print("\n>>> Loading Breast Cancer dataset (sklearn built-in) ...")
bc = load_breast_cancer()
df_cancer = pd.DataFrame(bc.data, columns=bc.feature_names)
df_cancer["target"] = bc.target
X_c = df_cancer.drop("target", axis=1)
y_c = df_cancer["target"]
X_c = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_c),
    columns=X_c.columns
)
df_cancer = pd.concat([X_c, y_c.reset_index(drop=True)], axis=1)
train_and_save(df_cancer, "target", "cancer")

# ─────────────────────────────────────────────────────────────
#  4. KIDNEY DISEASE
# ─────────────────────────────────────────────────────────────
print("\n>>> Loading Kidney Disease dataset ...")

KIDNEY_URLS = [
    "https://raw.githubusercontent.com/patilgirish815/Kidney_Cancer_Prediction_Using_Machine_Learning/main/dataset/kidney_disease.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/kidney_disease.csv",
    "https://raw.githubusercontent.com/vijayasri-m/Chronic-Kidney-Disease-Prediction/main/kidney_disease.csv",
]

df_kidney = None
for url in KIDNEY_URLS:
    try:
        df_kidney = fetch_csv(url)
        print(f"  Loaded from: {url}")
        break
    except Exception as e:
        print(f"  Failed ({url}): {e}")

if df_kidney is not None:
    # Drop id column if present
    if "id" in df_kidney.columns:
        df_kidney = df_kidney.drop("id", axis=1)

    # Map target to binary
    if df_kidney["classification"].dtype == object:
        df_kidney["classification"] = (
            df_kidney["classification"]
            .str.strip()
            .map({"ckd": 1, "notckd": 0, "ckd\t": 1})
        )
    df_kidney = df_kidney.rename(columns={"classification": "class"})

    # Binary string columns → numeric
    binary_map = {
        "yes": 1, "no": 0, "good": 1, "poor": 0,
        "present": 1, "notpresent": 0, "normal": 0, "abnormal": 1,
        "\tyes": 1, "\tno": 0, " yes": 1, " no": 0,
        "ckd": 1, "notckd": 0,
    }
    for col in df_kidney.columns:
        s = df_kidney[col].astype(str).str.strip().str.lower()
        mapped = s.map(binary_map)
        df_kidney[col] = pd.to_numeric(mapped.fillna(s), errors="coerce")

    X_k = df_kidney.drop("class", axis=1)
    y_k = df_kidney["class"]

    X_k = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_k),
        columns=X_k.columns
    )
    df_kidney = pd.concat([X_k, y_k.reset_index(drop=True)], axis=1)
    df_kidney = df_kidney.dropna(subset=["class"]).copy()
    df_kidney["class"] = df_kidney["class"].astype(int)

    train_and_save(df_kidney, "class", "kidney")
else:
    print("\n  Warning: Could not load Kidney dataset — skipping.")
    print("    Download kidney_disease.csv manually and place it next to this script,")
    print("    then re-run to train the kidney model.")

print("\n" + "="*60)
print("  All done! Models saved to ./models/")
print("  You can now run:  streamlit run app.py")
print("="*60)