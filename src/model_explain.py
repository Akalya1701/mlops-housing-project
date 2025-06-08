import shap
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load model
model = joblib.load("src/housing_model.pkl")

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data

# Sample 100 rows for SHAP
X_sample = X.sample(100, random_state=42)

# Explain with SHAP
explainer = shap.Explainer(model.predict, X_sample)
shap_values = explainer(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample)

