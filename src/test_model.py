#  in this file we are going to test the registered model by giving inputs

import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

run_id = "a05f0909bfa04fc2b8fc40b691be8254"
model_name = "HousingPriceModel"

def validate_model(model):
    # Load some test data (or synthetic data)
    data = fetch_california_housing(as_frame=True)
    X = data.data

    # Predict on first 5 rows
    preds = model.predict(X.head(5))
    
    # Simple check: predictions should be numeric and non-negative
    assert isinstance(preds, np.ndarray), "Predictions should be a NumPy array"
    assert all(preds >= 0), "Some predictions are negative!"

    print("✅ Model passed validation tests.")
    return True

if __name__ == "__main__":
    model_name = "HousingPriceModel"

    # ✅ Load model from registry (Staging stage)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Staging")


    if validate_model(model):
        print("🚀 Ready to transition model to Production manually or via API.")
        
