# in this file we are going to tarin our very first model to predict house price

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib
import os

# Load dataset
data = fetch_california_housing(as_frame=True)

# Save dataset as CSV locally
df = data.frame
os.makedirs("data", exist_ok=True)
df.to_csv("data/housing.csv", index=False)
print("✅ housing.csv has been saved in the 'data' folder.")

X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Start MLflow run and log model + params + metric
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        model, 
        artifact_path="housingmodel",           # This artifact path must be consistent
        registered_model_name="HousingPriceModel"
    )
    mlflow.log_params(model.get_params())
    test_score = model.score(X_test, y_test)
    mlflow.log_metric("test_r2_score", test_score)

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"Test R2 score: {test_score:.4f}")

# Save a local copy using joblib (optional)
os.makedirs("src", exist_ok=True)
joblib.dump(model, "src/housing_model.pkl")
print("✅ Model saved locally as src/housing_model.pkl")
