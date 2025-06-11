from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from fairlearn.metrics import MetricFrame
import mlflow
import mlflow.sklearn
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

# 📥 Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# ✅ Add sensitive feature: income group (0 = low, 1 = high)
df["income_group"] = (df["MedInc"] > 3.5).astype(int)

# 💾 Save dataset locally
os.makedirs("data", exist_ok=True)
df.to_csv("data/housing.csv", index=False)
print("✅ housing.csv has been saved in the 'data' folder.")

# 📊 Prepare features and labels
X = df.drop(columns=["MedHouseVal", "income_group"])
y = df["MedHouseVal"]
sensitive_feature = df["income_group"]

# 🔀 Split into train/test (keeping sensitive info)
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42
)

# 🤖 Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📈 Predict
y_pred = model.predict(X_test)

# ✅ Manual Fairness Evaluation
mae_low = mean_absolute_error(y_test[sens_test == 0], y_pred[sens_test == 0])
mae_high = mean_absolute_error(y_test[sens_test == 1], y_pred[sens_test == 1])
mae_gap = abs(mae_high - mae_low)

print(f"MAE (Low income): {mae_low:.2f}")
print(f"MAE (High income): {mae_high:.2f}")
print(f"MAE Gap (High - Low): {mae_gap:.2f}")

# ✅ Fairlearn MetricFrame Evaluation
metric_frame = MetricFrame(
    metrics=mean_absolute_error,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

print("\n📊 Fairness Report (MAE by income group):")
print(metric_frame.by_group)

print("\n📉 Performance disparity (MAE difference between groups):")
print(metric_frame.difference())

# plot

metric_frame.by_group.plot(kind="bar")
plt.title("MAE by Income Group")
plt.ylabel("Mean Absolute Error")
plt.show()


# 🚀 Log to MLflow
with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.sklearn.log_model(
        model,
        artifact_path="housingmodel",
        registered_model_name="HousingPriceModel"
    )
    mlflow.log_params(model.get_params())
    mlflow.log_metric("test_r2_score", model.score(X_test, y_test))
    mlflow.log_metric("mae_low_income", mae_low)
    mlflow.log_metric("mae_high_income", mae_high)
    mlflow.log_metric("mae_gap", mae_gap)
    mlflow.log_metric("mae_gap_fairlearn", metric_frame.difference())

    print(f"\n📝 MLflow Run ID: {run_id}")

    # 💾 Save run ID to file for use in CI/CD
    with open("run_id.txt", "w") as f:
        f.write(run_id)

# 💽 Save model locally
os.makedirs("src", exist_ok=True)
joblib.dump(model, "src/housing_model.pkl")
print("✅ Model saved locally as src/housing_model.pkl")
