#  in this file we are going to log the model performace 
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np

def load_model(stage="Production", model_name="HousingPriceModel"):
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)

def monitor_model():
    df = pd.read_csv("data/new_data.csv")

    X = df.drop("Target", axis=1)
    y_true = df["Target"]

    model = load_model()
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"🔍 RMSE on new data: {rmse:.4f}")

    # Log metrics and params to MLflow
    with mlflow.start_run(run_name="monitoring_eval"):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("model_stage", "Production")
        mlflow.set_tag("monitoring", "true")


if __name__ == "__main__":
    monitor_model()

    