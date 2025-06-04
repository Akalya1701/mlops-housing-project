# in this file we are going to register our model in mlflow registry to explore more about staging,tags and versions

import mlflow
from mlflow.tracking import MlflowClient

run_id = "a05f0909bfa04fc2b8fc40b691be8254"  # use your actual run_id after logging
model_name = "HousingPriceModel"

client = MlflowClient()

# Register model from the run's artifact path "model"
mv = mlflow.register_model(f"runs:/{run_id}/housingmodel", model_name)
version = mv.version

# Update version description (optional)
client.update_model_version(
    name=model_name,
    version=version,
    description="RandomForest v2 trained on May-26-2025 data"
)

# Transition version to "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

print(f"Model {model_name} version {version} is now in Staging ✅")

# Check latest versions in "Staging"
model_versions = client.get_latest_versions(model_name, stages=["Staging"])
print(model_versions)
