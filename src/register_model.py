import mlflow
from mlflow.tracking import MlflowClient
import os

# 🔁 Load run_id dynamically from file (produced by train.py)
run_id_path = "run_id.txt"
if not os.path.exists(run_id_path):
    raise FileNotFoundError("❌ 'run_id.txt' not found. Ensure train.py ran successfully and logged the run ID.")

with open(run_id_path, "r") as f:
    run_id = f.read().strip()

model_name = "HousingPriceModel"
artifact_path = "housingmodel"

client = MlflowClient()

# 🏷️ Register model (creates new version if already exists)
print(f"📦 Registering model from run ID: {run_id}")
model_uri = f"runs:/{run_id}/{artifact_path}"
mv = mlflow.register_model(model_uri, model_name)
version = mv.version

# 📝 Update version description
client.update_model_version(
    name=model_name,
    version=version,
    description="RandomForest v2 trained on June 6, 2025"
)

# 🚦 Move model to "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

print(f"✅ Model '{model_name}' version {version} is now in 'Staging'")

# 📄 List latest model versions in Staging
staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
print("\n📚 Models in 'Staging':")
for mv in staging_versions:
    print(f"🔹 Version {mv.version} | Status: {mv.status} | Description: {mv.description}")
