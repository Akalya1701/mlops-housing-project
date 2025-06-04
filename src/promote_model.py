import mlflow
from mlflow.tracking import MlflowClient

model_name = "HousingPriceModel"

client = MlflowClient()

# Get latest version in Staging
staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

if not staging_versions:
    print("❌ No model in Staging to promote.")
else:
    staging_model = staging_versions[0]
    version = staging_model.version
    print(f"🔍 Found model in Staging: version {version}")

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True  # Archive old Production versions automatically
    )

    print(f"🚀 Model {model_name} version {version} has been promoted to Production ✅")
  
