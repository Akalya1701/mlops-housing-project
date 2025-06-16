# add_model_metadata.py

import mlflow
from mlflow import MlflowClient

client = MlflowClient()

model_name = "HousingPriceModel"
version = client.get_latest_versions(model_name, stages=["Production"])[0].version
# Add description
client.update_registered_model(
    name=model_name,
    description="🏠 Predicts house prices in California based on 8 features"
)

# Add tags
client.set_model_version_tag(name=model_name, version=version, key="environment", value="production")
client.set_model_version_tag(name=model_name, version=version, key="owner", value="Akalya")
client.set_model_version_tag(name=model_name, version=version, key="model_type", value="regression")
client.set_model_version_tag(
    name=model_name,
    version=version,
    key="approved_by",
    value="akalya"
)
client.update_model_version(
    name=model_name,
    version=version,
    description="""
✅ Approved for production
🔍 Reviewed by: Akalya
📅 Date: 2025-06-16
📝 Notes: Model shows stable RMSE over test sets and no fairness violations detected.
"""
)