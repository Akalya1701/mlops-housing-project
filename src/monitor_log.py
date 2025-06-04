#  in this file we are going to log the model performace 
# We’ll log the RMSE and timestamp to a simple CSV or JSON file for record-keeping. 
import json
from datetime import datetime

log_data = {
    "timestamp": datetime.now().isoformat(),
    "rmse": 0.6563  # Replace with your actual RMSE variable if dynamic
}

with open("monitoring_logs.json", "a") as f:
    f.write(json.dumps(log_data) + "\n")

THRESHOLD = 1.0

if log_data["rmse"] > THRESHOLD:
    print("⚠️  ALERT: RMSE exceeded threshold! Investigate the model.")
else:
    print("✅ Model RMSE is within the acceptable range.")
