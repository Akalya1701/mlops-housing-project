# in this file we are going to test our flask api by posting data

from sklearn.datasets import fetch_california_housing
import requests

# Load the housing dataset
data = fetch_california_housing(as_frame=True)
X = data.data

# Select 1 or 2 samples
sample = X.iloc[[0, 1]].values.tolist()

# Correct key name to match what the Flask app expects
payload = {"inputs": sample}
headers = {"Content-Type": "application/json"}

# Make the POST request
url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json=payload)

# Print the prediction response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

