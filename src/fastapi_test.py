import requests

url = "http://127.0.0.1:8000/predict"
data = {
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}

response = requests.post(url, json=data)
print("Prediction:", response.json())

