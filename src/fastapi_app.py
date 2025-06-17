from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("src/housing_model.pkl")

# Define input schema
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to Housing Price Prediction API!"}

@app.post("/predict")
def predict(data: HousingData):
    input_data = np.array([[ 
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]])
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}

