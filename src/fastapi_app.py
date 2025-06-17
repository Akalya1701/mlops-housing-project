from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
import joblib
import numpy as np

# ✅ Load the model
model = joblib.load("src/housing_model.pkl")

# ✅ Initialize FastAPI app
app = FastAPI(title="Housing Price Prediction API")

# ✅ Define request schema using Pydantic

@app.get("/")
def read_root():
    return {"message": "Hello World"}
class HousingFeatures(BaseModel):
    features: List[float]

    @field_validator("features")
    def validate_features_length(cls, v):
        if len(v) != 8:
            raise ValueError("Exactly 8 features are required")
        return v

# ✅ Define the POST endpoint
@app.post("/predict")
def predict(data: HousingFeatures):
    try:
        input_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
