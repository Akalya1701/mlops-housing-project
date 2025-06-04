# src/predict.py
import joblib
import pandas as pd

def predict(input_data):
    model = joblib.load("src/housing_model.pkl")
    df = pd.DataFrame(input_data)
    return model.predict(df)

def main():
    sample = [[8.3, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]]
    prediction = predict(sample)
    print("Prediction:", prediction)

