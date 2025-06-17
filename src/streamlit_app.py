import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("src/housing_model.pkl")

# Page title
st.title("🏡 California Housing Price Predictor")

# Sidebar - User inputs
st.sidebar.header("Input Features")

def user_input_features():
    MedInc = st.sidebar.slider('Median Income (10k)', 0.0, 15.0, 5.0)
    HouseAge = st.sidebar.slider('House Age', 0, 50, 25)
    AveRooms = st.sidebar.slider('Average Rooms', 0.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider('Average Bedrooms', 0.0, 5.0, 1.0)
    Population = st.sidebar.slider('Population', 0.0, 5000.0, 1000.0)
    AveOccup = st.sidebar.slider('Average Occupancy', 0.0, 10.0, 3.0)
    Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 36.0)
    Longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -120.0)

    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Panel
st.subheader("User Input Parameters")
st.write(input_df)

# Predict
prediction = model.predict(input_df)
st.subheader("Predicted Median House Value (in $100,000s)")
st.write(f"💰 {prediction[0]:.2f}")

