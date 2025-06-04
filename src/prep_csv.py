# in this file we are going to create a csv file which contains new data for prediction and monitorinbg the model performance

import pandas as pd

data = {
    "MedInc": [8.3252, 5.6431, 6.7890, 4.2154, 7.1234],
    "HouseAge": [41, 28, 15, 30, 25],
    "AveRooms": [6.9841, 6.2381, 5.7210, 4.3210, 6.5123],
    "AveBedrms": [1.0238, 1.1321, 1.1050, 0.9500, 1.2345],
    "Population": [322, 430, 380, 290, 410],
    "AveOccup": [2.5556, 3.1234, 2.9850, 2.7450, 2.8900],
    "Latitude": [37.88, 37.49, 37.65, 37.70, 37.72],
    "Longitude": [-122.23, -121.98, -122.00, -121.95, -122.10],
    "Target": [4.526, 3.912, 3.750, 2.890, 4.120]
}

df = pd.DataFrame(data)
df.to_csv("data/new_data.csv", index=False)

