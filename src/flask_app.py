#  in this file we are going to create flask app to serve our model to predict client values

from flask import Flask, request, jsonify
import joblib
import numpy as np

# 1. Create Flask app
app = Flask(__name__)

# 2. Load the trained model (from local file)
model = joblib.load("src/housing_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is working!"})

# 3. Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        inputs = np.array(data["inputs"])

        # Make predictions
        predictions = model.predict(inputs).tolist()

        # Return as JSON
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)})

# 4. Run app
if __name__ == "__main__":
    app.run(port=5000, debug=True)
