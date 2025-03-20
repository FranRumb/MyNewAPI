from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("RFCmodel.pkl")

# Initialize Flask app
app = Flask(__name__)


@app.route("/")
def home():
    return "Machine Learning API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get input data as JSON
    features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
    prediction = model.predict(features).tolist()  # Predict
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
