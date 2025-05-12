from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for frontend interaction

# Load model and scaler
MODEL_PATH = "model/diabetes_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at path: {MODEL_PATH}")

model, scaler = joblib.load(MODEL_PATH)


@app.route('/')
def home():
    return "✅ Diabetes Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("features")
        if not data or len(data) != 8:
            return jsonify({"error": "Invalid input. Please provide 8 features."}), 400

        # Reshape and scale
        input_data = np.array(data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
