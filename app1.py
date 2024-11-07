import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodell = pickle.load(open('regmodell.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Housing Prediction API"

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Received data:", data)
    
    # Ensure the required features are in the correct order and provide default values if missing
    required_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", 
        "AveOccup", "Latitude", "Longitude"
    ]
    
    # Fill missing features with default values (for example, 0 or mean values)
    # If a feature is missing, you can set it to 0 or other reasonable defaults based on your dataset
    for feature in required_features:
        if feature not in data:
            data[feature] = 0.0  # Set a default value like 0 for missing features
    
    # Ensure the data is in the correct order
    input_data = [data[feature] for feature in required_features]
    
    print("Prepared input data:", input_data)
    
    # Convert to numpy array and reshape for model input
    input_data_array = np.array(input_data).reshape(1, -1)
    
    # Scale the input data
    scaled_input = scalar.transform(input_data_array)
    
    # Make a prediction
    prediction = regmodell.predict(scaled_input)
    print("Prediction result:", prediction[0])
    
    # Return the prediction as a response
    return jsonify(prediction=float(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
