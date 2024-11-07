import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodell = pickle.load(open('regmodell.pkl', 'rb'))  # Your trained regression model
scalar = pickle.load(open('scaling.pkl', 'rb'))  # Your fitted scaler (StandardScaler or similar)

@app.route('/')
def home():
    return render_template('home.html')  # Renders the HTML form for prediction

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get the input data
        data = request.json['data']
        print("Received data:", data)

        # Ensure the required features are in the correct order
        required_features = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", 
            "AveOccup", "Latitude", "Longitude"
        ]
        
        # Fill missing features with default values (0.0)
        for feature in required_features:
            if feature not in data:
                data[feature] = 0.0  # Provide a default value if feature is missing
        
        # Prepare the input data for prediction
        input_data = [data[feature] for feature in required_features]
        print("Prepared input data:", input_data)
        
        # Convert to numpy array and reshape for model input
        input_data_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input data
        scaled_input = scalar.transform(input_data_array)
        
        # Make the prediction
        prediction = regmodell.predict(scaled_input)
        print("Prediction result:", prediction[0])
        
        # Return the prediction in a valid JSON format
        return jsonify(prediction=float(prediction[0]))

    except Exception as e:
        # Log the error and return a JSON error message
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while making the prediction. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
