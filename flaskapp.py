from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = Flask(__name__)

# Load the service account credentials for Firestore
cred_path = "/Users/dee/Desktop/Cloud_Project_Uni/stockpreddb-firebase-adminsdk-lsfi8-c7503dc169.json"  # Full path to the service account JSON
try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firestore initialized successfully.")
except Exception as e:
    print(f"Error initializing Firestore: {e}")
    db = None

# Load the pre-trained stock prediction model
MODEL_PATH = "stock_prediction_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def fetch_stock_data(stock_symbol, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    stock = yf.Ticker(stock_symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

def predict_stock_price(stock_symbol, start_date, end_date):
    """
    Predict stock prices using the pre-trained model.
    """
    try:
        # Fetch stock data
        data = fetch_stock_data(stock_symbol, start_date, end_date)
        if data.empty:
            return {"error": "No stock data available for the given date range."}

        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

        # Prepare the input for the model
        X_test = np.array([scaled_data[-60:]])  # Use the last 60 days of data
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Make predictions
        prediction = model.predict(X_test)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        return {"prediction": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}

def log_prediction_to_firestore(stock_symbol, start_date, end_date, predicted_price):
    """
    Log the prediction data to Firestore.
    """
    if db is None:
        print("Firestore is not initialized. Skipping log.")
        return

    try:
        # Add data to the "predictions" collection
        doc_ref = db.collection("predictions").document()
        doc_ref.set({
            "stock_symbol": stock_symbol,
            "start_date": start_date,
            "end_date": end_date,
            "predicted_price": predicted_price
        })
        print("Prediction logged to Firestore successfully.")
    except Exception as e:
        print(f"Error logging prediction to Firestore: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({"status": "API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict stock prices.
    """
    data = request.json
    stock_symbol = data.get("stock_symbol")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    # Validate inputs
    if not all([stock_symbol, start_date, end_date]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Make prediction
    prediction_result = predict_stock_price(stock_symbol, start_date, end_date)
    if "error" in prediction_result:
        return jsonify(prediction_result), 400

    predicted_price = prediction_result["prediction"]

    # Log to Firestore
    log_prediction_to_firestore(stock_symbol, start_date, end_date, predicted_price)

    return jsonify({
        "stock_symbol": stock_symbol,
        "start_date": start_date,
        "end_date": end_date,
        "predicted_price": predicted_price
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
