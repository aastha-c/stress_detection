"""
Script to connect to Phyphox app on a smartphone, fetch accelerometer data, compute activity level, and send to stress detection pipeline.

Requirements:
- requests
- numpy
- time

Instructions:
- Ensure your phone and computer are on the same WiFi network.
- In the Phyphox app, enable remote access and note the IP address and port (default: 8080).
- Update the PHYPHOX_URL variable below with your phone's IP if needed.
"""



import requests
import numpy as np
import time

import argparse
from model.model_loader import load_model
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# LSTM model import (defer torch import)
def try_import_lstm():
    try:
        import importlib
        torch = importlib.import_module('torch')
        StressLSTM = getattr(importlib.import_module('src.model'), 'StressLSTM')
        return torch, StressLSTM
    except ImportError:
        return None, None

# URL of the Phyphox remote access API (update with your phone's IP if needed)
PHYPHOX_URL = "http://192.168.1.100:8080/"  # Example IP, change as needed
ACC_ENDPOINT = PHYPHOX_URL + "get?acceleration"

# Function to fetch accelerometer data from Phyphox
def fetch_accelerometer_data():
    try:
        response = requests.get(ACC_ENDPOINT, timeout=2)
        response.raise_for_status()
        data = response.json()
        # Extract latest values for accX, accY, accZ
        accX = data["buffer"]["accX"]["buffer"][-1]
        accY = data["buffer"]["accY"]["buffer"][-1]
        accZ = data["buffer"]["accZ"]["buffer"][-1]
        return accX, accY, accZ
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

# Function to compute activity level (magnitude of acceleration vector)
def compute_activity_level(accX, accY, accZ):
    vec = np.array([accX, accY, accZ])
    magnitude = np.linalg.norm(vec)
    return magnitude

# Dummy function to send data to stress detection pipeline


# --- AI Inference: Model Selection ---
def get_model(algo):
    if algo == "randomforest":
        model, feature_names = load_model()
        return model, feature_names, "sklearn"
    elif algo == "logreg":
        model = LogisticRegression()
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        return model, ["accX", "accY", "accZ", "activity_level"], "sklearn"
    elif algo == "xgboost" and xgb_available:
        model = XGBClassifier()
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        return model, ["accX", "accY", "accZ", "activity_level"], "sklearn"
    elif algo == "lstm":
        torch, StressLSTM = try_import_lstm()
        if torch is None or StressLSTM is None:
            raise ImportError("PyTorch or LSTM model not available. Please install torch and ensure src/model.py exists.")
        model_path = os.path.join(os.path.dirname(__file__), "model", "stress_lstm.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("LSTM model file not found. Please train and save the model as 'model/stress_lstm.pth'.")
        model = StressLSTM(input_size=4)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, ["accX", "accY", "accZ", "activity_level"], "lstm"
    else:
        raise ValueError(f"Unknown or unavailable algorithm: {algo}")

def extract_features(accX, accY, accZ, activity_level):
    """
    Minimal feature extraction for accelerometer-based stress detection.
    This is a placeholder. For best results, use the same features as in training.
    """
    # Example features: mean, std, and magnitude
    features = [accX, accY, accZ, activity_level]
    return np.array(features).reshape(1, -1)


def send_to_stress_pipeline(accX, accY, accZ, activity_level, model, model_type):
    features = extract_features(accX, accY, accZ, activity_level)
    try:
        if model_type == "sklearn":
            pred = model.predict(features)[0]
        elif model_type == "lstm":
            torch, _ = try_import_lstm()
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # (batch, seq_len=1, input_size=4)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
        else:
            raise ValueError("Unknown model type")
        label = 'Stress' if pred == 1 else 'Baseline'
        print(f"[AI] Prediction: {label}")
    except Exception as e:
        print(f"[AI] Prediction error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phyphox Accelerometer Stress Detection")
    parser.add_argument("--algo", choices=["randomforest", "logreg", "xgboost", "lstm"], default="randomforest", help="Algorithm to use")
    args = parser.parse_args()

    print(f"Connecting to Phyphox at: {ACC_ENDPOINT}")
    print(f"Using algorithm: {args.algo}")
    model, feature_names, model_type = get_model(args.algo)

    while True:
        accX, accY, accZ = fetch_accelerometer_data()
        if None not in (accX, accY, accZ):
            activity_level = compute_activity_level(accX, accY, accZ)
            send_to_stress_pipeline(accX, accY, accZ, activity_level, model, model_type)
            print(f"accX: {accX:.3f}, accY: {accY:.3f}, accZ: {accZ:.3f}, Activity Level: {activity_level:.3f}")
        else:
            print("Waiting for valid sensor data...")
        time.sleep(1.5)  # Update every 1–2 seconds
