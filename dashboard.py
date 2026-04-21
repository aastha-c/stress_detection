"""
Real-Time Wearable Stress Monitoring Dashboard
---------------------------------------------
Streamlit app for live visualization of physiological signals and stress prediction.

Instructions:
1. Ensure your FastAPI backend is running (default: http://localhost:8000).
2. Run this dashboard with:
   streamlit run dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from model.model_loader import load_model
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

import joblib
def get_model(algo):
    if algo == "randomforest":
        model, feature_names = load_model()
        return model, feature_names, "sklearn"
    elif algo == "logreg":
        # Load the trained LogisticRegression model with 5 features
        try:
            data = joblib.load("model/logreg_model.pkl")
            model = data["model"]
            feature_names = data["feature_names"]
            return model, feature_names, "sklearn"
        except Exception as e:
            return None, None, None
    elif algo == "xgboost" and xgb_available:
        model = XGBClassifier()
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        return model, ["accX", "accY", "accZ", "activity_level", "skin_temp"], "sklearn"
    else:
        return None, None, None

def extract_features(accX, accY, accZ, activity_level, temp):
    features = [accX, accY, accZ, activity_level, temp]
    return np.array(features).reshape(1, -1)

def predict_all(accX, accY, accZ, activity_level, temp):
    algos = ["randomforest", "logreg"]
    if xgb_available:
        algos.append("xgboost")
    results = {}
    for algo in algos:
        model, feature_names, model_type = get_model(algo)
        if model is not None:
            features = extract_features(accX, accY, accZ, activity_level, temp)
            try:
                pred = model.predict(features)[0]
                label = 'Stress' if pred == 1 else 'Baseline'
                results[algo] = label
            except Exception as e:
                results[algo] = f"Error: {e}"
        else:
            results[algo] = "Unavailable"
    return results

# --- Config ---
API_URL = "http://localhost:8000/latest-result"
REFRESH_INTERVAL = 3  # seconds

# --- Sidebar: Device Status ---
st.sidebar.title("Device Status")
device_status = st.sidebar.empty()


# --- Main Title ---
st.title("Real-Time Wearable Stress Monitoring")

# --- Algorithm Comparison Page ---
st.header("Manual Algorithm Comparison (Simulated Data)")
st.write("Enter accelerometer values to see predictions from all algorithms.")
accX = st.slider("accX", -10.0, 10.0, 0.0, key="accX")
accY = st.slider("accY", -10.0, 10.0, 0.0, key="accY")
accZ = st.slider("accZ", -10.0, 10.0, 0.0, key="accZ")
activity_level = st.slider("Activity Level", 0.0, 20.0, 1.0, key="activity_level")
temp = st.slider("Skin Temperature (°C)", 25.0, 40.0, 33.0, key="temp")
if st.button("Predict", key="predict_btn"):
    results = predict_all(accX, accY, accZ, activity_level, temp)
    st.subheader("Predictions:")
    for algo, label in results.items():
        st.write(f"**{algo}**: {label}")

# --- Initialize session state for history ---
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = []
if 'sdnn_history' not in st.session_state:
    st.session_state.sdnn_history = []
if 'rmssd_history' not in st.session_state:
    st.session_state.rmssd_history = []
if 'acc_history' not in st.session_state:
    st.session_state.acc_history = []
if 'temp_history' not in st.session_state:
    st.session_state.temp_history = []
if 'stress_history' not in st.session_state:
    st.session_state.stress_history = []

# --- Main Loop ---
placeholder = st.empty()

while True:
    try:
        response = requests.get(API_URL, timeout=2)
        if response.status_code == 200:
            result = response.json()
            features = result.get('features', {})
            prediction = result.get('prediction', 0)
            # Update histories
            st.session_state.hr_history.append(features.get('mean_hr', np.nan))
            st.session_state.sdnn_history.append(features.get('sdnn', np.nan))
            st.session_state.rmssd_history.append(features.get('rmssd', np.nan))
            st.session_state.acc_history.append(features.get('acc_mean', np.nan))
            st.session_state.temp_history.append(features.get('temp_mean', np.nan))
            st.session_state.stress_history.append(prediction)
            device_status.success("Device Connected")
        else:
            features = {}
            prediction = 0
            device_status.warning("No data from device")
    except Exception:
        features = {}
        prediction = 0
        device_status.error("Device Disconnected or API Unreachable")

    # --- Layout ---
    with placeholder.container():
        col1, col2 = st.columns([2, 1])
        # --- Main Graphs ---
        with col1:
            st.subheader("Live Physiological Signals")
            fig, axs = plt.subplots(3, 1, figsize=(7, 7))
            axs[0].plot(st.session_state.hr_history, label="Heart Rate (BPM)", color='tab:red')
            axs[0].set_ylabel("BPM")
            axs[0].legend(loc='upper right')
            axs[1].plot(st.session_state.sdnn_history, label="SDNN", color='tab:blue')
            axs[1].plot(st.session_state.rmssd_history, label="RMSSD", color='tab:green')
            axs[1].set_ylabel("HRV (ms)")
            axs[1].legend(loc='upper right')
            axs[2].plot(st.session_state.acc_history, label="Activity", color='tab:orange')
            axs[2].set_ylabel("Accel")
            axs[2].set_xlabel("Time (intervals)")
            axs[2].legend(loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
            st.subheader("Skin Temperature")
            st.line_chart(st.session_state.temp_history)
        # --- Predictions & Indicators ---
        with col2:
            st.subheader("Stress State")
            if prediction == 1:
                st.markdown('<div style="background-color:#ff4d4d;padding:20px;text-align:center;border-radius:10px;"><h2 style="color:white;">STRESS DETECTED</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color:#4CAF50;padding:20px;text-align:center;border-radius:10px;"><h2 style="color:white;">RELAXED</h2></div>', unsafe_allow_html=True)
            st.markdown("---")
            def safe_metric(val, fmt):
                try:
                    return fmt.format(float(val))
                except (ValueError, TypeError):
                    return str(val)

            st.metric("Current Heart Rate", safe_metric(features.get('mean_hr', 'N/A'), "{:.1f} BPM"))
            st.metric("SDNN", safe_metric(features.get('sdnn', 'N/A'), "{:.2f} ms"))
            st.metric("RMSSD", safe_metric(features.get('rmssd', 'N/A'), "{:.2f} ms"))
            st.metric("Skin Temp", safe_metric(features.get('temp_mean', 'N/A'), "{:.2f} °C"))
            st.metric("Activity", safe_metric(features.get('acc_mean', 'N/A'), "{:.3f}"))
            st.markdown("---")
            # --- Summary Statistics ---
            st.subheader("Summary Stats")
            avg_hr = np.nanmean(st.session_state.hr_history)
            avg_sdnn = np.nanmean(st.session_state.sdnn_history)
            avg_rmssd = np.nanmean(st.session_state.rmssd_history)
            stress_prob = np.mean(st.session_state.stress_history) if st.session_state.stress_history else 0
            st.write(f"**Avg Heart Rate:** {avg_hr:.1f} BPM")
            st.write(f"**Avg SDNN:** {avg_sdnn:.2f} ms")
            st.write(f"**Avg RMSSD:** {avg_rmssd:.2f} ms")
            st.write(f"**Stress Probability:** {stress_prob:.2%}")
    time.sleep(REFRESH_INTERVAL)
