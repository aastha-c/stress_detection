"""
Main Pipeline
==============
End-to-end stress detection pipeline that:
    1. Simulates physiological signals (EDA, BVP, Temperature)
    2. Preprocesses and windows the signals
    3. Extracts features from each window
    4. Trains a RandomForest classifier
    5. Demonstrates real-time prediction on new data

Run this script to train the model before launching the Streamlit dashboard:
    python main_pipeline.py
"""

import os
import sys
import numpy as np

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.signal_simulator import simulate_all_signals
from features.feature_extraction import extract_all_features
from utils.windowing import create_multimodal_windows
from model.train_model import run_training_pipeline
import joblib


def demo_realtime_prediction():
    """
    Demonstrate the trained model making predictions on new simulated data.

    Simulates a short session, windows it, extracts features, and predicts
    stress vs baseline for each window — mimicking a real-time loop.
    """
    model_path = os.path.join("model", "stress_model.pkl")
    if not os.path.exists(model_path):
        print("  No trained model found. Please run training first.")
        return

    # Load model
    saved = joblib.load(model_path)
    clf = saved["model"]
    feature_names = saved["feature_names"]

    print("\n" + "=" * 60)
    print("  Real-Time Prediction Demo")
    print("=" * 60)

    # Simulate a new 3-minute mixed session
    # First half baseline, second half stressed
    print("\n  Simulating 3-minute mixed session...")
    baseline_signals = simulate_all_signals(duration_sec=90, stressed=False)
    stress_signals = simulate_all_signals(duration_sec=90, stressed=True)

    # Process baseline segment
    print("\n  --- Baseline Segment (0–90s) ---")
    windows = create_multimodal_windows(baseline_signals, window_sec=60.0, overlap=0.5)
    for i, w in enumerate(windows):
        feats = extract_all_features(
            w["eda"], w["bvp"], w["temperature"],
            eda_fs=baseline_signals["eda_fs"],
            bvp_fs=baseline_signals["bvp_fs"],
            temp_fs=baseline_signals["temp_fs"],
        )
        x = np.array([[feats[name] for name in feature_names]])
        # Replace NaN
        x = np.nan_to_num(x, nan=0.0)

        pred = clf.predict(x)[0]
        prob = clf.predict_proba(x)[0]
        state = "Relaxed ✓" if pred == 0 else "STRESS ✗"
        print(f"    Window {i + 1}: {state}  (P(stress)={prob[1]:.2f})")

    # Process stress segment
    print("\n  --- Stress Segment (90–180s) ---")
    windows = create_multimodal_windows(stress_signals, window_sec=60.0, overlap=0.5)
    for i, w in enumerate(windows):
        feats = extract_all_features(
            w["eda"], w["bvp"], w["temperature"],
            eda_fs=stress_signals["eda_fs"],
            bvp_fs=stress_signals["bvp_fs"],
            temp_fs=stress_signals["temp_fs"],
        )
        x = np.array([[feats[name] for name in feature_names]])
        x = np.nan_to_num(x, nan=0.0)

        pred = clf.predict(x)[0]
        prob = clf.predict_proba(x)[0]
        state = "Relaxed ✓" if pred == 0 else "STRESS ✗"
        print(f"    Window {i + 1}: {state}  (P(stress)={prob[1]:.2f})")

    print("\n" + "=" * 60)
    print("  Demo complete! Launch the dashboard with:")
    print("    streamlit run app.py")
    print("=" * 60)


def main():
    """
    Main entry point: train the model, then run a prediction demo.
    """
    # Step 1: Train the model
    run_training_pipeline()

    # Step 2: Demonstrate real-time prediction
    demo_realtime_prediction()


if __name__ == "__main__":
    main()
