"""
Motion-Based Stress Model Training
====================================
Trains a RandomForest classifier to detect stress from phone
accelerometer and gyroscope motion patterns.

Simulates many sessions of baseline (calm) and stressed (restless)
motion data, extracts features, and trains a binary classifier.

Run:
    python model/train_motion_model.py
"""

import os
import sys
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.motion_features import extract_motion_features, simulate_motion_data


def generate_motion_training_data(
    n_baseline: int = 50,
    n_stress: int = 50,
    duration_sec: float = 10.0,
    fs: float = 20.0,
) -> tuple:
    """
    Generate labeled feature vectors from simulated motion data.

    Parameters
    ----------
    n_baseline : int
        Number of baseline (relaxed) sessions.
    n_stress : int
        Number of stressed sessions.
    duration_sec : float
        Duration of each session in seconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Labels: 0 = baseline, 1 = stress.
    feature_names : list of str
    """
    all_features = []
    all_labels = []

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"  Generating {n_baseline} baseline motion sessions...")
    for i in range(n_baseline):
        data = simulate_motion_data(duration_sec=duration_sec, fs=fs, stressed=False)
        feats = extract_motion_features(
            data["acc_x"], data["acc_y"], data["acc_z"],
            data["gyro_alpha"], data["gyro_beta"], data["gyro_gamma"],
            fs=fs,
        )
        all_features.append(feats)
        all_labels.append(0)

    print(f"  Generating {n_stress} stressed motion sessions...")
    for i in range(n_stress):
        data = simulate_motion_data(duration_sec=duration_sec, fs=fs, stressed=True)
        feats = extract_motion_features(
            data["acc_x"], data["acc_y"], data["acc_z"],
            data["gyro_alpha"], data["gyro_beta"], data["gyro_gamma"],
            fs=fs,
        )
        all_features.append(feats)
        all_labels.append(1)

    feature_names = list(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_labels)

    # Replace any NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    return X, y, feature_names


def train_motion_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForest on motion features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Baseline", "Stress"]))

    return clf


def run_motion_training():
    """Full pipeline: simulate → extract features → train → save."""
    print("=" * 60)
    print("  Motion-Based Stress Detection — Model Training")
    print("=" * 60)

    print("\n[1/3] Generating synthetic motion training data...")
    X, y, feature_names = generate_motion_training_data(
        n_baseline=100, n_stress=100, duration_sec=10.0
    )
    print(f"  Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Baseline: {np.sum(y == 0)}, Stress: {np.sum(y == 1)}")

    print("\n[2/3] Training RandomForest classifier...")
    clf = train_motion_model(X, y)

    print("\n[3/3] Saving model...")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "motion_stress_model.pkl")
    joblib.dump({"model": clf, "feature_names": feature_names}, model_path)
    print(f"  Model saved to {model_path}")

    print("\n  Feature Importances:")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"    {feature_names[i]:25s} {importances[i]:.4f}")

    print("\n" + "=" * 60)
    print("  Motion model training complete!")
    print("=" * 60)
    return clf, feature_names


if __name__ == "__main__":
    run_motion_training()
