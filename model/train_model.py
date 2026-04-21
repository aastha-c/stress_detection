"""
Model Training Module
======================
Trains a RandomForestClassifier to classify stress vs baseline using
features extracted from simulated physiological signals.

Workflow:
    1. Generate synthetic baseline and stress signals
    2. Window the signals with 60s windows, 50% overlap
    3. Extract features from each window
    4. Train a RandomForest on the feature vectors
    5. Save the trained model with joblib

The RandomForest is chosen for its:
    - Robustness to feature scale differences
    - Built-in feature importance ranking
    - Good performance on tabular data with moderate sample sizes
    - Resistance to overfitting via ensemble averaging
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

from simulation.signal_simulator import simulate_all_signals
from features.feature_extraction import extract_all_features
from utils.windowing import create_multimodal_windows


def generate_training_data(n_baseline: int = 10, n_stress: int = 10,
                           duration_sec: float = 300,
                           window_sec: float = 60.0,
                           overlap: float = 0.5) -> tuple:
    """
    Generate labeled feature vectors from simulated signals.

    Creates multiple sessions of baseline and stress data, windows each
    session, and extracts features from every window.

    Parameters
    ----------
    n_baseline : int
        Number of baseline recording sessions to simulate.
    n_stress : int
        Number of stressed recording sessions to simulate.
    duration_sec : float
        Duration of each session in seconds.
    window_sec : float
        Window length for feature extraction.
    overlap : float
        Overlap fraction for sliding windows.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_windows, n_features).
    y : np.ndarray
        Labels: 0 = baseline, 1 = stress.
    feature_names : list of str
        Names of the features (column names).
    """
    all_features = []
    all_labels = []

    # Suppress filter warnings for short windows
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"  Generating {n_baseline} baseline sessions ({duration_sec}s each)...")
    for i in range(n_baseline):
        signals = simulate_all_signals(duration_sec=duration_sec, stressed=False)
        windows = create_multimodal_windows(signals, window_sec=window_sec, overlap=overlap)

        for w in windows:
            feats = extract_all_features(
                w["eda"], w["bvp"], w["temperature"],
                eda_fs=signals["eda_fs"],
                bvp_fs=signals["bvp_fs"],
                temp_fs=signals["temp_fs"],
            )
            all_features.append(feats)
            all_labels.append(0)  # baseline

    print(f"  Generating {n_stress} stress sessions ({duration_sec}s each)...")
    for i in range(n_stress):
        signals = simulate_all_signals(duration_sec=duration_sec, stressed=True)
        windows = create_multimodal_windows(signals, window_sec=window_sec, overlap=overlap)

        for w in windows:
            feats = extract_all_features(
                w["eda"], w["bvp"], w["temperature"],
                eda_fs=signals["eda_fs"],
                bvp_fs=signals["bvp_fs"],
                temp_fs=signals["temp_fs"],
            )
            all_features.append(feats)
            all_labels.append(1)  # stress

    # Convert list of dicts to numpy array
    feature_names = list(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_labels)

    # Replace NaN values with column means (some HRV features may be NaN)
    col_means = np.nanmean(X, axis=0)
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        X[nan_mask, col_idx] = col_means[col_idx]

    return X, y, feature_names


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Labels (0 = baseline, 1 = stress).

    Returns
    -------
    RandomForestClassifier
        Trained model.
    """
    # Split into train/test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    # Train RandomForest with balanced class weights
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Baseline", "Stress"]))

    return clf


def save_model(clf: RandomForestClassifier, feature_names: list,
               model_dir: str = None):
    """
    Save trained model and feature names using joblib.

    Parameters
    ----------
    clf : RandomForestClassifier
        Trained classifier.
    feature_names : list
        List of feature names for reference.
    model_dir : str
        Directory to save the model (default: project model/ folder).
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "stress_model.pkl")
    joblib.dump({"model": clf, "feature_names": feature_names}, model_path)
    print(f"  Model saved to {model_path}")


def run_training_pipeline():
    """
    Complete training pipeline: simulate → extract features → train → save.
    """
    print("=" * 60)
    print("  Stress Detection — Model Training")
    print("=" * 60)

    # Step 1: Generate synthetic training data
    print("\n[1/3] Generating synthetic training data...")
    X, y, feature_names = generate_training_data(
        n_baseline=10, n_stress=10, duration_sec=300
    )
    print(f"  Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Baseline: {np.sum(y == 0)}, Stress: {np.sum(y == 1)}")

    # Step 2: Train model
    print("\n[2/3] Training RandomForest classifier...")
    clf = train_model(X, y)

    # Step 3: Save model
    print("\n[3/3] Saving model...")
    save_model(clf, feature_names)

    # Print feature importances
    print("\n  Feature Importances:")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"    {feature_names[i]:25s} {importances[i]:.4f}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)
    return clf, feature_names


if __name__ == "__main__":
    run_training_pipeline()
