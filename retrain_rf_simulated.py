"""
Retrain RandomForestClassifier for 5 features: accX, accY, accZ, activity_level, skin_temp
Simulated data for demonstration purposes.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Simulate data
np.random.seed(42)
N = 500  # number of samples
accX = np.random.uniform(-10, 10, N)
accY = np.random.uniform(-10, 10, N)
accZ = np.random.uniform(-10, 10, N)
activity_level = np.abs(accX) + np.abs(accY) + np.abs(accZ) + np.random.normal(0, 2, N)
skin_temp = np.random.uniform(28, 38, N)

# Simulate binary labels: stress if high activity or low temp
labels = ((activity_level > 20) | (skin_temp < 32)).astype(int)

X = np.column_stack([accX, accY, accZ, activity_level, skin_temp])
feature_names = ["accX", "accY", "accZ", "activity_level", "skin_temp"]


# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, labels)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X, labels)

# Save models
os.makedirs("model", exist_ok=True)
joblib.dump({"model": clf, "feature_names": feature_names}, "model/stress_model.pkl")
joblib.dump({"model": logreg, "feature_names": feature_names}, "model/logreg_model.pkl")
print("RandomForest and LogisticRegression models trained and saved with features:", feature_names)
