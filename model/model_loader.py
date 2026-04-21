"""
Model loader for stress detection RandomForestClassifier.
"""
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "stress_model.pkl")

_model = None
_feature_names = None

def load_model():
    global _model, _feature_names
    if _model is None or _feature_names is None:
        data = joblib.load(MODEL_PATH)
        _model = data["model"]
        _feature_names = data["feature_names"]
    return _model, _feature_names
