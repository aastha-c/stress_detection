"""
Configuration
=============
Central configuration for the stress detection system.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "WESAD")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── WESAD Dataset ────────────────────────────────────────────────────────────
WESAD_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

WESAD_LABELS = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

# Empatica E4 sampling rates (wrist device)
SAMPLING_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}

LABEL_SR = 700  # RespiBAN label sampling rate

# ── Preprocessing ────────────────────────────────────────────────────────────
TARGET_SR = 4.0           # Common resampling rate (Hz)
WINDOW_SEC = 60.0         # Window duration (seconds)
WINDOW_OVERLAP = 0.5      # Overlap fraction

# ── Model Hyperparameters ────────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 8

# ── Binary classification labels ─────────────────────────────────────────────
LABEL_MAP = {0: "Baseline", 1: "Stress"}
