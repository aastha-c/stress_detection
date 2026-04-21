"""
Data Loader
============
Load and manage the WESAD dataset (wrist-worn Empatica E4 signals).
"""

import os
import pickle
import numpy as np
from typing import Dict, Optional

from src.config import LABEL_SR


class WESADLoader:
    """Load raw WESAD pickle data for a single subject."""

    def __init__(self, data_path: str, subject_id: int):
        self.data_path = data_path
        self.subject_id = subject_id
        self.subject_key = f"S{subject_id}"
        self.raw_data: Optional[Dict] = None
        self.wrist_signals: Dict[str, np.ndarray] = {}
        self.labels: Optional[np.ndarray] = None

    def load(self) -> "WESADLoader":
        """Load raw pickle data for one subject."""
        pkl_path = os.path.join(
            self.data_path, self.subject_key, f"{self.subject_key}.pkl"
        )
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"WESAD file not found: {pkl_path}\n"
                f"Download from: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx"
            )
        with open(pkl_path, "rb") as f:
            self.raw_data = pickle.load(f, encoding="latin1")

        wrist = self.raw_data["signal"]["wrist"]
        self.wrist_signals = {
            "ACC": np.array(wrist["ACC"]),
            "BVP": np.array(wrist["BVP"]).flatten(),
            "EDA": np.array(wrist["EDA"]).flatten(),
            "TEMP": np.array(wrist["TEMP"]).flatten(),
        }
        self.labels = np.array(self.raw_data["label"]).flatten()
        return self

    def get_label_at_rate(self, target_sr: float) -> np.ndarray:
        """Resample labels from 700 Hz to target sampling rate."""
        n_target = int(len(self.labels) * target_sr / LABEL_SR)
        indices = np.linspace(0, len(self.labels) - 1, n_target).astype(int)
        return self.labels[indices]
