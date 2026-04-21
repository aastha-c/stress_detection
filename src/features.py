"""
Feature Extraction
==================
Statistical and time-series feature extraction from preprocessed signal windows.
"""

import numpy as np
from scipy import stats
from typing import Dict, List


def _statistical_features(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extract statistical features from a 1-D signal segment."""
    return {
        f"{prefix}_mean": np.mean(sig),
        f"{prefix}_std": np.std(sig),
        f"{prefix}_min": np.min(sig),
        f"{prefix}_max": np.max(sig),
        f"{prefix}_range": np.ptp(sig),
        f"{prefix}_median": np.median(sig),
        f"{prefix}_skew": float(stats.skew(sig)),
        f"{prefix}_kurtosis": float(stats.kurtosis(sig)),
        f"{prefix}_iqr": float(np.subtract(*np.percentile(sig, [75, 25]))),
        f"{prefix}_rms": float(np.sqrt(np.mean(sig ** 2))),
    }


def _timeseries_features(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extract time-series features from a 1-D signal segment."""
    diff1 = np.diff(sig)
    diff2 = np.diff(sig, n=2)

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0)

    # Slope sign changes
    slope_changes = np.sum(np.diff(np.sign(diff1)) != 0) if len(diff1) > 1 else 0

    # Auto-correlation at lag 1
    if np.std(sig) > 1e-10 and len(sig) > 1:
        autocorr = np.corrcoef(sig[:-1], sig[1:])[0, 1]
    else:
        autocorr = 0.0

    return {
        f"{prefix}_diff_mean": np.mean(np.abs(diff1)) if len(diff1) > 0 else 0.0,
        f"{prefix}_diff_std": np.std(diff1) if len(diff1) > 0 else 0.0,
        f"{prefix}_diff2_mean": np.mean(np.abs(diff2)) if len(diff2) > 0 else 0.0,
        f"{prefix}_zcr": float(zero_crossings / len(sig)) if len(sig) > 0 else 0.0,
        f"{prefix}_slope_changes": float(slope_changes),
        f"{prefix}_autocorr_1": float(autocorr),
        f"{prefix}_energy": float(np.sum(sig ** 2)),
        f"{prefix}_line_length": float(np.sum(np.abs(diff1))) if len(diff1) > 0 else 0.0,
    }


def extract_window_features(window: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract all features from a single window of signals."""
    features: Dict[str, float] = {}
    for sig_name, sig_data in window.items():
        features.update(_statistical_features(sig_data, sig_name))
        features.update(_timeseries_features(sig_data, sig_name))
    return features


def extract_all_features(
    windows: List[Dict[str, np.ndarray]],
    labels: List[int],
) -> tuple:
    """
    Extract features from all windows.

    Returns
    -------
    feature_matrix : np.ndarray, shape (n_windows, n_features)
    feature_names  : list of str
    labels_array   : np.ndarray, shape (n_windows,)
    """
    all_features = []
    for win in windows:
        feat = extract_window_features(win)
        all_features.append(feat)

    feature_names = list(all_features[0].keys())
    feature_matrix = np.array(
        [[f[name] for name in feature_names] for f in all_features], dtype=np.float32
    )

    # Replace NaN/Inf with 0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_matrix, feature_names, np.array(labels, dtype=np.int64)


def prepare_sequences(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    seq_length: int = 5,
) -> tuple:
    """
    Create overlapping sequences for LSTM input.

    Parameters
    ----------
    feature_matrix : (n_windows, n_features)
    labels         : (n_windows,)
    seq_length     : number of consecutive windows per sequence

    Returns
    -------
    X : np.ndarray, shape (n_sequences, seq_length, n_features)
    y : np.ndarray, shape (n_sequences,)
    """
    X, y = [], []
    for i in range(len(feature_matrix) - seq_length + 1):
        X.append(feature_matrix[i : i + seq_length])
        y.append(labels[i + seq_length - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
