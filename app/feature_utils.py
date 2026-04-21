"""
Feature extraction utilities for HRV and sensor data.
"""
import numpy as np
import pandas as pd

def heart_rate_to_ibi(hr_series):
    """Convert heart rate (bpm) to inter-beat intervals (ms)."""
    ibi = 60000.0 / hr_series
    return ibi

def compute_hrv_features(ibi_series):
    """Compute HRV features: mean HR, SDNN, RMSSD, pNN50."""
    if len(ibi_series) < 2:
        return {"mean_hr": np.nan, "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    diff_ibi = np.diff(ibi_series)
    sdnn = np.std(ibi_series)
    rmssd = np.sqrt(np.mean(diff_ibi ** 2))
    nn50 = np.sum(np.abs(diff_ibi) > 50)
    pnn50 = 100.0 * nn50 / len(diff_ibi)
    mean_hr = 60000.0 / np.mean(ibi_series)
    return {
        "mean_hr": mean_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50
    }

def extract_acc_features(df):
    """Extract activity features from accelerometer data."""
    acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    return {
        "acc_mean": acc_mag.mean(),
        "acc_std": acc_mag.std(),
        "acc_max": acc_mag.max(),
    }

def extract_temp_features(df):
    """Extract temperature features."""
    return {
        "temp_mean": df['temperature'].mean(),
        "temp_std": df['temperature'].std(),
    }
