"""
Feature Extraction Module
==========================
Extracts physiological features from preprocessed signals for stress classification.

Features are computed from windowed segments of each signal type:

EDA Features:
    - Mean tonic level (SCL) — baseline skin conductance
    - Std of tonic — variability of baseline
    - Mean phasic amplitude (SCR) — intensity of arousal responses
    - Number of SCR peaks — frequency of arousal events
    - Max phasic amplitude — strongest arousal response

BVP / HRV Features:
    - Mean heart rate (BPM)
    - Mean inter-beat interval (ms)
    - SDNN — standard deviation of IBI (overall HRV)
    - RMSSD — root mean square of successive IBI differences (short-term HRV)
    - pNN50 — percentage of successive IBIs differing by >50ms (vagal tone)

Temperature Features:
    - Mean temperature
    - Std of temperature
    - Mean gradient (rate of change)
    - Max gradient — fastest temperature shift
"""

import numpy as np
from preprocessing.eda_preprocessor import EDAPreprocessor
from preprocessing.bvp_preprocessor import BVPPreprocessor
from preprocessing.temp_preprocessor import TemperaturePreprocessor


def extract_eda_features(eda_window: np.ndarray, fs: float = 4.0) -> dict:
    """
    Extract features from a window of raw EDA signal.

    Parameters
    ----------
    eda_window : np.ndarray
        Raw EDA signal segment.
    fs : float
        Sampling rate (default 4 Hz).

    Returns
    -------
    dict
        Dictionary of EDA feature names to values.
    """
    preprocessor = EDAPreprocessor(fs=fs)
    components = preprocessor.process(eda_window)

    tonic = components["tonic"]
    phasic = components["phasic"]

    # Count SCR peaks: phasic values exceeding a threshold
    scr_threshold = 0.05  # μS
    scr_peaks = np.sum(np.diff((phasic > scr_threshold).astype(int)) == 1)

    return {
        "eda_tonic_mean": np.mean(tonic),
        "eda_tonic_std": np.std(tonic),
        "eda_phasic_mean": np.mean(np.abs(phasic)),
        "eda_scr_count": scr_peaks,
        "eda_phasic_max": np.max(phasic),
    }


def extract_bvp_features(bvp_window: np.ndarray, fs: float = 64.0) -> dict:
    """
    Extract HRV and heart rate features from a window of raw BVP signal.

    Parameters
    ----------
    bvp_window : np.ndarray
        Raw BVP signal segment.
    fs : float
        Sampling rate (default 64 Hz).

    Returns
    -------
    dict
        Dictionary of BVP/HRV feature names to values.
    """
    preprocessor = BVPPreprocessor(fs=fs)
    result = preprocessor.process(bvp_window)
    hrv = result["hrv"]

    return {
        "bvp_mean_hr": hrv["mean_hr"],
        "bvp_mean_ibi": hrv["mean_ibi"],
        "bvp_sdnn": hrv["sdnn"],
        "bvp_rmssd": hrv["rmssd"],
        "bvp_pnn50": hrv["pnn50"],
    }


def extract_temp_features(temp_window: np.ndarray, fs: float = 4.0) -> dict:
    """
    Extract features from a window of raw temperature signal.

    Parameters
    ----------
    temp_window : np.ndarray
        Raw temperature signal segment.
    fs : float
        Sampling rate (default 4 Hz).

    Returns
    -------
    dict
        Dictionary of temperature feature names to values.
    """
    preprocessor = TemperaturePreprocessor(fs=fs)
    result = preprocessor.process(temp_window)

    clean_temp = result["clean"]
    gradient = result["gradient"]

    return {
        "temp_mean": np.mean(clean_temp),
        "temp_std": np.std(clean_temp),
        "temp_gradient_mean": np.mean(gradient),
        "temp_gradient_max": np.max(np.abs(gradient)),
    }


def extract_all_features(eda_window: np.ndarray, bvp_window: np.ndarray,
                          temp_window: np.ndarray,
                          eda_fs: float = 4.0, bvp_fs: float = 64.0,
                          temp_fs: float = 4.0) -> dict:
    """
    Extract all features from one time window of all three signals.

    Combines EDA, BVP/HRV, and temperature features into a single
    feature vector for the machine learning model.

    Parameters
    ----------
    eda_window : np.ndarray
        EDA signal segment for this window.
    bvp_window : np.ndarray
        BVP signal segment for this window.
    temp_window : np.ndarray
        Temperature signal segment for this window.
    eda_fs, bvp_fs, temp_fs : float
        Sampling rates for each signal.

    Returns
    -------
    dict
        Combined feature dictionary with 14 features total.
    """
    features = {}
    features.update(extract_eda_features(eda_window, fs=eda_fs))
    features.update(extract_bvp_features(bvp_window, fs=bvp_fs))
    features.update(extract_temp_features(temp_window, fs=temp_fs))
    return features
