"""
Windowing Utility
==================
Creates sliding windows from continuous physiological signals.

Sliding windows allow us to compute features over fixed-duration segments
of the signal, producing one feature vector per window for the ML model.

Window parameters:
    - Window length: 60 seconds (captures enough heartbeats for reliable HRV)
    - Overlap: 50% (30-second step) — balances temporal resolution and redundancy
"""

import numpy as np


def create_sliding_windows(signal: np.ndarray, fs: float,
                           window_sec: float = 60.0,
                           overlap: float = 0.5) -> list:
    """
    Split a continuous signal into overlapping fixed-duration windows.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    fs : float
        Sampling rate in Hz.
    window_sec : float
        Window length in seconds (default 60s).
    overlap : float
        Fraction of overlap between consecutive windows (default 0.5 = 50%).

    Returns
    -------
    list of np.ndarray
        List of signal segments, each of length int(window_sec * fs).

    Example
    -------
    For a 300s signal at 4 Hz with 60s windows and 50% overlap:
    - Window size = 240 samples
    - Step size = 120 samples
    - Number of windows = (1200 - 240) / 120 + 1 = 9
    """
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap))

    windows = []
    start = 0
    while start + window_samples <= len(signal):
        windows.append(signal[start: start + window_samples])
        start += step_samples

    return windows


def create_multimodal_windows(signals: dict, window_sec: float = 60.0,
                               overlap: float = 0.5) -> list:
    """
    Create aligned sliding windows across multiple signals with different
    sampling rates.

    Each signal is windowed independently using its own sampling rate,
    but all share the same time boundaries.

    Parameters
    ----------
    signals : dict
        Must contain keys: 'eda', 'bvp', 'temperature',
        'eda_fs', 'bvp_fs', 'temp_fs'.
    window_sec : float
        Window length in seconds.
    overlap : float
        Overlap fraction.

    Returns
    -------
    list of dict
        Each element is a dict with windowed segments for all three signals,
        e.g. {'eda': array, 'bvp': array, 'temperature': array}.
    """
    eda_fs = signals["eda_fs"]
    bvp_fs = signals["bvp_fs"]
    temp_fs = signals["temp_fs"]

    eda_windows = create_sliding_windows(signals["eda"], eda_fs, window_sec, overlap)
    bvp_windows = create_sliding_windows(signals["bvp"], bvp_fs, window_sec, overlap)
    temp_windows = create_sliding_windows(signals["temperature"], temp_fs, window_sec, overlap)

    # Use the minimum number of windows across all signals
    n_windows = min(len(eda_windows), len(bvp_windows), len(temp_windows))

    multimodal_windows = []
    for i in range(n_windows):
        multimodal_windows.append({
            "eda": eda_windows[i],
            "bvp": bvp_windows[i],
            "temperature": temp_windows[i],
        })

    return multimodal_windows
