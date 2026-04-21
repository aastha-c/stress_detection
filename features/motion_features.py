"""
Motion Feature Extraction
==========================
Extracts stress-related features from accelerometer and gyroscope data
captured via phone sensors (DeviceMotion API).

Features are designed to capture restlessness, jitter, fidgeting, and
other motion patterns correlated with psychological stress.

Feature Groups:
    Accelerometer (acc_x, acc_y, acc_z):
        - Magnitude statistics (mean, std, max)
        - Jerk (rate of change of acceleration)
        - Stillness ratio
    Gyroscope (gyro_alpha, gyro_beta, gyro_gamma):
        - Rotation magnitude statistics
        - Angular jerk
    Combined:
        - Movement intensity (energy)
        - Dominant frequency via FFT
        - Signal entropy
"""

import numpy as np
from typing import Optional


def _magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute the Euclidean magnitude of a 3-axis signal."""
    return np.sqrt(x**2 + y**2 + z**2)


def _jerk(signal: np.ndarray, fs: float) -> np.ndarray:
    """Compute jerk (derivative) of a signal."""
    return np.diff(signal) * fs


def _dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """Find the dominant frequency in a signal using FFT."""
    if len(signal) < 4:
        return 0.0
    # Remove DC component
    signal = signal - np.mean(signal)
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Ignore DC (index 0)
    if len(fft_vals) > 1:
        dominant_idx = np.argmax(fft_vals[1:]) + 1
        return float(freqs[dominant_idx])
    return 0.0


def _signal_entropy(signal: np.ndarray, bins: int = 20) -> float:
    """Compute Shannon entropy of a signal's amplitude distribution."""
    if len(signal) < 2:
        return 0.0
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    # Normalize to probability
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def extract_motion_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    gyro_alpha: Optional[np.ndarray] = None,
    gyro_beta: Optional[np.ndarray] = None,
    gyro_gamma: Optional[np.ndarray] = None,
    fs: float = 20.0,
) -> dict:
    """
    Extract motion-based stress features from accelerometer and gyroscope data.

    Parameters
    ----------
    acc_x, acc_y, acc_z : np.ndarray
        Accelerometer readings (m/s²) along three axes.
    gyro_alpha, gyro_beta, gyro_gamma : np.ndarray or None
        Gyroscope readings (deg/s) along three axes.
        If None, gyroscope features will be set to 0.
    fs : float
        Sampling rate in Hz (default 20 Hz from DeviceMotion API).

    Returns
    -------
    dict
        Dictionary of feature names to values (12 features total).
    """
    features = {}

    # ── Accelerometer Features ───────────────────────────────────────────
    acc_mag = _magnitude(acc_x, acc_y, acc_z)

    features["acc_magnitude_mean"] = float(np.mean(acc_mag))
    features["acc_magnitude_std"] = float(np.std(acc_mag))
    features["acc_magnitude_max"] = float(np.max(acc_mag))

    # Jerk: how abruptly acceleration changes (stress → more sudden moves)
    acc_jerk = _jerk(acc_mag, fs)
    features["acc_jerk_mean"] = float(np.mean(np.abs(acc_jerk)))
    features["acc_jerk_std"] = float(np.std(acc_jerk))

    # Stillness ratio: fraction of time acceleration is near gravity (9.81 ± 0.5)
    stillness_threshold = 0.5  # m/s² deviation from gravity
    near_still = np.abs(acc_mag - 9.81) < stillness_threshold
    features["stillness_ratio"] = float(np.mean(near_still))

    # ── Gyroscope Features ───────────────────────────────────────────────
    if gyro_alpha is not None and gyro_beta is not None and gyro_gamma is not None:
        gyro_mag = _magnitude(gyro_alpha, gyro_beta, gyro_gamma)
        features["gyro_magnitude_mean"] = float(np.mean(gyro_mag))
        features["gyro_magnitude_std"] = float(np.std(gyro_mag))

        gyro_jerk = _jerk(gyro_mag, fs)
        features["gyro_jerk_mean"] = float(np.mean(np.abs(gyro_jerk)))
    else:
        features["gyro_magnitude_mean"] = 0.0
        features["gyro_magnitude_std"] = 0.0
        features["gyro_jerk_mean"] = 0.0

    # ── Combined Features ────────────────────────────────────────────────
    # Movement intensity: RMS of all acceleration axes
    movement_energy = np.mean(acc_x**2 + acc_y**2 + acc_z**2)
    features["movement_intensity"] = float(np.sqrt(movement_energy))

    # Dominant frequency: rhythmic vs chaotic motion
    features["dominant_frequency"] = _dominant_frequency(acc_mag, fs)

    # Signal entropy: randomness of motion pattern
    features["motion_entropy"] = _signal_entropy(acc_mag)

    return features


def simulate_motion_data(
    duration_sec: float = 10.0,
    fs: float = 20.0,
    stressed: bool = False,
) -> dict:
    """
    Simulate realistic phone accelerometer/gyroscope data.

    Baseline (relaxed): phone held steady or on a table, minimal movement.
    Stressed: restless fidgeting, phone being shifted, tapped, picked up.

    Parameters
    ----------
    duration_sec : float
        Duration of simulation in seconds.
    fs : float
        Sampling rate in Hz.
    stressed : bool
        If True, simulate stressed/restless motion patterns.

    Returns
    -------
    dict
        Dictionary with acc_x, acc_y, acc_z, gyro_alpha, gyro_beta, gyro_gamma arrays.
    """
    n = int(duration_sec * fs)
    t = np.arange(n) / fs

    if stressed:
        # Stressed: irregular fidgeting, more hand tremor, phone shifting
        # Gravity + tremor + irregular movements
        acc_x = (
            0.0
            + 0.8 * np.sin(2 * np.pi * np.random.uniform(3, 8) * t)  # tremor
            + 1.5 * np.random.randn(n)  # jitter
            + 0.5 * np.sin(2 * np.pi * 0.3 * t)  # slow shifting
        )
        acc_y = (
            0.0
            + 0.6 * np.sin(2 * np.pi * np.random.uniform(4, 9) * t + 1.0)
            + 1.2 * np.random.randn(n)
            + 0.4 * np.sin(2 * np.pi * 0.25 * t)
        )
        acc_z = (
            9.81
            + 0.4 * np.sin(2 * np.pi * np.random.uniform(2, 6) * t + 2.0)
            + 0.8 * np.random.randn(n)
        )

        # Occasional sudden movements (picking up phone, shifting position)
        n_bursts = np.random.randint(2, 6)
        for _ in range(n_bursts):
            burst_start = np.random.randint(0, max(1, n - int(fs)))
            burst_len = int(np.random.uniform(0.3, 1.0) * fs)
            burst_end = min(burst_start + burst_len, n)
            acc_x[burst_start:burst_end] += np.random.uniform(2, 5) * np.random.randn(burst_end - burst_start)
            acc_y[burst_start:burst_end] += np.random.uniform(2, 4) * np.random.randn(burst_end - burst_start)

        # Gyroscope: more rotation from fidgeting
        gyro_alpha = 15.0 * np.random.randn(n) + 8.0 * np.sin(2 * np.pi * 1.5 * t)
        gyro_beta = 12.0 * np.random.randn(n) + 6.0 * np.sin(2 * np.pi * 2.0 * t)
        gyro_gamma = 10.0 * np.random.randn(n) + 5.0 * np.sin(2 * np.pi * 1.8 * t)
    else:
        # Baseline: phone relatively still, minor natural hand tremor
        acc_x = 0.0 + 0.05 * np.random.randn(n)
        acc_y = 0.0 + 0.04 * np.random.randn(n)
        acc_z = 9.81 + 0.03 * np.random.randn(n)

        # Very subtle breathing-related movement
        acc_x += 0.02 * np.sin(2 * np.pi * 0.25 * t)
        acc_y += 0.015 * np.sin(2 * np.pi * 0.25 * t + 0.5)

        # Gyroscope: minimal rotation
        gyro_alpha = 0.5 * np.random.randn(n)
        gyro_beta = 0.4 * np.random.randn(n)
        gyro_gamma = 0.3 * np.random.randn(n)

    return {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_alpha": gyro_alpha,
        "gyro_beta": gyro_beta,
        "gyro_gamma": gyro_gamma,
        "fs": fs,
        "duration_sec": duration_sec,
    }
