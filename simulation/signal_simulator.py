"""
Signal Simulator
=================
Simulates realistic wearable sensor signals for stress detection demos.

Generates three physiological signals:
- EDA (Electrodermal Activity) at 4 Hz
- BVP (Blood Volume Pulse) at 64 Hz
- Skin Temperature at 4 Hz

Two conditions are simulated:
- Baseline (label=0): calm resting state
- Stress   (label=1): elevated arousal
"""

import numpy as np


def simulate_eda(duration_sec: float, fs: float = 4.0, stressed: bool = False) -> np.ndarray:
    """
    Simulate an EDA (skin conductance) signal.

    EDA has two components:
    - Slow-varying tonic level (SCL)
    - Occasional phasic peaks (SCR) triggered by arousal

    Under stress: higher baseline, more frequent and larger phasic peaks.

    Parameters
    ----------
    duration_sec : float
        Duration of the signal in seconds.
    fs : float
        Sampling rate in Hz (default 4 Hz).
    stressed : bool
        If True, simulate a stressed condition.

    Returns
    -------
    np.ndarray
        Simulated EDA signal in microSiemens (μS).
    """
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    if stressed:
        # Stressed: higher baseline (5 μS), larger slow fluctuations
        tonic = 5.0 + 1.5 * np.sin(2 * np.pi * 0.04 * t)
        noise = 0.3 * np.random.randn(n_samples)

        # More frequent phasic peaks (SCR events every ~5-10 seconds)
        phasic = np.zeros(n_samples)
        peak_times = np.random.uniform(0, duration_sec, size=int(duration_sec / 7))
        for pt in peak_times:
            idx = int(pt * fs)
            if idx < n_samples - int(4 * fs):
                # SCR shape: fast rise, slow decay (bilateral exponential)
                rise_len = int(1.0 * fs)
                decay_len = int(3.0 * fs)
                amplitude = np.random.uniform(0.5, 2.0)
                rise = np.linspace(0, amplitude, rise_len)
                decay = amplitude * np.exp(-np.linspace(0, 3, decay_len))
                scr = np.concatenate([rise, decay])
                end_idx = min(idx + len(scr), n_samples)
                phasic[idx:end_idx] += scr[: end_idx - idx]
    else:
        # Baseline: lower baseline (2 μS), gentle fluctuations
        tonic = 2.0 + 0.3 * np.sin(2 * np.pi * 0.02 * t)
        noise = 0.1 * np.random.randn(n_samples)

        # Fewer, smaller phasic peaks (every ~15-25 seconds)
        phasic = np.zeros(n_samples)
        peak_times = np.random.uniform(0, duration_sec, size=int(duration_sec / 20))
        for pt in peak_times:
            idx = int(pt * fs)
            if idx < n_samples - int(4 * fs):
                rise_len = int(1.0 * fs)
                decay_len = int(3.0 * fs)
                amplitude = np.random.uniform(0.1, 0.5)
                rise = np.linspace(0, amplitude, rise_len)
                decay = amplitude * np.exp(-np.linspace(0, 3, decay_len))
                scr = np.concatenate([rise, decay])
                end_idx = min(idx + len(scr), n_samples)
                phasic[idx:end_idx] += scr[: end_idx - idx]

    eda = tonic + phasic + noise
    return eda


def simulate_bvp(duration_sec: float, fs: float = 64.0, stressed: bool = False) -> np.ndarray:
    """
    Simulate a BVP (Blood Volume Pulse) signal.

    BVP is a pseudo-periodic signal where each cycle represents one heartbeat.
    The fundamental frequency corresponds to heart rate (1.0–1.3 Hz = 60–78 BPM).

    Under stress: faster heart rate (~1.5 Hz = 90 BPM), lower HRV.

    Parameters
    ----------
    duration_sec : float
        Duration in seconds.
    fs : float
        Sampling rate (default 64 Hz).
    stressed : bool
        If True, simulate elevated heart rate with lower variability.

    Returns
    -------
    np.ndarray
        Simulated BVP signal (arbitrary units).
    """
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    if stressed:
        # ~90 BPM with low variability
        base_freq = 1.5
        freq_variation = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    else:
        # ~72 BPM with natural variability (respiratory sinus arrhythmia)
        base_freq = 1.2
        freq_variation = 0.15 * np.sin(2 * np.pi * 0.25 * t)

    # Instantaneous frequency modulation to simulate HRV
    phase = 2 * np.pi * np.cumsum(base_freq + freq_variation) / fs

    # BVP waveform: fundamental + harmonics for realistic pulse shape
    bvp = (
        -np.sin(phase)                        # fundamental (inverted for E4-style)
        - 0.3 * np.sin(2 * phase + 0.5)       # 2nd harmonic
        - 0.1 * np.sin(3 * phase + 1.0)       # 3rd harmonic
    )

    # Add small amount of noise
    noise = 0.05 * np.random.randn(n_samples)
    bvp = bvp + noise

    return bvp


def simulate_temperature(duration_sec: float, fs: float = 4.0, stressed: bool = False) -> np.ndarray:
    """
    Simulate a skin temperature signal.

    Skin temperature is very slow-varying. Under stress, peripheral
    vasoconstriction causes slight temperature decrease.

    Parameters
    ----------
    duration_sec : float
        Duration in seconds.
    fs : float
        Sampling rate (default 4 Hz).
    stressed : bool
        If True, simulate lower temperature with more variability.

    Returns
    -------
    np.ndarray
        Simulated temperature signal in °C.
    """
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    if stressed:
        # Stressed: slightly lower temp, more variability
        temp = 33.0 + 0.3 * np.sin(2 * np.pi * 0.005 * t) + 0.08 * np.random.randn(n_samples)
    else:
        # Baseline: comfortable skin temp ~33.5°C, minimal variability
        temp = 33.5 + 0.1 * np.sin(2 * np.pi * 0.003 * t) + 0.03 * np.random.randn(n_samples)

    return temp


def simulate_all_signals(duration_sec: float = 300, stressed: bool = False) -> dict:
    """
    Simulate all three physiological signals for a given condition.

    Parameters
    ----------
    duration_sec : float
        Duration in seconds (default 300 = 5 minutes).
    stressed : bool
        If True, simulate stressed condition.

    Returns
    -------
    dict
        Dictionary with keys 'eda', 'bvp', 'temperature' and their
        respective sampling rates and time arrays.
    """
    eda_fs = 4.0
    bvp_fs = 64.0
    temp_fs = 4.0

    eda = simulate_eda(duration_sec, fs=eda_fs, stressed=stressed)
    bvp = simulate_bvp(duration_sec, fs=bvp_fs, stressed=stressed)
    temperature = simulate_temperature(duration_sec, fs=temp_fs, stressed=stressed)

    return {
        "eda": eda,
        "bvp": bvp,
        "temperature": temperature,
        "eda_fs": eda_fs,
        "bvp_fs": bvp_fs,
        "temp_fs": temp_fs,
        "duration_sec": duration_sec,
        "label": 1 if stressed else 0,
    }
