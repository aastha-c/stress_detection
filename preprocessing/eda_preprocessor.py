"""
EDA (Electrodermal Activity) Preprocessor
==========================================
Processes raw skin conductance signals from wearable sensors.

EDA reflects sympathetic nervous system activity:
- Tonic component (SCL): slow-varying baseline skin conductance level
- Phasic component (SCR): rapid skin conductance responses to stimuli

Stress increases both SCL and the frequency/amplitude of SCR events.
"""

import numpy as np
from scipy.signal import medfilt
from preprocessing.filters import apply_lowpass_filter


class EDAPreprocessor:
    """
    Preprocess Electrodermal Activity (skin conductance) signal.

    Pipeline:
        1. Median filter to remove spike artifacts
        2. Low-pass filter at 1 Hz (EDA changes slowly)
        3. Decompose into tonic (SCL) and phasic (SCR) components

    Parameters
    ----------
    fs : float
        Sampling rate of the EDA signal in Hz (default 4 Hz for Empatica E4).
    """

    def __init__(self, fs: float = 4.0):
        self.fs = fs

    def clean(self, eda_raw: np.ndarray) -> np.ndarray:
        """
        Remove artifacts and smooth the raw EDA signal.

        Step 1: Median filter (kernel=5) removes short spike artifacts
        Step 2: Low-pass at 1 Hz removes high-frequency noise
        """
        # Median filter removes impulsive noise (motion artifacts)
        eda_median = medfilt(eda_raw, kernel_size=5)

        # Low-pass filter — EDA is a slow signal (< 1 Hz)
        eda_filtered = apply_lowpass_filter(eda_median, cutoff=1.0, fs=self.fs, order=4)
        return eda_filtered

    def decompose(self, eda_clean: np.ndarray) -> dict:
        """
        Decompose clean EDA into tonic and phasic components.

        - Tonic (SCL): very low frequency baseline (< 0.05 Hz)
        - Phasic (SCR): eda_clean minus tonic = rapid responses
        """
        # Tonic = very low frequency envelope of the signal
        tonic = apply_lowpass_filter(eda_clean, cutoff=0.05, fs=self.fs, order=2)

        # Phasic = what remains after removing the slow baseline
        phasic = eda_clean - tonic

        return {"tonic": tonic, "phasic": phasic, "clean": eda_clean}

    def process(self, eda_raw: np.ndarray) -> dict:
        """
        Full EDA preprocessing pipeline.

        Returns dict with keys: 'tonic', 'phasic', 'clean'
        """
        eda_clean = self.clean(eda_raw)
        components = self.decompose(eda_clean)
        return components
