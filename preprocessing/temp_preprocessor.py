"""
Temperature Preprocessor
=========================
Processes raw skin temperature signals from wearable sensors.

Skin temperature is measured at the wrist and reflects peripheral
blood flow regulated by the autonomic nervous system. Under stress,
vasoconstriction typically causes a slight decrease in peripheral
skin temperature.

The signal is very slow-varying (< 0.1 Hz), so aggressive low-pass
filtering is appropriate.
"""

import numpy as np
from scipy.signal import medfilt
from preprocessing.filters import apply_lowpass_filter


class TemperaturePreprocessor:
    """
    Preprocess skin temperature signal.

    Pipeline:
        1. Median filter (kernel=3) to remove spike artifacts
        2. Low-pass filter at 0.1 Hz (temperature changes very slowly)
        3. Compute gradient (rate of change)

    Parameters
    ----------
    fs : float
        Sampling rate in Hz (default 4 Hz for Empatica E4).
    """

    def __init__(self, fs: float = 4.0):
        self.fs = fs

    def clean(self, temp_raw: np.ndarray) -> np.ndarray:
        """
        Smooth temperature signal and remove outlier spikes.

        Median filter removes impulsive noise, low-pass at 0.1 Hz
        keeps only the very slow temperature trend.
        """
        temp_median = medfilt(temp_raw, kernel_size=3)
        temp_filtered = apply_lowpass_filter(temp_median, cutoff=0.1, fs=self.fs, order=2)
        return temp_filtered

    def process(self, temp_raw: np.ndarray) -> dict:
        """
        Full temperature preprocessing pipeline.

        Returns dict with:
        - 'clean': smoothed temperature signal
        - 'gradient': rate of temperature change (°C/s)
        """
        temp_clean = self.clean(temp_raw)

        # Gradient = rate of change of temperature over time
        temp_gradient = np.gradient(temp_clean, 1.0 / self.fs)

        return {"clean": temp_clean, "gradient": temp_gradient}
