"""
BVP (Blood Volume Pulse) Preprocessor
=======================================
Processes raw BVP signals to extract heart rate and HRV features.

BVP is captured by photoplethysmography (PPG) on wearable devices.
Each pulse corresponds to one heartbeat. By detecting peaks (systolic
points) we can compute:
- Inter-Beat Intervals (IBI): time between successive heartbeats
- Heart Rate Variability (HRV): variability in IBI, a key stress marker

Under stress, HRV typically decreases (less variability) and heart rate
increases.
"""

import numpy as np
from scipy import signal as scipy_signal
from preprocessing.filters import apply_bandpass_filter


class BVPPreprocessor:
    """
    Preprocess Blood Volume Pulse signal for heart rate and HRV extraction.

    Pipeline:
        1. Band-pass filter 0.5–8 Hz (keeps heartbeat frequency range)
        2. Peak detection to find heartbeats
        3. Compute IBI (inter-beat intervals)
        4. Compute HRV metrics from IBI series

    Parameters
    ----------
    fs : float
        Sampling rate in Hz (default 64 Hz for Empatica E4).
    """

    def __init__(self, fs: float = 64.0):
        self.fs = fs

    def clean(self, bvp_raw: np.ndarray) -> np.ndarray:
        """
        Filter BVP signal using band-pass 0.5–8 Hz.

        - 0.5 Hz lower bound: removes baseline wander and respiration artifacts
        - 8.0 Hz upper bound: removes high-frequency noise while keeping
          the fundamental heartbeat frequency (0.8–3.3 Hz = 48–200 BPM)
          and its harmonics
        """
        bvp_filtered = apply_bandpass_filter(
            bvp_raw, lowcut=0.5, highcut=8.0, fs=self.fs, order=3
        )
        return bvp_filtered

    def detect_peaks(self, bvp_clean: np.ndarray) -> np.ndarray:
        """
        Detect systolic peaks (heartbeats) in the clean BVP signal.

        The BVP signal from Empatica E4 is inverted, so we detect
        peaks in the negated signal. Minimum distance between peaks
        is set to 0.4s (corresponding to 150 BPM maximum).
        """
        min_distance = int(0.4 * self.fs)  # ~150 BPM max

        peaks, _ = scipy_signal.find_peaks(
            -bvp_clean,  # Negate: E4 BVP signal is inverted
            distance=min_distance,
            prominence=np.std(bvp_clean) * 0.3,
        )
        return peaks

    def compute_ibi(self, peaks: np.ndarray) -> np.ndarray:
        """
        Compute Inter-Beat Intervals (IBI) from detected peaks.

        IBI = time between consecutive heartbeats, in milliseconds.
        We filter out physiologically implausible values:
        - < 300 ms would mean > 200 BPM (too fast)
        - > 1500 ms would mean < 40 BPM (too slow)
        """
        ibi = np.diff(peaks) / self.fs * 1000.0  # Convert samples to ms

        # Keep only physiologically plausible IBIs (40–200 BPM)
        valid_mask = (ibi > 300) & (ibi < 1500)
        return ibi[valid_mask]

    def compute_hrv_metrics(self, ibi: np.ndarray) -> dict:
        """
        Compute time-domain Heart Rate Variability (HRV) metrics.

        HRV Metrics:
        - mean_hr:  Average heart rate in BPM
        - sdnn:     Standard deviation of IBI (overall variability)
        - rmssd:    Root mean square of successive IBI differences
                    (short-term variability, parasympathetic marker)
        - pnn50:    Percentage of successive IBIs differing by > 50ms
                    (vagal tone indicator)
        - mean_ibi: Average inter-beat interval in ms

        Under stress: HR↑, SDNN↓, RMSSD↓, pNN50↓
        """
        if len(ibi) < 5:
            return {
                "mean_hr": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                "pnn50": np.nan, "mean_ibi": np.nan,
            }

        diff_ibi = np.diff(ibi)
        mean_ibi = np.mean(ibi)
        mean_hr = 60000.0 / mean_ibi if mean_ibi > 0 else np.nan
        sdnn = np.std(ibi, ddof=1)
        rmssd = np.sqrt(np.mean(diff_ibi ** 2))
        nn50 = np.sum(np.abs(diff_ibi) > 50)
        pnn50 = (nn50 / len(diff_ibi)) * 100 if len(diff_ibi) > 0 else 0.0

        return {
            "mean_hr": mean_hr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "mean_ibi": mean_ibi,
        }

    def process(self, bvp_raw: np.ndarray) -> dict:
        """
        Full BVP preprocessing pipeline: raw → clean → peaks → IBI → HRV.

        Returns dict with keys: 'bvp_clean', 'peaks', 'ibi', 'hrv'
        """
        bvp_clean = self.clean(bvp_raw)
        peaks = self.detect_peaks(bvp_clean)
        ibi = self.compute_ibi(peaks)
        hrv = self.compute_hrv_metrics(ibi)
        return {
            "bvp_clean": bvp_clean,
            "peaks": peaks,
            "ibi": ibi,
            "hrv": hrv,
        }
