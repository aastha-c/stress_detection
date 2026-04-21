"""
Signal Preprocessing
====================
EDA, BVP/HRV, and Temperature preprocessing from Empatica E4 wrist signals.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt
from typing import Dict, List, Tuple

from src.config import SAMPLING_RATES, TARGET_SR


# ── Filter utilities ─────────────────────────────────────────────────────────

def butter_lowpass(cutoff: float, fs: float, order: int = 4) -> Tuple:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return b, a


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band", analog=False)
    return b, a


def apply_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def apply_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)


def resample_signal(sig: np.ndarray, original_sr: float, target_sr: float) -> np.ndarray:
    n_target = int(len(sig) * target_sr / original_sr)
    return signal.resample(sig, n_target)


# ── EDA Preprocessor ─────────────────────────────────────────────────────────

class EDAPreprocessor:
    """Preprocess Electrodermal Activity (skin conductance)."""

    def __init__(self, fs: float = 4.0):
        self.fs = fs

    def clean(self, eda_raw: np.ndarray) -> np.ndarray:
        eda_med = medfilt(eda_raw, kernel_size=5)
        return apply_lowpass_filter(eda_med, cutoff=1.0, fs=self.fs, order=4)

    def decompose(self, eda_clean: np.ndarray) -> Dict[str, np.ndarray]:
        tonic = apply_lowpass_filter(eda_clean, cutoff=0.05, fs=self.fs, order=2)
        phasic = eda_clean - tonic
        return {"tonic": tonic, "phasic": phasic, "clean": eda_clean}

    def process(self, eda_raw: np.ndarray) -> Dict[str, np.ndarray]:
        return self.decompose(self.clean(eda_raw))


# ── BVP / HRV Preprocessor ───────────────────────────────────────────────────

class BVPPreprocessor:
    """Preprocess Blood Volume Pulse for HRV extraction."""

    def __init__(self, fs: float = 64.0):
        self.fs = fs

    def clean(self, bvp_raw: np.ndarray) -> np.ndarray:
        return apply_bandpass_filter(bvp_raw, 0.5, 8.0, self.fs, order=3)

    def detect_peaks(self, bvp_clean: np.ndarray) -> np.ndarray:
        min_distance = int(0.4 * self.fs)
        peaks, _ = signal.find_peaks(
            -bvp_clean,
            distance=min_distance,
            prominence=np.std(bvp_clean) * 0.3,
        )
        return peaks

    def compute_ibi(self, peaks: np.ndarray) -> np.ndarray:
        ibi = np.diff(peaks) / self.fs * 1000.0
        valid = (ibi > 300) & (ibi < 1500)
        return ibi[valid]

    def compute_hrv_metrics(self, ibi: np.ndarray) -> Dict[str, float]:
        if len(ibi) < 5:
            return {"mean_hr": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                    "pnn50": np.nan, "mean_ibi": np.nan}

        diff_ibi = np.diff(ibi)
        mean_ibi = np.mean(ibi)
        return {
            "mean_hr": 60000.0 / mean_ibi if mean_ibi > 0 else np.nan,
            "sdnn": np.std(ibi, ddof=1),
            "rmssd": np.sqrt(np.mean(diff_ibi ** 2)),
            "pnn50": (np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi)) * 100 if len(diff_ibi) > 0 else 0,
            "mean_ibi": mean_ibi,
        }

    def process(self, bvp_raw: np.ndarray) -> Dict:
        bvp_clean = self.clean(bvp_raw)
        peaks = self.detect_peaks(bvp_clean)
        ibi = self.compute_ibi(peaks)
        hrv = self.compute_hrv_metrics(ibi)
        return {"bvp_clean": bvp_clean, "peaks": peaks, "ibi": ibi, "hrv": hrv}


# ── Temperature Preprocessor ─────────────────────────────────────────────────

class TemperaturePreprocessor:
    """Preprocess skin temperature signal."""

    def __init__(self, fs: float = 4.0):
        self.fs = fs

    def clean(self, temp_raw: np.ndarray) -> np.ndarray:
        temp_med = medfilt(temp_raw, kernel_size=3)
        return apply_lowpass_filter(temp_med, cutoff=0.1, fs=self.fs, order=2)

    def process(self, temp_raw: np.ndarray) -> Dict[str, np.ndarray]:
        temp_clean = self.clean(temp_raw)
        return {"clean": temp_clean, "gradient": np.gradient(temp_clean, 1.0 / self.fs)}


# ── Windowing ────────────────────────────────────────────────────────────────

def create_windows(
    signals: Dict[str, np.ndarray],
    labels: np.ndarray,
    window_sec: float = 60.0,
    overlap: float = 0.5,
    target_sr: float = 4.0,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Segment signals into fixed-length windows. Binary labels: 0=baseline, 1=stress."""
    window_len = int(window_sec * target_sr)
    step = int(window_len * (1 - overlap))
    n_samples = min(len(labels), *(len(s) for s in signals.values()))

    windows: List[Dict[str, np.ndarray]] = []
    window_labels: List[int] = []

    for start in range(0, n_samples - window_len + 1, step):
        end = start + window_len
        seg_labels = labels[start:end]
        unique, counts = np.unique(seg_labels, return_counts=True)
        majority = unique[np.argmax(counts)]

        if majority not in (1, 2):
            continue

        win = {name: sig[start:end] for name, sig in signals.items()}
        windows.append(win)
        window_labels.append(0 if majority == 1 else 1)

    return windows, window_labels


# ── Full subject pipeline ────────────────────────────────────────────────────

def preprocess_subject(
    data_path: str,
    subject_id: int,
    target_sr: float = TARGET_SR,
    window_sec: float = 60.0,
    overlap: float = 0.5,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Full preprocessing pipeline for one WESAD subject."""
    from src.data_loader import WESADLoader

    loader = WESADLoader(data_path, subject_id).load()

    eda_result = EDAPreprocessor(SAMPLING_RATES["EDA"]).process(loader.wrist_signals["EDA"])
    bvp_result = BVPPreprocessor(SAMPLING_RATES["BVP"]).process(loader.wrist_signals["BVP"])
    temp_result = TemperaturePreprocessor(SAMPLING_RATES["TEMP"]).process(loader.wrist_signals["TEMP"])

    bvp_resampled = resample_signal(bvp_result["bvp_clean"], SAMPLING_RATES["BVP"], target_sr)

    signals = {
        "eda_tonic": eda_result["tonic"],
        "eda_phasic": eda_result["phasic"],
        "eda_clean": eda_result["clean"],
        "bvp": bvp_resampled,
        "temp": temp_result["clean"],
        "temp_grad": temp_result["gradient"],
    }

    min_len = min(len(s) for s in signals.values())
    signals = {k: v[:min_len] for k, v in signals.items()}
    labels = loader.get_label_at_rate(target_sr)[:min_len]

    return create_windows(signals, labels, window_sec, overlap, target_sr)
