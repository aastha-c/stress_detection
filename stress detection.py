"""
Signal Preprocessing Module
============================
Handles loading WESAD dataset and preprocessing EDA, BVP (for HRV),
and temperature signals from wrist-worn Empatica E4 device.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt
import pickle
import os
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# ── WESAD constants ──────────────────────────────────────────────────────────
WESAD_LABELS = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

# Empatica E4 sampling rates (wrist)
SAMPLING_RATES = {
    "ACC": 32,    # Accelerometer
    "BVP": 64,    # Blood Volume Pulse
    "EDA": 4,     # Electrodermal Activity
    "TEMP": 4,    # Skin Temperature
}

LABEL_SR = 700  # Label sampling rate (from RespiBAN)


# ── Filter utilities ─────────────────────────────────────────────────────────
def butter_lowpass(cutoff: float, fs: float, order: int = 4) -> Tuple:
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple:
    """Design a Butterworth band-pass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a


def apply_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply a low-pass Butterworth filter to the signal."""
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def apply_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a band-pass Butterworth filter to the signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)


# ── Dataset loader ───────────────────────────────────────────────────────────
class WESADLoader:
    """Load and preprocess the WESAD dataset for a single subject."""

    def __init__(self, data_path: str, subject_id: int):
        self.data_path = data_path
        self.subject_id = subject_id
        self.subject_key = f"S{subject_id}"
        self.raw_data: Optional[Dict] = None
        self.wrist_signals: Dict[str, np.ndarray] = {}
        self.labels: Optional[np.ndarray] = None

    def load(self) -> "WESADLoader":
        """Load raw pickle data for one subject."""
        pkl_path = os.path.join(
            self.data_path, self.subject_key, f"{self.subject_key}.pkl"
        )
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"WESAD file not found: {pkl_path}\n"
                f"Download the dataset from: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx"
            )
        with open(pkl_path, "rb") as f:
            self.raw_data = pickle.load(f, encoding="latin1")

        # Extract wrist sensor data
        wrist = self.raw_data["signal"]["wrist"]
        self.wrist_signals = {
            "ACC": np.array(wrist["ACC"]),
            "BVP": np.array(wrist["BVP"]).flatten(),
            "EDA": np.array(wrist["EDA"]).flatten(),
            "TEMP": np.array(wrist["TEMP"]).flatten(),
        }
        self.labels = np.array(self.raw_data["label"]).flatten()
        return self

    def get_label_at_rate(self, target_sr: float) -> np.ndarray:
        """Resample labels from 700 Hz to a target sampling rate."""
        n_target = int(len(self.labels) * target_sr / LABEL_SR)
        indices = np.linspace(0, len(self.labels) - 1, n_target).astype(int)
        return self.labels[indices]


# ── Signal preprocessors ─────────────────────────────────────────────────────
class EDAPreprocessor:
    """Preprocess Electrodermal Activity (skin conductance) signal."""

    def __init__(self, fs: float = 4.0):
        self.fs = fs

    def clean(self, eda_raw: np.ndarray) -> np.ndarray:
        """Remove artifacts and smooth the EDA signal."""
        # Median filter to remove spikes
        eda_med = medfilt(eda_raw, kernel_size=5)
        # Low-pass filter at 1 Hz (EDA is slow-varying)
        eda_filtered = apply_lowpass_filter(eda_med, cutoff=1.0, fs=self.fs, order=4)
        return eda_filtered

    def decompose(self, eda_clean: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose EDA into tonic (SCL) and phasic (SCR) components."""
        # Tonic: very low-frequency component (baseline skin conductance level)
        tonic = apply_lowpass_filter(eda_clean, cutoff=0.05, fs=self.fs, order=2)
        # Phasic: rapid changes (skin conductance responses)
        phasic = eda_clean - tonic
        return {"tonic": tonic, "phasic": phasic, "clean": eda_clean}

    def process(self, eda_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Full preprocessing pipeline for EDA."""
        eda_clean = self.clean(eda_raw)
        components = self.decompose(eda_clean)
        return components


class BVPPreprocessor:
    """Preprocess Blood Volume Pulse signal for HRV extraction."""

    def __init__(self, fs: float = 64.0):
        self.fs = fs

    def clean(self, bvp_raw: np.ndarray) -> np.ndarray:
        """Filter BVP signal: bandpass 0.5–8 Hz."""
        bvp_filtered = apply_bandpass_filter(
            bvp_raw, lowcut=0.5, highcut=8.0, fs=self.fs, order=3
        )
        return bvp_filtered

    def detect_peaks(self, bvp_clean: np.ndarray) -> np.ndarray:
        """Detect systolic peaks (heartbeats) in the clean BVP signal."""
        # Minimum distance between peaks ≈ 0.4 s (150 BPM max)
        min_distance = int(0.4 * self.fs)
        peaks, _ = signal.find_peaks(
            -bvp_clean,  # BVP from E4 is inverted
            distance=min_distance,
            prominence=np.std(bvp_clean) * 0.3,
        )
        return peaks

    def compute_ibi(self, peaks: np.ndarray) -> np.ndarray:
        """Compute inter-beat intervals (IBI) in milliseconds."""
        ibi = np.diff(peaks) / self.fs * 1000.0  # convert to ms
        # Remove physiologically implausible IBIs
        valid_mask = (ibi > 300) & (ibi < 1500)  # 40–200 BPM
        return ibi[valid_mask]

    def compute_hrv_metrics(self, ibi: np.ndarray) -> Dict[str, float]:
        """Compute time-domain HRV metrics from IBI series."""
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
        pnn50 = (nn50 / len(diff_ibi)) * 100 if len(diff_ibi) > 0 else 0

        return {
            "mean_hr": mean_hr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "mean_ibi": mean_ibi,
        }

    def process(self, bvp_raw: np.ndarray) -> Dict:
        """Full preprocessing pipeline for BVP → HRV."""
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


class TemperaturePreprocessor:
    """Preprocess skin temperature signal."""

    def __init__(self, fs: float = 4.0):
        self.fs = fs

    def clean(self, temp_raw: np.ndarray) -> np.ndarray:
        """Smooth temperature signal and remove outliers."""
        # Median filter for spike removal
        temp_med = medfilt(temp_raw, kernel_size=3)
        # Low-pass at 0.1 Hz (temperature changes very slowly)
        temp_filtered = apply_lowpass_filter(temp_med, cutoff=0.1, fs=self.fs, order=2)
        return temp_filtered

    def process(self, temp_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Full preprocessing pipeline for temperature."""
        temp_clean = self.clean(temp_raw)
        # Compute rate of change
        temp_gradient = np.gradient(temp_clean, 1.0 / self.fs)
        return {"clean": temp_clean, "gradient": temp_gradient}


# ── Windowed data generator ──────────────────────────────────────────────────
def create_windows(
    signals: Dict[str, np.ndarray],
    labels: np.ndarray,
    window_sec: float = 60.0,
    overlap: float = 0.5,
    target_sr: float = 4.0,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """
    Segment signals into fixed-length windows with overlap.

    Parameters
    ----------
    signals : dict mapping signal name → 1-D array (all at target_sr)
    labels  : 1-D label array at target_sr
    window_sec : window duration in seconds
    overlap : fraction of overlap (0–1)
    target_sr : common sampling rate

    Returns
    -------
    windows : list of dicts  {signal_name: np.ndarray}
    window_labels : list of int (majority vote per window)
    """
    window_len = int(window_sec * target_sr)
    step = int(window_len * (1 - overlap))

    n_samples = min(len(labels), *(len(s) for s in signals.values()))

    windows = []
    window_labels = []

    for start in range(0, n_samples - window_len + 1, step):
        end = start + window_len
        seg_labels = labels[start:end]

        # Majority-vote label for the window
        unique, counts = np.unique(seg_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        # Keep only baseline (1) and stress (2) for binary classification
        if majority_label not in (1, 2):
            continue

        win = {name: sig[start:end] for name, sig in signals.items()}
        windows.append(win)
        # Binary label: 0 = baseline, 1 = stress
        window_labels.append(0 if majority_label == 1 else 1)

    return windows, window_labels


def resample_signal(sig: np.ndarray, original_sr: float, target_sr: float) -> np.ndarray:
    """Resample a signal to a target sampling rate."""
    n_target = int(len(sig) * target_sr / original_sr)
    return signal.resample(sig, n_target)


def preprocess_subject(
    data_path: str,
    subject_id: int,
    target_sr: float = 4.0,
    window_sec: float = 60.0,
    overlap: float = 0.5,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """
    Complete preprocessing pipeline for one WESAD subject.

    Returns windowed signal segments and binary labels.
    """
    # Load raw data
    loader = WESADLoader(data_path, subject_id)
    loader.load()

    # Preprocess each modality
    eda_proc = EDAPreprocessor(fs=SAMPLING_RATES["EDA"])
    bvp_proc = BVPPreprocessor(fs=SAMPLING_RATES["BVP"])
    temp_proc = TemperaturePreprocessor(fs=SAMPLING_RATES["TEMP"])

    eda_result = eda_proc.process(loader.wrist_signals["EDA"])
    bvp_result = bvp_proc.process(loader.wrist_signals["BVP"])
    temp_result = temp_proc.process(loader.wrist_signals["TEMP"])

    # Resample BVP-derived features to target SR
    bvp_resampled = resample_signal(
        bvp_result["bvp_clean"], SAMPLING_RATES["BVP"], target_sr
    )

    # Build signal dict at common SR
    signals = {
        "eda_tonic": eda_result["tonic"],
        "eda_phasic": eda_result["phasic"],
        "eda_clean": eda_result["clean"],
        "bvp": bvp_resampled,
        "temp": temp_result["clean"],
        "temp_grad": temp_result["gradient"],
    }

    # Trim all signals to same length
    min_len = min(len(s) for s in signals.values())
    signals = {k: v[:min_len] for k, v in signals.items()}

    # Get labels at target SR
    labels = loader.get_label_at_rate(target_sr)[:min_len]

    # Create windows
    windows, window_labels = create_windows(
        signals, labels, window_sec=window_sec, overlap=overlap, target_sr=target_sr
    )

    return windows, window_labels


# ── Demo with synthetic data ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Stress Detection — Signal Preprocessing Demo")
    print("=" * 60)

    duration_sec = 300  # 5 minutes of fake data
    np.random.seed(42)

    # --- Synthetic EDA signal (4 Hz) ---
    fs_eda = SAMPLING_RATES["EDA"]
    n_eda = duration_sec * fs_eda
    t_eda = np.arange(n_eda) / fs_eda
    eda_raw = 2.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t_eda) + 0.1 * np.random.randn(n_eda)

    print(f"\n[EDA] Raw signal: {n_eda} samples @ {fs_eda} Hz ({duration_sec}s)")
    eda_proc = EDAPreprocessor(fs=fs_eda)
    eda_result = eda_proc.process(eda_raw)
    print(f"  Tonic  — mean: {eda_result['tonic'].mean():.3f}, std: {eda_result['tonic'].std():.3f}")
    print(f"  Phasic — mean: {eda_result['phasic'].mean():.3f}, std: {eda_result['phasic'].std():.3f}")

    # --- Synthetic BVP signal (64 Hz) ---
    fs_bvp = SAMPLING_RATES["BVP"]
    n_bvp = duration_sec * fs_bvp
    t_bvp = np.arange(n_bvp) / fs_bvp
    heart_rate_hz = 1.2  # ~72 BPM
    bvp_raw = -np.sin(2 * np.pi * heart_rate_hz * t_bvp) + 0.3 * np.random.randn(n_bvp)

    print(f"\n[BVP] Raw signal: {n_bvp} samples @ {fs_bvp} Hz ({duration_sec}s)")
    bvp_proc = BVPPreprocessor(fs=fs_bvp)
    bvp_result = bvp_proc.process(bvp_raw)
    print(f"  Peaks detected: {len(bvp_result['peaks'])}")
    print(f"  IBI samples:    {len(bvp_result['ibi'])}")
    for k, v in bvp_result["hrv"].items():
        print(f"  HRV {k:>10s}: {v:.2f}")

    # --- Synthetic Temperature signal (4 Hz) ---
    fs_temp = SAMPLING_RATES["TEMP"]
    n_temp = duration_sec * fs_temp
    t_temp = np.arange(n_temp) / fs_temp
    temp_raw = 33.0 + 0.5 * np.sin(2 * np.pi * 0.005 * t_temp) + 0.05 * np.random.randn(n_temp)

    print(f"\n[TEMP] Raw signal: {n_temp} samples @ {fs_temp} Hz ({duration_sec}s)")
    temp_proc = TemperaturePreprocessor(fs=fs_temp)
    temp_result = temp_proc.process(temp_raw)
    print(f"  Clean — mean: {temp_result['clean'].mean():.2f} C, std: {temp_result['clean'].std():.3f}")

    # --- Windowing demo ---
    bvp_resampled = resample_signal(bvp_result["bvp_clean"], fs_bvp, fs_eda)
    signals = {
        "eda_tonic": eda_result["tonic"],
        "eda_phasic": eda_result["phasic"],
        "bvp": bvp_resampled,
        "temp": temp_result["clean"],
    }
    min_len = min(len(s) for s in signals.values())
    signals = {k: v[:min_len] for k, v in signals.items()}

    # Fake labels: first half baseline (1), second half stress (2)
    labels = np.concatenate([
        np.ones(min_len // 2, dtype=int),
        np.full(min_len - min_len // 2, 2, dtype=int),
    ])
    windows, window_labels = create_windows(
        signals, labels, window_sec=60, overlap=0.5, target_sr=fs_eda
    )

    n_baseline = sum(1 for lbl in window_labels if lbl == 0)
    n_stress = sum(1 for lbl in window_labels if lbl == 1)
    print(f"\n[WINDOWS] 60s windows, 50% overlap")
    print(f"  Total: {len(windows)} | Baseline: {n_baseline} | Stress: {n_stress}")
    print(f"  Signals per window: {list(windows[0].keys())}")
    print(f"  Samples per window: {len(windows[0]['eda_tonic'])}")

    print("\n" + "=" * 60)
    print("  Demo complete! All preprocessing modules working.")
    print("=" * 60)
