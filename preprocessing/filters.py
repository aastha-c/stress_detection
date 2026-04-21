"""
Butterworth Filter Utilities
=============================
Provides low-pass and band-pass Butterworth filters used across
all signal preprocessing modules (EDA, BVP, Temperature).

A Butterworth filter has a maximally flat frequency response in the
passband, making it ideal for physiological signal processing where
we want to preserve signal shape while removing noise.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """
    Design a Butterworth low-pass filter.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies above this are attenuated.
    fs : float
        Sampling rate of the signal in Hz.
    order : int
        Filter order. Higher = sharper cutoff but more ringing.

    Returns
    -------
    b, a : ndarray
        Numerator and denominator polynomial coefficients of the filter.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return b, a


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Design a Butterworth band-pass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling rate of the signal in Hz.
    order : int
        Filter order.

    Returns
    -------
    b, a : ndarray
        Filter coefficients.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a


def apply_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase low-pass Butterworth filter.

    Uses filtfilt (forward-backward filtering) to avoid phase distortion,
    which is critical for preserving temporal alignment of physiological events.

    Parameters
    ----------
    data : np.ndarray
        Input signal (1-D array).
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling rate in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal with same length as input.
    """
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def apply_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase band-pass Butterworth filter.

    Keeps only frequencies between lowcut and highcut Hz.

    Parameters
    ----------
    data : np.ndarray
        Input signal (1-D array).
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling rate in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)
