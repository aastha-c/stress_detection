"""
Real-Time Stress Detection via Webcam using rPPG
====================================================
This script captures video from the webcam, detects the face using MediaPipe,
extracts the average green channel value from the face region, and estimates
heart rate using remote photoplethysmography (rPPG). It then computes simple
HRV features and predicts stress level using rules or a classifier.

Requirements:
- opencv-python
- mediapipe
- numpy
- scipy
- matplotlib

Run: python stress_rppg_webcam.py
"""

import cv2
import numpy as np
import os
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
import time
from collections import deque

# ------------------- rPPG Signal Processing Functions -------------------
def detrend_signal(signal):
    """Remove linear trend from the signal."""
    return scipy.signal.detrend(signal)

def smooth_signal(signal, window_size=5):
    """Smooth signal with moving average."""
    if len(signal) < window_size:
        return signal
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def estimate_heart_rate(signal, fs):
    """Estimate heart rate (BPM) from the dominant frequency in the signal."""
    n = len(signal)
    if n < fs:
        return 0  # Not enough data
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft = np.abs(np.fft.rfft(signal))
    # Typical heart rate range: 0.8–3 Hz (48–180 BPM)
    idx = np.where((freqs >= 0.8) & (freqs <= 3.0))
    if len(idx[0]) == 0:
        return 0
    peak_freq = freqs[idx][np.argmax(fft[idx])]
    bpm = peak_freq * 60
    return bpm

def compute_hrv_features(signal, fs):
    """Compute simple HRV features from the rPPG signal."""
    # Find peaks (heart beats)
    peaks, _ = scipy.signal.find_peaks(signal, distance=fs/2.5)  # min 0.4s between beats
    if len(peaks) < 2:
        return {'ibi_mean': 0, 'ibi_std': 0, 'hrv_rmssd': 0}
    ibi = np.diff(peaks) / fs  # Inter-beat intervals in seconds
    ibi_mean = np.mean(ibi)
    ibi_std = np.std(ibi)
    rmssd = np.sqrt(np.mean(np.square(np.diff(ibi))))
    return {'ibi_mean': ibi_mean, 'ibi_std': ibi_std, 'hrv_rmssd': rmssd}

def predict_stress(bpm, hrv):
    """Rule-based stress prediction: high HR + low HRV = stress."""
    # Example rules (customize as needed)
    if bpm > 95 and hrv['hrv_rmssd'] < 0.04:
        return 'STRESS'
    elif bpm < 60:
        return 'RELAXED'
    else:
        return 'NORMAL'

# ------------------- Main Webcam rPPG Pipeline -------------------
def main():
    # Parameters
    fs = 30  # Sampling rate (video FPS)
    buffer_size = fs * 10  # 10 seconds of data
    green_buffer = deque(maxlen=buffer_size)
    time_buffer = deque(maxlen=buffer_size)

    # OpenCV Haar Cascade face detection
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print('Error: Haar cascade file not found.')
        return
    face_cascade = cv2.CascadeClassifier(haar_path)

    # OpenCV video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='g')
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlim(0, buffer_size)
    ax.set_title('rPPG Green Channel Signal')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Normalized Green')

    last_bpm = 0
    last_stress = 'NORMAL'
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        face_box = None
        if len(faces) > 0:
            (x, y, w_box, h_box) = faces[0]
            face_box = (x, y, x + w_box, y + h_box)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0,255,0), 2)
            # Extract face ROI
            face_roi = frame[y:y + h_box, x:x + w_box]
            if face_roi.size > 0:
                green = np.mean(face_roi[:,:,1])  # Green channel mean
                green_buffer.append(green)
                time_buffer.append(time.time() - t0)
        else:
            # No face detected, append last value or zero
            if len(green_buffer) > 0:
                green_buffer.append(green_buffer[-1])
            else:
                green_buffer.append(0)
            time_buffer.append(time.time() - t0)

        # Signal processing and HR estimation
        if len(green_buffer) >= fs * 5:  # At least 5 seconds of data
            sig = np.array(green_buffer)
            std = np.std(sig)
            if std == 0 or np.isnan(std):
                # Avoid division by zero or NaN
                sig = np.zeros_like(sig)
            else:
                sig = (sig - np.mean(sig)) / std  # Normalize
            sig = detrend_signal(sig)
            sig = smooth_signal(sig, window_size=5)
            bpm = estimate_heart_rate(sig, fs)
            hrv = compute_hrv_features(sig, fs)
            stress = predict_stress(bpm, hrv)
            last_bpm = bpm
            last_stress = stress
        else:
            bpm = last_bpm
            stress = last_stress

        # Display results
        label = f'HR: {bpm:.0f} BPM | Stress: {stress}'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if stress=='STRESS' else (0,255,0), 2)

        cv2.imshow('Stress Detection (rPPG)', frame)

        # Update plot
        if len(green_buffer) > 0:
            ydata = np.array(green_buffer)[-buffer_size:]
            if np.all(np.isfinite(ydata)) and ydata.size > 0:
                line.set_ydata(ydata)
                line.set_xdata(np.arange(len(green_buffer))[-buffer_size:])
                ax.set_xlim(max(0, len(green_buffer)-buffer_size), len(green_buffer))
                ax.figure.canvas.draw()
                ax.figure.canvas.flush_events()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
