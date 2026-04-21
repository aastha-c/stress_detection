"""
Real-Time Stress Detection Dashboard
======================================
Streamlit application that demonstrates live physiological signal
monitoring and stress prediction.

Features:
    - Real-time signal visualization (EDA, Heart Rate, Temperature)
    - Live stress/relaxed classification with color-coded indicators
    - Signal history plots updated every few seconds
    - Feature display for the current window

Launch with:
    streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.signal_simulator import simulate_eda, simulate_bvp, simulate_temperature
from features.feature_extraction import extract_all_features
from preprocessing.bvp_preprocessor import BVPPreprocessor

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Detection — Live Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────
EDA_FS = 4.0      # EDA sampling rate (Hz)
BVP_FS = 64.0     # BVP sampling rate (Hz)
TEMP_FS = 4.0     # Temperature sampling rate (Hz)
WINDOW_SEC = 60   # Feature extraction window (seconds)
MODEL_PATH = os.path.join("model", "stress_model.pkl")


# ── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    """Load the trained RandomForest model from disk."""
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved["feature_names"]
    return None, None


def get_heart_rate_series(bvp_signal: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """
    Convert raw BVP signal to an approximate heart rate time series.

    Detects peaks in the BVP signal and computes instantaneous HR
    between each pair of beats, then interpolates to a uniform time axis.
    """
    preprocessor = BVPPreprocessor(fs=fs)
    bvp_clean = preprocessor.clean(bvp_signal)
    peaks = preprocessor.detect_peaks(bvp_clean)

    if len(peaks) < 2:
        return np.full(len(bvp_signal), 72.0)

    # Instantaneous HR at each peak
    ibi_samples = np.diff(peaks)
    ibi_sec = ibi_samples / fs
    instant_hr = 60.0 / ibi_sec

    # Clamp to physiological range
    instant_hr = np.clip(instant_hr, 40, 200)

    # Interpolate to full signal length
    peak_times = peaks[1:] / fs
    full_times = np.arange(len(bvp_signal)) / fs
    hr_series = np.interp(full_times, peak_times, instant_hr)

    return hr_series


# ── Main Dashboard ───────────────────────────────────────────────────────────
def main():
    st.title("🧠 Real-Time Stress Detection Dashboard")
    st.markdown(
        "Live monitoring of simulated physiological signals with "
        "machine learning-based stress prediction."
    )

    # Load model
    clf, feature_names = load_trained_model()

    if clf is None:
        st.error(
            "⚠️ No trained model found! Please run the training pipeline first:\n\n"
            "```\npython main_pipeline.py\n```"
        )
        st.stop()

    # ── Sidebar Controls ─────────────────────────────────────────────────
    st.sidebar.header("⚙️ Simulation Controls")
    sim_state = st.sidebar.radio(
        "Simulated Condition",
        ["Baseline (Relaxed)", "Stress", "Random (Mixed)"],
        index=2,
    )
    update_interval = st.sidebar.slider(
        "Update Interval (seconds)", min_value=2, max_value=10, value=3
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**How it works:**\n"
        "1. Sensor signals are simulated in real-time\n"
        "2. 60-second windows are extracted\n"
        "3. Physiological features are computed\n"
        "4. RandomForest predicts stress vs relaxed"
    )

    # ── Session State Initialization ─────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = {
            "predictions": [],
            "probabilities": [],
            "timestamps": [],
            "hr_values": [],
            "eda_values": [],
            "temp_values": [],
        }

    # ── Main Layout ──────────────────────────────────────────────────────
    status_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns(4)
    metric_hr = col1.empty()
    metric_eda = col2.empty()
    metric_temp = col3.empty()
    metric_stress = col4.empty()

    st.markdown("---")

    # Signal plots
    chart_col1, chart_col2 = st.columns(2)
    eda_chart = chart_col1.empty()
    hr_chart = chart_col2.empty()

    chart_col3, chart_col4 = st.columns(2)
    temp_chart = chart_col3.empty()
    pred_chart = chart_col4.empty()

    # Feature display
    feature_placeholder = st.empty()

    # ── Real-Time Simulation Loop ────────────────────────────────────────
    st.markdown("---")
    st.info("🔄 The dashboard updates automatically. Press **Stop** in the top-right to halt.")

    iteration = 0
    while True:
        iteration += 1

        # Determine current state
        if sim_state == "Baseline (Relaxed)":
            stressed = False
        elif sim_state == "Stress":
            stressed = True
        else:
            # Random: alternate with some randomness
            stressed = np.random.random() > 0.5

        # ── Simulate one window of data ──────────────────────────────
        eda_signal = simulate_eda(WINDOW_SEC, fs=EDA_FS, stressed=stressed)
        bvp_signal = simulate_bvp(WINDOW_SEC, fs=BVP_FS, stressed=stressed)
        temp_signal = simulate_temperature(WINDOW_SEC, fs=TEMP_FS, stressed=stressed)

        # ── Extract features ─────────────────────────────────────────
        features = extract_all_features(
            eda_signal, bvp_signal, temp_signal,
            eda_fs=EDA_FS, bvp_fs=BVP_FS, temp_fs=TEMP_FS,
        )

        # Build feature vector in correct order
        x = np.array([[features.get(name, 0.0) for name in feature_names]])
        x = np.nan_to_num(x, nan=0.0)

        # ── Predict ──────────────────────────────────────────────────
        prediction = clf.predict(x)[0]
        probability = clf.predict_proba(x)[0]
        stress_prob = probability[1]

        # ── Compute display values ───────────────────────────────────
        hr_series = get_heart_rate_series(bvp_signal, fs=BVP_FS)
        current_hr = np.mean(hr_series)
        current_eda = np.mean(eda_signal)
        current_temp = np.mean(temp_signal)

        # ── Update history ───────────────────────────────────────────
        hist = st.session_state.history
        hist["predictions"].append(prediction)
        hist["probabilities"].append(stress_prob)
        hist["timestamps"].append(iteration * update_interval)
        hist["hr_values"].append(current_hr)
        hist["eda_values"].append(current_eda)
        hist["temp_values"].append(current_temp)

        # Keep last 50 data points
        max_history = 50
        for key in hist:
            if len(hist[key]) > max_history:
                hist[key] = hist[key][-max_history:]

        # ── Status Banner ────────────────────────────────────────────
        if prediction == 1:
            status_placeholder.markdown(
                '<div style="background-color:#ff4b4b;padding:20px;border-radius:10px;'
                'text-align:center;">'
                '<h2 style="color:white;margin:0;">🔴 STRESS DETECTED</h2>'
                f'<p style="color:white;margin:5px 0 0 0;">Confidence: {stress_prob:.0%}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            status_placeholder.markdown(
                '<div style="background-color:#21c354;padding:20px;border-radius:10px;'
                'text-align:center;">'
                '<h2 style="color:white;margin:0;">🟢 RELAXED</h2>'
                f'<p style="color:white;margin:5px 0 0 0;">Confidence: {1 - stress_prob:.0%}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Metric Cards ────────────────────────────────────────────
        metric_hr.metric("❤️ Heart Rate", f"{current_hr:.0f} BPM")
        metric_eda.metric("⚡ EDA Level", f"{current_eda:.2f} μS")
        metric_temp.metric("🌡️ Temperature", f"{current_temp:.1f} °C")
        metric_stress.metric("🎯 Stress Prob", f"{stress_prob:.0%}")

        # ── Signal Charts ────────────────────────────────────────────
        # EDA signal plot
        fig_eda, ax_eda = plt.subplots(figsize=(6, 2.5))
        t_eda = np.arange(len(eda_signal)) / EDA_FS
        ax_eda.plot(t_eda, eda_signal, color="#636EFA", linewidth=1)
        ax_eda.set_title("EDA Signal (Current Window)", fontsize=10)
        ax_eda.set_xlabel("Time (s)", fontsize=8)
        ax_eda.set_ylabel("μS", fontsize=8)
        ax_eda.tick_params(labelsize=7)
        fig_eda.tight_layout()
        eda_chart.pyplot(fig_eda)
        plt.close(fig_eda)

        # Heart rate plot
        fig_hr, ax_hr = plt.subplots(figsize=(6, 2.5))
        t_hr = np.arange(len(hr_series)) / BVP_FS
        ax_hr.plot(t_hr, hr_series, color="#EF553B", linewidth=1)
        ax_hr.set_title("Heart Rate (Current Window)", fontsize=10)
        ax_hr.set_xlabel("Time (s)", fontsize=8)
        ax_hr.set_ylabel("BPM", fontsize=8)
        ax_hr.tick_params(labelsize=7)
        fig_hr.tight_layout()
        hr_chart.pyplot(fig_hr)
        plt.close(fig_hr)

        # Temperature plot
        fig_temp, ax_temp = plt.subplots(figsize=(6, 2.5))
        t_temp = np.arange(len(temp_signal)) / TEMP_FS
        ax_temp.plot(t_temp, temp_signal, color="#00CC96", linewidth=1)
        ax_temp.set_title("Skin Temperature (Current Window)", fontsize=10)
        ax_temp.set_xlabel("Time (s)", fontsize=8)
        ax_temp.set_ylabel("°C", fontsize=8)
        ax_temp.tick_params(labelsize=7)
        fig_temp.tight_layout()
        temp_chart.pyplot(fig_temp)
        plt.close(fig_temp)

        # Prediction history
        fig_pred, ax_pred = plt.subplots(figsize=(6, 2.5))
        times = hist["timestamps"]
        probs = hist["probabilities"]
        colors = ["#ff4b4b" if p > 0.5 else "#21c354" for p in probs]
        ax_pred.bar(range(len(probs)), probs, color=colors, width=0.8)
        ax_pred.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
        ax_pred.set_title("Stress Probability History", fontsize=10)
        ax_pred.set_xlabel("Window #", fontsize=8)
        ax_pred.set_ylabel("P(Stress)", fontsize=8)
        ax_pred.set_ylim(0, 1)
        ax_pred.tick_params(labelsize=7)
        fig_pred.tight_layout()
        pred_chart.pyplot(fig_pred)
        plt.close(fig_pred)

        # ── Feature Table ────────────────────────────────────────────
        with feature_placeholder.container():
            st.subheader("📋 Extracted Features (Current Window)")
            feat_col1, feat_col2, feat_col3 = st.columns(3)

            with feat_col1:
                st.markdown("**EDA Features**")
                st.write(f"• Tonic Mean (SCL): {features.get('eda_tonic_mean', 0):.3f} μS")
                st.write(f"• Tonic Std: {features.get('eda_tonic_std', 0):.4f}")
                st.write(f"• Phasic Mean (SCR): {features.get('eda_phasic_mean', 0):.4f}")
                st.write(f"• SCR Count: {features.get('eda_scr_count', 0):.0f}")
                st.write(f"• Phasic Max: {features.get('eda_phasic_max', 0):.4f}")

            with feat_col2:
                st.markdown("**HRV Features**")
                st.write(f"• Mean HR: {features.get('bvp_mean_hr', 0):.1f} BPM")
                st.write(f"• Mean IBI: {features.get('bvp_mean_ibi', 0):.1f} ms")
                st.write(f"• SDNN: {features.get('bvp_sdnn', 0):.2f} ms")
                st.write(f"• RMSSD: {features.get('bvp_rmssd', 0):.2f} ms")
                st.write(f"• pNN50: {features.get('bvp_pnn50', 0):.1f}%")

            with feat_col3:
                st.markdown("**Temperature Features**")
                st.write(f"• Mean Temp: {features.get('temp_mean', 0):.2f} °C")
                st.write(f"• Temp Std: {features.get('temp_std', 0):.4f}")
                st.write(f"• Gradient Mean: {features.get('temp_gradient_mean', 0):.6f} °C/s")
                st.write(f"• Gradient Max: {features.get('temp_gradient_max', 0):.6f} °C/s")

        # ── Wait before next update ──────────────────────────────────
        time.sleep(update_interval)


if __name__ == "__main__":
    main()
