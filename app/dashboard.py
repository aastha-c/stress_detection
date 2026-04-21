"""
Streamlit Dashboard
===================
Interactive stress detection dashboard with real-time prediction and visualization.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, OUTPUT_DIR, LABEL_MAP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT
from src.model import StressLSTM

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Detection AI",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load trained model, scaler, and metadata."""
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    model_path = os.path.join(MODEL_DIR, "stress_lstm.pth")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not all(os.path.exists(p) for p in [meta_path, model_path, scaler_path]):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)
"""
Minimal Streamlit dashboard for mobile stress detection system.
Displays latest heart rate, activity, and stress state from FastAPI backend.
"""
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/latest-sensor-data"

st.set_page_config(page_title="Mobile Stress Detection Dashboard", layout="centered")
st.title("📱 Mobile Stress Detection Dashboard")

sensor_placeholder = st.empty()
activity_placeholder = st.empty()
stress_placeholder = st.empty()

refresh_interval = 3  # seconds

while True:
    try:
        response = requests.get(API_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            heart_rate = data.get("heart_rate", 0)
            activity = data.get("activity", 0)
            stress_state = data.get("stress_state", "Unknown")
            timestamp = data.get("timestamp", 0)

            sensor_placeholder.metric("Heart Rate (bpm)", f"{heart_rate:.1f}")
            activity_placeholder.metric("Activity Level", f"{activity:.2f}")
            stress_placeholder.metric("Stress State", stress_state)
        else:
            st.warning("Failed to fetch data from API.")
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
    time.sleep(refresh_interval)


def plot_training_curves(history):
    """Plot training and validation loss curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Curves", "Validation Accuracy"),
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                   line=dict(color="#636EFA")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                   line=dict(color="#EF553B")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history["val_acc"], name="Val Accuracy",
                   line=dict(color="#00CC96")),
        row=1, col=2,
    )

    fig.update_layout(height=400, template="plotly_white")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    return fig


def plot_confusion_matrix(cm):
    """Plot confusion matrix heatmap."""
    labels = [LABEL_MAP[0], LABEL_MAP[1]]
    fig = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True,
    )
    fig.update_layout(height=350, width=400, template="plotly_white",
                      title="Confusion Matrix")
    return fig


def simulate_realtime_signals(duration_sec=120, fs=4.0, stressed=False):
    """Generate synthetic physiological signals for demo."""
    np.random.seed(None)
    n = int(duration_sec * fs)
    t = np.arange(n) / fs

    if stressed:
        eda = 5.0 + 2.0 * np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.random.randn(n)
        hr = 95 + 15 * np.sin(2 * np.pi * 0.03 * t) + 5 * np.random.randn(n)
        temp = 34.5 - 0.3 * np.sin(2 * np.pi * 0.01 * t) + 0.05 * np.random.randn(n)
    else:
        eda = 2.0 + 0.3 * np.sin(2 * np.pi * 0.02 * t) + 0.1 * np.random.randn(n)
        hr = 72 + 5 * np.sin(2 * np.pi * 0.01 * t) + 2 * np.random.randn(n)
        temp = 33.5 + 0.1 * np.sin(2 * np.pi * 0.005 * t) + 0.02 * np.random.randn(n)

    return t, eda, hr, temp


def plot_physiological_signals(t, eda, hr, temp):
    """Plot EDA, heart rate, and temperature signals."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("EDA (μS)", "Heart Rate (BPM)", "Skin Temperature (°C)"),
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(x=t, y=eda, name="EDA", line=dict(color="#636EFA")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=hr, name="HR", line=dict(color="#EF553B")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp, name="Temp", line=dict(color="#00CC96")), row=3, col=1)

    fig.update_layout(height=600, template="plotly_white", showlegend=False)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


# --- Real-time REST API integration for live wearable data ---
API_URL = "http://localhost:8000/latest-result"

# Store history for plotting
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = []
if 'hrv_history' not in st.session_state:
    st.session_state.hrv_history = []
if 'activity_history' not in st.session_state:
    st.session_state.activity_history = []
if 'stress_history' not in st.session_state:
    st.session_state.stress_history = []

REFRESH_INTERVAL = 5  # seconds

# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("🧠 Stress Detection AI Dashboard")
    st.markdown("Real-time stress detection from wearable physiological signals (WESAD)")

    model, scaler, meta = load_model()
    metrics = load_evaluation_metrics()
    history = load_training_history()

    tab1, tab2, tab3 = st.tabs(["📊 Live Monitor", "📈 Model Performance", "🔬 Signal Explorer"])

    # ── Tab 1: Live Monitor ──────────────────────────────────────────────
    with tab1:
        st.header("Stress Prediction Monitor")

        if metrics and "predictions" in metrics:
            preds = metrics["predictions"]
            probs = metrics["probabilities"]

            col1, col2, col3, col4 = st.columns(4)
            stress_pct = sum(1 for p in preds if p == 1) / len(preds) * 100

            with col1:
                st.metric("Current State",
                          "🔴 Stressed" if preds[-1] == 1 else "🟢 Relaxed")
            with col2:
                st.metric("Stress Probability", f"{probs[-1][1]:.1%}")
            with col3:
                st.metric("Stress Periods", f"{stress_pct:.0f}%")
            with col4:
                st.metric("Windows Analyzed", len(preds))

            st.plotly_chart(plot_stress_timeline(preds, probs), use_container_width=True)
        else:
            st.info("No prediction data available. Train the model first with `python main.py`.")

            # Demo mode
            st.subheader("Demo: Simulated Signals")
            col1, col2 = st.columns(2)
            with col1:
                demo_state = st.selectbox("Simulated state", ["Baseline", "Stressed"])
            with col2:
                demo_duration = st.slider("Duration (seconds)", 30, 300, 120)

            t, eda, hr, temp = simulate_realtime_signals(
                demo_duration, stressed=(demo_state == "Stressed")
            )
            st.plotly_chart(plot_physiological_signals(t, eda, hr, temp), use_container_width=True)

    # ── Tab 2: Model Performance ─────────────────────────────────────────
    with tab2:
        st.header("Model Performance")

        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")

            col_left, col_right = st.columns([1.5, 1])
            with col_left:
                if history:
                    st.plotly_chart(plot_training_curves(history), use_container_width=True)
            with col_right:
                cm = np.array(metrics["confusion_matrix"])
                st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

            with st.expander("Model Architecture"):
                if meta:
                    st.json({
                        "Type": "Bidirectional LSTM + Attention",
                        "Input Features": meta["input_size"],
                        "Sequence Length": meta["seq_length"],
                        "Hidden Size": meta["hidden_size"],
                        "LSTM Layers": meta["num_layers"],
                        "Dropout": meta["dropout"],
                    })
        else:
            st.info("No evaluation data. Train the model first with `python main.py`.")

    # ── Tab 3: Signal Explorer ───────────────────────────────────────────
    with tab3:
        st.header("Signal Explorer")
        st.markdown("Explore how physiological signals differ between baseline and stress.")

        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Signal duration (s)", 30, 600, 180)

        # Baseline signals
        t_b, eda_b, hr_b, temp_b = simulate_realtime_signals(duration, stressed=False)
        # Stressed signals
        t_s, eda_s, hr_s, temp_s = simulate_realtime_signals(duration, stressed=True)

        signal_choice = st.selectbox("Signal", ["EDA", "Heart Rate", "Temperature"])

        fig = go.Figure()
        if signal_choice == "EDA":
            fig.add_trace(go.Scatter(x=t_b, y=eda_b, name="Baseline", line=dict(color="#636EFA")))
            fig.add_trace(go.Scatter(x=t_s, y=eda_s, name="Stressed", line=dict(color="#EF553B")))
            fig.update_yaxes(title_text="EDA (μS)")
        elif signal_choice == "Heart Rate":
            fig.add_trace(go.Scatter(x=t_b, y=hr_b, name="Baseline", line=dict(color="#636EFA")))
            fig.add_trace(go.Scatter(x=t_s, y=hr_s, name="Stressed", line=dict(color="#EF553B")))
            fig.update_yaxes(title_text="BPM")
        else:
            fig.add_trace(go.Scatter(x=t_b, y=temp_b, name="Baseline", line=dict(color="#636EFA")))
            fig.add_trace(go.Scatter(x=t_s, y=temp_s, name="Stressed", line=dict(color="#EF553B")))
            fig.update_yaxes(title_text="°C")

        fig.update_layout(
            height=400, template="plotly_white",
            xaxis_title="Time (s)",
            title=f"{signal_choice}: Baseline vs Stress",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics comparison
        st.subheader("Signal Statistics Comparison")
        stats_data = {
            "Metric": ["Mean EDA", "Std EDA", "Mean HR", "Std HR", "Mean Temp", "Std Temp"],
            "Baseline": [
                f"{np.mean(eda_b):.2f}", f"{np.std(eda_b):.2f}",
                f"{np.mean(hr_b):.1f}", f"{np.std(hr_b):.1f}",
                f"{np.mean(temp_b):.2f}", f"{np.std(temp_b):.3f}",
            ],
            "Stressed": [
                f"{np.mean(eda_s):.2f}", f"{np.std(eda_s):.2f}",
                f"{np.mean(hr_s):.1f}", f"{np.std(hr_s):.1f}",
                f"{np.mean(temp_s):.2f}", f"{np.std(temp_s):.3f}",
            ],
        }
        st.table(stats_data)


if __name__ == "__main__":
    main()
