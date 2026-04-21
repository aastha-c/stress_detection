# 🧠 Stress Detection AI System

Real-time stress detection from wearable physiological signals using deep learning.

Built on the **WESAD** (Wearable Stress and Affect Detection) dataset with signals from the Empatica E4 wrist device — EDA, BVP/HRV, and skin temperature.

## Architecture

```
stress_detection/
├── src/
│   ├── config.py          # Central configuration & hyperparameters
│   ├── data_loader.py     # WESAD dataset loading
│   ├── preprocessing.py   # EDA, BVP/HRV, Temperature preprocessing
│   ├── features.py        # Statistical & time-series feature extraction
│   ├── model.py           # Bidirectional LSTM + Attention network
│   ├── train.py           # Training pipeline with early stopping
│   └── evaluate.py        # Accuracy, precision, recall, F1 evaluation
├── app/
│   └── dashboard.py       # Streamlit interactive dashboard
├── main.py                # CLI entry point — full pipeline
├── requirements.txt
└── README.md
```

## Features

- **Signal Preprocessing**: Butterworth filters, EDA decomposition (tonic/phasic), BVP peak detection, IBI computation, HRV metrics
- **Feature Extraction**: 18 features per signal channel (statistical + time-series) including mean, std, skewness, kurtosis, zero-crossing rate, autocorrelation, energy
- **LSTM Model**: Bidirectional LSTM with attention mechanism, batch normalization, gradient clipping, class-weighted loss
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Dashboard**: Streamlit app with stress timeline, training curves, signal explorer

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd stress_detection
pip install -r requirements.txt
```

### 2. Download the WESAD dataset

Download from: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx

Extract into `data/WESAD/` so the structure is:
```
data/WESAD/
├── S2/S2.pkl
├── S3/S3.pkl
├── ...
└── S17/S17.pkl
```

### 3. Train the model

```bash
# Full training with WESAD data
python main.py

# Demo mode (synthetic data, no dataset needed)
python main.py --demo

# Custom options
python main.py --subjects 2 3 4 --epochs 30
```

### 4. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

## Model Details

| Component | Detail |
|-----------|--------|
| Architecture | Bidirectional LSTM + Attention |
| Hidden Size | 128 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience = 8 epochs |
| Classification | Binary (Baseline vs Stress) |

## Signal Processing Pipeline

1. **EDA**: Median filter → 1 Hz low-pass → Tonic/Phasic decomposition
2. **BVP → HRV**: 0.5–8 Hz bandpass → Peak detection → IBI → HRV metrics (SDNN, RMSSD, pNN50)
3. **Temperature**: Median filter → 0.1 Hz low-pass → Gradient computation
4. All signals resampled to 4 Hz → 60s sliding windows (50% overlap)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Dataset Reference

Schmidt, P., Reiss, A., Duerichen, R., Marber, C., & Van Laerhoven, K. (2018).
*Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.*
ICMI 2018.
