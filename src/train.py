"""
Training Pipeline
=================
End-to-end training: data loading, feature extraction, LSTM training with early stopping.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import pickle

from src.config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR, WESAD_SUBJECTS, TARGET_SR,
    WINDOW_SEC, WINDOW_OVERLAP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
)
from src.preprocessing import preprocess_subject
from src.features import extract_all_features, prepare_sequences
from src.model import StressLSTM


def load_and_extract(
    data_path: str,
    subject_ids: List[int],
    seq_length: int = 5,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load WESAD subjects, preprocess, and extract features."""
    all_windows: List[Dict[str, np.ndarray]] = []
    all_labels: List[int] = []

    for sid in subject_ids:
        print(f"  Processing subject S{sid}...")
        try:
            windows, labels = preprocess_subject(
                data_path, sid, TARGET_SR, WINDOW_SEC, WINDOW_OVERLAP
            )
            all_windows.extend(windows)
            all_labels.extend(labels)
            print(f"    → {len(windows)} windows (baseline={labels.count(0)}, stress={labels.count(1)})")
        except FileNotFoundError as e:
            print(f"    ⚠ Skipped: {e}")
            continue

    if not all_windows:
        raise RuntimeError("No data loaded. Ensure WESAD dataset is available.")

    print(f"\n  Total windows: {len(all_windows)}")
    feature_matrix, feature_names, labels_arr = extract_all_features(all_windows, all_labels)
    print(f"  Feature matrix: {feature_matrix.shape}")

    X_seq, y_seq = prepare_sequences(feature_matrix, labels_arr, seq_length)
    print(f"  Sequences: {X_seq.shape}")
    return X_seq, y_seq, feature_names


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    test_size: float = 0.2,
    device: str = "auto",
) -> Dict:
    """
    Train LSTM model with early stopping.

    Returns dict with model, scaler, metrics, and training history.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # ── Train/test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Normalize features ───────────────────────────────────────────────
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, seq_len, n_feat)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_feat)).reshape(
        X_test.shape[0], seq_len, n_feat
    )

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ── Class weights ────────────────────────────────────────────────────
    class_counts = np.bincount(y_train)
    weights = torch.FloatTensor(1.0 / class_counts).to(device)
    weights = weights / weights.sum()

    # ── Model ────────────────────────────────────────────────────────────
    model = StressLSTM(
        input_size=n_feat,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # ── Training loop ────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n  Training for up to {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(test_ds)
        val_acc = correct / len(test_ds)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "history": history,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "device": device,
        "input_size": n_feat,
        "seq_length": X.shape[1],
    }


def save_artifacts(result: Dict) -> str:
    """Save trained model, scaler, and metadata."""
    model_path = os.path.join(MODEL_DIR, "stress_lstm.pth")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")

    torch.save(result["model"].state_dict(), model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(result["scaler"], f)

    meta = {
        "input_size": result["input_size"],
        "seq_length": result["seq_length"],
        "feature_names": result["feature_names"],
        "hidden_size": LSTM_HIDDEN_SIZE,
        "num_layers": LSTM_NUM_LAYERS,
        "dropout": LSTM_DROPOUT,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)

    print(f"\n  Model saved to {model_path}")
    print(f"  Scaler saved to {scaler_path}")
    print(f"  Metadata saved to {meta_path}")
    return model_path
