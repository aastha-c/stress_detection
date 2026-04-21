"""
Stress Detection AI — Main Entry Point
=======================================
Full pipeline: preprocess → extract features → train LSTM → evaluate → save.
"""

import sys
import os
import argparse
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR, WESAD_SUBJECTS
from src.train import load_and_extract, train_model, save_artifacts
from src.evaluate import evaluate_model


def run_demo_training():
    """Run training with synthetic data when WESAD dataset is not available."""
    from src.features import prepare_sequences

    print("\n  WESAD dataset not found — running demo with synthetic data.\n")

    np.random.seed(42)
    n_windows = 200
    n_features = 108  # typical feature count for 6 signals × 18 features
    seq_length = 5

    # Simulate feature matrix: stress windows have higher mean
    X_baseline = np.random.randn(n_windows // 2, n_features).astype(np.float32)
    X_stress = np.random.randn(n_windows // 2, n_features).astype(np.float32) + 0.8
    feature_matrix = np.vstack([X_baseline, X_stress])
    labels = np.array([0] * (n_windows // 2) + [1] * (n_windows // 2), dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(labels))
    feature_matrix = feature_matrix[idx]
    labels = labels[idx]

    X_seq, y_seq = prepare_sequences(feature_matrix, labels, seq_length)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"  Synthetic sequences: {X_seq.shape}")
    print(f"  Class distribution: baseline={np.sum(y_seq == 0)}, stress={np.sum(y_seq == 1)}")

    return X_seq, y_seq, feature_names


def main():
    parser = argparse.ArgumentParser(description="Stress Detection AI System")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to WESAD dataset directory")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject IDs to use (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data for demonstration")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stress Detection AI System")
    print("  WESAD Dataset — LSTM Classifier")
    print("=" * 60)

    if args.epochs:
        import src.config as cfg
        cfg.NUM_EPOCHS = args.epochs

    # ── Step 1: Load & Extract Features ──────────────────────────────────
    subjects = args.subjects or WESAD_SUBJECTS
    use_demo = args.demo

    if not use_demo:
        # Check if any subject data exists
        found = any(
            os.path.exists(os.path.join(args.data_dir, f"S{s}", f"S{s}.pkl"))
            for s in subjects
        )
        if not found:
            use_demo = True

    if use_demo:
        X, y, feature_names = run_demo_training()
    else:
        print(f"\n[1/4] Loading & preprocessing {len(subjects)} subjects...")
        X, y, feature_names = load_and_extract(args.data_dir, subjects)

    # ── Step 2: Train ────────────────────────────────────────────────────
    print(f"\n[2/4] Training LSTM model...")
    result = train_model(X, y, feature_names)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────
    print(f"\n[3/4] Evaluating model...")
    metrics = evaluate_model(result)

    # ── Step 4: Save ─────────────────────────────────────────────────────
    print(f"\n[4/4] Saving model artifacts...")
    save_artifacts(result)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Launch dashboard: streamlit run app/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
