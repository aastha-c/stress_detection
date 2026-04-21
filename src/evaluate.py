"""
Evaluation
==========
Model evaluation with accuracy, precision, recall, F1, confusion matrix.
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from typing import Dict

from src.config import OUTPUT_DIR, LABEL_MAP


def evaluate_model(result: Dict) -> Dict:
    """
    Evaluate the trained model on the test set.

    Parameters
    ----------
    result : dict returned by train_model()

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1, confusion matrix, report
    """
    model = result["model"]
    device = result["device"]
    X_test = torch.FloatTensor(result["X_test"]).to(device)
    y_test = result["y_test"]

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_prob = torch.softmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": y_prob.tolist(),
        "ground_truth": y_test.tolist(),
        "report": report,
    }

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    serializable = {k: v for k, v in metrics.items() if k != "report"}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{report}")
    print(f"  Metrics saved to {metrics_path}")

    return metrics
