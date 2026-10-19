"""
Evaluation and Benchmarking Module.
Computes accuracy, precision, recall, F1, confusion matrices, and latency stats.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from src.config import CANONICAL_EMOTIONS
from src.utils.logger import get_logger

logger = get_logger("evaluation.evaluator")


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = CANONICAL_EMOTIONS
) -> Dict[str, Any]:
    """
    Calculate classification metrics:
    Accuracy, Macro F1, Weighted F1, Per-class breakdown, and Confusion Matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")

    acc = float(accuracy_score(y_true, y_pred))

    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1_score": round(float(f1_per[i]), 4),
            "support": int(support_per[i])
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "sample_count": len(y_true)
    }


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute mean, median, standard deviation, and p95 latency."""
    if not latencies_ms:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0}

    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2)
    }
