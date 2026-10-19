"""
Automated Tests for Evaluation Metrics and Latency Statistics.
"""

from src.evaluation.evaluator import compute_metrics, compute_latency_stats
from src.config import CANONICAL_EMOTIONS


def test_compute_metrics():
    y_true = ["happy", "sad", "angry", "happy", "neutral", "surprise", "fear"]
    y_pred = ["happy", "sad", "angry", "sad", "neutral", "surprise", "fear"]

    metrics = compute_metrics(y_true, y_pred, labels=CANONICAL_EMOTIONS)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] == round(6 / 7, 4)
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == len(CANONICAL_EMOTIONS)
    assert metrics["sample_count"] == 7


def test_latency_stats():
    sample_latencies = [45.2, 52.1, 48.9, 61.3, 44.0, 50.5]
    stats = compute_latency_stats(sample_latencies)

    assert stats["mean_ms"] > 0
    assert stats["median_ms"] > 0
    assert stats["p95_ms"] >= stats["median_ms"]
