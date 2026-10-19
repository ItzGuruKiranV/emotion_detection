"""
Evaluation and Benchmarking Script.
Evaluates Vision, Audio, Text, and Multimodal Fusion on benchmark datasets.
Outputs Accuracy, Macro/Weighted F1, Confusion Matrices, and Latency statistics.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from PIL import Image

from src.config import CANONICAL_EMOTIONS, DATA_SAMPLE_DIR
from src.vision.classifier import VisionEmotionClassifier
from src.audio.classifier import AudioEmotionClassifier
from src.text.classifier import TextEmotionClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.evaluation.evaluator import compute_metrics, compute_latency_stats


def evaluate_text_modality():
    """Evaluate text classifier on benchmark dataset."""
    text_benchmark_path = DATA_SAMPLE_DIR / "texts" / "sample_benchmark_texts.json"
    if not text_benchmark_path.exists():
        print("Text benchmark dataset missing. Run scripts/create_sample_assets.py first.")
        return None

    with open(text_benchmark_path, "r") as f:
        samples = json.load(f)

    classifier = TextEmotionClassifier.get_instance()

    y_true, y_pred, latencies = [], [], []
    for item in samples:
        t0 = time.perf_counter()
        res = classifier.analyze_text(item["text"])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        y_true.append(item["expected_emotion"])
        y_pred.append(res["dominant_emotion"])

    metrics = compute_metrics(y_true, y_pred, labels=CANONICAL_EMOTIONS)
    lat_stats = compute_latency_stats(latencies)

    return {
        "modality": "text",
        "sample_count": len(samples),
        "metrics": metrics,
        "latency": lat_stats
    }


def evaluate_audio_modality():
    """Evaluate audio classifier on sample acoustic files."""
    audio_dir = DATA_SAMPLE_DIR / "audio"
    if not audio_dir.exists():
        return None

    ground_truth = {
        "sample_happy_tone.wav": "happy",
        "sample_sad_tone.wav": "sad",
        "sample_angry_tone.wav": "angry",
        "sample_neutral_tone.wav": "neutral"
    }

    classifier = AudioEmotionClassifier.get_instance()
    y_true, y_pred, latencies = [], [], []

    for fname, expected in ground_truth.items():
        p = audio_dir / fname
        if not p.exists():
            continue
        t0 = time.perf_counter()
        res = classifier.analyze_audio(p)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        y_true.append(expected)
        y_pred.append(res["dominant_emotion"])

    if not y_true:
        return None

    metrics = compute_metrics(y_true, y_pred, labels=CANONICAL_EMOTIONS)
    lat_stats = compute_latency_stats(latencies)

    return {
        "modality": "audio",
        "sample_count": len(y_true),
        "metrics": metrics,
        "latency": lat_stats
    }


def evaluate_vision_modality():
    """Evaluate vision classifier on sample face images."""
    img_dir = DATA_SAMPLE_DIR / "images"
    if not img_dir.exists():
        return None

    ground_truth = {
        "sample_happy_face.jpg": "happy",
        "sample_sad_face.jpg": "sad"
    }

    classifier = VisionEmotionClassifier.get_instance()
    y_true, y_pred, latencies = [], [], []

    for fname, expected in ground_truth.items():
        p = img_dir / fname
        if not p.exists():
            continue
        img = Image.open(p)
        t0 = time.perf_counter()
        res = classifier.analyze_image(img, annotate=False)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        y_true.append(expected)
        y_pred.append(res["dominant_emotion"])

    if not y_true:
        return None

    metrics = compute_metrics(y_true, y_pred, labels=CANONICAL_EMOTIONS)
    lat_stats = compute_latency_stats(latencies)

    return {
        "modality": "vision",
        "sample_count": len(y_true),
        "metrics": metrics,
        "latency": lat_stats
    }


def main():
    print("=" * 70)
    print("MULTIMODAL EMOTION ANALYSIS PLATFORM - BENCHMARK EVALUATION")
    print("=" * 70)

    results = {}

    print("\n1. Evaluating Text Emotion Classifier (BERT)...")
    text_res = evaluate_text_modality()
    if text_res:
        results["text"] = text_res
        m = text_res["metrics"]
        lat = text_res["latency"]
        print(f"   Accuracy: {m['accuracy']*100:.1f}% | Macro F1: {m['macro_f1']:.3f} | Latency: {lat['mean_ms']} ms")

    print("\n2. Evaluating Audio Emotion Classifier (Acoustic SER)...")
    audio_res = evaluate_audio_modality()
    if audio_res:
        results["audio"] = audio_res
        m = audio_res["metrics"]
        lat = audio_res["latency"]
        print(f"   Accuracy: {m['accuracy']*100:.1f}% | Macro F1: {m['macro_f1']:.3f} | Latency: {lat['mean_ms']} ms")

    print("\n3. Evaluating Vision Emotion Classifier (MobileNetV2)...")
    vision_res = evaluate_vision_modality()
    if vision_res:
        results["vision"] = vision_res
        m = vision_res["metrics"]
        lat = vision_res["latency"]
        print(f"   Accuracy: {m['accuracy']*100:.1f}% | Macro F1: {m['macro_f1']:.3f} | Latency: {lat['mean_ms']} ms")

    report_path = PROJECT_ROOT / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed evaluation report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
