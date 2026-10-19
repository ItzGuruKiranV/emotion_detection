"""
Automated Tests for Audio Modality (Speech Emotion Recognition).
"""

import pytest
import numpy as np

from src.audio.preprocessor import load_and_preprocess_audio
from src.audio.feature_extractor import extract_acoustic_features
from src.audio.classifier import AudioEmotionClassifier
from src.config import CANONICAL_EMOTIONS, AUDIO_TARGET_SR
from src.utils.validation import ValidationError


@pytest.fixture
def synthetic_audio_clip():
    """Generate 1.5 seconds of synthetic audio tone."""
    sr = AUDIO_TARGET_SR
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 220Hz tone with slight harmonic
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


def test_audio_preprocessor_validation(synthetic_audio_clip):
    # Test valid signal
    waveform, sr, duration = load_and_preprocess_audio(synthetic_audio_clip.tobytes())
    assert sr == AUDIO_TARGET_SR
    assert duration >= 1.0
    assert waveform.ndim == 1


def test_audio_feature_extraction(synthetic_audio_clip):
    features, mel_spec, summary = extract_acoustic_features(synthetic_audio_clip, sr=AUDIO_TARGET_SR)
    assert features.ndim == 1
    assert len(features) == 49
    assert mel_spec.shape[0] == 128  # 128 mel frequency bins
    assert "mean_pitch_hz" in summary
    assert "mean_rms_energy" in summary


def test_audio_classifier_inference(synthetic_audio_clip):
    classifier = AudioEmotionClassifier.get_instance()
    result = classifier.analyze_audio(synthetic_audio_clip.tobytes())

    assert result["modality"] == "audio"
    assert result["dominant_emotion"] in CANONICAL_EMOTIONS
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == len(CANONICAL_EMOTIONS)
    assert result["duration_sec"] > 0
    assert "latency_ms" in result
