"""
Automated Tests for Text Modality.
"""

import pytest

from src.text.preprocessor import preprocess_text
from src.text.classifier import TextEmotionClassifier
from src.config import CANONICAL_EMOTIONS
from src.utils.validation import ValidationError


def test_text_preprocessor():
    raw = "   I am   feeling extremely happy   today!   "
    cleaned = preprocess_text(raw)
    assert cleaned == "I am feeling extremely happy today!"


def test_text_preprocessor_invalid_length():
    with pytest.raises(ValidationError):
        preprocess_text("a")


def test_text_classifier_inference():
    classifier = TextEmotionClassifier.get_instance()
    result = classifier.analyze_text("I am feeling absolutely joyful and delighted today!")

    assert result["modality"] == "text"
    assert result["dominant_emotion"] in CANONICAL_EMOTIONS
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == len(CANONICAL_EMOTIONS)
    assert result["dominant_emotion"] == "happy"
    assert "latency_ms" in result
