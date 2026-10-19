"""
Automated Tests for Vision Modality.
"""

import pytest
import numpy as np
from PIL import Image

from src.vision.face_detector import FaceDetector
from src.vision.preprocessor import preprocess_face_crop
from src.vision.classifier import VisionEmotionClassifier
from src.config import CANONICAL_EMOTIONS, IMAGE_INPUT_SIZE


@pytest.fixture
def sample_image():
    """Create a test RGB image."""
    arr = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_face_detector_instantiation():
    detector = FaceDetector.get_instance()
    assert detector is not None
    assert detector.cascade is not None


def test_vision_preprocessor(sample_image):
    tensor = preprocess_face_crop(sample_image, bbox=None, target_size=IMAGE_INPUT_SIZE)
    assert tensor.shape == (1, 96, 96, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_vision_classifier_analyze(sample_image):
    classifier = VisionEmotionClassifier.get_instance()
    result = classifier.analyze_image(sample_image, annotate=True)

    assert result["modality"] == "vision"
    assert result["dominant_emotion"] in CANONICAL_EMOTIONS
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == len(CANONICAL_EMOTIONS)
    assert "latency_ms" in result
    assert result["annotated_image"] is not None
