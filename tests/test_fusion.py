"""
Automated Tests for Multimodal Late Fusion Engine.
"""

import pytest
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.config import CANONICAL_EMOTIONS


@pytest.fixture
def fusion_engine():
    return MultimodalFusionEngine()


def test_single_modality_passthrough(fusion_engine):
    vision_mock = {
        "dominant_emotion": "happy",
        "confidence": 0.85,
        "probabilities": {e: 0.05 for e in CANONICAL_EMOTIONS}
    }
    vision_mock["probabilities"]["happy"] = 0.70

    result = fusion_engine.fuse({"vision": vision_mock})
    assert result["fused_emotion"] == "happy"
    assert result["confidence"] == 0.85
    assert result["conflict_detected"] is False
    assert result["active_modalities"] == ["vision"]


def test_concordant_multimodal_fusion(fusion_engine):
    # Both vision and text predict 'happy'
    vision_mock = {
        "dominant_emotion": "happy",
        "confidence": 0.80,
        "probabilities": {e: 0.05 for e in CANONICAL_EMOTIONS}
    }
    vision_mock["probabilities"]["happy"] = 0.70

    text_mock = {
        "dominant_emotion": "happy",
        "confidence": 0.90,
        "probabilities": {e: 0.05 for e in CANONICAL_EMOTIONS}
    }
    text_mock["probabilities"]["happy"] = 0.70

    result = fusion_engine.fuse({"vision": vision_mock, "text": text_mock})
    assert result["fused_emotion"] == "happy"
    assert result["confidence"] >= 0.70
    assert result["conflict_detected"] is False


def test_conflicting_multimodal_fusion(fusion_engine):
    # Vision predicts 'happy' (positive), Audio predicts 'angry' (negative high arousal)
    vision_mock = {
        "dominant_emotion": "happy",
        "confidence": 0.80,
        "probabilities": {e: 0.05 for e in CANONICAL_EMOTIONS}
    }
    vision_mock["probabilities"]["happy"] = 0.70

    audio_mock = {
        "dominant_emotion": "angry",
        "confidence": 0.85,
        "probabilities": {e: 0.05 for e in CANONICAL_EMOTIONS}
    }
    audio_mock["probabilities"]["angry"] = 0.70

    result = fusion_engine.fuse({"vision": vision_mock, "audio": audio_mock})
    assert result["conflict_detected"] is True
    assert result["conflict_details"] is not None
    assert "explanation" in result
