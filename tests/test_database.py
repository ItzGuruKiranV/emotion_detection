"""
Automated Tests for SQLite Prediction History and Privacy Deletion.
"""

import pytest
from src.database.models import (
    add_prediction,
    get_history,
    get_emotion_distribution,
    clear_history
)
from src.config import CANONICAL_EMOTIONS


def test_database_crud_operations():
    # Insert test prediction
    test_probs = {e: 0.1 for e in CANONICAL_EMOTIONS}
    test_probs["happy"] = 0.40

    rec_id = add_prediction(
        modality="multimodal",
        dominant_emotion="happy",
        confidence=0.88,
        probabilities=test_probs,
        metadata={"test_run": True},
        notes="Automated test entry"
    )
    assert rec_id > 0

    # Retrieve history
    history = get_history(limit=5)
    assert len(history) > 0
    recent = history[0]
    assert recent["dominant_emotion"] == "happy"
    assert recent["modality"] == "multimodal"

    # Emotion distribution count
    dist = get_emotion_distribution()
    assert "happy" in dist
    assert dist["happy"] >= 1

    # Delete single record
    deleted = clear_history(record_id=rec_id)
    assert deleted == 1
