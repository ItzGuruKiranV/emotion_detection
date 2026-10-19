"""
Database Access Methods for Prediction History and Analytics.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from src.database.session import get_connection, init_db
from src.utils.logger import get_logger

logger = get_logger("database.models")

# Ensure DB schema exists on import
init_db()


def add_prediction(
    modality: str,
    dominant_emotion: str,
    confidence: float,
    probabilities: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None
) -> int:
    """
    Persist an analysis record into the SQLite database.
    Does NOT store raw user media files for user privacy.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (
                timestamp, modality, dominant_emotion, confidence,
                probabilities_json, metadata_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                modality,
                dominant_emotion,
                round(float(confidence), 4),
                json.dumps(probabilities),
                json.dumps(metadata or {}),
                notes
            )
        )
        conn.commit()
        record_id = cursor.lastrowid
        logger.info(f"Recorded prediction ID {record_id} ({modality} -> {dominant_emotion})")
        return record_id


def get_history(limit: int = 50, modality: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch recent prediction history records."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if modality:
            cursor.execute(
                """
                SELECT id, timestamp, modality, dominant_emotion, confidence,
                       probabilities_json, metadata_json, notes
                FROM prediction_history
                WHERE modality = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (modality, limit)
            )
        else:
            cursor.execute(
                """
                SELECT id, timestamp, modality, dominant_emotion, confidence,
                       probabilities_json, metadata_json, notes
                FROM prediction_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "modality": r["modality"],
                "dominant_emotion": r["dominant_emotion"],
                "confidence": r["confidence"],
                "probabilities": json.loads(r["probabilities_json"]),
                "metadata": json.loads(r["metadata_json"] or "{}"),
                "notes": r["notes"]
            })
        return results


def get_emotion_distribution(modality: Optional[str] = None) -> Dict[str, int]:
    """Return count of predictions per dominant emotion."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if modality:
            cursor.execute(
                """
                SELECT dominant_emotion, COUNT(*) as count
                FROM prediction_history
                WHERE modality = ?
                GROUP BY dominant_emotion
                """,
                (modality,)
            )
        else:
            cursor.execute(
                """
                SELECT dominant_emotion, COUNT(*) as count
                FROM prediction_history
                GROUP BY dominant_emotion
                """
            )
        return {row["dominant_emotion"]: row["count"] for row in cursor.fetchall()}


def clear_history(record_id: Optional[int] = None) -> int:
    """Delete history records (single ID or clear entire history)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if record_id is not None:
            cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        else:
            cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        deleted = cursor.rowcount
        logger.info(f"Deleted {deleted} prediction record(s).")
        return deleted
