"""
SQLite Database Session and Initialization.
"""

import sqlite3
from typing import Generator
from contextlib import contextmanager
from src.config import DATABASE_PATH
from src.utils.logger import get_logger

logger = get_logger("database.session")


def init_db() -> None:
    """Initialize SQLite database schema if not present."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                modality TEXT NOT NULL,          -- 'vision', 'audio', 'text', 'multimodal'
                dominant_emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                metadata_json TEXT,
                notes TEXT
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON prediction_history(timestamp);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_modality ON prediction_history(modality);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_emotion ON prediction_history(dominant_emotion);
        """)
        conn.commit()
        logger.info(f"Database initialized at {DATABASE_PATH}")


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
