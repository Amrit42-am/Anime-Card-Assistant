"""
src/memory/database.py

SQLite persistent database for the Learned Blur Memory system.
Stores metadata, confirmation status, and paths for all user-confirmed
and corrected blurred card examples.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
from typing import List, Optional, Dict, Any

logger = logging.getLogger("memory_database")

DEFAULT_DB_PATH = Path("data/card_memory.db")


@dataclass
class BlurSampleRecord:
    sample_id: int
    card_id: int
    card_name: str
    image_path: str
    is_confirmed: bool
    confidence: float
    source: str
    notes: Optional[str]
    created_at: str
    updated_at: str


class MemoryDatabase:
    """Manages SQLite persistence for learned blur samples."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS learned_blurs (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER NOT NULL,
                card_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                is_confirmed INTEGER NOT NULL DEFAULT 1,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'user_confirmed',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_card_id ON learned_blurs (card_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_confirmed ON learned_blurs (is_confirmed);")
            conn.commit()

    def insert_sample(
        self,
        card_id: int,
        card_name: str,
        image_path: str,
        is_confirmed: bool = True,
        confidence: float = 1.0,
        source: str = "user_confirmed",
        notes: str = ""
    ) -> int:
        """Inserts a new learned blur record and returns sample_id."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO learned_blurs (
                card_id, card_name, image_path, is_confirmed, confidence, source, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (card_id, card_name, str(image_path), 1 if is_confirmed else 0, confidence, source, notes))
            conn.commit()
            return cur.lastrowid

    def get_all_confirmed(self) -> List[BlurSampleRecord]:
        """Returns all confirmed active blur samples."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT * FROM learned_blurs
            WHERE is_confirmed = 1
            ORDER BY sample_id ASC;
            """)
            rows = cur.fetchall()
            return [
                BlurSampleRecord(
                    sample_id=r["sample_id"],
                    card_id=r["card_id"],
                    card_name=r["card_name"],
                    image_path=r["image_path"],
                    is_confirmed=bool(r["is_confirmed"]),
                    confidence=float(r["confidence"]),
                    source=r["source"],
                    notes=r["notes"],
                    created_at=str(r["created_at"]),
                    updated_at=str(r["updated_at"])
                )
                for r in rows
            ]

    def update_card_identity(self, sample_id: int, new_card_id: int, new_card_name: str):
        """Corrects the assigned identity of a learned sample."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            UPDATE learned_blurs
            SET card_id = ?, card_name = ?, source = 'manual_correction', updated_at = CURRENT_TIMESTAMP
            WHERE sample_id = ?;
            """, (new_card_id, new_card_name, sample_id))
            conn.commit()
            logger.info("Updated sample %d identity to #%d %s", sample_id, new_card_id, new_card_name)

    def invalidate_sample(self, sample_id: int):
        """Deactivates a false or erroneous sample without deleting file."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            UPDATE learned_blurs
            SET is_confirmed = 0, updated_at = CURRENT_TIMESTAMP
            WHERE sample_id = ?;
            """, (sample_id,))
            conn.commit()
            logger.info("Invalidated sample %d from active memory.", sample_id)

    def count_confirmed(self) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM learned_blurs WHERE is_confirmed = 1;")
            return cur.fetchone()[0]

