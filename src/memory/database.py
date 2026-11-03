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
    clear_image_path: Optional[str]
    is_confirmed: bool
    confidence: float
    source: str
    verification_status: str
    options_json: Optional[str]
    blur_hash: Optional[str]
    session_id: Optional[str]
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
                clear_image_path TEXT,
                is_confirmed INTEGER NOT NULL DEFAULT 1,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'user_confirmed',
                verification_status TEXT DEFAULT 'VERIFIED',
                options_json TEXT,
                blur_hash TEXT,
                session_id TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Safe migration for existing databases: check existing columns
            cur.execute("PRAGMA table_info(learned_blurs);")
            columns = [row["name"] for row in cur.fetchall()]

            new_cols = [
                ("clear_image_path", "TEXT"),
                ("verification_status", "TEXT DEFAULT 'VERIFIED'"),
                ("options_json", "TEXT"),
                ("blur_hash", "TEXT"),
                ("session_id", "TEXT"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in columns:
                    cur.execute(f"ALTER TABLE learned_blurs ADD COLUMN {col_name} {col_type};")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_card_id ON learned_blurs (card_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_confirmed ON learned_blurs (is_confirmed);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_hash ON learned_blurs (blur_hash);")

            # Dedicated separate table for disagreement cases to preserve debugging records
            # without contaminating active confirmed recognition memory
            cur.execute("""
            CREATE TABLE IF NOT EXISTS disagreement_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                green_card_id INTEGER,
                green_card_name TEXT,
                clear_matched_id INTEGER,
                clear_matched_name TEXT,
                clear_confidence REAL,
                options_json TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

    def insert_sample(
        self,
        card_id: int,
        card_name: str,
        image_path: str,
        clear_image_path: Optional[str] = None,
        is_confirmed: bool = True,
        confidence: float = 1.0,
        source: str = "user_confirmed",
        verification_status: str = "VERIFIED",
        options_json: Optional[str] = None,
        blur_hash: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: str = ""
    ) -> int:
        """Inserts a new learned blur record and returns sample_id."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO learned_blurs (
                card_id, card_name, image_path, clear_image_path,
                is_confirmed, confidence, source, verification_status,
                options_json, blur_hash, session_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_id, card_name, str(image_path),
                str(clear_image_path) if clear_image_path else None,
                1 if is_confirmed else 0, confidence, source,
                verification_status, options_json, blur_hash, session_id, notes
            ))
            conn.commit()
            return cur.lastrowid

    def get_all_confirmed(self) -> List[BlurSampleRecord]:
        """Returns all confirmed active blur samples with VERIFIED status."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT * FROM learned_blurs
            WHERE is_confirmed = 1 AND (verification_status = 'VERIFIED' OR verification_status IS NULL)
            ORDER BY sample_id ASC;
            """)
            rows = cur.fetchall()
            return [
                BlurSampleRecord(
                    sample_id=r["sample_id"],
                    card_id=r["card_id"],
                    card_name=r["card_name"],
                    image_path=r["image_path"],
                    clear_image_path=r["clear_image_path"] if "clear_image_path" in r.keys() else None,
                    is_confirmed=bool(r["is_confirmed"]),
                    confidence=float(r["confidence"]),
                    source=r["source"],
                    verification_status=r["verification_status"] if "verification_status" in r.keys() else "VERIFIED",
                    options_json=r["options_json"] if "options_json" in r.keys() else None,
                    blur_hash=r["blur_hash"] if "blur_hash" in r.keys() else None,
                    session_id=r["session_id"] if "session_id" in r.keys() else None,
                    notes=r["notes"],
                    created_at=str(r["created_at"]),
                    updated_at=str(r["updated_at"])
                )
                for r in rows
            ]

    def has_blur_hash(self, card_id: int, blur_hash: str) -> bool:
        """Checks if an identical blur hash already exists for this card."""
        if not blur_hash:
            return False
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT COUNT(*) FROM learned_blurs
            WHERE card_id = ? AND blur_hash = ? AND is_confirmed = 1;
            """, (card_id, blur_hash))
            return cur.fetchone()[0] > 0

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

    def log_disagreement(
        self,
        session_id: str,
        green_card_id: int,
        green_card_name: str,
        clear_matched_id: int,
        clear_matched_name: str,
        clear_confidence: float = 0.0,
        options_json: Optional[str] = None,
        notes: str = ""
    ) -> int:
        """Stores a separate unverified disagreement record for debugging."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO disagreement_logs (
                session_id, green_card_id, green_card_name,
                clear_matched_id, clear_matched_name, clear_confidence,
                options_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, green_card_id, green_card_name,
                clear_matched_id, clear_matched_name, clear_confidence,
                options_json, notes
            ))
            conn.commit()
            return cur.lastrowid

    def get_all_disagreements(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM disagreement_logs ORDER BY log_id DESC;")
            return [dict(r) for r in cur.fetchall()]

