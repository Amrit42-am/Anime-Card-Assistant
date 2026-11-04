"""
src/memory/self_learning_db.py

Cold-Start Self-Learning SQLite Database for Anime Card Assistant.
Stores learned cards, revealed clear artwork, verified blur samples,
question sessions, and prediction history.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
from typing import List, Optional, Dict, Any, Generator, Tuple
import numpy as np

logger = logging.getLogger("self_learning_db")

DEFAULT_DB_PATH = Path("data/self_learning.db")


@dataclass
class CardRecord:
    card_id: int
    canonical_name: str
    clear_artwork_path: Optional[str]
    image_hash: Optional[str]
    embedding: Optional[np.ndarray]
    first_seen: str
    last_seen: str
    observation_count: int
    clear_sharpness: float
    verification_status: str


@dataclass
class BlurSampleRecord:
    sample_id: int
    card_id: int
    blur_image_path: str
    clear_image_path: Optional[str]
    blur_embedding: Optional[np.ndarray]
    blur_hash: Optional[str]
    capture_conditions: Optional[str]
    source_session_id: Optional[str]
    verification_status: str
    created_at: str


@dataclass
class QuestionSessionRecord:
    session_id: str
    blurred_artwork_path: Optional[str]
    option_tl: Optional[str]
    option_tr: Optional[str]
    option_bl: Optional[str]
    option_br: Optional[str]
    predicted_name: Optional[str]
    predicted_slot: Optional[int]
    predicted_confidence: Optional[float]
    actual_name: Optional[str]
    actual_slot: Optional[int]
    outcome: str  # "CORRECT", "WRONG", "UNVERIFIED"
    timestamp: float
    created_at: str


@dataclass
class PredictionHistoryRecord:
    history_id: int
    session_id: str
    predicted_name: Optional[str]
    actual_name: Optional[str]
    predicted_slot: Optional[int]
    actual_slot: Optional[int]
    confidence: Optional[float]
    is_correct: Optional[int]  # 1, 0, or None
    failure_category: Optional[str]
    timestamp: float
    created_at: str


class SelfLearningDatabase:
    """Manages transactional SQLite storage for the autonomous self-learning pipeline."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        with self._get_connection() as conn:
            cur = conn.cursor()

            # 1. Cards Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                card_id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT UNIQUE NOT NULL COLLATE NOCASE,
                clear_artwork_path TEXT,
                image_hash TEXT,
                embedding BLOB,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                observation_count INTEGER DEFAULT 1,
                clear_sharpness REAL DEFAULT 0.0,
                verification_status TEXT DEFAULT 'VERIFIED'
            );
            """)

            # 2. Question Sessions Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS question_sessions (
                session_id TEXT PRIMARY KEY,
                blurred_artwork_path TEXT,
                option_tl TEXT,
                option_tr TEXT,
                option_bl TEXT,
                option_br TEXT,
                predicted_name TEXT,
                predicted_slot INTEGER,
                predicted_confidence REAL,
                actual_name TEXT,
                actual_slot INTEGER,
                outcome TEXT DEFAULT 'UNVERIFIED',
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 3. Blur Samples Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS blur_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
                blur_image_path TEXT NOT NULL,
                clear_image_path TEXT,
                blur_embedding BLOB,
                blur_hash TEXT,
                capture_conditions TEXT,
                source_session_id TEXT REFERENCES question_sessions(session_id) ON DELETE SET NULL,
                verification_status TEXT DEFAULT 'VERIFIED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. Prediction History Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES question_sessions(session_id) ON DELETE SET NULL,
                predicted_name TEXT,
                actual_name TEXT,
                predicted_slot INTEGER,
                actual_slot INTEGER,
                confidence REAL,
                is_correct INTEGER,
                failure_category TEXT,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Indexes for fast lookup
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_canonical_name ON cards(canonical_name);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_samples_card_id ON blur_samples(card_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_blur_samples_hash ON blur_samples(blur_hash);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON question_sessions(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_history_is_correct ON prediction_history(is_correct);")

            conn.commit()

    @staticmethod
    def _embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
        if embedding is None:
            return None
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        return arr.tobytes()

    @staticmethod
    def _blob_to_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if not blob:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    def get_all_cards(self) -> List[CardRecord]:
        """Returns all known cards ordered by canonical name."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM cards ORDER BY canonical_name ASC;")
            rows = cur.fetchall()
            return [
                CardRecord(
                    card_id=r["card_id"],
                    canonical_name=r["canonical_name"],
                    clear_artwork_path=r["clear_artwork_path"],
                    image_hash=r["image_hash"],
                    embedding=self._blob_to_embedding(r["embedding"]),
                    first_seen=str(r["first_seen"]),
                    last_seen=str(r["last_seen"]),
                    observation_count=int(r["observation_count"]),
                    clear_sharpness=float(r["clear_sharpness"] or 0.0),
                    verification_status=r["verification_status"]
                )
                for r in rows
            ]

    def get_card_by_id(self, card_id: int) -> Optional[CardRecord]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM cards WHERE card_id = ?;", (card_id,))
            r = cur.fetchone()
            if not r:
                return None
            return CardRecord(
                card_id=r["card_id"],
                canonical_name=r["canonical_name"],
                clear_artwork_path=r["clear_artwork_path"],
                image_hash=r["image_hash"],
                embedding=self._blob_to_embedding(r["embedding"]),
                first_seen=str(r["first_seen"]),
                last_seen=str(r["last_seen"]),
                observation_count=int(r["observation_count"]),
                clear_sharpness=float(r["clear_sharpness"] or 0.0),
                verification_status=r["verification_status"]
            )

    def get_card_by_name(self, name: str) -> Optional[CardRecord]:
        clean = name.strip()
        if not clean:
            return None
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM cards WHERE canonical_name = ? COLLATE NOCASE;", (clean,))
            r = cur.fetchone()
            if not r:
                return None
            return CardRecord(
                card_id=r["card_id"],
                canonical_name=r["canonical_name"],
                clear_artwork_path=r["clear_artwork_path"],
                image_hash=r["image_hash"],
                embedding=self._blob_to_embedding(r["embedding"]),
                first_seen=str(r["first_seen"]),
                last_seen=str(r["last_seen"]),
                observation_count=int(r["observation_count"]),
                clear_sharpness=float(r["clear_sharpness"] or 0.0),
                verification_status=r["verification_status"]
            )

    def get_or_create_card(
        self,
        canonical_name: str,
        clear_artwork_path: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        image_hash: Optional[str] = None,
        clear_sharpness: float = 0.0
    ) -> Tuple[CardRecord, bool]:
        """
        Retrieves existing card or creates a new card record.
        If existing card has lower sharpness clear artwork, safely upgrades it.
        Returns (CardRecord, created: bool).
        """
        clean_name = canonical_name.strip()
        existing = self.get_card_by_name(clean_name)
        if existing:
            # Update observation count and last_seen
            with self._get_connection() as conn:
                cur = conn.cursor()
                # Check if we should upgrade clear artwork (e.g. higher sharpness crop)
                should_update_artwork = (
                    clear_artwork_path and
                    (not existing.clear_artwork_path or clear_sharpness > existing.clear_sharpness + 10.0)
                )
                if should_update_artwork:
                    cur.execute("""
                    UPDATE cards
                    SET last_seen = CURRENT_TIMESTAMP,
                        observation_count = observation_count + 1,
                        clear_artwork_path = ?,
                        embedding = ?,
                        image_hash = ?,
                        clear_sharpness = ?
                    WHERE card_id = ?;
                    """, (
                        clear_artwork_path,
                        self._embedding_to_blob(embedding) if embedding is not None else self._embedding_to_blob(existing.embedding),
                        image_hash or existing.image_hash,
                        max(clear_sharpness, existing.clear_sharpness),
                        existing.card_id
                    ))
                else:
                    cur.execute("""
                    UPDATE cards
                    SET last_seen = CURRENT_TIMESTAMP,
                        observation_count = observation_count + 1
                    WHERE card_id = ?;
                    """, (existing.card_id,))
                conn.commit()
            return self.get_card_by_id(existing.card_id), False

        # Create new card record
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO cards (
                canonical_name, clear_artwork_path, image_hash, embedding,
                clear_sharpness, observation_count, verification_status
            ) VALUES (?, ?, ?, ?, ?, 1, 'VERIFIED');
            """, (
                clean_name,
                clear_artwork_path,
                image_hash,
                self._embedding_to_blob(embedding),
                clear_sharpness
            ))
            new_id = cur.lastrowid
            conn.commit()

        logger.info("Discovered and registered NEW card in self-learning database: '%s' (Card ID: %d)", clean_name, new_id)
        return self.get_card_by_id(new_id), True

    def insert_blur_sample(
        self,
        card_id: int,
        blur_image_path: str,
        clear_image_path: Optional[str] = None,
        blur_embedding: Optional[np.ndarray] = None,
        blur_hash: Optional[str] = None,
        capture_conditions: Optional[str] = None,
        source_session_id: Optional[str] = None,
        verification_status: str = "VERIFIED"
    ) -> int:
        """Saves a verified blur sample attached to a known card_id."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Prevent duplicate identical blur hash on same card
            if blur_hash:
                cur.execute(
                    "SELECT sample_id FROM blur_samples WHERE card_id = ? AND blur_hash = ?;",
                    (card_id, blur_hash)
                )
                existing = cur.fetchone()
                if existing:
                    logger.debug("Duplicate blur hash %s for card_id %d skipped.", blur_hash, card_id)
                    return existing["sample_id"]

            cur.execute("""
            INSERT INTO blur_samples (
                card_id, blur_image_path, clear_image_path, blur_embedding,
                blur_hash, capture_conditions, source_session_id, verification_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                card_id,
                str(blur_image_path),
                str(clear_image_path) if clear_image_path else None,
                self._embedding_to_blob(blur_embedding),
                blur_hash,
                capture_conditions,
                source_session_id,
                verification_status
            ))
            sample_id = cur.lastrowid
            conn.commit()
            return sample_id

    def get_blur_samples_for_card(self, card_id: int) -> List[BlurSampleRecord]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM blur_samples WHERE card_id = ? AND verification_status = 'VERIFIED' ORDER BY sample_id ASC;",
                (card_id,)
            )
            rows = cur.fetchall()
            return [
                BlurSampleRecord(
                    sample_id=r["sample_id"],
                    card_id=r["card_id"],
                    blur_image_path=r["blur_image_path"],
                    clear_image_path=r["clear_image_path"],
                    blur_embedding=self._blob_to_embedding(r["blur_embedding"]),
                    blur_hash=r["blur_hash"],
                    capture_conditions=r["capture_conditions"],
                    source_session_id=r["source_session_id"],
                    verification_status=r["verification_status"],
                    created_at=str(r["created_at"])
                )
                for r in rows
            ]

    def get_all_verified_blur_samples(self) -> List[BlurSampleRecord]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM blur_samples WHERE verification_status = 'VERIFIED' ORDER BY sample_id ASC;")
            rows = cur.fetchall()
            return [
                BlurSampleRecord(
                    sample_id=r["sample_id"],
                    card_id=r["card_id"],
                    blur_image_path=r["blur_image_path"],
                    clear_image_path=r["clear_image_path"],
                    blur_embedding=self._blob_to_embedding(r["blur_embedding"]),
                    blur_hash=r["blur_hash"],
                    capture_conditions=r["capture_conditions"],
                    source_session_id=r["source_session_id"],
                    verification_status=r["verification_status"],
                    created_at=str(r["created_at"])
                )
                for r in rows
            ]

    def record_session(
        self,
        session_id: str,
        blurred_artwork_path: Optional[str],
        options: List[str],
        predicted_name: Optional[str],
        predicted_slot: Optional[int],
        predicted_confidence: Optional[float],
        actual_name: Optional[str],
        actual_slot: Optional[int],
        outcome: str,
        timestamp: float
    ):
        """Records or updates a complete quiz session lifecycle record."""
        opt_tl = options[0] if len(options) > 0 else None
        opt_tr = options[1] if len(options) > 1 else None
        opt_bl = options[2] if len(options) > 2 else None
        opt_br = options[3] if len(options) > 3 else None

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO question_sessions (
                session_id, blurred_artwork_path, option_tl, option_tr, option_bl, option_br,
                predicted_name, predicted_slot, predicted_confidence,
                actual_name, actual_slot, outcome, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                session_id,
                str(blurred_artwork_path) if blurred_artwork_path else None,
                opt_tl, opt_tr, opt_bl, opt_br,
                predicted_name, predicted_slot, predicted_confidence,
                actual_name, actual_slot, outcome, timestamp
            ))
            conn.commit()

    def record_prediction(
        self,
        session_id: str,
        predicted_name: Optional[str],
        actual_name: Optional[str],
        predicted_slot: Optional[int],
        actual_slot: Optional[int],
        confidence: Optional[float],
        is_correct: Optional[bool],
        failure_category: Optional[str],
        timestamp: float
    ) -> int:
        """Stores the prediction history for accurate tracking and learning analysis."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO prediction_history (
                session_id, predicted_name, actual_name, predicted_slot, actual_slot,
                confidence, is_correct, failure_category, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                session_id,
                predicted_name,
                actual_name,
                predicted_slot,
                actual_slot,
                confidence,
                1 if is_correct is True else (0 if is_correct is False else None),
                failure_category,
                timestamp
            ))
            row_id = cur.lastrowid
            conn.commit()
            return row_id

    def get_stats(self) -> Dict[str, Any]:
        """Calculates dashboard metrics strictly based on verified data."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM cards;")
            total_discovered_cards = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM cards WHERE verification_status = 'VERIFIED';")
            total_verified_cards = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM blur_samples WHERE verification_status = 'VERIFIED';")
            total_blur_samples = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM prediction_history WHERE is_correct = 1;")
            total_correct = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM prediction_history WHERE is_correct = 0;")
            total_incorrect = cur.fetchone()[0]

            total_verified_predictions = total_correct + total_incorrect
            accuracy = (
                (float(total_correct) / float(total_verified_predictions)) * 100.0
                if total_verified_predictions > 0 else 0.0
            )

            # Recent learning events
            cur.execute("""
            SELECT qs.session_id, qs.actual_name, qs.predicted_name, qs.outcome, qs.predicted_slot, qs.actual_slot, qs.timestamp
            FROM question_sessions qs
            WHERE qs.outcome IN ('CORRECT', 'WRONG')
            ORDER BY qs.timestamp DESC LIMIT 10;
            """)
            recent_rows = cur.fetchall()
            recent_events = [
                {
                    "session_id": r["session_id"],
                    "card_name": r["actual_name"],
                    "previous_prediction": r["predicted_name"] or "None",
                    "outcome": r["outcome"],
                    "predicted_slot": r["predicted_slot"],
                    "actual_slot": r["actual_slot"],
                    "timestamp": r["timestamp"]
                }
                for r in recent_rows
            ]

            # Most frequently encountered cards
            cur.execute("""
            SELECT canonical_name, observation_count, clear_artwork_path
            FROM cards
            ORDER BY observation_count DESC, last_seen DESC LIMIT 10;
            """)
            freq_rows = cur.fetchall()
            most_frequent = [
                {
                    "name": r["canonical_name"],
                    "count": r["observation_count"],
                    "has_artwork": bool(r["clear_artwork_path"])
                }
                for r in freq_rows
            ]

            # Last learning event
            last_event = recent_events[0] if recent_events else None

            return {
                "total_discovered_cards": total_discovered_cards,
                "total_verified_cards": total_verified_cards,
                "total_blur_samples": total_blur_samples,
                "total_correct_predictions": total_correct,
                "total_incorrect_predictions": total_incorrect,
                "total_verified_predictions": total_verified_predictions,
                "accuracy_percent": accuracy,
                "recent_learning_events": recent_events,
                "most_frequent_cards": most_frequent,
                "last_event": last_event
            }
