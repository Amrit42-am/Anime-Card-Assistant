"""
src/recognition/image_matcher.py

Self-Learning Dynamic Image Matching Engine for Anime Card Assistant.
Supports cold-start with 0 cards, incremental memory growth,
candidate-restricted similarity scoring against clean artwork and verified blur samples,
and integration with SelfLearningDatabase.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from rapidfuzz import process, fuzz

from src.recognition.confidence import ConfidenceCalibrator
from src.memory.self_learning_db import SelfLearningDatabase

logger = logging.getLogger("image_matcher")


@dataclass
class MatchResult:
    card_name: str
    card_id: Optional[int]
    raw_score: float
    confidence: float
    margin: float
    matched_from: str
    alternatives: List[Dict[str, Any]]
    status: str
    confidence_report: Optional[Any] = None

    @property
    def similarity(self) -> float:
        return self.raw_score

    @property
    def source(self) -> str:
        return self.matched_from

    @property
    def all_candidate_matches(self) -> List[Dict[str, Any]]:
        return self.alternatives

    @property
    def top_matches(self) -> List[Dict[str, Any]]:
        return self.alternatives

    def __getitem__(self, item: str) -> Any:
        if item in ("similarity", "score"):
            return self.raw_score
        if item == "source":
            return self.matched_from
        if item in ("all_candidate_matches", "top_matches"):
            return self.alternatives
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) or item in ("similarity", "score", "source", "all_candidate_matches", "top_matches")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "card_name": self.card_name,
            "card_id": self.card_id,
            "confidence": self.confidence,
            "confidence_report": self.confidence_report,
            "matched_from": self.matched_from,
            "raw_score": self.raw_score,
            "margin": self.margin,
            "alternatives": self.alternatives
        }


class ImageMatcher:
    """Dynamic Image Matcher that grows as new cards and blur samples are learned."""

    def __init__(
        self,
        db: Optional[SelfLearningDatabase] = None,
        embeddings_path: Optional[Path] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ):
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.db = db

        self.card_ids: List[int] = []
        self.card_names: List[str] = []
        self.clean_embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.blur_embeddings_map: Dict[int, List[np.ndarray]] = {}

        self.name_to_index: Dict[str, int] = {}
        self.id_to_index: Dict[int, int] = {}

        # 1. Load from legacy embeddings_path if provided and exists
        if embeddings_path and Path(embeddings_path).exists():
            self._load_legacy_embeddings(Path(embeddings_path))
        # 2. Otherwise load from SelfLearningDatabase
        else:
            if self.db is None:
                self.db = SelfLearningDatabase()
            self._load_from_db()

    def _load_from_db(self):
        """Loads all verified cards and blur samples from the self-learning database."""
        cards = self.db.get_all_cards()
        if not cards:
            logger.info("Self-learning ImageMatcher initialized with 0 cards (Cold Start).")
            return

        cids = []
        names = []
        embs = []

        for c in cards:
            cids.append(c.card_id)
            names.append(c.canonical_name)
            if c.embedding is not None and len(c.embedding) == 384:
                embs.append(c.embedding.astype(np.float32))
            else:
                embs.append(np.zeros(384, dtype=np.float32))

        self.card_ids = cids
        self.card_names = names
        self.clean_embeddings = np.vstack(embs).astype(np.float32)

        # Build name & id lookup indexes
        self.name_to_index = {name.lower().strip(): idx for idx, name in enumerate(self.card_names)}
        self.id_to_index = {cid: idx for idx, cid in enumerate(self.card_ids)}

        # Load blur samples
        samples = self.db.get_all_verified_blur_samples()
        for s in samples:
            if s.blur_embedding is not None and len(s.blur_embedding) == 384:
                if s.card_id not in self.blur_embeddings_map:
                    self.blur_embeddings_map[s.card_id] = []
                self.blur_embeddings_map[s.card_id].append(s.blur_embedding.astype(np.float32))

        logger.info(
            "Self-learning ImageMatcher loaded %d cards and %d blur samples from database.",
            len(self.card_ids), len(samples)
        )

    def _load_legacy_embeddings(self, path: Path):
        """Loads from an existing .npz file (e.g. for synthetic tests)."""
        logger.info("Loading reference embeddings from %s...", path)
        data = np.load(path, allow_pickle=True)
        self.card_ids = [int(cid) for cid in data["card_ids"]]
        self.card_names = [str(name) for name in data["card_names"]]
        self.clean_embeddings = data["clean_embeddings"].astype(np.float32)

        if "blur_embeddings" in data:
            blurs = data["blur_embeddings"].astype(np.float32)
            for idx, cid in enumerate(self.card_ids):
                # blurs[idx] shape: (variants, 384)
                self.blur_embeddings_map[cid] = [v for v in blurs[idx]]

        self.name_to_index = {name.lower().strip(): idx for idx, name in enumerate(self.card_names)}
        self.id_to_index = {cid: idx for idx, cid in enumerate(self.card_ids)}

    def add_card(self, card_id: int, canonical_name: str, embedding: Optional[np.ndarray] = None):
        """Dynamically registers or updates a card in memory."""
        clean_name = canonical_name.strip()
        emb = embedding.astype(np.float32) if embedding is not None else np.zeros(384, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        if card_id in self.id_to_index:
            idx = self.id_to_index[card_id]
            self.card_names[idx] = clean_name
            if embedding is not None:
                self.clean_embeddings[idx] = emb
            self.name_to_index[clean_name.lower()] = idx
        else:
            idx = len(self.card_ids)
            self.card_ids.append(card_id)
            self.card_names.append(clean_name)
            if self.clean_embeddings.size == 0:
                self.clean_embeddings = emb.reshape(1, 384)
            else:
                self.clean_embeddings = np.vstack([self.clean_embeddings, emb.reshape(1, 384)])
            self.id_to_index[card_id] = idx
            self.name_to_index[clean_name.lower()] = idx

        logger.debug("ImageMatcher updated card #%d: '%s'", card_id, clean_name)

    def add_blur_sample(self, card_id: int, blur_embedding: np.ndarray):
        """Adds a verified blur sample embedding to a card."""
        emb = blur_embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        if card_id not in self.blur_embeddings_map:
            self.blur_embeddings_map[card_id] = []
        self.blur_embeddings_map[card_id].append(emb)
        logger.debug("ImageMatcher added blur sample to card #%d (Total samples: %d)",
                     card_id, len(self.blur_embeddings_map[card_id]))

    def find_card_index(self, name: str, threshold: float = 75.0) -> Optional[int]:
        """Find card index from string name, with fuzzy matching fallback."""
        if not self.card_names:
            return None
        clean = name.lower().strip()
        if clean in self.name_to_index:
            return self.name_to_index[clean]

        best_match = process.extractOne(
            clean,
            list(self.name_to_index.keys()),
            scorer=fuzz.token_sort_ratio
        )
        if best_match and best_match[1] >= threshold:
            return self.name_to_index[best_match[0]]

        return None

    def match_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_names: List[str],
        learned_memory: Optional[Any] = None
    ) -> MatchResult:
        """
        Matches query embedding against candidate cards:
        - If database is empty or cards are unseen: returns cold-start provisional result.
        - If cards are known: computes cosine similarities against clear and blur samples.
        """
        query_embedding = query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding /= q_norm

        # Cold start handling: zero known cards in database
        if len(self.card_ids) == 0:
            first_cand = candidate_names[0] if candidate_names else "Unknown"
            return MatchResult(
                status="unseen",
                card_name=first_cand,
                card_id=None,
                raw_score=0.0,
                confidence=0.30,
                margin=0.0,
                matched_from="cold_start_unseen",
                alternatives=[],
                confidence_report=None
            )

        # Check candidate resolution
        candidate_scores = []
        for name in candidate_names:
            clean_name = name.strip()
            idx = self.find_card_index(clean_name)
            if idx is not None:
                cid = self.card_ids[idx]
                cname = self.card_names[idx]
                # 1. Similarity with clear reference artwork
                clean_vec = self.clean_embeddings[idx]
                clean_sim = float(np.dot(query_embedding, clean_vec)) if np.linalg.norm(clean_vec) > 0 else 0.0

                # 2. Similarity with all collected blur samples for this card
                blur_sims = []
                for b_emb in self.blur_embeddings_map.get(cid, []):
                    blur_sims.append(float(np.dot(query_embedding, b_emb)))
                max_blur_sim = float(max(blur_sims)) if blur_sims else 0.0

                # Composite score: maximum of blur samples and clean artwork
                best_sim = max(max_blur_sim, clean_sim)

                candidate_scores.append({
                    "card_id": cid,
                    "card_name": cname,
                    "score": best_sim,
                    "clean_similarity": clean_sim,
                    "blur_similarity": max_blur_sim,
                    "is_known": True
                })
            else:
                # Unseen candidate card
                candidate_scores.append({
                    "card_id": None,
                    "card_name": clean_name,
                    "score": 0.0,
                    "clean_similarity": 0.0,
                    "blur_similarity": 0.0,
                    "is_known": False
                })

        # Check if any candidate was known
        known_candidates = [c for c in candidate_scores if c["is_known"]]
        if not known_candidates:
            first_cand = candidate_names[0] if candidate_names else "Unknown"
            return MatchResult(
                status="unseen",
                card_name=first_cand,
                card_id=None,
                raw_score=0.0,
                confidence=0.30,
                margin=0.0,
                matched_from="cold_start_unseen",
                alternatives=candidate_scores,
                confidence_report=None
            )

        # Rank known candidates descending
        candidate_scores.sort(key=lambda x: x["score"], reverse=True)
        top = candidate_scores[0]
        second = candidate_scores[1] if len(candidate_scores) > 1 else None
        second_score = second["score"] if second else 0.0

        report = self.calibrator.evaluate(
            top_score=top["score"],
            second_score=second_score,
            source="learned_memory" if self.blur_embeddings_map.get(top["card_id"]) else "reference_artwork"
        )

        return MatchResult(
            status="matched" if (report.is_confident and top["is_known"] and top["score"] >= 0.50) else "uncertain",
            card_name=top["card_name"],
            card_id=top["card_id"],
            confidence=report.calibrated_confidence,
            confidence_report=report,
            matched_from="learned_blur_memory" if self.blur_embeddings_map.get(top["card_id"]) else "clear_artwork",
            raw_score=top["score"],
            margin=report.margin,
            alternatives=candidate_scores[1:]
        )

    def match_all(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        learned_memory: Optional[Any] = None
    ) -> MatchResult:
        """Fallback search across all learned cards in the database."""
        query_embedding = query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding /= q_norm

        if len(self.card_ids) == 0:
            return MatchResult(
                status="unseen",
                card_name="Unknown",
                card_id=None,
                raw_score=0.0,
                confidence=0.25,
                margin=0.0,
                matched_from="cold_start_unseen",
                alternatives=[],
                confidence_report=None
            )

        scores = []
        for idx, cid in enumerate(self.card_ids):
            clean_sim = float(np.dot(query_embedding, self.clean_embeddings[idx])) if np.linalg.norm(self.clean_embeddings[idx]) > 0 else 0.0
            blur_sims = [float(np.dot(query_embedding, b)) for b in self.blur_embeddings_map.get(cid, [])]
            max_blur = max(blur_sims) if blur_sims else 0.0
            best_sim = max(clean_sim, max_blur)
            scores.append({
                "card_id": cid,
                "card_name": self.card_names[idx],
                "score": best_sim
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[0]
        second_score = scores[1]["score"] if len(scores) > 1 else 0.0

        report = self.calibrator.evaluate(
            top_score=top["score"],
            second_score=second_score,
            source="fallback_all_cards"
        )

        return MatchResult(
            status="matched" if (report.is_confident and top["score"] >= 0.50) else "uncertain",
            card_name=top["card_name"],
            card_id=top["card_id"],
            confidence=report.calibrated_confidence,
            confidence_report=report,
            matched_from="fallback_all_cards",
            raw_score=top["score"],
            margin=report.margin,
            alternatives=scores[1:top_k]
        )
