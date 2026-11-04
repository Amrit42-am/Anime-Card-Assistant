"""
src/recognition/image_matcher.py

Phase 2.4 & 7: Core Image Matching Engine.
Implements the 4-candidate matching strategy with full 286-card fallback,
cosine similarity computation across clean and synthetic blur embeddings,
and integration with learned blur memory.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from rapidfuzz import process, fuzz

from src.recognition.confidence import ConfidenceCalibrator

logger = logging.getLogger("image_matcher")


@dataclass
class MatchResult:
    card_name: str
    card_id: int
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
    """Matches query card embeddings against precomputed reference and blur embeddings."""

    def __init__(
        self,
        embeddings_path: Path = Path("models/embeddings/card_embeddings.npz"),
        calibrator: Optional[ConfidenceCalibrator] = None,
    ):
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

        logger.info("Loading precomputed card embeddings from %s...", embeddings_path)
        data = np.load(embeddings_path, allow_pickle=True)

        self.card_ids = data["card_ids"]
        self.card_names = [str(name) for name in data["card_names"]]
        self.clean_embeddings = data["clean_embeddings"].astype(np.float32)       # (N, D)
        self.blur_embeddings = data["blur_embeddings"].astype(np.float32)         # (N, 6, D)
        self.composite_embeddings = data["composite_embeddings"].astype(np.float32)  # (N, D)

        self.num_cards, self.num_variants, self.dim = self.blur_embeddings.shape
        self.calibrator = calibrator or ConfidenceCalibrator()

        # Build name lookup indexes
        self.name_to_index = {name.lower().strip(): idx for idx, name in enumerate(self.card_names)}
        self.id_to_index = {int(cid): idx for idx, cid in enumerate(self.card_ids)}

        logger.info("Loaded %d cards (dim: %d, variants: %d) into memory.", self.num_cards, self.dim, self.num_variants)

    def find_card_index(self, name: str, threshold: float = 75.0) -> Optional[int]:
        """Find card index from string name, with fuzzy matching fallback."""
        clean = name.lower().strip()
        if clean in self.name_to_index:
            return self.name_to_index[clean]

        # Fuzzy match against canonical names
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
    ) -> Dict[str, Any]:
        """
        4-Candidate Recognition Strategy:
        1. Resolve candidate names to database card indices.
        2. If learned memory exists, check for confirmed matches among candidates.
        3. Compare query embedding against candidate blur & clean representations.
        4. Rank candidates and return top match with calibrated confidence.
        """
        query_embedding = query_embedding.astype(np.float32).flatten()
        # Ensure unit length
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding /= q_norm

        candidate_indices = []
        resolved_names = []
        for name in candidate_names:
            idx = self.find_card_index(name)
            if idx is not None:
                candidate_indices.append(idx)
                resolved_names.append(self.card_names[idx])
            else:
                logger.debug("Candidate name '%s' could not be resolved in database.", name)

        # Fallback to all cards if candidate resolution failed or yielded < 2 candidates
        if len(candidate_indices) < 2:
            logger.warning("Fewer than 2 candidates resolved (%d found), falling back to full database search.", len(candidate_indices))
            return self.match_all(query_embedding, top_k=4, learned_memory=learned_memory)

        # Check Learned Memory first (Phase 6 integration)
        if learned_memory:
            memory_match = learned_memory.match_sample(
                query_embedding=query_embedding,
                allowed_card_ids=[int(self.card_ids[i]) for i in candidate_indices]
            )
            if memory_match and memory_match.get("confidence", 0) >= 0.82:
                matched_id = memory_match["card_id"]
                idx = self.id_to_index[matched_id]
                report = self.calibrator.evaluate(
                    top_score=memory_match["confidence"],
                    second_score=memory_match.get("second_confidence", 0.5),
                    source="learned_memory"
                )
                return MatchResult(
                    status="matched" if report.is_confident else "uncertain",
                    card_name=self.card_names[idx],
                    card_id=int(self.card_ids[idx]),
                    confidence=report.calibrated_confidence,
                    confidence_report=report,
                    matched_from="learned_memory",
                    raw_score=memory_match["confidence"],
                    margin=report.margin,
                    alternatives=memory_match.get("alternatives", [])
                )

        # Compare against candidate embeddings
        candidate_scores = []
        for idx in candidate_indices:
            # 1. Similarity with clean reference
            clean_sim = float(np.dot(query_embedding, self.clean_embeddings[idx]))

            # 2. Similarity with all synthetic blur variants (take maximum)
            # blur_embeddings[idx] shape: (num_variants, dim)
            variant_sims = np.dot(self.blur_embeddings[idx], query_embedding)
            max_variant_sim = float(np.max(variant_sims))

            # 3. Composite blur similarity
            comp_sim = float(np.dot(query_embedding, self.composite_embeddings[idx]))

            # Overall score: weighted combination favoring closest blur variant
            score = max(max_variant_sim, 0.9 * comp_sim, 0.85 * clean_sim)

            candidate_scores.append({
                "card_id": int(self.card_ids[idx]),
                "card_name": self.card_names[idx],
                "score": score,
                "clean_similarity": clean_sim,
                "blur_similarity": max_variant_sim,
                "composite_similarity": comp_sim
            })

        # Rank descending
        candidate_scores.sort(key=lambda x: x["score"], reverse=True)
        top = candidate_scores[0]
        second = candidate_scores[1] if len(candidate_scores) > 1 else None
        second_score = second["score"] if second else 0.0

        report = self.calibrator.evaluate(
            top_score=top["score"],
            second_score=second_score,
            source="synthetic_blurs"
        )

        return MatchResult(
            status="matched" if report.is_confident else "uncertain",
            card_name=top["card_name"],
            card_id=top["card_id"],
            confidence=report.calibrated_confidence,
            confidence_report=report,
            matched_from="synthetic_blurs",
            raw_score=top["score"],
            margin=report.margin,
            alternatives=candidate_scores[1:]
        )

    def match_all(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        learned_memory: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Fallback search across all 286 cards when candidates are unavailable.
        """
        query_embedding = query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding /= q_norm

        # Check learned memory across all cards
        if learned_memory:
            memory_match = learned_memory.match_sample(query_embedding=query_embedding)
            if memory_match and memory_match.get("confidence", 0) >= 0.88:
                matched_id = memory_match["card_id"]
                idx = self.id_to_index[matched_id]
                report = self.calibrator.evaluate(
                    top_score=memory_match["confidence"],
                    second_score=memory_match.get("second_confidence", 0.5),
                    source="learned_memory"
                )
                return MatchResult(
                    status="matched" if report.is_confident else "uncertain",
                    card_name=self.card_names[idx],
                    card_id=int(self.card_ids[idx]),
                    confidence=report.calibrated_confidence,
                    confidence_report=report,
                    matched_from="learned_memory",
                    raw_score=memory_match["confidence"],
                    margin=report.margin,
                    alternatives=[]
                )

        # Vectorized dot product against all composite embeddings
        comp_scores = np.dot(self.composite_embeddings, query_embedding)  # (N,)

        # Vectorized dot product against all blur variants: shape (N, 6)
        # blur_embeddings: (N, 6, D) dot (D,) -> (N, 6)
        variant_sims = np.tensordot(self.blur_embeddings, query_embedding, axes=([2], [0]))
        max_variant_sims = np.max(variant_sims, axis=1)  # (N,)

        combined_scores = np.maximum(max_variant_sims, comp_scores)
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        ranked = []
        for rank_idx in top_indices:
            ranked.append({
                "card_id": int(self.card_ids[rank_idx]),
                "card_name": self.card_names[rank_idx],
                "score": float(combined_scores[rank_idx])
            })

        top = ranked[0]
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0

        report = self.calibrator.evaluate(
            top_score=top["score"],
            second_score=second_score,
            source="fallback_all_cards"
        )

        return MatchResult(
            status="matched" if report.is_confident else "uncertain",
            card_name=top["card_name"],
            card_id=top["card_id"],
            confidence=report.calibrated_confidence,
            confidence_report=report,
            matched_from="fallback_all_cards",
            raw_score=top["score"],
            margin=report.margin,
            alternatives=ranked[1:]
        )
