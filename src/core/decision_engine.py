"""
src/core/decision_engine.py

Phase 7: Central Answer Decision Engine.
Coordinates OCR candidate extraction, visual embedding matching,
learned blur memory prioritization, margin scoring, and confidence calibration.
"""

from dataclasses import dataclass, asdict
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

from src.ocr.option_reader import OptionReader, CandidateOption
from src.recognition.image_encoder import BaseImageEncoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem

logger = logging.getLogger("decision_engine")


@dataclass
class AnswerResult:
    status: str  # "matched", "uncertain", "error"
    card_name: str
    card_id: Optional[int]
    confidence: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    matched_from: str      # "learned_memory", "synthetic_blurs", "fallback_all_cards"
    raw_score: float
    margin: float
    candidates: List[str]
    alternatives: List[Dict[str, Any]]
    explanation: str
    processing_time_ms: float
    captured_image: Optional[np.ndarray] = None
    query_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("captured_image", None)
        d.pop("query_embedding", None)
        return d


class AnswerDecisionEngine:
    """Combines OCR evidence, visual embeddings, and learned blur memory to resolve quiz questions."""

    def __init__(
        self,
        matcher: ImageMatcher,
        encoder: BaseImageEncoder,
        option_reader: OptionReader,
        memory: Optional[BlurMemorySystem] = None
    ):
        self.matcher = matcher
        self.encoder = encoder
        self.option_reader = option_reader
        self.memory = memory

    def process_question(
        self,
        card_frame: np.ndarray,
        button_frames: Optional[List[np.ndarray]] = None,
        options_frame: Optional[np.ndarray] = None
    ) -> AnswerResult:
        """
        Main decision workflow:
        1. Read 4 options from button or options frame.
        2. Encode the blurred card frame into a unit embedding.
        3. Match against candidates and learned memory.
        4. Validate margin and score confidence.
        """
        t0 = time.time()

        if card_frame is None or card_frame.size == 0:
            return AnswerResult(
                status="error",
                card_name="No Card Frame",
                card_id=None,
                confidence=0.0,
                confidence_level="LOW",
                matched_from="none",
                raw_score=0.0,
                margin=0.0,
                candidates=[],
                alternatives=[],
                explanation="Captured card frame was empty or missing.",
                processing_time_ms=0.0
            )

        # 1. OCR Extraction
        ocr_candidates: List[CandidateOption] = []
        if button_frames and len(button_frames) == 4:
            ocr_candidates = self.option_reader.read_options_from_buttons(button_frames)
        elif options_frame is not None and options_frame.size > 0:
            ocr_candidates = self.option_reader.read_options_from_panel(options_frame)

        valid_candidate_names = [c.canonical_name for c in ocr_candidates if c.is_valid]
        logger.info("OCR extracted %d valid candidate names: %s", len(valid_candidate_names), valid_candidate_names)

        # 2. Visual Embedding Extraction
        query_embedding = self.encoder.encode_image(card_frame)

        # 3. Candidate Matching
        if len(valid_candidate_names) >= 2:
            match_res = self.matcher.match_candidates(
                query_embedding=query_embedding,
                candidate_names=valid_candidate_names,
                learned_memory=self.memory
            )
        else:
            logger.warning("Fewer than 2 candidates from OCR, executing full database search fallback.")
            match_res = self.matcher.match_all(
                query_embedding=query_embedding,
                top_k=4,
                learned_memory=self.memory
            )

        elapsed_ms = (time.time() - t0) * 1000.0
        conf_report = match_res.get("confidence_report")

        return AnswerResult(
            status=match_res.get("status", "uncertain"),
            card_name=match_res["card_name"],
            card_id=match_res["card_id"],
            confidence=match_res["confidence"],
            confidence_level=conf_report.confidence_level if conf_report else "MEDIUM",
            matched_from=match_res["matched_from"],
            raw_score=match_res["raw_score"],
            margin=match_res["margin"],
            candidates=[c.canonical_name for c in ocr_candidates],
            alternatives=match_res.get("alternatives", []),
            explanation=conf_report.explanation if conf_report else "",
            processing_time_ms=round(elapsed_ms, 2),
            captured_image=card_frame.copy(),
            query_embedding=query_embedding.copy()
        )

