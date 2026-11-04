"""
src/core/decision_engine.py

Phase 7: Decision Engine & Answer Reasoner.
Fuses candidate OCR evidence, visual embeddings from DINOv2, and verified
learned blur memory to identify cards within the active quiz window.
"""

from dataclasses import dataclass, asdict, field
import logging
import time
from typing import List, Dict, Any, Optional

import numpy as np

from src.ocr.option_reader import OptionReader, CandidateOption, normalize_detected_options
from src.recognition.image_encoder import BaseImageEncoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.ai.candidate_reasoner import CandidateReasoner, ReasonedAnswer
from src.ai.confidence_engine import ConfidenceEngine

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
    candidates: List[str]  # 4-slot list: [TL, TR, BL, BR]
    alternatives: List[Dict[str, Any]]
    explanation: str
    processing_time_ms: float
    button_slot: Optional[int] = None
    runner_up_name: Optional[str] = None
    runner_up_score: Optional[float] = None
    captured_image: Optional[np.ndarray] = None
    query_embedding: Optional[np.ndarray] = None
    prediction_stage: str = "CONFIDENT"  # "PROVISIONAL", "STABILIZING", "CONFIDENT", "UNCERTAIN"
    temporal_agreement: float = 1.0
    observation_count: int = 1
    detected_options: List[Dict[str, Any]] = field(default_factory=list)
    visual_similarity: float = 0.0
    is_constrained: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("captured_image", None)
        d.pop("query_embedding", None)
        return d


class AnswerDecisionEngine:
    """Combines OCR evidence, visual embeddings, and candidate reasoning to resolve quiz questions."""

    def __init__(
        self,
        matcher: ImageMatcher,
        encoder: BaseImageEncoder,
        option_reader: OptionReader,
        memory: Optional[BlurMemorySystem] = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        candidate_reasoner: Optional[CandidateReasoner] = None
    ):
        self.matcher = matcher
        self.encoder = encoder
        self.option_reader = option_reader
        self.memory = memory
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.reasoner = candidate_reasoner or CandidateReasoner(
            image_matcher=self.matcher,
            image_encoder=self.encoder,
            option_reader=self.option_reader,
            confidence_engine=self.confidence_engine
        )

    def process_question(
        self,
        card_frame: np.ndarray,
        button_frames: Optional[List[np.ndarray]] = None,
        options_frame: Optional[np.ndarray] = None,
        temporal_agreement: float = 1.0,
        ocr_candidates: Optional[List[CandidateOption]] = None,
        prediction_stage: str = "CONFIDENT",
        observation_count: int = 1
    ) -> AnswerResult:
        """
        Main decision workflow:
        1. Read options from button frames or options panel (if not precomputed).
        2. Execute CandidateReasoner for candidate-constrained matching and slot assignment.
        3. Convert to AnswerResult.
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
                candidates=["", "", "", ""],
                alternatives=[],
                explanation="Captured card frame was empty or missing.",
                processing_time_ms=0.0,
                prediction_stage="UNCERTAIN",
                temporal_agreement=temporal_agreement,
                observation_count=observation_count,
                detected_options=normalize_detected_options([]),
                visual_similarity=0.0,
                is_constrained=False
            )

        # 1. OCR Extraction (reuse precomputed if provided, otherwise extract)
        resolved_candidates: List[CandidateOption] = []
        if ocr_candidates is not None:
            resolved_candidates = ocr_candidates
        elif button_frames and len(button_frames) == 4:
            resolved_candidates = self.option_reader.read_options_from_buttons(button_frames, use_retry=False)
        elif options_frame is not None and options_frame.size > 0:
            resolved_candidates = self.option_reader.read_options_from_panel(options_frame)

        # 2. Candidate Reasoner matching
        reasoned: ReasonedAnswer = self.reasoner.reason_answer(
            card_frame=card_frame,
            ocr_candidates=resolved_candidates,
            learned_memory=self.memory,
            button_frames=button_frames,
            temporal_agreement=temporal_agreement
        )

        elapsed_ms = (time.time() - t0) * 1000.0
        normalized_opts = normalize_detected_options(resolved_candidates)
        canonical_candidates = [opt["canonical_name"] if opt["is_valid"] else "" for opt in normalized_opts]
        valid_count = sum(1 for opt in normalized_opts if opt["is_valid"] and opt["card_id"] is not None)
        unique_ids = set(opt["card_id"] for opt in normalized_opts if opt["card_id"] is not None)
        is_constrained = (valid_count == 4 and len(unique_ids) == 4 and reasoned.slot_index is not None)

        return AnswerResult(
            status=reasoned.status,
            card_name=reasoned.card_name,
            card_id=reasoned.card_id,
            confidence=reasoned.confidence,
            confidence_level=reasoned.confidence_level,
            matched_from=reasoned.source,
            raw_score=reasoned.raw_match.get("raw_score", reasoned.confidence),
            margin=reasoned.margin,
            candidates=canonical_candidates,
            alternatives=reasoned.raw_match.get("alternatives", []),
            explanation=reasoned.raw_match.get("explanation", ""),
            processing_time_ms=round(elapsed_ms, 2),
            button_slot=reasoned.slot_index if is_constrained else None,
            runner_up_name=reasoned.runner_up_name,
            runner_up_score=reasoned.runner_up_score,
            captured_image=card_frame.copy(),
            query_embedding=reasoned.query_embedding,
            prediction_stage=prediction_stage,
            temporal_agreement=round(temporal_agreement, 3),
            observation_count=observation_count,
            detected_options=normalized_opts,
            visual_similarity=float(reasoned.raw_match.get("raw_score", reasoned.confidence)),
            is_constrained=is_constrained
        )
