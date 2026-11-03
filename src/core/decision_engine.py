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
    button_slot: Optional[int] = None
    runner_up_name: Optional[str] = None
    runner_up_score: Optional[float] = None
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

        # 1. OCR Extraction (Prefer 4 distinct and correctly resolved option IDs)
        ocr_candidates: List[CandidateOption] = []
        if button_frames and len(button_frames) == 4:
            ocr_candidates = self.option_reader.read_options_from_buttons(button_frames, use_retry=False)

            # Check distinct resolved cards
            distinct_cids = {c.card_id for c in ocr_candidates if c.is_valid and c.card_id is not None}
            if len(distinct_cids) < 4:
                logger.info("Only %d/4 distinct candidates resolved. Retrying preprocessing and OCR...", len(distinct_cids))
                # Step 1 & 2: Retry preprocessing & OCR on button crops
                retry_candidates = self.option_reader.read_options_from_buttons(button_frames, use_retry=True)
                for i in range(4):
                    if not ocr_candidates[i].is_valid and retry_candidates[i].is_valid:
                        logger.info("OCR Retry successfully recovered slot %d: '%s'", i, retry_candidates[i].canonical_name)
                        ocr_candidates[i] = retry_candidates[i]

        elif options_frame is not None and options_frame.size > 0:
            ocr_candidates = self.option_reader.read_options_from_panel(options_frame)

        # Unique valid candidates preserving order
        seen_ids = set()
        valid_candidates: List[str] = []
        for c in ocr_candidates:
            if c.is_valid and c.card_id is not None and c.card_id not in seen_ids:
                seen_ids.add(c.card_id)
                valid_candidates.append(c.canonical_name)

        logger.info("OCR extracted %d distinct valid candidate cards: %s", len(valid_candidates), valid_candidates)

        # 2. Visual Embedding Extraction
        query_embedding = self.encoder.encode_image(card_frame)

        # 3. Candidate Matching with Controlled Fallback
        if len(valid_candidates) >= 4:
            # Ideal: 4 distinct candidate options
            match_res = self.matcher.match_candidates(
                query_embedding=query_embedding,
                candidate_names=valid_candidates,
                learned_memory=self.memory
            )
        elif len(valid_candidates) == 3:
            # 3 options resolved: match among the 3 candidates with controlled validation
            match_res = self.matcher.match_candidates(
                query_embedding=query_embedding,
                candidate_names=valid_candidates,
                learned_memory=self.memory
            )
            # Slight confidence penalty for 1 missing candidate
            match_res["confidence"] = round(match_res["confidence"] * 0.92, 4)
        else:
            # Insufficient candidates (<3): controlled fallback matching across all cards
            logger.warning("Fewer than 3 candidates from OCR (%d found). Running controlled full database search fallback.", len(valid_candidates))
            match_res = self.matcher.match_all(
                query_embedding=query_embedding,
                top_k=4,
                learned_memory=self.memory
            )
            # Return uncertainty when evidence is insufficient
            if match_res["confidence"] < 0.85 or match_res["margin"] < 0.12:
                match_res["status"] = "uncertain"

        elapsed_ms = (time.time() - t0) * 1000.0
        conf_report = match_res.get("confidence_report")

        # Determine button slot (0: TL, 1: TR, 2: BL, 3: BR)
        assigned_slot = None
        target_name = match_res["card_name"].lower().strip()
        for cand in ocr_candidates:
            if cand.canonical_name and cand.canonical_name.lower().strip() == target_name:
                assigned_slot = cand.slot_index
                break

        # Extract runner-up name and score
        alts = match_res.get("alternatives", [])
        runner_up_name = alts[0]["card_name"] if alts else None
        runner_up_score = alts[0].get("similarity") if alts else None

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
            button_slot=assigned_slot,
            runner_up_name=runner_up_name,
            runner_up_score=runner_up_score,
            captured_image=card_frame.copy(),
            query_embedding=query_embedding.copy()
        )

