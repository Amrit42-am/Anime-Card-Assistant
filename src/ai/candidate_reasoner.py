"""
src/ai/candidate_reasoner.py

Candidate Reasoning Engine for Anime Card Assistant.
Constrains visual matching to the 4 recognized OCR candidates, resolves ambiguous
typos via strict fuzzy matching, compares embeddings against synthetic blur profiles
and verified blur memory, and computes decision margins.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.ocr.option_reader import CandidateOption, OptionReader
from src.recognition.image_matcher import ImageMatcher
from src.recognition.image_encoder import ImageEncoder
from src.ai.confidence_engine import ConfidenceEngine, ConfidenceVerdict

logger = logging.getLogger("candidate_reasoner")


@dataclass
class ReasonedAnswer:
    card_name: str
    card_id: Optional[int]
    slot_index: int
    confidence: float
    margin: float
    is_confident: bool
    runner_up_name: Optional[str]
    runner_up_score: float
    source: str
    status: str  # "matched", "uncertain"
    raw_match: Dict[str, Any]


class CandidateReasoner:
    """
    Ranks and validates answer candidates by fusing OCR extractions
    with candidate-constrained DINOv2 visual similarity matching.
    """

    def __init__(
        self,
        image_matcher: ImageMatcher,
        image_encoder: ImageEncoder,
        option_reader: OptionReader,
        confidence_engine: Optional[ConfidenceEngine] = None
    ):
        self.matcher = image_matcher
        self.encoder = image_encoder
        self.option_reader = option_reader
        self.confidence_engine = confidence_engine or ConfidenceEngine()

    def reason_answer(
        self,
        card_frame: np.ndarray,
        ocr_candidates: List[CandidateOption],
        learned_memory: Optional[Any] = None
    ) -> ReasonedAnswer:
        """
        Processes query card frame and OCR candidate options.
        Produces a reasoned answer with confidence calibration and slot assignment.
        """
        # 1. Compute query image embedding
        query_embedding = self.encoder.encode_image(card_frame)

        # 2. Extract valid candidate names
        valid_candidates = [c for c in ocr_candidates if c.is_valid and c.canonical_name]
        candidate_names = [c.canonical_name for c in valid_candidates]

        is_constrained = len(valid_candidates) >= 2

        # 3. Match candidates via ImageMatcher
        if is_constrained:
            match_res = self.matcher.match_candidates(
                query_embedding=query_embedding,
                candidate_names=candidate_names,
                learned_memory=learned_memory
            )
        else:
            match_res = self.matcher.match_all(
                query_embedding=query_embedding,
                top_k=4,
                learned_memory=learned_memory
            )

        top_name = match_res.get("card_name", "Unknown")
        top_id = match_res.get("card_id")
        top_sim = float(match_res.get("similarity", 0.0))
        match_source = match_res.get("source", "visual")

        # Determine runner-up score and name
        runner_up_name = None
        runner_up_score = 0.0

        all_matches = match_res.get("all_candidate_matches", []) or match_res.get("top_matches", [])
        if len(all_matches) > 1:
            runner_up_name = all_matches[1].get("card_name")
            runner_up_score = float(all_matches[1].get("similarity", 0.0))

        # 4. Calibrate confidence through ConfidenceEngine
        verdict = self.confidence_engine.evaluate(
            top_score=top_sim,
            runner_up_score=runner_up_score,
            candidate_constrained=is_constrained,
            source=match_source
        )

        # 5. Resolve button slot index (0: TL, 1: TR, 2: BL, 3: BR)
        assigned_slot = 0
        found_slot = False
        norm_top = top_name.lower().strip()

        for cand in ocr_candidates:
            if cand.canonical_name and cand.canonical_name.lower().strip() == norm_top:
                assigned_slot = cand.slot_index
                found_slot = True
                break

        if not found_slot and ocr_candidates:
            # Fuzzy match to candidate raw texts if canonical didn't match slot directly
            best_fuzzy = 0.0
            for cand in ocr_candidates:
                from rapidfuzz import fuzz
                score = fuzz.token_sort_ratio(cand.canonical_name or cand.raw_text, top_name)
                if score > best_fuzzy:
                    best_fuzzy = score
                    assigned_slot = cand.slot_index

        status = "matched" if verdict.is_confident else "uncertain"

        return ReasonedAnswer(
            card_name=top_name,
            card_id=top_id,
            slot_index=assigned_slot,
            confidence=verdict.calibrated_confidence,
            margin=verdict.margin_of_victory,
            is_confident=verdict.is_confident,
            runner_up_name=runner_up_name,
            runner_up_score=runner_up_score,
            source=match_source,
            status=status,
            raw_match=match_res
        )
