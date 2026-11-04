"""
src/ai/candidate_reasoner.py

Candidate Reasoning Engine for Anime Card Assistant.
Constrains visual matching strictly to 4 distinct validated OCR candidates,
handles multi-line text and typos via strict fuzzy matching, compares embeddings
against synthetic blur profiles and verified blur memory, and resolves 2x2 grid slots.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from rapidfuzz import fuzz

from src.ocr.option_reader import CandidateOption, OptionReader, normalize_detected_options
from src.recognition.image_matcher import ImageMatcher
from src.recognition.image_encoder import BaseImageEncoder
from src.ai.confidence_engine import ConfidenceEngine

logger = logging.getLogger("candidate_reasoner")


@dataclass
class ReasonedAnswer:
    card_name: str
    card_id: Optional[int]
    slot_index: Optional[int]  # 0: TL, 1: TR, 2: BL, 3: BR, or None if unmapped
    confidence: float
    margin: float
    is_confident: bool
    runner_up_name: Optional[str]
    runner_up_score: float
    source: str
    status: str  # "matched", "uncertain"
    raw_match: Dict[str, Any]
    confidence_level: str = "MEDIUM"  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    query_embedding: Optional[np.ndarray] = None
    captured_image: Optional[np.ndarray] = None
    detected_options: List[Dict[str, Any]] = field(default_factory=list)


class CandidateReasoner:
    """
    Ranks and validates answer candidates by fusing OCR extractions
    with candidate-constrained DINOv2 visual similarity matching.
    """

    def __init__(
        self,
        image_matcher: ImageMatcher,
        image_encoder: BaseImageEncoder,
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
        learned_memory: Optional[Any] = None,
        button_frames: Optional[List[np.ndarray]] = None,
        temporal_agreement: float = 1.0
    ) -> ReasonedAnswer:
        """
        Processes query card frame and OCR candidate options.
        Enforces candidate restriction rule: candidate-constrained matching is ONLY
        performed when all 4 distinct answer options are successfully resolved.
        """
        # 1. Compute query image embedding
        query_embedding = self.encoder.encode_image(card_frame)

        # 2. Validate distinct candidate options
        valid_candidates = [
            c for c in ocr_candidates
            if c.is_valid and c.canonical_name and c.card_id is not None
        ]
        unique_ids = set(c.card_id for c in valid_candidates)
        has_four_distinct = (len(valid_candidates) == 4 and len(unique_ids) == 4)

        # Standalone invocation fallback: only read buttons if candidates were not provided
        if not ocr_candidates and button_frames and len(button_frames) == 4:
            ocr_candidates = self.option_reader.read_options_from_buttons(button_frames, use_retry=False)
            valid_candidates = [
                c for c in ocr_candidates
                if c.is_valid and c.canonical_name and c.card_id is not None
            ]
            unique_ids = set(c.card_id for c in valid_candidates)
            has_four_distinct = (len(valid_candidates) == 4 and len(unique_ids) == 4)

        # Candidate restriction rule: strictly require 4 distinct resolved candidates
        # to restrict visual matching. Otherwise, fall back to controlled full database matching.
        is_constrained = has_four_distinct

        if is_constrained:
            candidate_names = [c.canonical_name for c in valid_candidates]
            match_res = self.matcher.match_candidates(
                query_embedding=query_embedding,
                candidate_names=candidate_names,
                learned_memory=learned_memory
            )
        else:
            logger.info("Fewer than 4 distinct candidates resolved (%d/4 distinct). Running full-database fallback.",
                        len(unique_ids))
            match_res = self.matcher.match_all(
                query_embedding=query_embedding,
                top_k=4,
                learned_memory=learned_memory
            )

        top_name = match_res.get("card_name", "Unknown")
        top_id = match_res.get("card_id")
        top_sim = float(match_res.get("raw_score", match_res.get("similarity", 0.0)))
        match_source = match_res.get("matched_from", match_res.get("source", "visual"))

        # Extract runner-up name and score
        alts = match_res.get("alternatives", match_res.get("all_candidate_matches", []))
        runner_up_name = None
        runner_up_score = 0.0
        if alts:
            first_alt = alts[0]
            runner_up_name = first_alt.get("card_name")
            runner_up_score = float(first_alt.get("score", first_alt.get("similarity", 0.0)))

        # Average OCR quality of candidate choices
        if valid_candidates:
            ocr_quality = float(np.mean([c.ocr_confidence for c in valid_candidates]))
        else:
            ocr_quality = 0.0

        # Calibrate confidence through ConfidenceEngine
        verdict = self.confidence_engine.evaluate(
            top_score=top_sim,
            runner_up_score=runner_up_score,
            candidate_constrained=is_constrained,
            source=match_source,
            ocr_quality=ocr_quality,
            candidate_count=len(unique_ids),
            temporal_agreement=temporal_agreement
        )

        # Resolve 2x2 button slot index (0: TL, 1: TR, 2: BL, 3: BR)
        assigned_slot: Optional[int] = None
        norm_top = top_name.lower().strip()

        # Exact canonical matching to slot
        for cand in ocr_candidates:
            if cand.canonical_name and cand.canonical_name.lower().strip() == norm_top:
                assigned_slot = cand.slot_index
                break

        # If not found directly, try strict fuzzy match with threshold >= 80.0
        if assigned_slot is None and ocr_candidates:
            best_fuzzy = 0.0
            best_slot = None
            for cand in ocr_candidates:
                target_text = cand.canonical_name or cand.raw_text
                score = fuzz.token_sort_ratio(target_text, top_name)
                if score > best_fuzzy:
                    best_fuzzy = score
                    best_slot = cand.slot_index

            if best_fuzzy >= 80.0:
                assigned_slot = best_slot

        # Slot assignment & candidate restriction verification:
        # If fewer than 4 distinct candidates resolved:
        # - Never assign a final answer button slot (assigned_slot must be None)
        # - Never mark as confident or final matched
        # - Provide a provisional visual guess clearly marked as unconstrained
        if not is_constrained:
            assigned_slot = None
            is_confident = False
            status = "uncertain"
            slot_labels = ["TL", "TR", "BL", "BR"]
            failed_slots = [
                slot_labels[i] for i in range(4)
                if not any(c.slot_index == i and c.is_valid and c.card_id is not None for c in ocr_candidates)
            ]
            failed_str = f" (unresolved: {', '.join(failed_slots)})" if failed_slots else ""
            confidence_level = "WAITING FOR VALID OPTIONS" if not valid_candidates else "OCR RECOVERY"
            explanation = f"Provisional visual match: {top_name} ({len(unique_ids)}/4 distinct options resolved{failed_str})"
            display_card_name = top_name
            calibrated_conf = min(verdict.calibrated_confidence, 0.45) if verdict.calibrated_confidence > 0 else 0.35
        elif assigned_slot is None:
            is_confident = False
            status = "uncertain"
            confidence_level = "UNCERTAIN"
            explanation = f"Matched card '{top_name}' could not be assigned to a button slot"
            display_card_name = top_name
            calibrated_conf = verdict.calibrated_confidence
        else:
            is_confident = verdict.is_confident
            status = "matched" if is_confident else "uncertain"
            confidence_level = verdict.confidence_level if is_confident else "UNCERTAIN"
            explanation = verdict.explanation
            display_card_name = top_name
            calibrated_conf = verdict.calibrated_confidence

        logger.debug("Reasoned answer: %s (slot %s, conf %.2f): %s", display_card_name, assigned_slot, calibrated_conf, explanation)

        return ReasonedAnswer(
            card_name=display_card_name,
            card_id=top_id,
            slot_index=assigned_slot,
            confidence=calibrated_conf,
            confidence_level=confidence_level,
            margin=verdict.margin_of_victory if is_constrained else 0.0,
            is_confident=is_confident,
            runner_up_name=runner_up_name if is_constrained else None,
            runner_up_score=runner_up_score if is_constrained else 0.0,
            source=match_source if is_constrained else "waiting_options",
            status=status,
            raw_match=match_res.to_dict() if hasattr(match_res, "to_dict") else dict(match_res),
            query_embedding=query_embedding.copy(),
            captured_image=card_frame.copy(),
            detected_options=normalize_detected_options(ocr_candidates)
        )
