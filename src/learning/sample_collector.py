"""
src/learning/sample_collector.py

Phase: Automatic Sample Collector & Question Lifecycle Linker.
Pairs active blurred question samples with their revealed result screens.
Extracts ground-truth identity from the green button, cross-validates against
the unblurred card artwork with an independent gallery match, and persists
clean, confirmed samples into permanent learned blur memory.
"""

from dataclasses import dataclass
import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np

from src.recognition.image_encoder import BaseImageEncoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.learning.result_detector import ResultDetectionEvent

logger = logging.getLogger("sample_collector")


@dataclass
class PendingQuestion:
    session_id: str
    timestamp: float
    blurred_image: np.ndarray
    blurred_embedding: np.ndarray
    blur_hash: str
    option_names: List[str]
    option_ids: List[int]
    predicted_card_name: str
    predicted_card_id: Optional[int]


@dataclass
class CollectionReport:
    session_id: str
    card_id: int
    card_name: str
    verified: bool
    sample_id: Optional[int]
    green_button_name: str
    clear_image_match_name: str
    status: str
    message: str


class AutomaticSampleCollector:
    """Coordinates automatic learning from live Roblox result screens."""

    def __init__(
        self,
        memory: BlurMemorySystem,
        matcher: ImageMatcher,
        encoder: BaseImageEncoder,
        max_pairing_delay: float = 12.0,
        min_pairing_delay: float = 0.4
    ):
        self.memory = memory
        self.matcher = matcher
        self.encoder = encoder
        self.max_pairing_delay = max_pairing_delay
        self.min_pairing_delay = min_pairing_delay

        self.pending_question: Optional[PendingQuestion] = None
        self.last_learned_session: Optional[str] = None

    def register_active_question(
        self,
        card_frame: np.ndarray,
        option_names: List[str],
        option_ids: List[Optional[int]],
        predicted_name: str,
        predicted_id: Optional[int],
        card_hash: str,
        timestamp: float,
        blurred_embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Called when ACTIVE_QUESTION appears.
        Temporarily buffers the blurred image and options awaiting result reveal.
        """
        if blurred_embedding is None:
            blurred_embedding = self.encoder.encode_image(card_frame)

        session_id = f"q_{int(timestamp)}_{card_hash[:8]}"
        valid_ids = [cid for cid in option_ids if cid is not None]

        self.pending_question = PendingQuestion(
            session_id=session_id,
            timestamp=timestamp,
            blurred_image=card_frame.copy(),
            blurred_embedding=blurred_embedding.copy(),
            blur_hash=card_hash,
            option_names=list(option_names),
            option_ids=valid_ids,
            predicted_card_name=predicted_name,
            predicted_card_id=predicted_id
        )

        logger.info("Buffered pending active question %s (%s, %d options)",
                    session_id, predicted_name, len(option_names))
        return session_id

    def process_result(
        self,
        clear_frame: np.ndarray,
        button_frames: List[np.ndarray],
        result_event: ResultDetectionEvent,
        timestamp: float
    ) -> Optional[CollectionReport]:
        """
        Called when RESULT_REVEALED is detected.
        Cross-validates green button answer with clear card match,
        and saves sample to permanent memory if confident.
        """
        if not self.pending_question:
            logger.debug("Result detected but no pending question is buffered.")
            return None

        # 1. Pairing Window Check
        delay = timestamp - self.pending_question.timestamp
        if delay < self.min_pairing_delay or delay > self.max_pairing_delay:
            logger.warning("Discarding pending question %s: result delay %.2fs outside valid window (%.1fs - %.1fs)",
                           self.pending_question.session_id, delay, self.min_pairing_delay, self.max_pairing_delay)
            self.pending_question = None
            return None

        # Avoid processing the exact same session multiple times on repeated result frames
        if self.pending_question.session_id == self.last_learned_session:
            return None

        # 2. Extract and Validate Ground Truth from Green Button
        correct_data = result_event.correct_option
        if not correct_data or not correct_data.verified or correct_data.card_id is None:
            logger.warning("Could not identify verified correct card from green button. Auto-learning skipped.")
            return None

        ground_truth_id = correct_data.card_id
        ground_truth_name = correct_data.correct_option

        # Verify ground truth was among the candidate options
        name_lower = ground_truth_name.lower().strip()
        opt_names_lower = [o.lower().strip() for o in self.pending_question.option_names]
        if ground_truth_id not in self.pending_question.option_ids and name_lower not in opt_names_lower:
            logger.warning("Green button card '%s' (#%d) was not among the active question's 4 options %s. Mismatched round.",
                           ground_truth_name, ground_truth_id, self.pending_question.option_names)
            return None

        # 3. Independent Verification: Match Clear Artwork against Reference Gallery
        clear_emb = self.encoder.encode_image(clear_frame)
        clear_match = self.matcher.match_all(clear_emb, top_k=2)
        clear_matched_id = clear_match["card_id"]
        clear_matched_name = clear_match["card_name"]

        # Check for agreement between Green Button and Clear Artwork
        is_agreed = (clear_matched_id == ground_truth_id) or (clear_matched_name.lower().strip() == name_lower)

        if is_agreed:
            verification_status = "VERIFIED"
            notes = f"Auto-confirmed from green button (slot {correct_data.slot_index}) & clear artwork match"
            msg = f"Successfully verified! Ground truth: {ground_truth_name}"
        else:
            verification_status = "UNVERIFIED_DISAGREEMENT"
            notes = f"Disagreement: green button='{ground_truth_name}' vs clear match='{clear_matched_name}'"
            msg = f"Disagreement recorded: green button '{ground_truth_name}' vs clear artwork '{clear_matched_name}'"
            logger.warning(notes)

        # 4. Save to Persistent Blur Memory
        sample_id = self.memory.add_confirmed_sample(
            card_id=ground_truth_id,
            card_name=ground_truth_name,
            blurred_image=self.pending_question.blurred_image,
            clear_image=clear_frame,
            options=self.pending_question.option_names,
            embedding=self.pending_question.blurred_embedding,
            confidence=correct_data.confidence if is_agreed else 0.5,
            source="auto_green_button" if is_agreed else "auto_disagreement",
            verification_status=verification_status,
            blur_hash=self.pending_question.blur_hash,
            session_id=self.pending_question.session_id,
            notes=notes
        )

        self.last_learned_session = self.pending_question.session_id
        session_id = self.pending_question.session_id
        self.pending_question = None

        return CollectionReport(
            session_id=session_id,
            card_id=ground_truth_id,
            card_name=ground_truth_name,
            verified=is_agreed,
            sample_id=sample_id,
            green_button_name=ground_truth_name,
            clear_image_match_name=clear_matched_name,
            status=verification_status,
            message=msg
        )
