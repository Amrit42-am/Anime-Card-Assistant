"""
src/learning/sample_collector.py

Phase: Automatic Self-Learning Sample Collector & Lifecycle Linker.
Pairs active blurred question samples with their revealed result screens.
Extracts ground-truth identity from the green button, saves verified clear artwork
and blur samples into the SelfLearningDatabase, updates in-memory matcher and memory,
and records prediction outcomes (CORRECT vs WRONG).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import List, Optional, Any
import cv2
import numpy as np

from src.recognition.image_encoder import BaseImageEncoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.memory.self_learning_db import SelfLearningDatabase
from src.learning.result_detector import ResultDetectionEvent, compute_image_sharpness

logger = logging.getLogger("sample_collector")


@dataclass
class PendingQuestion:
    session_id: str
    timestamp: float
    blurred_image: np.ndarray
    blurred_embedding: np.ndarray
    blur_hash: str
    option_names: List[str]
    option_ids: List[Optional[int]]
    predicted_card_name: str
    predicted_card_id: Optional[int]
    predicted_slot: Optional[int] = None
    predicted_confidence: float = 0.0


@dataclass
class CollectionReport:
    session_id: str
    card_id: int
    card_name: str
    verified: bool
    sample_id: Optional[int]
    green_button_name: str
    clear_image_match_name: str
    status: str  # "CORRECT", "WRONG", "EXPIRED_TIMEOUT", "UNVERIFIED"
    message: str


class AutomaticSampleCollector:
    """Coordinates automatic learning from live Roblox result screens."""

    def __init__(
        self,
        memory: BlurMemorySystem,
        matcher: ImageMatcher,
        encoder: BaseImageEncoder,
        db: Optional[SelfLearningDatabase] = None,
        option_reader: Optional[Any] = None,
        max_pairing_delay: float = 12.0,
        min_pairing_delay: float = 0.3
    ):
        self.memory = memory
        self.matcher = matcher
        self.encoder = encoder
        self.db = db or SelfLearningDatabase()
        self.option_reader = option_reader
        self.max_pairing_delay = max_pairing_delay
        self.min_pairing_delay = min_pairing_delay

        self.cards_dir = Path("data/learned_cards")
        self.blurs_dir = Path("data/learned_blurs")
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        self.blurs_dir.mkdir(parents=True, exist_ok=True)

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
        blurred_embedding: Optional[np.ndarray] = None,
        predicted_slot: Optional[int] = None,
        predicted_confidence: float = 0.0
    ) -> str:
        """
        Called when ACTIVE_QUESTION appears.
        Temporarily buffers the blurred image and options awaiting result reveal.
        """
        if blurred_embedding is None:
            blurred_embedding = self.encoder.encode_image(card_frame)

        session_id = f"q_{int(timestamp)}_{card_hash[:8]}"

        self.pending_question = PendingQuestion(
            session_id=session_id,
            timestamp=timestamp,
            blurred_image=card_frame.copy(),
            blurred_embedding=blurred_embedding.copy(),
            blur_hash=card_hash,
            option_names=list(option_names),
            option_ids=list(option_ids),
            predicted_card_name=predicted_name,
            predicted_card_id=predicted_id,
            predicted_slot=predicted_slot,
            predicted_confidence=predicted_confidence
        )

        logger.info(
            "Buffered pending active question %s (Prediction: '%s', Slot: %s, Options: %s)",
            session_id, predicted_name, predicted_slot, option_names
        )
        return session_id

    def update_pending_candidates(
        self,
        option_names: List[str],
        option_ids: List[Optional[int]]
    ) -> bool:
        """Updates candidate options and IDs for the pending question."""
        if not self.pending_question:
            return False
        self.pending_question.option_names = list(option_names)
        self.pending_question.option_ids = list(option_ids)
        return True

    def invalidate_pending_candidates(self) -> bool:
        """Invalidates candidate options for pending question when OCR fails."""
        if not self.pending_question:
            return False
        self.pending_question.option_names = []
        self.pending_question.option_ids = []
        return True

    def process_result(
        self,
        clear_frame: np.ndarray,
        button_frames: List[np.ndarray],
        result_event: ResultDetectionEvent,
        timestamp: float
    ) -> Optional[CollectionReport]:
        """
        Called when RESULT_REVEALED is detected.
        Extracts ground-truth identity from the green button, learns the card and blur sample,
        evaluates prediction outcome, and records history.
        """
        if not self.pending_question:
            logger.debug("Result detected but no pending question is buffered.")
            return None

        # 1. Pairing Window Check
        delay = timestamp - self.pending_question.timestamp
        if delay < self.min_pairing_delay:
            # Result appeared too fast (<0.3s), wait for stable screen
            return None

        if delay > self.max_pairing_delay:
            logger.warning(
                "Discarding pending question %s: result delay %.2fs exceeded max window (%.1fs)",
                self.pending_question.session_id, delay, self.max_pairing_delay
            )
            sid = self.pending_question.session_id
            self.pending_question = None
            return CollectionReport(
                session_id=sid,
                card_id=-1,
                card_name="",
                verified=False,
                sample_id=None,
                green_button_name="",
                clear_image_match_name="",
                status="EXPIRED_TIMEOUT",
                message=f"Result pairing window expired after {delay:.1f}s"
            )

        if self.pending_question.session_id == self.last_learned_session:
            return None

        # 2. Extract Ground Truth from Green Button
        correct_data = result_event.correct_option
        if not correct_data or not correct_data.verified:
            logger.debug("Green button not yet verified on current frame. Retrying...")
            return None

        ground_truth_name = correct_data.correct_option.strip()
        winning_slot = correct_data.slot_index

        session_id = self.pending_question.session_id
        pred_name = self.pending_question.predicted_card_name.strip()
        pred_slot = self.pending_question.predicted_slot
        pred_conf = self.pending_question.predicted_confidence

        # 3. Determine Prediction Outcome (CORRECT vs WRONG)
        name_match = bool(pred_name and pred_name.lower() == ground_truth_name.lower())
        slot_match = bool(pred_slot is not None and pred_slot == winning_slot)
        is_correct = name_match or slot_match
        outcome = "CORRECT" if is_correct else "WRONG"

        # 4. Measure Clear Frame Sharpness and Encode
        sharpness = compute_image_sharpness(clear_frame)
        clear_emb = self.encoder.encode_image(clear_frame) if clear_frame is not None and clear_frame.size > 0 else None

        # 5. Persist Clear Artwork & Register Card in SelfLearningDatabase
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", ground_truth_name.lower()).strip("_")
        clear_img_filename = f"card_{safe_name}_clear.png"
        clear_img_path = self.cards_dir / clear_img_filename

        if clear_frame is not None and clear_frame.size > 0:
            cv2.imwrite(str(clear_img_path), clear_frame)

        card_record, created = self.db.get_or_create_card(
            canonical_name=ground_truth_name,
            clear_artwork_path=str(clear_img_path) if clear_img_path.exists() else None,
            embedding=clear_emb,
            image_hash=None,
            clear_sharpness=sharpness
        )
        ground_truth_id = card_record.card_id

        # Update in-memory matcher and option reader dynamically
        if self.matcher and clear_emb is not None:
            self.matcher.add_card(ground_truth_id, ground_truth_name, clear_emb)
        if self.option_reader and hasattr(self.option_reader, "register_card"):
            self.option_reader.register_card(ground_truth_id, ground_truth_name)

        # 6. Record Question Session first (parent of blur_samples and prediction_history)
        blur_filename = f"blur_{ground_truth_id}_{int(timestamp)}.png"
        blur_img_path = self.blurs_dir / blur_filename
        blurred_img = self.pending_question.blurred_image
        if blurred_img is not None and blurred_img.size > 0:
            cv2.imwrite(str(blur_img_path), blurred_img)

        self.db.record_session(
            session_id=session_id,
            blurred_artwork_path=str(blur_img_path) if blur_img_path.exists() else None,
            options=self.pending_question.option_names,
            predicted_name=pred_name,
            predicted_slot=pred_slot,
            predicted_confidence=pred_conf,
            actual_name=ground_truth_name,
            actual_slot=winning_slot,
            outcome=outcome,
            timestamp=timestamp
        )

        failure_category = "NONE" if is_correct else "WRONG_PREDICTION"
        self.db.record_prediction(
            session_id=session_id,
            predicted_name=pred_name,
            actual_name=ground_truth_name,
            predicted_slot=pred_slot,
            actual_slot=winning_slot,
            confidence=pred_conf,
            is_correct=is_correct,
            failure_category=failure_category,
            timestamp=timestamp
        )

        # 7. Persist Blurred Sample & Register Blur Sample
        blur_emb = self.pending_question.blurred_embedding
        sample_id = self.db.insert_blur_sample(
            card_id=ground_truth_id,
            blur_image_path=str(blur_img_path),
            clear_image_path=str(clear_img_path) if clear_img_path.exists() else None,
            blur_embedding=blur_emb,
            blur_hash=self.pending_question.blur_hash,
            capture_conditions=f"slot_{winning_slot}",
            source_session_id=session_id,
            verification_status="VERIFIED"
        )

        # Update in-memory matcher and memory system with the new blur sample
        if self.matcher and blur_emb is not None:
            self.matcher.add_blur_sample(ground_truth_id, blur_emb)
        if self.memory and hasattr(self.memory, "add_confirmed_sample"):
            self.memory.add_confirmed_sample(
                card_id=ground_truth_id,
                card_name=ground_truth_name,
                blurred_image=blurred_img,
                clear_image=clear_frame,
                options=self.pending_question.option_names,
                embedding=blur_emb,
                confidence=correct_data.confidence,
                source="auto_green_button",
                verification_status="VERIFIED",
                blur_hash=self.pending_question.blur_hash,
                session_id=session_id,
                notes=f"Auto-learned outcome: {outcome}"
            )

        if is_correct:
            msg = f"Prediction CORRECT! Verified ground truth: {ground_truth_name} (Slot: {winning_slot})"
            logger.info("✓ [Session %s] SUCCESS: %s", session_id, msg)
        else:
            msg = f"Learned from ground truth: {ground_truth_name} (AI guessed: {pred_name or 'Unseen'})"
            logger.info("! [Session %s] CORRECTION LEARNED: %s", session_id, msg)

        self.last_learned_session = session_id
        self.pending_question = None

        return CollectionReport(
            session_id=session_id,
            card_id=ground_truth_id,
            card_name=ground_truth_name,
            verified=True,
            sample_id=sample_id,
            green_button_name=ground_truth_name,
            clear_image_match_name=ground_truth_name,
            status=outcome,
            message=msg
        )

    def expire_pending_question(self, reason: str = "new_question_started"):
        """Safely expires or closes any pending question when a new question arrives."""
        if self.pending_question:
            logger.info("Closing previous pending question session %s (%s)", self.pending_question.session_id, reason)
            self.pending_question = None
