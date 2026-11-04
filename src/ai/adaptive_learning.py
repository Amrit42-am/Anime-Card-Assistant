"""
src/ai/adaptive_learning.py

Adaptive Result-Based Learning and Ground-Truth Dual Verification.
Orchestrates automatic sample collection from revealed Roblox result screens,
validating green-button OCR against clear card artwork similarity, logging
disagreements separately, and updating verified local blur memory.
"""

from dataclasses import dataclass
import logging
import time
from typing import Optional, List, Dict, Any
import numpy as np

from src.recognition.image_encoder import ImageEncoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.learning.result_detector import ResultDetectionEvent
from src.learning.sample_collector import AutomaticSampleCollector, CollectionReport

logger = logging.getLogger("adaptive_learning")


@dataclass
class LearningOutcome:
    session_id: str
    card_id: Optional[int]
    card_name: str
    verified: bool
    status: str  # "LEARNED", "DISAGREEMENT", "RETRYING", "TIMEOUT", "IGNORED"
    sample_id: Optional[int]
    message: str
    ground_truth_source: str


class AdaptiveLearning:
    """
    Manages the lifecycle of result verification and safe memory accumulation.
    """

    def __init__(
        self,
        memory: BlurMemorySystem,
        matcher: ImageMatcher,
        encoder: ImageEncoder,
        sample_collector: Optional[AutomaticSampleCollector] = None
    ):
        self.memory = memory
        self.matcher = matcher
        self.encoder = encoder
        self.collector = sample_collector or AutomaticSampleCollector(
            memory=memory,
            matcher=matcher,
            encoder=encoder
        )

    def register_question(
        self,
        card_frame: np.ndarray,
        option_names: List[str],
        option_ids: List[Optional[int]],
        predicted_name: str,
        predicted_id: Optional[int],
        card_hash: str,
        timestamp: float
    ) -> str:
        """Registers active question blurred frame for subsequent result verification."""
        return self.collector.register_active_question(
            card_frame=card_frame,
            option_names=option_names,
            option_ids=option_ids,
            predicted_name=predicted_name,
            predicted_id=predicted_id,
            card_hash=card_hash,
            timestamp=timestamp
        )

    def expire_pending_question(self, reason: str = "new_question_started"):
        """Safely expires any pending active question."""
        self.collector.expire_pending_question(reason)

    def update_pending_candidates(
        self,
        option_names: List[str],
        option_ids: List[Optional[int]]
    ) -> bool:
        """Safely updates candidate options and IDs for pending question without touching original blurred image."""
        return self.collector.update_pending_candidates(option_names, option_ids)

    def invalidate_pending_candidates(self) -> bool:
        """Invalidates candidate options for pending question when OCR fails or is invalid."""
        return self.collector.invalidate_pending_candidates()

    @property
    def pending_question(self):
        """Returns the pending active question from collector."""
        return self.collector.pending_question

    def process_result(
        self,
        clear_frame: np.ndarray,
        button_frames: List[np.ndarray],
        result_event: ResultDetectionEvent,
        timestamp: float
    ) -> Optional[CollectionReport]:
        """
        Coordinates dual-verification result processing, returning CollectionReport.
        """
        return self.collector.process_result(
            clear_frame=clear_frame,
            button_frames=button_frames,
            result_event=result_event,
            timestamp=timestamp
        )

    def process_result_frame(
        self,
        clear_frame: np.ndarray,
        button_frames: List[np.ndarray],
        result_event: ResultDetectionEvent,
        timestamp: float
    ) -> Optional[LearningOutcome]:
        """
        Attempts to verify and learn the result frame.
        Retries cleanly on subsequent frames if OCR is not yet stable.
        """
        report: Optional[CollectionReport] = self.collector.process_result(
            clear_frame=clear_frame,
            button_frames=button_frames,
            result_event=result_event,
            timestamp=timestamp
        )

        if report is None:
            return None

        status = report.status
        if status == "VERIFIED":
            outcome_status = "LEARNED"
        elif status == "DISAGREEMENT":
            outcome_status = "DISAGREEMENT"
        elif "TIMEOUT" in status:
            outcome_status = "TIMEOUT"
        else:
            outcome_status = "IGNORED"

        return LearningOutcome(
            session_id=report.session_id,
            card_id=report.card_id if report.card_id >= 0 else None,
            card_name=report.card_name,
            verified=report.verified,
            status=outcome_status,
            sample_id=report.sample_id,
            message=report.message,
            ground_truth_source=report.green_button_name
        )

