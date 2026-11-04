"""
src/ai/context_analyzer.py

Quiz Lifecycle State Machine and Context Analyzer.
Manages explicit state transitions across:
SEARCHING -> QUESTION_DETECTED -> ANALYZING -> ANSWER_DISPLAYED ->
RESULT_DETECTED -> VERIFYING -> LEARNING -> TRANSITION.
"""

from dataclasses import dataclass
from enum import Enum, auto
import logging
import time
from typing import Optional, List

logger = logging.getLogger("context_analyzer")


class QuizState(Enum):
    SEARCHING = auto()
    QUESTION_DETECTED = auto()
    ANALYZING = auto()
    ANSWER_DISPLAYED = auto()
    RESULT_DETECTED = auto()
    VERIFYING = auto()
    LEARNING = auto()
    TRANSITION = auto()

    # Backwards compatibility properties / aliases
    @property
    def is_active_question(self) -> bool:
        return self in (QuizState.QUESTION_DETECTED, QuizState.ANALYZING, QuizState.ANSWER_DISPLAYED)

    @property
    def is_result_screen(self) -> bool:
        return self in (QuizState.RESULT_DETECTED, QuizState.VERIFYING, QuizState.LEARNING)


# Provide aliases for legacy references if any
QuizState.IDLE = QuizState.SEARCHING
QuizState.ACTIVE_QUESTION = QuizState.QUESTION_DETECTED
QuizState.RESULT_REVEALED = QuizState.RESULT_DETECTED


@dataclass
class QuestionSession:
    session_id: str
    card_hash: str
    options: List[str]
    start_time: float
    last_update: float
    predicted_name: Optional[str] = None
    predicted_slot: Optional[int] = None
    has_predicted: bool = False
    result_processed: bool = False
    ocr_validated: bool = False
    attempts: int = 0


class ContextAnalyzer:
    """
    Finite State Machine tracking quiz states, isolating question sessions,
    and handling timeout expiration.
    """

    def __init__(self, session_timeout: float = 18.0):
        self.session_timeout = session_timeout
        self.current_state: QuizState = QuizState.SEARCHING
        self.active_session: Optional[QuestionSession] = None
        self._session_counter: int = 0
        self._last_question_hash: Optional[str] = None
        self._last_question_time: float = 0.0

    @staticmethod
    def hex_hash_distance(h1: str, h2: str) -> int:
        """Computes Hamming distance between two hex hashes."""
        if not h1 or not h2:
            return 999
        try:
            val1 = int(h1, 16)
            val2 = int(h2, 16)
            return bin(val1 ^ val2).count("1")
        except ValueError:
            return sum(c1 != c2 for c1, c2 in zip(h1, h2))

    def evaluate_transition(
        self,
        is_active_question: bool,
        is_result_screen: bool,
        card_hash: Optional[str] = None,
        options: Optional[List[str]] = None,
        timestamp: Optional[float] = None
    ) -> QuizState:
        """
        Transitions the state machine according to screen perception signals.
        Ensures consecutive questions with the same card get distinct session IDs.
        """
        now = timestamp or time.time()

        # 1. Session Expiration Check
        if self.active_session is not None:
            if now - self.active_session.start_time > self.session_timeout:
                logger.info("Session %s timed out after %.1fs. Returning to SEARCHING.",
                            self.active_session.session_id, now - self.active_session.start_time)
                self.active_session = None
                self.current_state = QuizState.SEARCHING

        # 2. Result state transition
        if is_result_screen:
            if self.active_session and not self.active_session.result_processed:
                self.current_state = QuizState.RESULT_DETECTED
                self.active_session.last_update = now
            else:
                self.current_state = QuizState.TRANSITION

        # 3. Active question state transition
        elif is_active_question:
            # Detect new question round:
            # - No active session, OR
            # - Materially different card hash (Hamming distance > 8 bits, not minor unblurring), OR
            # - Options list changed materially, OR
            # - Active session already had its result processed, OR
            # - Previously in result/transition/searching state
            hash_mismatch = False
            if card_hash and self._last_question_hash:
                hash_mismatch = (self.hex_hash_distance(card_hash, self._last_question_hash) > 8)

            options_mismatch = False
            if options and self.active_session and self.active_session.options:
                common = set(o.lower().strip() for o in options) & set(o.lower().strip() for o in self.active_session.options)
                if len(common) <= 1 and len(options) >= 3:
                    options_mismatch = True

            was_in_result_or_transition = (
                self.current_state in (
                    QuizState.RESULT_DETECTED,
                    QuizState.VERIFYING,
                    QuizState.LEARNING,
                    QuizState.TRANSITION
                )
            )

            is_new_question = (
                self.active_session is None or
                hash_mismatch or
                options_mismatch or
                self.active_session.result_processed or
                was_in_result_or_transition
            )

            if is_new_question:
                self._session_counter += 1
                sid = f"q_{int(now)}_{self._session_counter}"
                self.active_session = QuestionSession(
                    session_id=sid,
                    card_hash=card_hash or f"hash_{self._session_counter}",
                    options=options or [],
                    start_time=now,
                    last_update=now
                )
                self._last_question_hash = card_hash
                self._last_question_time = now
                self.current_state = QuizState.QUESTION_DETECTED
                logger.info("ContextAnalyzer: Started new QuestionSession %s", sid)
            else:
                if self.active_session.has_predicted:
                    self.current_state = QuizState.ANSWER_DISPLAYED
                else:
                    self.current_state = QuizState.ANALYZING

            self.active_session.last_update = now

        # 4. Idle / Transition
        else:
            if self.current_state in (QuizState.RESULT_DETECTED, QuizState.VERIFYING, QuizState.LEARNING, QuizState.TRANSITION):
                self.current_state = QuizState.TRANSITION
            else:
                self.current_state = QuizState.SEARCHING

        return self.current_state

    def mark_ocr_status(self, is_valid: bool, options: Optional[List[str]] = None):
        """Records current OCR validation status for the session."""
        if self.active_session is not None:
            self.active_session.ocr_validated = is_valid
            if options:
                self.active_session.options = options

    def mark_analyzing(self):
        """Marks state as actively computing visual inferences."""
        if self.active_session:
            self.current_state = QuizState.ANALYZING

    def mark_predicted(self, card_name: str, slot_index: Optional[int]):
        """Records that a valid prediction was emitted for the active question."""
        if self.active_session is not None:
            # Only mark has_predicted if an actual card was predicted (not a waiting/recovery message)
            if card_name and not card_name.lower().startswith("waiting") and slot_index is not None:
                self.active_session.predicted_name = card_name
                self.active_session.predicted_slot = slot_index
                self.active_session.has_predicted = True
                self.current_state = QuizState.ANSWER_DISPLAYED
            else:
                self.current_state = QuizState.ANALYZING

    def mark_verifying(self):
        """Marks that result verification is in progress."""
        self.current_state = QuizState.VERIFYING

    def mark_learning(self):
        """Marks that sample is being written to memory."""
        self.current_state = QuizState.LEARNING

    def mark_result_completed(self):
        """Marks result processing complete so duplicate frames are skipped."""
        if self.active_session is not None:
            self.active_session.result_processed = True
            self.current_state = QuizState.TRANSITION

    @property
    def current_session_id(self) -> Optional[str]:
        return self.active_session.session_id if self.active_session else None

    def should_predict(self) -> bool:
        """Determines if a prediction should be emitted for the current frame."""
        return (
            self.current_state in (QuizState.QUESTION_DETECTED, QuizState.ANALYZING) and
            self.active_session is not None and
            not self.active_session.has_predicted
        )

    def is_in_active_question(self) -> bool:
        """Returns True if a question session is currently active and awaiting result."""
        return (
            self.active_session is not None and
            not self.active_session.result_processed and
            self.current_state in (
                QuizState.QUESTION_DETECTED,
                QuizState.ANALYZING,
                QuizState.ANSWER_DISPLAYED
            )
        )

    def should_process_result(self) -> bool:
        """Determines if the result screen should be processed."""
        return (
            self.current_state in (QuizState.RESULT_DETECTED, QuizState.VERIFYING) and
            self.active_session is not None and
            not self.active_session.result_processed
        )

    def reset(self):
        """Resets the state machine."""
        self.current_state = QuizState.SEARCHING
        self.active_session = None
        self._last_question_hash = None
        self._last_question_time = 0.0
