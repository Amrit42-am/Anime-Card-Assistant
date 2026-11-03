"""
src/ai/context_analyzer.py

Quiz Lifecycle State Machine and Context Analyzer.
Orchestrates transitions between IDLE, ACTIVE_QUESTION, RESULT_REVEALED,
VERIFYING, and TRANSITION states, managing question sessions and timeout expiration.
"""

from dataclasses import dataclass
from enum import Enum, auto
import logging
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("context_analyzer")


class QuizState(Enum):
    IDLE = auto()
    ACTIVE_QUESTION = auto()
    RESULT_REVEALED = auto()
    TRANSITION = auto()


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
    attempts: int = 0


class ContextAnalyzer:
    """
    Finite State Machine tracking question lifecycle, preventing duplicate
    predictions and managing result verification windows.
    """

    def __init__(self, session_timeout: float = 18.0):
        self.session_timeout = session_timeout
        self.current_state: QuizState = QuizState.IDLE
        self.active_session: Optional[QuestionSession] = None
        self._session_counter: int = 0

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
        """
        now = timestamp or time.time()

        # Check for session expiration
        if self.active_session is not None:
            if now - self.active_session.start_time > self.session_timeout:
                logger.info("Session %s timed out after %.1fs. Resetting to IDLE.",
                            self.active_session.session_id, now - self.active_session.start_time)
                self.active_session = None
                self.current_state = QuizState.IDLE

        if is_result_screen:
            self.current_state = QuizState.RESULT_REVEALED
            if self.active_session:
                self.active_session.last_update = now

        elif is_active_question:
            # Check if this is a brand new question or continuation of current
            if self.active_session is None or (card_hash and self.active_session.card_hash != card_hash):
                self._session_counter += 1
                sid = f"q_{int(now)}_{self._session_counter}"
                self.active_session = QuestionSession(
                    session_id=sid,
                    card_hash=card_hash or f"hash_{self._session_counter}",
                    options=options or [],
                    start_time=now,
                    last_update=now
                )
                logger.info("Started new QuestionSession: %s", sid)

            self.current_state = QuizState.ACTIVE_QUESTION
            self.active_session.last_update = now

        else:
            if self.current_state == QuizState.RESULT_REVEALED:
                self.current_state = QuizState.TRANSITION
            else:
                self.current_state = QuizState.IDLE

        return self.current_state

    def mark_predicted(self, card_name: str, slot_index: int):
        """Records that prediction was emitted for the active question."""
        if self.active_session is not None:
            self.active_session.predicted_name = card_name
            self.active_session.predicted_slot = slot_index
            self.active_session.has_predicted = True

    def mark_result_completed(self):
        """Marks result processing complete so duplicate frames are skipped."""
        if self.active_session is not None:
            self.active_session.result_processed = True

    def should_predict(self) -> bool:
        """Determines if a prediction should be emitted for the current frame."""
        return (
            self.current_state == QuizState.ACTIVE_QUESTION and
            self.active_session is not None and
            not self.active_session.has_predicted
        )

    def should_process_result(self) -> bool:
        """Determines if the result screen should be processed."""
        return (
            self.current_state == QuizState.RESULT_REVEALED and
            self.active_session is not None and
            not self.active_session.result_processed
        )

    def reset(self):
        """Resets the state machine."""
        self.current_state = QuizState.IDLE
        self.active_session = None
