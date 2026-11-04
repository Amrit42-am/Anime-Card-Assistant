"""
src/ai/temporal_analyzer.py

Multi-frame temporal tracking and confidence stabilization for Anime Card Assistant.
Mitigates single-frame noise, resolves momentary OCR errors via temporal voting,
monitors timer depletion rates using actual elapsed timestamps, and isolates predictions
by question session ID.
"""

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Optional, Dict, Any

from src.ai.candidate_reasoner import ReasonedAnswer

logger = logging.getLogger("temporal_analyzer")


@dataclass
class TemporalSnapshot:
    session_id: str
    answer: ReasonedAnswer
    timestamp: float
    timer_value: Optional[float]
    is_result_frame: bool


class TemporalAnalyzer:
    """
    Maintains a rolling buffer of consecutive frame inferences to stabilize
    predictions, damp jitter, and detect question/result transitions.
    Isolates snapshots per question session to avoid cross-question contamination.
    """

    def __init__(self, window_size: int = 5, min_agreement_ratio: float = 0.5):
        self.window_size = window_size
        self.min_agreement_ratio = min_agreement_ratio
        self.history: deque[TemporalSnapshot] = deque(maxlen=window_size)
        self.current_session_id: Optional[str] = None
        self.last_timer: Optional[float] = None
        self.last_timer_time: Optional[float] = None
        self.timer_trend: float = 0.0  # units of timer change per second

    def set_session(self, session_id: str):
        """Switches active session and clears history if new question begins."""
        if self.current_session_id != session_id:
            logger.debug("TemporalAnalyzer switching from session %s to %s",
                         self.current_session_id, session_id)
            self.clear()
            self.current_session_id = session_id

    def push(
        self,
        answer: Optional[ReasonedAnswer],
        session_id: Optional[str] = None,
        timer_val: Optional[float] = None,
        is_result: bool = False,
        timestamp: Optional[float] = None
    ):
        """Records a new inference snapshot into the temporal window."""
        now = timestamp or time.time()

        if session_id and session_id != self.current_session_id:
            self.set_session(session_id)

        # Calculate timer trend using actual timestamp delta
        if timer_val is not None:
            if self.last_timer is not None and self.last_timer_time is not None:
                dt = max(0.001, now - self.last_timer_time)
                self.timer_trend = (timer_val - self.last_timer) / dt
            self.last_timer = timer_val
            self.last_timer_time = now

        if answer is not None:
            sid = session_id or self.current_session_id or "default"
            self.history.append(TemporalSnapshot(
                session_id=sid,
                answer=answer,
                timestamp=now,
                timer_value=timer_val,
                is_result_frame=is_result
            ))

    def get_stabilized_answer(
        self,
        session_id: Optional[str] = None,
        return_agreement: bool = False,
        return_details: bool = False
    ) -> Any:
        """
        Computes the temporally stabilized answer across the rolling window.
        Returns:
          - (best_answer, agreement_ratio, observation_count) if return_details=True
          - (best_answer, agreement_ratio) if return_agreement=True
          - best_answer otherwise
        Filters snapshots to ensure only answers from the requested session are counted.
        """
        if not self.history:
            if return_details:
                return (None, 0.0, 0)
            return (None, 0.0) if return_agreement else None

        target_sid = session_id or self.current_session_id

        # Filter out result frames and cross-session frames
        active_snaps = [
            s for s in self.history
            if not s.is_result_frame and (target_sid is None or s.session_id == target_sid)
        ]

        if not active_snaps:
            latest = self.history[-1].answer
            if return_details:
                return (latest, 1.0, 1)
            return (latest, 1.0) if return_agreement else latest

        # Accumulate weights per card name
        votes: Dict[str, float] = {}
        candidate_map: Dict[str, ReasonedAnswer] = {}

        for snap in active_snaps:
            ans = snap.answer
            name = ans.card_name
            weight = max(0.1, ans.confidence)
            votes[name] = votes.get(name, 0.0) + weight
            candidate_map[name] = ans

        top_name = max(votes, key=votes.get)
        top_weight = votes[top_name]
        total_weight = sum(votes.values())

        agreement_ratio = top_weight / total_weight if total_weight > 0 else 0.0
        best_answer = candidate_map[top_name]

        if return_details:
            return best_answer, round(agreement_ratio, 4), len(active_snaps)
        if return_agreement:
            return best_answer, round(agreement_ratio, 4)
        return best_answer

    def is_transition_detected(self) -> bool:
        """
        Detects sudden upward jump in timer indicating a new question round has begun.
        """
        if self.timer_trend > 2.0:  # Timer jumped up (e.g. from 0.5s to 10.0s)
            return True
        return False

    def clear(self):
        """Clears temporal buffer for a new question round."""
        self.history.clear()
        self.last_timer = None
        self.last_timer_time = None
        self.timer_trend = 0.0

    def reset(self):
        """Resets the analyzer and clears history and session."""
        self.clear()
        self.current_session_id = None
