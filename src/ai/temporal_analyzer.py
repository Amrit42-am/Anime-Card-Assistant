"""
src/ai/temporal_analyzer.py

Multi-frame temporal tracking and confidence stabilization for Anime Card Assistant.
Mitigates single-frame noise, resolves momentary OCR errors via temporal voting,
monitors timer depletion rates, and detects state transitions.
"""

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Optional, List, Dict, Any

from src.ai.candidate_reasoner import ReasonedAnswer

logger = logging.getLogger("temporal_analyzer")


@dataclass
class TemporalSnapshot:
    answer: ReasonedAnswer
    timestamp: float
    timer_value: Optional[float]
    is_result_frame: bool


class TemporalAnalyzer:
    """
    Maintains a rolling buffer of consecutive frame inferences to stabilize
    predictions, damp jitter, and detect question/result transitions.
    """

    def __init__(self, window_size: int = 5, min_agreement_ratio: float = 0.5):
        self.window_size = window_size
        self.min_agreement_ratio = min_agreement_ratio
        self.history: deque[TemporalSnapshot] = deque(maxlen=window_size)
        self.last_timer: Optional[float] = None
        self.timer_trend: float = 0.0  # rate of change

    def push(
        self,
        answer: Optional[ReasonedAnswer],
        timer_val: Optional[float] = None,
        is_result: bool = False,
        timestamp: Optional[float] = None
    ):
        """Records a new inference snapshot into the temporal window."""
        now = timestamp or time.time()
        if timer_val is not None and self.last_timer is not None:
            dt = 0.1  # typical frame interval
            self.timer_trend = (timer_val - self.last_timer) / dt
        self.last_timer = timer_val

        if answer is not None:
            self.history.append(TemporalSnapshot(
                answer=answer,
                timestamp=now,
                timer_value=timer_val,
                is_result_frame=is_result
            ))

    def get_stabilized_answer(self) -> Optional[ReasonedAnswer]:
        """
        Computes the temporally stabilized answer across the rolling window.
        Uses confidence-weighted voting.
        """
        if not self.history:
            return None

        # Filter out result frames when seeking active answer
        active_snaps = [s for s in self.history if not s.is_result_frame]
        if not active_snaps:
            return self.history[-1].answer

        # Accumulate weights per card name
        votes: Dict[str, float] = {}
        candidate_map: Dict[str, ReasonedAnswer] = {}

        for snap in active_snaps:
            ans = snap.answer
            name = ans.card_name
            weight = max(0.1, ans.confidence)
            votes[name] = votes.get(name, 0.0) + weight
            candidate_map[name] = ans

        # Find candidate with highest accumulated weight
        top_name = max(votes, key=votes.get)
        top_weight = votes[top_name]
        total_weight = sum(votes.values())

        agreement_ratio = top_weight / total_weight if total_weight > 0 else 0.0
        best_answer = candidate_map[top_name]

        if agreement_ratio >= self.min_agreement_ratio:
            return best_answer

        # If voting is fragmented, return the latest answer with lower confidence flag
        latest = active_snaps[-1].answer
        return latest

    def is_transition_detected(self) -> bool:
        """
        Detects sudden jump in timer or blanking indicating a new question.
        """
        if self.timer_trend > 2.0:  # Timer jumped up (e.g. from 1.0s to 10.0s)
            return True
        return False

    def clear(self):
        """Clears temporal buffer for a new question round."""
        self.history.clear()
        self.last_timer = None
        self.timer_trend = 0.0
