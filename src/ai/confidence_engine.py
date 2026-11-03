"""
src/ai/confidence_engine.py

Calibrated confidence estimation and decision boundary verification.
Translates raw cosine similarities into probabilistic confidence scores
using temperature scaling, runner-up margin analysis, and candidate constraints.
"""

from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger("confidence_engine")


@dataclass
class ConfidenceVerdict:
    calibrated_confidence: float
    margin_of_victory: float
    is_confident: bool
    explanation: str
    top_score: float
    runner_up_score: float
    candidate_constrained: bool


class ConfidenceEngine:
    """
    Evaluates visual match confidence by measuring similarity magnitude,
    runner-up margin of victory, and candidate search constraint.
    """

    def __init__(
        self,
        min_top_similarity: float = 0.62,
        min_margin: float = 0.04,
        confident_threshold: float = 0.70,
        temperature: float = 0.08
    ):
        self.min_top_similarity = min_top_similarity
        self.min_margin = min_margin
        self.confident_threshold = confident_threshold
        self.temperature = temperature

    def evaluate(
        self,
        top_score: float,
        runner_up_score: float,
        candidate_constrained: bool = True,
        source: str = "visual"
    ) -> ConfidenceVerdict:
        """
        Calculates calibrated confidence and determines if the prediction is trustworthy.
        """
        top_score = max(0.0, min(1.0, float(top_score)))
        runner_up_score = max(0.0, min(1.0, float(runner_up_score)))
        margin = max(0.0, top_score - runner_up_score)

        # Baseline sigmoid temperature calibration over margin and top similarity
        # Shift anchor based on candidate constraint
        anchor_sim = 0.60 if candidate_constrained else 0.72
        norm_diff = (top_score - anchor_sim) + (margin * 1.8)

        # Sigmoid scaling
        calibrated = 1.0 / (1.0 + math.exp(-norm_diff / self.temperature))
        calibrated = round(float(calibrated), 4)

        # Verification rules:
        # 1. Top score must meet minimum similarity
        # 2. Margin must meet minimum threshold unless top score is overwhelmingly high (>0.92)
        # 3. If source is confirmed learned blur memory, confidence receives positive prior
        if "learned" in source.lower():
            calibrated = min(1.0, calibrated + 0.08)

        is_confident = False
        reasons = []

        if top_score < self.min_top_similarity:
            reasons.append(f"Top similarity ({top_score:.3f}) < min threshold ({self.min_top_similarity:.3f})")
        elif margin < self.min_margin and top_score < 0.90:
            reasons.append(f"Margin of victory ({margin:.3f}) < min margin ({self.min_margin:.3f})")
        elif calibrated < self.confident_threshold:
            reasons.append(f"Calibrated score ({calibrated:.2f}) < threshold ({self.confident_threshold:.2f})")
        else:
            is_confident = True
            reasons.append(f"Confident match (sim={top_score:.3f}, margin={margin:.3f}, conf={calibrated:.2f})")

        explanation = "; ".join(reasons)

        return ConfidenceVerdict(
            calibrated_confidence=calibrated,
            margin_of_victory=round(margin, 4),
            is_confident=is_confident,
            explanation=explanation,
            top_score=round(top_score, 4),
            runner_up_score=round(runner_up_score, 4),
            candidate_constrained=candidate_constrained
        )
