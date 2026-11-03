"""
src/recognition/confidence.py

Calibrated confidence estimation for card recognition.
Combines raw cosine similarity, top-1 vs top-2 candidate margin,
and learned blur confirmation status to avoid over-confident false positives.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfidenceReport:
    raw_score: float
    margin: float
    calibrated_confidence: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    is_confident: bool
    explanation: str


class ConfidenceCalibrator:
    """
    Calibrates decision confidence from visual similarity scores.
    """

    def __init__(
        self,
        high_threshold: float = 0.70,
        medium_threshold: float = 0.55,
        min_margin: float = 0.04,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.min_margin = min_margin

    def evaluate(
        self,
        top_score: float,
        second_score: Optional[float] = None,
        source: str = "synthetic_blurs"
    ) -> ConfidenceReport:
        """
        Evaluate confidence based on top score, second score, and evidence source.
        """
        if second_score is None:
            margin = 1.0
        else:
            margin = float(top_score - second_score)

        # Learned confirmed blurs receive a reliability bonus
        source_bonus = 0.05 if source == "learned_memory" else 0.0

        # Calibrated score combines raw similarity and margin
        # A large margin between candidate 1 and candidate 2 provides strong evidence
        margin_factor = min(1.0, max(0.0, margin / 0.15))
        calibrated = float(0.65 * top_score + 0.35 * margin_factor + source_bonus)
        calibrated = max(0.0, min(0.99, calibrated))

        if top_score >= self.high_threshold and margin >= self.min_margin:
            level = "HIGH"
            is_conf = True
            expl = f"Strong match ({top_score:.2f}) with clear margin (+{margin:.2f}) from {source}."
        elif top_score >= self.medium_threshold and margin >= (self.min_margin * 0.7):
            level = "MEDIUM"
            is_conf = True
            expl = f"Moderate match ({top_score:.2f}) with acceptable margin (+{margin:.2f})."
        elif margin < (self.min_margin * 0.5) and top_score >= self.medium_threshold:
            level = "UNCERTAIN"
            is_conf = False
            expl = f"Ambiguous candidates: top score {top_score:.2f} but margin only +{margin:.2f}."
        else:
            level = "LOW"
            is_conf = False
            expl = f"Low similarity score ({top_score:.2f}) below threshold."

        return ConfidenceReport(
            raw_score=round(top_score, 4),
            margin=round(margin, 4),
            calibrated_confidence=round(calibrated, 4),
            confidence_level=level,
            is_confident=is_conf,
            explanation=expl
        )
