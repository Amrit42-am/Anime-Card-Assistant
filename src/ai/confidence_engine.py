"""
src/ai/confidence_engine.py

Evidence-based calibrated confidence estimation and decision boundary verification.
Translates visual similarity, victory margin, OCR quality, candidate completeness,
and temporal agreement into calibrated confidence scores with explainable verdicts.
"""

from dataclasses import dataclass
import logging
import math
from typing import Optional, List

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
    confidence_level: str  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"


class ConfidenceEngine:
    """
    Evaluates visual match confidence by measuring similarity magnitude,
    runner-up margin of victory, candidate completeness, and temporal stability.
    """

    def __init__(
        self,
        min_top_similarity: float = 0.62,
        min_margin: float = 0.04,
        confident_threshold: float = 0.70,
        temperature: float = 0.08,
        fallback_min_top_similarity: float = 0.78,
        fallback_min_margin: float = 0.08
    ):
        self.min_top_similarity = min_top_similarity
        self.min_margin = min_margin
        self.confident_threshold = confident_threshold
        self.temperature = temperature
        self.fallback_min_top_similarity = fallback_min_top_similarity
        self.fallback_min_margin = fallback_min_margin

    def evaluate(
        self,
        top_score: float,
        runner_up_score: float,
        candidate_constrained: bool = True,
        source: str = "visual",
        ocr_quality: float = 1.0,
        candidate_count: int = 4,
        temporal_agreement: float = 1.0
    ) -> ConfidenceVerdict:
        """
        Calculates calibrated confidence and determines if the prediction is trustworthy.
        Uses evidence-based multi-factor temperature scaling without arbitrary hardcoded bonuses.
        """
        top_score = max(0.0, min(1.0, float(top_score)))
        runner_up_score = max(0.0, min(1.0, float(runner_up_score)))
        margin = max(0.0, top_score - runner_up_score)

        # Determine threshold requirements based on constraint level
        if candidate_constrained:
            req_top_sim = self.min_top_similarity
            req_margin = self.min_margin
            anchor_sim = 0.60
        else:
            req_top_sim = self.fallback_min_top_similarity
            req_margin = self.fallback_min_margin
            anchor_sim = 0.75

        # Incorporate evidence: OCR fidelity and candidate completeness factor
        completeness_factor = min(1.0, candidate_count / 4.0) if candidate_constrained else 0.85
        evidence_weight = (0.7 * top_score) + (0.3 * (ocr_quality * completeness_factor))

        norm_diff = (evidence_weight - anchor_sim) + (margin * 2.0)
        calibrated = 1.0 / (1.0 + math.exp(-norm_diff / self.temperature))
        calibrated = round(float(calibrated), 4)

        reasons: List[str] = []
        is_confident = True

        if top_score < req_top_sim:
            is_confident = False
            reasons.append(f"Top similarity ({top_score:.3f}) < required ({req_top_sim:.3f})")

        if margin < req_margin and top_score < 0.92:
            is_confident = False
            reasons.append(f"Margin of victory ({margin:.3f}) < required ({req_margin:.3f})")

        if calibrated < self.confident_threshold:
            is_confident = False
            reasons.append(f"Calibrated confidence ({calibrated:.2f}) < threshold ({self.confident_threshold:.2f})")

        if candidate_constrained and candidate_count < 4:
            is_confident = False
            reasons.append(f"Incomplete candidate set ({candidate_count}/4 resolved)")

        if temporal_agreement < 0.60:
            is_confident = False
            calibrated = min(calibrated, round(calibrated * temporal_agreement, 4))
            reasons.append(f"Low temporal agreement ({temporal_agreement:.2f} < 0.60)")

        if is_confident:
            if calibrated >= 0.85:
                level = "HIGH"
            elif calibrated >= 0.70:
                level = "MEDIUM"
            else:
                level = "LOW"
            reasons.append(f"Confident match (sim={top_score:.3f}, margin={margin:.3f}, conf={calibrated:.2f})")
        else:
            level = "UNCERTAIN"

        explanation = "; ".join(reasons)

        return ConfidenceVerdict(
            calibrated_confidence=calibrated,
            margin_of_victory=round(margin, 4),
            is_confident=is_confident,
            explanation=explanation,
            top_score=round(top_score, 4),
            runner_up_score=round(runner_up_score, 4),
            candidate_constrained=candidate_constrained,
            confidence_level=level
        )
