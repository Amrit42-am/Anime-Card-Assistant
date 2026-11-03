"""
src/ai package

Autonomous offline AI subsystem for Anime Card Assistant:
- Model Manager: Local offline backbone & encoder access.
- Confidence Engine: Calibrated confidence and victory-margin analysis.
- Candidate Reasoner: Candidate-constrained visual matching & ranking.
- Temporal Analyzer: Multi-frame temporal smoothing and state tracking.
- Context Analyzer: Quiz lifecycle finite state machine.
- Perception Engine: Unified display monitoring, cropping, and OCR.
- Adaptive Learning: Dual-verified result learning and memory updates.
"""

from src.ai.model_manager import ModelManager
from src.ai.confidence_engine import ConfidenceEngine, ConfidenceVerdict
from src.ai.candidate_reasoner import CandidateReasoner, ReasonedAnswer
from src.ai.temporal_analyzer import TemporalAnalyzer
from src.ai.context_analyzer import ContextAnalyzer, QuizState
from src.ai.perception_engine import PerceptionEngine, PerceptionFrame
from src.ai.adaptive_learning import AdaptiveLearning, LearningOutcome

__all__ = [
    "ModelManager",
    "ConfidenceEngine",
    "ConfidenceVerdict",
    "CandidateReasoner",
    "ReasonedAnswer",
    "TemporalAnalyzer",
    "ContextAnalyzer",
    "QuizState",
    "PerceptionEngine",
    "PerceptionFrame",
    "AdaptiveLearning",
    "LearningOutcome",
]
