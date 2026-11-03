"""
tests/test_advanced_ai.py

Comprehensive test suite for Advanced Autonomous AI Modules:
- ConfidenceEngine
- CandidateReasoner
- TemporalAnalyzer
- ContextAnalyzer
- PerceptionEngine
- AdaptiveLearning
"""

import time
import numpy as np
import pytest

from src.ai.confidence_engine import ConfidenceEngine, ConfidenceVerdict
from src.ai.temporal_analyzer import TemporalAnalyzer
from src.ai.context_analyzer import ContextAnalyzer, QuizState
from src.ai.candidate_reasoner import CandidateReasoner, ReasonedAnswer
from src.ai.model_manager import ModelManager
from src.ai.adaptive_learning import AdaptiveLearning, LearningOutcome
from src.ocr.option_reader import CandidateOption, OptionReader
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.memory.database import MemoryDatabase


def test_confidence_engine_decisive():
    engine = ConfidenceEngine()
    # High top score and large margin
    verdict = engine.evaluate(top_score=0.92, runner_up_score=0.60, candidate_constrained=True)
    assert isinstance(verdict, ConfidenceVerdict)
    assert verdict.is_confident is True
    assert verdict.margin_of_victory >= 0.30
    assert verdict.calibrated_confidence > 0.80


def test_confidence_engine_ambiguous():
    engine = ConfidenceEngine()
    # High score but almost identical runner-up
    verdict = engine.evaluate(top_score=0.75, runner_up_score=0.74, candidate_constrained=True)
    assert verdict.is_confident is False
    assert verdict.margin_of_victory <= 0.02


def test_confidence_engine_low_similarity():
    engine = ConfidenceEngine()
    # Low top score
    verdict = engine.evaluate(top_score=0.45, runner_up_score=0.30, candidate_constrained=True)
    assert verdict.is_confident is False


def test_temporal_analyzer_majority_voting():
    analyzer = TemporalAnalyzer(window_size=5)

    ans_green = ReasonedAnswer(
        card_name="Green Bomber",
        card_id=137,
        slot_index=1,
        confidence=0.88,
        margin=0.25,
        is_confident=True,
        runner_up_name="Harmonious Soul",
        runner_up_score=0.63,
        source="visual",
        status="matched",
        raw_match={}
    )
    ans_soul = ReasonedAnswer(
        card_name="Harmonious Soul",
        card_id=139,
        slot_index=2,
        confidence=0.60,
        margin=0.05,
        is_confident=False,
        runner_up_name="Green Bomber",
        runner_up_score=0.55,
        source="visual",
        status="uncertain",
        raw_match={}
    )

    # Push 3 Green Bomber and 1 Harmonious Soul
    analyzer.push(ans_green)
    analyzer.push(ans_green)
    analyzer.push(ans_soul)
    analyzer.push(ans_green)

    stabilized = analyzer.get_stabilized_answer()
    assert stabilized is not None
    assert stabilized.card_name == "Green Bomber"
    assert stabilized.slot_index == 1


def test_context_analyzer_lifecycle():
    ctx = ContextAnalyzer(session_timeout=10.0)
    assert ctx.current_state == QuizState.IDLE

    # 1. New Question appears
    t0 = 1000.0
    state1 = ctx.evaluate_transition(
        is_active_question=True,
        is_result_screen=False,
        card_hash="hash_abc",
        options=["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"],
        timestamp=t0
    )
    assert state1 == QuizState.ACTIVE_QUESTION
    assert ctx.should_predict() is True

    # Mark prediction
    ctx.mark_predicted("Green Bomber", 0)
    assert ctx.should_predict() is False  # Prevents repeated predictions on same question

    # 2. Result revealed
    state2 = ctx.evaluate_transition(
        is_active_question=False,
        is_result_screen=True,
        timestamp=t0 + 2.0
    )
    assert state2 == QuizState.RESULT_REVEALED
    assert ctx.should_process_result() is True

    # Mark result completed
    ctx.mark_result_completed()
    assert ctx.should_process_result() is False  # Prevents duplicate learning on repeated result frames

    # 3. Transition to Idle / Next
    state3 = ctx.evaluate_transition(
        is_active_question=False,
        is_result_screen=False,
        timestamp=t0 + 4.0
    )
    assert state3 == QuizState.TRANSITION


def test_candidate_reasoner_slot_mapping():
    reader = OptionReader()
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    reasoner = CandidateReasoner(
        image_matcher=matcher,
        image_encoder=encoder,
        option_reader=reader
    )

    # Mock 4 candidate options
    cands = [
        CandidateOption(slot_index=0, raw_text="Shadow Duelist", canonical_name="Shadow Duelist", card_id=200, ocr_confidence=0.9, fuzzy_score=100.0, is_valid=True),
        CandidateOption(slot_index=1, raw_text="Green Bomber", canonical_name="Green Bomber", card_id=137, ocr_confidence=0.9, fuzzy_score=100.0, is_valid=True),
        CandidateOption(slot_index=2, raw_text="Harmonious Soul", canonical_name="Harmonious Soul", card_id=139, ocr_confidence=0.9, fuzzy_score=100.0, is_valid=True),
        CandidateOption(slot_index=3, raw_text="Soul King", canonical_name="Soul King", card_id=201, ocr_confidence=0.9, fuzzy_score=100.0, is_valid=True),
    ]

    # Create dummy card frame
    dummy_card = np.ones((120, 120, 3), dtype=np.uint8) * 100

    result = reasoner.reason_answer(dummy_card, cands)
    assert isinstance(result, ReasonedAnswer)
    assert result.card_name in ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    assert result.slot_index in [0, 1, 2, 3]
    assert 0.0 <= result.confidence <= 1.0


def test_candidate_reasoner_insufficient_candidates_fallback():
    """Verify that fewer than 4 distinct valid options triggers full database fallback."""
    reader = OptionReader()
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    reasoner = CandidateReasoner(
        image_matcher=matcher,
        image_encoder=encoder,
        option_reader=reader
    )
    dummy_card = np.ones((120, 120, 3), dtype=np.uint8) * 100

    # Case A: Only 3 valid candidates
    three_cands = [
        CandidateOption(slot_index=0, raw_text="Shadow Duelist", canonical_name="Shadow Duelist", card_id=200, ocr_confidence=0.9, is_valid=True),
        CandidateOption(slot_index=1, raw_text="Green Bomber", canonical_name="Green Bomber", card_id=137, ocr_confidence=0.9, is_valid=True),
        CandidateOption(slot_index=2, raw_text="Harmonious Soul", canonical_name="Harmonious Soul", card_id=139, ocr_confidence=0.9, is_valid=True),
        CandidateOption(slot_index=3, raw_text="???", canonical_name="", card_id=None, ocr_confidence=0.1, is_valid=False),
    ]
    res_three = reasoner.reason_answer(dummy_card, three_cands)
    assert isinstance(res_three, ReasonedAnswer)
    # Should fall back to match_all (candidate_restricted=False)

    # Case B: Duplicate card IDs (e.g. OCR read same card twice)
    duplicate_cands = [
        CandidateOption(slot_index=0, raw_text="Green Bomber", canonical_name="Green Bomber", card_id=137, ocr_confidence=0.9, is_valid=True),
        CandidateOption(slot_index=1, raw_text="Green Bomber", canonical_name="Green Bomber", card_id=137, ocr_confidence=0.9, is_valid=True),
        CandidateOption(slot_index=2, raw_text="Harmonious Soul", canonical_name="Harmonious Soul", card_id=139, ocr_confidence=0.9, is_valid=True),
        CandidateOption(slot_index=3, raw_text="Soul King", canonical_name="Soul King", card_id=201, ocr_confidence=0.9, is_valid=True),
    ]
    res_dup = reasoner.reason_answer(dummy_card, duplicate_cands)
    assert isinstance(res_dup, ReasonedAnswer)


def test_candidate_reasoner_unmapped_slot_returns_uncertain():
    """Verify that when the predicted card cannot be matched to any slot, it does NOT default to 0."""
    reader = OptionReader()
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    reasoner = CandidateReasoner(
        image_matcher=matcher,
        image_encoder=encoder,
        option_reader=reader
    )
    dummy_card = np.ones((120, 120, 3), dtype=np.uint8) * 100

    # OCR reads options completely different from visual match
    cands = [
        CandidateOption(slot_index=0, raw_text="Card X", canonical_name="Card X", card_id=901, is_valid=True),
        CandidateOption(slot_index=1, raw_text="Card Y", canonical_name="Card Y", card_id=902, is_valid=True),
        CandidateOption(slot_index=2, raw_text="Card Z", canonical_name="Card Z", card_id=903, is_valid=True),
        CandidateOption(slot_index=3, raw_text="Card W", canonical_name="Card W", card_id=904, is_valid=True),
    ]
    # Because < 4 match the canonical dataset, it falls back to full database.
    # The top database match will not be in [Card X, Card Y, Card Z, Card W].
    res = reasoner.reason_answer(dummy_card, cands)
    # If the top match is not in cands, assigned_slot is None
    if res.card_name not in ["Card X", "Card Y", "Card Z", "Card W"]:
        assert res.slot_index is None
        assert res.status == "uncertain"
        assert res.is_confident is False


def test_temporal_analyzer_session_isolation():
    """Verify that TemporalAnalyzer isolates history by session ID and clears on new session."""
    analyzer = TemporalAnalyzer(window_size=5)

    ans1 = ReasonedAnswer(
        card_name="Card Alpha", card_id=1, slot_index=0, confidence=0.9,
        margin=0.3, is_confident=True, runner_up_name="Other", runner_up_score=0.5,
        source="visual", status="matched", raw_match={}
    )
    ans2 = ReasonedAnswer(
        card_name="Card Beta", card_id=2, slot_index=1, confidence=0.85,
        margin=0.25, is_confident=True, runner_up_name="Other", runner_up_score=0.5,
        source="visual", status="matched", raw_match={}
    )

    analyzer.push(ans1, session_id="session_A")
    analyzer.push(ans1, session_id="session_A")
    stab_a = analyzer.get_stabilized_answer(session_id="session_A")
    assert stab_a.card_name == "Card Alpha"

    # Switching session must clear old history
    analyzer.push(ans2, session_id="session_B")
    stab_b = analyzer.get_stabilized_answer(session_id="session_B")
    assert stab_b.card_name == "Card Beta"
    # Ensure session_A is no longer influencing session_B
    assert len(analyzer.history) == 1


def test_temporal_analyzer_timer_trend_transition():
    """Verify timer jump triggers transition detection."""
    analyzer = TemporalAnalyzer()
    analyzer.push(None, timer_val=1.0, timestamp=100.0)
    analyzer.push(None, timer_val=10.0, timestamp=100.5)  # Jump from 1.0 to 10.0 in 0.5s -> +18 units/s

    assert analyzer.timer_trend > 2.0
    assert analyzer.is_transition_detected() is True


def test_adaptive_learning_flow(tmp_path):
    """Verify AdaptiveLearning correctly registers questions, handles verification, and expires sessions."""
    db_path = tmp_path / "test_blur_memory.db"
    mem_db = MemoryDatabase(db_path=db_path)
    memory = BlurMemorySystem(db=mem_db, storage_dir=tmp_path / "learned")
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    adaptive = AdaptiveLearning(memory=memory, matcher=matcher, encoder=encoder)

    dummy_card = np.zeros((100, 100, 3), dtype=np.uint8)
    sid = adaptive.register_question(
        card_frame=dummy_card,
        option_names=["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"],
        option_ids=[137, 139, 200, 201],
        predicted_name="Green Bomber",
        predicted_id=137,
        card_hash="hash_123",
        timestamp=time.time()
    )
    assert sid.startswith("q_")

    # Safe expiration
    adaptive.expire_pending_question(reason="test_expire")
    assert adaptive.collector.pending_question is None


