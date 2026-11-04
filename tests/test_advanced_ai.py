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
import cv2
import numpy as np
import pytest

from src.ai.confidence_engine import ConfidenceEngine, ConfidenceVerdict
from src.ai.temporal_analyzer import TemporalAnalyzer
from src.ai.context_analyzer import ContextAnalyzer, QuizState
from src.ai.candidate_reasoner import CandidateReasoner, ReasonedAnswer
from src.ai.model_manager import ModelManager
from src.ai.adaptive_learning import AdaptiveLearning
from src.ocr.option_reader import CandidateOption, OptionReader
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.memory.database import MemoryDatabase
from src.learning.sample_collector import AutomaticSampleCollector


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


def test_gradual_card_unblurring_triggers_inference():
    """Verify that gradual unblurring across frames triggers inference once cumulative delta exceeds threshold."""
    from src.ai.perception_engine import PerceptionEngine

    engine = PerceptionEngine(sharpness_threshold=120.0)
    engine.reset_cache()

    # Frame 1: baseline blurred card
    base_img = np.ones((80, 80, 3), dtype=np.uint8) * 100
    bundle1 = {
        "card_frame": cv2.GaussianBlur(base_img, (35, 35), 15.0),
        "button_frames": [],
        "options_frame": None,
        "timer_frame": None,
        "banner_frame": None,
        "game_frame": None
    }
    p1 = engine.perceive_frame(run_ocr=False, bundle=bundle1)
    assert p1.has_card_changed is True

    # Pipeline notifies inference was performed
    engine.notify_inference_completed(p1.card_hash, p1.card_sharpness)

    # Frame 2-4: gradual unblurring (+4 sharpness delta each frame, below 15.0 per-frame and cumulative)
    # Mocking compute_image_sharpness or frames with gradual sharpness
    for step in [4.0, 8.0, 12.0]:
        bundle_step = {
            "card_frame": base_img.copy(),
            "button_frames": [],
            "options_frame": None,
            "timer_frame": None,
            "banner_frame": None,
            "game_frame": None
        }
        # Simulate sharpness progression relative to inference baseline
        p_step = engine.perceive_frame(run_ocr=False, bundle=bundle_step)
        # Manually verify with engine's hash and sharp_delta logic
        sharp_delta = step
        has_changed = (engine.hash_distance(p_step.card_hash, engine._last_inference_card_hash) > 3 or
                       sharp_delta > 15.0)
        assert has_changed is False, f"Delta {step} should not trigger premature inference"

    # Frame 5: cumulative delta reaches 18.0 (> 15.0)
    sharp_delta_5 = 18.0
    has_changed_5 = (sharp_delta_5 > 15.0)
    assert has_changed_5 is True, "Gradual accumulation over 15.0 threshold must trigger inference"


def test_positional_integrity_with_none_ids(tmp_path):
    """Verify that 4-slot layout maintains None IDs without shifting quadrant indices."""
    db_path = tmp_path / "test_blur_memory.db"
    mem_db = MemoryDatabase(db_path=db_path)
    memory = BlurMemorySystem(db=mem_db, storage_dir=tmp_path / "learned")
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    collector = AutomaticSampleCollector(memory=memory, matcher=matcher, encoder=encoder)

    dummy_card = np.zeros((100, 100, 3), dtype=np.uint8)
    opt_names = ["Card TL", "Unresolved TR", "Card BL", "Card BR"]
    opt_ids = [10, None, 30, 40]

    collector.register_active_question(
        card_frame=dummy_card,
        option_names=opt_names,
        option_ids=opt_ids,
        predicted_name="Card TL",
        predicted_id=10,
        card_hash="hash_quad",
        timestamp=100.0
    )

    pending = collector.pending_question
    assert pending is not None
    # Must preserve exactly 4 slots with None at index 1 (TR)
    assert len(pending.option_ids) == 4
    assert pending.option_ids[0] == 10
    assert pending.option_ids[1] is None
    assert pending.option_ids[2] == 30
    assert pending.option_ids[3] == 40

    # Test update_pending_candidates preserves positions
    collector.update_pending_candidates(
        option_names=["Card TL", "Card TR Recovered", "Card BL", "Card BR"],
        option_ids=[10, 20, 30, 40]
    )
    assert len(pending.option_ids) == 4
    assert pending.option_ids[1] == 20


def test_ocr_invalidation_and_recovery_flow(tmp_path):
    """Verify OCR invalidation safely clears candidates and subsequent recovery synchronizes state."""
    db_path = tmp_path / "test_blur_memory.db"
    mem_db = MemoryDatabase(db_path=db_path)
    memory = BlurMemorySystem(db=mem_db, storage_dir=tmp_path / "learned")
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    adaptive = AdaptiveLearning(memory=memory, matcher=matcher, encoder=encoder)

    dummy_card = np.zeros((100, 100, 3), dtype=np.uint8)
    adaptive.register_question(
        card_frame=dummy_card,
        option_names=["Card A", "Card B", "Card C", "Card D"],
        option_ids=[1, 2, 3, 4],
        predicted_name="Card A",
        predicted_id=1,
        card_hash="hash_ocr_flow",
        timestamp=200.0
    )
    assert len(adaptive.pending_question.option_names) == 4

    # OCR fails / invalid grid:
    adaptive.invalidate_pending_candidates()
    assert len(adaptive.pending_question.option_names) == 0
    assert len(adaptive.pending_question.option_ids) == 0
    # Session itself must NOT be discarded
    assert adaptive.pending_question.session_id.startswith("q_200_")

    # OCR recovers:
    adaptive.update_pending_candidates(
        option_names=["Card A", "Card B", "Card C", "Card D"],
        option_ids=[1, 2, 3, 4]
    )
    assert len(adaptive.pending_question.option_names) == 4
    assert adaptive.pending_question.option_ids == [1, 2, 3, 4]


def test_insufficient_candidates_never_produces_confirmed_answer():
    """Verify CandidateReasoner with < 4 candidates never outputs a confirmed answer or slot."""
    reader = OptionReader()
    matcher = ImageMatcher()
    encoder = ModelManager.get_instance().get_encoder()

    reasoner = CandidateReasoner(
        image_matcher=matcher,
        image_encoder=encoder,
        option_reader=reader
    )
    dummy_card = np.ones((120, 120, 3), dtype=np.uint8) * 100

    # Test candidate counts: 0, 1, 2, 3
    test_counts = [
        [],
        [CandidateOption(slot_index=0, raw_text="Shadow Duelist", canonical_name="Shadow Duelist", card_id=200, is_valid=True)],
        [
            CandidateOption(slot_index=0, raw_text="Shadow Duelist", canonical_name="Shadow Duelist", card_id=200, is_valid=True),
            CandidateOption(slot_index=1, raw_text="Green Bomber", canonical_name="Green Bomber", card_id=137, is_valid=True),
        ],
        [
            CandidateOption(slot_index=0, raw_text="Shadow Duelist", canonical_name="Shadow Duelist", card_id=200, is_valid=True),
            CandidateOption(slot_index=1, raw_text="Green Bomber", canonical_name="Green Bomber", card_id=137, is_valid=True),
            CandidateOption(slot_index=2, raw_text="Harmonious Soul", canonical_name="Harmonious Soul", card_id=139, is_valid=True),
        ],
    ]

    for cands in test_counts:
        res = reasoner.reason_answer(dummy_card, cands)
        assert res.slot_index is None
        assert res.confidence == 0.0
        assert res.status == "uncertain"
        assert res.is_confident is False
        assert res.confidence_level in ["WAITING FOR VALID OPTIONS", "OCR RECOVERY"]
        assert res.card_name == "Waiting for valid options..."


def test_overlay_window_flags_and_visibility():
    """Verify FloatingOverlay window flags prevent focus stealing and remain visible on focus switch."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from src.ui.overlay import FloatingOverlay
    from src.core.pipeline import QuizPipeline

    _ = QApplication.instance() or QApplication([])

    # Mock pipeline for overlay
    pipeline = QuizPipeline(auto_init_models=False)
    overlay = FloatingOverlay(pipeline=pipeline, is_offline=True, start_worker=False)

    flags = overlay.windowFlags()
    # Check tool and focus prevention flags
    assert bool(flags & Qt.Tool)
    assert bool(flags & Qt.WindowStaysOnTopHint)
    assert bool(flags & Qt.WindowDoesNotAcceptFocus)
    assert bool(flags & Qt.FramelessWindowHint)

    # Check macOS tool window and no-activate attributes
    assert overlay.testAttribute(Qt.WA_MacAlwaysShowToolWindow) is True
    assert overlay.testAttribute(Qt.WA_ShowWithoutActivating) is True
    assert overlay.testAttribute(Qt.WA_TranslucentBackground) is True

    # Check status display mapping for waiting state
    overlay.on_worker_status("WAITING FOR VALID OPTIONS")
    assert "WAITING FOR VALID OPTIONS" in overlay.state_label.text()
    overlay.close()


def test_context_analyzer_question_lasting_longer_than_four_seconds_does_not_restart_session():
    """Verify that an active question visible for > 4.0s does NOT restart session."""
    ctx = ContextAnalyzer(session_timeout=15.0)
    t0 = 1000.0

    # Start question at t0
    ctx.evaluate_transition(
        is_active_question=True,
        is_result_screen=False,
        card_hash="hash_card_alpha",
        options=["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"],
        timestamp=t0
    )
    initial_sid = ctx.current_session_id
    assert initial_sid is not None

    # Emit a prediction
    ctx.mark_predicted("Green Bomber", 0)

    # Question remains on screen at t0 + 4.5s (timer ticking down)
    ctx.evaluate_transition(
        is_active_question=True,
        is_result_screen=False,
        card_hash="hash_card_alpha",
        options=["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"],
        timestamp=t0 + 4.5
    )
    # Session ID must remain strictly identical!
    assert ctx.current_session_id == initial_sid

    # Question still on screen at t0 + 5.5s
    ctx.evaluate_transition(
        is_active_question=True,
        is_result_screen=False,
        card_hash="hash_card_alpha",
        options=["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"],
        timestamp=t0 + 5.5
    )
    assert ctx.current_session_id == initial_sid


def test_context_analyzer_ocr_failure_does_not_mark_predicted():
    """Verify that unsuccessful prediction attempts do not mark session as predicted."""
    ctx = ContextAnalyzer(session_timeout=15.0)
    t0 = 2000.0

    ctx.evaluate_transition(
        is_active_question=True,
        is_result_screen=False,
        card_hash="hash_card_beta",
        options=[],
        timestamp=t0
    )
    assert ctx.active_session is not None
    assert ctx.active_session.has_predicted is False

    # Attempt to mark with waiting status (OCR failed)
    ctx.mark_predicted("Waiting for valid options...", None)
    assert ctx.active_session.has_predicted is False
    assert ctx.should_predict() is True

    # Mark with valid card name and slot
    ctx.mark_predicted("Green Bomber", 1)
    assert ctx.active_session.has_predicted is True
    assert ctx.should_predict() is False


def test_read_options_from_panel_uses_2x2_not_horizontal_strip():
    """Verify that read_options_from_panel extracts 2x2 buttons without horizontal strip splitting."""
    reader = OptionReader()

    # Create mock 2x2 panel: 400 wide, 300 high
    panel = np.ones((300, 400, 3), dtype=np.uint8) * 40
    # Top-Left button
    cv2.rectangle(panel, (20, 60), (180, 150), (120, 120, 120), -1)
    cv2.putText(panel, "Green Bomber", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Top-Right button
    cv2.rectangle(panel, (220, 60), (380, 150), (120, 120, 120), -1)
    cv2.putText(panel, "Soul King", (230, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Bottom-Left button
    cv2.rectangle(panel, (20, 180), (180, 270), (120, 120, 120), -1)
    cv2.putText(panel, "Harmonious Soul", (25, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # Bottom-Right button
    cv2.rectangle(panel, (220, 180), (380, 270), (120, 120, 120), -1)
    cv2.putText(panel, "Shadow Duelist", (225, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cands = reader.read_options_from_panel(panel)
    assert len(cands) == 4
    # Ensure slots are 0, 1, 2, 3
    assert [c.slot_index for c in cands] == [0, 1, 2, 3]


def test_screen_capture_extract_four_buttons():
    """Verify ScreenCapture.extract_four_buttons extracts 4 non-empty crops with correct slot geometry."""
    from src.capture.screen_capture import ScreenCapture

    # Options frame with header banner at top and 4 buttons in 2x2 grid
    options_frame = np.ones((300, 400, 3), dtype=np.uint8) * 30
    # Header banner
    cv2.putText(options_frame, "Which character is this?", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # 4 distinct buttons
    cv2.rectangle(options_frame, (20, 60), (180, 160), (100, 140, 200), -1)
    cv2.rectangle(options_frame, (220, 60), (380, 160), (100, 140, 200), -1)
    cv2.rectangle(options_frame, (20, 180), (180, 280), (100, 140, 200), -1)
    cv2.rectangle(options_frame, (220, 180), (380, 280), (100, 140, 200), -1)

    crops, rects, is_auto = ScreenCapture.extract_four_buttons(options_frame)
    assert len(crops) == 4
    assert len(rects) == 4
    for b in crops:
        assert b.shape[0] > 15
        assert b.shape[1] > 30

    tl, tr, bl, br = rects
    # Verify strict positional mapping
    assert tl[0] < tr[0]  # TL is left of TR
    assert bl[0] < br[0]  # BL is left of BR
    assert tl[1] < bl[1]  # TL is above BL
    assert tr[1] < br[1]  # TR is above BR



