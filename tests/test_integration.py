"""
tests/test_integration.py

End-to-End Integration Tests for Anime Card Assistant:
- Simulates quiz screen capture bundle
- Tests QuestionDetector -> OptionReader -> DecisionEngine -> BlurMemory
- Verifies state transitions and duplicate question suppression
- Verifies manual correction and confirmation persistence
"""

import cv2
import numpy as np
import pytest

from src.core.pipeline import QuizPipeline
from src.core.decision_engine import AnswerResult
from src.ai.context_analyzer import QuizState
from src.ai.model_manager import ModelManager


def make_mock_card_image() -> np.ndarray:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = (35, 45, 65)
    cv2.circle(img, (112, 112), 70, (200, 120, 40), -1)
    cv2.rectangle(img, (60, 60), (164, 164), (50, 220, 180), 3)
    cv2.line(img, (20, 20), (204, 204), (255, 255, 255), 2)
    return img


@pytest.fixture(scope="module")
def pipeline():
    ModelManager.reset_instance()
    p = QuizPipeline(auto_init_models=True)
    card_img = make_mock_card_image()
    emb = p.encoder.encode_image(card_img)
    test_cards = [
        (137, "Green Bomber", emb),
        (139, "Harmonious Soul", np.zeros(384, dtype=np.float32)),
        (140, "Shadow Duelist", np.zeros(384, dtype=np.float32)),
        (141, "Soul King", np.zeros(384, dtype=np.float32)),
        (150, "Celestial Warrior", np.zeros(384, dtype=np.float32)),
    ]
    for cid, cname, c_emb in test_cards:
        if p.option_reader:
            p.option_reader.register_card(cid, cname)
        if p.matcher:
            p.matcher.add_card(cid, cname, embedding=c_emb)
    yield p
    ModelManager.reset_instance()


def create_mock_quiz_frame(card_name: str, options: list[str], blur_k: int = 35) -> dict:
    """Creates a mock game frame bundle with blurred card artwork and 4 2x2 option buttons."""
    # 1. Base card image: patterned synthetic card artwork
    real_img = make_mock_card_image()

    # Apply blur
    k = max(3, blur_k if blur_k % 2 == 1 else blur_k + 1)
    blurred_card = cv2.GaussianBlur(real_img, (k, k), float(k) / 3.0)

    # 2. 2x2 Options frame with 4 buttons: TL, TR, BL, BR
    btn_frames = []
    btn_h = 60
    btn_w = 200

    for i, name in enumerate(options):
        btn = np.ones((btn_h, btn_w, 3), dtype=np.uint8) * 40
        cv2.putText(btn, name, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        btn_frames.append(btn)

    top_row = np.hstack([btn_frames[0], btn_frames[1]])
    bottom_row = np.hstack([btn_frames[2], btn_frames[3]])
    options_frame = np.vstack([top_row, bottom_row])

    # Game frame mock
    game_frame = np.ones((500, 700, 3), dtype=np.uint8) * 30

    return {
        "game_frame": game_frame,
        "card_frame": blurred_card,
        "options_frame": options_frame,
        "button_frames": btn_frames,
    }


def test_pipeline_decision_engine(pipeline):
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    mock_bundle = create_mock_quiz_frame(card_name="Green Bomber", options=options)

    result = pipeline.decision_engine.process_question(
        card_frame=mock_bundle["card_frame"],
        button_frames=mock_bundle["button_frames"],
        options_frame=mock_bundle["options_frame"]
    )

    assert isinstance(result, AnswerResult)
    assert result.status == "matched"
    assert result.card_name == "Green Bomber"
    assert result.confidence > 0.7

    # Warm inference run
    result_warm = pipeline.decision_engine.process_question(
        card_frame=mock_bundle["card_frame"],
        button_frames=mock_bundle["button_frames"],
        options_frame=mock_bundle["options_frame"]
    )
    assert result_warm.processing_time_ms < 500.0


def test_detector_duplicate_prevention(pipeline):
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle1 = create_mock_quiz_frame("Green Bomber", options)

    # First presentation
    event1 = pipeline.detector.process_frame_bundle(bundle1)
    assert event1.question_detected is True

    # Immediate next frame with same question
    event2 = pipeline.detector.process_frame_bundle(bundle1)
    assert event2.question_detected is False, "Repeated identical frame must not trigger another question event!"


def test_pipeline_confirm_and_learn(pipeline):
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle = create_mock_quiz_frame("Green Bomber", options)

    pipeline.last_result = pipeline.decision_engine.process_question(
        card_frame=bundle["card_frame"],
        button_frames=bundle["button_frames"]
    )

    initial_memory_count = pipeline.memory.total_samples
    sid = pipeline.confirm_last_answer(notes="Integration test confirmation")
    assert sid is not None
    assert pipeline.memory.total_samples == initial_memory_count + 1

    # Cleanup test sample so it does not pollute real memory
    pipeline.memory.invalidate_sample(sid)


def test_pipeline_step_lifecycle(pipeline):
    """
    Tests full pipeline.step() lifecycle:
    Active Question -> Answer Prediction (<100ms) -> Result Screen -> Auto-Learning
    """
    import time
    options = ["Shadow Duelist", "Green Bomber", "Harmonious Soul", "Soul King"]

    # 1. Active Question Screen
    active_bundle = create_mock_quiz_frame("Green Bomber", options)
    # Monkey-patch pipeline.capture.capture_quiz_bundle to return active_bundle
    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: active_bundle

    pipeline.detector.reset()

    # Step 1: Active question frame (Provisional answer)
    answer, report = pipeline.step()
    assert answer is not None
    assert answer.status == "matched"
    assert answer.card_name == "Green Bomber"
    assert answer.prediction_stage == "PROVISIONAL"
    assert report is None

    # Step 2: Next frame during active question (Multi-frame temporal stabilization with evolving frame)
    active_bundle_frame2 = create_mock_quiz_frame("Green Bomber", options, blur_k=25)
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: active_bundle_frame2
    answer2, report2 = pipeline.step()
    assert answer2 is not None
    assert answer2.card_name == "Green Bomber"
    assert answer2.prediction_stage in ("STABILIZING", "CONFIDENT")
    assert answer2.observation_count >= 2
    assert report2 is None

    # 2. Result Revealed Screen (Green Bomber button is green)
    real_card = make_mock_card_image()
    result_buttons = []
    opt_w = 260
    for i, name in enumerate(options):
        btn = np.ones((56, opt_w - 8, 3), dtype=np.uint8)
        if name == "Green Bomber":
            btn[:, :] = (30, 180, 30)  # Green
        else:
            btn[:, :] = (40, 40, 40)   # Neutral
        cv2.putText(btn, name, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        result_buttons.append(btn)

    result_bundle = {
        "game_frame": active_bundle["game_frame"],
        "card_frame": real_card,
        "options_frame": active_bundle["options_frame"],
        "button_frames": result_buttons,
        "timer_frame": None,
    }

    # Simulate 0.5s delay for game result reveal
    time.sleep(0.5)
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: result_bundle

    # Step 3: Result frame
    answer3, report3 = pipeline.step()
    assert answer3 is None
    assert report3 is not None
    assert report3.verified is True
    assert report3.card_name == "Green Bomber"

    # Cleanup
    pipeline.capture.capture_quiz_bundle = orig_capture
    if report3.sample_id:
        pipeline.memory.invalidate_sample(report3.sample_id)


def test_pipeline_advanced_ai_modules_wired(pipeline, monkeypatch):
    """
    Verifies that all Advanced Autonomous AI modules are instantiated,
    properly wired, and actively invoked during pipeline processing.
    """
    # 1. Check all modules exist on the pipeline
    assert hasattr(pipeline, "perception_engine")
    assert hasattr(pipeline, "candidate_reasoner")
    assert hasattr(pipeline, "temporal_analyzer")
    assert hasattr(pipeline, "context_analyzer")
    assert hasattr(pipeline, "adaptive_learning")
    assert hasattr(pipeline, "game_locator")
    assert hasattr(pipeline, "confidence_engine")
    assert hasattr(pipeline, "model_manager")

    # 2. Track calls to candidate_reasoner and temporal_analyzer
    called_tracker = {"reasoner": 0, "temporal": 0}

    orig_reason = pipeline.candidate_reasoner.reason_answer
    def spy_reason(*args, **kwargs):
        called_tracker["reasoner"] += 1
        return orig_reason(*args, **kwargs)

    orig_push = pipeline.temporal_analyzer.push
    def spy_push(*args, **kwargs):
        called_tracker["temporal"] += 1
        return orig_push(*args, **kwargs)

    monkeypatch.setattr(pipeline.candidate_reasoner, "reason_answer", spy_reason)
    monkeypatch.setattr(pipeline.temporal_analyzer, "push", spy_push)

    # Trigger a question step
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle = create_mock_quiz_frame("Green Bomber", options)
    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle
    pipeline.detector.reset()

    try:
        answer, _ = pipeline.step()
        assert answer is not None
        assert called_tracker["reasoner"] >= 1
        assert called_tracker["temporal"] >= 1
    finally:
        pipeline.capture.capture_quiz_bundle = orig_capture


def test_perception_engine_eliminates_redundant_ocr(pipeline, monkeypatch):
    """
    Verifies that PerceptionEngine is the live perception entry point and its
    precomputed OCR candidates are directly consumed by DecisionEngine without
    triggering duplicate OCR in OptionReader.
    """
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle = create_mock_quiz_frame("Green Bomber", options)
    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle
    pipeline.detector.reset()

    # Spy on OptionReader.read_options_from_buttons and read_options_from_grid
    ocr_calls = {"from_buttons": 0}
    orig_read_buttons = pipeline.option_reader.read_options_from_buttons

    def spy_read_buttons(*args, **kwargs):
        ocr_calls["from_buttons"] += 1
        return orig_read_buttons(*args, **kwargs)

    monkeypatch.setattr(pipeline.option_reader, "read_options_from_buttons", spy_read_buttons)

    # Spy on PerceptionEngine.perceive_frame
    perceive_called = {"count": 0}
    orig_perceive = pipeline.perception_engine.perceive_frame

    def spy_perceive(*args, **kwargs):
        perceive_called["count"] += 1
        return orig_perceive(*args, **kwargs)

    monkeypatch.setattr(pipeline.perception_engine, "perceive_frame", spy_perceive)

    try:
        answer, _ = pipeline.step()
        assert answer is not None
        assert perceive_called["count"] == 1, "PerceptionEngine.perceive_frame must be invoked by pipeline.step()"
        # In perception_engine, read_options_from_buttons or read_options_from_grid is called ONCE.
        # DecisionEngine must NOT invoke it again because ocr_candidates was passed!
        # If DecisionEngine had re-run OCR, ocr_calls["from_buttons"] would be >= 2.
        assert ocr_calls["from_buttons"] <= 1, "DecisionEngine must not re-run OCR when PerceptionFrame provides candidates"
    finally:
        pipeline.capture.capture_quiz_bundle = orig_capture


def test_temporal_contradictory_evidence_penalizes_confidence(pipeline):
    """
    Verifies that contradictory evidence across frames dynamically degrades confidence
    and marks prediction_stage as UNCERTAIN.
    """
    options_a = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle_a = create_mock_quiz_frame("Green Bomber", options_a)

    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.detector.reset()
    pipeline.context_analyzer.reset()
    pipeline.temporal_analyzer.reset()

    # Frame 1: Sees Green Bomber
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle_a
    ans1, _ = pipeline.step()
    assert ans1 is not None
    assert ans1.card_name == "Green Bomber"
    assert ans1.prediction_stage == "PROVISIONAL"

    # Frame 2 & 3: Artificially inject contradictory answers into temporal analyzer for the same session
    session_id = pipeline.context_analyzer.current_session_id
    from src.ai.candidate_reasoner import ReasonedAnswer
    contradictory_ans = ReasonedAnswer(
        card_name="Celestial Warrior",
        card_id=1,
        slot_index=0,
        confidence=0.85,
        margin=0.2,
        is_confident=True,
        runner_up_name="Green Bomber",
        runner_up_score=0.6,
        source="visual",
        status="matched",
        raw_match={}
    )
    pipeline.temporal_analyzer.push(contradictory_ans, session_id=session_id)
    pipeline.temporal_analyzer.push(contradictory_ans, session_id=session_id)

    # Frame 4: Run another step under the same question session with evolving visual frame
    bundle_a_frame4 = create_mock_quiz_frame("Green Bomber", options_a, blur_k=25)
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle_a_frame4
    ans_contra, _ = pipeline.step()
    assert ans_contra is not None
    # Now there are 2 Green Bomber and 2 Celestial Warrior -> agreement ratio is 0.50 (< 0.60)
    assert ans_contra.prediction_stage == "UNCERTAIN"
    assert ans_contra.confidence <= 0.40

    pipeline.capture.capture_quiz_bundle = orig_capture


def test_complete_quiz_lifecycle_state_machine(pipeline):
    """
    Tests complete lifecycle sequence:
    SEARCHING -> QUESTION_DETECTED / ANSWER_DISPLAYED -> RESULT_DETECTED / VERIFYING -> TRANSITION / SEARCHING
    """
    import time
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    active_bundle = create_mock_quiz_frame("Green Bomber", options)
    orig_capture = pipeline.capture.capture_quiz_bundle

    pipeline.detector.reset()
    pipeline.context_analyzer.reset()
    pipeline.temporal_analyzer.reset()
    assert pipeline.context_analyzer.current_state in (QuizState.SEARCHING, QuizState.TRANSITION)

    # 1. State: ACTIVE_QUESTION (Frame 1 -> PROVISIONAL)
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: active_bundle
    ans_frame1, report1 = pipeline.step()
    assert ans_frame1 is not None
    assert ans_frame1.prediction_stage == "PROVISIONAL"
    assert pipeline.context_analyzer.current_state in (
        QuizState.QUESTION_DETECTED,
        QuizState.ANALYZING,
        QuizState.ANSWER_DISPLAYED
    )
    assert report1 is None

    # 2. State: ACTIVE_QUESTION (Frame 2 -> STABILIZING with evolving unblur frame)
    active_bundle_frame2 = create_mock_quiz_frame("Green Bomber", options, blur_k=25)
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: active_bundle_frame2
    ans_frame2, report2 = pipeline.step()
    assert ans_frame2 is not None
    assert ans_frame2.prediction_stage in ("STABILIZING", "CONFIDENT")
    assert ans_frame2.observation_count >= 2
    assert pipeline.context_analyzer.current_state == QuizState.ANSWER_DISPLAYED
    assert report2 is None

    # 3. State: RESULT_REVEALED (Green button detected + clear artwork)
    real_card = make_mock_card_image()
    result_buttons = []
    for name in options:
        btn = np.ones((56, 252, 3), dtype=np.uint8)
        if name == "Green Bomber":
            btn[:, :] = (30, 180, 30)  # Green
        else:
            btn[:, :] = (40, 40, 40)
        cv2.putText(btn, name, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        result_buttons.append(btn)

    result_bundle = {
        "game_frame": active_bundle["game_frame"],
        "card_frame": real_card,
        "options_frame": active_bundle["options_frame"],
        "button_frames": result_buttons,
        "timer_frame": None,
    }
    time.sleep(0.5)
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: result_bundle

    ans_result, report_result = pipeline.step()
    assert ans_result is None
    assert report_result is not None
    assert report_result.verified is True
    assert pipeline.context_analyzer.current_state in (
        QuizState.RESULT_DETECTED,
        QuizState.VERIFYING,
        QuizState.LEARNING,
        QuizState.TRANSITION
    )

    # 4. State: TRANSITION / SEARCHING (Next frame after result has completed)
    neutral_bundle = {
        "game_frame": active_bundle["game_frame"],
        "card_frame": np.zeros((100, 100, 3), dtype=np.uint8),
        "options_frame": np.zeros((100, 100, 3), dtype=np.uint8),
        "button_frames": [np.zeros((50, 100, 3), dtype=np.uint8) for _ in range(4)],
        "timer_frame": None,
    }
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: neutral_bundle
    ans_trans, report_trans = pipeline.step()
    assert ans_trans is None
    assert report_trans is None
    assert pipeline.context_analyzer.current_state in (QuizState.TRANSITION, QuizState.SEARCHING)

    pipeline.capture.capture_quiz_bundle = orig_capture
    if report_result and report_result.sample_id:
        pipeline.memory.invalidate_sample(report_result.sample_id)


def test_perception_engine_ocr_cache_and_refresh(pipeline, monkeypatch):
    """
    Verifies that PerceptionEngine caches candidates when the options grid remains stable,
    and refreshes/re-runs OCR when the candidate options change.
    """
    options_a = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle_a = create_mock_quiz_frame("Green Bomber", options_a)

    options_b = ["Celestial Warrior", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle_b = create_mock_quiz_frame("Celestial Warrior", options_b)

    pipeline.perception_engine.reset_cache()

    ocr_call_count = {"count": 0}
    orig_read = pipeline.option_reader.read_options_from_buttons

    def counting_read(*args, **kwargs):
        ocr_call_count["count"] += 1
        return orig_read(*args, **kwargs)

    monkeypatch.setattr(pipeline.option_reader, "read_options_from_buttons", counting_read)

    # 1. Frame 1 with bundle_a: Initial OCR run
    p1 = pipeline.perception_engine.perceive_frame(bundle=bundle_a)
    assert ocr_call_count["count"] >= 1
    initial_calls = ocr_call_count["count"]

    # 2. Frame 2 with identical bundle_a: Grid is visually stable, OCR must be cached!
    p2 = pipeline.perception_engine.perceive_frame(bundle=bundle_a)
    assert ocr_call_count["count"] == initial_calls, "OCR should not re-run when options grid is visually stable"
    assert [c.canonical_name for c in p2.ocr_candidates] == [c.canonical_name for c in p1.ocr_candidates]

    # 3. Frame 3 with bundle_b: Options grid visually changed, OCR must refresh!
    p3 = pipeline.perception_engine.perceive_frame(bundle=bundle_b)
    assert ocr_call_count["count"] > initial_calls, "OCR must re-run when options grid changes"
    assert "Celestial Warrior" in [c.canonical_name for c in p3.ocr_candidates]


def test_adaptive_visual_inference_skips_unchanged_frame(pipeline, monkeypatch):
    """
    Verifies that on identical consecutive frames:
    1. DINOv2 visual inference is skipped to save GPU/CPU cycles.
    2. Temporal observation count is NOT artificially incremented.
    3. The existing provisional answer is returned smoothly for the UI overlay.
    """
    options = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle = create_mock_quiz_frame("Green Bomber", options)
    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle
    pipeline.reset()

    # Track calls to encoder.encode_image
    encode_calls = {"count": 0}
    orig_encode = pipeline.encoder.encode_image

    def spy_encode(*args, **kwargs):
        encode_calls["count"] += 1
        return orig_encode(*args, **kwargs)

    monkeypatch.setattr(pipeline.encoder, "encode_image", spy_encode)

    try:
        # Frame 1: First question detection & provisional prediction
        ans1, _ = pipeline.step()
        assert ans1 is not None
        assert ans1.card_name == "Green Bomber"
        assert ans1.observation_count == 1
        assert encode_calls["count"] >= 1
        initial_encodes = encode_calls["count"]

        # Frame 2: Identical frame bundle
        ans2, _ = pipeline.step()
        assert ans2 is not None
        assert ans2.card_name == "Green Bomber"
        # Observation count MUST NOT increase on unchanged frame
        assert ans2.observation_count == 1, "Repeated identical frames must not artificially increase temporal observation count!"
        # Encoder MUST NOT be called again
        assert encode_calls["count"] == initial_encodes, "Adaptive inference must skip DINOv2 encode on unchanged card frame!"
    finally:
        pipeline.capture.capture_quiz_bundle = orig_capture


def test_refreshed_ocr_candidates_synchronize_pipeline(pipeline):
    """
    Verifies that when PerceptionEngine signals REFRESHED candidates on grid change:
    1. Pipeline updates self.current_candidates.
    2. Pending question options and valid IDs in AdaptiveLearning are safely updated.
    3. The original blurred question image is NOT overwritten.
    """
    options_a = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle_a = create_mock_quiz_frame("Green Bomber", options_a)

    options_b = ["Green Bomber", "Harmonious Soul", "Celestial Warrior", "Soul King"]
    bundle_b = create_mock_quiz_frame("Green Bomber", options_b)

    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.reset()

    try:
        # Frame 1: Initial question with options_a
        pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle_a
        ans1, _ = pipeline.step()
        assert ans1 is not None
        assert "Celestial Warrior" not in pipeline.current_candidates
        assert pipeline.adaptive_learning.pending_question is not None
        assert "Celestial Warrior" not in pipeline.adaptive_learning.pending_question.option_names
        orig_blurred_img = pipeline.adaptive_learning.pending_question.blurred_image.copy()

        # Frame 2: Options grid changes to options_b
        pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle_b
        ans2, _ = pipeline.step()

        # Verify pipeline.current_candidates updated
        current_names = [c.canonical_name for c in pipeline.current_candidates]
        assert "Celestial Warrior" in current_names, "Pipeline must update current_candidates when grid changes!"

        # Verify pending question in adaptive learning updated options without touching blurred image
        pending_names = pipeline.adaptive_learning.pending_question.option_names
        assert "Celestial Warrior" in pending_names, "AdaptiveLearning pending question options must be updated!"
        assert np.array_equal(
            pipeline.adaptive_learning.pending_question.blurred_image,
            orig_blurred_img
        ), "Original blurred question image must never be overwritten on subsequent inference frames!"
    finally:
        pipeline.capture.capture_quiz_bundle = orig_capture


def test_invalid_ocr_grid_change_invalidates_stale_candidates(pipeline):
    """
    Verifies that when a grid change occurs but OCR fails to resolve 4 valid candidates:
    1. PerceptionFrame signals OCRSource.INVALID.
    2. Stale cached candidates are cleared and pipeline.current_candidates is emptied.
    3. Stale candidates are not incorrectly reused.
    """
    options_a = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    bundle_a = create_mock_quiz_frame("Green Bomber", options_a)

    # Frame with black/unreadable buttons
    black_buttons = [np.zeros((60, 140, 3), dtype=np.uint8) for _ in range(4)]
    top_row = np.hstack([black_buttons[0], black_buttons[1]])
    bot_row = np.hstack([black_buttons[2], black_buttons[3]])
    options_frame_invalid = np.vstack([top_row, bot_row])

    bundle_invalid = {
        "game_frame": bundle_a["game_frame"],
        "card_frame": bundle_a["card_frame"],
        "options_frame": options_frame_invalid,
        "button_frames": black_buttons,
        "timer_frame": None,
        "banner_frame": None
    }

    orig_capture = pipeline.capture.capture_quiz_bundle
    pipeline.reset()

    try:
        # Frame 1: Valid initial question
        pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle_a
        ans1, _ = pipeline.step()
        assert len(pipeline.current_candidates) == 4

        # Frame 2: Grid changed to unreadable/invalid buttons
        pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: bundle_invalid
        ans2, _ = pipeline.step()

        # Pipeline current_candidates must be cleared (not stale!)
        assert len(pipeline.current_candidates) == 0, "Stale candidates must be cleared when OCR fails on grid change!"
    finally:
        pipeline.capture.capture_quiz_bundle = orig_capture




