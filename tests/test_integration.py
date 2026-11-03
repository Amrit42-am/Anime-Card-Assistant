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

from src.capture.region_manager import CalibrationConfig
from src.core.pipeline import QuizPipeline
from src.core.decision_engine import AnswerResult


@pytest.fixture(scope="module")
def pipeline():
    return QuizPipeline(auto_init_models=True)


def create_mock_quiz_frame(card_name: str, options: list[str]) -> dict:
    """Creates a mock game frame bundle with blurred card artwork and 4 option buttons."""
    # 1. Base card image: let's load a real card artwork from assets
    card_path = "assets/cards/card_137_green-bomber.png"
    real_img = cv2.imread(card_path)
    if real_img is None:
        real_img = np.ones((224, 224, 3), dtype=np.uint8) * 120

    # Apply heavy blur
    blurred_card = cv2.GaussianBlur(real_img, (35, 35), 12.0)

    # 2. Options frame with 4 buttons
    btn_frames = []
    opt_h = 240
    opt_w = 260
    options_frame = np.ones((opt_h, opt_w, 3), dtype=np.uint8) * 20

    for i, name in enumerate(options):
        btn = np.ones((56, opt_w - 8, 3), dtype=np.uint8) * 40
        cv2.putText(btn, name, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        btn_frames.append(btn)
        options_frame[i * 60 + 2:(i + 1) * 60 - 2, 4:opt_w - 4] = btn

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

    # Step 1: Active question frame
    answer, report = pipeline.step()
    assert answer is not None
    assert answer.status == "matched"
    assert answer.card_name == "Green Bomber"
    assert report is None

    # Step 2: Next frame during active question (no duplicate prediction)
    answer2, report2 = pipeline.step()
    assert answer2 is None
    assert report2 is None

    # 2. Result Revealed Screen (Green Bomber button is green)
    real_card = cv2.imread("assets/cards/card_137_green-bomber.png")
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
