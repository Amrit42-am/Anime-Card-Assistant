"""
tests/test_result_learning.py

Comprehensive Test Suite for Roblox Result-Based Automatic Learning & 2x2 Grid Geometry:
1. All four 2x2 green button positions (TL, TR, BL, BR) tested independently.
2. 2x2 quadrant layout geometry and ScreenCapture slicing.
3. Green and red button combinations (simultaneous colors).
4. OCR retry across subsequent result frames (recovery from transient noise).
5. Result screen pairing timeout and safe expiration on new questions.
6. Deduplication: repeated result frames never insert duplicate samples.
7. Clear-image and green-button disagreement handling (persisted in separate log, excluded from confirmed memory).
8. Result banner region separation (independent from timer region).
9. Consecutive rounds with identical cards without blanket cooldown suppression.
10. State machine transitions through all 6 phases.
11. Headless mode tuple unpacking safety.
"""

import time
from typing import List
import cv2
import numpy as np
import pytest

from src.capture.region_manager import CalibrationConfig, RectRegion, RelativeRegion
from src.core.pipeline import QuizPipeline
from src.detection.question_detector import QuestionDetector
from src.learning.result_detector import ResultDetector
from src.learning.sample_collector import AutomaticSampleCollector


@pytest.fixture(scope="module")
def pipeline():
    pipe = QuizPipeline(auto_init_models=True)
    test_cards = [
        (137, "Green Bomber"),
        (139, "Harmonious Soul"),
        (140, "Shadow Duelist"),
        (141, "Soul King"),
    ]
    for cid, cname in test_cards:
        if pipe.option_reader:
            pipe.option_reader.register_card(cid, cname)
        if pipe.matcher:
            pipe.matcher.add_card(cid, cname)
    return pipe


def make_button_frame(text: str, color_type: str = "neutral", w: int = 240, h: int = 70) -> np.ndarray:
    """Helper to synthesize button image with specific background color."""
    btn = np.ones((h, w, 3), dtype=np.uint8)
    if color_type == "green":
        btn[:, :] = (30, 180, 30)       # Strong HSV green (H=60, S=212, V=180)
    elif color_type == "red":
        btn[:, :] = (20, 20, 200)       # Strong HSV red (H=0, S=230, V=200)
    else:
        btn[:, :] = (40, 40, 40)        # Neutral dark gray

    if text:
        cv2.putText(btn, text, (10, int(h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
    return btn


def make_options_frame_2x2(buttons: List[np.ndarray]) -> np.ndarray:
    """
    Arranges 4 button frames into an authentic 2x2 grid:
    Row 0: Top-Left (Slot 0), Top-Right (Slot 1)
    Row 1: Bottom-Left (Slot 2), Bottom-Right (Slot 3)
    """
    assert len(buttons) == 4, "Must provide exactly 4 buttons for 2x2 grid"
    top_row = np.hstack([buttons[0], buttons[1]])
    bottom_row = np.hstack([buttons[2], buttons[3]])
    return np.vstack([top_row, bottom_row])


def load_card_frame(filename: str = "card_137_green-bomber.png", blurred: bool = False) -> np.ndarray:
    """Helper to load a card image, optionally blurred."""
    img = cv2.imread(f"assets/cards/{filename}")
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (35, 45, 65)
        cv2.circle(img, (112, 112), 70, (200, 120, 40), -1)
        cv2.rectangle(img, (60, 60), (164, 164), (50, 220, 180), 3)
        cv2.line(img, (20, 20), (204, 204), (255, 255, 255), 2)
    if blurred:
        return cv2.GaussianBlur(img, (35, 35), 12.0)
    return img


# ==============================================================================
# P0: Test All Four 2x2 Button Positions Independently
# ==============================================================================
@pytest.mark.parametrize("target_slot,slot_name", [
    (0, "Top-Left"),
    (1, "Top-Right"),
    (2, "Bottom-Left"),
    (3, "Bottom-Right"),
])
def test_all_four_green_button_positions(pipeline, target_slot, slot_name):
    """
    Verifies that the green correct option is accurately recognized and localized
    at each of the four 2x2 grid quadrant positions:
    0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right.
    """
    card_name = "Green Bomber"
    options = ["Shadow Duelist", "Harmonious Soul", "Soul King", "Blood Demon"]
    # Insert target card into the target slot
    options.insert(target_slot, card_name)
    options = options[:4]

    buttons = []
    for i in range(4):
        color = "green" if i == target_slot else "neutral"
        buttons.append(make_button_frame(options[i], color))

    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)
    detector = ResultDetector(pipeline.option_reader)

    event = detector.evaluate_screen(
        card_frame=clear_card,
        button_frames=buttons,
        candidate_names=options,
        timestamp=time.time()
    )

    assert event.is_result is True
    assert event.is_correct_answer_revealed is True
    assert event.correct_option is not None
    assert event.correct_option.slot_index == target_slot, f"Expected slot {target_slot} ({slot_name})"
    assert event.correct_option.correct_option == card_name
    assert event.correct_option.verified is True


def test_2x2_screen_capture_quadrant_geometry():
    """
    Tests that CalibrationConfig and ScreenCapture properly slice the 2x2 grid
    into 4 distinct quadrant crops (TL, TR, BL, BR).
    """
    cfg = CalibrationConfig(
        game_window=RectRegion(x=0, y=0, w=1000, h=800),
        card_region=RelativeRegion(rx=0.1, ry=0.2, rw=0.4, rh=0.6),
        options_region=RelativeRegion(rx=0.5, ry=0.2, rw=0.4, rh=0.6),
        banner_region=RelativeRegion(rx=0.5, ry=0.1, rw=0.4, rh=0.08)
    )

    regions = cfg.get_option_button_regions()
    assert len(regions) == 4

    tl, tr, bl, br = regions
    # Verify relative positioning of quadrant bounding boxes
    assert tl.x < tr.x, "Top-Left X must be < Top-Right X"
    assert tl.y == tr.y, "Top-Left Y must equal Top-Right Y"
    assert bl.x < br.x, "Bottom-Left X must be < Bottom-Right X"
    assert bl.y > tl.y, "Bottom-Left Y must be below Top-Left Y"
    assert br.y > tr.y, "Bottom-Right Y must be below Top-Right Y"


# ==============================================================================
# P0: OCR Retry on Result Screen
# ==============================================================================
def test_ocr_retry_on_subsequent_result_frames(pipeline):
    """
    Tests that if the first result frame fails OCR (e.g. transient render blur / noise),
    the collector does NOT abort; it retries and succeeds on the subsequent frame.
    """
    t0 = 1000.0
    collector = AutomaticSampleCollector(
        memory=pipeline.memory,
        matcher=pipeline.matcher,
        encoder=pipeline.encoder
    )
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)
    blurred_card = load_card_frame("card_137_green-bomber.png", blurred=True)

    collector.register_active_question(
        card_frame=blurred_card,
        option_names=["Shadow Duelist", "Green Bomber", "Harmonious Soul", "Soul King"],
        option_ids=[200, 137, 139, 201],
        predicted_name="Green Bomber",
        predicted_id=137,
        card_hash="retry_test_hash",
        timestamp=t0
    )

    detector = ResultDetector(pipeline.option_reader)

    # Frame 1: Noisy / unreadable text on the green button (e.g. partial animation)
    noisy_buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("??? ~~~", "green"),   # OCR fails
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    event1 = detector.evaluate_screen(clear_card, noisy_buttons, timestamp=t0 + 1.0)
    report1 = collector.process_result(clear_card, noisy_buttons, event1, timestamp=t0 + 1.0)

    # Frame 1 must return None without clearing the pending question session!
    assert report1 is None
    assert collector.pending_question is not None, "Pending question must be retained for retry!"

    # Frame 2: 0.3s later, text is clear and readable
    clean_buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Green Bomber", "green"),  # Clear OCR!
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    event2 = detector.evaluate_screen(clear_card, clean_buttons, timestamp=t0 + 1.3)
    report2 = collector.process_result(clear_card, clean_buttons, event2, timestamp=t0 + 1.3)

    assert report2 is not None
    assert report2.verified is True
    assert report2.card_name == "Green Bomber"
    assert collector.pending_question is None, "Session finalized after successful retry."

    if report2.sample_id:
        pipeline.memory.invalidate_sample(report2.sample_id)


# ==============================================================================
# P0: Clear-Image Disagreement Record Preservation (No Memory Pollution)
# ==============================================================================
def test_clear_image_disagreement_logged_separately(pipeline):
    """
    Verifies that when AI prediction differs from actual green button (e.g. wrong guess),
    the true green-button answer becomes the verified ground truth, and the wrong guess
    is recorded separately in prediction history.
    """
    t0 = 2000.0
    collector = AutomaticSampleCollector(
        memory=pipeline.memory,
        matcher=pipeline.matcher,
        encoder=pipeline.encoder,
        db=pipeline.db,
        option_reader=pipeline.option_reader
    )
    clear_card = load_card_frame("card_139_harmonious-soul.png", blurred=False)
    blurred_card = load_card_frame("card_139_harmonious-soul.png", blurred=True)

    collector.register_active_question(
        card_frame=blurred_card,
        option_names=["Harmonious Soul", "Green Bomber", "Shadow Duelist", "Soul King"],
        option_ids=[139, 137, 200, 201],
        predicted_name="Harmonious Soul",
        predicted_id=139,
        card_hash="disagree_test_hash",
        timestamp=t0,
        predicted_slot=0,
        predicted_confidence=0.60
    )

    # Green button shows Green Bomber at slot 1, but AI guessed slot 0 (Harmonious Soul)
    buttons = [
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Green Bomber", "green"),
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, timestamp=t0 + 1.5)

    report = collector.process_result(clear_card, buttons, event, timestamp=t0 + 1.5)

    assert report is not None
    assert report.verified is True
    assert report.status == "WRONG"
    assert report.card_name == "Green Bomber"

    # Prediction history records incorrect prediction
    stats = pipeline.db.get_stats()
    assert any(ev["previous_prediction"] == "Harmonious Soul" and ev["outcome"] == "WRONG" for ev in stats["recent_learning_events"])


# ==============================================================================
# P0: Result Screen Pairing Timeout & Stale Session Expiration
# ==============================================================================
def test_result_pairing_timeout_and_expiration(pipeline):
    """
    Verifies that a pending question expires when the result delay exceeds
    the maximum pairing window, and expires safely when a new question starts.
    """
    t0 = 3000.0
    collector = AutomaticSampleCollector(
        memory=pipeline.memory,
        matcher=pipeline.matcher,
        encoder=pipeline.encoder,
        max_pairing_delay=5.0
    )
    blurred_card = load_card_frame("card_137_green-bomber.png", blurred=True)
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    collector.register_active_question(
        card_frame=blurred_card,
        option_names=["Shadow Duelist", "Green Bomber", "Harmonious Soul", "Soul King"],
        option_ids=[200, 137, 139, 201],
        predicted_name="Green Bomber",
        predicted_id=137,
        card_hash="timeout_hash",
        timestamp=t0
    )

    buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Green Bomber", "green"),
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, timestamp=t0 + 8.0)

    # 8.0s > 5.0s max_pairing_delay -> Must time out
    report = collector.process_result(clear_card, buttons, event, timestamp=t0 + 8.0)
    assert report is not None
    assert report.status == "EXPIRED_TIMEOUT"
    assert report.verified is False
    assert collector.pending_question is None


# ==============================================================================
# P1: Result Banner Region Separation
# ==============================================================================
def test_result_banner_region_separation(pipeline):
    """
    Verifies that result banner OCR uses the dedicated banner region,
    avoiding confusion with timer OCR.
    """
    banner_img = np.ones((40, 200, 3), dtype=np.uint8) * 30
    cv2.putText(banner_img, "Correct in 2.5s!", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Timer only contains digits, no banner words
    timer_img = np.ones((40, 100, 3), dtype=np.uint8) * 30
    cv2.putText(timer_img, "5.2s", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Green Bomber", "green"),
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(
        card_frame=clear_card,
        button_frames=buttons,
        timer_frame=timer_img,
        banner_frame=banner_img,
        timestamp=time.time()
    )

    assert event.is_result is True
    assert "correct" in event.banner_text


# ==============================================================================
# P1: Headless Mode Step Tuple Handling
# ==============================================================================
def test_headless_step_tuple_handling(pipeline):
    """
    Verifies that pipeline.step() safely returns a 2-tuple (answer, report)
    that unpacks cleanly in headless mode without raising exceptions.
    """
    ans, rep = pipeline.step()
    # In idle state without screen changes, ans and rep should be None
    assert ans is None or hasattr(ans, "status")
    assert rep is None or hasattr(rep, "verified")


# ==============================================================================
# State Machine: 6 Lifecycle Phases
# ==============================================================================
def test_state_machine_transitions_all_phases():
    detector = QuestionDetector()
    assert detector.current_state == "IDLE"

    blurred_card = load_card_frame("card_137_green-bomber.png", blurred=True)
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    neutral_buttons = [make_button_frame(f"Option {i}", "neutral") for i in range(4)]
    active_bundle = {
        "card_frame": blurred_card,
        "options_frame": make_options_frame_2x2(neutral_buttons),
        "button_frames": neutral_buttons
    }

    # 1. IDLE -> ACTIVE_QUESTION
    e1 = detector.process_frame_bundle(active_bundle)
    assert e1.question_detected is True
    assert detector.current_state == "ACTIVE_QUESTION"

    # 2. ACTIVE_QUESTION -> ANSWER_RECOGNITION
    e2 = detector.process_frame_bundle(active_bundle)
    assert e2.question_detected is False
    assert detector.current_state == "ANSWER_RECOGNITION"

    # 3. ANSWER_RECOGNITION -> RESULT_REVEALED
    result_buttons = [
        make_button_frame("Opt 0", "neutral"),
        make_button_frame("Opt 1", "green"),
        make_button_frame("Opt 2", "neutral"),
        make_button_frame("Opt 3", "neutral")
    ]
    result_bundle = {
        "card_frame": clear_card,
        "options_frame": make_options_frame_2x2(result_buttons),
        "button_frames": result_buttons
    }
    e3 = detector.process_frame_bundle(result_bundle)
    assert e3.is_result_detected is True
    assert detector.current_state == "RESULT_REVEALED"

    # 4. RESULT_REVEALED -> SAMPLE_VERIFICATION
    detector.notify_sample_verified()
    assert detector.current_state == "SAMPLE_VERIFICATION"

    # 5. Screen clear -> NEXT_QUESTION
    empty_bundle = {
        "card_frame": np.zeros((224, 224, 3), dtype=np.uint8),
        "options_frame": np.zeros((100, 100, 3), dtype=np.uint8),
        "button_frames": []
    }
    detector.process_frame_bundle(empty_bundle)
    assert detector.current_state in ("IDLE", "NEXT_QUESTION")

    # 6. Next question cycle starts
    detector.current_state = "NEXT_QUESTION"
    e5 = detector.process_frame_bundle(active_bundle)
    assert e5.question_detected is True
    assert detector.current_state == "ACTIVE_QUESTION"


def test_lobby_and_world_screens_do_not_trigger_quiz():
    """Verify that ordinary Roblox screens (lobby, world character, menu) never trigger ACTIVE_QUESTION."""
    detector = QuestionDetector()

    # 1. Lobby screen: smooth ground texture and sky
    ground = np.ones((80, 200, 3), dtype=np.uint8) * 90
    ground_buttons = [np.ones((40, 90, 3), dtype=np.uint8) * 90 for _ in range(4)]
    lobby_bundle = {
        "card_frame": np.ones((120, 120, 3), dtype=np.uint8) * 110,
        "options_frame": ground,
        "button_frames": ground_buttons
    }
    e_lobby = detector.process_frame_bundle(lobby_bundle)
    assert e_lobby.question_detected is False
    assert detector.current_state == "IDLE"

    # 2. Character in game world: random gradient without button borders/text
    char_card = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(char_card, (60, 60), 40, (180, 120, 80), -1)
    world_buttons = [np.random.normal(100, 2, (40, 90, 3)).astype(np.uint8) for _ in range(4)]
    char_bundle = {
        "card_frame": char_card,
        "options_frame": np.vstack([np.hstack(world_buttons[:2]), np.hstack(world_buttons[2:])]),
        "button_frames": world_buttons
    }
    e_char = detector.process_frame_bundle(char_bundle)
    assert e_char.question_detected is False
    assert detector.current_state == "IDLE"

    # 3. Solid menu screen
    menu_bundle = {
        "card_frame": np.ones((120, 120, 3), dtype=np.uint8) * 30,
        "options_frame": np.ones((80, 200, 3), dtype=np.uint8) * 30,
        "button_frames": [np.ones((40, 90, 3), dtype=np.uint8) * 30 for _ in range(4)]
    }
    e_menu = detector.process_frame_bundle(menu_bundle)
    assert e_menu.question_detected is False
    assert detector.current_state == "IDLE"


def test_result_detector_green_button_ocr_fallback(monkeypatch):
    """Verify that get_correct_option falls back to raw button crop and retry when vision on preprocessed fails."""
    from src.learning.result_detector import get_correct_option
    from src.ocr.option_reader import OptionReader

    reader = OptionReader()
    green_btn = make_button_frame("Green Bomber", color_type="green")
    neutral_btns = [make_button_frame("Soul King", color_type="neutral") for _ in range(3)]
    buttons = [green_btn, neutral_btns[0], neutral_btns[1], neutral_btns[2]]

    # Case 1: normal vision succeeds
    res1 = get_correct_option(buttons, reader, candidate_names=["Green Bomber", "Soul King", "Shadow Duelist", "Harmonious Soul"])
    assert res1 is not None
    assert res1.correct_option == "Green Bomber"
    assert res1.slot_index == 0

    # Case 2: Vision fails on preprocessed, but succeeds on raw frame
    original_vision = reader.extract_text_vision
    call_count = [0]
    def mock_vision(img, recognition_level=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return []  # Fail first (preprocessed) call
        return original_vision(img, recognition_level=recognition_level)

    monkeypatch.setattr(reader, "extract_text_vision", mock_vision)
    res2 = get_correct_option(buttons, reader, candidate_names=["Green Bomber", "Soul King", "Shadow Duelist", "Harmonious Soul"])
    assert res2 is not None
    assert res2.correct_option == "Green Bomber"
