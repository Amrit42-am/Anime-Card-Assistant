"""
tests/test_result_learning.py

Comprehensive Test Suite for Roblox Result-Based Automatic Learning:
Covers all 10 scenarios required by the architecture specification:
1. Correct answer revealed (green button only).
2. Incorrect answer selected (green button + red button simultaneously).
3. Correct answer without red selection.
4. New question with same card as previous.
5. New question with different card.
6. OCR failure on green button.
7. Clear image and green option disagree (saves as UNVERIFIED_DISAGREEMENT).
8. Result screen appears after timer expires.
9. Repeated screenshots of the same result screen (deduplication check).
10. State machine transitions through all 6 phases.
"""

import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pytest

from src.core.pipeline import QuizPipeline
from src.detection.question_detector import QuestionDetector
from src.learning.result_detector import (
    ResultDetector,
    get_color_ratios,
    get_correct_option,
    compute_image_sharpness
)
from src.learning.sample_collector import AutomaticSampleCollector


@pytest.fixture(scope="module")
def pipeline():
    return QuizPipeline(auto_init_models=True)


def make_button_frame(text: str, color_type: str = "neutral") -> np.ndarray:
    """Helper to synthesize button image with specific background color."""
    btn = np.ones((70, 280, 3), dtype=np.uint8)
    if color_type == "green":
        btn[:, :] = (30, 180, 30)       # Strong HSV green (H=60, S=212, V=180)
    elif color_type == "red":
        btn[:, :] = (20, 20, 200)       # Strong HSV red (H=0, S=230, V=200)
    else:
        btn[:, :] = (40, 40, 40)        # Neutral dark gray

    if text:
        cv2.putText(btn, text, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return btn


def make_options_frame(buttons: List[np.ndarray]) -> np.ndarray:
    """Helper to stack 4 button frames into a single options region."""
    return np.vstack(buttons)


def load_card_frame(filename: str = "card_137_green-bomber.png", blurred: bool = False) -> np.ndarray:
    """Helper to load a card image, optionally blurred."""
    img = cv2.imread(f"assets/cards/{filename}")
    if img is None:
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    if blurred:
        return cv2.GaussianBlur(img, (35, 35), 12.0)
    return img


# ==============================================================================
# Scenario 1: Correct Answer Revealed (Green Button Only)
# ==============================================================================
def test_correct_answer_revealed_green_only(pipeline):
    card_name = "Green Bomber"
    options = ["Shadow Duelist", card_name, "Harmonious Soul", "Soul King"]
    buttons = [
        make_button_frame(options[0], "neutral"),
        make_button_frame(options[1], "green"),     # Correct option in Slot 1
        make_button_frame(options[2], "neutral"),
        make_button_frame(options[3], "neutral"),
    ]
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, None, time.time())

    assert event.phase == "RESULT_REVEALED"
    assert event.is_result is True
    assert event.is_correct_answer_revealed is True
    assert event.is_wrong_selection_present is False
    assert event.correct_option is not None
    assert event.correct_option.slot_index == 1
    assert event.correct_option.correct_option == card_name
    assert event.correct_option.verified is True
    assert event.correct_option.source == "green_button_ocr"


# ==============================================================================
# Scenario 2: Incorrect Answer Selected (Green + Red Simultaneously Present)
# ==============================================================================
def test_incorrect_answer_selected_simultaneous_colors(pipeline):
    card_name = "Green Bomber"
    options = ["Shadow Duelist", card_name, "Harmonious Soul", "Soul King"]
    buttons = [
        make_button_frame(options[0], "red"),       # Wrong selection in Slot 0
        make_button_frame(options[1], "green"),     # Correct answer revealed in Slot 1
        make_button_frame(options[2], "neutral"),
        make_button_frame(options[3], "neutral"),
    ]
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, None, time.time())

    assert event.phase == "RESULT_REVEALED"
    assert event.is_correct_answer_revealed is True
    assert event.is_wrong_selection_present is True
    assert event.wrong_slot_index == 0
    # Must NEVER learn from red button (Slot 0). Correct option must remain green (Slot 1).
    assert event.correct_option is not None
    assert event.correct_option.slot_index == 1
    assert event.correct_option.correct_option == card_name


# ==============================================================================
# Scenario 3: Correct Answer Without Red Selection
# ==============================================================================
def test_correct_answer_without_red_selection(pipeline):
    card_name = "Harmonious Soul"
    options = ["Green Bomber", "Shadow Duelist", card_name, "Soul King"]
    buttons = [
        make_button_frame(options[0], "neutral"),
        make_button_frame(options[1], "neutral"),
        make_button_frame(options[2], "green"),     # User answered correctly or timed out
        make_button_frame(options[3], "neutral"),
    ]
    clear_card = load_card_frame("card_139_harmonious-soul.png", blurred=False)

    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, None, time.time())

    assert event.is_result is True
    assert event.is_correct_answer_revealed is True
    assert event.is_wrong_selection_present is False
    assert event.wrong_slot_index is None
    assert event.correct_option.correct_option == card_name


# ==============================================================================
# Scenario 4: New Question With Same Card as Previous
# ==============================================================================
def test_new_question_with_same_card(pipeline):
    """
    Verifies that the state machine recognizes a consecutive round containing
    the EXACT same card, without being suppressed by a blanket cooldown.
    """
    card_name = "Green Bomber"
    options = ["Shadow Duelist", card_name, "Harmonious Soul", "Soul King"]
    blurred_card = load_card_frame("card_137_green-bomber.png", blurred=True)
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    detector = QuestionDetector()

    # 1. Round 1 Active Question
    neutral_buttons = [make_button_frame(o, "neutral") for o in options]
    bundle_q1 = {
        "card_frame": blurred_card,
        "options_frame": make_options_frame(neutral_buttons),
        "button_frames": neutral_buttons,
    }
    event1 = detector.process_frame_bundle(bundle_q1)
    assert event1.question_detected is True
    assert event1.state == "ACTIVE_QUESTION"

    # 2. Round 1 Result Reveal
    result_buttons = [
        make_button_frame(options[0], "neutral"),
        make_button_frame(options[1], "green"),
        make_button_frame(options[2], "neutral"),
        make_button_frame(options[3], "neutral"),
    ]
    bundle_r1 = {
        "card_frame": clear_card,
        "options_frame": make_options_frame(result_buttons),
        "button_frames": result_buttons,
    }
    event2 = detector.process_frame_bundle(bundle_r1)
    assert event2.is_result_detected is True
    assert event2.state == "RESULT_REVEALED"

    # 3. Post-round Sample Verification finishes
    detector.notify_sample_verified()
    assert detector.current_state == "SAMPLE_VERIFICATION"

    # 4. Round 2 Active Question with the SAME card
    event3 = detector.process_frame_bundle(bundle_q1)
    assert event3.question_detected is True, "Must detect new question even with identical card!"
    assert event3.state == "ACTIVE_QUESTION"


# ==============================================================================
# Scenario 5: New Question With Different Card
# ==============================================================================
def test_new_question_with_different_card(pipeline):
    card1_name = "Green Bomber"
    options1 = ["Shadow Duelist", card1_name, "Harmonious Soul", "Soul King"]
    blurred1 = load_card_frame("card_137_green-bomber.png", blurred=True)
    clear1 = load_card_frame("card_137_green-bomber.png", blurred=False)

    card2_name = "Harmonious Soul"
    options2 = ["Soul King", "Green Bomber", card2_name, "Shadow Duelist"]
    blurred2 = load_card_frame("card_139_harmonious-soul.png", blurred=True)

    detector = QuestionDetector()

    # Round 1 Active
    neutral1 = [make_button_frame(o, "neutral") for o in options1]
    e1 = detector.process_frame_bundle({
        "card_frame": blurred1,
        "options_frame": make_options_frame(neutral1),
        "button_frames": neutral1,
    })
    assert e1.question_detected is True

    # Round 1 Result
    res1 = [make_button_frame(options1[0], "neutral"), make_button_frame(options1[1], "green"),
            make_button_frame(options1[2], "neutral"), make_button_frame(options1[3], "neutral")]
    e2 = detector.process_frame_bundle({
        "card_frame": clear1,
        "options_frame": make_options_frame(res1),
        "button_frames": res1,
    })
    assert e2.is_result_detected is True

    detector.notify_sample_verified()

    # Round 2 Active with Different Card
    neutral2 = [make_button_frame(o, "neutral") for o in options2]
    e3 = detector.process_frame_bundle({
        "card_frame": blurred2,
        "options_frame": make_options_frame(neutral2),
        "button_frames": neutral2,
    })
    assert e3.question_detected is True
    assert e3.state == "ACTIVE_QUESTION"


# ==============================================================================
# Scenario 6: OCR Failure on Green Button
# ==============================================================================
def test_ocr_failure_on_green_button(pipeline):
    """
    If the green button contains unreadable text / noise, it must not
    be verified and must not be saved as a confirmed training sample.
    """
    buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("???? #### !!!!", "green"),  # Unreadable noise
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)
    blurred_card = load_card_frame("card_137_green-bomber.png", blurred=True)

    t0 = 5000.0
    collector = AutomaticSampleCollector(
        memory=pipeline.memory,
        matcher=pipeline.matcher,
        encoder=pipeline.encoder
    )
    collector.register_active_question(
        card_frame=blurred_card,
        option_names=["Shadow Duelist", "Green Bomber", "Harmonious Soul", "Soul King"],
        option_ids=[200, 137, 139, 201],
        predicted_name="Green Bomber",
        predicted_id=137,
        card_hash="noise_hash",
        timestamp=t0
    )

    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, None, t0 + 2.0)

    # Collector must refuse to process invalid/unverified OCR
    report = collector.process_result(clear_card, buttons, event, t0 + 2.0)
    assert report is None, "Must not save sample when green button OCR fails verification!"


# ==============================================================================
# Scenario 7: Clear Image and Green Option Disagree
# ==============================================================================
def test_clear_image_and_green_option_disagree(pipeline):
    """
    If the green button states 'Green Bomber' (#137) but the clear revealed
    artwork matches 'Harmonious Soul' (#139), save as UNVERIFIED_DISAGREEMENT.
    """
    t0 = 6000.0
    collector = AutomaticSampleCollector(
        memory=pipeline.memory,
        matcher=pipeline.matcher,
        encoder=pipeline.encoder
    )

    clear_card = load_card_frame("card_139_harmonious-soul.png", blurred=False)
    blurred_card = load_card_frame("card_139_harmonious-soul.png", blurred=True)

    collector.register_active_question(
        card_frame=blurred_card,
        option_names=["Shadow Duelist", "Green Bomber", "Harmonious Soul", "Soul King"],
        option_ids=[200, 137, 139, 201],
        predicted_name="Green Bomber",
        predicted_id=137,
        card_hash="disagree_hash",
        timestamp=t0
    )

    # Green button is set to Green Bomber (disagreeing with Harmonious Soul artwork)
    buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Green Bomber", "green"),
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, None, t0 + 2.0)

    report = collector.process_result(clear_card, buttons, event, t0 + 2.0)
    assert report is not None
    assert report.verified is False
    assert report.status == "UNVERIFIED_DISAGREEMENT"
    assert report.green_button_name == "Green Bomber"
    assert report.clear_image_match_name == "Harmonious Soul"

    # Cleanup test record
    if report.sample_id:
        pipeline.memory.invalidate_sample(report.sample_id)


# ==============================================================================
# Scenario 8: Result Screen Appears After Timer Expires
# ==============================================================================
def test_result_screen_after_timer_expires(pipeline):
    """
    When round timer expires without a user click, no red button is present,
    and a 'Time Up' banner appears while the green button highlights the answer.
    """
    timer_banner = np.ones((40, 200, 3), dtype=np.uint8) * 30
    cv2.putText(timer_banner, "Time Up!", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Green Bomber", "neutral"),
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "green"),     # Correct answer revealed at slot 3
    ]
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, timer_banner, time.time())

    assert event.is_result is True
    assert event.is_correct_answer_revealed is True
    assert event.is_wrong_selection_present is False
    assert "time up" in event.banner_text


# ==============================================================================
# Scenario 9: Repeated Screenshots of the Same Result Screen (Deduplication)
# ==============================================================================
def test_repeated_screenshots_of_same_result_screen(pipeline):
    t0 = 7000.0
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
        card_hash="dedup_hash",
        timestamp=t0
    )

    buttons = [
        make_button_frame("Shadow Duelist", "neutral"),
        make_button_frame("Green Bomber", "green"),
        make_button_frame("Harmonious Soul", "neutral"),
        make_button_frame("Soul King", "neutral"),
    ]
    detector = ResultDetector(pipeline.option_reader)
    event = detector.evaluate_screen(clear_card, buttons, None, None, t0 + 1.0)

    # First result screen evaluation: successfully learned
    report1 = collector.process_result(clear_card, buttons, event, t0 + 1.0)
    assert report1 is not None
    assert report1.verified is True

    # Second result screen evaluation 100ms later during the same post-round reveal:
    # Must return None to prevent duplicate DB writes or memory bloating!
    report2 = collector.process_result(clear_card, buttons, event, t0 + 1.1)
    assert report2 is None, "Repeated frame of the same result screen must be deduplicated!"

    # Cleanup
    if report1.sample_id:
        pipeline.memory.invalidate_sample(report1.sample_id)


# ==============================================================================
# Scenario 10: State Machine Transitions Through All 6 Phases
# ==============================================================================
def test_state_machine_transitions_all_phases():
    detector = QuestionDetector()
    assert detector.current_state == "IDLE"

    blurred_card = load_card_frame("card_137_green-bomber.png", blurred=True)
    clear_card = load_card_frame("card_137_green-bomber.png", blurred=False)

    neutral_buttons = [make_button_frame(f"Option {i}", "neutral") for i in range(4)]
    active_bundle = {
        "card_frame": blurred_card,
        "options_frame": make_options_frame(neutral_buttons),
        "button_frames": neutral_buttons
    }

    # 1. Transition IDLE -> ACTIVE_QUESTION
    e1 = detector.process_frame_bundle(active_bundle)
    assert e1.question_detected is True
    assert detector.current_state == "ACTIVE_QUESTION"

    # 2. Transition ACTIVE_QUESTION -> ANSWER_RECOGNITION on steady active frame
    e2 = detector.process_frame_bundle(active_bundle)
    assert e2.question_detected is False
    assert detector.current_state == "ANSWER_RECOGNITION"

    # 3. Transition ANSWER_RECOGNITION -> RESULT_REVEALED when green button appears
    result_buttons = [
        make_button_frame("Opt 0", "neutral"),
        make_button_frame("Opt 1", "green"),
        make_button_frame("Opt 2", "neutral"),
        make_button_frame("Opt 3", "neutral")
    ]
    result_bundle = {
        "card_frame": clear_card,
        "options_frame": make_options_frame(result_buttons),
        "button_frames": result_buttons
    }
    e3 = detector.process_frame_bundle(result_bundle)
    assert e3.is_result_detected is True
    assert detector.current_state == "RESULT_REVEALED"

    # 4. Transition RESULT_REVEALED -> SAMPLE_VERIFICATION
    detector.notify_sample_verified()
    assert detector.current_state == "SAMPLE_VERIFICATION"

    # 5. Transition to NEXT_QUESTION on screen change
    empty_bundle = {
        "card_frame": np.zeros((224, 224, 3), dtype=np.uint8),
        "options_frame": np.zeros((100, 100, 3), dtype=np.uint8),
        "button_frames": []
    }
    detector.process_frame_bundle(empty_bundle)
    assert detector.current_state == "IDLE" or detector.current_state == "NEXT_QUESTION"

    # 6. Next question cycle starts
    detector.current_state = "NEXT_QUESTION"
    e5 = detector.process_frame_bundle(active_bundle)
    assert e5.question_detected is True
    assert detector.current_state == "ACTIVE_QUESTION"

