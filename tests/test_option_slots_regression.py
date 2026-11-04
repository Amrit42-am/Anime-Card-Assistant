"""
tests/test_option_slots_regression.py

Mandatory regression test suite for 2x2 option slot positioning and
visual guess contamination prevention.
"""

from typing import Optional
import pytest
from PySide6.QtWidgets import QApplication
from src.ocr.option_reader import CandidateOption, normalize_detected_options
from src.core.decision_engine import AnswerResult
from src.ui.overlay import FloatingOverlay
from src.core.pipeline import QuizPipeline


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def overlay(qapp):
    pipeline = QuizPipeline(auto_init_models=False)
    overlay_widget = FloatingOverlay(pipeline=pipeline, is_offline=True, start_worker=False)
    yield overlay_widget
    overlay_widget.close()


def make_answer_result(
    card_name: str,
    confidence: float,
    button_slot: Optional[int],
    candidates: list,
    detected_options: list,
    visual_similarity: float = 0.8,
    is_constrained: bool = False,
) -> AnswerResult:
    return AnswerResult(
        status="matched",
        card_name=card_name,
        card_id=1,
        confidence=confidence,
        confidence_level="HIGH" if confidence > 0.8 else "LOW",
        matched_from="synthetic_blurs",
        raw_score=visual_similarity,
        margin=0.2,
        candidates=candidates,
        alternatives=[],
        explanation="Test explanation",
        processing_time_ms=10.0,
        button_slot=button_slot,
        detected_options=detected_options,
        visual_similarity=visual_similarity,
        is_constrained=is_constrained,
    )


def test_regression_scenario_1_only_bl_recognized(overlay):
    """
    Test 1: Only BL recognized.
    Input: TL invalid, TR invalid, BL="The Returner" valid, BR invalid.
    Verify:
    - [TL] —
    - [TR] —
    - [BL]    The Returner
    - [BR] —
    - No target slot assigned.
    """
    candidates = [
        CandidateOption(slot_index=0, position="TL", raw_text="", canonical_name=None, card_id=None, ocr_confidence=0.0, is_valid=False),
        CandidateOption(slot_index=1, position="TR", raw_text="", canonical_name=None, card_id=None, ocr_confidence=0.0, is_valid=False),
        CandidateOption(slot_index=2, position="BL", raw_text="The Returner", canonical_name="The Returner", card_id=42, ocr_confidence=0.92, is_valid=True),
        CandidateOption(slot_index=3, position="BR", raw_text="", canonical_name=None, card_id=None, ocr_confidence=0.0, is_valid=False),
    ]
    detected = normalize_detected_options(candidates)

    result = make_answer_result(
        card_name="The Returner",
        confidence=0.45,
        button_slot=None,
        candidates=["", "", "The Returner", ""],
        detected_options=detected,
        visual_similarity=0.78,
        is_constrained=False,
    )

    overlay.display_answer(result)

    assert overlay.cand_labels[0].text() == "[TL] —"
    assert overlay.cand_labels[1].text() == "[TR] —"
    assert overlay.cand_labels[2].text() == "[BL]    The Returner"
    assert overlay.cand_labels[3].text() == "[BR] —"
    # Ensure slot badge is hidden (no target slot)
    assert overlay.slot_badge.isHidden()


def test_regression_scenario_2_no_ocr_candidates_visual_prediction_separation(overlay):
    """
    Test 2: No OCR candidates recognized.
    Visual prediction is "The Returner".
    Verify:
    - All 4 option slots are empty: [TL] —, [TR] —, [BL] —, [BR] —.
    - The Returner appears ONLY in AI Visual Guess section with a question mark.
    - No slot is highlighted in green.
    """
    detected = normalize_detected_options([])

    result = make_answer_result(
        card_name="The Returner",
        confidence=0.35,
        button_slot=None,
        candidates=["", "", "", ""],
        detected_options=detected,
        visual_similarity=0.72,
        is_constrained=False,
    )

    overlay.display_answer(result)

    for i, prefix in enumerate(["[TL]", "[TR]", "[BL]", "[BR]"]):
        assert overlay.cand_labels[i].text() == f"{prefix} —"
        assert "background: rgba(34, 197, 94" not in overlay.cand_labels[i].styleSheet()

    # Visual guess section shows the prediction
    assert "THE RETURNER" in overlay.card_name_label.text()
    assert "?" in overlay.card_name_label.text()
    assert overlay.slot_badge.isHidden()


def test_regression_scenario_3_only_tr_recognized(overlay):
    """
    Test 3: Only TR recognized.
    Verify:
    - Stays in [TR], other slots empty.
    """
    candidates = [
        CandidateOption(slot_index=0, position="TL", raw_text="", canonical_name=None, card_id=None, ocr_confidence=0.0, is_valid=False),
        CandidateOption(slot_index=1, position="TR", raw_text="Heir of Death", canonical_name="Heir of Death", card_id=88, ocr_confidence=0.95, is_valid=True),
        CandidateOption(slot_index=2, position="BL", raw_text="", canonical_name=None, card_id=None, ocr_confidence=0.0, is_valid=False),
        CandidateOption(slot_index=3, position="BR", raw_text="", canonical_name=None, card_id=None, ocr_confidence=0.0, is_valid=False),
    ]
    detected = normalize_detected_options(candidates)

    result = make_answer_result(
        card_name="Heir of Death",
        confidence=0.40,
        button_slot=None,
        candidates=["", "Heir of Death", "", ""],
        detected_options=detected,
        visual_similarity=0.81,
        is_constrained=False,
    )

    overlay.display_answer(result)

    assert overlay.cand_labels[0].text() == "[TL] —"
    assert overlay.cand_labels[1].text() == "[TR]    Heir of Death"
    assert overlay.cand_labels[2].text() == "[BL] —"
    assert overlay.cand_labels[3].text() == "[BR] —"
    assert overlay.slot_badge.isHidden()


def test_regression_scenario_4_all_four_options_recognized_exact_slots(overlay):
    """
    Test 4: All four options recognized.
    TL: Soul Student, TR: Heir of Death, BL: The Returner, BR: Lunar Inverter.
    Verify:
    - Exact positions preserved.
    - If 4/4 constrained and Heir of Death matches TR, TR is highlighted as target.
    """
    candidates = [
        CandidateOption(slot_index=0, position="TL", raw_text="Soul Student", canonical_name="Soul Student", card_id=1, ocr_confidence=0.91, is_valid=True),
        CandidateOption(slot_index=1, position="TR", raw_text="Heir of Death", canonical_name="Heir of Death", card_id=2, ocr_confidence=0.95, is_valid=True),
        CandidateOption(slot_index=2, position="BL", raw_text="The Returner", canonical_name="The Returner", card_id=3, ocr_confidence=0.93, is_valid=True),
        CandidateOption(slot_index=3, position="BR", raw_text="Lunar Inverter", canonical_name="Lunar Inverter", card_id=4, ocr_confidence=0.90, is_valid=True),
    ]
    detected = normalize_detected_options(candidates)

    result = make_answer_result(
        card_name="Heir of Death",
        confidence=0.96,
        button_slot=1,
        candidates=["Soul Student", "Heir of Death", "The Returner", "Lunar Inverter"],
        detected_options=detected,
        visual_similarity=0.94,
        is_constrained=True,
    )

    overlay.display_answer(result)

    assert overlay.cand_labels[0].text() == "[TL]    Soul Student"
    assert overlay.cand_labels[1].text() == "[TR] ➤ Heir of Death"
    assert overlay.cand_labels[2].text() == "[BL]    The Returner"
    assert overlay.cand_labels[3].text() == "[BR]    Lunar Inverter"

    # TR should be highlighted green
    assert "background: rgba(34, 197, 94" in overlay.cand_labels[1].styleSheet()
    assert not overlay.slot_badge.isHidden()
    assert "[TR]" in overlay.slot_badge.text()


def test_regression_scenario_5_result_screen_styling_no_shifting(overlay):
    """
    Test 5: Actual result screen.
    Options: TL=Soul Student, TR=Heir of Death, BL=The Returner, BR=Lunar Inverter.
    Correct: Heir of Death (TR, slot 1). User clicked: The Returner (BL, slot 2).
    Verify:
    - TR is green with checkmark.
    - BL is red with cross mark.
    - The Returner stays in BL (never shifts to TL).
    """
    options = [
        {"slot_index": 0, "canonical_name": "Soul Student"},
        {"slot_index": 1, "canonical_name": "Heir of Death"},
        {"slot_index": 2, "canonical_name": "The Returner"},
        {"slot_index": 3, "canonical_name": "Lunar Inverter"},
    ]

    overlay.display_result_screen(
        correct_slot=1,
        correct_name="Heir of Death",
        wrong_slot=2,
        detected_options=options
    )

    assert overlay.cand_labels[0].text() == "[TL]   Soul Student"
    assert overlay.cand_labels[1].text() == "[TR] ✓ Heir of Death"
    assert overlay.cand_labels[2].text() == "[BL] ✕ The Returner"
    assert overlay.cand_labels[3].text() == "[BR]   Lunar Inverter"

    # Styling verification
    assert "#4ade80" in overlay.cand_labels[1].styleSheet()  # Green
    assert "#ef4444" in overlay.cand_labels[2].styleSheet()  # Red
    assert "CORRECT ANSWER: [TR]" in overlay.slot_badge.text()


def test_regression_scenario_6_clear_options_display_on_round_reset(overlay):
    """
    Test 6: New round clears old options and slot highlights.
    """
    # First populate options
    test_regression_scenario_4_all_four_options_recognized_exact_slots(overlay)
    assert overlay.cand_labels[1].text() != "[TR] —"

    # Reset round
    overlay.clear_options_display()

    for i, prefix in enumerate(["[TL]", "[TR]", "[BL]", "[BR]"]):
        assert overlay.cand_labels[i].text() == f"{prefix} —"
        assert "background: rgba(34, 197, 94" not in overlay.cand_labels[i].styleSheet()
    assert overlay.slot_badge.isHidden()
