"""
tests/test_self_learning.py

Comprehensive test suite for the Guess -> Observe -> Verify -> Learn architecture:
- Empty database cold start handling
- Correct prediction verification & learning
- Incorrect prediction learning (actual green button becomes true label, wrong guess logged separately)
- Repeat appearance recognition improvement
- Failure scenario safety (unverified results never contaminate memory)
- Complete question session linking with unique IDs
"""

import cv2
import numpy as np
import pytest

from src.memory.self_learning_db import SelfLearningDatabase
from src.recognition.image_matcher import ImageMatcher
from src.recognition.image_encoder import get_image_encoder
from src.learning.sample_collector import AutomaticSampleCollector
from src.learning.result_detector import ResultDetectionEvent, CorrectOptionResult


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "self_learning_test.db"
    return SelfLearningDatabase(db_path=db_file)


@pytest.fixture
def encoder():
    return get_image_encoder("dinov2")


@pytest.fixture
def matcher(test_db):
    return ImageMatcher(db=test_db)


@pytest.fixture
def collector(test_db, matcher, encoder, tmp_path):
    collector_obj = AutomaticSampleCollector(
        memory=None,
        matcher=matcher,
        encoder=encoder,
        db=test_db,
        max_pairing_delay=10.0,
        min_pairing_delay=0.1
    )
    collector_obj.cards_dir = tmp_path / "learned_cards"
    collector_obj.blurs_dir = tmp_path / "learned_blurs"
    collector_obj.cards_dir.mkdir(parents=True, exist_ok=True)
    collector_obj.blurs_dir.mkdir(parents=True, exist_ok=True)
    return collector_obj


def test_cold_start_unseen_card(matcher, encoder):
    """Verify application starts with 0 cards and handles unseen cards with honest low confidence."""
    assert len(matcher.card_names) == 0

    dummy_blur = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    query_emb = encoder.encode_image(dummy_blur)

    candidates = ["Manga Writer", "Tutorial Boss", "White Flame Defender", "Royal Paladin"]
    res = matcher.match_candidates(query_emb, candidates)

    assert res.status == "unseen"
    assert res.matched_from == "cold_start_unseen"
    assert res.confidence <= 0.40
    assert res.card_id is None


def test_correct_prediction_lifecycle(collector, test_db, matcher, encoder):
    """
    Test Case A: The AI prediction is CORRECT.
    - AI predicts Heir of Death on TR (slot 1)
    - Game reveals green button on TR (slot 1, Heir of Death)
    - Outcome is verified as CORRECT
    - Clear artwork and blur sample are paired with same question ID
    - History records CORRECT
    """
    t0 = 1000.0
    blurred_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(blurred_img, (112, 112), 60, (200, 100, 50), -1)

    clear_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(clear_img, (112, 112), 60, (255, 120, 70), -1)

    options = ["Soul Student", "Heir of Death", "The Returner", "Lunar Inverter"]

    # 1. Active Question Phase: Register prediction
    sid = collector.register_active_question(
        card_frame=blurred_img,
        option_names=options,
        option_ids=[None, None, None, None],
        predicted_name="Heir of Death",
        predicted_id=None,
        card_hash="hash_alpha",
        timestamp=t0,
        predicted_slot=1,
        predicted_confidence=0.75
    )
    assert collector.pending_question is not None
    assert collector.pending_question.session_id == sid

    # 2. Result Reveal Phase: Game reveals green button at slot 1 ("Heir of Death")
    btn_frames = [np.zeros((30, 80, 3), dtype=np.uint8) for _ in range(4)]
    res_event = ResultDetectionEvent(
        phase="RESULT_REVEALED",
        is_result=True,
        is_correct_answer_revealed=True,
        is_wrong_selection_present=False,
        correct_option=CorrectOptionResult(
            correct_option="Heir of Death",
            card_id=None,
            slot_index=1,
            confidence=0.95,
            source="green_button_ocr",
            verified=True,
            raw_text="Heir of Death",
            green_ratio=0.45
        ),
        wrong_slot_index=None,
        card_sharpness=120.0,
        is_card_unblurred=True,
        banner_text="CORRECT",
        timestamp=t0 + 1.0
    )

    report = collector.process_result(
        clear_frame=clear_img,
        button_frames=btn_frames,
        result_event=res_event,
        timestamp=t0 + 1.0
    )

    # 3. Verification
    assert report is not None
    assert report.verified is True
    assert report.status == "CORRECT"
    assert report.card_name == "Heir of Death"
    assert report.card_id > 0

    # Verify database state
    card = test_db.get_card_by_name("Heir of Death")
    assert card is not None
    assert card.observation_count == 1
    assert card.clear_artwork_path is not None

    samples = test_db.get_blur_samples_for_card(card.card_id)
    assert len(samples) == 1
    assert samples[0].source_session_id == sid

    # Verify prediction history
    stats = test_db.get_stats()
    assert stats["total_correct_predictions"] == 1
    assert stats["total_incorrect_predictions"] == 0
    assert stats["accuracy_percent"] == 100.0


def test_incorrect_prediction_lifecycle(collector, test_db, matcher, encoder):
    """
    Test Case B: The AI prediction is WRONG.
    - AI predicts The Returner on BL (slot 2)
    - Game reveals green button on TR (slot 1, Heir of Death)
    - Outcome is WRONG
    - The Returner is NOT saved as the identity of the card!
    - The TRUE answer (Heir of Death) becomes the training label!
    - The wrong guess (The Returner) is recorded separately in prediction history.
    """
    t0 = 2000.0
    blurred_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(blurred_img, (50, 50), (170, 170), (100, 200, 50), -1)

    clear_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(clear_img, (50, 50), (170, 170), (120, 255, 60), -1)

    options = ["Soul Student", "Heir of Death", "The Returner", "Lunar Inverter"]

    # 1. AI made WRONG prediction: "The Returner" on BL (slot 2)
    sid = collector.register_active_question(
        card_frame=blurred_img,
        option_names=options,
        option_ids=[None, None, None, None],
        predicted_name="The Returner",
        predicted_id=None,
        card_hash="hash_beta",
        timestamp=t0,
        predicted_slot=2,
        predicted_confidence=0.45
    )
    assert sid.startswith("q_")

    # 2. Result Reveal: Green button is actually TR (slot 1, "Heir of Death")
    btn_frames = [np.zeros((30, 80, 3), dtype=np.uint8) for _ in range(4)]
    res_event = ResultDetectionEvent(
        phase="RESULT_REVEALED",
        is_result=True,
        is_correct_answer_revealed=True,
        is_wrong_selection_present=True,
        correct_option=CorrectOptionResult(
            correct_option="Heir of Death",
            card_id=None,
            slot_index=1,
            confidence=0.92,
            source="green_button_ocr",
            verified=True,
            raw_text="Heir of Death",
            green_ratio=0.50
        ),
        wrong_slot_index=2,
        card_sharpness=115.0,
        is_card_unblurred=True,
        banner_text="WRONG",
        timestamp=t0 + 1.2
    )

    report = collector.process_result(
        clear_frame=clear_img,
        button_frames=btn_frames,
        result_event=res_event,
        timestamp=t0 + 1.2
    )

    # 3. Verification of Learning Outcome
    assert report is not None
    assert report.verified is True
    assert report.status == "WRONG"
    # Ground truth learned must be Heir of Death, NOT The Returner!
    assert report.card_name == "Heir of Death"

    # Confirm "The Returner" was NOT created as the card
    wrong_card = test_db.get_card_by_name("The Returner")
    assert wrong_card is None

    # Confirm "Heir of Death" WAS created as the true card
    true_card = test_db.get_card_by_name("Heir of Death")
    assert true_card is not None
    assert true_card.canonical_name == "Heir of Death"

    # Verify the blur sample was attached to the true card
    samples = test_db.get_blur_samples_for_card(true_card.card_id)
    assert len(samples) == 1
    assert samples[0].card_id == true_card.card_id

    # Verify history recorded the wrong guess
    stats = test_db.get_stats()
    assert stats["total_correct_predictions"] == 0
    assert stats["total_incorrect_predictions"] == 1
    assert stats["accuracy_percent"] == 0.0
    assert len(stats["recent_learning_events"]) == 1
    assert stats["recent_learning_events"][0]["previous_prediction"] == "The Returner"
    assert stats["recent_learning_events"][0]["outcome"] == "WRONG"


def test_repeat_appearance_improvement(collector, test_db, matcher, encoder):
    """
    Verify that after learning a card:
    - Subsequent query recognizes it with higher similarity and confidence
    - New blur sample is attached to existing card without creating duplicates
    - Observation count increments
    """
    # First learn "Manga Writer" via round 1
    test_correct_prediction_lifecycle(collector, test_db, matcher, encoder)

    card = test_db.get_card_by_name("Heir of Death")
    assert card is not None
    initial_obs = card.observation_count

    # Round 2: Heir of Death appears again
    t2 = 3000.0
    new_blur = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(new_blur, (112, 112), 60, (210, 110, 55), -1)

    # Matcher now recognizes Heir of Death from round 1!
    query_emb = encoder.encode_image(new_blur)
    res = matcher.match_candidates(query_emb, ["Soul Student", "Heir of Death", "The Returner", "Lunar Inverter"])
    assert res.card_name == "Heir of Death"
    assert res.raw_score > 0.90

    # Register round 2
    sid2 = collector.register_active_question(
        card_frame=new_blur,
        option_names=["Soul Student", "Heir of Death", "The Returner", "Lunar Inverter"],
        option_ids=[None, card.card_id, None, None],
        predicted_name="Heir of Death",
        predicted_id=card.card_id,
        card_hash="hash_alpha_2",
        timestamp=t2,
        predicted_slot=1,
        predicted_confidence=res.confidence
    )
    assert sid2.startswith("q_")

    btn_frames = [np.zeros((30, 80, 3), dtype=np.uint8) for _ in range(4)]
    res_event = ResultDetectionEvent(
        phase="RESULT_REVEALED",
        is_result=True,
        is_correct_answer_revealed=True,
        is_wrong_selection_present=False,
        correct_option=CorrectOptionResult(
            correct_option="Heir of Death",
            card_id=card.card_id,
            slot_index=1,
            confidence=0.96,
            source="green_button_ocr",
            verified=True,
            raw_text="Heir of Death",
            green_ratio=0.48
        ),
        wrong_slot_index=None,
        card_sharpness=125.0,
        is_card_unblurred=True,
        banner_text="CORRECT",
        timestamp=t2 + 1.0
    )

    collector.process_result(new_blur, btn_frames, res_event, t2 + 1.0)

    # Verify duplicate was prevented and observation count increased
    card_updated = test_db.get_card_by_id(card.card_id)
    assert card_updated.observation_count == initial_obs + 1

    # Verify second blur sample was saved
    samples = test_db.get_blur_samples_for_card(card.card_id)
    assert len(samples) == 2


def test_unverified_result_never_contaminates_database(collector, test_db):
    """Verify that unverified or ambiguous result frames are rejected without saving unverified cards."""
    t0 = 5000.0
    blurred_img = np.zeros((224, 224, 3), dtype=np.uint8)
    collector.register_active_question(
        card_frame=blurred_img,
        option_names=["Opt A", "Opt B", "Opt C", "Opt D"],
        option_ids=[None, None, None, None],
        predicted_name="Opt A",
        predicted_id=None,
        card_hash="hash_unverified",
        timestamp=t0
    )

    # Result with unverified green button
    res_event = ResultDetectionEvent(
        phase="RESULT_REVEALED",
        is_result=True,
        is_correct_answer_revealed=False,
        is_wrong_selection_present=False,
        correct_option=None,  # Not verified
        wrong_slot_index=None,
        card_sharpness=20.0,  # Still blurred
        is_card_unblurred=False,
        banner_text="",
        timestamp=t0 + 1.0
    )

    report = collector.process_result(blurred_img, [], res_event, t0 + 1.0)
    assert report is None

    # Database must remain completely empty
    assert len(test_db.get_all_cards()) == 0
    assert len(test_db.get_all_verified_blur_samples()) == 0
