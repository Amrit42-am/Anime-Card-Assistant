"""
tests/test_dataset.py

Validates the SelfLearningDatabase integrity and cold-start learning architecture:
- Cold-start empty database initialization
- Safe dynamic card creation and duplicate prevention
- Foreign-key integrity between cards, blur samples, and sessions
- Observation tracking and sharpness-based artwork upgrading
- Prediction accuracy metrics calculation
"""

import numpy as np
import pytest

from src.memory.self_learning_db import SelfLearningDatabase


@pytest.fixture
def fresh_db(tmp_path):
    db_file = tmp_path / "test_self_learning.db"
    return SelfLearningDatabase(db_path=db_file)


def test_empty_database_starts_with_zero_records(fresh_db):
    """Verify fresh self-learning database starts with 0 cards and 0 samples."""
    cards = fresh_db.get_all_cards()
    assert len(cards) == 0

    stats = fresh_db.get_stats()
    assert stats["total_discovered_cards"] == 0
    assert stats["total_blur_samples"] == 0
    assert stats["total_correct_predictions"] == 0
    assert stats["total_incorrect_predictions"] == 0
    assert stats["accuracy_percent"] == 0.0


def test_get_or_create_card_and_duplicate_prevention(fresh_db):
    """Verify cards are discovered once and duplicate names update existing records."""
    dummy_emb = np.random.randn(384).astype(np.float32)
    card1, created1 = fresh_db.get_or_create_card(
        canonical_name="Heir of Death",
        clear_artwork_path="/path/to/heir_clear.png",
        embedding=dummy_emb,
        clear_sharpness=120.0
    )
    assert created1 is True
    assert card1.canonical_name == "Heir of Death"
    assert card1.card_id is not None
    assert card1.observation_count == 1
    assert card1.clear_sharpness == 120.0

    # Repeat appearance with case-insensitivity: should update observation count without creating duplicate
    card2, created2 = fresh_db.get_or_create_card(
        canonical_name="heir of death",
        clear_sharpness=110.0
    )
    assert created2 is False
    assert card2.card_id == card1.card_id
    assert card2.observation_count == 2
    # Lower sharpness should not overwrite original higher sharpness artwork
    assert card2.clear_sharpness == 120.0

    all_cards = fresh_db.get_all_cards()
    assert len(all_cards) == 1


def test_blur_sample_foreign_key_and_retrieval(fresh_db):
    """Verify blur samples attach strictly to existing cards and recorded sessions."""
    card, _ = fresh_db.get_or_create_card("Manga Writer")
    sample_emb = np.random.randn(384).astype(np.float32)

    # First record the session so foreign key succeeds
    fresh_db.record_session(
        session_id="q_100_manga",
        blurred_artwork_path="/path/to/manga_blur.png",
        options=["Manga Writer", "Tutorial Boss", "White Flame Defender", "Royal Paladin"],
        predicted_name="Manga Writer",
        predicted_slot=0,
        predicted_confidence=0.85,
        actual_name="Manga Writer",
        actual_slot=0,
        outcome="CORRECT",
        timestamp=100.0
    )

    sid = fresh_db.insert_blur_sample(
        card_id=card.card_id,
        blur_image_path="/path/to/manga_blur.png",
        blur_embedding=sample_emb,
        blur_hash="hash_manga_1",
        capture_conditions="slot_0",
        source_session_id="q_100_manga"
    )
    assert sid > 0

    samples = fresh_db.get_blur_samples_for_card(card.card_id)
    assert len(samples) == 1
    assert samples[0].sample_id == sid
    assert samples[0].card_id == card.card_id
    assert samples[0].source_session_id == "q_100_manga"
    assert np.allclose(samples[0].blur_embedding, sample_emb)


def test_prediction_history_accuracy_calculation(fresh_db):
    """Verify prediction history tracks correct and wrong outcomes accurately."""
    fresh_db.record_session(
        session_id="s1",
        blurred_artwork_path=None,
        options=["Heir of Death", "The Returner", "Mochi Emperor", "Beast Giant"],
        predicted_name="Heir of Death",
        predicted_slot=0,
        predicted_confidence=0.88,
        actual_name="Heir of Death",
        actual_slot=0,
        outcome="CORRECT",
        timestamp=100.0
    )
    fresh_db.record_prediction(
        session_id="s1",
        predicted_name="Heir of Death",
        actual_name="Heir of Death",
        predicted_slot=0,
        actual_slot=0,
        confidence=0.88,
        is_correct=True,
        failure_category="NONE",
        timestamp=100.0
    )

    fresh_db.record_session(
        session_id="s2",
        blurred_artwork_path=None,
        options=["The Returner", "Soul Student", "Lunar Inverter", "Heir of Death"],
        predicted_name="The Returner",
        predicted_slot=0,
        predicted_confidence=0.45,
        actual_name="Heir of Death",
        actual_slot=3,
        outcome="WRONG",
        timestamp=115.0
    )
    fresh_db.record_prediction(
        session_id="s2",
        predicted_name="The Returner",
        actual_name="Heir of Death",
        predicted_slot=0,
        actual_slot=3,
        confidence=0.45,
        is_correct=False,
        failure_category="WRONG_PREDICTION",
        timestamp=115.0
    )

    stats = fresh_db.get_stats()
    assert stats["total_correct_predictions"] == 1
    assert stats["total_incorrect_predictions"] == 1
    assert stats["total_verified_predictions"] == 2
    assert stats["accuracy_percent"] == 50.0
    assert len(stats["recent_learning_events"]) == 2
    assert stats["last_event"]["card_name"] == "Heir of Death"
