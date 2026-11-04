"""
tests/test_recognition.py

Unit tests for DINOv2 visual encoder and self-learning candidate matcher:
- Feature vector dimensionality & normalization
- Cold-start handling with zero known cards
- Dynamic card learning and candidate ranking
- Blur sample integration enhancing recognition confidence
- Confidence calibration & margin evaluation
"""

import cv2
import numpy as np
import pytest

from src.recognition.image_encoder import get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.recognition.confidence import ConfidenceCalibrator
from src.memory.self_learning_db import SelfLearningDatabase


@pytest.fixture(scope="module")
def encoder():
    return get_image_encoder("dinov2")


@pytest.fixture
def empty_matcher(tmp_path):
    db_file = tmp_path / "test_matcher.db"
    db = SelfLearningDatabase(db_path=db_file)
    return ImageMatcher(db=db)


def test_encoder_output(encoder):
    """Verify DINOv2 produces 384-dim unit-normalized feature vectors."""
    dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    vec = encoder.encode_image(dummy)
    assert vec.ndim == 1
    assert len(vec) == 384
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 0.001) == 1.0


def test_matcher_cold_start_unseen(empty_matcher, encoder):
    """Verify empty matcher cleanly handles unseen cards without crashing."""
    dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    query = encoder.encode_image(dummy)

    candidates = ["Manga Writer", "Tutorial Boss", "White Flame Defender", "Royal Paladin"]
    res = empty_matcher.match_candidates(query, candidates)

    assert res.status == "unseen"
    assert res.matched_from == "cold_start_unseen"
    assert res.card_id is None
    assert res.confidence <= 0.40


def test_matcher_dynamic_learning_and_repeat_appearance(empty_matcher, encoder):
    """Verify that learning a card and blur sample enables confident subsequent matching."""
    # Create two synthetic clear card images
    card_img_a = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(card_img_a, (112, 112), 80, (255, 100, 50), -1)

    card_img_b = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(card_img_b, (40, 40), (184, 184), (50, 200, 255), -1)

    emb_a = encoder.encode_image(card_img_a)
    emb_b = encoder.encode_image(card_img_b)

    # Register both cards in the matcher
    empty_matcher.add_card(card_id=1, canonical_name="Card Alpha", embedding=emb_a)
    empty_matcher.add_card(card_id=2, canonical_name="Card Beta", embedding=emb_b)

    # Query with Card Alpha
    res_clean = empty_matcher.match_candidates(emb_a, ["Card Alpha", "Card Beta", "Card Gamma", "Card Delta"])
    assert res_clean.card_name == "Card Alpha"
    assert res_clean.raw_score > 0.95

    # Query with a heavily blurred Card Alpha
    blurred_a = cv2.GaussianBlur(card_img_a, (45, 45), 15.0)
    emb_blur_a = encoder.encode_image(blurred_a)

    # Add this verified blur sample to Card Alpha
    empty_matcher.add_blur_sample(card_id=1, blur_embedding=emb_blur_a)

    # Subsequent match with blurred query now leverages the learned blur sample
    res_blur = empty_matcher.match_candidates(emb_blur_a, ["Card Alpha", "Card Beta", "Card Gamma", "Card Delta"])
    assert res_blur.card_name == "Card Alpha"
    assert res_blur.raw_score > 0.95
    assert res_blur.matched_from == "learned_blur_memory"


def test_confidence_calibrator():
    calibrator = ConfidenceCalibrator()
    # High score, high margin
    rep_high = calibrator.evaluate(top_score=0.88, second_score=0.62)
    assert rep_high.confidence_level == "HIGH"
    assert rep_high.is_confident is True

    # High score, tiny margin (ambiguous)
    rep_amb = calibrator.evaluate(top_score=0.75, second_score=0.74)
    assert rep_amb.confidence_level == "UNCERTAIN"
    assert rep_amb.is_confident is False
