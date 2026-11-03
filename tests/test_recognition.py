"""
tests/test_recognition.py

Unit tests for visual encoder and candidate matcher:
- Feature vector dimensionality & normalization
- Matching clear and synthetic blurred cards against 4 candidates
- Top candidate ranking accuracy
- Confidence calibration & margin evaluation
"""

import cv2
from pathlib import Path
import numpy as np
import pytest

from src.recognition.image_encoder import get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.recognition.confidence import ConfidenceCalibrator

EMBEDDINGS_NPZ = Path("models/embeddings/card_embeddings.npz")


@pytest.fixture(scope="module")
def matcher():
    assert EMBEDDINGS_NPZ.exists(), "Embeddings file must exist before testing matcher."
    return ImageMatcher(embeddings_path=EMBEDDINGS_NPZ)


@pytest.fixture(scope="module")
def encoder():
    return get_image_encoder("dinov2")


def test_encoder_output(encoder):
    # Dummy image: 224x224 RGB
    dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    vec = encoder.encode_image(dummy)
    assert vec.ndim == 1
    assert len(vec) == 384
    # Check L2 normalization
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 0.001) == 1.0


def test_matcher_independent_image_query(matcher, encoder):
    """
    Leakage-free test: Loads image from disk and encodes it dynamically.
    Never uses pre-stored embeddings from the reference database.
    """
    card_path = "assets/cards/card_137_green-bomber.png"
    img = cv2.imread(card_path)
    assert img is not None

    query = encoder.encode_image(img)
    target_name = "Green Bomber"

    candidates = [target_name, matcher.card_names[1], matcher.card_names[2], matcher.card_names[3]]
    result = matcher.match_candidates(query, candidates)

    assert result["card_name"] == target_name
    assert result["confidence"] > 0.8
    assert result["raw_score"] > 0.90
    assert result["status"] == "matched"


def test_matcher_independent_synthetic_blur(matcher, encoder):
    """
    Leakage-free test: Generates an independent, non-cached blur transformation
    (kernel 31x31, sigma 11.5) on card #139, encodes dynamically, and tests candidate matching.
    """
    card_path = "assets/cards/card_139_harmonious-soul.png"
    img = cv2.imread(card_path)
    assert img is not None

    # Apply independent blur parameters not present in precomputed NPZ
    independent_blur = cv2.GaussianBlur(img, (31, 31), 11.5)
    query = encoder.encode_image(independent_blur)
    target_name = "Harmonious Soul"

    candidates = [
        "Green Bomber",
        target_name,
        "Shadow Duelist",
        "Soul King"
    ]

    result = matcher.match_candidates(query, candidates)
    assert result["card_name"] == target_name
    assert result["confidence"] > 0.65
    assert result["margin"] > 0.04


def test_matcher_fallback_all_independent_image(matcher, encoder):
    """
    Leakage-free fallback test: Encodes an independent test image dynamically.
    """
    card_path = "assets/cards/card_137_green-bomber.png"
    img = cv2.imread(card_path)
    assert img is not None

    query = encoder.encode_image(img)
    result = matcher.match_all(query, top_k=5)
    assert result["card_name"] == "Green Bomber"
    assert result["status"] == "matched"


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
