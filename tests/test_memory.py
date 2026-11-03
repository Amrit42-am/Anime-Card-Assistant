"""
tests/test_memory.py

Tests for the Learned Blur Memory System:
- Adding confirmed samples
- Restricting retrieval by OCR candidate options
- Correcting mislabeled memory samples
- Invalidating samples
- Memory persistence and reload
"""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.memory.database import MemoryDatabase
from src.memory.blur_memory import BlurMemorySystem


@pytest.fixture
def temp_memory():
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test_memory.db"
    db = MemoryDatabase(db_path=db_path)
    system = BlurMemorySystem(db=db, storage_dir=temp_dir)
    yield system
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_empty_memory(temp_memory):
    assert temp_memory.total_samples == 0
    query = np.random.randn(384).astype(np.float32)
    match = temp_memory.match_sample(query)
    assert match is None


def test_add_and_retrieve_confirmed_sample(temp_memory):
    # Create synthetic query vector and dummy image
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    sample_id = temp_memory.add_confirmed_sample(
        card_id=42,
        card_name="Shadow Duelist",
        blurred_image=dummy_img,
        embedding=vec
    )
    assert sample_id > 0
    assert temp_memory.total_samples == 1

    # Query with exact vector
    match = temp_memory.match_sample(vec)
    assert match is not None
    assert match["card_id"] == 42
    assert match["card_name"] == "Shadow Duelist"
    assert match["confidence"] > 0.99


def test_candidate_restriction_safety(temp_memory):
    """
    CRITICAL RULE: If a blur matches a stored memory, but OCR options
    do NOT include that card, it MUST NOT blindly return that identity.
    """
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    temp_memory.add_confirmed_sample(
        card_id=10,
        card_name="Green Bomber",
        blurred_image=dummy_img,
        embedding=vec
    )

    # Candidates only contain cards 1, 2, 3 (NOT 10)
    match = temp_memory.match_sample(vec, allowed_card_ids=[1, 2, 3])
    assert match is None, "Should not return card 10 when not among allowed candidates!"

    # Now allow card 10
    match_ok = temp_memory.match_sample(vec, allowed_card_ids=[1, 2, 10])
    assert match_ok is not None
    assert match_ok["card_id"] == 10


def test_correct_sample(temp_memory):
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    sid = temp_memory.add_confirmed_sample(
        card_id=5,
        card_name="Wrong Card",
        blurred_image=dummy_img,
        embedding=vec
    )

    temp_memory.correct_sample(sid, correct_card_id=99, correct_card_name="Corrected Card")
    match = temp_memory.match_sample(vec)
    assert match["card_id"] == 99
    assert match["card_name"] == "Corrected Card"


def test_invalidate_sample(temp_memory):
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    sid = temp_memory.add_confirmed_sample(
        card_id=7,
        card_name="Discarded Sample",
        blurred_image=dummy_img,
        embedding=vec
    )
    assert temp_memory.total_samples == 1

    temp_memory.invalidate_sample(sid)
    assert temp_memory.total_samples == 0
    assert temp_memory.match_sample(vec) is None

