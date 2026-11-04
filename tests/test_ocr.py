"""
tests/test_ocr.py

Tests for OptionReader:
- Cleaning noisy strings and button prefixes
- Fuzzy resolution against canonical card database
- Multiline card names
- Handling empty and corrupt inputs
"""

import cv2
from pathlib import Path
import numpy as np
import pytest
from src.ocr.option_reader import OptionReader


@pytest.fixture(scope="module")
def reader():
    return OptionReader()


def test_clean_text(reader):
    assert reader.clean_text("1. Shadow Commander") == "Shadow Commander"
    assert reader.clean_text("[A]  Monkey Deity ") == "Monkey Deity"
    assert reader.clean_text(" - Blood Demon Triad") == "Blood Demon Triad"
    assert reader.clean_text("3) Fearsome Beast") == "Fearsome Beast"


def test_fuzzy_resolution_exact(reader):
    canon, cid, score = reader.resolve_canonical_name("Green Bomber")
    assert canon == "Green Bomber"
    assert cid is not None
    assert score == 100.0


def test_fuzzy_resolution_typos(reader):
    # Minor OCR typos or spacing issues
    canon, cid, score = reader.resolve_canonical_name("Grren Bombr")
    assert canon == "Green Bomber"
    assert cid is not None
    assert score > 70.0


def test_fuzzy_resolution_multiline(reader):
    canon, cid, score = reader.resolve_canonical_name("Harmonious\nSoul")
    assert canon == "Harmonious Soul"
    assert cid is not None


def test_ocr_synthetic_buttons(reader):
    # Create 4 synthetic button images with card names
    names = ["Green Bomber", "Harmonious Soul", "Shadow Duelist", "Soul King"]
    buttons = []
    for n in names:
        img = np.ones((70, 280, 3), dtype=np.uint8) * 35
        cv2.putText(img, n, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        buttons.append(img)

    results = reader.read_options_from_buttons(buttons)
    assert len(results) == 4
    for idx, opt in enumerate(results):
        assert opt.canonical_name == names[idx]
        assert opt.card_id is not None
        assert opt.is_valid is True


def test_empty_results(reader):
    # Blank black image
    blank = np.zeros((60, 200, 3), dtype=np.uint8)
    results = reader.read_options_from_buttons([blank])
    assert len(results) == 1
    assert results[0].is_valid is False


def test_option_reader_tesseract_missing_no_error_spam(monkeypatch, reader, caplog):
    """Verify that when Tesseract binary is not present, extract_text_fallback returns [] cleanly without spamming errors."""
    import logging
    import src.ocr.option_reader as opt_mod
    monkeypatch.setattr(opt_mod, "TESSERACT_AVAILABLE", False)

    blank = np.zeros((60, 200, 3), dtype=np.uint8)
    with caplog.at_level(logging.ERROR):
        res = reader.extract_text_fallback(blank)
    assert res == []
    # No error logs emitted
    error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_logs) == 0


def test_diagnose_quiz_crops(tmp_path):
    """Verify that ScreenCapture.diagnose_quiz_crops produces annotated screenshots, numbered crops, and json logs."""
    from src.capture.screen_capture import ScreenCapture
    capture = ScreenCapture()
    diag_dir = tmp_path / "diag"
    rep = capture.diagnose_quiz_crops(output_dir=diag_dir)

    assert "game_dimensions" in rep
    assert (diag_dir / "debug_screen_crops.png").exists()
    assert (diag_dir / "04_annotated_regions.png").exists()
    assert (diag_dir / "ocr_results.json").exists()
    assert (diag_dir / "detection_results.json").exists()
    assert len(rep["crop_paths"]) == 4
    for cp in rep["crop_paths"]:
        assert Path(cp).exists()


def test_option_reader_recording_card_names(reader):
    """Verify resolution of canonical names from the gameplay recording."""
    # Exact canonical matches
    canon, cid, score = reader.resolve_canonical_name("White Flame Defender")
    assert canon == "White Flame Defender"
    assert cid is not None
    assert score == 100.0

    canon, cid, score = reader.resolve_canonical_name("Merged Beast")
    assert canon == "Merged Beast"
    assert cid is not None
    assert score == 100.0

    canon, cid, score = reader.resolve_canonical_name("Copy Ninja")
    assert canon == "Copy Ninja"
    assert cid is not None
    assert score == 100.0

    canon, cid, score = reader.resolve_canonical_name("Awakened Light Admiral")
    assert canon == "Awakened Light Admiral"
    assert cid is not None
    assert score == 100.0

    # In-game shorthand: 'Stark Gunner' resolves to canonical 'Awakened Stark Gunner'
    canon, cid, score = reader.resolve_canonical_name("Stark Gunner")
    assert canon == "Awakened Stark Gunner"
    assert cid is not None
    assert score > 70.0


def test_option_reader_winning_margin_rejection(reader):
    """Verify that gibberish or ambiguous text without winning margin is not accepted."""
    # Completely random text
    canon, cid, score = reader.resolve_canonical_name("XkL999_zzq")
    assert cid is None
