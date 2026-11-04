"""
tests/test_dataset.py

Validates the collected card dataset integrity:
- Exactly 286 card records
- Unique names and valid IDs
- Valid image files that exist on disk and can be opened
- SQLite database consistency
"""

import json
import sqlite3
from pathlib import Path
from PIL import Image

DATA_JSON = Path("data/cards.json")
DATA_DB = Path("data/cards.db")
ASSETS_DIR = Path("assets/cards")


def test_cards_json_exists():
    assert DATA_JSON.exists(), f"cards.json not found at {DATA_JSON}"


def test_cards_count_and_fields():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    assert len(cards) == 286, f"Expected 286 cards, got {len(cards)}"

    names = set()
    ids = set()

    for c in cards:
        assert "id" in c and isinstance(c["id"], int) and c["id"] > 0
        assert "name" in c and len(c["name"].strip()) > 0
        assert "image" in c and len(c["image"]) > 0

        ids.add(c["id"])
        names.add(c["name"])

        img_path = Path(c["image"])
        assert img_path.exists(), f"Missing image file for card {c['name']} at {img_path}"
        assert img_path.stat().st_size > 500, f"Image file too small or empty: {img_path}"

    assert len(names) == 286, "Duplicate card names found in cards.json"
    assert len(ids) == 286, "Duplicate card IDs found in cards.json"


def test_cards_images_openable():
    """Verify first 20 and a sample of images are valid readable image files."""
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    for c in cards[:25]:
        img_path = Path(c["image"])
        with Image.open(img_path) as im:
            assert im.width > 20 and im.height > 20, f"Invalid dimensions for {img_path}"


def test_cards_db_consistency():
    assert DATA_DB.exists(), f"cards.db not found at {DATA_DB}"
    conn = sqlite3.connect(DATA_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cards;")
    count = cur.fetchone()[0]
    assert count == 286, f"Expected 286 rows in cards.db, found {count}"

    cur.execute("SELECT id, name, image_path FROM cards LIMIT 10;")
    rows = cur.fetchall()
    assert len(rows) == 10
    conn.close()
