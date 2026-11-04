#!/usr/bin/env python3
"""
scripts/collect_cards.py

Phase 1: Card Data Collection
Extracts canonical card references from animecardclashcalculator.com,
validates the card count (286 expected), downloads original card artwork,
and populates data/cards.json and data/cards.db (SQLite).
Supports resuming interrupted downloads and generates a validation report.
"""

import json
import logging
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("collect_cards")

BASE_URL = "https://animecardclashcalculator.com"
PAGE_URL = f"{BASE_URL}/anime-card-clash-all-cards"
DATA_DIR = Path("data")
ASSETS_DIR = Path("assets/cards")
CARDS_JSON_PATH = DATA_DIR / "cards.json"
CARDS_DB_PATH = DATA_DIR / "cards.db"
REPORT_PATH = DATA_DIR / "collection_report.json"

EXPECTED_COUNT = 286


def slugify(text: str) -> str:
    """Generate clean slug for filenames and identifiers."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text


def fetch_card_definitions() -> list[dict]:
    """
    Fetch card definitions by inspecting the page and extracting
    the card database embedded in the Next.js static chunk.
    """
    logger.info("Fetching main page %s ...", PAGE_URL)
    req = urllib.request.Request(PAGE_URL, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8")

    # Find the chunk containing the cards payload
    chunk_matches = re.findall(r'/_next/static/chunks/[a-zA-Z0-9_-]+\.js', html)
    logger.info("Discovered %d Next.js chunks in page HTML", len(chunk_matches))

    card_data = None
    for chunk_rel in chunk_matches:
        chunk_url = urllib.parse.urljoin(BASE_URL, chunk_rel)
        try:
            creq = urllib.request.Request(chunk_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(creq, timeout=15) as cresp:
                chunk_content = cresp.read().decode("utf-8")
                if "JSON.parse(" in chunk_content and "rarity" in chunk_content and "ability" in chunk_content:
                    pattern = r'JSON\.parse\(\'((?:\\.|[^\'\\])*)\'\)'
                    matches = re.findall(pattern, chunk_content)
                    for m in matches:
                        try:
                            unescaped = m.encode("utf-8").decode("unicode_escape")
                            parsed = json.loads(unescaped)
                            if isinstance(parsed, list) and len(parsed) >= 200:
                                card_data = parsed
                                logger.info("Found card list in chunk %s (%d cards)", chunk_rel, len(parsed))
                                break
                        except Exception:
                            continue
                if card_data:
                    break
        except Exception as e:
            logger.debug("Skipping chunk %s: %s", chunk_url, e)

    if not card_data:
        raise RuntimeError("Failed to extract card database from reference website chunks.")

    return card_data


def init_database(db_path: Path):
    """Initialize SQLite schema for canonical card records."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        slug TEXT NOT NULL UNIQUE,
        image_path TEXT NOT NULL,
        image_url TEXT NOT NULL,
        rarity TEXT,
        tier TEXT,
        card_type TEXT,
        category TEXT,
        ability TEXT,
        obtain_method TEXT,
        source_url TEXT,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_card_name ON cards (name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_card_slug ON cards (slug);")
    conn.commit()
    conn.close()


def download_image(url: str, dest_path: Path, max_retries: int = 3, card_name: str = "") -> bool:
    """Download single image file with retries and content verification."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True  # Already downloaded and non-empty

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    urls_to_try = [url]
    # Specific known fallbacks for special-character URLs
    if "Humanity" in url or "Humanity" in card_name:
        urls_to_try.append(
            "https://static.wikia.nocookie.net/anime-card-clash-official/images/8/84/Humanity%27s_Weapon.png/revision/latest"
        )

    for target_url in urls_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(
                    target_url,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status == 200:
                        data = resp.read()
                        if len(data) < 500:
                            raise ValueError(f"Downloaded payload too small: {len(data)} bytes")
                        temp_path = dest_path.with_suffix(".tmp")
                        with open(temp_path, "wb") as f:
                            f.write(data)
                        temp_path.replace(dest_path)
                        return True
            except Exception as err:
                logger.warning("Attempt %d failed for %s: %s", attempt, target_url, err)
                time.sleep(0.5 * attempt)

    return False


def run_collection(force_refresh: bool = False):
    """Main collection pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    init_database(CARDS_DB_PATH)

    raw_cards = fetch_card_definitions()
    total_found = len(raw_cards)
    logger.info("Total cards retrieved from source: %d (Expected: %d)", total_found, EXPECTED_COUNT)

    processed_cards = []
    seen_names = set()
    failed_downloads = []

    # Sort deterministically by name for consistent IDs
    raw_cards.sort(key=lambda c: c.get("name", "").strip())

    for idx, raw in enumerate(raw_cards, start=1):
        name = raw.get("name", "").strip()
        if not name:
            logger.error("Card at index %d has empty name: %s", idx, raw)
            continue
        if name in seen_names:
            logger.warning("Duplicate card name detected: %s", name)
            continue
        seen_names.add(name)

        slug = slugify(name)
        img_rel = raw.get("image", "")
        if img_rel.startswith("/"):
            img_url = urllib.parse.urljoin(BASE_URL, img_rel)
        elif img_rel.startswith("http"):
            img_url = img_rel
        else:
            img_url = f"{BASE_URL}/{img_rel.lstrip('/')}"

        ext = Path(urllib.parse.urlparse(img_url).path).suffix or ".png"
        filename = f"card_{idx:03d}_{slug}{ext}"
        local_path = ASSETS_DIR / filename

        card_record = {
            "id": idx,
            "name": name,
            "slug": slug,
            "image": str(local_path),
            "image_url": img_url,
            "rarity": raw.get("rarity"),
            "tier": raw.get("tier"),
            "type": raw.get("type"),
            "category": raw.get("category"),
            "ability": raw.get("ability"),
            "obtainMethod": raw.get("obtainMethod"),
            "source_url": raw.get("url"),
            "raw_metadata": raw
        }
        processed_cards.append(card_record)

    logger.info("Downloading %d card artwork images to %s ...", len(processed_cards), ASSETS_DIR)
    for i, c in enumerate(processed_cards, 1):
        success = download_image(c["image_url"], Path(c["image"]), card_name=c["name"])
        if not success:
            logger.error("Failed to download image for %s from %s", c["name"], c["image_url"])
            failed_downloads.append({"id": c["id"], "name": c["name"], "url": c["image_url"]})
        if i % 50 == 0 or i == len(processed_cards):
            logger.info("Progress: %d / %d images checked/downloaded", i, len(processed_cards))

    # Save to cards.json
    with open(CARDS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_cards, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d cards to %s", len(processed_cards), CARDS_JSON_PATH)

    # Save to SQLite cards.db
    conn = sqlite3.connect(CARDS_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM cards;")
    for c in processed_cards:
        cur.execute("""
        INSERT INTO cards (
            id, name, slug, image_path, image_url,
            rarity, tier, card_type, category, ability,
            obtain_method, source_url, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c["id"],
            c["name"],
            c["slug"],
            c["image"],
            c["image_url"],
            c["rarity"],
            c["tier"],
            c["type"],
            c["category"],
            c["ability"],
            c["obtainMethod"],
            c["source_url"],
            json.dumps(c["raw_metadata"])
        ))
    conn.commit()
    conn.close()
    logger.info("Populated %s with %d records", CARDS_DB_PATH, len(processed_cards))

    # Validation Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_url": PAGE_URL,
        "expected_card_count": EXPECTED_COUNT,
        "actual_card_count": len(processed_cards),
        "count_matches_expected": len(processed_cards) == EXPECTED_COUNT,
        "unique_names_count": len(seen_names),
        "failed_downloads_count": len(failed_downloads),
        "failed_downloads": failed_downloads,
        "status": "PASSED" if (len(processed_cards) == EXPECTED_COUNT and len(failed_downloads) == 0) else "WARNING"
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report written to %s (Status: %s)", REPORT_PATH, report["status"])

    return report


if __name__ == "__main__":
    rep = run_collection()
    if rep["status"] != "PASSED":
        logger.warning("Collection completed with warnings: %s", rep)
        sys.exit(1)
    logger.info("Collection successfully verified! All %d cards downloaded.", rep["actual_card_count"])
