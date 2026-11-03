#!/usr/bin/env python3
"""
scripts/generate_blurs.py

Phase 2.1: Image Preprocessing & Synthetic Blur Generation
Creates multiple synthetic blur variants of each reference card
simulating different blur strengths (light, medium, heavy, box, compound shader).
Outputs to data/blur_variants/card_XXX/
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.recognition.blur_generator import BlurAugmentor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("generate_blurs")

CARDS_JSON = Path("data/cards.json")
OUTPUT_DIR = Path("data/blur_variants")


def generate_all_blurs(cards_json: Path = CARDS_JSON, output_dir: Path = OUTPUT_DIR):
    if not cards_json.exists():
        raise FileNotFoundError(f"Missing cards database at {cards_json}")

    with open(cards_json, "r", encoding="utf-8") as f:
        cards = json.load(f)

    augmentor = BlurAugmentor(target_size=(224, 224))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic blur variants for %d cards...", len(cards))
    total_variants_count = 0

    for idx, card in enumerate(cards, start=1):
        card_id = card["id"]
        card_dir = output_dir / f"card_{card_id:03d}"
        card_dir.mkdir(parents=True, exist_ok=True)

        img_path = Path(card["image"])
        if not img_path.exists():
            logger.warning("Image missing for card %d: %s", card_id, img_path)
            continue

        variants = augmentor.generate_blur_variants(img_path)

        for name, img in variants.items():
            out_file = card_dir / f"{name}.png"
            cv2.imwrite(str(out_file), img)
            total_variants_count += 1

        if idx % 50 == 0 or idx == len(cards):
            logger.info("Processed %d / %d cards (%d blur variants generated)", idx, len(cards), total_variants_count)

    logger.info("Successfully generated %d total synthetic blur images in %s", total_variants_count, output_dir)
    return total_variants_count


if __name__ == "__main__":
    generate_all_blurs()
