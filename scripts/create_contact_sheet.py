#!/usr/bin/env python3
"""
scripts/create_contact_sheet.py

Generates a visual contact sheet grid of all collected card artworks
with labels for visual inspection and dataset validation.
"""

import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATA_JSON = Path("data/cards.json")
OUTPUT_IMAGE = Path("data/card_contact_sheet.jpg")


def create_contact_sheet(
    cards_json: Path = DATA_JSON,
    output_path: Path = OUTPUT_IMAGE,
    thumb_size: tuple[int, int] = (120, 160),
    columns: int = 16,
):
    if not cards_json.exists():
        raise FileNotFoundError(f"Missing cards database at {cards_json}")

    with open(cards_json, "r", encoding="utf-8") as f:
        cards = json.load(f)

    total = len(cards)
    if total == 0:
        print("No cards found.")
        return

    rows = math.ceil(total / columns)
    thumb_w, thumb_h = thumb_size
    label_h = 24
    cell_w = thumb_w + 10
    cell_h = thumb_h + label_h + 10

    sheet_w = columns * cell_w + 20
    sheet_h = rows * cell_h + 60

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(25, 27, 34))
    draw = ImageDraw.Draw(sheet)

    # Title header
    header_text = f"Anime Card Clash — Gallery Contact Sheet ({total} Cards)"
    draw.text((20, 20), header_text, fill=(240, 240, 240))

    for idx, card in enumerate(cards):
        col = idx % columns
        row = idx // columns
        x = 20 + col * cell_w
        y = 60 + row * cell_h

        img_path = Path(card["image"])
        if img_path.exists():
            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    # Center thumbnail in cell
                    offset_x = x + (thumb_w - im_rgb.width) // 2
                    offset_y = y + (thumb_h - im_rgb.height) // 2
                    sheet.paste(im_rgb, (offset_x, offset_y))
            except Exception as e:
                draw.rectangle([x, y, x + thumb_w, y + thumb_h], fill=(120, 30, 30))
                draw.text((x + 5, y + 5), "ERROR", fill=(255, 255, 255))
        else:
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], fill=(80, 80, 80))
            draw.text((x + 5, y + 5), "MISSING", fill=(255, 255, 255))

        # Card label
        name = card["name"]
        short_name = (name[:14] + "..") if len(name) > 16 else name
        draw.text((x, y + thumb_h + 4), f"#{card['id']} {short_name}", fill=(200, 200, 200))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=85)
    print(f"Contact sheet saved to {output_path} ({sheet_w}x{sheet_h} px)")


if __name__ == "__main__":
    create_contact_sheet()
