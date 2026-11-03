#!/usr/bin/env python3
"""
scripts/build_embeddings.py

Phase 2.3: Image Embeddings Precomputation
Computes normalized feature embeddings for all 286 canonical cards
and their synthetic blur variants using DINOv2.
Stores the precomputed vectors in models/embeddings/card_embeddings.npz.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.recognition.image_encoder import get_image_encoder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("build_embeddings")

CARDS_JSON = Path("data/cards.json")
BLURS_DIR = Path("data/blur_variants")
OUTPUT_DIR = Path("models/embeddings")
EMBEDDINGS_NPZ = OUTPUT_DIR / "card_embeddings.npz"
META_JSON = OUTPUT_DIR / "embeddings_meta.json"


def build_all_embeddings(
    cards_json: Path = CARDS_JSON,
    blurs_dir: Path = BLURS_DIR,
    output_npz: Path = EMBEDDINGS_NPZ,
    encoder_name: str = "dinov2",
    batch_size: int = 32
):
    if not cards_json.exists():
        raise FileNotFoundError(f"Missing cards database at {cards_json}")

    with open(cards_json, "r", encoding="utf-8") as f:
        cards = json.load(f)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    encoder = get_image_encoder(encoder_name)

    logger.info("Building embeddings for %d cards using %s...", len(cards), encoder.__class__.__name__)
    t0 = time.time()

    card_ids = []
    card_names = []
    clean_images = []
    blur_variant_names = ["base_clean", "blur_light", "blur_medium", "blur_heavy", "blur_box", "blur_compound"]

    for c in cards:
        card_ids.append(c["id"])
        card_names.append(c["name"])
        clean_images.append(c["image"])

    # 1. Compute clean embeddings in batches
    logger.info("Computing clean artwork embeddings in batches of %d...", batch_size)
    clean_embed_list = []
    for i in range(0, len(clean_images), batch_size):
        batch_paths = clean_images[i:i + batch_size]
        batch_embs = encoder.encode_batch(batch_paths)
        clean_embed_list.append(batch_embs)
    clean_embeddings = np.vstack(clean_embed_list).astype(np.float32)

    # 2. Compute blur variant embeddings: shape (N_cards, N_variants, Dim)
    logger.info("Computing synthetic blur embeddings for %d variants per card...", len(blur_variant_names))
    blur_embeddings = np.zeros((len(cards), len(blur_variant_names), encoder.dimension), dtype=np.float32)

    # Flatten all variant paths for efficient batch encoding
    all_variant_paths = []
    index_map = []  # (card_idx, variant_idx)
    for card_idx, c in enumerate(cards):
        c_id = c["id"]
        for v_idx, v_name in enumerate(blur_variant_names):
            v_path = blurs_dir / f"card_{c_id:03d}" / f"{v_name}.png"
            if v_path.exists():
                all_variant_paths.append(str(v_path))
            else:
                all_variant_paths.append(c["image"])
            index_map.append((card_idx, v_idx))

    for i in range(0, len(all_variant_paths), batch_size):
        batch_paths = all_variant_paths[i:i + batch_size]
        batch_embs = encoder.encode_batch(batch_paths)
        for sub_i, emb in enumerate(batch_embs):
            c_idx, v_idx = index_map[i + sub_i]
            blur_embeddings[c_idx, v_idx] = emb
        if (i // batch_size) % 10 == 0:
            logger.info("Progress: %d / %d blur variants encoded", min(i + batch_size, len(all_variant_paths)), len(all_variant_paths))

    # Also compute composite mean embedding for each card across its blur representations
    # (mean of normalized vectors, then re-normalized)
    composite_embeddings = np.mean(blur_embeddings, axis=1)
    composite_norms = np.linalg.norm(composite_embeddings, axis=1, keepdims=True)
    composite_norms = np.maximum(composite_norms, 1e-12)
    composite_embeddings = (composite_embeddings / composite_norms).astype(np.float32)

    # Save to compressed NPZ
    np.savez_compressed(
        output_npz,
        card_ids=np.array(card_ids, dtype=np.int32),
        card_names=np.array(card_names, dtype=object),
        clean_embeddings=clean_embeddings,
        blur_embeddings=blur_embeddings,
        composite_embeddings=composite_embeddings,
        variant_names=np.array(blur_variant_names, dtype=object)
    )

    elapsed = time.time() - t0
    logger.info("Saved embeddings to %s (Shape: clean %s, blur %s, composite %s) in %.2fs",
                output_npz, clean_embeddings.shape, blur_embeddings.shape, composite_embeddings.shape, elapsed)

    meta = {
        "model_name": encoder_name,
        "encoder_class": encoder.__class__.__name__,
        "dimension": encoder.dimension,
        "card_count": len(cards),
        "variant_names": blur_variant_names,
        "clean_shape": list(clean_embeddings.shape),
        "blur_shape": list(blur_embeddings.shape),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "time_seconds": round(elapsed, 2)
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", META_JSON)

    return meta


if __name__ == "__main__":
    build_all_embeddings()
