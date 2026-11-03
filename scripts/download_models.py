#!/usr/bin/env python3
"""
scripts/download_models.py

Online Preparation Script for Anime Card Assistant:
Downloads required vision model weights once into the local models directory,
verifies cryptographic checksums against models/manifest.json, and prepares
the environment for subsequent completely offline execution.
"""

import hashlib
import json
import logging
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("download_models")

MANIFEST_PATH = PROJECT_ROOT / "models" / "manifest.json"


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found at %s", MANIFEST_PATH)
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    models = manifest.get("models", {})
    all_ok = True

    for model_key, spec in models.items():
        rel_path = spec["filename"]
        expected_sha = spec["sha256"]
        local_path = PROJECT_ROOT / "models" / rel_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            logger.info("Checking existing model checkpoint: %s", local_path)
            actual_sha = compute_sha256(local_path)
            if actual_sha == expected_sha:
                logger.info("✓ Checkpoint '%s' verified (SHA256: %s...)", model_key, actual_sha[:12])
                continue
            else:
                logger.warning("Checksum mismatch for %s. Re-downloading...", local_path)

        logger.info("Downloading checkpoint for '%s'...", model_key)
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="timm/vit_small_patch14_dinov2.lvd142m",
                filename="model.safetensors",
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False
            )
            # Ensure proper filename
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != local_path.resolve():
                downloaded_path.rename(local_path)

            actual_sha = compute_sha256(local_path)
            if actual_sha == expected_sha:
                logger.info("✓ Checkpoint '%s' successfully downloaded and verified!", model_key)
            else:
                logger.error("Downloaded file checksum (%s) does not match manifest (%s)!", actual_sha, expected_sha)
                all_ok = False
        except Exception as e:
            logger.error("Failed to download model '%s': %s", model_key, e)
            all_ok = False

    if all_ok:
        logger.info("All offline model checkpoints are verified and ready for completely offline execution.")
    else:
        logger.error("One or more models failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
