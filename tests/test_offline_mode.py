"""
tests/test_offline_mode.py

Unit tests for completely offline operation, model checkpoint verification,
and failure handling when local weights are absent.
"""

import json
from pathlib import Path
import pytest

from src.ai.model_manager import ModelManager
from src.recognition.image_encoder import DINOv2Encoder


def test_manifest_verification():
    manager = ModelManager.get_instance()
    is_ok, msg = manager.verify_manifest()
    assert is_ok is True
    assert "verified" in msg.lower()


def test_offline_encoder_local_safetensors():
    """Verify DINOv2Encoder loads cleanly with local weights without network access."""
    weights_path = Path("models/dinov2/vit_small_patch14_dinov2.safetensors")
    assert weights_path.exists(), "Local model weights must exist in models/dinov2/"

    encoder = DINOv2Encoder(
        model_name="vit_small_patch14_dinov2.lvd142m",
        weights_path=weights_path,
        require_offline=True
    )
    assert encoder.dimension == 384


def test_missing_local_model_raises_clear_error():
    """Verify that when offline execution is required, missing weights raise a clear FileNotFoundError."""
    non_existent = Path("models/dinov2/non_existent_weights.safetensors")

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = DINOv2Encoder(
            model_name="vit_small_patch14_dinov2.lvd142m",
            weights_path=non_existent,
            require_offline=True
        )

    err_msg = str(exc_info.value)
    assert "Offline execution requirement failed" in err_msg
    assert "download_models.py" in err_msg


def test_manifest_incompatible_embeddings_dimension(tmp_path):
    """Verify that manifest validation catches incompatible embedding dimensions."""
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True)

    fake_manifest = {
        "version": "1.0.0",
        "models": {
            "dinov2_small": {
                "arch": "vit_small_patch14_dinov2.lvd142m",
                "filename": "fake_weights.safetensors",
                "sha256": "abc",
                "expected_dimension": 384
            }
        }
    }
    with open(fake_models_dir / "manifest.json", "w") as f:
        json.dump(fake_manifest, f)

    manager = ModelManager(models_dir=fake_models_dir, require_offline=False)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "missing" in msg.lower()


def test_manifest_sha256_mismatch(tmp_path):
    """Verify that corrupt or tampered model weights fail SHA256 checksum check."""
    fake_models_dir = tmp_path / "models"
    dinov2_dir = fake_models_dir / "dinov2"
    dinov2_dir.mkdir(parents=True)

    fake_weights = dinov2_dir / "vit_small.safetensors"
    with open(fake_weights, "wb") as f:
        f.write(b"corrupted_weights_content_12345")

    fake_manifest = {
        "version": "1.0.0",
        "models": {
            "dinov2_small": {
                "arch": "vit_small_patch14_dinov2.lvd142m",
                "filename": "dinov2/vit_small.safetensors",
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "expected_dimension": 384
            }
        }
    }
    with open(fake_models_dir / "manifest.json", "w") as f:
        json.dump(fake_manifest, f)

    manager = ModelManager(models_dir=fake_models_dir, require_offline=False)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "sha256 mismatch" in msg.lower()


def test_manifest_file_size_mismatch(tmp_path):
    """Verify that truncated or unexpected file sizes fail manifest verification."""
    fake_models_dir = tmp_path / "models"
    dinov2_dir = fake_models_dir / "dinov2"
    dinov2_dir.mkdir(parents=True)

    fake_weights = dinov2_dir / "vit_small.safetensors"
    with open(fake_weights, "wb") as f:
        f.write(b"short")

    fake_manifest = {
        "version": "1.0.0",
        "models": {
            "dinov2_small": {
                "arch": "vit_small_patch14_dinov2.lvd142m",
                "filename": "dinov2/vit_small.safetensors",
                "sha256": ModelManager.compute_sha256(fake_weights),
                "expected_dimension": 384,
                "size_bytes": 999999
            }
        }
    }
    with open(fake_models_dir / "manifest.json", "w") as f:
        json.dump(fake_manifest, f)

    manager = ModelManager(models_dir=fake_models_dir, require_offline=False)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "size mismatch" in msg.lower()


def test_manifest_incompatible_card_count(tmp_path):
    """Verify that embeddings with unexpected card count (< 286) fail verification."""
    fake_models_dir = tmp_path / "models"
    emb_dir = fake_models_dir / "embeddings"
    emb_dir.mkdir(parents=True)

    # Save dummy embeddings with 100 cards instead of 286
    emb_file = emb_dir / "card_embeddings.npz"
    import numpy as np
    dummy_clean = np.zeros((100, 384), dtype=np.float32)
    np.savez_compressed(emb_file, clean_embeddings=dummy_clean)

    # Also put real or dummy weights
    dinov2_dir = fake_models_dir / "dinov2"
    dinov2_dir.mkdir(parents=True)
    fake_weights = dinov2_dir / "weights.safetensors"
    with open(fake_weights, "wb") as f:
        f.write(b"dummy")

    fake_manifest = {
        "version": "1.0.0",
        "models": {
            "dinov2_small": {
                "arch": "vit_small_patch14_dinov2.lvd142m",
                "filename": "dinov2/weights.safetensors",
                "sha256": ModelManager.compute_sha256(fake_weights),
                "expected_dimension": 384,
                "size_bytes": 5
            }
        },
        "embeddings": {
            "filename": "embeddings/card_embeddings.npz",
            "expected_cards": 286,
            "dimension": 384
        }
    }
    with open(fake_models_dir / "manifest.json", "w") as f:
        json.dump(fake_manifest, f)

    manager = ModelManager(models_dir=fake_models_dir, require_offline=False)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "incompatible embeddings card count" in msg.lower()


