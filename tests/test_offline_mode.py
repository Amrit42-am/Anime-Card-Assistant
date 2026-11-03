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

    manager = ModelManager(models_dir=fake_models_dir)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "missing" in msg.lower()
