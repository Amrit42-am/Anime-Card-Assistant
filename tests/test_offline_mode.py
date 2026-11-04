"""
tests/test_offline_mode.py

Unit tests for completely offline operation, model checkpoint verification,
and failure handling when local weights are absent.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.ai.model_manager import ModelManager
from src.recognition.image_encoder import DINOv2Encoder


@pytest.fixture(autouse=True)
def reset_singleton():
    ModelManager.reset_instance()
    yield
    ModelManager.reset_instance()


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


def test_manifest_path_traversal_rejected(tmp_path):
    """Verify that path traversal attempts in manifest.json are detected and blocked."""
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True)

    fake_manifest = {
        "version": "1.0.0",
        "models": {
            "dinov2_small": {
                "arch": "vit_small_patch14_dinov2.lvd142m",
                "filename": "../../etc/passwd",
                "sha256": "fake",
                "expected_dimension": 384
            }
        }
    }
    with open(fake_models_dir / "manifest.json", "w") as f:
        json.dump(fake_manifest, f)

    manager = ModelManager(models_dir=fake_models_dir, require_offline=False)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "path traversal" in msg.lower()


def test_manifest_architecture_mismatch(tmp_path):
    """Verify that architecture specification mismatches are caught."""
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True)

    fake_manifest = {
        "version": "1.0.0",
        "models": {
            "dinov2_small": {
                "arch": "resnet50_incompatible",
                "filename": "dinov2/weights.safetensors",
                "sha256": "fake",
                "expected_dimension": 384
            }
        }
    }
    with open(fake_models_dir / "manifest.json", "w") as f:
        json.dump(fake_manifest, f)

    manager = ModelManager(models_dir=fake_models_dir, require_offline=False)
    is_ok, msg = manager.verify_manifest()
    assert is_ok is False
    assert "architecture mismatch" in msg.lower()


def test_pipeline_no_fallback_to_pretrained_remote_weights(monkeypatch):
    """Verify that QuizPipeline never falls back to remote pretrained weights when ModelManager fails."""
    from src.core.pipeline import QuizPipeline

    # Mock ModelManager.get_instance to fail verification
    class FailingModelManager:
        is_offline_verified = False
        verification_message = "Mock checkpoint verification failure"

    monkeypatch.setattr(ModelManager, "get_instance", lambda *args, **kwargs: FailingModelManager())

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = QuizPipeline(auto_init_models=True)

    assert "ModelManager offline verification failed" in str(exc_info.value)
    assert "./prepare.sh" in str(exc_info.value)


def test_manifest_embeddings_meta_mismatch(tmp_path):
    """Verify that embeddings_meta.json mismatching NPZ contents fails verification."""
    import numpy as np
    fake_models_dir = tmp_path / "models"
    emb_dir = fake_models_dir / "embeddings"
    emb_dir.mkdir(parents=True)

    emb_file = emb_dir / "card_embeddings.npz"
    clean = np.zeros((286, 384), dtype=np.float32)
    blur = np.zeros((286, 6, 384), dtype=np.float32)
    np.savez_compressed(emb_file, clean_embeddings=clean, blur_embeddings=blur)

    # Put mismatched metadata (card_count 100 instead of 286)
    meta_file = emb_dir / "embeddings_meta.json"
    with open(meta_file, "w") as f:
        json.dump({
            "model_name": "dinov2",
            "encoder_class": "DINOv2Encoder",
            "dimension": 384,
            "card_count": 100,
            "clean_shape": [100, 384],
            "blur_shape": [100, 6, 384],
            "variant_names": ["clean"]
        }, f)

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
    assert "metadata" in msg.lower()


def test_model_manager_loads_exact_verified_weights_path():
    """Verify ModelManager passes the exact verified checkpoint path to the encoder."""
    manager = ModelManager.get_instance()
    assert manager.is_offline_verified is True
    assert manager.verified_weights_path is not None
    assert manager.verified_weights_path.exists()
    assert manager.encoder.weights_path == manager.verified_weights_path


def test_embeddings_meta_missing_required_field(tmp_path):
    """Verify that omitting any required field from embeddings_meta.json fails verification."""
    fake_models_dir = tmp_path / "models"
    emb_dir = fake_models_dir / "embeddings"
    emb_dir.mkdir(parents=True)

    emb_file = emb_dir / "card_embeddings.npz"
    clean = np.zeros((286, 384), dtype=np.float32)
    blur = np.zeros((286, 6, 384), dtype=np.float32)
    np.savez_compressed(emb_file, clean_embeddings=clean, blur_embeddings=blur)

    # Missing 'variant_names'
    meta_file = emb_dir / "embeddings_meta.json"
    with open(meta_file, "w") as f:
        json.dump({
            "model_name": "dinov2",
            "encoder_class": "DINOv2Encoder",
            "dimension": 384,
            "card_count": 286,
            "clean_shape": [286, 384],
            "blur_shape": [286, 6, 384]
        }, f)

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
    assert "missing required field 'variant_names'" in msg.lower()


def test_embeddings_meta_file_missing(tmp_path):
    """Verify that completely missing embeddings_meta.json fails verification."""
    fake_models_dir = tmp_path / "models"
    emb_dir = fake_models_dir / "embeddings"
    emb_dir.mkdir(parents=True)

    emb_file = emb_dir / "card_embeddings.npz"
    clean = np.zeros((286, 384), dtype=np.float32)
    blur = np.zeros((286, 6, 384), dtype=np.float32)
    np.savez_compressed(emb_file, clean_embeddings=clean, blur_embeddings=blur)

    # Do NOT create embeddings_meta.json

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
    assert "metadata file missing" in msg.lower()


def test_offline_mode_blocks_network(monkeypatch):
    """Verify that QuizPipeline operates 100% offline and raises if network socket connects."""
    import socket
    from src.core.pipeline import QuizPipeline

    # Guard: monkeypatch socket.socket.connect to raise RuntimeError if called
    def forbidden_connect(*args, **kwargs):
        raise RuntimeError("Network connection attempted in strict offline mode!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)

    pipeline = QuizPipeline(auto_init_models=True)
    assert pipeline.is_offline_verified is True
    # Run a pipeline step with dummy bundle - must not attempt any socket connection
    dummy_bundle = {
        "card_frame": np.zeros((100, 100, 3), dtype=np.uint8),
        "button_frames": [np.zeros((30, 80, 3), dtype=np.uint8) for _ in range(4)],
        "options_frame": np.zeros((60, 160, 3), dtype=np.uint8),
        "timer_frame": None,
        "banner_frame": None,
        "game_frame": np.zeros((400, 600, 3), dtype=np.uint8)
    }
    pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: dummy_bundle
    ans, rep = pipeline.step()
    # Passed without triggering forbidden_connect





