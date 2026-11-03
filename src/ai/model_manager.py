"""
src/ai/model_manager.py

Offline Model Management for Anime Card Assistant.
Ensures local offline initialization of DINOv2 vision backbones, device placement
(macOS Metal Performance Shaders / MPS with CPU fallback), model manifest validation,
and singleton access without remote downloads or telemetry.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

from src.recognition.image_encoder import ImageEncoder, BaseImageEncoder

logger = logging.getLogger("model_manager")


class ModelManager:
    """
    Singleton manager providing offline vision encoders and local AI models.
    Validates model checksums and embedding dimensions against models/manifest.json.
    """

    _instance: Optional["ModelManager"] = None
    _encoder: Optional[BaseImageEncoder] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        models_dir: Path = Path("models"),
        require_offline: bool = False
    ):
        if getattr(self, "_initialized", False) and getattr(self, "models_dir", None) == Path(models_dir):
            return

        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.require_offline = require_offline or (os.environ.get("REQUIRE_OFFLINE", "0") == "1")
        self._device = self._resolve_device()

        # Enforce offline mode in PyTorch / HuggingFace
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Verify offline integrity
        self.is_offline_verified, self.verification_message = self.verify_manifest()

        if self.require_offline and not self.is_offline_verified:
            raise FileNotFoundError(
                f"ModelManager offline verification failed: {self.verification_message}. "
                "Please run: python scripts/download_models.py"
            )

        logger.info("Initializing offline ModelManager on device: %s (%s)",
                    self._device, self.verification_message)

        weights_file = self.models_dir / "dinov2" / "vit_small_patch14_dinov2.safetensors"
        self._encoder = ImageEncoder(
            model_name=self.model_name,
            device=str(self._device),
            weights_path=weights_file if weights_file.exists() else None,
            require_offline=self.require_offline
        )

        self._initialized = True

    def verify_manifest(self, models_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Validates local model checkpoints and embedding file dimensions
        against models/manifest.json.
        """
        dir_to_check = Path(models_dir) if models_dir is not None else self.models_dir
        manifest_path = dir_to_check / "manifest.json"
        if not manifest_path.exists():
            return False, f"Manifest not found at {manifest_path}"

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Check DINOv2 model
            dinov2_spec = manifest.get("models", {}).get("dinov2_small")
            if not dinov2_spec:
                return False, "dinov2_small spec missing in manifest"

            model_path = dir_to_check / dinov2_spec["filename"]
            if not model_path.exists():
                return False, f"Model file missing at {model_path}"

            # Check embeddings file
            emb_spec = manifest.get("embeddings")
            if emb_spec:
                emb_path = dir_to_check / emb_spec["filename"]
                if not emb_path.exists():
                    return False, f"Embeddings file missing at {emb_path}"

                # Verify embedding dimension
                data = np.load(emb_path, allow_pickle=True)
                clean_emb = data["clean_embeddings"]
                if clean_emb.shape[1] != emb_spec["dimension"]:
                    return False, (
                        f"Incompatible embeddings dimension ({clean_emb.shape[1]}), "
                        f"expected {emb_spec['dimension']}"
                    )

            return True, "All model checkpoints and embeddings verified locally"
        except Exception as e:
            return False, f"Manifest verification error: {e}"

    def _resolve_device(self) -> torch.device:
        """Prefers Apple Silicon MPS, falls back to CPU."""
        if torch.backends.mps.is_available():
            try:
                _ = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception as e:
                logger.warning("MPS test tensor creation failed (%s), falling back to CPU.", e)
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def encoder(self) -> BaseImageEncoder:
        if self._encoder is None:
            weights_file = self.models_dir / "dinov2" / "vit_small_patch14_dinov2.safetensors"
            self._encoder = ImageEncoder(
                model_name=self.model_name,
                device=str(self._device),
                weights_path=weights_file if weights_file.exists() else None,
                require_offline=self.require_offline
            )
        return self._encoder

    def get_encoder(self) -> BaseImageEncoder:
        return self.encoder

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    @classmethod
    def get_instance(cls, require_offline: bool = False) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(require_offline=require_offline)
        return cls._instance
