"""
src/ai/model_manager.py

Offline Model Management for Anime Card Assistant.
Ensures local offline initialization of DINOv2 vision backbones, device placement
(macOS Metal Performance Shaders / MPS with CPU fallback), and singleton access
without remote downloads or telemetry.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from src.recognition.image_encoder import ImageEncoder

logger = logging.getLogger("model_manager")


class ModelManager:
    """
    Singleton manager providing offline vision encoders and local AI models.
    """

    _instance: Optional["ModelManager"] = None
    _encoder: Optional[ImageEncoder] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        models_dir: Path = Path("models")
    ):
        if getattr(self, "_initialized", False):
            return

        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self._device = self._resolve_device()

        # Enforce offline mode in PyTorch / HuggingFace
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.info("Initializing offline ModelManager on device: %s", self._device)
        self._encoder = ImageEncoder(model_name=self.model_name, device=str(self._device))
        self._initialized = True

    def _resolve_device(self) -> torch.device:
        """Prefers Apple Silicon MPS, falls back to CPU."""
        if torch.backends.mps.is_available():
            try:
                # Test MPS capability
                _ = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception as e:
                logger.warning("MPS test tensor creation failed (%s), falling back to CPU.", e)
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def encoder(self) -> ImageEncoder:
        if self._encoder is None:
            self._encoder = ImageEncoder(model_name=self.model_name, device=str(self._device))
        return self._encoder

    def get_encoder(self) -> ImageEncoder:
        return self.encoder

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
