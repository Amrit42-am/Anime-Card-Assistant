"""
src/recognition/image_encoder.py

Visual model abstraction for computing dense, normalized image embeddings.
Supports DINOv2 and MobileNetV3 with Apple Silicon MPS acceleration and CPU fallback.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Union, List, Optional

# Enforce offline mode at runtime
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

logger = logging.getLogger("image_encoder")


class BaseImageEncoder(ABC):
    """Abstract interface for all visual encoders."""

    def __init__(self, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        logger.info("Initialized %s on device: %s", self.__class__.__name__, self.device)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    def encode_image(self, image: Union[np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        """
        Encode a single image into a 1D L2-normalized float32 embedding vector.
        """
        pass

    @abstractmethod
    def encode_batch(self, images: List[Union[np.ndarray, Image.Image, Path, str]]) -> np.ndarray:
        """
        Encode a batch of images into an (N, D) L2-normalized float32 matrix.
        """
        pass

    def _prepare_pil_image(self, img_input: Union[np.ndarray, Image.Image, Path, str]) -> Image.Image:
        """Converts diverse image inputs into PIL RGB Image."""
        if isinstance(img_input, (str, Path)):
            return Image.open(str(img_input)).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 4:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")


class DINOv2Encoder(BaseImageEncoder):
    """
    DINOv2 (Vision Transformer) encoder.
    Highly robust to heavy blur, color shifts, and downsampling.
    Feature dimension: 384 (vit_small).
    Loads weights strictly from local safetensors checkpoints when available.
    """

    DEFAULT_LOCAL_WEIGHTS = Path("models/dinov2/vit_small_patch14_dinov2.safetensors")

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        device: str = None,
        weights_path: Optional[Path] = None,
        require_offline: bool = True
    ):
        super().__init__(device=device)
        import timm

        target_weights = Path(weights_path) if weights_path else self.DEFAULT_LOCAL_WEIGHTS
        self.weights_path = target_weights

        if not target_weights.exists():
            raise FileNotFoundError(
                f"Offline execution requirement failed: local model checkpoint not found at '{target_weights}'. "
                "Normal gameplay requires locally verified safetensors weights. "
                "Please run one-time preparation first: python scripts/download_models.py or ./prepare.sh"
            )

        logger.info("Loading DINOv2 architecture '%s' and local weights from %s...", model_name, target_weights)
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True
        ).to(self.device)

        from safetensors.torch import load_file
        state_dict = load_file(str(target_weights))
        self.model.load_state_dict(state_dict, strict=True)
        logger.info("Successfully loaded offline DINOv2 weights from local safetensors checkpoint.")

        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._dim = self.model.num_features
        if "vit_small" in model_name and self._dim != 384:
            raise ValueError(f"DINOv2 encoder output dimension {self._dim} != expected 384")

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_image(self, image: Union[np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        pil_img = self._prepare_pil_image(image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
            # L2 normalize
            norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
            vec = norm_features.cpu().numpy().flatten().astype(np.float32)
        return vec

    def encode_batch(self, images: List[Union[np.ndarray, Image.Image, Path, str]]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        tensors = [self.transform(self._prepare_pil_image(im)) for im in images]
        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model(batch_tensor)
            norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
            matrix = norm_features.cpu().numpy().astype(np.float32)
        return matrix


class MobileNetV3Encoder(BaseImageEncoder):
    """
    Lightweight, ultra-fast CNN encoder fallback.
    Feature dimension: 576.
    """

    def __init__(self, device: str = None):
        super().__init__(device=device)
        import torchvision.models as models

        logger.info("Loading MobileNetV3-Small encoder...")
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Extract features before classification head
        self.feature_extractor = base.features
        self.avgpool = base.avgpool
        self.model = torch.nn.Sequential(self.feature_extractor, self.avgpool, torch.nn.Flatten()).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._dim = 576

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_image(self, image: Union[np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        pil_img = self._prepare_pil_image(image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
            norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
            vec = norm_features.cpu().numpy().flatten().astype(np.float32)
        return vec

    def encode_batch(self, images: List[Union[np.ndarray, Image.Image, Path, str]]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        tensors = [self.transform(self._prepare_pil_image(im)) for im in images]
        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model(batch_tensor)
            norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
            matrix = norm_features.cpu().numpy().astype(np.float32)
        return matrix


# Convenient alias
ImageEncoder = DINOv2Encoder


def get_image_encoder(name: str = "dinov2", device: str = None) -> BaseImageEncoder:
    """Factory function for creating visual encoders."""
    name_clean = name.lower().strip()
    if name_clean in ("dinov2", "dino"):
        return DINOv2Encoder(device=device)
    elif name_clean in ("mobilenet", "mobilenetv3"):
        return MobileNetV3Encoder(device=device)
    else:
        logger.warning("Unknown encoder '%s', defaulting to DINOv2", name)
        return DINOv2Encoder(device=device)
