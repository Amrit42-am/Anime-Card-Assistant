"""
src/recognition/blur_generator.py

Image preprocessing and synthetic blur generation using OpenCV and NumPy.
Applies Gaussian blur, box blur, downsampling/upsampling (pixelation/re-interpolation),
brightness/contrast adjustments, and center cropping to emulate various game shader blur levels.
"""

import cv2
import numpy as np
from pathlib import Path


class BlurAugmentor:
    """Generates synthetic blur variants simulating Roblox shader effects."""

    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def center_crop_and_resize(self, image: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """Center crop and resize to target_size."""
        h, w = image.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

    def adjust_brightness_contrast(
        self, image: np.ndarray, alpha: float = 1.0, beta: int = 0
    ) -> np.ndarray:
        """
        Adjust contrast (alpha: 1.0 = unchanged) and brightness (beta: -100 to 100).
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def apply_gaussian_blur(self, image: np.ndarray, ksize: int, sigma: float = 0) -> np.ndarray:
        """Applies Gaussian blur with odd kernel size."""
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    def apply_box_blur(self, image: np.ndarray, ksize: int) -> np.ndarray:
        """Applies Box blur with averaging kernel."""
        return cv2.boxFilter(image, -1, (ksize, ksize))

    def apply_downsample_upsample(self, image: np.ndarray, scale: float = 0.12) -> np.ndarray:
        """
        Downsamples image to simulate Roblox low-resolution texture / mipmap
        and upsamples back with bilinear or bicubic filtering.
        """
        h, w = image.shape[:2]
        small_w = max(4, int(w * scale))
        small_h = max(4, int(h * scale))
        down = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    def generate_blur_variants(self, image_input: np.ndarray | str | Path) -> dict[str, np.ndarray]:
        """
        Generate multiple blur variants simulating different blur strengths.
        Returns a dictionary: { variant_name: processed_image_bgr }
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
        else:
            img = image_input

        # Standardize base size
        base = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        variants = {}

        # 1. Base clean resized
        variants["base_clean"] = base

        # 2. Light Gaussian Blur (k=15, sigma=5)
        variants["blur_light"] = self.apply_gaussian_blur(base, ksize=15, sigma=5.0)

        # 3. Medium Gaussian Blur (k=31, sigma=11)
        variants["blur_medium"] = self.apply_gaussian_blur(base, ksize=31, sigma=11.0)

        # 4. Heavy Gaussian Blur (k=55, sigma=21)
        variants["blur_heavy"] = self.apply_gaussian_blur(base, ksize=55, sigma=21.0)

        # 5. Box blur + slight contrast (simulates Roblox UI blur element)
        box_blurred = self.apply_box_blur(base, ksize=29)
        variants["blur_box"] = self.adjust_brightness_contrast(box_blurred, alpha=1.05, beta=5)

        # 6. Downsample-upsample + Gaussian blur (compound shader blur)
        down_up = self.apply_downsample_upsample(base, scale=0.10)
        variants["blur_compound"] = self.apply_gaussian_blur(down_up, ksize=21, sigma=8.0)

        return variants
