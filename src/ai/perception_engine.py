"""
src/ai/perception_engine.py

Unified Perception Engine for Anime Card Assistant.
Captures screen regions, isolates the 2x2 button grid (TL, TR, BL, BR),
measures card image sharpness, reads candidate texts via local OCR,
and extracts result banner messages.
"""

import re
from dataclasses import dataclass
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.capture.screen_capture import ScreenCapture
from src.capture.game_locator import GameLocator
from src.ocr.option_reader import OptionReader, CandidateOption
from src.learning.result_detector import compute_image_sharpness

logger = logging.getLogger("perception_engine")


@dataclass
class PerceptionFrame:
    game_frame: np.ndarray
    card_frame: np.ndarray
    button_frames: List[np.ndarray]  # 0: TL, 1: TR, 2: BL, 3: BR
    options_frame: np.ndarray
    timer_frame: Optional[np.ndarray]
    banner_frame: Optional[np.ndarray]
    ocr_candidates: List[CandidateOption]
    card_sharpness: float
    is_card_unblurred: bool
    banner_text: str
    timestamp: float
    timer_value: Optional[float] = None


class PerceptionEngine:
    """
    Perception pipeline capturing, slicing, and extracting OCR data from the Roblox game window.
    """

    def __init__(
        self,
        screen_capture: Optional[ScreenCapture] = None,
        game_locator: Optional[GameLocator] = None,
        option_reader: Optional[OptionReader] = None,
        sharpness_threshold: float = 120.0
    ):
        self.capture = screen_capture or ScreenCapture()
        self.locator = game_locator or GameLocator()
        self.option_reader = option_reader or OptionReader()
        self.sharpness_threshold = sharpness_threshold
        self._last_options_hash: Optional[str] = None
        self._cached_candidates: List[CandidateOption] = []

    @staticmethod
    def compute_options_hash(options_frame: Optional[np.ndarray]) -> str:
        """Computes a fast 16x16 gradient hash to detect when option buttons change."""
        if options_frame is None or options_frame.size == 0:
            return ""
        resized = cv2.resize(options_frame, (17, 16), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        diff = gray[:, 1:] > gray[:, :-1]
        return "".join(["1" if b else "0" for b in diff.flatten()])

    @staticmethod
    def hash_distance(h1: str, h2: str) -> int:
        """Computes Hamming distance between two binary hashes."""
        if not h1 or not h2 or len(h1) != len(h2):
            return 999
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))

    def reset_cache(self):
        """Clears cached OCR candidate options."""
        self._last_options_hash = None
        self._cached_candidates = []

    def perceive_frame(
        self,
        run_ocr: bool = True,
        bundle: Optional[Dict[str, Any]] = None
    ) -> PerceptionFrame:
        """
        Captures the current game state, crops subregions, and executes OCR.
        Uses supplied bundle if provided (e.g. for testing/mocking).
        Reuses cached candidates while the options grid remains visually stable.
        """
        now = time.time()
        b = bundle if bundle is not None else self.capture.capture_quiz_bundle()

        card_frame = b.get("card_frame")
        button_frames = b.get("button_frames", [])
        options_frame = b.get("options_frame")
        timer_frame = b.get("timer_frame")
        banner_frame = b.get("banner_frame")
        game_frame = b.get("game_frame")

        sharpness = compute_image_sharpness(card_frame) if card_frame is not None else 0.0
        is_unblurred = sharpness >= self.sharpness_threshold

        # Check whether options grid has visually changed
        opt_hash = self.compute_options_hash(options_frame)
        grid_changed = (
            self._last_options_hash is None or
            self.hash_distance(opt_hash, self._last_options_hash) > 4 or
            len(self._cached_candidates) < 4
        )

        # Run OCR on 2x2 buttons
        candidates: List[CandidateOption] = []
        if run_ocr and button_frames and len(button_frames) == 4:
            if not grid_changed and len(self._cached_candidates) == 4 and all(c.is_valid for c in self._cached_candidates):
                # Reuse cached OCR candidates while grid remains visually stable
                candidates = self._cached_candidates
            else:
                candidates = self.option_reader.read_options_from_buttons(button_frames, use_retry=False)
                valid_count = sum(1 for c in candidates if c.is_valid and c.card_id is not None)
                if valid_count < 4:
                    # Retry with alternative aggressive preprocessing
                    retry_candidates = self.option_reader.read_options_from_buttons(button_frames, use_retry=True)
                    retry_valid = sum(1 for c in retry_candidates if c.is_valid and c.card_id is not None)
                    if retry_valid > valid_count:
                        candidates = retry_candidates

                if len(candidates) == 4:
                    self._cached_candidates = candidates
                    self._last_options_hash = opt_hash

        # Timer numeric extraction
        timer_value: Optional[float] = None
        if timer_frame is not None and timer_frame.size > 0:
            timer_texts = self.option_reader.extract_text_vision(timer_frame)
            if not timer_texts:
                timer_texts = self.option_reader.extract_text_fallback(timer_frame)
            for t_str, _, _ in timer_texts:
                match = re.search(r"(\d+(\.\d+)?)", t_str)
                if match:
                    try:
                        timer_value = float(match.group(1))
                        break
                    except ValueError:
                        pass

        # Banner text extraction
        banner_text = ""
        target_banner = banner_frame if (banner_frame is not None and banner_frame.size > 0) else timer_frame
        if target_banner is not None and target_banner.size > 0:
            vision_results = self.option_reader.extract_text_vision(target_banner)
            if not vision_results:
                vision_results = self.option_reader.extract_text_fallback(target_banner)
            if vision_results:
                banner_text = " ".join([b[0] for b in vision_results]).lower()

        return PerceptionFrame(
            game_frame=game_frame,
            card_frame=card_frame,
            button_frames=button_frames,
            options_frame=options_frame,
            timer_frame=timer_frame,
            banner_frame=banner_frame,
            ocr_candidates=candidates,
            card_sharpness=round(sharpness, 2),
            is_card_unblurred=is_unblurred,
            banner_text=banner_text,
            timestamp=now,
            timer_value=timer_value
        )

