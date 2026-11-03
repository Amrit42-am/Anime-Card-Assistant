"""
src/ai/perception_engine.py

Unified Perception Engine for Anime Card Assistant.
Captures screen regions, isolates the 2x2 button grid (TL, TR, BL, BR),
measures card image sharpness, reads candidate texts via local OCR,
and extracts result banner messages.
"""

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

    def perceive_frame(self, run_ocr: bool = True) -> PerceptionFrame:
        """
        Captures the current game state, crops subregions, and executes OCR.
        """
        now = time.time()
        bundle = self.capture.capture_quiz_bundle()

        card_frame = bundle.get("card_frame")
        button_frames = bundle.get("button_frames", [])
        options_frame = bundle.get("options_frame")
        timer_frame = bundle.get("timer_frame")
        banner_frame = bundle.get("banner_frame")
        game_frame = bundle.get("game_frame")

        sharpness = compute_image_sharpness(card_frame) if card_frame is not None else 0.0
        is_unblurred = sharpness >= self.sharpness_threshold

        # Run OCR on 2x2 buttons
        candidates: List[CandidateOption] = []
        if run_ocr and button_frames and len(button_frames) == 4:
            candidates = self.option_reader.read_options_from_buttons(button_frames)

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
            timestamp=now
        )
