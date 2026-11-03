"""
src/learning/result_detector.py

Phase: Result-Based Automatic Learning.
Distinguishes between ACTIVE_QUESTION and RESULT_REVEALED game states.
Detects green (correct answer) and red (wrong selection) button colors,
measures card artwork sharpness, and reads the ground-truth card name
from the green button using OCR.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.ocr.option_reader import OptionReader

logger = logging.getLogger("result_detector")


@dataclass
class CorrectOptionResult:
    correct_option: str
    card_id: Optional[int]
    slot_index: int
    confidence: float
    source: str
    verified: bool
    raw_text: str
    green_ratio: float


@dataclass
class ResultDetectionEvent:
    phase: str  # "ACTIVE_QUESTION", "RESULT_REVEALED", "TRANSITION", "IDLE"
    is_result: bool
    is_correct_answer_revealed: bool
    is_wrong_selection_present: bool
    correct_option: Optional[CorrectOptionResult]
    wrong_slot_index: Optional[int]
    card_sharpness: float
    is_card_unblurred: bool
    banner_text: str
    timestamp: float


def compute_image_sharpness(image: np.ndarray) -> float:
    """
    Computes Laplacian variance of image.
    Heavily blurred game artwork typically yields variance < 50.
    Unblurred / revealed clear artwork typically yields variance > 200.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Standardize scale so variance is size-independent
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(resized, cv2.CV_64F)
    return float(lap.var())


def get_color_ratios(button_img: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the proportion of green and red pixels in button image.
    Returns (green_ratio, red_ratio).
    """
    if button_img is None or button_img.size == 0:
        return 0.0, 0.0

    hsv = cv2.cvtColor(button_img, cv2.COLOR_BGR2HSV)

    # Green mask in HSV: Hue 35 to 88, Saturation >= 50, Value >= 45
    mask_green = cv2.inRange(hsv, (35, 50, 45), (88, 255, 255))
    green_ratio = float(np.count_nonzero(mask_green) / mask_green.size)

    # Red mask in HSV (wraps around 0 and 180): Hue 0-10 and 165-180
    mask_red1 = cv2.inRange(hsv, (0, 55, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (165, 55, 50), (180, 255, 255))
    mask_red = mask_red1 | mask_red2
    red_ratio = float(np.count_nonzero(mask_red) / mask_red.size)

    return green_ratio, red_ratio


def get_correct_option(
    button_frames: List[np.ndarray],
    option_reader: OptionReader,
    min_green_ratio: float = 0.18
) -> Optional[CorrectOptionResult]:
    """
    Identifies the green correct-answer button among the 4 choices and reads its text.
    Returns CorrectOptionResult or None if no green button is present.
    """
    if not button_frames or len(button_frames) < 2:
        return None

    best_slot = -1
    highest_green = 0.0

    for idx, btn in enumerate(button_frames):
        green_ratio, red_ratio = get_color_ratios(btn)
        # Button must have dominant green and not be confused with red
        if green_ratio > min_green_ratio and green_ratio > red_ratio and green_ratio > highest_green:
            highest_green = green_ratio
            best_slot = idx

    if best_slot == -1:
        return None

    # OCR the text inside the green button
    green_frame = button_frames[best_slot]
    preprocessed = option_reader.preprocess_button_image(green_frame)
    vision_results = option_reader.extract_text_vision(preprocessed)

    if vision_results:
        vision_results.sort(key=lambda item: item[2])
        raw_text = " ".join([t[0] for t in vision_results])
        avg_conf = float(np.mean([t[1] for t in vision_results]))
    else:
        raw_text = ""
        avg_conf = 0.0

    canonical_name, card_id, fuzzy_score = option_reader.resolve_canonical_name(raw_text)
    is_verified = (card_id is not None) and (fuzzy_score >= 65.0)

    logger.info("Found green correct button at slot %d: '%s' (Canonical: %s, ID: %s, Green: %.2f)",
                best_slot, raw_text, canonical_name, card_id, highest_green)

    return CorrectOptionResult(
        correct_option=canonical_name if is_verified else raw_text,
        card_id=card_id,
        slot_index=best_slot,
        confidence=round(avg_conf, 4),
        source="green_button_ocr",
        verified=is_verified,
        raw_text=raw_text,
        green_ratio=round(highest_green, 4)
    )


class ResultDetector:
    """
    Monitors captured game regions to detect quiz state transitions
    and identify revealed results.
    """

    def __init__(
        self,
        option_reader: OptionReader,
        sharpness_threshold: float = 120.0,
        min_button_color_ratio: float = 0.18
    ):
        self.option_reader = option_reader
        self.sharpness_threshold = sharpness_threshold
        self.min_button_color_ratio = min_button_color_ratio

    def evaluate_screen(
        self,
        card_frame: Optional[np.ndarray],
        button_frames: Optional[List[np.ndarray]],
        options_frame: Optional[np.ndarray],
        timer_frame: Optional[np.ndarray],
        timestamp: float
    ) -> ResultDetectionEvent:
        """
        Evaluates current screen to determine if the quiz is in
        ACTIVE_QUESTION, RESULT_REVEALED, TRANSITION, or IDLE phase.
        """
        if card_frame is None or card_frame.size == 0 or not button_frames:
            return ResultDetectionEvent(
                phase="IDLE",
                is_result=False,
                is_correct_answer_revealed=False,
                is_wrong_selection_present=False,
                correct_option=None,
                wrong_slot_index=None,
                card_sharpness=0.0,
                is_card_unblurred=False,
                banner_text="",
                timestamp=timestamp
            )

        # 1. Card Sharpness
        sharpness = compute_image_sharpness(card_frame)
        is_unblurred = sharpness >= self.sharpness_threshold

        # 2. Check for Green (correct) and Red (wrong) buttons
        correct_result = get_correct_option(button_frames, self.option_reader, self.min_button_color_ratio)

        wrong_slot = None
        highest_red = 0.0
        for idx, btn in enumerate(button_frames):
            _, red_r = get_color_ratios(btn)
            if red_r > self.min_button_color_ratio and red_r > highest_red:
                # Ensure it's not the green button slot
                if correct_result is None or idx != correct_result.slot_index:
                    highest_red = red_r
                    wrong_slot = idx

        # 3. Check for Result Banner OCR text if timer frame is provided
        banner_text = ""
        if timer_frame is not None and timer_frame.size > 0:
            banner_results = self.option_reader.extract_text_vision(timer_frame)
            if banner_results:
                banner_text = " ".join([b[0] for b in banner_results]).lower()

        # 4. Determine Phase
        is_result_banner = any(w in banner_text for w in ["correct", "wrong", "time up", "result", "bonus"])
        is_active_prompt = any(w in banner_text for w in ["choose", "matching", "guess"])

        # Result is revealed if green button is present OR unblurred card with result banner
        has_result_signals = (correct_result is not None) or (is_unblurred and (is_result_banner or wrong_slot is not None))

        if has_result_signals:
            phase = "RESULT_REVEALED"
        elif not is_unblurred and (is_active_prompt or not is_result_banner):
            phase = "ACTIVE_QUESTION"
        else:
            phase = "TRANSITION"

        return ResultDetectionEvent(
            phase=phase,
            is_result=(phase == "RESULT_REVEALED"),
            is_correct_answer_revealed=(correct_result is not None),
            is_wrong_selection_present=(wrong_slot is not None),
            correct_option=correct_result,
            wrong_slot_index=wrong_slot,
            card_sharpness=round(sharpness, 2),
            is_card_unblurred=is_unblurred,
            banner_text=banner_text,
            timestamp=timestamp
        )

