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
from typing import List, Optional, Tuple
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
    min_green_ratio: float = 0.18,
    min_ocr_confidence: float = 0.35,
    min_fuzzy_score: float = 80.0,
    candidate_names: Optional[List[str]] = None
) -> Optional[CorrectOptionResult]:
    """
    Identifies the unambiguous green correct-answer button among 4 valid button crops,
    verifies appropriate OCR confidence and strict fuzzy-match score, and confirms
    the card belongs to the active options.
    """
    # 1. Require four valid button crops
    if not button_frames or len(button_frames) != 4:
        return None

    for btn in button_frames:
        if btn is None or btn.size == 0 or btn.shape[0] < 15 or btn.shape[1] < 15:
            return None

    # 2. Identify one unambiguous green button
    qualifying_green: List[Tuple[int, float]] = []
    for idx, btn in enumerate(button_frames):
        green_ratio, red_ratio = get_color_ratios(btn)
        if green_ratio > min_green_ratio and green_ratio > red_ratio:
            qualifying_green.append((idx, green_ratio))

    if len(qualifying_green) == 0:
        return None
    elif len(qualifying_green) > 1:
        # Require unambiguous dominance
        qualifying_green.sort(key=lambda x: x[1], reverse=True)
        top_slot, top_gr = qualifying_green[0]
        second_slot, second_gr = qualifying_green[1]
        if top_gr < 1.8 * second_gr or second_gr >= 0.25:
            logger.warning("Ambiguous green buttons detected: %s. Refusing ambiguous answer.", qualifying_green)
            return None
        best_slot, highest_green = top_slot, top_gr
    else:
        best_slot, highest_green = qualifying_green[0]

    # 3. OCR the text inside the green button
    green_frame = button_frames[best_slot]
    preprocessed = option_reader.preprocess_button_image(green_frame)
    vision_results = option_reader.extract_text_vision(preprocessed)

    if not vision_results:
        return None

    vision_results.sort(key=lambda item: item[2])
    raw_text = " ".join([t[0] for t in vision_results]).strip()
    avg_conf = float(np.mean([t[1] for t in vision_results]))

    if len(raw_text) < 3 or avg_conf < min_ocr_confidence:
        logger.debug("Green button OCR below confidence threshold (conf=%.2f, text='%s')", avg_conf, raw_text)
        return None

    # 4. Strict canonical-name resolution
    canonical_name, card_id, fuzzy_score = option_reader.resolve_canonical_name(raw_text)
    is_verified = (card_id is not None) and (fuzzy_score >= min_fuzzy_score)

    if not is_verified:
        logger.debug("Green button text '%s' resolved to '%s' (fuzzy=%.1f) but below strict threshold %.1f",
                     raw_text, canonical_name, fuzzy_score, min_fuzzy_score)
        return None

    # 5. Verify the detected card belongs to the active options if provided
    if candidate_names:
        norm_cands = [c.lower().strip() for c in candidate_names]
        if canonical_name.lower().strip() not in norm_cands:
            logger.warning("Green button card '%s' (#%d) does not belong to active options %s",
                           canonical_name, card_id, candidate_names)
            return None

    logger.info("Found unambiguous green correct button at slot %d: '%s' (Canonical: %s, ID: %s, Green: %.2f)",
                best_slot, raw_text, canonical_name, card_id, highest_green)

    return CorrectOptionResult(
        correct_option=canonical_name,
        card_id=card_id,
        slot_index=best_slot,
        confidence=round(avg_conf, 4),
        source="green_button_ocr",
        verified=True,
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
        min_button_color_ratio: float = 0.18,
        min_fuzzy_score: float = 80.0
    ):
        self.option_reader = option_reader
        self.sharpness_threshold = sharpness_threshold
        self.min_button_color_ratio = min_button_color_ratio
        self.min_fuzzy_score = min_fuzzy_score

    def evaluate_screen(
        self,
        card_frame: Optional[np.ndarray],
        button_frames: Optional[List[np.ndarray]],
        options_frame: Optional[np.ndarray] = None,
        timer_frame: Optional[np.ndarray] = None,
        banner_frame: Optional[np.ndarray] = None,
        candidate_names: Optional[List[str]] = None,
        timestamp: float = 0.0
    ) -> ResultDetectionEvent:
        """
        Evaluates current screen to determine if the quiz is in
        ACTIVE_QUESTION, RESULT_REVEALED, TRANSITION, or IDLE phase.
        Uses independent result banner region rather than timer frame for banner text.
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
        correct_result = get_correct_option(
            button_frames=button_frames,
            option_reader=self.option_reader,
            min_green_ratio=self.min_button_color_ratio,
            min_fuzzy_score=self.min_fuzzy_score,
            candidate_names=candidate_names
        )

        wrong_slot = None
        highest_red = 0.0
        for idx, btn in enumerate(button_frames):
            _, red_r = get_color_ratios(btn)
            if red_r > self.min_button_color_ratio and red_r > highest_red:
                # Ensure it's not the green button slot
                if correct_result is None or idx != correct_result.slot_index:
                    highest_red = red_r
                    wrong_slot = idx

        # 3. Check for Result Banner OCR text (prioritize dedicated banner_frame)
        banner_text = ""
        target_banner = banner_frame if (banner_frame is not None and banner_frame.size > 0) else timer_frame
        if target_banner is not None and target_banner.size > 0:
            # First try Vision OCR
            banner_results = self.option_reader.extract_text_vision(target_banner)
            # Fallback to pytesseract if Vision yields no results
            if not banner_results:
                banner_results = self.option_reader.extract_text_fallback(target_banner)
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

