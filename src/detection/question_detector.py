"""
src/detection/question_detector.py

Phase: State-Machine Question & Transition Detector.
Implements the full Roblox game lifecycle state machine:
IDLE -> ACTIVE_QUESTION -> ANSWER_RECOGNITION -> RESULT_REVEALED -> SAMPLE_VERIFICATION -> NEXT_QUESTION

Allows repeated cards across consecutive rounds without blanket cooldown suppression.
"""

from dataclasses import dataclass
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

logger = logging.getLogger("question_detector")


@dataclass
class QuestionEvent:
    question_detected: bool       # True ONLY on the fresh transition into ACTIVE_QUESTION
    is_result_detected: bool      # True on the fresh transition into RESULT_REVEALED
    state: str                    # Current state machine state
    timestamp: float
    card_hash: str
    card_sharpness: float
    change_delta: float
    card_frame: Optional[np.ndarray] = None
    options_frame: Optional[np.ndarray] = None
    button_frames: Optional[List[np.ndarray]] = None
    timer_frame: Optional[np.ndarray] = None
    banner_frame: Optional[np.ndarray] = None


class QuestionDetector:
    """State machine tracker for Roblox Anime Card Clash quiz lifecycle."""

    STATES = [
        "IDLE",
        "ACTIVE_QUESTION",
        "ANSWER_RECOGNITION",
        "RESULT_REVEALED",
        "SAMPLE_VERIFICATION",
        "NEXT_QUESTION"
    ]

    def __init__(
        self,
        min_change_threshold: float = 6.0,
        button_edge_threshold: float = 14.0,
        sharpness_threshold: float = 120.0,
    ):
        self.min_change_threshold = min_change_threshold
        self.button_edge_threshold = button_edge_threshold
        self.sharpness_threshold = sharpness_threshold

        self.current_state = "IDLE"
        self.last_card_gray: Optional[np.ndarray] = None
        self.last_card_hash: Optional[str] = None
        self.active_question_hash: Optional[str] = None
        self.last_state_change_time: float = time.time()

    @staticmethod
    def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
        """Computes difference hash (dHash) of card image."""
        if image is None or image.size == 0:
            return "0" * (hash_size * hash_size // 4)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = "".join(["1" if b else "0" for b in diff.flatten()])
        return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)

    @staticmethod
    def compute_sharpness(card_frame: np.ndarray) -> float:
        """Computes Laplacian edge variance to measure blur vs clear artwork."""
        if card_frame is None or card_frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(card_frame, cv2.COLOR_BGR2GRAY) if card_frame.ndim == 3 else card_frame
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(resized, cv2.CV_64F)
        return float(lap.var())

    def has_button_structure(
        self,
        options_frame: np.ndarray,
        button_frames: Optional[List[np.ndarray]] = None
    ) -> bool:
        """Verifies button slot boundaries in options area (supporting both 2x2 grid and options panel)."""
        if button_frames and len(button_frames) == 4:
            valid_btns = [b for b in button_frames if b is not None and b.size > 0 and (np.std(b) > 6.0 or np.mean(b) > 15.0)]
            if len(valid_btns) == 4:
                return True

        if options_frame is None or options_frame.shape[0] < 40 or options_frame.shape[1] < 40:
            return False
        gray = cv2.cvtColor(options_frame, cv2.COLOR_BGR2GRAY) if options_frame.ndim == 3 else options_frame
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_energy = float(np.mean(np.abs(sobel_y) + np.abs(sobel_x)))
        return edge_energy > self.button_edge_threshold

    def has_card_content(self, card_frame: np.ndarray) -> bool:
        """Verifies card artwork box is not solid background."""
        if card_frame is None or card_frame.size == 0:
            return False
        gray = cv2.cvtColor(card_frame, cv2.COLOR_BGR2GRAY) if card_frame.ndim == 3 else card_frame
        return float(np.std(gray)) > 8.0

    @staticmethod
    def has_result_colors(button_frames: Optional[List[np.ndarray]]) -> Tuple[bool, bool]:
        """
        Detects if any button is green (correct) or red (incorrect selection).
        Returns (has_green, has_red).
        """
        if not button_frames:
            return False, False

        has_green = False
        has_red = False

        for btn in button_frames:
            if btn is None or btn.size == 0:
                continue
            hsv = cv2.cvtColor(btn, cv2.COLOR_BGR2HSV)

            # Green mask
            mg = cv2.inRange(hsv, (35, 50, 45), (88, 255, 255))
            if (np.count_nonzero(mg) / mg.size) > 0.18:
                has_green = True

            # Red mask
            mr1 = cv2.inRange(hsv, (0, 55, 50), (10, 255, 255))
            mr2 = cv2.inRange(hsv, (165, 55, 50), (180, 255, 255))
            mr = mr1 | mr2
            if (np.count_nonzero(mr) / mr.size) > 0.18:
                has_red = True

        return has_green, has_red

    def process_frame_bundle(self, bundle: Any) -> QuestionEvent:
        """
        Advances the state machine based on visual evidence from frame bundle
        or PerceptionFrame.
        """
        now = time.time()
        if isinstance(bundle, dict):
            card_frame = bundle.get("card_frame")
            options_frame = bundle.get("options_frame")
            button_frames = bundle.get("button_frames")
            timer_frame = bundle.get("timer_frame")
            banner_frame = bundle.get("banner_frame")
        else:
            card_frame = getattr(bundle, "card_frame", None)
            options_frame = getattr(bundle, "options_frame", None)
            button_frames = getattr(bundle, "button_frames", None)
            timer_frame = getattr(bundle, "timer_frame", None)
            banner_frame = getattr(bundle, "banner_frame", None)

        if card_frame is None or options_frame is None or not self.has_card_content(card_frame):
            if self.current_state != "IDLE":
                self._transition("IDLE", now)
            return QuestionEvent(
                question_detected=False,
                is_result_detected=False,
                state=self.current_state,
                timestamp=now,
                card_hash="",
                card_sharpness=0.0,
                change_delta=0.0,
                banner_frame=banner_frame
            )

        card_gray = cv2.cvtColor(card_frame, cv2.COLOR_BGR2GRAY) if card_frame.ndim == 3 else card_frame
        current_hash = self.compute_dhash(card_gray)
        sharpness = self.compute_sharpness(card_frame)

        if self.last_card_gray is not None and self.last_card_gray.shape == card_gray.shape:
            delta = float(np.mean(np.abs(card_gray.astype(float) - self.last_card_gray.astype(float))))
        else:
            delta = 100.0

        self.last_card_gray = card_gray.copy()
        self.last_card_hash = current_hash

        has_buttons = self.has_button_structure(options_frame, button_frames)
        has_green, has_red = self.has_result_colors(button_frames)
        is_result_colors = has_green or has_red

        new_question_triggered = False
        result_triggered = False

        # State Machine Transitions
        if self.current_state in ("IDLE", "NEXT_QUESTION", "SAMPLE_VERIFICATION"):
            # If buttons exist and no result colors are present -> active question!
            if has_buttons and not is_result_colors:
                self._transition("ACTIVE_QUESTION", now)
                self.active_question_hash = current_hash
                new_question_triggered = True
                logger.info("Detector State: ACTIVE_QUESTION (Hash: %s, Sharpness: %.1f)", current_hash, sharpness)

        elif self.current_state in ("ACTIVE_QUESTION", "ANSWER_RECOGNITION"):
            # Check for result reveal: green/red button colors OR sudden sharpness jump
            if is_result_colors or (sharpness > self.sharpness_threshold and delta > self.min_change_threshold):
                self._transition("RESULT_REVEALED", now)
                result_triggered = True
                logger.info("Detector State: RESULT_REVEALED (Green: %s, Red: %s, Sharpness: %.1f)",
                            has_green, has_red, sharpness)
            else:
                self.current_state = "ANSWER_RECOGNITION"

        elif self.current_state == "RESULT_REVEALED":
            # Screen transition away from result screen or timeout safeguard
            if not is_result_colors and (not has_buttons or delta > self.min_change_threshold):
                self._transition("NEXT_QUESTION", now)
                self.active_question_hash = None
                logger.info("Detector State: NEXT_QUESTION (Round ended, awaiting next)")
            elif now - self.last_state_change_time > 6.0:
                self._transition("NEXT_QUESTION", now)
                self.active_question_hash = None
                logger.info("Detector State: NEXT_QUESTION (Result reveal timeout after 6.0s)")

        return QuestionEvent(
            question_detected=new_question_triggered,
            is_result_detected=result_triggered,
            state=self.current_state,
            timestamp=now,
            card_hash=current_hash,
            card_sharpness=sharpness,
            change_delta=delta,
            card_frame=card_frame,
            options_frame=options_frame,
            button_frames=button_frames,
            timer_frame=timer_frame,
            banner_frame=banner_frame
        )

    def _transition(self, new_state: str, timestamp: float):
        self.current_state = new_state
        self.last_state_change_time = timestamp

    def notify_sample_verified(self):
        """Called by pipeline after sample verification finishes."""
        if self.current_state == "RESULT_REVEALED":
            self.current_state = "SAMPLE_VERIFICATION"

    def reset(self):
        self.current_state = "IDLE"
        self.last_card_gray = None
        self.last_card_hash = None
        self.active_question_hash = None
