"""
src/detection/question_detector.py

Phase 4: Automatic Question Detection.
Monitors the calibrated Roblox game region to detect when a new quiz appears.
Distinguishes between gameplay, loading, active quiz, and answered rounds.
Uses visual difference analysis, perceptual hashing, and edge detection on answer buttons
to prevent repeated triggers on the same question.
"""

from dataclasses import dataclass
import logging
import time
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

logger = logging.getLogger("question_detector")


@dataclass
class QuestionEvent:
    question_detected: bool
    state: str  # "IDLE", "LOADING", "ACTIVE_QUIZ", "ANSWERED"
    timestamp: float
    card_hash: str
    change_delta: float
    card_frame: Optional[np.ndarray] = None
    options_frame: Optional[np.ndarray] = None
    button_frames: Optional[list[np.ndarray]] = None


class QuestionDetector:
    """Detects active quiz states and transitions from captured video frames."""

    def __init__(
        self,
        min_change_threshold: float = 8.0,
        button_edge_threshold: float = 18.0,
        hash_similarity_threshold: int = 6,
        cooldown_seconds: float = 2.5,
    ):
        self.min_change_threshold = min_change_threshold
        self.button_edge_threshold = button_edge_threshold
        self.hash_similarity_threshold = hash_similarity_threshold
        self.cooldown_seconds = cooldown_seconds

        self.last_card_gray: Optional[np.ndarray] = None
        self.last_options_gray: Optional[np.ndarray] = None
        self.last_active_hash: Optional[str] = None
        self.last_event_time: float = 0.0
        self.current_state: str = "IDLE"

    @staticmethod
    def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
        """
        Computes difference hash (dHash) of an image region.
        Robust to minor brightness/scaling shifts, sensitive to structure changes.
        """
        if image is None or image.size == 0:
            return "0" * (hash_size * hash_size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = "".join(["1" if b else "0" for b in diff.flatten()])
        return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Computes bitwise Hamming distance between two hex hashes."""
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            return bin(val1 ^ val2).count("1")
        except Exception:
            return 999

    def has_button_structure(self, options_frame: np.ndarray) -> bool:
        """
        Checks for horizontal button outlines in the options area.
        Quiz active state displays 4 prominent rectangular button shapes.
        """
        if options_frame is None or options_frame.shape[0] < 40 or options_frame.shape[1] < 40:
            return False

        gray = cv2.cvtColor(options_frame, cv2.COLOR_BGR2GRAY) if options_frame.ndim == 3 else options_frame
        # Horizontal Sobel edge response
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = float(np.mean(np.abs(sobel_y)))
        return edge_energy > self.button_edge_threshold

    def has_card_content(self, card_frame: np.ndarray) -> bool:
        """
        Verifies that the card artwork area is non-empty and has reasonable variance
        (not a pure black or solid background color).
        """
        if card_frame is None or card_frame.size == 0:
            return False

        gray = cv2.cvtColor(card_frame, cv2.COLOR_BGR2GRAY) if card_frame.ndim == 3 else card_frame
        std_dev = float(np.std(gray))
        # Solid screen has std_dev < 10
        return std_dev > 14.0

    def process_frame_bundle(self, bundle: Dict[str, Any]) -> QuestionEvent:
        """
        Evaluates captured frame bundle.
        Returns a QuestionEvent indicating whether a fresh quiz question is present.
        """
        now = time.time()
        card_frame = bundle.get("card_frame")
        options_frame = bundle.get("options_frame")
        button_frames = bundle.get("button_frames")

        if card_frame is None or options_frame is None:
            return QuestionEvent(
                question_detected=False,
                state="IDLE",
                timestamp=now,
                card_hash="",
                change_delta=0.0
            )

        card_gray = cv2.cvtColor(card_frame, cv2.COLOR_BGR2GRAY) if card_frame.ndim == 3 else card_frame
        current_hash = self.compute_dhash(card_gray)

        # 1. Check frame delta compared to previous capture
        if self.last_card_gray is not None and self.last_card_gray.shape == card_gray.shape:
            delta = float(np.mean(np.abs(card_gray.astype(float) - self.last_card_gray.astype(float))))
        else:
            delta = 100.0

        self.last_card_gray = card_gray.copy()

        # 2. Structural verification
        card_ok = self.has_card_content(card_frame)
        buttons_ok = self.has_button_structure(options_frame)

        is_quiz_layout = card_ok and buttons_ok

        if not is_quiz_layout:
            self.current_state = "IDLE"
            self.last_active_hash = None
            return QuestionEvent(
                question_detected=False,
                state="IDLE",
                timestamp=now,
                card_hash=current_hash,
                change_delta=delta
            )

        # 3. Duplicate check and cooldown against active question
        if self.last_active_hash is not None:
            time_since_last = now - self.last_event_time
            dist = self.hamming_distance(current_hash, self.last_active_hash)

            # Suppress repeat if within cooldown or if visual change is minimal
            if time_since_last < self.cooldown_seconds or (delta < self.min_change_threshold and dist <= 12):
                return QuestionEvent(
                    question_detected=False,
                    state="ACTIVE_QUIZ",
                    timestamp=now,
                    card_hash=current_hash,
                    change_delta=delta,
                    card_frame=card_frame,
                    options_frame=options_frame,
                    button_frames=button_frames
                )

        # 4. New Question Detected!
        self.last_active_hash = current_hash
        self.last_event_time = now
        self.current_state = "ACTIVE_QUIZ"
        logger.info("New quiz question detected! Hash: %s, Delta: %.2f", current_hash, delta)

        return QuestionEvent(
            question_detected=True,
            state="ACTIVE_QUIZ",
            timestamp=now,
            card_hash=current_hash,
            change_delta=delta,
            card_frame=card_frame,
            options_frame=options_frame,
            button_frames=button_frames
        )

    def reset(self):
        """Reset state tracking."""
        self.last_card_gray = None
        self.last_options_gray = None
        self.last_active_hash = None
        self.current_state = "IDLE"

