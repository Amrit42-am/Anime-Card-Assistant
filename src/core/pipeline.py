"""
src/core/pipeline.py

Phase 9 & 13: Main Application Pipeline.
Coordinates ScreenCapture, QuestionDetector, OptionReader, ImageEncoder,
ImageMatcher, BlurMemorySystem, and AnswerDecisionEngine.
Manages the application state machine:
IDLE -> DETECTING -> QUESTION_FOUND -> OCR_PROCESSING -> IMAGE_MATCHING -> ANSWER_READY / UNCERTAIN
"""

import logging
import time
from typing import Optional, Dict, Any

from src.capture.region_manager import CalibrationConfig
from src.capture.screen_capture import ScreenCapture
from src.detection.question_detector import QuestionDetector, QuestionEvent
from src.ocr.option_reader import OptionReader
from src.recognition.image_encoder import BaseImageEncoder, get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.core.decision_engine import AnswerDecisionEngine, AnswerResult

logger = logging.getLogger("pipeline")


class QuizPipeline:
    """End-to-end processing pipeline for Roblox card recognition."""

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        encoder_name: str = "dinov2",
        auto_init_models: bool = True
    ):
        self.calibration = calibration or CalibrationConfig.load()
        self.state = "IDLE"
        self.last_result: Optional[AnswerResult] = None

        logger.info("Initializing pipeline modules...")
        self.capture = ScreenCapture(calibration=self.calibration)
        self.detector = QuestionDetector()
        self.option_reader = OptionReader()

        if auto_init_models:
            self.encoder: BaseImageEncoder = get_image_encoder(encoder_name)
            self.matcher = ImageMatcher()
            self.memory = BlurMemorySystem(encoder=self.encoder)
            self.decision_engine = AnswerDecisionEngine(
                matcher=self.matcher,
                encoder=self.encoder,
                option_reader=self.option_reader,
                memory=self.memory
            )
        else:
            self.encoder = None
            self.matcher = None
            self.memory = None
            self.decision_engine = None

        logger.info("QuizPipeline successfully initialized.")

    def step(self) -> Optional[AnswerResult]:
        """
        Executes a single monitoring step:
        1. Capture screen bundle.
        2. Run question detector.
        3. If fresh question detected, run decision engine.
        4. Return AnswerResult if resolved, else None.
        """
        bundle = self.capture.capture_quiz_bundle(self.calibration)
        event: QuestionEvent = self.detector.process_frame_bundle(bundle)

        if not event.question_detected:
            self.state = "DETECTING" if event.state == "ACTIVE_QUIZ" else "IDLE"
            return None

        # Fresh Question Found!
        self.state = "QUESTION_FOUND"
        logger.info("Pipeline: New question detected! Commencing analysis...")

        card_frame = event.card_frame
        button_frames = event.button_frames
        options_frame = event.options_frame

        self.state = "OCR_PROCESSING"
        # Decision engine handles OCR + Embedding + Matching
        result = self.decision_engine.process_question(
            card_frame=card_frame,
            button_frames=button_frames,
            options_frame=options_frame
        )

        self.last_result = result
        self.state = "ANSWER_READY" if result.status == "matched" else "UNCERTAIN"
        logger.info("Pipeline answer: %s (Confidence: %.2f from %s) in %.1f ms",
                    result.card_name, result.confidence, result.matched_from, result.processing_time_ms)

        return result

    def confirm_last_answer(self, notes: str = "") -> Optional[int]:
        """
        Confirms the last predicted answer into permanent learned blur memory.
        """
        if not self.last_result or self.last_result.card_id is None:
            logger.warning("No valid card in last result to confirm.")
            return None

        if self.last_result.captured_image is None:
            logger.warning("No captured image saved in last result.")
            return None

        sid = self.memory.add_confirmed_sample(
            card_id=self.last_result.card_id,
            card_name=self.last_result.card_name,
            blurred_image=self.last_result.captured_image,
            embedding=self.last_result.query_embedding,
            confidence=self.last_result.confidence,
            notes=notes
        )
        logger.info("Confirmed last answer into memory: Sample ID %d", sid)
        return sid

    def correct_last_answer(self, correct_card_name: str) -> Optional[int]:
        """
        Manually corrects the last card identity and confirms it into memory.
        """
        if not self.last_result or self.last_result.captured_image is None:
            logger.warning("No last result to correct.")
            return None

        idx = self.matcher.find_card_index(correct_card_name)
        if idx is None:
            logger.error("Could not find card '%s' in database.", correct_card_name)
            return None

        correct_id = int(self.matcher.card_ids[idx])
        correct_canonical = self.matcher.card_names[idx]

        sid = self.memory.add_confirmed_sample(
            card_id=correct_id,
            card_name=correct_canonical,
            blurred_image=self.last_result.captured_image,
            embedding=self.last_result.query_embedding,
            confidence=1.0,
            notes=f"Corrected from {self.last_result.card_name}"
        )
        logger.info("Saved manual correction into memory: %s (#%d), Sample %d",
                    correct_canonical, correct_id, sid)
        return sid

