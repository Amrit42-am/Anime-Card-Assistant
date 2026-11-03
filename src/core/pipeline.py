"""
src/core/pipeline.py

Phase 9 & 13: Main Application Pipeline.
Coordinates ScreenCapture, QuestionDetector, OptionReader, ImageEncoder,
ImageMatcher, BlurMemorySystem, AnswerDecisionEngine, ResultDetector,
and AutomaticSampleCollector.

Manages the full quiz lifecycle:
IDLE -> ACTIVE_QUESTION (live sub-100ms prediction) -> RESULT_REVEALED (auto-ground-truth learning) -> NEXT_QUESTION
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple

from src.capture.region_manager import CalibrationConfig
from src.capture.screen_capture import ScreenCapture
from src.detection.question_detector import QuestionDetector, QuestionEvent
from src.ocr.option_reader import OptionReader
from src.recognition.image_encoder import BaseImageEncoder, get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.core.decision_engine import AnswerDecisionEngine, AnswerResult
from src.learning.result_detector import ResultDetector, ResultDetectionEvent
from src.learning.sample_collector import AutomaticSampleCollector, CollectionReport

logger = logging.getLogger("pipeline")


class QuizPipeline:
    """End-to-end processing pipeline for Roblox card recognition and automatic learning."""

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        encoder_name: str = "dinov2",
        auto_init_models: bool = True
    ):
        self.calibration = calibration or CalibrationConfig.load()
        self.state = "IDLE"
        self.last_result: Optional[AnswerResult] = None
        self.last_collection_report: Optional[CollectionReport] = None

        logger.info("Initializing pipeline modules...")
        self.capture = ScreenCapture(calibration=self.calibration)
        self.detector = QuestionDetector()
        self.option_reader = OptionReader()
        self.result_detector = ResultDetector(option_reader=self.option_reader)

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
            self.sample_collector = AutomaticSampleCollector(
                memory=self.memory,
                matcher=self.matcher,
                encoder=self.encoder
            )
        else:
            self.encoder = None
            self.matcher = None
            self.memory = None
            self.decision_engine = None
            self.sample_collector = None

        logger.info("QuizPipeline successfully initialized with automatic result learning.")

    def step(self) -> Tuple[Optional[AnswerResult], Optional[CollectionReport]]:
        """
        Executes a single monitoring step:
        1. Capture screen bundle.
        2. Advance QuestionDetector state machine.
        3. If ACTIVE_QUESTION: run live candidate prediction immediately (<100ms) and buffer question.
        4. If RESULT_REVEALED: run result detector, verify green button + clear artwork, and auto-learn.
        Returns (AnswerResult or None, CollectionReport or None).
        """
        bundle = self.capture.capture_quiz_bundle(self.calibration)
        event: QuestionEvent = self.detector.process_frame_bundle(bundle)

        answer_res: Optional[AnswerResult] = None
        collection_rep: Optional[CollectionReport] = None

        # 1. State: ACTIVE_QUESTION -> Deliver live prediction immediately
        if event.question_detected:
            # Safely close or expire any pending session from a previous question
            if self.sample_collector:
                self.sample_collector.expire_pending_question("new_question_started")

            self.state = "ACTIVE_QUESTION"
            logger.info("Pipeline: New question detected! Commencing sub-100ms candidate analysis...")

            card_frame = event.card_frame
            button_frames = event.button_frames
            options_frame = event.options_frame

            result = self.decision_engine.process_question(
                card_frame=card_frame,
                button_frames=button_frames,
                options_frame=options_frame
            )

            self.last_result = result
            self.state = "ANSWER_READY" if result.status == "matched" else "UNCERTAIN"
            logger.info("Pipeline live prediction: %s (Confidence: %.2f from %s) in %.1f ms",
                        result.card_name, result.confidence, result.matched_from, result.processing_time_ms)

            # Register with sample collector for upcoming result pairing
            if self.sample_collector:
                opt_ids = [self.matcher.find_card_index(name) for name in result.candidates]
                opt_cids = [int(self.matcher.card_ids[idx]) if idx is not None else None for idx in opt_ids]

                self.sample_collector.register_active_question(
                    card_frame=card_frame,
                    option_names=result.candidates,
                    option_ids=opt_cids,
                    predicted_name=result.card_name,
                    predicted_id=result.card_id,
                    card_hash=event.card_hash,
                    timestamp=event.timestamp,
                    blurred_embedding=result.query_embedding
                )

            answer_res = result

        # 2. State: RESULT_REVEALED -> Process ground-truth automatic learning (with retry across subsequent frames)
        elif (event.is_result_detected or self.detector.current_state == "RESULT_REVEALED") and self.sample_collector and self.sample_collector.pending_question is not None:
            self.state = "RESULT_REVEALED"
            logger.debug("Pipeline: Processing result screen frame...")

            res_event: ResultDetectionEvent = self.result_detector.evaluate_screen(
                card_frame=event.card_frame,
                button_frames=event.button_frames,
                options_frame=event.options_frame,
                timer_frame=event.timer_frame,
                banner_frame=event.banner_frame,
                candidate_names=self.sample_collector.pending_question.option_names,
                timestamp=event.timestamp
            )

            report = self.sample_collector.process_result(
                clear_frame=event.card_frame,
                button_frames=event.button_frames,
                result_event=res_event,
                timestamp=event.timestamp
            )

            if report is not None:
                self.last_collection_report = report
                collection_rep = report
                self.detector.notify_sample_verified()
                logger.info("Pipeline auto-learning outcome: %s (Sample ID: %s, Status: %s)",
                            report.message, report.sample_id, report.status)
            else:
                # Keep state as RESULT_REVEALED to retry on subsequent result frames
                logger.debug("Result verification pending; will retry on next frame...")

        else:
            self.state = event.state

        return answer_res, collection_rep

    def confirm_last_answer(self, notes: str = "") -> Optional[int]:
        """Manually confirms the last predicted answer into permanent learned blur memory."""
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
            clear_image=None,
            options=self.last_result.candidates,
            embedding=self.last_result.query_embedding,
            confidence=self.last_result.confidence,
            source="manual_confirmation",
            verification_status="VERIFIED",
            notes=notes
        )
        logger.info("Manually confirmed last answer into memory: Sample ID %s", sid)
        return sid

    def correct_last_answer(self, correct_card_name: str) -> Optional[int]:
        """Manually corrects the last card identity and confirms it into memory."""
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
            clear_image=None,
            options=self.last_result.candidates,
            embedding=self.last_result.query_embedding,
            confidence=1.0,
            source="manual_correction",
            verification_status="VERIFIED",
            notes=f"Corrected from {self.last_result.card_name}"
        )
        logger.info("Saved manual correction into memory: %s (#%d), Sample %s",
                    correct_canonical, correct_id, sid)
        return sid
