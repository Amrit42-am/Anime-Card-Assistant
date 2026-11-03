"""
src/core/pipeline.py

Phase 9 & 13: Main Autonomous AI Application Pipeline.
Coordinates:
DisplayMonitor -> WindowTracker -> GameLocator -> ScreenCapture ->
PerceptionEngine -> QuestionDetector -> ContextAnalyzer ->
CandidateReasoner -> TemporalAnalyzer -> ConfidenceEngine -> AnswerResult.

Result verification & learning lifecycle:
ResultDetector -> AdaptiveLearning -> Dual Verification -> Memory Update.
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple

from src.capture.region_manager import CalibrationConfig
from src.capture.display_monitor import DisplayMonitor
from src.capture.window_tracker import WindowTracker
from src.capture.game_locator import GameLocator
from src.capture.screen_capture import ScreenCapture
from src.detection.question_detector import QuestionDetector, QuestionEvent
from src.ocr.option_reader import OptionReader
from src.recognition.image_encoder import BaseImageEncoder, get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.core.decision_engine import AnswerDecisionEngine, AnswerResult
from src.learning.result_detector import ResultDetector, ResultDetectionEvent
from src.learning.sample_collector import AutomaticSampleCollector, CollectionReport
from src.ai.model_manager import ModelManager
from src.ai.perception_engine import PerceptionEngine, PerceptionFrame
from src.ai.context_analyzer import ContextAnalyzer, QuizState
from src.ai.candidate_reasoner import CandidateReasoner, ReasonedAnswer
from src.ai.temporal_analyzer import TemporalAnalyzer
from src.ai.confidence_engine import ConfidenceEngine
from src.ai.adaptive_learning import AdaptiveLearning, LearningOutcome

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

        logger.info("Initializing autonomous AI pipeline modules...")
        self.display_monitor = DisplayMonitor()
        self.window_tracker = WindowTracker()
        self.game_locator = GameLocator(
            calibration=self.calibration,
            display_monitor=self.display_monitor,
            window_tracker=self.window_tracker
        )
        self.capture = ScreenCapture(
            calibration=self.calibration,
            game_locator=self.game_locator
        )
        self.option_reader = OptionReader()
        self.perception_engine = PerceptionEngine(
            screen_capture=self.capture,
            game_locator=self.game_locator,
            option_reader=self.option_reader
        )
        self.detector = QuestionDetector()
        self.context_analyzer = ContextAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.confidence_engine = ConfidenceEngine()
        self.result_detector = ResultDetector(option_reader=self.option_reader)

        if auto_init_models:
            # Load local offline vision encoder
            try:
                self.model_manager = ModelManager.get_instance()
                self.encoder: BaseImageEncoder = self.model_manager.get_encoder()
            except Exception as e:
                logger.warning("ModelManager initialization notice (%s), falling back to get_image_encoder", e)
                self.encoder: BaseImageEncoder = get_image_encoder(encoder_name)

            self.matcher = ImageMatcher()
            self.memory = BlurMemorySystem(encoder=self.encoder)

            self.candidate_reasoner = CandidateReasoner(
                image_matcher=self.matcher,
                image_encoder=self.encoder,
                option_reader=self.option_reader,
                confidence_engine=self.confidence_engine
            )

            self.decision_engine = AnswerDecisionEngine(
                matcher=self.matcher,
                encoder=self.encoder,
                option_reader=self.option_reader,
                memory=self.memory,
                confidence_engine=self.confidence_engine,
                candidate_reasoner=self.candidate_reasoner
            )

            self.adaptive_learning = AdaptiveLearning(
                memory=self.memory,
                matcher=self.matcher,
                encoder=self.encoder
            )
            self.sample_collector = self.adaptive_learning.collector
        else:
            self.model_manager = None
            self.encoder = None
            self.matcher = None
            self.memory = None
            self.candidate_reasoner = None
            self.decision_engine = None
            self.adaptive_learning = None
            self.sample_collector = None

        logger.info("QuizPipeline successfully initialized with autonomous AI architecture.")

    def step(self) -> Tuple[Optional[AnswerResult], Optional[CollectionReport]]:
        """
        Executes a single monitoring step:
        1. Capture screen bundle (tracked via GameLocator/WindowTracker).
        2. Advance QuestionDetector & ContextAnalyzer state machine.
        3. If ACTIVE_QUESTION: execute CandidateReasoner + TemporalAnalyzer for live prediction.
        4. If RESULT_REVEALED: execute ResultDetector + AdaptiveLearning for dual-verification learning.
        Returns (AnswerResult or None, CollectionReport or None).
        """
        # 1. Screen capture bundle (uses automatic Roblox window tracking)
        bundle = self.capture.capture_quiz_bundle(self.calibration)
        event: QuestionEvent = self.detector.process_frame_bundle(bundle)

        # 2. Advance ContextAnalyzer
        state = self.context_analyzer.evaluate_transition(
            is_active_question=event.question_detected or (self.detector.current_state == "ACTIVE_QUESTION" and not (self.context_analyzer.active_session and self.context_analyzer.active_session.has_predicted)),
            is_result_screen=event.is_result_detected or self.detector.current_state == "RESULT_REVEALED",
            card_hash=event.card_hash,
            timestamp=event.timestamp
        )

        answer_res: Optional[AnswerResult] = None
        collection_rep: Optional[CollectionReport] = None

        # 3. State: ACTIVE_QUESTION -> Deliver live prediction immediately
        if event.question_detected or self.context_analyzer.should_predict():
            # Safely close or expire any pending session from a previous question
            if self.adaptive_learning:
                self.adaptive_learning.expire_pending_question("new_question_started")

            sid = self.context_analyzer.active_session.session_id if self.context_analyzer.active_session else f"q_{int(event.timestamp)}"
            self.temporal_analyzer.set_session(sid)

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

            # Push to temporal analyzer for multi-frame stabilization
            reasoned_proxy = ReasonedAnswer(
                card_name=result.card_name,
                card_id=result.card_id,
                slot_index=result.button_slot,
                confidence=result.confidence,
                confidence_level=result.confidence_level,
                margin=result.margin,
                is_confident=(result.status == "matched"),
                runner_up_name=result.runner_up_name,
                runner_up_score=result.runner_up_score or 0.0,
                source=result.matched_from,
                status=result.status,
                raw_match=result.to_dict()
            )
            self.temporal_analyzer.push(
                answer=reasoned_proxy,
                session_id=sid,
                timer_val=None,
                timestamp=event.timestamp
            )

            self.last_result = result
            self.state = "ANSWER_READY" if result.status == "matched" else "UNCERTAIN"
            logger.info("Pipeline live prediction: %s (Slot: %s, Confidence: %.2f from %s) in %.1f ms",
                        result.card_name, result.button_slot, result.confidence, result.matched_from, result.processing_time_ms)

            # Mark predicted in context analyzer
            self.context_analyzer.mark_predicted(result.card_name, result.button_slot)

            # Register with adaptive learning for upcoming result pairing
            if self.adaptive_learning and self.matcher:
                opt_ids = [self.matcher.find_card_index(name) for name in result.candidates]
                opt_cids = [int(self.matcher.card_ids[idx]) if idx is not None else None for idx in opt_ids]

                self.adaptive_learning.register_question(
                    card_frame=card_frame,
                    option_names=result.candidates,
                    option_ids=opt_cids,
                    predicted_name=result.card_name,
                    predicted_id=result.card_id,
                    card_hash=event.card_hash,
                    timestamp=event.timestamp
                )

            answer_res = result

        # 4. State: RESULT_REVEALED -> Process ground-truth automatic learning
        elif (event.is_result_detected or self.detector.current_state == "RESULT_REVEALED" or self.context_analyzer.should_process_result()) and self.sample_collector and self.sample_collector.pending_question is not None:
            self.state = "RESULT_REVEALED"
            self.context_analyzer.mark_verifying()
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
                self.context_analyzer.mark_result_completed()
                logger.info("Pipeline auto-learning outcome: %s (Sample ID: %s, Status: %s)",
                            report.message, report.sample_id, report.status)
            else:
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
