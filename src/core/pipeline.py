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
from typing import Optional, Tuple, List
import numpy as np

from src.capture.region_manager import CalibrationConfig
from src.capture.display_monitor import DisplayMonitor
from src.capture.window_tracker import WindowTracker
from src.capture.game_locator import GameLocator
from src.capture.screen_capture import ScreenCapture
from src.detection.question_detector import QuestionDetector, QuestionEvent
from src.ocr.option_reader import OptionReader, CandidateOption, normalize_detected_options
from src.recognition.image_encoder import BaseImageEncoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.core.decision_engine import AnswerDecisionEngine, AnswerResult
from src.learning.result_detector import ResultDetector, ResultDetectionEvent
from src.learning.sample_collector import CollectionReport, AutomaticSampleCollector
from src.memory.self_learning_db import SelfLearningDatabase
from src.ai.model_manager import ModelManager
from src.ai.perception_engine import PerceptionEngine, PerceptionFrame, OCRSource
from src.ai.context_analyzer import ContextAnalyzer
from src.ai.candidate_reasoner import CandidateReasoner, ReasonedAnswer
from src.ai.temporal_analyzer import TemporalAnalyzer
from src.ai.confidence_engine import ConfidenceEngine
from src.ai.adaptive_learning import AdaptiveLearning

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
        self.current_candidates: List[CandidateOption] = []
        self.current_session_id: Optional[str] = None

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
        self.db = SelfLearningDatabase()
        self.option_reader = OptionReader(db=self.db)
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
            # Strictly load verified local offline vision encoder
            self.model_manager = ModelManager.get_instance(require_offline=True)
            if not self.model_manager.is_offline_verified:
                raise FileNotFoundError(
                    f"ModelManager offline verification failed: {self.model_manager.verification_message}. "
                    "Please run: ./prepare.sh"
                )
            self.encoder: BaseImageEncoder = self.model_manager.get_encoder()

            self.matcher = ImageMatcher(db=self.db)
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
                encoder=self.encoder,
                sample_collector=AutomaticSampleCollector(
                    memory=self.memory,
                    matcher=self.matcher,
                    encoder=self.encoder,
                    db=self.db,
                    option_reader=self.option_reader
                )
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
        1. Capture and slice screen via PerceptionEngine (tracked via GameLocator/WindowTracker).
        2. Advance QuestionDetector & ContextAnalyzer state machine.
        3. If ACTIVE_QUESTION: execute CandidateReasoner + TemporalAnalyzer for live prediction & stabilization.
        4. If RESULT_REVEALED: execute ResultDetector + AdaptiveLearning for dual-verification learning.
        Returns (AnswerResult or None, CollectionReport or None).
        """
        t0 = time.perf_counter()
        # 1. PerceptionEngine unified perception entry point
        p_frame: PerceptionFrame = self.perception_engine.perceive_frame()
        event: QuestionEvent = self.detector.process_frame_bundle(p_frame)

        # 2. Advance ContextAnalyzer
        _ = self.context_analyzer.evaluate_transition(
            is_active_question=event.question_detected or (
                self.detector.current_state in ("ACTIVE_QUESTION", "ANSWER_RECOGNITION") and
                not (self.context_analyzer.active_session and self.context_analyzer.active_session.result_processed)
            ),
            is_result_screen=event.is_result_detected or self.detector.current_state in ("RESULT_REVEALED", "SAMPLE_VERIFICATION"),
            card_hash=event.card_hash,
            timestamp=event.timestamp
        )

        answer_res: Optional[AnswerResult] = None
        collection_rep: Optional[CollectionReport] = None

        # 3. State: ACTIVE_QUESTION
        if event.question_detected or self.context_analyzer.should_predict():
            # Fresh question detected!
            if self.adaptive_learning:
                self.adaptive_learning.expire_pending_question("new_question_started")

            sid = self.context_analyzer.active_session.session_id if self.context_analyzer.active_session else f"q_{int(event.timestamp)}"
            self.current_session_id = sid
            self.temporal_analyzer.set_session(sid)
            if p_frame.ocr_source != OCRSource.INVALID:
                self.current_candidates = p_frame.ocr_candidates
            else:
                self.current_candidates = []

            self.state = "ACTIVE_QUESTION"
            logger.info("Pipeline: New question detected (%s)! Commencing sub-100ms candidate analysis...", sid)

            card_frame = p_frame.card_frame
            button_frames = p_frame.button_frames
            options_frame = p_frame.options_frame

            result = self.decision_engine.process_question(
                card_frame=card_frame,
                button_frames=button_frames,
                options_frame=options_frame,
                ocr_candidates=self.current_candidates,
                prediction_stage="PROVISIONAL",
                observation_count=1,
                temporal_agreement=1.0
            )

            if hasattr(self.perception_engine, "notify_inference_completed"):
                self.perception_engine.notify_inference_completed(p_frame.card_hash, p_frame.card_sharpness)

            # Push initial snapshot to temporal analyzer
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
                raw_match=result.to_dict(),
                query_embedding=result.query_embedding,
                captured_image=result.captured_image
            )
            self.temporal_analyzer.push(
                answer=reasoned_proxy,
                session_id=sid,
                timer_val=p_frame.timer_value,
                timestamp=event.timestamp
            )

            self.last_result = result
            has_four_readable = (
                len(self.current_candidates) == 4 and
                all(c.is_valid and (c.canonical_name or c.raw_text) for c in self.current_candidates)
            )
            has_four_known = (
                has_four_readable and
                all(c.card_id is not None for c in self.current_candidates) and
                len(set(c.card_id for c in self.current_candidates)) == 4
            )

            if has_four_known and result.status == "matched":
                self.state = "ANSWER_READY"
            elif has_four_readable or (result.candidates and len([c for c in result.candidates if c]) == 4):
                self.state = "PROVISIONAL_GUESS" if getattr(result, "is_constrained", False) else "UNCERTAIN"
            else:
                self.state = "WAITING FOR VALID OPTIONS"

            logger.info("Pipeline live prediction: %s (Slot: %s, Confidence: %.2f from %s) in %.1f ms [%s]",
                        result.card_name, result.button_slot, result.confidence, result.matched_from, result.processing_time_ms, result.prediction_stage)

            # Mark predicted in context analyzer
            if (has_four_known or has_four_readable) and result.card_name:
                self.context_analyzer.mark_predicted(result.card_name, result.button_slot if result.button_slot is not None else 0)
            else:
                self.context_analyzer.mark_ocr_status(is_valid=False)

            # Register with adaptive learning for upcoming result pairing whenever 4 options are extracted
            if (has_four_readable or len(result.candidates) == 4) and self.adaptive_learning:
                opt_ids = [self.matcher.find_card_index(name) if self.matcher else None for name in result.candidates]
                opt_cids = [int(self.matcher.card_ids[idx]) if (idx is not None and self.matcher and idx < len(self.matcher.card_ids)) else None for idx in opt_ids]

                self.adaptive_learning.register_question(
                    card_frame=card_frame,
                    option_names=result.candidates,
                    option_ids=opt_cids,
                    predicted_name=result.card_name,
                    predicted_id=result.card_id,
                    card_hash=event.card_hash,
                    timestamp=event.timestamp,
                    predicted_slot=result.button_slot,
                    predicted_confidence=result.confidence
                )

            answer_res = result

        elif (
            self.context_analyzer.is_in_active_question() and
            not event.is_result_detected and
            self.detector.current_state not in ("RESULT_REVEALED", "SAMPLE_VERIFICATION")
        ):
            # Subsequent active-question frames: OCR synchronization & adaptive visual inference
            sid = self.context_analyzer.active_session.session_id if self.context_analyzer.active_session else self.current_session_id

            # 1. OCR candidate synchronization on grid change or recovery
            if p_frame.ocr_source in (OCRSource.REFRESHED, OCRSource.NEWLY_EXTRACTED):
                if p_frame.ocr_candidates and p_frame.ocr_candidates != self.current_candidates:
                    logger.info("Pipeline: PerceptionEngine signaled %s candidates; updating pipeline candidates.", p_frame.ocr_source.value)
                    self.current_candidates = list(p_frame.ocr_candidates)
                    has_now_four = (
                        len(self.current_candidates) == 4 and
                        all(c.is_valid and (c.canonical_name or c.raw_text) for c in self.current_candidates)
                    )
                    # Safely register or update pending question candidate names and IDs in adaptive_learning
                    if has_now_four and self.adaptive_learning:
                        opt_names = [c.canonical_name if c.canonical_name else c.raw_text for c in self.current_candidates]
                        opt_ids = [self.matcher.find_card_index(name) if self.matcher else None for name in opt_names]
                        opt_cids = [int(self.matcher.card_ids[idx]) if (idx is not None and self.matcher and idx < len(self.matcher.card_ids)) else None for idx in opt_ids]
                        if self.adaptive_learning.pending_question is None:
                            self.adaptive_learning.register_question(
                                card_frame=p_frame.card_frame if (p_frame.card_frame is not None and p_frame.card_frame.size > 0) else np.zeros((10, 10, 3), dtype=np.uint8),
                                option_names=opt_names,
                                option_ids=opt_cids,
                                predicted_name=self.last_result.card_name if self.last_result else "",
                                predicted_id=self.last_result.card_id if self.last_result else None,
                                card_hash=p_frame.card_hash,
                                timestamp=event.timestamp,
                                predicted_slot=self.last_result.button_slot if self.last_result else None,
                                predicted_confidence=self.last_result.confidence if self.last_result else 0.0
                            )
                        else:
                            self.adaptive_learning.update_pending_candidates(opt_names, opt_cids)
            elif p_frame.ocr_source == OCRSource.INVALID:
                logger.warning("Pipeline: PerceptionEngine signaled INVALID OCR on grid change; invalidating stale candidates.")
                self.current_candidates = []
                if self.adaptive_learning:
                    self.adaptive_learning.invalidate_pending_candidates()

            cands = self.current_candidates if (self.current_candidates and p_frame.ocr_source != OCRSource.INVALID) else (
                p_frame.ocr_candidates if p_frame.ocr_source != OCRSource.INVALID else []
            )

            # 2. Adaptive Visual Inference Check
            # If card region has NOT materially changed, options are cached, and we already hold a prediction:
            # Skip expensive DINOv2 inference and do NOT artificially increment temporal observation count!
            if (
                not p_frame.has_card_changed and
                p_frame.ocr_source == OCRSource.FROM_CACHE and
                self.last_result is not None
            ):
                stab_ans, agreement_ratio, obs_count = self.temporal_analyzer.get_stabilized_answer(
                    session_id=sid,
                    return_details=True
                )
                step_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if obs_count < 2 or stab_ans is None:
                    logger.debug("Pipeline: Card artwork and options visually unchanged; skipping DINOv2 inference (took %.2f ms).", step_elapsed_ms)
                    return self.last_result, None

                # Temporal observations already exist (>=2); evaluate stabilization without re-running DINOv2
                is_constrained = (len(cands) == 4 and len(set(c.card_id for c in cands if c.card_id)) == 4)
                verdict = self.confidence_engine.evaluate(
                    top_score=stab_ans.raw_match.get("raw_score", stab_ans.confidence),
                    runner_up_score=stab_ans.runner_up_score or 0.0,
                    candidate_constrained=is_constrained,
                    source=stab_ans.source,
                    ocr_quality=self.last_result.confidence,
                    candidate_count=len(cands),
                    temporal_agreement=agreement_ratio
                )

                if obs_count >= 3 and agreement_ratio >= 0.70 and verdict.is_confident and stab_ans.slot_index is not None:
                    stage = "CONFIDENT"
                    status = "matched"
                    final_conf = verdict.calibrated_confidence
                elif agreement_ratio >= 0.60 and verdict.is_confident and stab_ans.slot_index is not None:
                    stage = "STABILIZING"
                    status = "matched"
                    final_conf = verdict.calibrated_confidence
                else:
                    stage = "UNCERTAIN"
                    status = "uncertain"
                    final_conf = min(verdict.calibrated_confidence, 0.40)

                normalized_cands = normalize_detected_options(cands)
                canonical_cands = [opt["canonical_name"] if opt["is_valid"] else "" for opt in normalized_cands]

                stab_result = AnswerResult(
                    status=status,
                    card_name=stab_ans.card_name,
                    card_id=stab_ans.card_id,
                    confidence=final_conf,
                    confidence_level=verdict.confidence_level if status == "matched" else "UNCERTAIN",
                    matched_from=stab_ans.source,
                    raw_score=stab_ans.raw_match.get("raw_score", stab_ans.confidence),
                    margin=verdict.margin_of_victory,
                    candidates=canonical_cands,
                    alternatives=stab_ans.raw_match.get("alternatives", []),
                    explanation=f"{verdict.explanation} (Obs: {obs_count}, Agreement: {agreement_ratio*100:.0f}%)",
                    processing_time_ms=round(step_elapsed_ms, 2),
                    button_slot=stab_ans.slot_index if is_constrained else None,
                    runner_up_name=stab_ans.runner_up_name,
                    runner_up_score=stab_ans.runner_up_score,
                    captured_image=self.last_result.captured_image,
                    query_embedding=stab_ans.query_embedding,
                    prediction_stage=stage,
                    temporal_agreement=round(agreement_ratio, 3),
                    observation_count=obs_count,
                    detected_options=normalized_cands,
                    visual_similarity=float(stab_ans.raw_match.get("raw_score", stab_ans.confidence)),
                    is_constrained=is_constrained
                )
                self.last_result = stab_result
                return stab_result, None

            if p_frame.card_frame is not None and p_frame.card_frame.size > 0:
                frame_result = self.decision_engine.process_question(
                    card_frame=p_frame.card_frame,
                    button_frames=p_frame.button_frames,
                    options_frame=p_frame.options_frame,
                    ocr_candidates=cands,
                    prediction_stage="STABILIZING"
                )

                if hasattr(self.perception_engine, "notify_inference_completed"):
                    self.perception_engine.notify_inference_completed(p_frame.card_hash, p_frame.card_sharpness)

                frame_proxy = ReasonedAnswer(
                    card_name=frame_result.card_name,
                    card_id=frame_result.card_id,
                    slot_index=frame_result.button_slot,
                    confidence=frame_result.confidence,
                    confidence_level=frame_result.confidence_level,
                    margin=frame_result.margin,
                    is_confident=(frame_result.status == "matched"),
                    runner_up_name=frame_result.runner_up_name,
                    runner_up_score=frame_result.runner_up_score or 0.0,
                    source=frame_result.matched_from,
                    status=frame_result.status,
                    raw_match=frame_result.to_dict(),
                    query_embedding=frame_result.query_embedding,
                    captured_image=frame_result.captured_image
                )

                self.temporal_analyzer.push(
                    answer=frame_proxy,
                    session_id=sid,
                    timer_val=p_frame.timer_value,
                    timestamp=event.timestamp
                )

                # Retrieve temporally stabilized answer
                stab_ans, agreement_ratio, obs_count = self.temporal_analyzer.get_stabilized_answer(
                    session_id=sid,
                    return_details=True
                )

                if stab_ans is not None and obs_count >= 2:
                    is_constrained = (len(cands) == 4 and len(set(c.card_id for c in cands if c.card_id)) == 4)
                    verdict = self.confidence_engine.evaluate(
                        top_score=stab_ans.raw_match.get("raw_score", stab_ans.confidence),
                        runner_up_score=stab_ans.runner_up_score or 0.0,
                        candidate_constrained=is_constrained,
                        source=stab_ans.source,
                        ocr_quality=frame_result.confidence,
                        candidate_count=len(cands),
                        temporal_agreement=agreement_ratio
                    )

                    if obs_count >= 3 and agreement_ratio >= 0.70 and verdict.is_confident and stab_ans.slot_index is not None:
                        stage = "CONFIDENT"
                        status = "matched"
                        final_conf = verdict.calibrated_confidence
                    elif agreement_ratio >= 0.60 and verdict.is_confident and stab_ans.slot_index is not None:
                        stage = "STABILIZING"
                        status = "matched"
                        final_conf = verdict.calibrated_confidence
                    else:
                        stage = "UNCERTAIN"
                        status = "uncertain"
                        final_conf = min(verdict.calibrated_confidence, 0.40)

                    normalized_cands = normalize_detected_options(cands)
                    canonical_cands = [opt["canonical_name"] if opt["is_valid"] else "" for opt in normalized_cands]

                    stab_result = AnswerResult(
                        status=status,
                        card_name=stab_ans.card_name,
                        card_id=stab_ans.card_id,
                        confidence=final_conf,
                        confidence_level=verdict.confidence_level if status == "matched" else "UNCERTAIN",
                        matched_from=stab_ans.source,
                        raw_score=stab_ans.raw_match.get("raw_score", stab_ans.confidence),
                        margin=verdict.margin_of_victory,
                        candidates=canonical_cands,
                        alternatives=stab_ans.raw_match.get("alternatives", []),
                        explanation=f"{verdict.explanation} (Obs: {obs_count}, Agreement: {agreement_ratio*100:.0f}%)",
                        processing_time_ms=frame_result.processing_time_ms,
                        button_slot=stab_ans.slot_index if is_constrained else None,
                        runner_up_name=stab_ans.runner_up_name,
                        runner_up_score=stab_ans.runner_up_score,
                        captured_image=p_frame.card_frame.copy(),
                        query_embedding=stab_ans.query_embedding,
                        prediction_stage=stage,
                        temporal_agreement=round(agreement_ratio, 3),
                        observation_count=obs_count,
                        detected_options=normalized_cands,
                        visual_similarity=float(stab_ans.raw_match.get("raw_score", stab_ans.confidence)),
                        is_constrained=is_constrained
                    )

                    self.last_result = stab_result
                    answer_res = stab_result
                    if stab_result.status == "matched" and stab_result.button_slot is not None:
                        self.context_analyzer.mark_predicted(stab_result.card_name, stab_result.button_slot)

        # 4. State: RESULT_REVEALED -> Process ground-truth automatic learning
        elif (
            (event.is_result_detected or self.detector.current_state in ("RESULT_REVEALED", "SAMPLE_VERIFICATION") or self.context_analyzer.should_process_result()) and
            self.adaptive_learning and
            self.adaptive_learning.pending_question is not None
        ):
            self.state = "RESULT_REVEALED"
            self.context_analyzer.mark_verifying()
            logger.debug("Pipeline: Processing result screen frame...")

            res_event: ResultDetectionEvent = self.result_detector.evaluate_screen(
                card_frame=p_frame.card_frame,
                button_frames=p_frame.button_frames,
                options_frame=p_frame.options_frame,
                timer_frame=p_frame.timer_frame,
                banner_frame=p_frame.banner_frame,
                candidate_names=self.adaptive_learning.pending_question.option_names,
                timestamp=event.timestamp
            )

            report = self.adaptive_learning.process_result(
                clear_frame=p_frame.card_frame,
                button_frames=p_frame.button_frames,
                result_event=res_event,
                timestamp=event.timestamp
            )

            if report is not None:
                self.last_collection_report = report
                collection_rep = report
                self.detector.notify_sample_verified()
                self.context_analyzer.mark_result_completed()
                self.current_candidates = []
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

    @property
    def is_offline_verified(self) -> bool:
        """Returns True only when ModelManager successfully verified local checkpoints and embeddings."""
        if self.model_manager is not None:
            return bool(self.model_manager.is_offline_verified)
        return False

    def reset(self):
        """Resets the pipeline, detectors, context sessions, and perception caches."""
        self.state = "SEARCHING"
        self.current_candidates = []
        self.current_session_id = None
        self.last_result = None
        self.detector.reset()
        self.context_analyzer.reset()
        self.temporal_analyzer.reset()
        if hasattr(self.perception_engine, "reset_cache"):
            self.perception_engine.reset_cache()
        if self.adaptive_learning and hasattr(self.adaptive_learning, "collector"):
            self.adaptive_learning.collector.pending_question = None
            self.adaptive_learning.collector.last_learned_session = None

