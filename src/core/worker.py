"""
src/core/worker.py

Phase 9: Background Worker Thread.
Runs screen capture and quiz evaluation in a non-blocking background thread
using PySide6 QThread to ensure the floating overlay UI remains smooth and responsive.
"""

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.core.pipeline import QuizPipeline
from src.core.decision_engine import AnswerResult

logger = logging.getLogger("worker")


class PipelineWorker(QThread):
    """Background worker for continuous screen monitoring and quiz recognition."""

    sig_status = Signal(str)            # Current pipeline state string
    sig_fps = Signal(float)             # Current monitoring FPS
    sig_answer_ready = Signal(object)   # Emits AnswerResult when high confidence
    sig_uncertain = Signal(object)      # Emits AnswerResult when ambiguous
    sig_error = Signal(str)             # Emits error description
    sig_memory_updated = Signal(int)    # Emits total confirmed blur sample count

    def __init__(self, pipeline: QuizPipeline, target_fps: float = 8.0, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.target_fps = target_fps
        self.interval = 1.0 / target_fps
        self._running = False
        self._paused = False

    def run(self):
        self._running = True
        logger.info("PipelineWorker started monitoring at %.1f target FPS.", self.target_fps)
        self.sig_status.emit("MONITORING")

        fps_counter = 0
        fps_start = time.time()

        while self._running:
            loop_start = time.time()

            if self._paused:
                self.sig_status.emit("PAUSED")
                time.sleep(0.1)
                continue

            try:
                result: Optional[AnswerResult] = self.pipeline.step()

                if result is not None:
                    if result.status == "matched":
                        self.sig_status.emit("ANSWER_READY")
                        self.sig_answer_ready.emit(result)
                    else:
                        self.sig_status.emit("UNCERTAIN")
                        self.sig_uncertain.emit(result)
                else:
                    self.sig_status.emit(self.pipeline.state)

            except Exception as e:
                logger.error("Error in pipeline worker step: %s", e, exc_info=True)
                self.sig_error.emit(str(e))
                self.sig_status.emit("ERROR")

            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                self.sig_fps.emit(round(current_fps, 1))
                fps_counter = 0
                fps_start = time.time()

            # Maintain target loop rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.01, self.interval - elapsed)
            time.sleep(sleep_time)

        logger.info("PipelineWorker stopped.")
        self.sig_status.emit("STOPPED")

    def pause(self):
        self._paused = True
        logger.info("Worker paused.")

    def resume(self):
        self._paused = False
        logger.info("Worker resumed.")

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def stop(self):
        self._running = False
        self.wait(1500)

