"""
src/ui/overlay.py

Phase 8: Modern Floating Desktop Overlay Assistant.
Frameless, always-on-top, movable PySide6 window positioned on the right
side of the desktop to display real-time card predictions without obstructing gameplay.
"""

import logging
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMouseEvent, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QSlider, QGraphicsDropShadowEffect, QMessageBox
)

from src.core.pipeline import QuizPipeline
from src.core.worker import PipelineWorker
from src.core.decision_engine import AnswerResult
from src.capture.calibration import CalibrationDialog
from src.ui.components import ConfidenceBadge, SourceBadge, CorrectionDialog

logger = logging.getLogger("overlay_ui")


class FloatingOverlay(QWidget):
    """Floating transparent desktop assistant panel."""

    def __init__(self, pipeline: QuizPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.drag_position = QPoint()
        self.worker: Optional[PipelineWorker] = None
        self.last_result: Optional[AnswerResult] = None

        self.init_window_flags()
        self.init_ui()
        self.position_on_screen()
        self.init_worker()

    def init_window_flags(self):
        # Frameless, Always on Top, floating desktop window without stealing focus
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedWidth(380)

    def position_on_screen(self):
        """Positions overlay neatly on the right side of the screen."""
        screen = self.screen().geometry()
        x = screen.width() - self.width() - 30
        y = max(60, (screen.height() - 520) // 2)
        self.move(x, y)

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # Container Frame with dark gradient theme
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setStyleSheet("""
            QFrame#container {
                background-color: rgba(22, 25, 34, 240);
                border: 1px solid rgba(75, 85, 105, 180);
                border-radius: 12px;
            }
            QLabel { color: #f1f5f9; font-family: -apple-system, sans-serif; }
            QPushButton {
                background-color: #2e3646;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 10px;
                color: #e2e8f0;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3e485b; }
            QPushButton#confirmBtn { background-color: #1e3a2b; border-color: #10b981; color: #a7f3d0; }
            QPushButton#confirmBtn:hover { background-color: #14532d; }
            QPushButton#correctBtn { background-color: #3b2818; border-color: #f59e0b; color: #fde68a; }
            QPushButton#correctBtn:hover { background-color: #78350f; }
        """)

        # Drop shadow for sleek float effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 4)
        self.container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # 1. Header Bar (Draggable)
        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel("ANIME CARD ASSISTANT")
        title.setStyleSheet("font-weight: 800; font-size: 13px; letter-spacing: 0.5px; color: #38bdf8;")
        header.addWidget(title)
        header.addStretch()

        self.live_indicator = QLabel("● LIVE")
        self.live_indicator.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 11px;")
        header.addWidget(self.live_indicator)

        self.offline_indicator = QLabel("🔒 OFFLINE")
        self.offline_indicator.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 10px; background: rgba(56, 189, 248, 0.15); border-radius: 3px; padding: 2px 4px;")
        header.addWidget(self.offline_indicator)

        self.btn_compact = QPushButton("▾ Compact")
        self.btn_compact.setFixedHeight(22)
        self.btn_compact.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_compact.clicked.connect(self.toggle_compact_mode)
        header.addWidget(self.btn_compact)

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.setStyleSheet("background: transparent; border: none; font-size: 12px; color: #94a3b8;")
        btn_close.clicked.connect(self.close)
        header.addWidget(btn_close)
        layout.addLayout(header)

        # Divider
        div1 = QFrame()
        div1.setFrameShape(QFrame.HLine)
        div1.setStyleSheet("background-color: #334155;")
        layout.addWidget(div1)

        # 2. Hero Prediction Area
        hero_layout = QVBoxLayout()
        hero_layout.setSpacing(6)

        self.state_label = QLabel("MONITORING ROBLOX...")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold; letter-spacing: 0.8px;")
        hero_layout.addWidget(self.state_label)

        self.card_name_label = QLabel("Waiting for quiz...")
        self.card_name_label.setAlignment(Qt.AlignCenter)
        self.card_name_label.setWordWrap(True)
        self.card_name_label.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc; padding: 4px 0;")
        hero_layout.addWidget(self.card_name_label)

        # Badges row
        badges_row = QHBoxLayout()
        badges_row.setSpacing(6)
        self.conf_badge = ConfidenceBadge()
        self.source_badge = SourceBadge()
        badges_row.addWidget(self.conf_badge)
        badges_row.addWidget(self.source_badge)
        hero_layout.addLayout(badges_row)

        self.slot_badge = QLabel("")
        self.slot_badge.setAlignment(Qt.AlignCenter)
        self.slot_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold; background: rgba(56, 189, 248, 0.15); border-radius: 4px; padding: 2px 6px;")
        self.slot_badge.setVisible(False)
        hero_layout.addWidget(self.slot_badge)

        self.runner_up_label = QLabel("Runner-up: —")
        self.runner_up_label.setAlignment(Qt.AlignCenter)
        self.runner_up_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        hero_layout.addWidget(self.runner_up_label)

        self.latency_label = QLabel("Latency: —")
        self.latency_label.setAlignment(Qt.AlignCenter)
        self.latency_label.setStyleSheet("color: #64748b; font-size: 10px;")
        hero_layout.addWidget(self.latency_label)

        # Auto-learned notification banner
        self.auto_learn_banner = QLabel("")
        self.auto_learn_banner.setAlignment(Qt.AlignCenter)
        self.auto_learn_banner.setWordWrap(True)
        self.auto_learn_banner.setVisible(False)
        hero_layout.addWidget(self.auto_learn_banner)

        layout.addLayout(hero_layout)

        # 3. Candidate Options Box (2x2 Grid)
        self.cand_frame = QFrame()
        self.cand_frame.setStyleSheet("background-color: rgba(15, 23, 42, 140); border-radius: 8px; padding: 6px;")
        self.cand_layout = QGridLayout(self.cand_frame)
        self.cand_layout.setContentsMargins(6, 6, 6, 6)
        self.cand_layout.setSpacing(6)

        slot_names = ["[TL]", "[TR]", "[BL]", "[BR]"]
        self.cand_labels = []
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (r, c) in enumerate(positions):
            lbl = QLabel(f"{slot_names[i]} —")
            lbl.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 6px 8px; background: rgba(30, 41, 59, 120); border: 1px solid rgba(71, 85, 105, 100); border-radius: 6px;")
            lbl.setWordWrap(True)
            self.cand_layout.addWidget(lbl, r, c)
            self.cand_labels.append(lbl)

        layout.addWidget(self.cand_frame)

        # 4. Learning & Feedback Actions
        self.action_frame = QFrame()
        action_layout = QHBoxLayout(self.action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)

        self.btn_confirm = QPushButton("✓ Confirm")
        self.btn_confirm.setObjectName("confirmBtn")
        self.btn_confirm.setToolTip("Save this confirmed blur into permanent memory.")
        self.btn_confirm.clicked.connect(self.on_confirm_answer)
        action_layout.addWidget(self.btn_confirm)

        self.btn_correct = QPushButton("✎ Correct")
        self.btn_correct.setObjectName("correctBtn")
        self.btn_correct.setToolTip("Select correct card if prediction was wrong.")
        self.btn_correct.clicked.connect(self.on_correct_answer)
        action_layout.addWidget(self.btn_correct)

        self.btn_pause = QPushButton("❚❚ Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        action_layout.addWidget(self.btn_pause)

        self.btn_calibrate = QPushButton("⚙ Calibrate")
        self.btn_calibrate.clicked.connect(self.open_calibration)
        action_layout.addWidget(self.btn_calibrate)

        layout.addWidget(self.action_frame)

        # 5. Opacity slider row
        self.opacity_frame = QFrame()
        op_layout = QHBoxLayout(self.opacity_frame)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_lbl = QLabel("Opacity:")
        op_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        op_layout.addWidget(op_lbl)

        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(40, 100)
        self.slider_opacity.setValue(94)
        self.slider_opacity.valueChanged.connect(self.update_opacity)
        op_layout.addWidget(self.slider_opacity)
        layout.addWidget(self.opacity_frame)

        # 6. Status Footer
        footer = QHBoxLayout()
        self.db_label = QLabel(f"DB: {len(self.pipeline.matcher.card_names)} cards")
        self.db_label.setStyleSheet("color: #64748b; font-size: 10px;")
        footer.addWidget(self.db_label)

        footer.addStretch()

        self.learned_label = QLabel(f"Memory: {self.pipeline.memory.total_samples}")
        self.learned_label.setStyleSheet("color: #c084fc; font-size: 10px; font-weight: bold;")
        footer.addWidget(self.learned_label)

        footer.addStretch()

        self.fps_label = QLabel("FPS: 8.0")
        self.fps_label.setStyleSheet("color: #64748b; font-size: 10px;")
        footer.addWidget(self.fps_label)

        layout.addLayout(footer)

        root_layout.addWidget(self.container)

    def init_worker(self):
        """Spawns the background monitoring worker."""
        self.worker = PipelineWorker(self.pipeline, target_fps=8.0, parent=self)
        self.worker.sig_status.connect(self.on_worker_status)
        self.worker.sig_fps.connect(self.on_worker_fps)
        self.worker.sig_answer_ready.connect(self.display_answer)
        self.worker.sig_uncertain.connect(self.display_uncertain)
        self.worker.sig_auto_learned.connect(self.on_auto_learned)
        self.worker.sig_memory_updated.connect(self.on_memory_updated)
        self.worker.sig_error.connect(self.on_worker_error)
        self.worker.start()

    def update_opacity(self, val: int):
        self.setWindowOpacity(val / 100.0)

    def on_worker_status(self, status: str):
        if status == "PAUSED":
            self.live_indicator.setText("● PAUSED")
            self.live_indicator.setStyleSheet("color: #f59e0b; font-weight: bold;")
        elif status == "MONITORING" or status == "IDLE":
            self.live_indicator.setText("● LIVE")
            self.live_indicator.setStyleSheet("color: #4ade80; font-weight: bold;")
        elif status == "QUESTION_FOUND" or status == "OCR_PROCESSING":
            self.state_label.setText("RECOGNIZING CARD...")
            self.state_label.setStyleSheet("color: #38bdf8; font-weight: bold;")

    def on_worker_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_worker_error(self, err: str):
        self.state_label.setText("MONITOR ERROR")
        self.conf_badge.set_confidence(0.0, "ERROR")

    def display_answer(self, result: AnswerResult):
        """Displays high-confidence predicted card."""
        self.last_result = result
        self.auto_learn_banner.setVisible(False)
        self.state_label.setText("ANSWER FOUND")
        self.state_label.setStyleSheet("color: #4ade80; font-weight: bold;")

        self.card_name_label.setText(result.card_name.upper())
        self.card_name_label.setStyleSheet("font-size: 21px; font-weight: 800; color: #4ade80; padding: 4px 0;")

        # Update slot badge
        slot_names = ["TOP-LEFT [TL]", "TOP-RIGHT [TR]", "BOTTOM-LEFT [BL]", "BOTTOM-RIGHT [BR]"]
        if getattr(result, "button_slot", None) is not None and 0 <= result.button_slot < 4:
            self.slot_badge.setText(f"TARGET SLOT: {slot_names[result.button_slot]}")
            self.slot_badge.setVisible(True)
        else:
            self.slot_badge.setVisible(False)

        # Update runner-up
        if getattr(result, "runner_up_name", None):
            score_str = f" ({result.runner_up_score:.2f})" if result.runner_up_score else ""
            self.runner_up_label.setText(f"Runner-up: {result.runner_up_name}{score_str} | Margin: +{result.margin:.2f}")
        else:
            self.runner_up_label.setText(f"Margin: +{result.margin:.2f}")

        self.conf_badge.set_confidence(result.confidence, result.confidence_level)
        self.source_badge.set_source(result.matched_from)
        self.latency_label.setText(f"Latency: {result.processing_time_ms:.0f} ms")

        self.update_candidates_display(result)

    def display_uncertain(self, result: AnswerResult):
        """Displays prediction when confidence is low or ambiguous."""
        self.last_result = result
        self.auto_learn_banner.setVisible(False)
        self.state_label.setText("UNCERTAIN MATCH")
        self.state_label.setStyleSheet("color: #f59e0b; font-weight: bold;")

        self.card_name_label.setText(f"{result.card_name} ?")
        self.card_name_label.setStyleSheet("font-size: 20px; font-weight: 800; color: #fbbf24; padding: 4px 0;")

        slot_names = ["TOP-LEFT [TL]", "TOP-RIGHT [TR]", "BOTTOM-LEFT [BL]", "BOTTOM-RIGHT [BR]"]
        if getattr(result, "button_slot", None) is not None and 0 <= result.button_slot < 4:
            self.slot_badge.setText(f"POSSIBLE SLOT: {slot_names[result.button_slot]}")
            self.slot_badge.setVisible(True)
        else:
            self.slot_badge.setVisible(False)

        if getattr(result, "runner_up_name", None):
            score_str = f" ({result.runner_up_score:.2f})" if result.runner_up_score else ""
            self.runner_up_label.setText(f"Runner-up: {result.runner_up_name}{score_str} | Margin: +{result.margin:.2f}")
        else:
            self.runner_up_label.setText(f"Margin: +{result.margin:.2f}")

        self.conf_badge.set_confidence(result.confidence, "UNCERTAIN")
        self.source_badge.set_source(result.matched_from)
        self.latency_label.setText(f"Latency: {result.processing_time_ms:.0f} ms")

        self.update_candidates_display(result)

    def on_auto_learned(self, report):
        """Displays auto-learned result status."""
        if not report:
            return
        if report.verified:
            self.auto_learn_banner.setText(f"★ Auto-Learned: {report.card_name}")
            self.auto_learn_banner.setStyleSheet(
                "color: #4ade80; font-size: 11px; font-weight: bold; "
                "background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.4); "
                "border-radius: 4px; padding: 3px 6px;"
            )
        else:
            self.auto_learn_banner.setText(f"⚠ Result: {report.message}")
            self.auto_learn_banner.setStyleSheet(
                "color: #f59e0b; font-size: 11px; font-weight: bold; "
                "background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4); "
                "border-radius: 4px; padding: 3px 6px;"
            )
        self.auto_learn_banner.setVisible(True)

    def on_memory_updated(self, total: int):
        self.learned_label.setText(f"Memory: {total}")

    def toggle_compact_mode(self):
        """Toggles between full view and compact mini-HUD."""
        self.is_compact = not getattr(self, "is_compact", False)
        self.cand_frame.setVisible(not self.is_compact)
        self.action_frame.setVisible(not self.is_compact)
        self.opacity_frame.setVisible(not self.is_compact)
        self.runner_up_label.setVisible(not self.is_compact)
        self.btn_compact.setText("▴ Expand" if self.is_compact else "▾ Compact")
        self.adjustSize()

    def update_candidates_display(self, result: AnswerResult):
        candidates = result.candidates or [result.card_name]
        slot_prefixes = ["[TL]", "[TR]", "[BL]", "[BR]"]
        assigned_slot = getattr(result, "button_slot", None)

        for i in range(4):
            prefix = slot_prefixes[i]
            if i < len(candidates):
                name = candidates[i]
                is_winner = False
                if assigned_slot is not None and assigned_slot == i:
                    is_winner = True
                elif name.lower().strip() == result.card_name.lower().strip():
                    is_winner = True

                marker = "➤ " if is_winner else "   "
                self.cand_labels[i].setText(f"{prefix} {marker}{name}")
                if is_winner:
                    self.cand_labels[i].setStyleSheet(
                        "color: #4ade80; font-weight: bold; background: rgba(34, 197, 94, 0.25); "
                        "border: 1px solid #22c55e; border-radius: 6px; padding: 6px 8px;"
                    )
                else:
                    self.cand_labels[i].setStyleSheet(
                        "color: #94a3b8; font-weight: normal; background: rgba(30, 41, 59, 120); "
                        "border: 1px solid rgba(71, 85, 105, 100); border-radius: 6px; padding: 6px 8px;"
                    )
            else:
                self.cand_labels[i].setText(f"{prefix} —")
                self.cand_labels[i].setStyleSheet(
                    "color: #475569; background: rgba(15, 23, 42, 80); border: 1px solid rgba(51, 65, 85, 60); border-radius: 6px; padding: 6px 8px;"
                )

    def on_confirm_answer(self):
        """User confirms predicted card was correct; saves to learned memory."""
        if not self.last_result:
            return
        sid = self.pipeline.confirm_last_answer()
        if sid:
            self.learned_label.setText(f"Memory: {self.pipeline.memory.total_samples}")
            self.btn_confirm.setText("✓ Saved!")
            self.btn_confirm.setStyleSheet("background-color: #065f46; color: #6ee7b7;")
        else:
            QMessageBox.information(self, "Memory", "No recent card available to confirm.")

    def on_correct_answer(self):
        """User triggers manual correction dialog."""
        current = self.last_result.card_name if self.last_result else ""
        dialog = CorrectionDialog(
            all_card_names=self.pipeline.matcher.card_names,
            current_guess=current,
            parent=self
        )
        if dialog.exec() and dialog.selected_name:
            correct_name = dialog.selected_name
            sid = self.pipeline.correct_last_answer(correct_name)
            if sid:
                self.learned_label.setText(f"Memory: {self.pipeline.memory.total_samples}")
                self.card_name_label.setText(correct_name.upper())
                self.card_name_label.setStyleSheet("font-size: 21px; font-weight: 800; color: #a855f7;")
                self.source_badge.set_source("learned_memory")

    def toggle_pause(self):
        if self.worker:
            paused = self.worker.toggle_pause()
            self.btn_pause.setText("▶ Resume" if paused else "❚❚ Pause")

    def open_calibration(self):
        if self.worker:
            self.worker.pause()
        dialog = CalibrationDialog(screen_capture=self.pipeline.capture, parent=self)
        dialog.exec()
        if self.worker:
            self.worker.resume()

    # Drag window handlers
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

