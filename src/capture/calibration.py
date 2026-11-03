"""
src/capture/calibration.py

Phase 3: Interactive Visual Calibration Interface using PySide6.
Allows selecting and tuning the Roblox game window, card artwork region,
and answer options box with live visual preview.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout
)

from src.capture.region_manager import CalibrationConfig, RectRegion, RelativeRegion
from src.capture.screen_capture import ScreenCapture

logger = logging.getLogger("calibration_ui")


class CalibrationDialog(QDialog):
    """Visual calibration dialog for aligning screen capture regions."""

    def __init__(self, screen_capture: ScreenCapture, parent=None):
        super().__init__(parent)
        self.capture = screen_capture
        self.config = self.capture.calibration

        self.setWindowTitle("Anime Card Assistant — Screen Calibration")
        self.setMinimumSize(850, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1a1c23; color: #e0e0e0; font-family: -apple-system, sans-serif; }
            QLabel { color: #cfd3dc; }
            QGroupBox { border: 1px solid #3d4251; border-radius: 6px; margin-top: 12px; font-weight: bold; color: #64b5f6; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background-color: #2b313e; border: 1px solid #4a5568; border-radius: 4px; padding: 6px 12px; color: white; }
            QPushButton:hover { background-color: #3b4252; }
            QPushButton#saveBtn { background-color: #2e7d32; font-weight: bold; }
            QPushButton#saveBtn:hover { background-color: #388e3c; }
            QSpinBox, QDoubleSpinBox { background-color: #262b36; border: 1px solid #4a5568; color: white; padding: 4px; border-radius: 4px; }
        """)

        self.init_ui()
        self.refresh_preview()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left column: Controls
        ctrl_layout = QVBoxLayout()

        # 1. Game Window Box
        gw_box = QGroupBox("Roblox Game Window (Screen Coordinates)")
        gw_grid = QGridLayout(gw_box)

        self.spin_gx = QSpinBox()
        self.spin_gx.setRange(0, 3840)
        self.spin_gx.setValue(self.config.game_window.x)

        self.spin_gy = QSpinBox()
        self.spin_gy.setRange(0, 2160)
        self.spin_gy.setValue(self.config.game_window.y)

        self.spin_gw = QSpinBox()
        self.spin_gw.setRange(200, 3840)
        self.spin_gw.setValue(self.config.game_window.w)

        self.spin_gh = QSpinBox()
        self.spin_gh.setRange(200, 2160)
        self.spin_gh.setValue(self.config.game_window.h)

        gw_grid.addWidget(QLabel("X:"), 0, 0)
        gw_grid.addWidget(self.spin_gx, 0, 1)
        gw_grid.addWidget(QLabel("Y:"), 0, 2)
        gw_grid.addWidget(self.spin_gy, 0, 3)
        gw_grid.addWidget(QLabel("Width:"), 1, 0)
        gw_grid.addWidget(self.spin_gw, 1, 1)
        gw_grid.addWidget(QLabel("Height:"), 1, 2)
        gw_grid.addWidget(self.spin_gh, 1, 3)
        ctrl_layout.addWidget(gw_box)

        # 2. Card Artwork Box (Relative %)
        card_box = QGroupBox("Card Artwork Region (Proportional %)")
        card_grid = QGridLayout(card_box)

        self.spin_cx = QDoubleSpinBox()
        self.spin_cx.setRange(0.0, 1.0)
        self.spin_cx.setSingleStep(0.02)
        self.spin_cx.setValue(self.config.card_relative.rx)

        self.spin_cy = QDoubleSpinBox()
        self.spin_cy.setRange(0.0, 1.0)
        self.spin_cy.setSingleStep(0.02)
        self.spin_cy.setValue(self.config.card_relative.ry)

        self.spin_cw = QDoubleSpinBox()
        self.spin_cw.setRange(0.05, 1.0)
        self.spin_cw.setSingleStep(0.02)
        self.spin_cw.setValue(self.config.card_relative.rw)

        self.spin_ch = QDoubleSpinBox()
        self.spin_ch.setRange(0.05, 1.0)
        self.spin_ch.setSingleStep(0.02)
        self.spin_ch.setValue(self.config.card_relative.rh)

        card_grid.addWidget(QLabel("Offset X:"), 0, 0)
        card_grid.addWidget(self.spin_cx, 0, 1)
        card_grid.addWidget(QLabel("Offset Y:"), 0, 2)
        card_grid.addWidget(self.spin_cy, 0, 3)
        card_grid.addWidget(QLabel("Width:"), 1, 0)
        card_grid.addWidget(self.spin_cw, 1, 1)
        card_grid.addWidget(QLabel("Height:"), 1, 2)
        card_grid.addWidget(self.spin_ch, 1, 3)
        ctrl_layout.addWidget(card_box)

        # 3. Answer Options Box (Relative %)
        opt_box = QGroupBox("Answer Options Region (Proportional %)")
        opt_grid = QGridLayout(opt_box)

        self.spin_ox = QDoubleSpinBox()
        self.spin_ox.setRange(0.0, 1.0)
        self.spin_ox.setSingleStep(0.02)
        self.spin_ox.setValue(self.config.options_relative.rx)

        self.spin_oy = QDoubleSpinBox()
        self.spin_oy.setRange(0.0, 1.0)
        self.spin_oy.setSingleStep(0.02)
        self.spin_oy.setValue(self.config.options_relative.ry)

        self.spin_ow = QDoubleSpinBox()
        self.spin_ow.setRange(0.05, 1.0)
        self.spin_ow.setSingleStep(0.02)
        self.spin_ow.setValue(self.config.options_relative.rw)

        self.spin_oh = QDoubleSpinBox()
        self.spin_oh.setRange(0.05, 1.0)
        self.spin_oh.setSingleStep(0.02)
        self.spin_oh.setValue(self.config.options_relative.rh)

        opt_grid.addWidget(QLabel("Offset X:"), 0, 0)
        opt_grid.addWidget(self.spin_ox, 0, 1)
        opt_grid.addWidget(QLabel("Offset Y:"), 0, 2)
        opt_grid.addWidget(self.spin_oy, 0, 3)
        opt_grid.addWidget(QLabel("Width:"), 1, 0)
        opt_grid.addWidget(self.spin_ow, 1, 1)
        opt_grid.addWidget(QLabel("Height:"), 1, 2)
        opt_grid.addWidget(self.spin_oh, 1, 3)
        ctrl_layout.addWidget(opt_box)

        # Connect value changes to live update
        for spin in [self.spin_gx, self.spin_gy, self.spin_gw, self.spin_gh,
                     self.spin_cx, self.spin_cy, self.spin_cw, self.spin_ch,
                     self.spin_ox, self.spin_oy, self.spin_ow, self.spin_oh]:
            spin.valueChanged.connect(self.refresh_preview)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Snapshot")
        btn_refresh.clicked.connect(self.refresh_preview)
        btn_layout.addWidget(btn_refresh)

        btn_save = QPushButton("Save Calibration")
        btn_save.setObjectName("saveBtn")
        btn_save.clicked.connect(self.save_and_close)
        btn_layout.addWidget(btn_save)

        btn_cancel = QPushButton("Close")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        ctrl_layout.addLayout(btn_layout)
        main_layout.addLayout(ctrl_layout, stretch=1)

        # Right column: Live Snapshot Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #12141a; border: 1px solid #2d3340; border-radius: 6px;")
        main_layout.addWidget(self.preview_label, stretch=2)

    def refresh_preview(self):
        """Captures frame from game window and overlays calibrated boxes."""
        rect = RectRegion(
            x=self.spin_gx.value(),
            y=self.spin_gy.value(),
            w=self.spin_gw.value(),
            h=self.spin_gh.value()
        )

        try:
            frame = self.capture.capture_rect(rect)
        except Exception as e:
            logger.error("Calibration capture error: %s", e)
            frame = np.zeros((rect.h, rect.w, 3), dtype=np.uint8)

        gh, gw = frame.shape[:2]

        # Draw Card Region in Green
        cx = int(self.spin_cx.value() * gw)
        cy = int(self.spin_cy.value() * gh)
        cw = int(self.spin_cw.value() * gw)
        ch = int(self.spin_ch.value() * gh)
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
        cv2.putText(frame, "Card Artwork", (cx + 5, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Options Region in Orange/Yellow
        ox = int(self.spin_ox.value() * gw)
        oy = int(self.spin_oy.value() * gh)
        ow = int(self.spin_ow.value() * gw)
        oh = int(self.spin_oh.value() * gh)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 165, 255), 2)
        cv2.putText(frame, "4 Answer Options", (ox + 5, oy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Draw 4 button division guides inside options
        slot_h = oh // 4
        for i in range(1, 4):
            cv2.line(frame, (ox, oy + i * slot_h), (ox + ow, oy + i * slot_h), (0, 140, 220), 1)

        # Convert to QPixmap for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch_count = rgb.shape
        bytes_per_line = ch_count * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix.scaled(560, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_and_close(self):
        """Saves values to configuration and dismisses dialog."""
        self.config.game_window = RectRegion(
            x=self.spin_gx.value(),
            y=self.spin_gy.value(),
            w=self.spin_gw.value(),
            h=self.spin_gh.value()
        )
        self.config.card_relative = RelativeRegion(
            rx=self.spin_cx.value(),
            ry=self.spin_cy.value(),
            rw=self.spin_cw.value(),
            rh=self.spin_ch.value()
        )
        self.config.options_relative = RelativeRegion(
            rx=self.spin_ox.value(),
            ry=self.spin_oy.value(),
            rw=self.spin_ow.value(),
            rh=self.spin_oh.value()
        )
        self.config.save()
        self.accept()
