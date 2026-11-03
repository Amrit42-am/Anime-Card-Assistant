"""
src/capture/screen_capture.py

High-performance macOS screen capture using MSS and Quartz.
Handles Retina display scaling, coordinate translation, multi-monitor setups,
and crops game, card, and option button regions efficiently.
"""

import logging
from typing import Dict, Any, Optional, List
import cv2
import mss
import numpy as np

from src.capture.region_manager import CalibrationConfig, RectRegion

logger = logging.getLogger("screen_capture")


class ScreenCapture:
    """Manages high-rate screen monitoring and region cropping."""

    def __init__(self, calibration: Optional[CalibrationConfig] = None):
        self.calibration = calibration or CalibrationConfig.load()
        self._sct = mss.MSS()
        self.monitors = self._sct.monitors
        logger.info("Initialized ScreenCapture with %d monitors detected.", len(self.monitors))

    @staticmethod
    def check_permissions() -> bool:
        """Check macOS Screen Recording permission using Quartz preflight."""
        try:
            import Quartz
            if Quartz.CGPreflightScreenCaptureAccess():
                return True
            # Try a quick test grab with MSS as definitive fallback
            with mss.MSS() as s:
                img = s.grab({"top": 0, "left": 0, "width": 10, "height": 10})
                return img is not None and len(img.raw) > 0
        except Exception as e:
            logger.warning("Permission preflight check returned error: %s", e)
            return True

    @staticmethod
    def request_permissions():
        """Trigger macOS Screen Recording system prompt."""
        try:
            import Quartz
            Quartz.CGRequestScreenCaptureAccess()
        except Exception as e:
            logger.warning("Failed to invoke CGRequestScreenCaptureAccess: %s", e)

    def capture_full_screen(self, monitor_idx: int = 1) -> np.ndarray:
        """Capture entire screen as BGR numpy array."""
        mon = self.monitors[monitor_idx] if monitor_idx < len(self.monitors) else self.monitors[0]
        raw = self._sct.grab(mon)
        bgra = np.array(raw)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def capture_rect(self, rect: RectRegion) -> np.ndarray:
        """
        Capture a specific bounding box.
        Coordinates are automatically bounded within display limits.
        """
        primary = self.monitors[1] if len(self.monitors) > 1 else self.monitors[0]
        max_w = primary["width"]
        max_h = primary["height"]

        left = max(0, min(rect.x, max_w - 10))
        top = max(0, min(rect.y, max_h - 10))
        width = max(10, min(rect.w, max_w - left))
        height = max(10, min(rect.h, max_h - top))

        bbox = {"left": left, "top": top, "width": width, "height": height}
        raw = self._sct.grab(bbox)
        bgra = np.array(raw)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def capture_quiz_bundle(
        self,
        calibration: Optional[CalibrationConfig] = None
    ) -> Dict[str, Any]:
        """
        Captures the entire game window and slices out card, options, and button frames.
        Returns a dictionary containing cropped BGR frames and metadata.
        """
        cfg = calibration or self.calibration
        game_rect = cfg.game_window
        game_frame = self.capture_rect(game_rect)

        # Slice card artwork and options from game_frame using relative coordinates
        gh, gw = game_frame.shape[:2]

        card_rel = cfg.card_relative
        cy1 = max(0, int(card_rel.ry * gh))
        cy2 = min(gh, int((card_rel.ry + card_rel.rh) * gh))
        cx1 = max(0, int(card_rel.rx * gw))
        cx2 = min(gw, int((card_rel.rx + card_rel.rw) * gw))
        card_frame = game_frame[cy1:cy2, cx1:cx2]

        opts_rel = cfg.options_relative
        oy1 = max(0, int(opts_rel.ry * gh))
        oy2 = min(gh, int((opts_rel.ry + opts_rel.rh) * gh))
        ox1 = max(0, int(opts_rel.rx * gw))
        ox2 = min(gw, int((opts_rel.rx + opts_rel.rw) * gw))
        options_frame = game_frame[oy1:oy2, ox1:ox2]

        # Slice individual 4 option buttons
        button_frames: List[np.ndarray] = []
        if options_frame.shape[0] > 20 and options_frame.shape[1] > 20:
            oh, ow = options_frame.shape[:2]
            slot_h = oh // 4
            for i in range(4):
                by1 = i * slot_h
                by2 = (i + 1) * slot_h if i < 3 else oh
                # small inner padding to crop inside button borders
                btn_crop = options_frame[by1 + 2:by2 - 2, 4:ow - 4]
                button_frames.append(btn_crop)

        # Slice timer/header
        tim_rel = cfg.timer_relative
        ty1 = max(0, int(tim_rel.ry * gh))
        ty2 = min(gh, int((tim_rel.ry + tim_rel.rh) * gh))
        tx1 = max(0, int(tim_rel.rx * gw))
        tx2 = min(gw, int((tim_rel.rx + tim_rel.rw) * gw))
        timer_frame = game_frame[ty1:ty2, tx1:tx2]

        return {
            "game_frame": game_frame,
            "card_frame": card_frame,
            "options_frame": options_frame,
            "button_frames": button_frames,
            "timer_frame": timer_frame,
            "game_rect": game_rect,
        }

    def close(self):
        self._sct.close()
