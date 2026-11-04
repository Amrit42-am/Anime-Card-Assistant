"""
src/capture/screen_capture.py

High-performance macOS screen capture using MSS and Quartz.
Handles Retina display scaling, coordinate translation, multi-monitor setups,
and crops game, card, and option button regions efficiently.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import mss
import numpy as np

from src.capture.region_manager import CalibrationConfig, RectRegion
from src.capture.game_locator import GameLocator, ResolvedGameRegions

logger = logging.getLogger("screen_capture")


class ScreenCapture:
    """Manages high-rate screen monitoring, dynamic window tracking, and region cropping."""

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        game_locator: Optional[GameLocator] = None
    ):
        self.calibration = calibration or CalibrationConfig.load()
        self.game_locator = game_locator or GameLocator(calibration=self.calibration)
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
            return False
        except Exception as e:
            logger.warning("Permission preflight check returned error: %s", e)
            return False

    @staticmethod
    def request_permissions():
        """Trigger macOS Screen Recording system prompt."""
        try:
            import Quartz
            Quartz.CGRequestScreenCaptureAccess()
        except Exception as e:
            logger.warning("Failed to invoke CGRequestScreenCaptureAccess: %s", e)

    def capture_full_screen(self, monitor_idx: int = 1) -> np.ndarray:
        """Capture entire screen as BGR numpy array without writing to disk."""
        mon = self.monitors[monitor_idx] if monitor_idx < len(self.monitors) else self.monitors[0]
        raw = self._sct.grab(mon)
        bgra = np.array(raw)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def capture_rect(self, rect: RectRegion) -> np.ndarray:
        """
        Capture a specific bounding box across all monitors.
        Coordinates support multi-display desktop virtual spaces and negative coordinates.
        """
        all_mon = self.monitors[0]
        min_x = all_mon["left"]
        min_y = all_mon["top"]
        max_x = min_x + all_mon["width"]
        max_y = min_y + all_mon["height"]

        left = max(min_x, min(rect.x, max_x - 10))
        top = max(min_y, min(rect.y, max_y - 10))
        width = max(10, min(rect.w, max_x - left))
        height = max(10, min(rect.h, max_y - top))

        bbox = {"left": left, "top": top, "width": width, "height": height}
        raw = self._sct.grab(bbox)
        bgra = np.array(raw)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def capture_quiz_bundle(
        self,
        calibration: Optional[CalibrationConfig] = None
    ) -> Dict[str, Any]:
        """
        Captures the live game window using automatic Roblox window tracking
        and slices out card, options, and button frames.
        Falls back to manual calibration if Roblox is not detected.
        """
        if calibration is not None:
            self.game_locator.calibration = calibration
            cfg = calibration
        else:
            cfg = self.calibration

        resolved = self.game_locator.locate_game()
        game_rect = resolved.game_rect
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

        # Slice individual 4 option buttons in 2x2 grid:
        # 1. Top-Left, 2. Top-Right, 3. Bottom-Left, 4. Bottom-Right
        button_frames: List[np.ndarray] = []
        if options_frame.shape[0] > 20 and options_frame.shape[1] > 20:
            oh, ow = options_frame.shape[:2]
            half_h = oh // 2
            half_w = ow // 2
            pad_x = 4
            pad_y = 4

            # 1. Top-Left
            tl = options_frame[pad_y:half_h - pad_y, pad_x:half_w - pad_x]
            # 2. Top-Right
            tr = options_frame[pad_y:half_h - pad_y, half_w + pad_x:ow - pad_x]
            # 3. Bottom-Left
            bl = options_frame[half_h + pad_y:oh - pad_y, pad_x:half_w - pad_x]
            # 4. Bottom-Right
            br = options_frame[half_h + pad_y:oh - pad_y, half_w + pad_x:ow - pad_x]

            button_frames = [tl, tr, bl, br]

        # Slice timer / countdown region at top
        tim_rel = cfg.timer_relative
        ty1 = max(0, int(tim_rel.ry * gh))
        ty2 = min(gh, int((tim_rel.ry + tim_rel.rh) * gh))
        tx1 = max(0, int(tim_rel.rx * gw))
        tx2 = min(gw, int((tim_rel.rx + tim_rel.rw) * gw))
        timer_frame = game_frame[ty1:ty2, tx1:tx2]

        # Slice dedicated result / instruction banner region above options
        banner_rel = getattr(cfg, "banner_relative", None)
        if banner_rel is not None:
            by1 = max(0, int(banner_rel.ry * gh))
            by2 = min(gh, int((banner_rel.ry + banner_rel.rh) * gh))
            bx1 = max(0, int(banner_rel.rx * gw))
            bx2 = min(gw, int((banner_rel.rx + banner_rel.rw) * gw))
            banner_frame = game_frame[by1:by2, bx1:bx2]
        else:
            banner_frame = None

        return {
            "game_frame": game_frame,
            "card_frame": card_frame,
            "options_frame": options_frame,
            "button_frames": button_frames,
            "timer_frame": timer_frame,
            "banner_frame": banner_frame,
            "game_rect": game_rect,
            "is_auto_tracked": getattr(resolved, "is_auto_tracked", False),
            "source_window_title": getattr(resolved, "source_window_title", ""),
            "resolved_regions": resolved,
        }

    def diagnose_quiz_crops(self, output_dir: Path = Path("data/diagnostics")) -> Dict[str, Any]:
        """
        Diagnostic mode: Captures live game window, draws color-coded rectangles
        for Game Window, Card Artwork, Options Grid, and all 4 Button Slots (TL, TR, BL, BR),
        saves individual button crops and runs Apple Vision OCR on each.
        Outputs diagnostics to output_dir and prints summary report.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle = self.capture_quiz_bundle()
        game_frame = bundle["game_frame"]
        card_frame = bundle["card_frame"]
        options_frame = bundle["options_frame"]
        button_frames = bundle.get("button_frames", [])
        resolved = bundle.get("resolved_regions")

        annotated = game_frame.copy()
        gh, gw = game_frame.shape[:2]
        cfg = self.calibration

        # Draw Card Artwork Box (Magenta)
        card_rel = cfg.card_relative
        cx1, cy1 = int(card_rel.rx * gw), int(card_rel.ry * gh)
        cx2, cy2 = int((card_rel.rx + card_rel.rw) * gw), int((card_rel.ry + card_rel.rh) * gh)
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
        cv2.putText(annotated, "Card Artwork", (cx1, max(18, cy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Draw Options Area (Green)
        opts_rel = cfg.options_relative
        ox1, oy1 = int(opts_rel.rx * gw), int(opts_rel.ry * gh)
        ox2, oy2 = int((opts_rel.rx + opts_rel.rw) * gw), int((opts_rel.ry + opts_rel.rh) * gh)
        cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        cv2.putText(annotated, "Options Area", (ox1, max(18, oy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw 4 Buttons inside options area (Yellow)
        ow = ox2 - ox1
        oh = oy2 - oy1
        half_w = ow // 2
        half_h = oh // 2
        pad_x, pad_y = 4, 4

        slots_info = [
            (0, "TL", ox1 + pad_x, oy1 + pad_y, ox1 + half_w - pad_x, oy1 + half_h - pad_y),
            (1, "TR", ox1 + half_w + pad_x, oy1 + pad_y, ox1 + ow - pad_x, oy1 + half_h - pad_y),
            (2, "BL", ox1 + pad_x, oy1 + half_h + pad_y, ox1 + half_w - pad_x, oy1 + oh - pad_y),
            (3, "BR", ox1 + half_w + pad_x, oy1 + half_h + pad_y, ox1 + ow - pad_x, oy1 + oh - pad_y),
        ]

        for s_idx, s_label, bx1, by1, bx2, by2 in slots_info:
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(annotated, f"Slot {s_idx} [{s_label}]", (bx1 + 4, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Save annotated image
        annotated_path = output_dir / "debug_screen_crops.png"
        cv2.imwrite(str(annotated_path), annotated)

        # Save individual button crops
        crop_paths = []
        for i, b in enumerate(button_frames):
            b_path = output_dir / f"button_slot_{i}_{slots_info[i][1]}.png"
            if b is not None and b.size > 0:
                cv2.imwrite(str(b_path), b)
            crop_paths.append(b_path)

        # Run OCR on the crops
        from src.ocr.option_reader import OptionReader
        reader = OptionReader()
        candidates = reader.read_options_from_buttons(button_frames)

        report = {
            "game_dimensions": f"{gw}x{gh}",
            "is_auto_tracked": bundle.get("is_auto_tracked", False),
            "window_title": bundle.get("source_window_title", ""),
            "annotated_screenshot": str(annotated_path),
            "crop_paths": [str(p) for p in crop_paths],
            "candidates": [
                {
                    "slot": c.slot_index,
                    "raw_text": c.raw_text,
                    "canonical_name": c.canonical_name,
                    "card_id": c.card_id,
                    "is_valid": c.is_valid,
                    "confidence": c.ocr_confidence,
                    "fuzzy_score": c.fuzzy_score
                }
                for c in candidates
            ]
        }
        return report

    def close(self):
        self._sct.close()
