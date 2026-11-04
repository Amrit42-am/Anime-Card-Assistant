"""
src/capture/screen_capture.py

High-performance macOS screen capture using MSS and Quartz.
Handles Retina display scaling, coordinate translation, multi-monitor setups,
and crops game, card, and option button regions efficiently.
"""

import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional, List, Tuple
import cv2
import mss
import numpy as np

from src.capture.region_manager import CalibrationConfig, RectRegion
from src.capture.game_locator import GameLocator

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

    @staticmethod
    def extract_four_buttons(
        options_frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], bool]:
        """
        Extracts 4 distinct button crops and their bounding boxes (x, y, w, h)
        from options_frame with strict positional mapping:
        [0: TL, 1: TR, 2: BL, 3: BR].

        First attempts automatic contour detection to find the exact 4 button UI boundaries,
        excluding headers/banners and inter-button gaps.
        If contour detection does not find 4 valid buttons, falls back to an accurate
        calibrated 2x2 layout with banner offset and inter-button padding.
        Returns (button_frames, button_rects, is_auto_detected).
        """
        if options_frame is None or options_frame.size == 0 or options_frame.ndim < 2:
            return [], [], False

        oh, ow = options_frame.shape[:2]
        if oh < 30 or ow < 30:
            return [], [], False

        # --- Method 1: Automatic Button Contour Detection ---
        try:
            gray = cv2.cvtColor(options_frame, cv2.COLOR_BGR2GRAY) if options_frame.ndim == 3 else options_frame
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 120)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            opt_area = float(oh * ow)
            candidate_rects = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect = float(w) / max(1, h)
                # Check button criteria: reasonable size, aspect ratio, not banner
                if (0.02 * opt_area <= area <= 0.35 * opt_area and
                    0.20 * ow <= w <= 0.58 * ow and
                    0.10 * oh <= h <= 0.48 * oh and
                    1.2 <= aspect <= 4.8):
                    candidate_rects.append((x, y, w, h))

            if len(candidate_rects) >= 4:
                # Find best 4 forming a 2x2 grid
                candidate_rects.sort(key=lambda r: r[2] * r[3], reverse=True)
                top_candidates = candidate_rects[:6]

                # Partition into upper 2 and lower 2 by y-center
                sorted_by_y = sorted(top_candidates, key=lambda r: r[1] + r[3] / 2.0)
                upper = sorted_by_y[:2]
                lower = sorted_by_y[2:4]

                # Within upper, sort by x-center (TL, TR)
                tl = min(upper, key=lambda r: r[0])
                tr = max(upper, key=lambda r: r[0])
                # Within lower, sort by x-center (BL, BR)
                bl = min(lower, key=lambda r: r[0])
                br = max(lower, key=lambda r: r[0])

                # Validate 2x2 geometry
                tl_xc, tr_xc = tl[0] + tl[2] / 2.0, tr[0] + tr[2] / 2.0
                bl_xc, br_xc = bl[0] + bl[2] / 2.0, br[0] + br[2] / 2.0
                tl_yc, bl_yc = tl[1] + tl[3] / 2.0, bl[1] + bl[3] / 2.0
                tr_yc, br_yc = tr[1] + tr[3] / 2.0, br[1] + br[3] / 2.0

                if (tr_xc > tl_xc + 0.15 * ow and
                    br_xc > bl_xc + 0.15 * ow and
                    bl_yc > tl_yc + 0.12 * oh and
                    br_yc > tr_yc + 0.12 * oh):
                    rects = [tl, tr, bl, br]
                    crops = []
                    for (x, y, w, h) in rects:
                        px, py = max(1, int(0.02 * w)), max(1, int(0.04 * h))
                        btn = options_frame[y + py:y + h - py, x + px:x + w - px]
                        crops.append(btn)
                    return crops, rects, True
        except Exception as e:
            logger.debug("Automatic button contour detection notice: %s", e)

        # --- Method 2: Accurate Calibrated 2x2 Geometry Fallback ---
        # Exclude top instruction banner (top ~16% of options area)
        banner_offset_y = int(0.16 * oh)
        margin_bottom = int(0.03 * oh)
        margin_x = int(0.03 * ow)
        gap_x = int(0.04 * ow)
        gap_y = int(0.05 * oh)

        usable_w = max(20, ow - 2 * margin_x - gap_x)
        usable_h = max(20, oh - banner_offset_y - margin_bottom - gap_y)
        btn_w = usable_w // 2
        btn_h = usable_h // 2

        # 0: TL
        tl_x, tl_y = margin_x, banner_offset_y
        # 1: TR
        tr_x, tr_y = margin_x + btn_w + gap_x, banner_offset_y
        # 2: BL
        bl_x, bl_y = margin_x, banner_offset_y + btn_h + gap_y
        # 3: BR
        br_x, br_y = margin_x + btn_w + gap_x, banner_offset_y + btn_h + gap_y

        rects = [
            (tl_x, tl_y, btn_w, btn_h),
            (tr_x, tr_y, btn_w, btn_h),
            (bl_x, bl_y, btn_w, btn_h),
            (br_x, br_y, btn_w, btn_h),
        ]
        crops = [
            options_frame[tl_y:tl_y + btn_h, tl_x:tl_x + btn_w],
            options_frame[tr_y:tr_y + btn_h, tr_x:tr_x + btn_w],
            options_frame[bl_y:bl_y + btn_h, bl_x:bl_x + btn_w],
            options_frame[br_y:br_y + btn_h, br_x:br_x + btn_w],
        ]
        return crops, rects, False

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
        button_frames, button_rects, is_buttons_auto = self.extract_four_buttons(options_frame)

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
            "button_rects": button_rects,
            "is_buttons_auto": is_buttons_auto,
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
        button_frames = bundle.get("button_frames", [])

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

        button_rects = bundle.get("button_rects", [])
        labels = ["TL", "TR", "BL", "BR"]
        slots_info = []

        if button_rects and len(button_rects) == 4:
            for s_idx, (rx, ry, rw, rh) in enumerate(button_rects):
                bx1, by1 = ox1 + rx, oy1 + ry
                bx2, by2 = bx1 + rw, by1 + rh
                s_label = labels[s_idx]
                slots_info.append((s_idx, s_label, bx1, by1, bx2, by2))
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                cv2.putText(annotated, f"Slot {s_idx} [{s_label}]", (bx1 + 4, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            half_w = (ox2 - ox1) // 2
            half_h = (oy2 - oy1) // 2
            pad_x, pad_y = 4, 4
            default_slots = [
                (0, "TL", ox1 + pad_x, oy1 + pad_y, ox1 + half_w - pad_x, oy1 + half_h - pad_y),
                (1, "TR", ox1 + half_w + pad_x, oy1 + pad_y, ox1 + (ox2 - ox1) - pad_x, oy1 + half_h - pad_y),
                (2, "BL", ox1 + pad_x, oy1 + half_h + pad_y, ox1 + half_w - pad_x, oy1 + (oy2 - oy1) - pad_y),
                (3, "BR", ox1 + half_w + pad_x, oy1 + half_h + pad_y, ox1 + (ox2 - ox1) - pad_x, oy1 + (oy2 - oy1) - pad_y),
            ]
            for s_idx, s_label, bx1, by1, bx2, by2 in default_slots:
                slots_info.append((s_idx, s_label, bx1, by1, bx2, by2))
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                cv2.putText(annotated, f"Slot {s_idx} [{s_label}]", (bx1 + 4, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Draw Timer Box (Red)
        tim_rel = cfg.timer_relative
        tx1, ty1 = int(tim_rel.rx * gw), int(tim_rel.ry * gh)
        tx2, ty2 = int((tim_rel.rx + tim_rel.rw) * gw), int((tim_rel.ry + tim_rel.rh) * gh)
        cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        cv2.putText(annotated, "Timer", (tx1, max(18, ty1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw Banner Area (Orange) if present
        banner_rel = getattr(cfg, "banner_relative", None)
        if banner_rel is not None:
            bx1, by1 = int(banner_rel.rx * gw), int(banner_rel.ry * gh)
            bx2, by2 = int((banner_rel.rx + banner_rel.rw) * gw), int((banner_rel.ry + banner_rel.rh) * gh)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            cv2.putText(annotated, "Instruction/Result Banner", (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # Save annotated image and viewport crops
        annotated_path = output_dir / "debug_screen_crops.png"
        cv2.imwrite(str(annotated_path), annotated)
        cv2.imwrite(str(output_dir / "04_annotated_regions.png"), annotated)
        cv2.imwrite(str(output_dir / "03_game_viewport.png"), game_frame)

        card_frame = bundle.get("card_frame")
        if card_frame is not None and card_frame.size > 0:
            cv2.imwrite(str(output_dir / "05_card_crop.png"), card_frame)

        options_frame = bundle.get("options_frame")
        if options_frame is not None and options_frame.size > 0:
            cv2.imwrite(str(output_dir / "06_options_grid.png"), options_frame)

        # Save individual button crops (both legacy and standardized numbered format)
        crop_paths = []
        numbered_names = ["07_option_tl.png", "08_option_tr.png", "09_option_bl.png", "10_option_br.png"]
        for i, b in enumerate(button_frames):
            b_path = output_dir / f"button_slot_{i}_{slots_info[i][1]}.png"
            if b is not None and b.size > 0:
                cv2.imwrite(str(b_path), b)
                if i < len(numbered_names):
                    cv2.imwrite(str(output_dir / numbered_names[i]), b)
            crop_paths.append(b_path)

        # Full screen capture for diagnostic completeness
        try:
            full_screen = self.capture_full_screen(0)
            cv2.imwrite(str(output_dir / "01_full_screen.png"), full_screen)
        except Exception:
            pass

        # Run OCR on the crops
        from src.ocr.option_reader import OptionReader
        reader = OptionReader()
        candidates = reader.read_options_from_buttons(button_frames)

        ocr_report = [
            {
                "slot": c.slot_index,
                "slot_label": labels[c.slot_index] if c.slot_index < len(labels) else f"S{c.slot_index}",
                "raw_text": c.raw_text,
                "canonical_name": c.canonical_name,
                "card_id": c.card_id,
                "is_valid": c.is_valid,
                "confidence": round(c.ocr_confidence, 4),
                "fuzzy_score": round(c.fuzzy_score, 2),
                "failure_reason": None if c.is_valid else ("Low fuzzy score" if c.raw_text else "No text detected")
            }
            for c in candidates
        ]

        detection_report = {
            "timestamp": time.time(),
            "game_dimensions": f"{gw}x{gh}",
            "is_auto_tracked": bundle.get("is_auto_tracked", False),
            "window_title": bundle.get("source_window_title", ""),
            "is_buttons_auto": bundle.get("is_buttons_auto", False),
            "button_rectangles": [
                {"slot": i, "label": labels[i], "x": r[0], "y": r[1], "w": r[2], "h": r[3]}
                for i, r in enumerate(button_rects)
            ] if button_rects else []
        }

        with open(output_dir / "ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(ocr_report, f, indent=2)

        with open(output_dir / "detection_results.json", "w", encoding="utf-8") as f:
            json.dump(detection_report, f, indent=2)

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
