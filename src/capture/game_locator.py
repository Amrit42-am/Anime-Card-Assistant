"""
src/capture/game_locator.py

Combines display capture and window tracking to establish Regions of Interest (ROI)
for the Roblox Anime Card Clash game. Automatically adjusts to window movements
and resizes, falling back to manual calibration coordinates when necessary.
"""

from dataclasses import dataclass
import logging
from typing import Optional, List

from src.capture.display_monitor import DisplayMonitor
from src.capture.window_tracker import WindowTracker
from src.capture.region_manager import CalibrationConfig, RectRegion, RelRegion

logger = logging.getLogger("game_locator")


@dataclass
class ResolvedGameRegions:
    game_rect: RectRegion
    card_rect: RectRegion
    options_rect: RectRegion
    timer_rect: RectRegion
    banner_rect: RectRegion
    button_rects: List[RectRegion]  # 0: TL, 1: TR, 2: BL, 3: BR
    is_auto_tracked: bool
    source_window_title: str


class GameLocator:
    """
    Locates the active Roblox game viewport and computes sub-region bounding boxes.
    """

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        display_monitor: Optional[DisplayMonitor] = None,
        window_tracker: Optional[WindowTracker] = None
    ):
        self.calibration = calibration or CalibrationConfig.load()
        self.display_monitor = display_monitor or DisplayMonitor()
        self.window_tracker = window_tracker or WindowTracker()
        self.is_auto_tracked = False

    def locate_game(self) -> ResolvedGameRegions:
        """
        Attempts to find Roblox via Quartz WindowTracker.
        If found, dynamically derives sub-regions using relative ratios.
        Otherwise falls back to saved manual calibration.
        """
        window = self.window_tracker.track()
        cfg = self.calibration

        if window is not None and window.width > 300 and window.height > 200:
            self.is_auto_tracked = True
            game_rect = RectRegion(x=window.x, y=window.y, w=window.width, h=window.height)
            title = f"{window.owner_name} - {window.window_name}"
        else:
            self.is_auto_tracked = False
            game_rect = cfg.game_window
            title = "Manual Calibrated Viewport"

        gw = game_rect.w
        gh = game_rect.h
        gx = game_rect.x
        gy = game_rect.y

        # Compute Card Region
        c_rel = cfg.card_relative
        card_rect = RectRegion(
            x=gx + int(c_rel.rx * gw),
            y=gy + int(c_rel.ry * gh),
            w=int(c_rel.rw * gw),
            h=int(c_rel.rh * gh)
        )

        # Compute Options Panel Region
        o_rel = cfg.options_relative
        options_rect = RectRegion(
            x=gx + int(o_rel.rx * gw),
            y=gy + int(o_rel.ry * gh),
            w=int(o_rel.rw * gw),
            h=int(o_rel.rh * gh)
        )

        # Compute Timer Region
        t_rel = cfg.timer_relative
        timer_rect = RectRegion(
            x=gx + int(t_rel.rx * gw),
            y=gy + int(t_rel.ry * gh),
            w=int(t_rel.rw * gw),
            h=int(t_rel.rh * gh)
        )

        # Compute Result Banner Region
        b_rel = getattr(cfg, "banner_relative", RelRegion(0.28, 0.60, 0.44, 0.06))
        banner_rect = RectRegion(
            x=gx + int(b_rel.rx * gw),
            y=gy + int(b_rel.ry * gh),
            w=int(b_rel.rw * gw),
            h=int(b_rel.rh * gh)
        )

        # Compute 2x2 Button Rects (TL, TR, BL, BR) with banner offset & gaps
        banner_offset_y = int(0.16 * options_rect.h)
        margin_bottom = int(0.03 * options_rect.h)
        margin_x = int(0.03 * options_rect.w)
        gap_x = int(0.04 * options_rect.w)
        gap_y = int(0.05 * options_rect.h)

        usable_w = max(20, options_rect.w - 2 * margin_x - gap_x)
        usable_h = max(20, options_rect.h - banner_offset_y - margin_bottom - gap_y)
        btn_w = usable_w // 2
        btn_h = usable_h // 2

        tl = RectRegion(
            x=options_rect.x + margin_x,
            y=options_rect.y + banner_offset_y,
            w=btn_w,
            h=btn_h
        )
        tr = RectRegion(
            x=options_rect.x + margin_x + btn_w + gap_x,
            y=options_rect.y + banner_offset_y,
            w=btn_w,
            h=btn_h
        )
        bl = RectRegion(
            x=options_rect.x + margin_x,
            y=options_rect.y + banner_offset_y + btn_h + gap_y,
            w=btn_w,
            h=btn_h
        )
        br = RectRegion(
            x=options_rect.x + margin_x + btn_w + gap_x,
            y=options_rect.y + banner_offset_y + btn_h + gap_y,
            w=btn_w,
            h=btn_h
        )

        return ResolvedGameRegions(
            game_rect=game_rect,
            card_rect=card_rect,
            options_rect=options_rect,
            timer_rect=timer_rect,
            banner_rect=banner_rect,
            button_rects=[tl, tr, bl, br],
            is_auto_tracked=self.is_auto_tracked,
            source_window_title=title
        )

