"""
tests/test_capture_window.py

Unit tests for macOS Display Monitoring, Window Tracking, and Game Locating.
"""

import numpy as np
import pytest

from src.capture.display_monitor import DisplayMonitor, DisplayInfo
from src.capture.window_tracker import WindowTracker, WindowInfo
from src.capture.game_locator import GameLocator, ResolvedGameRegions
from src.capture.region_manager import CalibrationConfig, RectRegion, RelRegion


def test_display_monitor_enumeration():
    monitor = DisplayMonitor()
    displays = monitor.refresh_displays()
    assert len(displays) >= 1
    primary = monitor.get_primary_display()
    assert primary is not None
    assert primary.width > 0
    assert primary.height > 0


def test_display_monitor_capture_frame():
    monitor = DisplayMonitor()
    frame = monitor.capture_display()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[0] > 100
    assert frame.shape[1] > 100


def test_window_tracker_mock():
    tracker = WindowTracker(target_owner="RobloxPlayer")
    # In headless/CI test environment where Roblox isn't open
    win = tracker.find_target_window()
    # It safely returns None or WindowInfo without throwing exception
    assert win is None or isinstance(win, WindowInfo)


def test_game_locator_fallback():
    cfg = CalibrationConfig(
        game_window=RectRegion(100, 100, 800, 600),
        card_region=RelRegion(0.1, 0.1, 0.4, 0.4),
        options_region=RelRegion(0.5, 0.5, 0.4, 0.4),
        timer_region=RelRegion(0.4, 0.02, 0.2, 0.05)
    )
    locator = GameLocator(calibration=cfg)
    regions = locator.locate_game()

    assert isinstance(regions, ResolvedGameRegions)
    assert regions.game_rect.w > 0
    assert regions.card_rect.w > 0
    assert regions.options_rect.w > 0
    assert len(regions.button_rects) == 4

    # Verify 2x2 grid geometry:
    tl, tr, bl, br = regions.button_rects
    assert tl.x < tr.x
    assert tl.y == tr.y
    assert bl.x < br.x
    assert bl.y > tl.y
    assert br.y > tr.y
