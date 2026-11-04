"""
tests/test_capture_window.py

Unit tests for macOS Display Monitoring, Window Tracking, and Game Locating.
"""

import numpy as np

from src.capture.display_monitor import DisplayMonitor
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


def test_game_locator_window_movement():
    """Verify that when window moves, GameLocator updates subregion coordinates."""
    cfg = CalibrationConfig(
        game_window=RectRegion(0, 0, 800, 600),
        card_region=RelRegion(0.1, 0.1, 0.4, 0.4),
        options_region=RelRegion(0.5, 0.5, 0.4, 0.4),
        timer_region=RelRegion(0.4, 0.02, 0.2, 0.05)
    )

    class MockTracker(WindowTracker):
        def __init__(self, pos=(100, 100)):
            super().__init__()
            self.pos = pos
        def track(self):
            return WindowInfo(
                window_id=1, owner_name="RobloxPlayer", window_name="Roblox",
                x=self.pos[0], y=self.pos[1], width=800, height=600, is_on_screen=True
            )

    tracker = MockTracker(pos=(100, 100))
    locator = GameLocator(calibration=cfg, window_tracker=tracker)
    r1 = locator.locate_game()
    assert r1.game_rect.x == 100
    assert r1.game_rect.y == 100
    assert r1.card_rect.x == 100 + int(0.1 * 800)

    # Window moves to (500, 400)
    tracker.pos = (500, 400)
    r2 = locator.locate_game()
    assert r2.game_rect.x == 500
    assert r2.game_rect.y == 400
    assert r2.card_rect.x == 500 + int(0.1 * 800)
    assert r2.card_rect.y == 400 + int(0.1 * 600)


def test_game_locator_window_resize():
    """Verify that when window resizes, GameLocator scales all subregions proportionately."""
    cfg = CalibrationConfig(
        game_window=RectRegion(0, 0, 800, 600),
        card_region=RelRegion(0.1, 0.1, 0.4, 0.4),
        options_region=RelRegion(0.5, 0.5, 0.4, 0.4),
        timer_region=RelRegion(0.4, 0.02, 0.2, 0.05)
    )

    class MockTracker(WindowTracker):
        def __init__(self, size=(800, 600)):
            super().__init__()
            self.size = size
        def track(self):
            return WindowInfo(
                window_id=1, owner_name="RobloxPlayer", window_name="Roblox",
                x=0, y=0, width=self.size[0], height=self.size[1], is_on_screen=True
            )

    tracker = MockTracker(size=(800, 600))
    locator = GameLocator(calibration=cfg, window_tracker=tracker)
    r1 = locator.locate_game()
    assert r1.card_rect.w == int(0.4 * 800)

    # Window resizes to 1600x1200
    tracker.size = (1600, 1200)
    r2 = locator.locate_game()
    assert r2.card_rect.w == int(0.4 * 1600)
    assert r2.card_rect.h == int(0.4 * 1200)


def test_game_locator_secondary_and_negative_coordinates():
    """Verify that game window on secondary or negative coordinate monitors works properly."""
    cfg = CalibrationConfig(
        game_window=RectRegion(0, 0, 800, 600),
        card_region=RelRegion(0.1, 0.1, 0.4, 0.4),
        options_region=RelRegion(0.5, 0.5, 0.4, 0.4),
        timer_region=RelRegion(0.4, 0.02, 0.2, 0.05)
    )

    class MockTracker(WindowTracker):
        def __init__(self, pos):
            super().__init__()
            self.pos = pos
        def track(self):
            return WindowInfo(
                window_id=1, owner_name="RobloxPlayer", window_name="Roblox",
                x=self.pos[0], y=self.pos[1], width=1920, height=1080, is_on_screen=True
            )

    # Monitor placed to the left (negative X coordinates)
    tracker_neg = MockTracker(pos=(-1920, 0))
    locator_neg = GameLocator(calibration=cfg, window_tracker=tracker_neg)
    r_neg = locator_neg.locate_game()
    assert r_neg.game_rect.x == -1920
    assert r_neg.card_rect.x == -1920 + int(0.1 * 1920)

    # Monitor placed to the right (secondary screen)
    tracker_sec = MockTracker(pos=(2560, 100))
    locator_sec = GameLocator(calibration=cfg, window_tracker=tracker_sec)
    r_sec = locator_sec.locate_game()
    assert r_sec.game_rect.x == 2560
    assert r_sec.game_rect.y == 100


def test_screen_capture_permission_denied(monkeypatch):
    """Verify ScreenCapture.check_permissions returns False on permission failure."""
    from src.capture.screen_capture import ScreenCapture
    import Quartz

    monkeypatch.setattr(Quartz, "CGPreflightScreenCaptureAccess", lambda: False)
    assert ScreenCapture.check_permissions() is False


def test_browser_hosted_roblox_window_detection(monkeypatch):
    """Verify WindowTracker detects browser-hosted Roblox and prioritizes native RobloxPlayer."""
    import Quartz

    mock_windows = [
        # Unrelated window
        {
            Quartz.kCGWindowOwnerName: "Finder",
            Quartz.kCGWindowName: "Desktop",
            Quartz.kCGWindowBounds: {"X": 0, "Y": 0, "Width": 1512, "Height": 982},
            Quartz.kCGWindowLayer: 0,
            Quartz.kCGWindowIsOnscreen: True,
            Quartz.kCGWindowOwnerPID: 9999,
            Quartz.kCGWindowNumber: 101
        },
        # Browser-hosted Roblox
        {
            Quartz.kCGWindowOwnerName: "Google Chrome",
            Quartz.kCGWindowName: "Anime Card Clash - Roblox",
            Quartz.kCGWindowBounds: {"X": 0, "Y": 50, "Width": 750, "Height": 900},
            Quartz.kCGWindowLayer: 0,
            Quartz.kCGWindowIsOnscreen: True,
            Quartz.kCGWindowOwnerPID: 9998,
            Quartz.kCGWindowNumber: 102
        }
    ]

    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda *args: mock_windows)

    tracker = WindowTracker()
    target = tracker.find_target_window()
    assert target is not None
    assert target.owner_name == "Google Chrome"
    assert target.width == 750
    assert target.height == 900

    # Now add native RobloxPlayer: it must be prioritized over the browser window
    mock_windows.append({
        Quartz.kCGWindowOwnerName: "RobloxPlayer",
        Quartz.kCGWindowName: "Roblox",
        Quartz.kCGWindowBounds: {"X": 0, "Y": 0, "Width": 750, "Height": 900},
        Quartz.kCGWindowLayer: 0,
        Quartz.kCGWindowIsOnscreen: True,
        Quartz.kCGWindowOwnerPID: 9997,
        Quartz.kCGWindowNumber: 103
    })

    target_native = tracker.find_target_window()
    assert target_native is not None
    assert target_native.owner_name == "RobloxPlayer"


def test_roblox_split_screen_left_half_tracking():
    """Verify Roblox window occupying left half of screen (x=0, w=750) coordinates."""
    cfg = CalibrationConfig(
        game_window=RectRegion(0, 0, 800, 600),
        card_region=RelRegion(0.08, 0.22, 0.45, 0.68),
        options_region=RelRegion(0.55, 0.25, 0.40, 0.65),
        timer_region=RelRegion(0.30, 0.08, 0.40, 0.12)
    )

    class MockTracker(WindowTracker):
        def track(self):
            return WindowInfo(
                window_id=1, owner_name="RobloxPlayer", window_name="Roblox",
                x=0, y=50, width=750, height=900, is_on_screen=True
            )

    locator = GameLocator(calibration=cfg, window_tracker=MockTracker())
    resolved = locator.locate_game()
    assert resolved.is_auto_tracked is True
    assert resolved.game_rect.x == 0
    assert resolved.game_rect.y == 50
    assert resolved.game_rect.w == 750
    assert resolved.game_rect.h == 900

    # Ensure options rect is in the right half of the game window
    assert resolved.options_rect.x > resolved.card_rect.x
    assert len(resolved.button_rects) == 4


