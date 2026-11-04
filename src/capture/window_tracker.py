"""
src/capture/window_tracker.py

Automated Roblox game window tracking on macOS using Quartz CoreGraphics.
Locates the Roblox Player application window, monitors movement, resizing,
and fullscreen transitions, and filters out non-game overlay windows.
"""

from dataclasses import dataclass
import logging
import os
from typing import Optional, List

logger = logging.getLogger("window_tracker")

try:
    import Quartz
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False
    logger.warning("Quartz not available for window tracking.")


@dataclass
class WindowInfo:
    window_id: int
    owner_name: str
    window_name: str
    x: int
    y: int
    width: int
    height: int
    is_on_screen: bool


class WindowTracker:
    """
    Locates and tracks the Roblox Player window on macOS across moves and resizes.
    """

    TARGET_OWNERS = ["RobloxPlayer", "Roblox", "RobloxStudio"]

    def __init__(self, target_owner: Optional[str] = None):
        self.target_owner = target_owner
        self.last_known_window: Optional[WindowInfo] = None

    def find_target_window(self) -> Optional[WindowInfo]:
        """
        Queries Quartz for the top-most visible Roblox Player window.
        """
        if not QUARTZ_AVAILABLE:
            return None

        try:
            # Query on-screen visible windows
            options = (
                Quartz.kCGWindowListOptionOnScreenOnly |
                Quartz.kCGWindowListExcludeDesktopElements
            )
            window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

            current_pid = os.getpid()
            candidates: List[WindowInfo] = []
            for win in window_list:
                owner = str(win.get(Quartz.kCGWindowOwnerName, ""))
                name = str(win.get(Quartz.kCGWindowName, ""))
                bounds_dict = win.get(Quartz.kCGWindowBounds, {})
                layer = win.get(Quartz.kCGWindowLayer, 0)
                is_on_screen = bool(win.get(Quartz.kCGWindowIsOnscreen, True))
                win_pid = win.get(Quartz.kCGWindowOwnerPID, 0)

                # Ignore windows from our own assistant process or with overlay titles
                if win_pid == current_pid:
                    continue
                if any(x in owner.lower() or x in name.lower() for x in ["anime card", "assistant overlay", "calibration"]):
                    continue

                # Ignore non-standard window layers (like menus, popups, and floating HUDs)
                if layer != 0:
                    continue

                is_match = False
                if self.target_owner:
                    is_match = (self.target_owner.lower() in owner.lower())
                else:
                    is_match = any(t.lower() in owner.lower() for t in self.TARGET_OWNERS)

                if is_match and bounds_dict:
                    w = int(bounds_dict.get("Width", 0))
                    h = int(bounds_dict.get("Height", 0))
                    # Ignore tiny helper/menu/dock windows
                    if w >= 320 and h >= 240:
                        candidates.append(WindowInfo(
                            window_id=int(win.get(Quartz.kCGWindowNumber, 0)),
                            owner_name=owner,
                            window_name=name,
                            x=int(bounds_dict.get("X", 0)),
                            y=int(bounds_dict.get("Y", 0)),
                            width=w,
                            height=h,
                            is_on_screen=is_on_screen
                        ))

            if candidates:
                # Select the largest window by area (main game viewport)
                candidates.sort(key=lambda item: item.width * item.height, reverse=True)
                self.last_known_window = candidates[0]
                return self.last_known_window

        except Exception as e:
            logger.debug("Failed querying Quartz window list: %s", e)

        return None

    def track(self) -> Optional[WindowInfo]:
        """
        Polls the current position of the target window.
        Returns updated WindowInfo if found, or None if lost/closed.
        """
        win = self.find_target_window()
        if win is not None:
            if self.last_known_window is not None:
                if (win.x != self.last_known_window.x or
                    win.y != self.last_known_window.y or
                    win.width != self.last_known_window.width or
                    win.height != self.last_known_window.height):
                    logger.info("Roblox window changed: bounds (%d, %d, %d, %d)",
                                win.x, win.y, win.width, win.height)
            self.last_known_window = win
        return win

    def is_active(self) -> bool:
        """Returns True if the target window is currently tracked and on-screen."""
        win = self.find_target_window()
        return win is not None and win.is_on_screen

