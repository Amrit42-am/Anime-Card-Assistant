"""
src/capture/display_monitor.py

Display monitoring and capture management on macOS using Quartz and MSS.
Enumerates connected displays, tracks retina scaling, and provides
high-performance full-screen frame capture without writing to disk.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Dict, Any
import cv2
import mss
import numpy as np

logger = logging.getLogger("display_monitor")

try:
    import Quartz
    import AppKit
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False
    logger.warning("Quartz/AppKit not available. Falling back to MSS display enumeration.")


@dataclass
class DisplayInfo:
    id: int
    name: str
    x: int
    y: int
    width: int
    height: int
    scale_factor: float
    is_primary: bool


class DisplayMonitor:
    """
    Enumerates connected monitors, tracks scaling factors, and provides
    per-display screen capture.
    """

    def __init__(self):
        self._sct = mss.MSS()
        self.displays: List[DisplayInfo] = []
        self.refresh_displays()

    @staticmethod
    def check_permissions() -> Tuple[bool, str]:
        """
        Verifies macOS Screen Recording permissions.
        Returns (has_permission, message).
        """
        if QUARTZ_AVAILABLE:
            try:
                has_access = Quartz.CGPreflightScreenCaptureAccess()
                if has_access:
                    return True, "Screen Recording permission granted."
                return False, (
                    "macOS Screen Recording permission is denied or not yet granted. "
                    "Please allow Anime Card Assistant in System Settings -> Privacy & Security -> Screen Recording."
                )
            except Exception as e:
                logger.warning("Quartz screen capture preflight check error: %s", e)

        # Fallback probe via MSS grab
        try:
            with mss.MSS() as s:
                sample = s.grab({"top": 0, "left": 0, "width": 10, "height": 10})
                if sample and len(sample.raw) > 0:
                    return True, "Screen Recording access verified via MSS."
        except Exception as e:
            return False, f"Permission verification failed: {e}"

        return True, "Screen capture initialized."

    @staticmethod
    def request_permissions() -> bool:
        """Triggers macOS Screen Recording permission prompt."""
        if QUARTZ_AVAILABLE:
            try:
                return Quartz.CGRequestScreenCaptureAccess()
            except Exception as e:
                logger.warning("Failed to invoke CGRequestScreenCaptureAccess: %s", e)
        return False

    def refresh_displays(self) -> List[DisplayInfo]:
        """Scans connected displays and updates display metadata."""
        displays: List[DisplayInfo] = []

        if QUARTZ_AVAILABLE:
            try:
                screens = AppKit.NSScreen.screens()
                max_displays = 16
                err, active_displays, count = Quartz.CGGetActiveDisplayList(max_displays, None, None)
                if err == 0 and count > 0:
                    main_id = Quartz.CGMainDisplayID()
                    for idx, disp_id in enumerate(active_displays):
                        bounds = Quartz.CGDisplayBounds(disp_id)
                        scale = 1.0
                        if idx < len(screens):
                            scale = float(screens[idx].backingScaleFactor())
                        
                        is_primary = (disp_id == main_id)
                        disp_info = DisplayInfo(
                            id=int(disp_id),
                            name=f"Display {idx + 1} ({int(bounds.size.width)}x{int(bounds.size.height)})",
                            x=int(bounds.origin.x),
                            y=int(bounds.origin.y),
                            width=int(bounds.size.width),
                            height=int(bounds.size.height),
                            scale_factor=scale,
                            is_primary=is_primary
                        )
                        displays.append(disp_info)
            except Exception as e:
                logger.warning("Error fetching Quartz display list: %s. Using MSS.", e)

        if not displays:
            # Fallback to MSS monitors
            for idx, mon in enumerate(self._sct.monitors[1:], start=1):
                displays.append(DisplayInfo(
                    id=idx,
                    name=f"Monitor {idx} ({mon['width']}x{mon['height']})",
                    x=mon["left"],
                    y=mon["top"],
                    width=mon["width"],
                    height=mon["height"],
                    scale_factor=1.0,
                    is_primary=(idx == 1)
                ))

        self.displays = displays
        return self.displays

    def get_primary_display(self) -> Optional[DisplayInfo]:
        """Returns the primary active display."""
        for disp in self.displays:
            if disp.is_primary:
                return disp
        return self.displays[0] if self.displays else None

    def capture_display(self, display_id: Optional[int] = None) -> np.ndarray:
        """
        Captures the specified display into an in-memory BGR numpy array.
        Never saves images to disk.
        """
        target = None
        if display_id is not None:
            for d in self.displays:
                if d.id == display_id:
                    target = d
                    break

        if target is None:
            target = self.get_primary_display()

        if target is None:
            # Absolute fallback
            raw = self._sct.grab(self._sct.monitors[0])
            bgra = np.array(raw)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        bbox = {
            "left": target.x,
            "top": target.y,
            "width": target.width,
            "height": target.height
        }
        raw = self._sct.grab(bbox)
        bgra = np.array(raw)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def close(self):
        self._sct.close()
