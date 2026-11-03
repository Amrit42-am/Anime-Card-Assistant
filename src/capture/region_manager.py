"""
src/capture/region_manager.py

Coordinates and calibrated regions manager.
Stores both absolute monitor coordinates and relative proportions (percentages)
to accommodate macOS Retina scaling, window dragging, and screen resolution changes.
"""

from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("region_manager")

DEFAULT_CALIBRATION_PATH = Path("data/calibration.json")


@dataclass
class RectRegion:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RectRegion":
        return cls(x=int(d["x"]), y=int(d["y"]), w=int(d["w"]), h=int(d["h"]))


@dataclass
class RelativeRegion:
    """Relative coordinates (0.0 to 1.0) relative to the parent game window."""
    rx: float
    ry: float
    rw: float
    rh: float

    def to_absolute(self, parent: RectRegion) -> RectRegion:
        return RectRegion(
            x=int(parent.x + self.rx * parent.w),
            y=int(parent.y + self.ry * parent.h),
            w=max(10, int(self.rw * parent.w)),
            h=max(10, int(self.rh * parent.h)),
        )


class CalibrationConfig:
    """Main calibration configuration model."""

    def __init__(
        self,
        game_window: Optional[RectRegion] = None,
        card_region: Optional[RelativeRegion] = None,
        options_region: Optional[RelativeRegion] = None,
        timer_region: Optional[RelativeRegion] = None,
        banner_region: Optional[RelativeRegion] = None,
        retina_scale: float = 2.0,
    ):
        # Default fallback window: left half of standard 1440x900 / 1920x1080 display
        self.game_window = game_window or RectRegion(x=50, y=80, w=850, h=650)
        # Defaults based on standard Roblox Guess the Card UI
        # Card artwork occupies central-left area
        self.card_relative = card_region or RelativeRegion(rx=0.08, ry=0.22, rw=0.45, rh=0.68)
        # 4 options occupy right side in a 2x2 grid (excluding the banner above)
        self.options_relative = options_region or RelativeRegion(rx=0.55, ry=0.28, rw=0.40, rh=0.62)
        # Timer bar at top of game
        self.timer_relative = timer_region or RelativeRegion(rx=0.30, ry=0.08, rw=0.40, rh=0.12)
        # Result / Instruction banner directly above the answer buttons
        self.banner_relative = banner_region or RelativeRegion(rx=0.55, ry=0.18, rw=0.40, rh=0.08)
        self.retina_scale = retina_scale

    def get_absolute_card_region(self) -> RectRegion:
        return self.card_relative.to_absolute(self.game_window)

    def get_absolute_options_region(self) -> RectRegion:
        return self.options_relative.to_absolute(self.game_window)

    def get_absolute_timer_region(self) -> RectRegion:
        return self.timer_relative.to_absolute(self.game_window)

    def get_absolute_banner_region(self) -> RectRegion:
        return self.banner_relative.to_absolute(self.game_window)

    def get_option_button_regions(self) -> list[RectRegion]:
        """
        Divides the options area into a 2x2 grid of four independent quadrant crops:
        1. Top-left
        2. Top-right
        3. Bottom-left
        4. Bottom-right
        """
        opts = self.get_absolute_options_region()
        half_w = opts.w // 2
        half_h = opts.h // 2
        pad = 4

        return [
            # 1. Top-Left
            RectRegion(x=opts.x + pad, y=opts.y + pad, w=half_w - pad * 2, h=half_h - pad * 2),
            # 2. Top-Right
            RectRegion(x=opts.x + half_w + pad, y=opts.y + pad, w=opts.w - half_w - pad * 2, h=half_h - pad * 2),
            # 3. Bottom-Left
            RectRegion(x=opts.x + pad, y=opts.y + half_h + pad, w=half_w - pad * 2, h=opts.h - half_h - pad * 2),
            # 4. Bottom-Right
            RectRegion(x=opts.x + half_w + pad, y=opts.y + half_h + pad, w=opts.w - half_w - pad * 2, h=opts.h - half_h - pad * 2),
        ]

    def save(self, path: Path = DEFAULT_CALIBRATION_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "game_window": self.game_window.to_dict(),
            "card_relative": asdict(self.card_relative),
            "options_relative": asdict(self.options_relative),
            "timer_relative": asdict(self.timer_relative),
            "banner_relative": asdict(self.banner_relative),
            "retina_scale": self.retina_scale
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration to %s", path)

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATION_PATH) -> "CalibrationConfig":
        if not path.exists():
            logger.info("No saved calibration found at %s. Using default configuration.", path)
            cfg = cls()
            cfg.save(path)
            return cfg

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            banner_rel = (
                RelativeRegion(**data["banner_relative"])
                if "banner_relative" in data
                else None
            )
            return cls(
                game_window=RectRegion.from_dict(data["game_window"]),
                card_region=RelativeRegion(**data["card_relative"]),
                options_region=RelativeRegion(**data["options_relative"]),
                timer_region=RelativeRegion(**data["timer_relative"]),
                banner_region=banner_rel,
                retina_scale=data.get("retina_scale", 2.0)
            )
        except Exception as e:
            logger.error("Failed to load calibration from %s (%s). Falling back to defaults.", path, e)
            return cls()
