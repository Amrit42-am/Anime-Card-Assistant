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
        retina_scale: float = 2.0,
    ):
        # Default fallback window: left half of standard 1440x900 / 1920x1080 display
        self.game_window = game_window or RectRegion(x=50, y=80, w=850, h=650)
        # Defaults based on standard Roblox Guess the Card UI
        # Card artwork occupies central-left area
        self.card_relative = card_region or RelativeRegion(rx=0.08, ry=0.22, rw=0.45, rh=0.68)
        # 4 options occupy right side
        self.options_relative = options_region or RelativeRegion(rx=0.55, ry=0.25, rw=0.40, rh=0.65)
        # Timer / question banner at top
        self.timer_relative = timer_region or RelativeRegion(rx=0.30, ry=0.08, rw=0.40, rh=0.12)
        self.retina_scale = retina_scale

    def get_absolute_card_region(self) -> RectRegion:
        return self.card_relative.to_absolute(self.game_window)

    def get_absolute_options_region(self) -> RectRegion:
        return self.options_relative.to_absolute(self.game_window)

    def get_absolute_timer_region(self) -> RectRegion:
        return self.timer_relative.to_absolute(self.game_window)

    def get_option_button_regions(self) -> list[RectRegion]:
        """Divides the options area into 4 vertical button slots."""
        opts = self.get_absolute_options_region()
        btn_h = int(opts.h / 4)
        buttons = []
        for i in range(4):
            y_start = opts.y + i * btn_h
            # Add small padding between buttons
            buttons.append(RectRegion(
                x=opts.x + 4,
                y=y_start + 4,
                w=opts.w - 8,
                h=btn_h - 8
            ))
        return buttons

    def save(self, path: Path = DEFAULT_CALIBRATION_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "game_window": self.game_window.to_dict(),
            "card_relative": asdict(self.card_relative),
            "options_relative": asdict(self.options_relative),
            "timer_relative": asdict(self.timer_relative),
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
            return cls(
                game_window=RectRegion.from_dict(data["game_window"]),
                card_region=RelativeRegion(**data["card_relative"]),
                options_region=RelativeRegion(**data["options_relative"]),
                timer_region=RelativeRegion(**data["timer_relative"]),
                retina_scale=data.get("retina_scale", 2.0)
            )
        except Exception as e:
            logger.error("Failed to load calibration from %s (%s). Falling back to defaults.", path, e)
            return cls()
