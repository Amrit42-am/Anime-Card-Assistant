"""
src/ocr/option_reader.py

Phase 5: Option Reader using macOS Apple Vision Framework.
Extracts candidate card names from the 4 quiz option buttons,
normalizes wrapped text and common OCR artifacts, and fuzzy-matches
against the canonical 286 card database.
"""

from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from rapidfuzz import process, fuzz
import shutil
import os

from src.memory.self_learning_db import SelfLearningDatabase

logger = logging.getLogger("option_reader")

try:
    import Vision
    from Foundation import NSData
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("macOS Vision framework not available. PyObjC Vision required for OCR.")


def find_tesseract_executable() -> Optional[str]:
    """Finds available tesseract binary in PATH or common macOS Homebrew paths."""
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
        if Path(p).exists() and os.access(p, os.X_OK):
            return p
    return None


TESSERACT_AVAILABLE = False
_TESSERACT_INITIALIZED = False

try:
    import pytesseract
    _t_cmd = find_tesseract_executable()
    if _t_cmd:
        pytesseract.pytesseract.tesseract_cmd = _t_cmd
        TESSERACT_AVAILABLE = True
    else:
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False


SLOT_NAMES = ["TL", "TR", "BL", "BR"]


@dataclass
class CandidateOption:
    slot_index: int
    raw_text: str
    canonical_name: str
    card_id: Optional[int]
    ocr_confidence: float = 0.0
    fuzzy_score: float = 0.0
    is_valid: bool = False
    position: str = ""

    def __post_init__(self):
        if not self.position and 0 <= self.slot_index < len(SLOT_NAMES):
            self.position = SLOT_NAMES[self.slot_index]

    def to_dict(self) -> Dict[str, Any]:
        pos = self.position or (SLOT_NAMES[self.slot_index] if 0 <= self.slot_index < len(SLOT_NAMES) else "")
        return {
            "slot_index": self.slot_index,
            "position": pos,
            "raw_text": self.raw_text,
            "canonical_name": self.canonical_name,
            "card_id": self.card_id,
            "ocr_confidence": round(float(self.ocr_confidence), 3),
            "fuzzy_score": round(float(self.fuzzy_score), 2),
            "is_valid": bool(self.is_valid)
        }


def normalize_detected_options(candidates: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes candidate options into exactly 4 fixed slots [TL, TR, BL, BR].
    Guarantees that index 0 is always TL, 1 is always TR, 2 is always BL, and 3 is always BR.
    Missing or unmapped slots are represented as invalid empty entries.
    """
    slot_map: Dict[int, Dict[str, Any]] = {}
    if candidates:
        for c in candidates:
            if hasattr(c, "to_dict"):
                d = c.to_dict()
            elif isinstance(c, dict):
                d = dict(c)
            else:
                continue

            idx = d.get("slot_index")
            if idx is not None and 0 <= idx < 4:
                if not d.get("position"):
                    d["position"] = SLOT_NAMES[idx]
                slot_map[idx] = d

    out: List[Dict[str, Any]] = []
    for idx in range(4):
        if idx in slot_map:
            out.append(slot_map[idx])
        else:
            out.append({
                "slot_index": idx,
                "position": SLOT_NAMES[idx],
                "raw_text": "",
                "canonical_name": "",
                "card_id": None,
                "ocr_confidence": 0.0,
                "fuzzy_score": 0.0,
                "is_valid": False
            })
    return out


class OptionReader:
    """Reads and parses the 4 candidate card names from quiz button regions."""

    def __init__(
        self,
        cards_json_path: Optional[Path] = None,
        db: Optional[SelfLearningDatabase] = None
    ):
        self.cards: List[Dict[str, Any]] = []
        self.canonical_names: List[str] = []
        self.name_to_id: Dict[str, int] = {}
        self.db = db

        # 1. Load from database if available
        if self.db is None:
            self.db = SelfLearningDatabase()
        db_cards = self.db.get_all_cards()
        if db_cards:
            self.canonical_names = [c.canonical_name for c in db_cards]
            self.name_to_id = {c.canonical_name.lower().strip(): c.card_id for c in db_cards}
            logger.info("OptionReader initialized with %d canonical cards from database.", len(self.canonical_names))

        # 2. Backward compatibility: load from cards_json_path if provided and exists
        if cards_json_path and Path(cards_json_path).exists():
            with open(cards_json_path, "r", encoding="utf-8") as f:
                self.cards = json.load(f)
            for c in self.cards:
                cname = c["name"]
                cid = c["id"]
                if cname not in self.canonical_names:
                    self.canonical_names.append(cname)
                self.name_to_id[cname.lower().strip()] = cid
            logger.info("OptionReader loaded %d canonical cards from %s.", len(self.cards), cards_json_path)

        if not self.canonical_names:
            logger.info("OptionReader initialized with 0 canonical cards (Cold Start).")

    def register_card(self, card_id: int, canonical_name: str):
        """Dynamically registers a newly discovered card into known canonical names."""
        clean_name = canonical_name.strip()
        lower_name = clean_name.lower()
        if clean_name not in self.canonical_names:
            self.canonical_names.append(clean_name)
        self.name_to_id[lower_name] = card_id

    def preprocess_button_image(self, img: np.ndarray) -> np.ndarray:
        """
        Enhances text contrast and readability for game buttons.
        """
        if img is None or img.size == 0:
            return img

        # Upscale small crops for higher OCR fidelity
        h, w = img.shape[:2]
        if h < 60:
            scale = 2.0
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def retry_preprocess_button_image(self, img: np.ndarray) -> np.ndarray:
        """
        Alternative aggressive preprocessing for degraded or difficult button text:
        Upscaling (2.5x), higher contrast CLAHE (3.5), and Otsu thresholding.
        """
        if img is None or img.size == 0:
            return img

        h, w = img.shape[:2]
        scale = 2.5
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized.copy()

        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        contrast = clahe.apply(gray)
        blur = cv2.GaussianBlur(contrast, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh


    def extract_text_fallback(self, image: np.ndarray) -> List[Tuple[str, float, float]]:
        """Fallback OCR using pytesseract when Vision fails.
        Only invokes pytesseract when the executable is verified to exist.
        """
        global _TESSERACT_INITIALIZED
        if not _TESSERACT_INITIALIZED:
            _TESSERACT_INITIALIZED = True
            if not TESSERACT_AVAILABLE:
                logger.info(
                    "Tesseract binary not detected in PATH or Homebrew (/opt/homebrew/bin/tesseract, /usr/local/bin/tesseract). "
                    "Native Apple Vision will serve as primary OCR engine. "
                    "To enable optional Tesseract fallback: brew install tesseract"
                )

        if not TESSERACT_AVAILABLE or image is None or image.size == 0:
            return []

        # Ensure image is 3-channel RGB for pytesseract
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        try:
            # Get raw string and confidence data
            data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
            texts = []
            for txt, conf in zip(data.get('text', []), data.get('conf', [])):
                if txt.strip():
                    # pytesseract conf is 0-100; normalize to 0-1
                    confidence = float(conf) / 100.0 if conf != -1 else 0.0
                    texts.append((txt.strip(), confidence, 0.0))
            return texts
        except Exception as e:
            logger.debug("pytesseract fallback notice: %s", e)
            return []

    def extract_text_vision(
        self,
        image: np.ndarray,
        recognition_level: Optional[int] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Invokes Apple Vision framework on in-memory image.
        Returns list of (recognized_text, confidence, y_coord_bottom).
        Vision coordinates have origin at bottom-left (y=0 is bottom, y=1 is top).
        """
        if not VISION_AVAILABLE or image is None or image.size == 0:
            return []

        # Convert to PNG bytes
        success, encoded = cv2.imencode(".png", image)
        if not success:
            return []

        data_bytes = encoded.tobytes()
        nsdata = NSData.dataWithBytes_length_(data_bytes, len(data_bytes))

        request = Vision.VNRecognizeTextRequest.alloc().init()
        level = recognition_level if recognition_level is not None else Vision.VNRequestTextRecognitionLevelFast
        request.setRecognitionLevel_(level)
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(nsdata, None)
        ok, err = handler.performRequests_error_([request], None)

        results = []
        if ok and request.results():
            for obs in request.results():
                top = obs.topCandidates_(1)[0]
                text = top.string().strip()
                conf = float(top.confidence())
                box = obs.boundingBox()
                # y in Vision: 0 at bottom, 1 at top. We invert so top is 0 for intuitive sorting
                norm_y = 1.0 - (box.origin.y + box.size.height)
                results.append((text, conf, norm_y))

        return results

    def clean_text(self, text: str) -> str:
        """Strips noise, button numbers, hotkey letters, and stray punctuation."""
        # Replace newlines with spaces
        text = text.replace("\n", " ").replace("\r", " ")
        # Strip leading punctuation/bullets/spaces
        text = re.sub(r"^[\s\-\*\•\>\:\.\,]+", "", text)
        # Strip leading option markers like '1. ', 'A) ', '[B] ', '4: ', '[TL] ', 'TL: '
        text = re.sub(r"^(?:\[[0-9A-Za-z]+\]|\([0-9A-Za-z]+\)|[0-9A-Za-z]{1,2}[\.\)\:\-])\s*", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def resolve_canonical_name(
        self,
        raw_text: str,
        threshold: float = 65.0,
        min_margin: float = 4.0
    ) -> Tuple[str, Optional[int], float]:
        """
        Fuzzy matches extracted text to canonical card names.
        Enforces threshold and winning margin to prevent assigning arbitrary closest match.
        Returns (canonical_name, card_id, score).
        """
        clean = self.clean_text(raw_text)
        if not clean or not self.canonical_names:
            return raw_text, None, 0.0

        lower_clean = clean.lower()
        if lower_clean in self.name_to_id:
            for name in self.canonical_names:
                if name.lower() == lower_clean:
                    return name, self.name_to_id[lower_clean], 100.0

        top_matches = process.extract(
            clean,
            self.canonical_names,
            scorer=fuzz.token_sort_ratio,
            limit=2
        )
        if top_matches:
            best_name, best_score, _ = top_matches[0]
            second_score = top_matches[1][1] if len(top_matches) > 1 else 0.0

            # Require threshold AND winning margin over runner-up (unless score is very high >= 85.0)
            if best_score >= threshold and (best_score >= 85.0 or (best_score - second_score) >= min_margin):
                card_id = self.name_to_id.get(best_name.lower().strip())
                return best_name, card_id, float(best_score)

        return clean, None, float(top_matches[0][1]) if top_matches else 0.0

    def read_options_from_buttons(
        self,
        button_frames: List[np.ndarray],
        use_retry: bool = False
    ) -> List[CandidateOption]:
        """
        Reads candidate names from 4 pre-sliced button images.
        """
        candidates: List[CandidateOption] = []

        for i, btn in enumerate(button_frames):
            preprocessed = (
                self.retry_preprocess_button_image(btn)
                if use_retry
                else self.preprocess_button_image(btn)
            )
            level = Vision.VNRequestTextRecognitionLevelAccurate if (VISION_AVAILABLE and use_retry) else None
            vision_results = self.extract_text_vision(preprocessed, recognition_level=level)

            # If enhanced preprocessed image produced no text, try directly on raw button crop
            if not vision_results and btn is not None and btn.size > 0:
                vision_results = self.extract_text_vision(btn, recognition_level=level)

            # Fallback to pytesseract when Vision yields no results and tesseract executable exists
            if not vision_results:
                vision_results = self.extract_text_fallback(preprocessed)
            if not vision_results and btn is not None and btn.size > 0:
                vision_results = self.extract_text_fallback(btn)

            if vision_results:
                # Sort lines top-to-bottom
                vision_results.sort(key=lambda item: item[2])
                raw_text = " ".join([t[0] for t in vision_results])
                avg_conf = float(np.mean([t[1] for t in vision_results]))
            else:
                raw_text = ""
                avg_conf = 0.0

            canonical, cid, fuzzy_score = self.resolve_canonical_name(raw_text)
            clean_str = self.clean_text(raw_text)

            # Validation logic:
            # 1. If known card with good fuzzy score
            if cid is not None and fuzzy_score >= 65.0:
                is_valid = True
                display_name = canonical
            # 2. If unseen card with clean text >= 3 chars and sufficient OCR confidence
            elif len(clean_str) >= 3 and avg_conf >= 0.25:
                is_valid = True
                display_name = clean_str
            else:
                is_valid = False
                display_name = clean_str if clean_str else raw_text

            candidates.append(CandidateOption(
                slot_index=i,
                raw_text=raw_text,
                canonical_name=display_name,
                card_id=cid,
                ocr_confidence=avg_conf,
                fuzzy_score=fuzzy_score,
                is_valid=is_valid
            ))

        return candidates

    def read_options_from_panel(self, options_panel: np.ndarray) -> List[CandidateOption]:
        """
        Reads candidate names from the combined options panel using accurate
        2x2 button grid extraction (TL, TR, BL, BR), never horizontal strip splitting.
        """
        if options_panel is None or options_panel.size == 0:
            return []

        from src.capture.screen_capture import ScreenCapture
        button_frames, _, _ = ScreenCapture.extract_four_buttons(options_panel)
        if not button_frames or len(button_frames) != 4:
            return []
        return self.read_options_from_buttons(button_frames)