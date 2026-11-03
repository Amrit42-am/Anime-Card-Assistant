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

logger = logging.getLogger("option_reader")

try:
    import Vision
    import pytesseract
    from Foundation import NSData
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("macOS Vision framework not available. PyObjC Vision required for OCR.")


@dataclass
class CandidateOption:
    slot_index: int
    raw_text: str
    canonical_name: str
    card_id: Optional[int]
    ocr_confidence: float
    fuzzy_score: float
    is_valid: bool


class OptionReader:
    """Reads and parses the 4 candidate card names from quiz button regions."""

    def __init__(self, cards_json_path: Path = Path("data/cards.json")):
        self.cards: List[Dict[str, Any]] = []
        self.canonical_names: List[str] = []
        self.name_to_id: Dict[str, int] = {}

        if cards_json_path.exists():
            with open(cards_json_path, "r", encoding="utf-8") as f:
                self.cards = json.load(f)
            self.canonical_names = [c["name"] for c in self.cards]
            self.name_to_id = {c["name"].lower().strip(): c["id"] for c in self.cards}
            logger.info("OptionReader initialized with %d canonical cards.", len(self.canonical_names))
        else:
            logger.warning("cards.json not found at %s. Canonical resolution disabled.", cards_json_path)

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
        Returns a list of (text, confidence, dummy_y) where confidence is the average
        confidence reported by pytesseract (scaled 0-1). The y coordinate is set to 0.0
        because we cannot obtain it from pytesseract.
        """
        if image is None or image.size == 0:
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
            logger.error("pytesseract OCR failed: %s", e)
            return []
    def extract_text_vision(self, image: np.ndarray) -> List[Tuple[str, float, float]]:
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
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
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
        # Strip leading punctuation/bullets/spaces
        text = re.sub(r"^[\s\-\*\•\>\:\.\,]+", "", text)
        # Strip leading option markers like '1. ', 'A) ', '[B] ', '4: '
        text = re.sub(r"^(?:\[?[0-9A-Da-d][\.\)\:\-\]]|\([0-9A-Da-d]\))\s*", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def resolve_canonical_name(self, raw_text: str, threshold: float = 65.0) -> Tuple[str, Optional[int], float]:
        """
        Fuzzy matches extracted text to canonical card names.
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

        best = process.extractOne(
            clean,
            self.canonical_names,
            scorer=fuzz.token_sort_ratio
        )
        if best and best[1] >= threshold:
            canonical_name, score, _ = best
            card_id = self.name_to_id.get(canonical_name.lower().strip())
            return canonical_name, card_id, float(score)

        return clean, None, float(best[1]) if best else 0.0

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
            vision_results = self.extract_text_vision(preprocessed)

            # Fallback to pytesseract when Vision yields no results or low confidence
            if not vision_results:
                vision_results = self.extract_text_fallback(preprocessed)


            if vision_results:
                # Sort lines top-to-bottom
                vision_results.sort(key=lambda item: item[2])
                raw_text = " ".join([t[0] for t in vision_results])
                avg_conf = float(np.mean([t[1] for t in vision_results]))
            else:
                raw_text = ""
                avg_conf = 0.0

            canonical, cid, fuzzy_score = self.resolve_canonical_name(raw_text)
            is_valid = (cid is not None) and (fuzzy_score >= 65.0)

            candidates.append(CandidateOption(
                slot_index=i,
                raw_text=raw_text,
                canonical_name=canonical if is_valid else raw_text,
                card_id=cid,
                ocr_confidence=avg_conf,
                fuzzy_score=fuzzy_score,
                is_valid=is_valid
            ))

        return candidates

    def read_options_from_panel(self, options_panel: np.ndarray) -> List[CandidateOption]:
        """
        Reads candidate names from the combined options panel by either
        splitting into 4 vertical button slots or analyzing text bounding boxes.
        """
        if options_panel is None or options_panel.size == 0:
            return []

        h, w = options_panel.shape[:2]
        slot_h = h // 4
        button_frames = []
        for i in range(4):
            y1 = i * slot_h
            y2 = (i + 1) * slot_h if i < 3 else h
            btn = options_panel[y1 + 2:y2 - 2, 4:w - 4]
            button_frames.append(btn)

        return self.read_options_from_buttons(button_frames)