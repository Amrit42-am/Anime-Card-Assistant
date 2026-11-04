"""
src/memory/blur_memory.py

Phase 6: Learned Blur Memory System.
Caches confirmed game blur samples in RAM and persists both images and feature vectors.
Supports candidate-restricted similarity search, manual corrections, and sample invalidation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np

from src.memory.database import MemoryDatabase

logger = logging.getLogger("blur_memory")

LEARNED_DIR = Path("data/learned_blurs")
MEMORY_CACHE_NPZ = LEARNED_DIR / "memory_embeddings.npz"


class BlurMemorySystem:
    """Manages learned blurred card samples, embeddings cache, and fast retrieval."""

    def __init__(
        self,
        db: Optional[MemoryDatabase] = None,
        storage_dir: Path = LEARNED_DIR,
        encoder: Optional[Any] = None
    ):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or MemoryDatabase()
        self.encoder = encoder

        # In-memory arrays for fast matrix multiplication
        self.sample_ids: List[int] = []
        self.card_ids: List[int] = []
        self.card_names: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)

        self.load_memory()

    def load_memory(self):
        """Loads all active confirmed samples from SQLite and syncs embeddings."""
        records = self.db.get_all_confirmed()
        if not records:
            logger.info("Learned blur memory is currently empty (0 samples).")
            self.sample_ids = []
            self.card_ids = []
            self.card_names = []
            self.embeddings = np.empty((0, 384), dtype=np.float32)
            return

        # Check if cached npz exists and matches record count
        valid_records = []
        loaded_embs = []

        if MEMORY_CACHE_NPZ.exists():
            try:
                cached = np.load(MEMORY_CACHE_NPZ, allow_pickle=True)
                cached_ids = list(cached["sample_ids"])
                cached_embs = cached["embeddings"]

                id_to_emb = {s_id: emb for s_id, emb in zip(cached_ids, cached_embs)}
                for r in records:
                    if r.sample_id in id_to_emb:
                        valid_records.append(r)
                        loaded_embs.append(id_to_emb[r.sample_id])
                    elif self.encoder and Path(r.image_path).exists():
                        # Encode newly added sample if missing from cache
                        emb = self.encoder.encode_image(r.image_path)
                        valid_records.append(r)
                        loaded_embs.append(emb)
            except Exception as e:
                logger.warning("Failed loading memory cache npz (%s), rebuilding...", e)

        if not loaded_embs and self.encoder:
            # Rebuild from image files
            for r in records:
                p = Path(r.image_path)
                if p.exists():
                    try:
                        emb = self.encoder.encode_image(p)
                        valid_records.append(r)
                        loaded_embs.append(emb)
                    except Exception as err:
                        logger.warning("Could not encode sample %s: %s", p, err)

        self.sample_ids = [r.sample_id for r in valid_records]
        self.card_ids = [r.card_id for r in valid_records]
        self.card_names = [r.card_name for r in valid_records]

        if loaded_embs:
            self.embeddings = np.vstack(loaded_embs).astype(np.float32)
            self._save_cache_npz()
        else:
            self.embeddings = np.empty((0, 384), dtype=np.float32)

        logger.info("Learned memory loaded %d active confirmed blur samples.", len(self.sample_ids))

    def _save_cache_npz(self):
        """Persists current in-memory embeddings to NPZ."""
        if len(self.sample_ids) > 0 and len(self.embeddings) > 0:
            np.savez_compressed(
                MEMORY_CACHE_NPZ,
                sample_ids=np.array(self.sample_ids, dtype=np.int32),
                card_ids=np.array(self.card_ids, dtype=np.int32),
                card_names=np.array(self.card_names, dtype=object),
                embeddings=self.embeddings
            )

    def match_sample(
        self,
        query_embedding: np.ndarray,
        allowed_card_ids: Optional[List[int]] = None,
        min_threshold: float = 0.80
    ) -> Optional[Dict[str, Any]]:
        """
        Search learned blur memory.
        If allowed_card_ids is provided (e.g. 4 OCR options), restricts search
        strictly to those candidate cards to avoid false positives!
        """
        if len(self.sample_ids) == 0 or len(self.embeddings) == 0:
            return None

        query_embedding = query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding /= q_norm

        # Filter indices by allowed_card_ids if provided
        if allowed_card_ids is not None:
            allowed_set = set(allowed_card_ids)
            indices = [i for i, cid in enumerate(self.card_ids) if cid in allowed_set]
            if not indices:
                return None
        else:
            indices = list(range(len(self.sample_ids)))

        sub_embeddings = self.embeddings[indices]  # (K, D)
        scores = np.dot(sub_embeddings, query_embedding)  # (K,)

        best_sub_idx = int(np.argmax(scores))
        best_score = float(scores[best_sub_idx])

        if best_score < min_threshold:
            return None

        best_orig_idx = indices[best_sub_idx]

        # Second best score among different card IDs
        second_score = 0.0
        for s_idx, idx in enumerate(indices):
            if self.card_ids[idx] != self.card_ids[best_orig_idx]:
                if float(scores[s_idx]) > second_score:
                    second_score = float(scores[s_idx])

        return {
            "matched": True,
            "sample_id": self.sample_ids[best_orig_idx],
            "card_id": self.card_ids[best_orig_idx],
            "card_name": self.card_names[best_orig_idx],
            "confidence": best_score,
            "second_confidence": second_score,
            "margin": best_score - second_score
        }

    def is_duplicate_sample(
        self,
        card_id: int,
        embedding: np.ndarray,
        blur_hash: Optional[str] = None,
        sim_threshold: float = 0.985
    ) -> bool:
        """Checks if an identical or nearly identical sample already exists in memory."""
        if blur_hash and self.db.has_blur_hash(card_id, blur_hash):
            return True

        if len(self.card_ids) == 0 or len(self.embeddings) == 0:
            return False

        same_card_indices = [i for i, cid in enumerate(self.card_ids) if cid == card_id]
        if not same_card_indices:
            return False

        sub_embs = self.embeddings[same_card_indices]
        sims = np.dot(sub_embs, embedding)
        return float(np.max(sims)) >= sim_threshold

    def add_confirmed_sample(
        self,
        card_id: int,
        card_name: str,
        blurred_image: np.ndarray,
        clear_image: Optional[np.ndarray] = None,
        options: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        source: str = "user_confirmed",
        verification_status: str = "VERIFIED",
        blur_hash: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: str = ""
    ) -> Optional[int]:
        """
        Saves a confirmed blurred screenshot (and optional revealed clear screenshot) from game play.
        CRITICAL RULE: Confirmed samples must be verified (either automatically via green button
        cross-checked with clear image or manually by the user).
        Checks for duplicates before persisting.
        """
        import json

        if embedding is None:
            if not self.encoder:
                raise ValueError("Encoder required when adding sample without precomputed embedding.")
            embedding = self.encoder.encode_image(blurred_image)

        embedding = embedding.astype(np.float32).flatten()
        emb_norm = np.linalg.norm(embedding)
        if emb_norm > 0:
            embedding /= emb_norm

        # Duplicate detection check
        if self.is_duplicate_sample(card_id, embedding, blur_hash):
            logger.info("Sample for card #%d %s is already present in memory (duplicate). Skipping save.", card_id, card_name)
            return None

        card_folder = self.storage_dir / f"card_{card_id:03d}"
        card_folder.mkdir(parents=True, exist_ok=True)

        # Temporary filename until inserted
        temp_img_path = card_folder / "temp.png"
        cv2.imwrite(str(temp_img_path), blurred_image)

        options_json = json.dumps(options) if options else None

        sample_id = self.db.insert_sample(
            card_id=card_id,
            card_name=card_name,
            image_path=str(temp_img_path),
            clear_image_path=None,
            is_confirmed=(verification_status == "VERIFIED"),
            confidence=confidence,
            source=source,
            verification_status=verification_status,
            options_json=options_json,
            blur_hash=blur_hash,
            session_id=session_id,
            notes=notes
        )

        final_blur_path = card_folder / f"sample_{sample_id:04d}_blur.png"
        temp_img_path.replace(final_blur_path)

        final_clear_path = None
        if clear_image is not None and clear_image.size > 0:
            final_clear_path = card_folder / f"sample_{sample_id:04d}_clear.png"
            cv2.imwrite(str(final_clear_path), clear_image)

        # Update DB with final paths
        with self.db._get_connection() as conn:
            conn.cursor().execute(
                "UPDATE learned_blurs SET image_path = ?, clear_image_path = ? WHERE sample_id = ?;",
                (str(final_blur_path), str(final_clear_path) if final_clear_path else None, sample_id)
            )
            conn.commit()

        # Update in-memory state only if verified
        if verification_status == "VERIFIED":
            self.sample_ids.append(sample_id)
            self.card_ids.append(card_id)
            self.card_names.append(card_name)
            if len(self.embeddings) == 0:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])

            self._save_cache_npz()
            logger.info("Saved verified blur memory for #%d %s (Sample ID: %d, Source: %s)", card_id, card_name, sample_id, source)
        else:
            logger.warning("Saved sample #%d %s as UNVERIFIED (Sample ID: %d, Disagreement recorded)", card_id, card_name, sample_id)

        return sample_id

    def correct_sample(self, sample_id: int, correct_card_id: int, correct_card_name: str):
        """Corrects an erroneously identified sample."""
        if sample_id not in self.sample_ids:
            logger.warning("Sample %d not found in active memory.", sample_id)
            return

        self.db.update_card_identity(sample_id, correct_card_id, correct_card_name)
        idx = self.sample_ids.index(sample_id)
        self.card_ids[idx] = correct_card_id
        self.card_names[idx] = correct_card_name
        self._save_cache_npz()
        logger.info("Corrected sample %d to #%d %s", sample_id, correct_card_id, correct_card_name)

    def invalidate_sample(self, sample_id: int):
        """Invalidates sample so it is never used in recognition."""
        if sample_id in self.sample_ids:
            idx = self.sample_ids.index(sample_id)
            self.db.invalidate_sample(sample_id)
            self.sample_ids.pop(idx)
            self.card_ids.pop(idx)
            self.card_names.pop(idx)
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
            self._save_cache_npz()
            logger.info("Invalidated sample %d from blur memory.", sample_id)

    @property
    def total_samples(self) -> int:
        return len(self.sample_ids)

    def record_disagreement(
        self,
        session_id: str,
        green_card_id: int,
        green_card_name: str,
        clear_matched_id: int,
        clear_matched_name: str,
        clear_confidence: float = 0.0,
        options: Optional[List[str]] = None,
        notes: str = ""
    ) -> int:
        """Preserves disagreement records separately for debugging without polluting confirmed memory."""
        import json
        opt_json = json.dumps(options) if options else None
        return self.db.log_disagreement(
            session_id=session_id,
            green_card_id=green_card_id,
            green_card_name=green_card_name,
            clear_matched_id=clear_matched_id,
            clear_matched_name=clear_matched_name,
            clear_confidence=clear_confidence,
            options_json=opt_json,
            notes=notes
        )

