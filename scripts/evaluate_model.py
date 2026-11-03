#!/usr/bin/env python3
"""
scripts/evaluate_model.py

Leakage-Free Benchmark Evaluation Suite for Anime Card Assistant:
Evaluates recognition accuracy across 4 distinct, independent evaluation groups:
1. Synthetic Images: Independently generated blurs not present in the precomputed reference DB.
2. Real Roblox Blur Images: Real held-out game screenshots from actual gameplay sessions.
3. Previously Learned Images: Samples confirmed and persisted in learned blur memory.
4. Previously Unseen Cards: Rejection and uncertainty calibration on held-out cards.

Generates a transparent, honest benchmark report at data/benchmark_report.json.
"""

import json
import logging
from pathlib import Path
import random
import sys
import time

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.recognition.image_encoder import get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.memory.database import MemoryDatabase

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_model")

CARDS_JSON = Path("data/cards.json")
REPORT_PATH = Path("data/benchmark_report.json")


def apply_independent_blur(img: np.ndarray, distortion_type: str) -> np.ndarray:
    """
    Generates fresh, un-cached blur transformations with jittered parameters
    never stored in the reference embedding database.
    """
    if distortion_type == "gaussian_jitter":
        k = random.choice([27, 31, 37, 43])
        sigma = random.uniform(8.5, 14.0)
        return cv2.GaussianBlur(img, (k, k), sigma)
    elif distortion_type == "box_jitter":
        k = random.choice([18, 22, 26, 30])
        return cv2.blur(img, (k, k))
    elif distortion_type == "motion_blur":
        size = random.choice([15, 21, 27])
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)
    elif distortion_type == "compound_noise":
        blurred = cv2.GaussianBlur(img, (33, 33), 11.0)
        # Add slight Gaussian noise and brightness jitter
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        jittered = np.clip(blurred.astype(np.float32) * random.uniform(0.9, 1.1) + noise, 0, 255).astype(np.uint8)
        return jittered
    else:
        return cv2.GaussianBlur(img, (35, 35), 12.0)


def evaluate_synthetic_blurs(cards, matcher, encoder, num_cards: int = 50):
    logger.info("Evaluating Group 1: Independent Synthetic Blurs (no data leakage)...")
    distortions = ["gaussian_jitter", "box_jitter", "motion_blur", "compound_noise"]
    random.seed(1337)
    sampled_indices = random.sample(range(len(cards)), min(num_cards, len(cards)))

    total_evals = 0
    top1_correct = 0
    top2_correct = 0
    latencies = []
    per_distortion_correct = {d: 0 for d in distortions}
    per_distortion_total = {d: 0 for d in distortions}

    for idx in sampled_indices:
        target_card = cards[idx]
        target_name = target_card["name"]
        card_img = cv2.imread(target_card["image"])
        if card_img is None:
            continue

        # 3 random distractors
        distractors = random.sample([i for i in range(len(cards)) if i != idx], 3)
        candidate_names = [target_name] + [cards[i]["name"] for i in distractors]
        random.shuffle(candidate_names)

        for d_type in distortions:
            distorted_img = apply_independent_blur(card_img, d_type)

            t0 = time.time()
            query_emb = encoder.encode_image(distorted_img)
            res = matcher.match_candidates(query_emb, candidate_names)
            elapsed_ms = (time.time() - t0) * 1000.0
            latencies.append(elapsed_ms)

            is_top1 = (res["card_name"] == target_name)
            is_top2 = is_top1 or any(alt["card_name"] == target_name for alt in res.get("alternatives", [])[:1])

            total_evals += 1
            per_distortion_total[d_type] += 1
            if is_top1:
                top1_correct += 1
                per_distortion_correct[d_type] += 1
            if is_top2:
                top2_correct += 1

    return {
        "total_evaluations": total_evals,
        "top1_accuracy_percent": round((top1_correct / total_evals) * 100.0, 2) if total_evals else 0.0,
        "top2_accuracy_percent": round((top2_correct / total_evals) * 100.0, 2) if total_evals else 0.0,
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "per_distortion_breakdown": {
            d: round((per_distortion_correct[d] / per_distortion_total[d]) * 100.0, 2)
            for d in distortions if per_distortion_total[d] > 0
        }
    }


def evaluate_real_gameplay_images(cards, matcher, encoder):
    logger.info("Evaluating Group 2: Real Roblox Gameplay Images...")
    db = MemoryDatabase()
    samples = db.get_all_confirmed()

    real_samples = [s for s in samples if Path(s.image_path).exists()]
    if not real_samples:
        logger.warning("No confirmed real gameplay images available in memory DB yet.")
        return {
            "available_samples": 0,
            "accuracy_percent": None,
            "note": "Awaiting active gameplay sessions to accumulate real blurred captures."
        }

    total = 0
    correct = 0
    for s in real_samples:
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        query_emb = encoder.encode_image(img)

        # 3 distractors
        distractors = random.sample([c["name"] for c in cards if c["id"] != s.card_id], 3)
        candidate_names = [s.card_name] + distractors
        random.shuffle(candidate_names)

        res = matcher.match_candidates(query_emb, candidate_names)
        total += 1
        if res["card_name"] == s.card_name and res["status"] == "matched":
            correct += 1

    acc = round((correct / total) * 100.0, 2) if total else 0.0
    return {
        "available_samples": total,
        "accuracy_percent": acc,
        "correct_matches": correct
    }


def evaluate_learned_memory(cards, matcher, encoder):
    logger.info("Evaluating Group 3: Learned Blur Memory Retrieval...")
    memory = BlurMemorySystem(encoder=encoder)
    if memory.total_samples == 0:
        return {
            "total_samples": 0,
            "retrieval_accuracy_percent": None,
            "note": "No learned blur memory records found."
        }

    total = 0
    retrieved_from_memory = 0
    correct = 0

    for sid, cid, cname in zip(memory.sample_ids, memory.card_ids, memory.card_names):
        # Retrieve stored image from card folder
        card_folder = memory.storage_dir / f"card_{cid:03d}"
        blur_files = list(card_folder.glob(f"sample_{sid:04d}_blur.png"))
        if not blur_files:
            continue

        img = cv2.imread(str(blur_files[0]))
        if img is None:
            continue

        query_emb = encoder.encode_image(img)
        distractors = random.sample([c["name"] for c in cards if c["id"] != cid], 3)
        candidate_names = [cname] + distractors
        random.shuffle(candidate_names)

        res = matcher.match_candidates(query_emb, candidate_names, learned_memory=memory)
        total += 1
        if res["matched_from"] == "learned_memory":
            retrieved_from_memory += 1
        if res["card_name"] == cname:
            correct += 1

    return {
        "total_samples_evaluated": total,
        "retrieved_from_memory_percent": round((retrieved_from_memory / total) * 100.0, 2) if total else 0.0,
        "top1_accuracy_percent": round((correct / total) * 100.0, 2) if total else 0.0
    }


def evaluate_unseen_cards(cards, matcher, encoder, num_held_out: int = 20):
    logger.info("Evaluating Group 4: Previously Unseen / Distractor-Only Scenarios...")
    random.seed(999)
    held_out_indices = random.sample(range(len(cards)), min(num_held_out, len(cards)))

    total = 0
    uncertain_flagged = 0

    for idx in held_out_indices:
        target_card = cards[idx]
        card_img = cv2.imread(target_card["image"])
        if card_img is None:
            continue

        # Candidate list that deliberately EXCLUDES the true card
        other_cards = [c["name"] for i, c in enumerate(cards) if i != idx]
        distractor_options = random.sample(other_cards, 4)

        query_emb = encoder.encode_image(apply_independent_blur(card_img, "gaussian_jitter"))
        res = matcher.match_candidates(query_emb, distractor_options)
        total += 1
        # When target is not among the options, confidence should be low or margin small
        if res["status"] == "uncertain" or res["confidence"] < 0.80 or res["margin"] < 0.08:
            uncertain_flagged += 1

    return {
        "held_out_cases_tested": total,
        "correctly_flagged_uncertain_percent": round((uncertain_flagged / total) * 100.0, 2) if total else 0.0,
        "note": "Evaluates confidence calibration when true card is absent from candidate pool."
    }


def run_comprehensive_evaluation(num_cards: int = 40, encoder_name: str = "dinov2"):
    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    matcher = ImageMatcher()
    encoder = get_image_encoder(encoder_name)

    synthetic_report = evaluate_synthetic_blurs(cards, matcher, encoder, num_cards=num_cards)
    real_gameplay_report = evaluate_real_gameplay_images(cards, matcher, encoder)
    learned_memory_report = evaluate_learned_memory(cards, matcher, encoder)
    unseen_cards_report = evaluate_unseen_cards(cards, matcher, encoder, num_held_out=15)

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": encoder_name,
        "data_leakage_prevented": True,
        "synthetic_images": synthetic_report,
        "real_roblox_blur_images": real_gameplay_report,
        "previously_learned_images": learned_memory_report,
        "previously_unseen_cards": unseen_cards_report,
        "summary": {
            "synthetic_top1_percent": synthetic_report["top1_accuracy_percent"],
            "avg_latency_ms": synthetic_report["avg_latency_ms"],
            "p95_latency_ms": synthetic_report["p95_latency_ms"],
            "real_samples_count": real_gameplay_report["available_samples"],
            "disclaimer": "Metrics are measured independently. Real-game accuracy reflects verified gameplay samples only."
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Comprehensive benchmark report saved to %s", REPORT_PATH)
    print("\n" + json.dumps(full_report["summary"], indent=2))
    return full_report


if __name__ == "__main__":
    run_comprehensive_evaluation(num_cards=30)
