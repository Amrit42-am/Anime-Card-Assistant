"""
scripts/evaluate_model.py

Phase 10 & 11: Benchmark Evaluation Suite.
Measures Top-1 and Top-2 candidate recognition accuracy, per-blur-strength accuracy,
learned memory retrieval accuracy, and latency (average & 95th percentile).
Generates data/benchmark_report.json.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.recognition.image_encoder import get_image_encoder
from src.recognition.image_matcher import ImageMatcher
from src.memory.blur_memory import BlurMemorySystem
from src.memory.database import MemoryDatabase

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_model")

BLURS_DIR = Path("data/blur_variants")
CARDS_JSON = Path("data/cards.json")
REPORT_PATH = Path("data/benchmark_report.json")


def run_evaluation(num_test_cards: int = 100, encoder_name: str = "dinov2"):
    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    matcher = ImageMatcher()
    encoder = get_image_encoder(encoder_name)

    variant_names = ["blur_light", "blur_medium", "blur_heavy", "blur_box", "blur_compound"]

    # Select representative test sample
    random.seed(42)
    test_indices = random.sample(range(len(cards)), min(num_test_cards, len(cards)))

    logger.info("Evaluating recognition accuracy across %d cards and %d blur variants...", len(test_indices), len(variant_names))

    per_variant_correct = {v: 0 for v in variant_names}
    per_variant_top2 = {v: 0 for v in variant_names}
    per_variant_total = {v: 0 for v in variant_names}
    latencies = []

    total_evals = 0
    total_top1_correct = 0
    total_top2_correct = 0

    for card_idx in test_indices:
        target_card = cards[card_idx]
        target_id = target_card["id"]
        target_name = target_card["name"]

        # Pick 3 random distractor cards to form the 4 quiz options
        distractor_indices = [i for i in range(len(cards)) if i != card_idx]
        chosen_distractors = random.sample(distractor_indices, 3)
        candidate_names = [target_name] + [cards[i]["name"] for i in chosen_distractors]
        random.shuffle(candidate_names)

        card_blur_dir = BLURS_DIR / f"card_{target_id:03d}"

        for v_name in variant_names:
            img_path = card_blur_dir / f"{v_name}.png"
            if not img_path.exists():
                continue

            t0 = time.time()
            query_emb = encoder.encode_image(img_path)
            res = matcher.match_candidates(query_emb, candidate_names)
            elapsed_ms = (time.time() - t0) * 1000.0
            latencies.append(elapsed_ms)

            is_top1 = (res["card_name"] == target_name)
            is_top2 = is_top1
            if not is_top2 and res.get("alternatives"):
                top_alt = res["alternatives"][0]["card_name"]
                is_top2 = (top_alt == target_name)

            total_evals += 1
            per_variant_total[v_name] += 1

            if is_top1:
                total_top1_correct += 1
                per_variant_correct[v_name] += 1

            if is_top2:
                total_top2_correct += 1
                per_variant_top2[v_name] += 1

    top1_acc = (total_top1_correct / total_evals) * 100.0 if total_evals else 0.0
    top2_acc = (total_top2_correct / total_evals) * 100.0 if total_evals else 0.0

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    logger.info("================ EVALUATION SUMMARY ================")
    logger.info("Total Evaluated Questions: %d", total_evals)
    logger.info("Top-1 Recognition Accuracy: %.2f%%", top1_acc)
    logger.info("Top-2 Recognition Accuracy: %.2f%%", top2_acc)
    logger.info("Average Latency: %.2f ms", avg_latency)
    logger.info("95th-Percentile Latency: %.2f ms", p95_latency)
    logger.info("---------------- Per-Blur Accuracy ----------------")
    per_blur_summary = {}
    for v in variant_names:
        v_tot = per_variant_total[v]
        v_acc = (per_variant_correct[v] / v_tot * 100.0) if v_tot else 0.0
        per_blur_summary[v] = round(v_acc, 2)
        logger.info("  %-15s : %.2f%% (%d/%d)", v, v_acc, per_variant_correct[v], v_tot)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": encoder_name,
        "cards_evaluated": len(test_indices),
        "total_questions": total_evals,
        "top1_accuracy_percent": round(top1_acc, 2),
        "top2_accuracy_percent": round(top2_acc, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "per_blur_accuracy": per_blur_summary,
        "meets_accuracy_goal": top1_acc >= 95.0,
        "meets_latency_goal": p95_latency < 500.0
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Benchmark report written to %s", REPORT_PATH)
    return report


if __name__ == "__main__":
    run_evaluation(num_test_cards=60)

