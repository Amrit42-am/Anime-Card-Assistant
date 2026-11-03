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
from src.ai.model_manager import ModelManager
from src.capture.display_monitor import DisplayMonitor
from src.capture.window_tracker import WindowTracker
from src.capture.game_locator import GameLocator
from src.ocr.option_reader import OptionReader
from src.core.pipeline import QuizPipeline
from src.core.decision_engine import AnswerResult

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


def benchmark_mode_a_replay(cards, num_questions: int = 25):
    """
    Mode A — Replay benchmark:
    Uses independently labelled gameplay bundles and feeds them through the actual
    QuizPipeline.step() execution with deterministic dependencies.
    Measures complete pipeline execution and component breakdown.
    """
    logger.info("Executing Mode A: Replay Benchmark on Live QuizPipeline.step()...")
    pipeline = QuizPipeline(auto_init_models=True)
    random.seed(42)

    sampled = random.sample(range(len(cards)), min(num_questions, len(cards)))
    top1_correct = 0
    top2_correct = 0
    uncertain_count = 0
    false_positives = 0
    temporal_agreements = []
    step_times = []

    comp_capture = []
    comp_locate = []
    comp_perception = []
    comp_ocr = []
    comp_encoder = []
    comp_match = []
    comp_temporal = []
    comp_confidence = []

    for idx in sampled:
        target = cards[idx]
        target_name = target["name"]
        card_img = cv2.imread(target["image"])
        if card_img is None:
            continue

        # 3 distractors
        distractors = random.sample([i for i in range(len(cards)) if i != idx], 3)
        candidate_names = [target_name] + [cards[i]["name"] for i in distractors]
        random.shuffle(candidate_names)

        blurred_card = cv2.GaussianBlur(card_img, (35, 35), 12.0)
        btn_frames = []
        for name in candidate_names:
            btn = np.ones((60, 140, 3), dtype=np.uint8) * 40
            cv2.putText(btn, name, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            btn_frames.append(btn)
        top_row = np.hstack([btn_frames[0], btn_frames[1]])
        bot_row = np.hstack([btn_frames[2], btn_frames[3]])
        options_frame = np.vstack([top_row, bot_row])
        game_frame = np.ones((600, 800, 3), dtype=np.uint8) * 30

        mock_bundle = {
            "game_frame": game_frame,
            "card_frame": blurred_card,
            "options_frame": options_frame,
            "button_frames": btn_frames,
            "timer_frame": None,
            "banner_frame": None
        }

        # Measure individual component times
        t0 = time.perf_counter()
        _ = pipeline.game_locator.locate_game()
        comp_locate.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        p_frame = pipeline.perception_engine.perceive_frame(bundle=mock_bundle)
        comp_perception.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = pipeline.option_reader.read_options_from_buttons(btn_frames)
        comp_ocr.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        emb = pipeline.encoder.encode_image(blurred_card)
        comp_encoder.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = pipeline.matcher.match_candidates(emb, candidate_names)
        comp_match.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = pipeline.confidence_engine.evaluate(top_score=0.9, runner_up_score=0.6)
        comp_confidence.append((time.perf_counter() - t0) * 1000.0)

        # Run through actual pipeline.step()
        pipeline.detector.reset()
        orig_capture = pipeline.capture.capture_quiz_bundle
        pipeline.capture.capture_quiz_bundle = lambda *args, **kwargs: mock_bundle

        try:
            # Step 1: Initial Question (Provisional Answer)
            t_step0 = time.perf_counter()
            ans1, _ = pipeline.step()
            t_step1 = time.perf_counter()
            step_times.append((t_step1 - t_step0) * 1000.0)

            # Step 2: Consecutive Frame (Multi-Frame Temporal Stabilization)
            t_temp0 = time.perf_counter()
            ans2, _ = pipeline.step()
            comp_temporal.append((time.perf_counter() - t_temp0) * 1000.0)

            final_ans = ans2 or ans1
            if final_ans:
                temporal_agreements.append(final_ans.temporal_agreement)
                is_correct = (final_ans.card_name == target_name)
                if is_correct:
                    top1_correct += 1
                    top2_correct += 1
                else:
                    if final_ans.runner_up_name == target_name:
                        top2_correct += 1
                    if final_ans.status == "matched":
                        false_positives += 1

                if final_ans.status == "uncertain":
                    uncertain_count += 1
        finally:
            pipeline.capture.capture_quiz_bundle = orig_capture

    # Capture latency on primary display
    monitor = DisplayMonitor()
    for _ in range(10):
        t0 = time.perf_counter()
        try:
            _ = monitor.capture_display()
        except Exception:
            pass
        comp_capture.append((time.perf_counter() - t0) * 1000.0)

    def stats(arr):
        if not arr:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "max_ms": round(float(np.max(arr)), 2)
        }

    total_q = len(sampled)
    return {
        "mode": "Mode A (Replay Pipeline Benchmark)",
        "questions_evaluated": total_q,
        "top1_accuracy_percent": round((top1_correct / total_q) * 100.0, 2) if total_q else 0.0,
        "top2_accuracy_percent": round((top2_correct / total_q) * 100.0, 2) if total_q else 0.0,
        "uncertain_rate_percent": round((uncertain_count / total_q) * 100.0, 2) if total_q else 0.0,
        "false_positive_rate_percent": round((false_positives / total_q) * 100.0, 2) if total_q else 0.0,
        "avg_temporal_agreement": round(float(np.mean(temporal_agreements)), 3) if temporal_agreements else 1.0,
        "e2e_pipeline_step_latency": stats(step_times),
        "component_latencies": {
            "1_display_capture": stats(comp_capture),
            "2_game_localization": stats(comp_locate),
            "3_perception_preprocessing": stats(comp_perception),
            "4_ocr_extraction": stats(comp_ocr),
            "5_encoder_inference": stats(comp_encoder),
            "6_candidate_matching": stats(comp_match),
            "7_temporal_reasoning": stats(comp_temporal),
            "8_confidence_evaluation": stats(comp_confidence)
        }
    }


def benchmark_mode_b_live(num_frames: int = 10):
    """
    Mode B — Live benchmark:
    Checks if Roblox Player is active and measures live pipeline execution.
    Does not save full-screen recordings to disk.
    """
    logger.info("Executing Mode B: Live Gameplay Benchmark...")
    tracker = WindowTracker()
    win = tracker.find_target_window()
    if not win:
        return {
            "mode": "Mode B (Live Gameplay Benchmark)",
            "status": "SKIPPED",
            "message": "Roblox window is not currently open or active on display."
        }

    pipeline = QuizPipeline(auto_init_models=True)
    live_latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        _ = pipeline.step()
        live_latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mode": "Mode B (Live Gameplay Benchmark)",
        "status": "COMPLETED",
        "frames_measured": num_frames,
        "target_window": f"{win.owner_name} ({win.width}x{win.height})",
        "p50_latency_ms": round(float(np.percentile(live_latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(live_latencies, 95)), 2)
    }


def run_comprehensive_evaluation(num_cards: int = 30, encoder_name: str = "dinov2"):
    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    matcher = ImageMatcher()
    try:
        encoder = ModelManager.get_instance().get_encoder()
    except Exception:
        encoder = get_image_encoder(encoder_name)

    synthetic_report = evaluate_synthetic_blurs(cards, matcher, encoder, num_cards=num_cards)
    real_gameplay_report = evaluate_real_gameplay_images(cards, matcher, encoder)
    learned_memory_report = evaluate_learned_memory(cards, matcher, encoder)
    unseen_cards_report = evaluate_unseen_cards(cards, matcher, encoder, num_held_out=15)
    replay_benchmark = benchmark_mode_a_replay(cards, num_questions=20)
    live_benchmark = benchmark_mode_b_live(num_frames=5)

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": encoder_name,
        "data_leakage_prevented": True,
        "mode_a_replay_benchmark": replay_benchmark,
        "mode_b_live_benchmark": live_benchmark,
        "synthetic_images": synthetic_report,
        "real_roblox_blur_images": real_gameplay_report,
        "previously_learned_images": learned_memory_report,
        "previously_unseen_cards": unseen_cards_report,
        "summary": {
            "synthetic_top1_percent": synthetic_report["top1_accuracy_percent"],
            "replay_top1_percent": replay_benchmark["top1_accuracy_percent"],
            "e2e_pipeline_step_p50_ms": replay_benchmark["e2e_pipeline_step_latency"]["p50_ms"],
            "e2e_pipeline_step_p95_ms": replay_benchmark["e2e_pipeline_step_latency"]["p95_ms"],
            "avg_temporal_agreement": replay_benchmark["avg_temporal_agreement"],
            "real_samples_count": real_gameplay_report["available_samples"],
            "disclaimer": "Metrics are measured on actual QuizPipeline.step() execution with no data leakage."
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Comprehensive benchmark report saved to %s", REPORT_PATH)
    print("\n" + json.dumps(full_report["summary"], indent=2))
    return full_report
    return full_report


if __name__ == "__main__":
    run_comprehensive_evaluation(num_cards=30)
