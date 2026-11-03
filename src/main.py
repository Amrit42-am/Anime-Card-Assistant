"""
src/main.py

Phase 9: Main Application Entry Point.
Initializes configuration, verifies dependencies and screen permissions,
and starts the Anime Card Assistant desktop application with PySide6.
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Anime Card Assistant — Roblox Card Clash Recognition System")
    parser.add_argument("--calibrate", action="store_true", help="Launch calibration dialog directly.")
    parser.add_argument("--headless", action="store_true", help="Run monitoring loop in terminal without GUI.")
    parser.add_argument("--offline", action="store_true", default=True, help="Enforce strict offline execution (default: strictly offline).")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dinov2", "mobilenet"], help="Visual encoder model.")
    parser.add_argument("--fps", type=float, default=8.0, help="Target screen monitoring FPS.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Normal gameplay is strictly offline by default
    os.environ["REQUIRE_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from src.ai.model_manager import ModelManager
    manager = ModelManager.get_instance(require_offline=True)
    if not manager.is_offline_verified:
        logger.error("Offline execution failed verification: %s. Please run: ./prepare.sh", manager.verification_message)
        sys.exit(1)
    logger.info("✓ Strict offline execution verified: local checkpoints and embeddings validated.")

    # Preflight check for card dataset and embeddings
    cards_json = Path("data/cards.json")
    embeddings_npz = Path("models/embeddings/card_embeddings.npz")

    if not cards_json.exists():
        logger.error("Dataset not found at %s. Please run: ./prepare.sh", cards_json)
        sys.exit(1)

    if not embeddings_npz.exists():
        logger.error("Embeddings not found at %s. Please run: ./prepare.sh", embeddings_npz)
        sys.exit(1)

    # Lazy import heavy Qt & Pipeline modules after argument parsing
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from src.capture.screen_capture import ScreenCapture
    from src.capture.calibration import CalibrationDialog
    from src.core.pipeline import QuizPipeline
    from src.ui.overlay import FloatingOverlay

    # Check Screen Recording permissions on macOS
    has_perm = ScreenCapture.check_permissions()
    if not has_perm:
        logger.warning("Screen Recording permission might not be granted. Requesting access...")
        ScreenCapture.request_permissions()

    # Headless mode
    if args.headless:
        logger.info("Starting headless monitoring pipeline (Ctrl+C to quit)...")
        pipeline = QuizPipeline(encoder_name=args.model)
        try:
            while True:
                ans, rep = pipeline.step()
                if ans is not None:
                    if ans.status == "matched":
                        print(f"\n>>> RECOGNIZED CARD: {ans.card_name} ({ans.confidence*100:.1f}%) in {ans.processing_time_ms:.1f} ms [Margin: +{ans.margin:.2f}]")
                    else:
                        print(f"\n>>> UNCERTAIN PREDICTION: {ans.card_name} ({ans.confidence*100:.1f}%) in {ans.processing_time_ms:.1f} ms")

                if rep is not None:
                    if rep.verified:
                        print(f"\n[AUTO-LEARNED] Confirmed: {rep.card_name} (Sample ID: {rep.sample_id})")
                    else:
                        print(f"\n[AUTO-LEARN NOTICE] {rep.message} (Status: {rep.status})")

                time.sleep(1.0 / args.fps)
        except KeyboardInterrupt:
            logger.info("Headless pipeline stopped.")
            sys.exit(0)

    # GUI Application
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Anime Card Assistant")
    app.setOrganizationName("AnimeCardClash")

    # Calibration-only mode
    if args.calibrate:
        logger.info("Launching screen calibration...")
        capture = ScreenCapture()
        dlg = CalibrationDialog(capture)
        dlg.exec()
        sys.exit(0)

    # Initialize full pipeline
    pipeline = QuizPipeline(encoder_name=args.model)

    # Launch floating assistant overlay
    overlay = FloatingOverlay(pipeline=pipeline, is_offline=pipeline.is_offline_verified)
    overlay.show()
    overlay.raise_()
    overlay.activateWindow()

    # Bring to foreground on macOS
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyRegular
        ns_app = NSApplication.sharedApplication()
        ns_app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        ns_app.activateIgnoringOtherApps_(True)
    except Exception as e:
        logger.debug("AppKit activation notice: %s", e)

    # Allow clean Ctrl+C in terminal to exit Qt app
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    logger.info("Anime Card Assistant is active and monitoring!")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()