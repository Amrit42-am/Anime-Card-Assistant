#!/usr/bin/env bash
# ==============================================================================
# Anime Card Assistant — Strict Offline macOS Launch Script
# ==============================================================================
# Runs the Anime Card Assistant completely offline.
# Does NOT access the internet, run pip install, scrape data, or download weights.
# If any required asset is missing, it halts and directs to ./prepare.sh.
# ==============================================================================

set -e

# Enforce strict offline execution globally in launch environment
export REQUIRE_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "      ANIME CARD ASSISTANT (OFFLINE LAUNCH)       "
echo "=================================================="

# 1. Virtual Environment Check
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found."
    echo "Please run one-time setup first while connected to the internet:"
    echo "    ./prepare.sh"
    exit 1
fi

source .venv/bin/activate

# 2. Dependency Check (Strictly Offline: No pip install)
if ! python -c "import PySide6, torch, timm, cv2, mss, rapidfuzz, safetensors, Vision, Quartz" 2>/dev/null; then
    echo "ERROR: Required Python packages are missing in .venv."
    echo "Please run one-time preparation first while connected to the internet:"
    echo "    ./prepare.sh"
    exit 1
fi

# 3. Preflight Check: Dataset Existence
if [ ! -f "data/cards.json" ] || [ ! -f "data/cards.db" ]; then
    echo "ERROR: Canonical card dataset not found in data/."
    echo "Please run one-time preparation first while connected to the internet:"
    echo "    ./prepare.sh"
    exit 1
fi

# 4. Preflight Check: Embeddings Existence
if [ ! -f "models/embeddings/card_embeddings.npz" ]; then
    echo "ERROR: Precomputed card embeddings not found in models/embeddings/."
    echo "Please run one-time preparation first while connected to the internet:"
    echo "    ./prepare.sh"
    exit 1
fi

# 5. Preflight Check: Local Model Checkpoint Existence
if [ ! -f "models/dinov2/vit_small_patch14_dinov2.safetensors" ]; then
    echo "ERROR: Local DINOv2 model checkpoint not found in models/dinov2/."
    echo "Please run one-time preparation first while connected to the internet:"
    echo "    ./prepare.sh"
    exit 1
fi

# 6. Preflight Check: Local Manifest Verification (Strictly Offline)
if ! python -c "
import os
os.environ['REQUIRE_OFFLINE'] = '1'
from src.ai.model_manager import ModelManager
m = ModelManager.get_instance(require_offline=True)
if not m.is_offline_verified:
    raise RuntimeError(m.verification_message)
" 2>/dev/null; then
    echo "ERROR: Local model manifest verification failed."
    echo "Please run one-time preparation first while connected to the internet:"
    echo "    ./prepare.sh"
    exit 1
fi

# 7. Launch Main Application in Strict Offline Mode
echo "==> Launching Anime Card Assistant in strict offline mode..."
exec python src/main.py "$@"
