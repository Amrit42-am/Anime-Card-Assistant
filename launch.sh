#!/usr/bin/env bash
# ==============================================================================
# Anime Card Assistant — macOS Launch Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "          ANIME CARD ASSISTANT (macOS)            "
echo "=================================================="

# 1. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "==> Creating Python virtual environment in .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Check and install dependencies
if ! python -c "import PySide6, torch, timm, cv2, mss, rapidfuzz, safetensors, Vision, Quartz" &> /dev/null; then
    echo "==> Installing required packages from requirements.txt..."
    pip install -r requirements.txt
fi

# 3. Preflight: Dataset Check
if [ ! -f "data/cards.json" ] || [ ! -f "data/cards.db" ]; then
    echo "==> Card database missing. Collecting 286 card references..."
    python scripts/collect_cards.py
fi

# 4. Preflight: Precomputed Embeddings Check
if [ ! -f "models/embeddings/card_embeddings.npz" ]; then
    echo "==> Generating synthetic blur variants and building precomputed embeddings..."
    python scripts/generate_blurs.py
    python scripts/build_embeddings.py
fi

# 5. Launch Main Application
echo "==> Launching Anime Card Assistant..."
exec python src/main.py "$@"

