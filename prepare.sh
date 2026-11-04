#!/usr/bin/env bash
# ==============================================================================
# Anime Card Assistant — One-Time Online Preparation Script
# ==============================================================================
# Run this script with internet access to set up the virtual environment,
# download approved model checkpoints, generate precomputed embeddings,
# and verify offline integrity for subsequent gameplay.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "    ANIME CARD ASSISTANT — ONE-TIME PREPARATION   "
echo "=================================================="

# 1. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "==> Creating Python virtual environment in .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Install dependencies
echo "==> Installing / updating dependencies from requirements.txt..."
pip install -r requirements.txt

# 3. Download and cryptographically verify approved DINOv2 model weights
echo "==> Downloading approved model checkpoints and verifying SHA256 checksums..."
python scripts/download_models.py

# 4. Verify full offline integrity
echo "==> Validating complete offline installation..."
python -c "
from src.ai.model_manager import ModelManager
m = ModelManager.get_instance(require_offline=True)
if not m.is_offline_verified:
    raise RuntimeError(f'Verification failed: {m.verification_message}')
print('✓ Verified:', m.verification_message)
"

echo "=================================================="
echo "✓ PREPARATION COMPLETE!"
echo "You can now disconnect the internet and launch with:"
echo "    ./launch.sh"
echo "=================================================="

