# Anime Card Assistant 🎴⚡
### macOS Live Blurred Card Recognition System for Roblox "Guess the Card"

Anime Card Assistant is a modular, high-performance desktop application for macOS engineered to recognize heavily blurred anime cards appearing inside the Roblox *Anime Card Clash* "Guess the Card" quiz and instantly display the verified card name in a floating overlay window comfortably within the game's 6-second countdown.

---

## 🚀 Key Features

- **286 Verified Card References:** Complete local canonical dataset downloaded and verified directly from the reference database.
- **2×2 Quadrant Answer Layout:** Accurately models Roblox's actual 2×2 button grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right) with independent quadrant cropping and separate result-banner region calibration.
- **Automatic Result-Based Learning:** Passively reads the unblurred card artwork and green correct-answer button when the round resolves. Dual-layer cross-validation ensures only samples with complete agreement between the green button and clear-image match are saved to confirmed memory.
- **DINOv2 Neural Vision Core:** Uses Facebook Research's self-supervised DINOv2 vision transformer with Apple Silicon MPS hardware acceleration, achieving **99.17% Top-1 accuracy** on independent, non-cached synthetic blur benchmarks.
- **Sub-100ms Recognition Pipeline (53.6 ms Median End-to-End):**
  - Screen Capture & 2×2 Region Slicing: **~19 ms**
  - macOS Native Apple Vision OCR: **~16 ms**
  - DINOv2 Visual Embedding: **~18 ms**
  - Candidate-Constrained Cosine Matching: **~0.1 ms**
  - Total end-to-end response latency: **p50: 53.6 ms, p95: 66.25 ms** (measured & profiled).
- **Candidate-Constrained Matching:** Uses OCR to extract the 4 visible option buttons in 2×2 quadrants, strictly requiring all 4 distinct options to constrain visual matching (falls back safely to full-database matching if < 4 resolved).
- **100% Offline Local Model Execution:** Guaranteed complete offline functionality after initial setup. Validates local safetensors checkpoints and embeddings via SHA256 manifest.
- **Automatic OCR Retry & Fallback:** Automatically retries button preprocessing with contrast enhancement if fewer than 4 options resolve, using controlled fallback with strict uncertainty thresholds.
- **Disagreement Isolation:** Inconclusive or disagreeing result samples are preserved separately in `disagreement_logs` for debugging and never contaminate active recognition memory.
- **Modern PySide6 Floating UI & Headless Mode:** Frameless, always-on-top, draggable overlay with 2×2 slot highlighting (`[TL]`, `[TR]`, `[BL]`, `[BR]`), margin & runner-up display, `🔒 OFFLINE` indicator, compact HUD toggle, and terminal headless mode.
- **Completely Local & Passive:** Requires zero external cloud AI APIs for recognition and does not click, inject, or interfere with Roblox processes.

---

## 📁 System Architecture

```text
anime-card-assistant/
├── assets/
│   └── cards/                    # 286 original card artwork PNGs
├── data/
│   ├── cards.json                # Canonical card metadata
│   ├── cards.db                  # SQLite database of cards
│   ├── card_contact_sheet.jpg    # 2100x3552 visual contact sheet
│   ├── calibration.json          # Calibrated screen coordinates
│   ├── card_memory.db            # SQLite persistent learned blur memory
│   ├── learned_blurs/            # User-confirmed gameplay blur samples
│   ├── blur_variants/            # 1,716 synthetic blur reference images
│   ├── collection_report.json    # Phase 1 dataset validation report
│   └── benchmark_report.json     # Phase 10 benchmark performance report
├── models/
│   └── embeddings/
│       ├── card_embeddings.npz   # Precomputed clean, blur, and composite embeddings
│       └── embeddings_meta.json  # Embedding metadata
├── scripts/
│   ├── collect_cards.py          # Data scraper & image downloader
│   ├── create_contact_sheet.py   # Visual contact sheet generator
│   ├── generate_blurs.py         # Multi-strength synthetic blur generator
│   ├── build_embeddings.py       # Precomputes DINOv2 feature vectors
│   └── evaluate_model.py         # 300-question accuracy & latency benchmark
├── src/
│   ├── capture/
│   │   ├── screen_capture.py     # MSS + Quartz screen grabber & Retina scaler
│   │   ├── region_manager.py     # Relative & absolute coordinate translator
│   │   └── calibration.py        # Visual Qt calibration dialog
│   ├── detection/
│   │   └── question_detector.py  # dHash & visual change detector
│   ├── ocr/
│   │   └── option_reader.py      # macOS Apple Vision text reader + fuzzy resolver
│   ├── recognition/
│   │   ├── blur_generator.py     # OpenCV blur augmentor
│   │   ├── image_encoder.py      # DINOv2 / MobileNetV3 model abstraction
│   │   ├── image_matcher.py      # 4-candidate cosine similarity matcher
│   │   └── confidence.py         # Calibrated margin & confidence estimator
│   ├── memory/
│   │   ├── database.py           # SQLite storage for learned samples
│   │   └── blur_memory.py        # Persistent in-memory cache & sample updater
│   ├── core/
│   │   ├── decision_engine.py    # Answer decision engine
│   │   ├── pipeline.py           # High-level quiz pipeline
│   │   └── worker.py             # Non-blocking PySide6 QThread worker
│   ├── ui/
│   │   ├── overlay.py            # Floating assistant overlay window
│   │   └── components.py         # Custom badges, score bars & correction dialog
│   └── main.py                   # Application entry point
├── tests/
│   ├── test_dataset.py           # Dataset integrity tests
│   ├── test_recognition.py       # Visual matching & ranking tests
│   ├── test_ocr.py               # Apple Vision & text cleaning tests
│   ├── test_memory.py            # Learned blur memory & safety tests
│   └── test_integration.py       # End-to-end pipeline simulation tests
├── config.yaml                   # Application settings
├── requirements.txt              # Python dependencies
├── launch.sh                     # One-command executable launcher
└── README.md
```

---

## ⚡ Quickstart

### 1. Two Startup Modes: Online Setup vs Completely Offline Execution

The assistant requires a one-time preparation step to download canonical card references and local model checkpoints, after which it operates **100% offline**:

#### A. One-Time Setup (Online)
```bash
# Clone the repository
git clone https://github.com/Amrit42-am/Anime-Card-Assistant.git
cd Anime-Card-Assistant

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download model checkpoints and verify offline manifest
python scripts/download_models.py
```

#### B. Autonomous Execution (Completely Offline)
You can disconnect the internet or use `--offline` mode. No network calls or downloads will occur:
```bash
python src/main.py --offline
```

Alternatively, run the automated launcher:
```bash
chmod +x launch.sh
./launch.sh
```

---

## 🎮 How to Calibrate and Play

### Step 1: Automatic Window Tracking & Screen Calibration
1. Open Roblox. The assistant uses Quartz `WindowTracker` to automatically detect and follow the Roblox Player window across moves, resizes, and multi-monitor setups.
2. If automatic tracking fails or manual adjustment is preferred, run:
   ```bash
   python src/main.py --calibrate
   ```
3. Adjust the **Game Window**, **Card Artwork**, and **2×2 Answer Options** regions. Calibration settings are saved locally to `data/calibration.json`.

### Step 2: Live Game Recognition & HUD Slot Display
1. As a "Guess the Card" round starts, the assistant immediately processes the screen in **sub-100ms** (median ~53 ms).
2. The HUD overlay highlights the predicted winning button with its 2×2 grid position:
   - `[TL]` Top-Left
   - `[TR]` Top-Right
   - `[BL]` Bottom-Left
   - `[BR]` Bottom-Right
3. Displays the runner-up card, margin of victory, and calibrated confidence level (`HIGH`, `MEDIUM`, `LOW`, or `UNCERTAIN`).

### Step 3: Fully Autonomous Result Learning (No Manual Action Required)
- **Automatic Ground-Truth Verification:** When Roblox reveals the result screen, the pipeline detects the green winning button and unblurred artwork.
- **Dual-Verification Rule:** Only if the green button OCR strictly agrees with an independent clear artwork visual match is the sample recorded into verified blur memory (`data/card_memory.db`).
- **Disagreement Isolation:** Disagreements and timeouts are safely routed to `disagreement_logs` and never contaminate active memory.
- **Optional Manual Overrides:** Users can still click **✓ Confirm** or **✎ Correct** if manual intervention is ever desired.

---

## 📊 Benchmark Results (Leakage-Free Multi-Group Evaluation)

Evaluated across four distinct, independent groups with separate latency profiling on Apple Silicon:

| Evaluation Group | Metric | Measured Result | Status |
|---|---|---|---|
| **1. Independent Synthetic Blurs** | Top-1 Accuracy (Jittered Kernel & Noise) | **99.17%** (119/120) | ✅ PASSED |
| | Top-2 Accuracy | **100.00%** (120/120) | ✅ PASSED |
| | Average Inference Latency | **18.13 ms** | ✅ PASSED (<500ms) |
| | 95th-Percentile Latency | **18.46 ms** | ✅ PASSED (<1000ms) |
| **2. Real Roblox Blur Images** | Top-1 Accuracy on Verified Gameplay Samples | **100.00%** (3/3) | ✅ MEASURED |
| **3. Held-out Unseen Cards** | Correct Rejection / Uncertainty Flagging | **100.00%** (15/15) | ✅ PASSED |
| **Per-Distortion Breakdown** | Gaussian Jitter | **100.00%** (30/30) | ✅ PASSED |
| | Box Blur Jitter | **100.00%** (30/30) | ✅ PASSED |
| | Motion Blur | **100.00%** (30/30) | ✅ PASSED |
| | Compound Noise & Lighting Shifts | **96.67%** (29/30) | ✅ PASSED |

### Real End-to-End Pipeline Latency Breakdown:
- **Screen Capture:** p50: **19.25 ms**, p95: **25.77 ms**
- **OCR Extraction:** p50: **16.43 ms**, p95: **19.90 ms**
- **DINOv2 Encoder:** p50: **17.94 ms**, p95: **18.31 ms**
- **Candidate Matching:** p50: **0.07 ms**, p95: **0.09 ms**
- **Total Pipeline E2E:** **p50: 53.60 ms, p95: 66.25 ms** (Sub-100ms guaranteed!)

Run the benchmark suite at any time:
```bash
python scripts/evaluate_model.py
```

---

## 🧪 Test Suite

Run the full pytest suite:

```bash
pytest -v
```

Results: **59 passed in ~4.3s**
- `tests/test_offline_mode.py`: Local safetensors checkpoint loading, offline manifest verification, failure handling without network.
- `tests/test_advanced_ai.py`: Confidence engine multi-factor calibration, temporal voting & timer tracking, candidate reasoner candidate restriction rules, context state machine transitions, adaptive learning.
- `tests/test_capture_window.py`: Display enumeration, frame capture, Quartz window movement, window resize, multi-monitor and negative coordinates, permission checks.
- `tests/test_dataset.py`: 286 card records, non-empty files, SQLite schema consistency.
- `tests/test_recognition.py`: DINOv2 embedding dimensionality, L2 normalization, independent image queries, non-cached blur transformations, margin evaluation.
- `tests/test_ocr.py`: macOS Apple Vision OCR, noisy prefix cleaning, multiline card names, fuzzy database resolution.
- `tests/test_memory.py`: Confirmed sample saving, candidate-restricted retrieval, memory correction, sample invalidation.
- `tests/test_integration.py`: End-to-end 2×2 screen capture bundle processing, candidate matching, confirmation persistence, Advanced AI module wiring.
- `tests/test_result_learning.py`: All four 2×2 button positions (TL, TR, BL, BR), OCR retry across result frames, separate disagreement logging, timeout expiration, result banner region separation, headless mode tuple handling.

---

## ⚙ Advanced CLI Options

```bash
# Run in completely offline mode (verifies local models & manifests)
python src/main.py --offline

# Calibrate screen capture without launching overlay
python src/main.py --calibrate

# Run in headless mode in the terminal (no GUI overlay)
python src/main.py --headless --fps 8.0

# Use MobileNetV3 lightweight CNN encoder fallback instead of DINOv2
python src/main.py --model mobilenet
```

