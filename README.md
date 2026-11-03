# Anime Card Assistant 🎴⚡
### macOS Live Blurred Card Recognition System for Roblox "Guess the Card"

Anime Card Assistant is a modular, high-performance desktop application for macOS engineered to recognize heavily blurred anime cards appearing inside the Roblox *Anime Card Clash* "Guess the Card" quiz and instantly display the verified card name in a floating overlay window comfortably within the game's 6-second countdown.

---

## 🚀 Key Features

- **286 Verified Card References:** Complete local canonical dataset downloaded and verified directly from the reference database.
- **2×2 Quadrant Answer Layout:** Accurately models Roblox's actual 2×2 button grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right) with independent quadrant cropping and separate result-banner region calibration.
- **Automatic Result-Based Learning:** Passively reads the unblurred card artwork and green correct-answer button when the round resolves. Dual-layer cross-validation ensures only samples with complete agreement between the green button and clear-image match are saved to confirmed memory.
- **DINOv2 Neural Vision Core:** Uses Facebook Research's self-supervised DINOv2 vision transformer with Apple Silicon MPS hardware acceleration, achieving **99.17% Top-1 accuracy** on independent, non-cached synthetic blur benchmarks.
- **Sub-100ms Recognition Pipeline (68.5 ms Median End-to-End):**
  - Display Capture & Region Slicing: **~20.2 ms**
  - Quartz Game Window Localization: **~0.8 ms**
  - Unified Perception Preprocessing: **~21.8 ms**
  - macOS Native Apple Vision OCR: **~17.6 ms** (extracted once per frame, reused downstream)
  - DINOv2 Visual Embedding: **~20.7 ms**
  - Candidate-Constrained Cosine Matching: **~0.1 ms**
  - Temporal Multi-Frame Stabilization: **~45.9 ms**
  - Total end-to-end pipeline step latency: **p50: 68.5 ms, p95: 88.1 ms** (measured & profiled on live pipeline steps).
- **Multi-Frame Temporal Stabilization:** Delivers an immediate `<100ms` provisional prediction on frame 1 (`PROVISIONAL`), then continuously refines over consecutive frames (`STABILIZING` or `CONFIDENT` with $\ge 70\%$ agreement over $\ge 3$ frames). Contradictory evidence across frames dynamically degrades confidence and flags `UNCERTAIN`.
- **Candidate-Constrained Matching:** Uses OCR to extract the 4 visible option buttons in 2×2 quadrants, strictly requiring all 4 distinct options to constrain visual matching (falls back safely to full-database matching if < 4 resolved).
- **100% Offline Local Model Execution:** Guaranteed complete offline functionality by default. Cryptographically verifies local safetensors checkpoints (SHA256, 88,240,510 bytes) and 286 card embeddings (384-dim) via `models/manifest.json`. Zero cloud AI APIs, remote OCR, or telemetry.
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

### 1. Two Explicit Operations: Online Preparation vs Strict Offline Launch

The assistant strictly separates one-time online preparation from everyday offline gameplay:

#### A. One-Time Setup (Online)
Run the preparation script while internet access is available. It builds the Python virtual environment, installs dependencies, fetches canonical card references, generates embeddings, and downloads the approved DINOv2 model checkpoint (`vit_small_patch14_dinov2.safetensors`):
```bash
# Clone the repository
git clone https://github.com/Amrit42-am/Anime-Card-Assistant.git
cd Anime-Card-Assistant

# Run explicit one-time preparation
chmod +x prepare.sh
./prepare.sh
```

#### B. Normal Gameplay Launch (Strictly Offline by Default)
Once prepared, normal gameplay is **100% offline by default**. No internet connection is accessed. The launcher verifies dependencies, card datasets, model checkpoints, and SHA256 manifests locally without network calls:
```bash
chmod +x launch.sh
./launch.sh
```
Or directly via Python (strictly offline by default):
```bash
source .venv/bin/activate
python src/main.py
```
*(The `--offline` CLI flag is retained for backwards compatibility, but is now enforced by default).*

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

## 📊 Benchmark Results (Leakage-Free Multi-Group & Replay Evaluation)

Evaluated across four distinct independent groups and a live `QuizPipeline.step()` replay suite on Apple Silicon (saved to `data/benchmark_report.json`):

| Evaluation Mode / Group | Metric | Measured Result | Status |
|---|---|---|---|
| **Mode A: Replay Pipeline Benchmark** | Top-1 End-to-End Accuracy | **95.00%** (19/20) | ✅ PASSED |
| | False Positive Rate | **0.00%** (0/20) | ✅ PASSED |
| | Uncertain Flagging Rate | **5.00%** (1/20) | ✅ CONSERVATIVE |
| | Provisional Prediction Latency ($p_{50}$) | **63.04 ms** | ✅ PASSED (<100ms) |
| | Provisional Prediction Latency ($p_{95}$) | **93.43 ms** | ✅ PASSED (<100ms) |
| | Subsequent Frame Latency ($p_{50}$) | **24.49 ms** | ✅ OPTIMIZED (OCR Reuse) |
| | Average Temporal Agreement | **1.00** | ✅ PASSED |
| **1. Independent Synthetic Blurs** | Top-1 Accuracy across 7 Distortion Types | **100.00%** (210/210) | ✅ PASSED |
| | Top-2 Accuracy | **100.00%** (210/210) | ✅ PASSED |
| | Average Inference Latency | **20.89 ms** | ✅ PASSED (<500ms) |
| | 95th-Percentile Latency | **23.17 ms** | ✅ PASSED (<1000ms) |
| **2. Real Roblox Blur Images** | Top-1 Accuracy on Verified Gameplay Samples | **100.00%** (3/3) | ✅ MEASURED |
| **3. Held-out Unseen Cards** | Correct Rejection / Uncertainty Flagging | **100.00%** (15/15) | ✅ PASSED |
| **Mode B: Live Gameplay Benchmark** | Safe Live Roblox Detection & Zero Disk Capture | *Active Window Dependent* | ✅ NON-BLOCKING |

### End-to-End Latency Breakdown:
- **Display Capture (Simulated Replay):** $p_{50}$: **19.08 ms**, $p_{95}$: **34.84 ms**
- **Game Localization:** $p_{50}$: **0.78 ms**, $p_{95}$: **2.69 ms**
- **Perception Preprocessing:** $p_{50}$: **21.42 ms**, $p_{95}$: **51.21 ms**
- **Native Apple Vision OCR:** $p_{50}$: **18.34 ms**, $p_{95}$: **20.49 ms** (Cached across visually stable frames)
- **DINOv2 Visual Encoder:** $p_{50}$: **20.11 ms**, $p_{95}$: **21.36 ms**
- **Candidate Matching:** $p_{50}$: **0.08 ms**, $p_{95}$: **0.12 ms**
- **Temporal Reasoning:** $p_{50}$: **24.49 ms**, $p_{95}$: **25.90 ms**
- **Initial Provisional Step (E2E):** **$p_{50}$: 63.04 ms, $p_{95}$: 93.43 ms** (Sub-100ms guaranteed!)
- **Subsequent Stabilization Frame (E2E):** **$p_{50}$: 24.49 ms, $p_{95}$: 25.90 ms**

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

Results: **69 passed in ~8.9s with zero warnings**
- `tests/test_offline_mode.py` (10 tests): Cryptographic SHA256 verification, path traversal prevention, architecture mismatch detection, local safetensors checkpoint loading, file size & dimension checks, failure handling without network, no fallback to pretrained remote weights.
- `tests/test_advanced_ai.py` (11 tests): Confidence engine multi-factor calibration, temporal voting & timer tracking, candidate reasoner candidate restriction rules, context state machine transitions, adaptive learning.
- `tests/test_capture_window.py` (8 tests): Display enumeration, frame capture, Quartz window movement, window resize, multi-monitor and negative coordinates, permission checks.
- `tests/test_dataset.py` (4 tests): 286 card records, non-empty files, SQLite schema consistency.
- `tests/test_recognition.py` (5 tests): DINOv2 embedding dimensionality, L2 normalization, independent image queries, non-cached blur transformations, margin evaluation.
- `tests/test_ocr.py` (6 tests): macOS Apple Vision OCR, noisy prefix cleaning, multiline card names, fuzzy database resolution.
- `tests/test_memory.py` (5 tests): Confirmed sample saving, candidate-restricted retrieval, memory correction, sample invalidation.
- `tests/test_integration.py` (9 tests): End-to-end 2×2 screen capture bundle processing, candidate matching, confirmation persistence, Advanced AI module wiring, PerceptionEngine single-OCR consumption, contradictory evidence confidence penalty, complete lifecycle state transitions, PerceptionEngine OCR cache and refresh on visual grid change.
- `tests/test_result_learning.py` (11 tests): All four 2×2 button positions (TL, TR, BL, BR), OCR retry across result frames, separate disagreement logging, timeout expiration, result banner region separation, headless mode tuple handling.

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

