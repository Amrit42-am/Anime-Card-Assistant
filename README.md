# Anime Card Assistant 🎴⚡
### macOS Live Blurred Card Recognition System for Roblox "Guess the Card"

Anime Card Assistant is a modular, high-performance desktop application for macOS engineered to recognize heavily blurred anime cards appearing inside the Roblox *Anime Card Clash* "Guess the Card" quiz and instantly display the verified card name in a floating overlay window comfortably within the game's 6-second countdown.

---

## 🚀 Key Features

- **286 Verified Card References:** Complete local canonical dataset downloaded and verified directly from the reference database.
- **2×2 Quadrant Answer Layout:** Accurately models Roblox's actual 2×2 button grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right) with independent quadrant cropping and separate result-banner region calibration.
- **Automatic Result-Based Learning:** Passively reads the unblurred card artwork and green correct-answer button when the round resolves. Dual-layer cross-validation ensures only samples with complete agreement between the green button and clear-image match are saved to confirmed memory.
- **DINOv2 Neural Vision Core:** Uses Facebook Research's self-supervised DINOv2 vision transformer with Apple Silicon MPS hardware acceleration, achieving **98.33% Top-1 accuracy** on independent, non-cached synthetic blur benchmarks.
- **Sub-100ms Recognition Pipeline:**
  - Screen Capture & 2×2 Region Slicing: **~8 ms**
  - macOS Native Apple Vision OCR: **~25 ms**
  - Visual Embedding & Candidate Matching: **~16 ms**
  - Total warm response latency: **~49 ms** (benchmark verified).
- **Candidate-Constrained Matching:** Uses OCR to extract the 4 visible option buttons in 2×2 quadrants, restricting visual comparison strictly to those candidates.
- **Automatic OCR Retry & Fallback:** Automatically retries button preprocessing with contrast enhancement if fewer than 4 options resolve, using controlled fallback with strict uncertainty thresholds.
- **Disagreement Isolation:** Inconclusive or disagreeing result samples are preserved separately in `disagreement_logs` for debugging and never contaminate active recognition memory.
- **Modern PySide6 Floating UI & Headless Mode:** Frameless, always-on-top, draggable overlay positioned on the right side of the screen with opacity controls, live latency tracking, auto-learned confirmation pills, and support for terminal headless mode.
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

### One-Command Launch

Ensure execute permissions, then launch:

```bash
chmod +x launch.sh
./launch.sh
```

`launch.sh` automatically:
1. Detects or creates the Python virtual environment (`.venv`).
2. Installs required dependencies if missing.
3. Verifies local card dataset and precomputed embeddings.
4. Launches the Anime Card Assistant floating overlay on your desktop.

---

## 🛠 Manual Installation

If you prefer step-by-step setup:

```bash
# 1. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Collect cards and download all 286 artwork references
python scripts/collect_cards.py

# 4. Generate contact sheet for visual inspection
python scripts/create_contact_sheet.py

# 5. Generate synthetic blur variants
python scripts/generate_blurs.py

# 6. Precompute DINOv2 embeddings
python scripts/build_embeddings.py

# 7. Run test suite
pytest tests/

# 8. Start application
python src/main.py
```

---

## 🎮 How to Calibrate and Play

### Step 1: Calibrate Screen Regions
1. Open Roblox and place the window on the left side of your desktop.
2. In Anime Card Assistant, click **⚙ Calibrate** (or run `python src/main.py --calibrate`).
3. Adjust the **Game Window** coordinates to enclose the Roblox game area.
4. Adjust the **Card Artwork Region** (green outline) so it aligns with the blurred card box.
5. Adjust the **Answer Options Region** (orange outline) to surround the 4 answer buttons.
6. Click **Save Calibration**. Calibration settings are saved to `data/calibration.json`.

### Step 2: Live Game Recognition
1. As the "Guess the Card" round starts, the assistant automatically detects the quiz.
2. The card name appears in large green text with its confidence score and evidence source.
3. The 4 options are highlighted in the candidate box with the winner marked `➤`.

### Step 3: Self-Improving Blur Memory
- **Confirming Answers:** When the round reveals the answer, click **✓ Confirm**. The assistant saves the actual in-game blurred screenshot and its feature embedding into `data/learned_blurs/`. Future encounters with this card will recognize it via **Learned Memory** with near-instantaneous confidence.
- **Manual Correction:** If an ambiguous card was mispredicted, click **✎ Correct**, type the card's name, and select it. The memory updates immediately and replaces the incorrect sample.

---

## 📊 Benchmark Results (Leakage-Free Multi-Group Evaluation)

Evaluated across four distinct, independent groups on Apple Silicon (M-series MacBook Pro):

| Evaluation Group | Metric | Measured Result | Status |
|---|---|---|---|
| **1. Independent Synthetic Blurs** | Top-1 Accuracy (Jittered Kernel & Noise) | **98.33%** (118/120) | ✅ PASSED |
| | Top-2 Accuracy | **100.00%** (120/120) | ✅ PASSED |
| | Average Inference Latency | **49.36 ms** | ✅ PASSED (<500ms) |
| | 95th-Percentile Latency | **57.65 ms** | ✅ PASSED (<1000ms) |
| **2. Real Roblox Blur Images** | Top-1 Accuracy on Verified Gameplay Samples | **100.00%** (3/3) | ✅ MEASURED |
| **3. Held-out Unseen Cards** | Correct Rejection / Uncertainty Flagging | **100.00%** (15/15) | ✅ PASSED |
| **Per-Distortion Breakdown** | Gaussian Jitter | **100.00%** (30/30) | ✅ PASSED |
| | Box Blur Jitter | **100.00%** (30/30) | ✅ PASSED |
| | Motion Blur | **100.00%** (30/30) | ✅ PASSED |
| | Compound Noise & Lighting Shifts | **93.33%** (28/30) | ✅ PASSED |

> [!NOTE]
> All test queries are generated independently or drawn from held-out gameplay captures; reference database embeddings are never re-used as test inputs. Real-gameplay performance is continuously tracked as new confirmed samples are collected.

Run the benchmark suite yourself at any time:
```bash
python scripts/evaluate_model.py
```

---

## 🧪 Test Suite

Run the full pytest suite:

```bash
pytest tests/
```

Results: **35 passed in ~20s**
- `tests/test_dataset.py`: 286 card records, non-empty files, SQLite schema consistency.
- `tests/test_recognition.py`: DINOv2 embedding dimensionality, L2 normalization, independent image queries, non-cached blur transformations, margin evaluation.
- `tests/test_ocr.py`: macOS Apple Vision OCR, noisy prefix cleaning, multiline card names, fuzzy database resolution.
- `tests/test_memory.py`: Confirmed sample saving, candidate-restricted retrieval, memory correction, sample invalidation.
- `tests/test_integration.py`: End-to-end 2×2 screen capture bundle processing, candidate matching, confirmation persistence, full lifecycle simulation.
- `tests/test_result_learning.py`: All four 2×2 button positions (TL, TR, BL, BR), OCR retry across result frames, separate disagreement logging, timeout expiration, result banner region separation, headless mode tuple handling.

---

## ⚙ Advanced CLI Options

```bash
# Calibrate screen capture without launching overlay
python src/main.py --calibrate

# Run in headless mode in the terminal (no GUI overlay)
python src/main.py --headless --fps 8.0

# Use MobileNetV3 lightweight CNN encoder fallback instead of DINOv2
python src/main.py --model mobilenet
```

