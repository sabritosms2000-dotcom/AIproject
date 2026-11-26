# FINAL PROJECT HANDOVER — SignBridge AI (Person 2)

> **Date:** 2026-11-26  
> **Branch:** `mediapipe-feature-engineering` (HEAD: `dd7b34e`, uncommitted changes present)  
> **Scope:** Person 2 (MediaPipe Feature Engineering + BiLSTM)  
> **Hardware:** Dell Latitude 7400, Intel Core i5-8365U, 8 GB RAM, Intel UHD 620, CPU-only (no NVIDIA GPU / CUDA)

---

## 1. Project Overview

**SignBridge AI** is a 3-person university AI project aimed at recognizing Arabic Sign Language gestures from video and converting them into understandable text.

- **Dataset:** A 70-class subset of **KArSL-502** (Saudi Arabic Sign Language; *not* confirmed Egyptian Sign Language):
  - **560 MP4 videos** total (~8 videos per class), originally `1920x1080` at `30 FPS` (durations `0.37s`–`1.20s`, mean `0.55s`).
  - **70 classes** (`y` mapped to `0..69`):
    - Classes `0..30` (SignID `0001..0031`): 31 number signs, performed by signer `01` (`01_02_*`).
    - Classes `31..69` (SignID `0032..0070`): 39 letter/hamza-related signs, performed by signer `02` (`02_02_*`).
- **Team Roles:**
  - **Person 1:** Dataset collection, cleaning, preprocessing (`16` sampled frames, square crop, `224x224` resize for `R(2+1)D-18`), and shared train/val/test split (`split_date.csv`).
  - **Person 2 (this handover):** MediaPipe Holistic landmark extraction, feature engineering, feature caching, BiLSTM model development, CPU training, multi-seed evaluation, confusion matrix & per-class reporting, and documentation.
  - **Person 3:** AI Agent, application integration, and visualization.

---

## 2. Initial State vs Final State

When first inspected, Person 2's `mediapipe_lstm.py` (which already had uncommitted edits on top of commit `dd7b34e`) had been executed end-to-end (`v1`), producing `keypoints/{train,val,test}.npz`, `best_lstm_seed_{42,123,2026}.pt`, `confusion_matrix_seed_*.csv`, `per_class_accuracy_seed_*.csv`, and `lstm_results.csv`. However, **`v1` achieved `72.86%–84.29%` validation accuracy but `0.00%–0.49%` test accuracy**.

To preserve all original `v1` artifacts while fixing the pipeline, the updated implementation (`v2`) writes to separate versioned paths (`keypoints/cache_v2/`, `keypoints/*_v2.npz`, `best_lstm_v2_seed_*.pt`, `*_v2_*.csv`).

| Component | Initial State (`v1`) | Final State (`v2`) | Reason for Change |
|---|---|---|---|
| **MediaPipe Config** | `HolisticLandmarker` (`IMAGE` mode, `min_pose=0.3`, `min_hand=0.3`), `holistic_landmarker.task` | Unchanged | Hand detection coverage was already `>99.9%`. |
| **Frame Preprocessing** | Resize width to `960` (yielding `960x540`), sample `16` RGB frames uniformly | Same + tracks `aspect_ratio = w / float(h)` (`1.778`) | MediaPipe `x` is normalized by width (`960`) and `y` by height (`540`); multiplying `x` by `aspect_ratio` restores isotropic 2D geometry. |
| **Landmark Selection** | 6 pose (`11..16`) × 3 + 21 left hand × 3 + 21 right hand × 3 + 2 flags | 20 non-wrist left hand × 2 (`x,y`) + 20 non-wrist right hand × 2 (`x,y`) + 2 flags | Pose (`84–87%` date-ID accuracy) and wrist-to-shoulder (`74%` date-ID accuracy) leaked session/date shortcuts; `z` introduced cross-session depth shift. |
| **Feature Dimensions** | `146`/frame (`292` with velocity) | `82`/frame (`164` with velocity) | Smaller, session-invariant feature vector reduces overfitting on 285 training videos. |
| **Coordinate Normalization** | All pose & hand `(x,y,z)` centered on `(pose[11]+pose[12])/2` and divided by shoulder width | Each hand centered on its own wrist (`lm[0]`) and divided by palm size (`\|\|lm[9]-lm[0]\|\| + 1e-4`) | Eliminated the `hand.z - pose_origin.z` coordinate-space bug and shoulder-framing shift across recording days. |
| **Missing-Landmark Handling** | Undetected set to `0` before standardization; standardization shifted `0` to `-mean/std` | Undetected masked to `0.0` **both before and after** standardization; stats computed only on detected training frames | Prevents missing-hand frames from injecting non-zero bias after standardization. |
| **Feature Caching** | `keypoints/cache/<sha1(path)>.npy`, `keypoints/{train,val,test}.npz` | `keypoints/cache_v2/<sha1(version:path)>.npy`, `keypoints/{train,val,test}_v2.npz` | Prevents stale cache hits when features change; preserves `v1` files. |
| **BiLSTM Architecture** | `Linear(292,128)` → 2-layer BiLSTM (`hidden=128`, `drop=0.3`) → mean pool → `Linear(256,70)` | `Linear(164,128)` → 1-layer BiLSTM (`hidden=128`, `drop=0.5`) → mean pool → `Linear(256,70)` | 2-layer BiLSTM overfit 285 samples; 1-layer with `dropout=0.5` generalizes better. |
| **Training & Augmentation** | Scale (`0.9–1.1`), Gaussian noise (`0.02`), temporal shift (`-2..+2`); `AdamW(lr=1e-3, wd=1e-2)`, `batch=64`, `patience=15` | Added `±15°` 2D hand rotation, scale (`0.9–1.1`), noise (`0.03`), shift (`-2..+2`); `AdamW(lr=2e-3, wd=5e-2)`, `batch=32`, `patience=20` | Rotation augmentation bridges wrist-angle differences across sessions; stronger weight decay reduces overfitting. |
| **Script Modularity** | Top-level execution on import | Wrapped in functions + `if __name__ == "__main__":` guard | Allows safe importing for tests and integration without triggering training. |

---

## 3. Root Cause Investigation

### Observed Failure (`v1`)
- Validation accuracy: `72.86%` (seed 42), `81.43%` (seed 123), `84.29%` (seed 2026).
- Test accuracy: `0.00%` (seed 42), `0.00%` (seed 123), `0.49%` (seed 2026); mean `0.16%` (worse than `1/70 = 1.43%` random).

### Investigations & Eliminated Causes
- **Eliminated — Label / Split Mismatch:** Verified `np.array_equal` between `keypoints/{train,val,test}.npz["y"]` and `split_date.csv` (`285/70/205`, labels `0..69`).
- **Eliminated — Evaluation / Checkpoint Bug:** Re-running the saved `best_lstm_seed_*.pt` checkpoints confirmed `99.65%` train top-1, `72.86%–84.29%` val top-1, and `0.00%–0.49%` test top-1 (`5.37%–5.85%` test top-5).
- **Eliminated — Horizontal Mirroring:** Horizontally flipping test landmarks and swapping left/right hands yielded `3.41%` 1-NN test accuracy.

### Confirmed Root Causes & Evidence
1. **Coordinate-Space Mismatch in `Z`-Axis Normalization:**
   - `mediapipe_lstm.py` (`v1`) subtracted `origin = (pose[11] + pose[12]) / 2` from **both** pose and hand landmarks.
   - In MediaPipe Holistic, `pose.z` is **hip-centered** (e.g., class `0` `origin.z = -0.279` on `15-11-16` vs `-0.009` on `16-11-16`), whereas `hand.z` is **wrist-centered** (`wrist.z ≈ 0.000`).
   - Subtracting `pose_origin.z` shifted all test hand `z` coordinates relative to train/val (`test` mean `lh_z = -0.462`, `rh_z = -0.504` vs `train` `lh_z = -0.127`, `rh_z = -0.131`).
2. **Session/Date Shortcut Learning Induced by `split_date.csv`:**
   - In `split_date.csv`, **all 70/70 classes** have `Train` and `Val` on the **same recording date** (`15-11-16` for 51 classes, `16-11-16` for 19 classes) and `Test` exclusively on the **opposite date**.
   - Using `v1` confusion matrices, we verified that:
     - Seed 42: **185 / 205 (90.2%)** of test predictions were classes whose `train_date == test_date(y_true)`.
     - Seed 123: **183 / 205 (89.3%)** had `train_date == test_date(y_true)`.
     - Seed 2026: **173 / 205 (84.4%)** had `train_date == test_date(y_true)`.
   - Because `train_date[y] != test_date[y]` for every class, predicting a class trained on the test video's date **mathematically guarantees an error**.
3. **Pose & Wrist-to-Shoulder Session Leakage (Verified by 1-NN Date-ID Test):**
   - Predicting the recording date (`15-11-16` vs `16-11-16`) via 1-NN on test samples gave:
     - `Pose XYZ (18)`: **87.32%** date accuracy (`1.95%` sign accuracy)
     - `Pose XY (12)`: **84.39%** date accuracy (`1.46%` sign accuracy)
     - `Wrist-to-shoulder XY (4)`: **74.15%** date accuracy (`2.93%` sign accuracy)
     - `Wrist-centered palm-scaled hands XY`: **41.46%** date accuracy (**`27.32%`** sign accuracy)

### Remaining Uncertainty
- Even with session-invariant hand shape (`v2`), cross-day test accuracy (`24.23%`) remains much lower than same-session validation accuracy (`90.48%`). Because `v2` intentionally discards global hand position relative to the body (to prevent date leakage), signs that share identical handshapes at different spatial locations (e.g., near face vs. chest) cannot be disambiguated by handshape alone without a multi-session/multi-signer training set.

---

## 4. Exact Final Project Structure

```text
AIProject/
├── .venv/                              # Python 3.14.3 CPU virtual environment (required)
├── dataset_clean.csv                   # Person 1: cleaned metadata (560 rows, SignID, Arabic/English labels)
├── dataset_metadata.csv                # Person 1: raw dataset metadata
├── video_properties.csv                # Person 1: video resolution, FPS, frame count, duration
├── split_date.csv                      # Person 1: shared train/val/test split (285/70/205) [REQUIRED]
├── holistic_landmarker.task            # MediaPipe HolisticLandmarker model bundle (~13.7 MB) [REQUIRED]
├── mediapipe_lstm.py                   # Person 2: final v2 extraction, BiLSTM, training & eval [MODIFIED]
├── README.md                           # Project overview + Person 2 docs & results [MODIFIED]
├── PERSON2_HANDOVER.md                 # This handover document [NEW]
├── requirements.txt                    # Python dependencies [REQUIRED]
├── project.ipynb                       # Person 1 / R(2+1)D-18 notebook (untouched)
├── download_dataset.py                 # Dataset download helper (untracked)
├── sample_download/extracted/          # 560 MP4 videos in 70 folders (0001..0070) [REQUIRED for extraction]
├── keypoints/
│   ├── cache/                          # Original v1 per-video .npy cache (preserved)
│   ├── train.npz, val.npz, test.npz    # Original v1 combined datasets (preserved)
│   ├── cache_v2/                       # Final v2 per-video .npy cache (82 dims/frame) [NEW, generated]
│   └── train_v2.npz, val_v2.npz, test_v2.npz  # Final v2 combined datasets [NEW, generated]
├── best_lstm_seed_{42,123,2026}.pt     # Original v1 checkpoints (preserved)
├── best_lstm_v2_seed_{42,123,2026}.pt  # Final v2 checkpoints [NEW, generated]
├── lstm_results.csv                    # Original v1 results (preserved)
├── lstm_v2_results.csv                 # Final v2 3-seed summary [NEW, generated]
├── confusion_matrix_seed_*.csv         # Original v1 confusion matrices (preserved)
├── confusion_matrix_v2_seed_*.csv      # Final v2 confusion matrices (70x70) [NEW, generated]
├── per_class_accuracy_seed_*.csv       # Original v1 per-class CSVs (preserved)
└── per_class_accuracy_v2_seed_*.csv    # Final v2 per-class accuracy CSVs [NEW, generated]
```

---

## 5. Final Implementation Details (`mediapipe_lstm.py`)

### Feature Extraction Pipeline
- **Input video format:** Original `1920x1080` MP4 at `30 FPS`, resized to `FRAME_WIDTH = 960` (`960x540`, `aspect_ratio = 1.778`), converted to RGB.
- **Frame sampling:** `T = 16` frames uniformly sampled via `np.linspace(0, len(frames) - 1, 16).round().astype(int)`.
- **MediaPipe model/API:** `mediapipe.tasks.python.vision.HolisticLandmarker` (`IMAGE` mode, `holistic_landmarker.task`, `min_pose_detection_confidence=0.3`, `min_hand_landmarks_confidence=0.3`).
- **Landmark groups:** `left_hand_landmarks` (21 landmarks) and `right_hand_landmarks` (21 landmarks). Pose and face landmarks are not used in `v2` features.
- **Coordinate normalization:** For each detected hand, `x` is multiplied by `aspect_ratio`, wrist (`lm[0]`) is subtracted, coordinates are divided by palm size (`||lm[9] - lm[0]|| + 1e-4`), and wrist (`lm[0]`) is dropped → `20 × 2 = 40` features per hand.
- **Hand detection flags:** `2` binary floats `[left_detected, right_detected]` (`0.0` or `1.0`).
- **Features per frame:** `40 + 40 + 2 = 82` (`164` after concatenating temporal velocity `np.diff(x, axis=0, prepend=x[:1])`).
- **Missing landmark treatment:** Undetected hands are set to `0.0` during extraction and re-masked to `0.0` after `(x - mean) / std` standardization. Standardization `mean` and `std` are fit strictly on detected training hand frames.
- **Cache format & location:** Per-video `float32` `(16, 82)` `.npy` files in `keypoints/cache_v2/<sha1("v2_hand_iso_wc_palm_xy:" + path)>.npy`; split arrays in `keypoints/{train,val,test}_v2.npz`.

### BiLSTM Model & Training
- **Input dimensions:** `(batch, 16, 164)`
- **Architecture:** `Linear(164, 128) → LayerNorm(128) → ReLU → Dropout(0.5)` → `LSTM(128, 128, num_layers=1, bidirectional=True)` → temporal `mean(dim=1)` → `Dropout(0.5) → Linear(256, 70)`
- **Loss function:** `F.cross_entropy(..., label_smoothing=0.1)`
- **Optimizer & Scheduler:** `AdamW(lr=2e-3, weight_decay=5e-2)`, `CosineAnnealingLR(T_max=100)`, gradient norm clipped to `1.0`
- **Batch size / Epochs / Early stopping:** `BATCH_SIZE = 32`, `EPOCHS = 100`, early stopping on validation accuracy with `PATIENCE = 20`
- **Augmentation (train only):** Random 2D in-plane hand rotation (`±15°` independently per hand), coordinate scale (`0.9–1.1`), masked Gaussian noise (`σ = 0.03`), clipped temporal shift (`-2..+2`)
- **Random seeds:** `[42, 123, 2026]`

---

## 6. Final Experimental Results (`v2`)

| Metric | Result |
|---|---|
| Train accuracy (`v2`) | NOT MEASURED (not logged in `v2` training loop) |
| Validation accuracy (`v2`, 3 seeds) | `0.9048 ± 0.0297` (`92.86%`, `87.14%`, `91.43%`) |
| Test accuracy (`v2`, per seed) | `27.32%` (42), `23.41%` (123), `21.95%` (2026) |
| Mean test accuracy (`v2`) | `0.2423` (`24.23%`) |
| Standard deviation (`ddof=1`) | `0.0277` (`2.77%`) |
| Number of seeds | `3` (`42, 123, 2026`) |
| Best epoch (per seed) | `50` (seed 42), `31` (seed 123), `45` (seed 2026) |
| Total runtime (`task-99`) | ~11m 33s total (`10m 09s` extraction + `~1m 24s` 3-seed training/eval on CPU) |

### Per-Seed Breakdown & Artifacts

| Seed | Best Val Acc | Test Acc | Numbers (0–30) | Letters (31–69) | Checkpoint | Confusion Matrix | Per-Class CSV |
|---:|---:|---:|---:|---:|---|---|---|
| 42 | 92.86% | 27.32% | 27.91% | 26.89% | `best_lstm_v2_seed_42.pt` | `confusion_matrix_v2_seed_42.csv` | `per_class_accuracy_v2_seed_42.csv` |
| 123 | 87.14% | 23.41% | 22.09% | 24.37% | `best_lstm_v2_seed_123.pt` | `confusion_matrix_v2_seed_123.csv` | `per_class_accuracy_v2_seed_123.csv` |
| 2026 | 91.43% | 21.95% | 24.42% | 20.17% | `best_lstm_v2_seed_2026.pt` | `confusion_matrix_v2_seed_2026.csv` | `per_class_accuracy_v2_seed_2026.csv` |

- **Summary CSV:** `lstm_v2_results.csv`
- **Frames without any detected hand (`v2`):** Train `0.02%`, Val `0.00%`, Test `0.06%`.
- **Reproducibility:** Successfully executed end-to-end from raw videos in `task-99` (`exit code 0`).
- **Failed / Intermediate Prototype Experiments (preserved for reference):**
  - `v1` baseline (`lstm_results.csv`): Val `72.86%–84.29%`, Test `0.00%–0.49%`.
  - `task-59` (`v2` prototype keeping `pose_xy` + `wrist_xy`, 60 epochs, seed 42): Best Val `61.43%`, Test `6.34%` (still leaked session cues).
  - `task-64` (hand-only, 2-layer BiLSTM `hidden=128` without zero-masking after standardization, 80 epochs, seed 42): Best Val `65.71%–67.14%`, Test `6.83%–7.32%`.
  - `task-78` (hand-only, masked standardization, no rotation aug, 100 epochs, seed 42): Best Val `91.43%`, Test `19.51%`.

---

## 7. Dataset and Evaluation Limitations

- **Date-based split (`split_date.csv`), NOT random or signer-independent (Verified):** For all 70 classes, `train` + `val` come from one recording day (`15-11-16` or `16-11-16`) and `test` comes exclusively from the opposite day.
- **Same-session validation optimism (Verified):** `train` and `val` videos are consecutive takes recorded seconds apart, which inflates `val` accuracy (`90.48%`) relative to cross-day `test` accuracy (`24.23%`).
- **No signer-independent evaluation possible (Verified):** Signer `01` performs only classes `0..30` and signer `02` performs only classes `31..69` (as documented in `README.md` and `split_date.csv` filename prefixes). No class has multiple signers.
- **Class imbalance & small sample size (Verified):** Total 560 videos (~8/class); `train` has `3–6` videos/class, `val` has `1` video/class, `test` has `1–4` videos/class.
- **Short video frame repetition (Documented in `README.md`):** ~56% of videos have `<16` frames (min `0.37s` ≈ `11` frames), causing duplicate frames when sampling `T=16`.

---

## 8. How to Run the Final Pipeline (Windows CMD)

Run all commands from the project root `c:\Users\standalone\Desktop\AIProject`:

```cmd
cd /d c:\Users\standalone\Desktop\AIProject
.venv\Scripts\activate.bat
```

Install dependencies only if setting up a fresh environment:
```cmd
pip install -r requirements.txt
```

Extract `v2` features (uses `keypoints/cache_v2/` if present), train all 3 seeds, evaluate on test, and generate `lstm_v2_results.csv`, `confusion_matrix_v2_seed_*.csv`, and `per_class_accuracy_v2_seed_*.csv`:
```cmd
python mediapipe_lstm.py
```

- **Expected output:** Extraction progress bars (~10 min if cache is empty, <5s if `keypoints/cache_v2/` exists), training logs for seeds `42`, `123`, `2026` (~1.5 min total on CPU), and final summary (`Val Accuracy: 0.9048 ± 0.0297`, `Test Accuracy: 0.2423 ± 0.0277`).

---

## 9. Git and Repository Status

- **Current branch:** `mediapipe-feature-engineering`
- **Latest commit:** `dd7b34e` (`Initial commit`)
- **Working tree status:** **Uncommitted changes** (NOT clean; NOT pushed):
  - **Modified (tracked):** `mediapipe_lstm.py`, `README.md`
  - **Untracked (`v1` & `v2` outputs + `download_dataset.py` + `PERSON2_HANDOVER.md`):** `lstm_results.csv`, `lstm_v2_results.csv`, `confusion_matrix_seed_*.csv`, `confusion_matrix_v2_seed_*.csv`, `per_class_accuracy_seed_*.csv`, `per_class_accuracy_v2_seed_*.csv`, `download_dataset.py`, `PERSON2_HANDOVER.md` (note: `.pt`, `.npz`, `.task`, `.7z`, `keypoints/`, `sample_download/`, `credentials.json`, `token.json` are gitignored or untracked).
- **Original data/implementation preserved:** Yes. All original `v1` caches (`keypoints/cache/`), NPZs (`keypoints/{train,val,test}.npz`), checkpoints (`best_lstm_seed_*.pt`), and results (`lstm_results.csv`) were left untouched.

---

## 10. Remaining Work

### Critical
- None for Person 2 implementation or inference; only final Git commit/push (if desired by the team).

### Important
- **Compare `v2` against Person 1's `R(2+1)D-18` results** once `R(2+1)D-18` is evaluated on `split_date.csv` (currently `TBD` in `README.md`).

### Optional
- Explore signer-agnostic / cross-session spatial normalization (e.g., hand-to-face/chin distance invariant to camera framing) if higher cross-day accuracy than `24.23%` is needed without leaking date.
- Log `train_accuracy` during training and export training curves.

---

## 11. Integration Handover to Person 3

A standalone, tested inference module is now ready in `infer.py`:

- **Import & Call:**
  ```python
  from infer import predict_video

  result = predict_video("sample_download/extracted/0001/01_02_0001_(15_11_16_16_41_12)_c.mp4")
  ```
- **CLI Usage (Windows CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  python infer.py "path\to\video.mp4" --checkpoint best_lstm_v2_seed_42.pt --top-k 5 --device cpu
  ```
- **Required Artifacts:**
  - `best_lstm_v2_seed_42.pt` (default checkpoint; `27.32%` test accuracy)
  - `keypoints/standardization_v2.npz` (training `mean` and `std`, shape `(82,)`)
  - `holistic_landmarker.task` (MediaPipe model bundle)
  - `dataset_clean.csv` (`SignID`, `Arabic_Label`, `English_Label`; verified `class_index = SignID - 1`)
- **Expected model input (internal):** `torch.FloatTensor` of shape `(1, 16, 164)` (82 `v2` features + 82 temporal velocity features).
- **Return dictionary:** `class_index`, `sign_id`, `arabic_label`, `english_label`, `confidence`, `top_k`, `top_5`, `probabilities` (length 70), `frames_without_hand_pct`, `inference_time_seconds`, `checkpoint_path`, `device`.
- **Verified Inference Test Results (`test_inference.py`, Tests A–F all PASSED):**
  - `train` (`01_02_0001_(16_11_16_15_50_08)_c.mp4`): GT=`0` | Pred=`0` | Conf=`95.63%` | Correct=`True` | Time=`2.48s`
  - `val` (`01_02_0001_(16_11_16_15_50_51)_c.mp4`): GT=`0` | Pred=`0` | Conf=`59.04%` | Correct=`True` | Time=`2.28s`
  - `test` (`01_02_0001_(15_11_16_16_41_12)_c.mp4`): GT=`0` | Pred=`0` | Conf=`65.45%` | Correct=`True` | Time=`2.11s` (`2.23s` via CLI)
  - Reproducibility (Test E): Identical predictions and probabilities (`atol=1e-7`) across repeated runs.
  - Error handling (Test F): Verified `FileNotFoundError` (missing video/checkpoint/stats), `ValueError` (corrupt/zero-frame MP4), and `RuntimeError` (blank MP4 with 0 detected hands).
- **Real-time inference:** **Not supported** out-of-the-box; operates on complete video files (`~2.1–2.5s` per video on CPU).

---

## 12. Final Verification Checklist

- [x] **Completed:** Dataset loading (`split_date.csv` validated, 285/70/205)
- [x] **Completed:** Feature extraction (`v2`, 560 videos extracted cleanly)
- [x] **Completed:** Feature caching (`keypoints/cache_v2/`, `keypoints/*_v2.npz`)
- [x] **Completed:** Landmark normalization (isotropic wrist-centered, palm-scaled)
- [x] **Completed:** Standardization stats saved (`keypoints/standardization_v2.npz`)
- [x] **Completed:** BiLSTM training (3 seeds on CPU)
- [x] **Completed:** Validation (`90.48% ± 2.97%`)
- [x] **Completed:** Test evaluation (`24.23% ± 2.77%`)
- [x] **Completed:** Multi-seed experiments (`42, 123, 2026`)
- [x] **Completed:** Confusion matrix (`confusion_matrix_v2_seed_*.csv`)
- [x] **Completed:** Per-class accuracy (`per_class_accuracy_v2_seed_*.csv`)
- [x] **Completed:** Missing-hand analysis (`0.02%` train, `0.00%` val, `0.06%` test)
- [x] **Completed:** Reproducibility (`mediapipe_lstm.py` + `test_inference.py` verified)
- [x] **Completed:** Documentation (`README.md` + `PERSON2_HANDOVER.md`)
- [x] **Completed:** Person 3 integration readiness (`infer.py` + `test_inference.py` Tests A–F passed)

---

## 13. CONTEXT FOR THE NEXT AI ASSISTANT

- **Current Project State:** You are on branch `mediapipe-feature-engineering` (HEAD `dd7b34e`, uncommitted changes in `mediapipe_lstm.py` and `README.md`, plus `v2` outputs and `PERSON2_HANDOVER.md`). Python 3.14.3 `.venv` is CPU-only (`torch` CPU, `mediapipe`, `opencv-python`).
- **Final Working Implementation (`v2`):** `mediapipe_lstm.py` extracts 16 frames/video, computes isotropic (`x * aspect_ratio`) wrist-centered, palm-scaled 2D hand coordinates (20 non-wrist landmarks × 2 per hand + 2 detection flags = `82` dims; `164` with velocity), masks undetected hands to `0.0` before and after standardization, applies `±15°` 2D hand rotation augmentation, and trains a 1-layer BiLSTM (`hidden=128`, `dropout=0.5`) across seeds `42, 123, 2026`.
- **Verified Results (`split_date.csv`, unchanged):**
  - `v1` (broken): Val `72.86%–84.29%`, Test `0.00%–0.49%` (mean `0.16%`).
  - `v2` (final): Val `90.48% ± 2.97%`, Test `24.23% ± 2.77%` (`27.32%`, `23.41%`, `21.95%`).
- **What Was Attempted & Must NOT Be Repeated:**
  - Do **not** subtract `pose_origin.z` from `hand.z` (hip-centered vs. wrist-centered Z mismatch).
  - Do **not** feed raw shoulder-normalized `pose` or `wrist-to-shoulder` coordinates directly to the classifier on `split_date.csv` — they leak the recording date (`74%–87%` date-ID accuracy) and cause the model to predict only classes from the test video's date (`0%` test accuracy).
  - Do **not** switch `split_date.csv` to a random split; `split_date.csv` is the shared cross-session benchmark with Person 1's `R(2+1)D-18`.
  - Do **not** delete original `v1` caches/checkpoints without explicit user permission.
- **Next Recommended Task:**
  1. Commit Person 2's changes on `mediapipe-feature-engineering` if requested.
  2. If assisting Person 3, save standardization `(mean, std)` from `keypoints/train_v2.npz` alongside `best_lstm_v2_seed_42.pt` and build a clean `predict_video(path)` wrapper.
- **Known Risks:** `val` (`~90.5%`) is same-session and optimistic compared to cross-day `test` (`~24.2%`); dataset has only 2 non-overlapping signers (`01` = numbers, `02` = letters), so signer-independent generalization is untested.
