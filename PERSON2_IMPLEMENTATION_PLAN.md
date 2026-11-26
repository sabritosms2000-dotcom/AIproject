# SignBridge AI — Person 2 Implementation Plan

> **Audit Date:** 2026-11-26  
> **Project Root:** `C:\Users\standalone\Desktop\AIProject`  
> **Branch:** `mediapipe-feature-engineering` (HEAD: `570f06a`)  
> **Environment:** Windows 11, Python 3.14.3 (`.venv`), Intel Core i5-8365U, 8 GB RAM, CPU-only

---

## 1. Stage 0 Audit Summary — Model Input & Existing Artifacts

- **Features per frame:** `82` (`40` wrist-centered palm-scaled 2D non-wrist left-hand coords + `40` right-hand coords + `2` hand detection flags); `164` after concatenating temporal velocity (`np.diff(..., prepend=x[:1])`).
- **Sequence length:** `T = 16` frames (`(1, 16, 164)` tensor input).
- **Feature ordering:** `[left_hand_xy (40), right_hand_xy (40), left_flag (1), right_flag (1)]` followed by `velocity (82)`.
- **Normalization procedure:**
  - `x` scaled by `aspect_ratio = w / h` (`1.778` for 16:9).
  - Each hand centered on its wrist (`lm[0]`) and divided by palm size (`||lm[9] - lm[0]|| + 1e-4`), dropping wrist `lm[0]`.
  - Standardized with `(x - mean) / std` from `keypoints/standardization_v2.npz` (computed over detected training hand frames only).
  - Undetected hands re-masked to `0.0` **after** standardization.
- **Class mapping:** `0..69` (`70` classes) mapped via `dataset_clean.csv` (`class_index = SignID - 1`):
  - `0..30`: 31 Arabic numbers (`صفر` through `عشرة ملايين`).
  - `31..69`: 39 Arabic letters & hamza forms (`ا` through `ال`).
- **Checkpoint architecture (`best_lstm_v2_seed_42.pt`):** `BiLSTM(in_dim=164, hidden=128, num_layers=1, dropout=0.5, num_classes=70)` with temporal mean pooling.
- **Prediction output format:** `infer.predict_video()` returns `class_index`, `sign_id`, `arabic_label`, `english_label`, `confidence`, `top_k`, `top_5`, `probabilities` (len 70), `frames_without_hand_pct`, `inference_time_seconds`, `checkpoint_path`, `device`.

---

## 2. Existing Working Components

1. `mediapipe_lstm.py` — V2 feature extraction (`HolisticLandmarker`), training, evaluation (`90.48% ± 2.97%` val, `24.23% ± 2.77%` cross-day test; `27.32%` seed 42).
2. `infer.py` — Standalone CPU inference (`predict_video()`, `load_model()`, `load_standardization()`, `load_label_map()`, `preprocess_features_for_model()`).
3. `webcam_app.py` — Working OpenCV webcam loop (`WebcamSignRecognizer`, 16-frame rolling buffer, 5-frame smoothing, `0.40` threshold, hand skeleton overlay).
4. `best_lstm_v2_seed_42.pt`, `keypoints/standardization_v2.npz`, `holistic_landmarker.task`, `dataset_clean.csv`, `split_date.csv`, and 560 MP4 videos in `sample_download/extracted/`.
5. **Verified Audio/UI Capabilities in `.venv` & Windows 11:**
   - `tkinter` + `PIL` (`ImageTk`) + `cv2` (built-in / installed).
   - `Microsoft Hoda (ar-EG)` offline Arabic TTS voice via Windows WinRT `Windows.Media.SpeechSynthesis` (tested: synthesized `83,566` bytes WAV offline).
   - `sounddevice` + `scipy` installed; `Microphone Array (Realtek(R) Audio)` available.

---

## 3. Missing Components

1. Unified 6-tab desktop application (`app.py` + modular `signbridge/` package) covering:
   - Sign to Speech (live webcam + sentence builder + duplicate suppression + pause/resume + Arabic TTS)
   - Speech to Sign (microphone start/stop + STT adapter + manual Arabic text fallback)
   - Text to Sign (Arabic normalization + phrase/word vocabulary matching + ordered MP4 playback)
   - Translation History (persistent local SQLite/JSON store + viewer + clear)
   - AI Assistant (Person 3 integration adapter + honest placeholder UI)
   - Analytics Dashboard (Person 3 integration adapter + real verified model/history stats)
2. Arabic Text-to-Speech module (`signbridge/tts.py`) using offline Windows `Microsoft Hoda (ar-EG)`.
3. Arabic Speech-to-Text module (`signbridge/stt.py`) with `sounddevice` recording + replaceable engine adapter.
4. Text-to-Sign vocabulary & video lookup module (`signbridge/text_to_sign.py`) using real `dataset_clean.csv` + `sample_download/extracted/` videos.
5. Local history persistence (`signbridge/history.py`).
6. Person 3 integration interfaces (`signbridge/person3_adapter.py`).
7. Comprehensive end-to-end test suite (`tests/test_app_suite.py`) and Windows launcher (`run_app.bat`).

---

## 4. Confirmed Bugs & Limitations

1. **Cross-day test accuracy (`27.32%` seed 42):** `split_date.csv` splits train/val and test across different recording days (`15-11-16` vs `16-11-16`), and the dataset only has 2 non-overlapping signers (`01` = numbers, `02` = letters). Confidence is uncalibrated softmax probability.
2. **No offline Arabic STT engine installed:** Windows `SpeechRecognizer` only has `en-US` installed, and `vosk` / `faster-whisper` / `SpeechRecognition` are not installed in `.venv`. Manual Arabic text input fallback must remain first-class.
3. **Vocabulary scope:** Only the 70 verified KArSL-502 signs (31 numbers + 39 letters/hamza forms) have real videos; general Egyptian Sign Language sentences and grammar are not supported.

---

## 5. Proposed Application Architecture

Use **`tkinter` + `PIL` (`ImageTk`)** (lightweight, zero new pip installs, 8 GB RAM friendly, native OpenCV frame & video rendering) organized cleanly as:

```text
AIProject/
├── app.py                        # Main 6-tab Tkinter desktop application
├── run_app.bat                   # Windows launcher (.venv check + launch app.py)
├── signbridge/
│   ├── __init__.py
│   ├── config.py                 # Paths, constants, verified threshold (0.40) & limitations
│   ├── recognizer.py             # Wraps WebcamSignRecognizer + sentence builder & duplicate suppression
│   ├── tts.py                    # Offline Windows ar-EG SpeechSynthesis -> WAV + winsound playback
│   ├── stt.py                    # sounddevice mic recorder + replaceable Arabic STT adapter
│   ├── text_to_sign.py           # Arabic text normalization, phrase/word matching, real video lookup
│   ├── history.py                # Local JSON/SQLite translation history persistence
│   └── person3_adapter.py        # Clean interfaces & fallback placeholders for AI Assistant & Analytics
└── tests/
    └── test_app_suite.py         # Automated tests (mocked hardware + real offline model/TTS/T2S/history)
```

---

## 6. Implementation Order

1. **Stage 0:** Complete audit & `PERSON2_IMPLEMENTATION_PLAN.md` (Done).
2. **Stage 1 & 8:** `signbridge/config.py`, `signbridge/history.py`, `signbridge/person3_adapter.py`.
3. **Stage 2 & 3:** `signbridge/recognizer.py` (reusing `webcam_app.py` / `infer.py` / `mediapipe_lstm.py` + sentence builder & duplicate hold suppression).
4. **Stage 4:** `signbridge/tts.py` (offline Windows `ar-EG` TTS).
5. **Stage 5:** `signbridge/stt.py` (`sounddevice` mic recording + replaceable STT adapter).
6. **Stage 6:** `signbridge/text_to_sign.py` (Arabic normalization + 70-class vocabulary & real MP4 lookup).
7. **Stage 1 (UI):** `app.py` (6-tab Tkinter GUI with RTL Arabic support, live webcam, video player, history, Person 3 tabs) + `run_app.bat`.
8. **Stage 9:** `tests/test_app_suite.py` — run all automated tests + real offline audio/video/model tests.
9. **Stage 10:** `PERSON2_IMPLEMENTATION_REPORT.md`, `PERSON2_TEST_REPORT.md`, `README.md` update.

---

## 7. Required Dependencies

All required packages are already present in `.venv` (`numpy`, `pandas`, `opencv-python`, `mediapipe`, `torch`, `Pillow`, `sounddevice`, `scipy`, `tkinter`, `winsound`, Windows WinRT `ar-EG` voice). No heavyweight frameworks or external paid services are required.
