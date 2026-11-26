# SignBridge AI — Live Webcam Application Guide

## 1. How to Launch

### Option A — Double-Click Launcher (Windows)
Double-click **`run_webcam.bat`** in the project root (`C:\Users\standalone\Desktop\AIProject`).

### Option B — Command Prompt (CMD)
```cmd
cd /d C:\Users\standalone\Desktop\AIProject
.venv\Scripts\activate.bat
python webcam_app.py
```

Optional flags:
- `--camera-id 0` (change if using an external USB camera, e.g., `1`)
- `--threshold 0.40` (confidence threshold below which `Uncertain` is displayed)
- `--smooth 5` (number of recent predictions averaged to reduce flicker)
- `--frame-skip 1` (set to `2` on slower CPUs to process every 2nd frame)

---

## 2. Required Files

All of the following must be present in `C:\Users\standalone\Desktop\AIProject`:
- `webcam_app.py` (live webcam application)
- `infer.py` & `mediapipe_lstm.py` (V2 feature extraction & BiLSTM model definition)
- `best_lstm_v2_seed_42.pt` (trained V2 BiLSTM checkpoint)
- `keypoints\standardization_v2.npz` (training standardization `mean` and `std`)
- `holistic_landmarker.task` (MediaPipe HolisticLandmarker bundle)
- `dataset_clean.csv` (class `0..69` to SignID / Arabic / English label mapping)
- `.venv\` (Python 3.14.3 CPU virtual environment)

---

## 3. How to Use It

1. Launch the application; wait ~2–3 seconds for MediaPipe and the BiLSTM checkpoint to load on CPU.
2. When the camera window opens, position your upper body and hands clearly in view.
3. The app collects a **16-frame rolling buffer** (`Collecting frames (X/16)`) before displaying predictions.
4. Perform a sign from the 70-class KArSL subset.
5. **Keyboard Shortcuts:**
   - **`Q`** (or close the window `[X]`): Quit cleanly and release camera/MediaPipe resources.
   - **`R`**: Reset the 16-frame rolling buffer and smoothing history.

---

## 4. What Confidence Means

- Confidence is the smoothed softmax probability (`0%–100%`) averaged over the last `--smooth` (default `5`) predictions.
- If confidence is below `--threshold` (default `40%`), the overlay displays **`Uncertain (<label>?)`** instead of presenting a low-confidence guess as reliable.
- If no hand is detected across the 16-frame buffer, the overlay displays **`Uncertain (No hand)`**.

---

## 5. Known Limitations

- **Limited Cross-Session Accuracy:** On the 205-video cross-day test set (`split_date.csv`), `best_lstm_v2_seed_42.pt` achieved **`27.32%`** accuracy (`24.23% ± 2.77%` across 3 seeds), compared to `92.86%` same-session validation accuracy. Expect frequent `Uncertain` or misclassified predictions in live webcam use, especially for signs that differ primarily by body location rather than handshape.
- **2-Signer Training Subset:** The 70-class dataset was recorded by only two signers (`01` for numbers `0..30`, `02` for letters `31..69`). Generalization to new people is untested.
- **CPU Performance:** Running MediaPipe Holistic + BiLSTM on Intel Core i5-8365U (CPU-only) runs at ~10–15 FPS (~60–100 ms per processed frame). Use `--frame-skip 2` if the preview feels sluggish.

---

## 6. Troubleshooting

- **`Could not open webcam (camera_id=0)`:** Close other apps using the camera (Zoom, Teams, browser tabs), check Windows Camera Privacy settings, or try `python webcam_app.py --camera-id 1`.
- **`Missing file` errors:** Ensure you run from `C:\Users\standalone\Desktop\AIProject` and that `best_lstm_v2_seed_42.pt`, `keypoints\standardization_v2.npz`, `holistic_landmarker.task`, and `dataset_clean.csv` exist.
- **`No module named ...`:** Ensure `.venv` is activated (`run_webcam.bat` handles this automatically).
