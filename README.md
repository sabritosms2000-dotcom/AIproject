Analyzed README.md:L1-140
Edited README.md

# SignBridge AI – Egyptian Sign Language Recognition & Translation

## 🎯 Project Goal
SignBridge AI is a research‑grade application that recognises **70 Arabic sign‑language gestures** (31 numbers + 39 letters) from video and provides real‑time feedback. It showcases two independent pipelines:
1. **R(2+1)D‑18** – a pretrained 3‑D CNN fine‑tuned on raw video frames (outside Person 2’s scope).  
2. **MediaPipe Holistic + BiLSTM (v2)** – a handcrafted, CPU‑friendly pipeline built by **Person 2**.

The final deliverable is a **desktop UI** that:
- Captures webcam video, predicts the sign, displays the label and confidence, and speaks the translation via Arabic TTS.  
- Allows text‑to‑sign playback, speech‑to‑sign (currently a placeholder), and stores a history of predictions.  
- Exposes a clean Python API (`infer.py`) for **Person 3** to integrate the model into other services.

---

## 👥 Team Roles
| Role | Responsibility |
|------|-----------------|
| **Person 1** | Data collection, dataset preparation, baseline model training (R(2+1)D‑18). |
| **Person 2** | MediaPipe feature engineering, BiLSTM V2 model training, inference CLI, evaluation scripts, and the full desktop UI. |
| **Person 3** | Packaging, deployment, analytics dashboard, AI‑assistant integration, and downstream services. |

---

## 📂 Repository Structure
```
AIProject/
├─ .venv/                     # Python virtual environment (do not commit)
├─ keypoints/                 # Cached feature files & standardisation vectors
│   ├─ cache_v2/               # Per‑video .npy cache (feature version v2)
│   ├─ standardization_v2.npz # mean / std for 164‑dim input
│   └─ {train,val,test}_v2.npz
├─ sample_download/extracted/ # 560 MP4 videos (8 per class) – *not included*
├─ mediapipe_lstm.py          # Feature extraction + BiLSTM training code
├─ infer.py                   # Python API + CLI for single‑video inference
├─ evaluate_test_subset.py    # Quick sanity‑check on 10 test videos
├─ webcam_app.py / app.py     # Real‑time Tkinter UI (6 tabs)
├─ run_app.bat                # Windows shortcut to launch the UI
├─ dataset_clean.csv          # Label map: SignID → Arabic/English labels
├─ split_date.csv             # Train/val/test split (recording‑date based)
├─ best_lstm_v2_seed_42.pt    # Trained BiLSTM checkpoint (seed = 42)
├─ holistic_landmarker.task   # MediaPipe model bundle (download required)
├─ tests/                     # pytest suite – all 7 tests currently pass
└─ README.md                  # **You are reading this file**
```

---

## 📦 Setup & Installation (Windows 11, CPU only)

1. **Clone the repo** and open a terminal in `C:\Users\standalone\Desktop\AIProject`.  
2. **Create / activate the virtual environment** (already committed as `.venv`):
   ```powershell
   .\.venv\Scripts\activate.bat
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Download the MediaPipe model** (required only once):
   ```powershell
   curl -L -o holistic_landmarker.task `
        https://github.com/google/mediapipe/releases/download/v0.10.9/holistic_landmarker.task
   ```
5. **Place the dataset**:
   - Obtain the KArSL‑502 subset (560 videos) from the original source.  
   - Extract the videos to `sample_download\extracted\<SignID>\*.mp4`.  
   - `split_date.csv` already points to the correct paths.

---

## 🛠️ How to Run the Different Parts

### 1️⃣ Feature Extraction & Model Training (Person 2)
```powershell
python mediapipe_lstm.py   # run cell‑by‑cell (Jupyter/Colab) or as a script
```
The script will:
- Validate `split_date.csv`.  
- Extract hand‑keypoints (caching in `keypoints/cache_v2`).  
- Train the BiLSTM for the seeds defined in `SEEDS` ([42, 123, 2026]).  
- Save checkpoints, per‑seed metrics, and aggregated results.

> **Note:** The script is designed for interactive execution but can run non‑interactively; it pauses before long steps.

### 2️⃣ Single‑Video Inference (Person 3 API)
```powershell
python infer.py path\to\video.mp4          # defaults to seed 42 checkpoint
```
Optional flags: `--checkpoint <path>`, `--top-k <N>` (default 5), `--device cpu`.

Programmatic usage:
```python
from infer import predict_video
result = predict_video('sample_download/extracted/0001/01_02_0001.mp4')
print(result['arabic_label'], result['confidence'])
```
Returned dict includes class index, SignID, Arabic/English labels, confidence, top‑k list, full probability vector, inference time, etc.

### 3️⃣ Real‑Time Webcam Application (Person 2 → Person 3 hand‑off)
```powershell
run_app.bat   # launches the Tkinter UI
```
The UI contains six tabs:

1. **Sign → Speech** – live webcam, prediction, sentence builder, Arabic TTS.  
2. **Speech → Sign** – microphone recorder (STT currently unavailable for Arabic).  
3. **Text → Sign** – type Arabic text → video playlist playback.  
4. **History** – SQLite DB of past predictions.  
5. **AI Assistant** – placeholder for future conversational features.  
6. **Analytics** – basic stats from `lstm_v2_results.csv` (expandable).

---

## 📊 Evaluation & Metrics (Person 2)

- **Validation accuracy** (same‑day split) ≈ **90 %** (±3 %).  
- **Cross‑day test accuracy** ≈ **24 %** (±3 %).  
- Per‑class accuracy and confusion matrices are saved as `per_class_accuracy_v2_seed_<seed>.csv` and `confusion_matrix_v2_seed_<seed>.csv`.  
- Hand‑detection failure rate is < 0.1 %.  
- Inference time on the reference laptop (Intel i5‑8365U, 8 GB RAM) ≈ **1.5 s/video** (CPU).

Quick sanity‑check on a representative subset:
```powershell
python evaluate_test_subset.py
```
It prints a table of ground‑truth vs. prediction, overall accuracy, and saves `test_subset_results.csv`.

---

## 🧪 Running the Test Suite
```powershell
python -m pytest -q
```
All **7 tests** currently pass, covering:
- Feature pipeline consistency.  
- Model loading & inference correctness.  
- Arabic TTS generation.  
- History DB handling.  
- UI launch & tab navigation.  
- Text‑to‑Sign mapping.

---

## 📚 Known Limitations & Future Work (Person 3 guidance)

| Limitation | Impact | Suggested Fix |
|------------|--------|--------------|
| **Small, signer‑specific dataset** (2 signers) | Poor generalisation to new users or lighting. | Collect more diverse recordings; augment data (background, colour jitter). |
| **Arabic STT unavailable** | Speech‑to‑Sign tab cannot transcribe Arabic. | Install an Arabic speech‑recognition engine (e.g., Vosk `vosk-model-ar-mgup-0.22`). |
| **Low cross‑day test accuracy** (~24 %) | Many signs mis‑classified in real use. | Experiment with deeper models, add temporal augmentation, or fine‑tune on more data. |
| **Analytics dashboard is a placeholder** | No live performance monitoring. | Load full prediction logs (`lstm_v2_results.csv`) and visualise confusion matrix, per‑class recall, etc. |
| **AI Assistant tab is static** | No conversational support. | Hook to Gemini or another LLM, expose a `signbridge.assistant` module. |

---

## 📜 License & Citation
The code is released under the **MIT License**.  If you use this work in research, please cite the original KArSL‑502 dataset and the project repository.

---

## 🙋‍♀️ Getting Help
- **Issues / Bugs** – open a GitHub issue (if hosted) or contact the project maintainer.  
- **Questions** – use the `#signbridge‑ai` channel on the team Slack, or ask the AI Assistant placeholder (once implemented).

---

*Happy signing!*