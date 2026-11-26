"""Configuration, paths, and documented model parameters for SignBridge AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "best_lstm_v2_seed_42.pt"
STANDARDIZATION_PATH = PROJECT_ROOT / "keypoints" / "standardization_v2.npz"
LABELS_CSV = PROJECT_ROOT / "dataset_clean.csv"
SPLIT_CSV = PROJECT_ROOT / "split_date.csv"
MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "holistic_landmarker.task"
RESULTS_CSV = PROJECT_ROOT / "lstm_v2_results.csv"
HISTORY_DB_PATH = PROJECT_ROOT / "signbridge_history.db"

# Recognition & stability defaults
# Threshold 0.40 is an uncalibrated softmax probability cutoff chosen to filter
# low-confidence noise across 70 classes (random baseline = 1.43%).
# Model confidence is raw softmax probability (not temperature-calibrated).
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_HOLD_FRAMES_TO_COMMIT = 3
DEFAULT_COOLDOWN_SECONDS = 1.5

VOCABULARY_LIMITATION = (
    "Limited-vocabulary demonstration: supports only the 70 verified KArSL-502 signs "
    "(31 numbers + 39 Arabic letters/hamza forms). Does NOT support full Egyptian Sign Language."
)
MODEL_ACCURACY_LIMITATION = (
    "Cross-day test accuracy of BiLSTM V2 (seed 42) is 27.32% on 205 test videos "
    "(mean 24.23% ± 2.77% across 3 seeds; 92.86% same-session val). "
    "Confidence values are uncalibrated softmax probabilities."
)
