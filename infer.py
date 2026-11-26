"""
SignBridge AI — Person 2: Standalone BiLSTM Inference Module & CLI

Reuses exact V2 MediaPipe Holistic feature extraction, wrist-centered palm-scaled
normalization, masking, and BiLSTM architecture from `mediapipe_lstm.py`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from mediapipe_lstm import (
    BiLSTM,
    COORDS,
    COORDS_PER_HAND,
    F,
    FEATURE_VERSION,
    MODEL_PATH,
    NUM_CLASSES,
    T,
    read_video_frames,
    video_features,
)

DEFAULT_CHECKPOINT = Path("best_lstm_v2_seed_42.pt")
DEFAULT_STATS_PATH = Path("keypoints/standardization_v2.npz")
DEFAULT_LABELS_CSV = Path("dataset_clean.csv")


def load_label_map(labels_csv: str | Path = DEFAULT_LABELS_CSV) -> dict[int, dict[str, str]]:
    """
    Load class_index (0..69) -> {sign_id, arabic_label, english_label} from dataset_clean.csv.
    Verified mapping: class_index = SignID - 1.
    """
    csv_path = Path(labels_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Label mapping CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"SignID", "Arabic_Label", "English_Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    unique_rows = df[["SignID", "Arabic_Label", "English_Label"]].drop_duplicates()
    label_map: dict[int, dict[str, str]] = {}

    for _, row in unique_rows.iterrows():
        sign_num = int(row["SignID"])
        class_idx = sign_num - 1
        label_map[class_idx] = {
            "sign_id": f"{sign_num:04d}",
            "arabic_label": str(row["Arabic_Label"]),
            "english_label": str(row["English_Label"]),
        }

    expected = set(range(NUM_CLASSES))
    if set(label_map.keys()) != expected:
        raise ValueError(
            f"Label mapping incomplete: expected 0..{NUM_CLASSES - 1}, "
            f"missing {sorted(expected - set(label_map.keys()))}"
        )

    return label_map


def load_standardization(stats_path: str | Path = DEFAULT_STATS_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load training-derived mean and std vectors from standardization_v2.npz."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Standardization file not found: {path}")

    data = np.load(path)
    if "mean" not in data or "std" not in data:
        raise ValueError(f"Invalid standardization file {path}: missing 'mean' or 'std'")

    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

    if mean.shape != (F,) or std.shape != (F,):
        raise ValueError(f"Expected stats shape ({F},), got mean={mean.shape}, std={std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Standardization statistics contain NaN or Inf values.")

    if "feature_version" in data:
        saved_ver = str(data["feature_version"].item())
        if saved_ver != FEATURE_VERSION:
            raise ValueError(
                f"Feature version mismatch: stats={saved_ver}, pipeline={FEATURE_VERSION}"
            )

    return mean, std


def preprocess_features_for_model(
    raw_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Apply V2 standardization, undetected-hand re-masking, and temporal velocity.
    Input shape: (T, 82) -> Output shape: (1, T, 164).
    """
    if raw_features.shape != (T, F):
        raise ValueError(f"Expected raw features shape ({T}, {F}), got {raw_features.shape}")

    lf = (raw_features[:, -2:-1] > 0.5).astype(np.float32)
    rf = (raw_features[:, -1:] > 0.5).astype(np.float32)

    x = (raw_features - mean) / std
    x[:, :COORDS_PER_HAND] *= lf
    x[:, COORDS_PER_HAND:COORDS] *= rf

    velocity = np.diff(x, axis=0, prepend=x[:1])
    combined = np.concatenate([x, velocity], axis=1).astype(np.float32)
    return combined[None, :, :]


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> BiLSTM:
    """Load trained V2 BiLSTM weights on the requested device (CPU by default)."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}' is unavailable; CUDA is not available.")

    model = BiLSTM(in_dim=F * 2, hidden=128, num_layers=1, dropout=0.5, num_classes=NUM_CLASSES)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _create_landmarker(model_path: str | Path = MODEL_PATH) -> vision.HolisticLandmarker:
    """Initialize MediaPipe HolisticLandmarker with V2 configuration."""
    mp_model = Path(model_path)
    if not mp_model.exists():
        raise FileNotFoundError(f"MediaPipe model file not found: {mp_model}")

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(mp_model)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def predict_video(
    video_path: str | Path,
    checkpoint_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    labels_csv: str | Path | None = None,
    top_k: int = 5,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run end-to-end sign language prediction on a single MP4 video file.

    Returns a dictionary containing predicted class index, SignID, Arabic/English labels,
    confidence, top-k predictions, full 70-class probability list, and inference time.
    """
    t_start = time.perf_counter()

    vid_path = Path(video_path)
    if not vid_path.exists() or not vid_path.is_file():
        raise FileNotFoundError(f"Video file not found: {vid_path}")

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT
    stats = Path(stats_path) if stats_path is not None else DEFAULT_STATS_PATH
    labels = Path(labels_csv) if labels_csv is not None else DEFAULT_LABELS_CSV

    if top_k < 1 or top_k > NUM_CLASSES:
        raise ValueError(f"top_k must be between 1 and {NUM_CLASSES}, got {top_k}")

    mean, std = load_standardization(stats)
    label_map = load_label_map(labels)
    model = load_model(ckpt, device=device)

    landmarker = _create_landmarker(MODEL_PATH)
    try:
        try:
            frames, aspect_ratio = read_video_frames(vid_path)
        except Exception as exc:
            raise ValueError(f"Failed to read video '{vid_path}': {exc}") from exc

        raw_features = video_features(frames, aspect_ratio, landmarker)
    finally:
        landmarker.close()

    hand_flags = raw_features[:, -2:]
    if float(hand_flags.sum()) == 0.0:
        raise RuntimeError(
            f"MediaPipe failed to detect any left or right hand across all {T} frames in '{vid_path}'."
        )

    no_hand_pct = float((1.0 - hand_flags.max(axis=1).mean()) * 100.0)
    model_input = preprocess_features_for_model(raw_features, mean, std)
    tensor_x = torch.from_numpy(model_input).to(device)

    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    sorted_indices = np.argsort(-probs)
    best_idx = int(sorted_indices[0])
    best_info = label_map[best_idx]

    def _entry(idx: int) -> dict[str, Any]:
        info = label_map[idx]
        return {
            "class_index": idx,
            "sign_id": info["sign_id"],
            "arabic_label": info["arabic_label"],
            "english_label": info["english_label"],
            "probability": float(probs[idx]),
        }

    top_k_list = [_entry(int(i)) for i in sorted_indices[:top_k]]
    top_5_list = [_entry(int(i)) for i in sorted_indices[:5]]

    elapsed = time.perf_counter() - t_start

    return {
        "class_index": best_idx,
        "sign_id": best_info["sign_id"],
        "arabic_label": best_info["arabic_label"],
        "english_label": best_info["english_label"],
        "confidence": float(probs[best_idx]),
        "top_k": top_k_list,
        "top_5": top_5_list,
        "probabilities": [float(p) for p in probs],
        "frames_without_hand_pct": no_hand_pct,
        "inference_time_seconds": float(elapsed),
        "checkpoint_path": str(ckpt),
        "device": device.upper(),
    }


def _format_cli_output(video_path: str, result: dict[str, Any], top_k: int) -> str:
    ckpt_name = os.path.basename(result["checkpoint_path"])
    lines = [
        "=" * 40,
        "         SignBridge AI - Prediction",
        "=" * 40,
        "",
        f"Video: {os.path.basename(video_path)}",
        f"Model: BiLSTM V2 ({ckpt_name})",
        f"Device: {result['device']}",
        "",
        "Predicted Sign:",
        f"Arabic: {result['arabic_label']}",
        f"English: {result['english_label']}",
        f"SignID: {result['sign_id']}",
        f"Class Index: {result['class_index']}",
        f"Confidence: {result['confidence'] * 100:.2f}%",
        "",
        f"Top {top_k} Predictions:",
    ]
    for rank, item in enumerate(result["top_k"], start=1):
        lines.append(
            f"{rank}. {item['english_label']} ({item['arabic_label']}) - "
            f"{item['probability'] * 100:.2f}%"
        )
    lines.extend(["", f"Inference Time: {result['inference_time_seconds']:.2f}s", "=" * 40])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SignBridge AI — Person 2 BiLSTM V2 Video Inference"
    )
    parser.add_argument("video_path", type=str, help="Path to input MP4 video file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run on (default: cpu)",
    )
    args = parser.parse_args()

    result = predict_video(
        video_path=args.video_path,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        device=args.device,
    )
    print(_format_cli_output(args.video_path, result, args.top_k))


if __name__ == "__main__":
    main()
