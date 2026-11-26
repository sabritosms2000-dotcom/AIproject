"""
SignBridge AI — Person 2: Automated Inference Verification Suite

Covers Tests A through F:
  A. Checkpoint loading & architecture compatibility
  B. Standardization statistics & missing-hand masking equivalence
  C. Feature dimensions (16, 82) -> (1, 16, 164)
  D. Real video inference on 1 train, 1 val, and 1 test video
  E. Deterministic reproducibility across repeated runs
  F. Error handling (missing files, corrupt/zero-frame video, failed hand detection)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LABELS_CSV,
    DEFAULT_STATS_PATH,
    load_label_map,
    load_model,
    load_standardization,
    predict_video,
    preprocess_features_for_model,
)
from mediapipe_lstm import (
    COORDS,
    COORDS_PER_HAND,
    F,
    KPDataset,
    MODEL_PATH,
    NUM_CLASSES,
    T,
    read_video_frames,
    video_features,
)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def test_a_checkpoint_loading() -> None:
    print("[Test A] Checkpoint loading...")
    model = load_model(DEFAULT_CHECKPOINT, device="cpu")
    assert not model.training, "Model must be in eval() mode"
    dummy = torch.zeros((1, T, F * 2), dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Unexpected output shape: {out.shape}"
    print("  PASSED: Checkpoint loaded and verified on CPU.")


def test_b_standardization() -> None:
    print("[Test B] Standardization statistics & masking...")
    mean, std = load_standardization(DEFAULT_STATS_PATH)
    assert mean.shape == (F,) and std.shape == (F,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std > 0).all()

    # Zero-hand frame must remain zero in coordinate slots after masking
    zero_raw = np.zeros((T, F), dtype=np.float32)
    proc = preprocess_features_for_model(zero_raw, mean, std)
    assert proc.shape == (1, T, F * 2)
    assert np.all(proc[0, :, :COORDS] == 0.0), "Undetected hand coords were not masked to zero"

    # Must match KPDataset(train=False) bit-for-bit on real training sample
    tr = np.load("keypoints/train_v2.npz")
    ds = KPDataset(tr["X"], tr["y"], mean, std, train=False)
    ds_x, _ = ds[0]
    proc_x = preprocess_features_for_model(tr["X"][0], mean, std)[0]
    assert np.allclose(ds_x.numpy(), proc_x, atol=1e-6), "Mismatch with KPDataset preprocessing"
    print("  PASSED: Standardization stats finite, masking exact, matches KPDataset.")


def test_c_feature_dimensions() -> None:
    print("[Test C] Feature dimensions...")
    df = pd.read_csv("split_date.csv", encoding="utf-8-sig")
    sample_path = df.iloc[0]["path"]
    frames, ar = read_video_frames(sample_path)
    assert len(frames) == T

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    try:
        raw = video_features(frames, ar, landmarker)
    finally:
        landmarker.close()

    assert raw.shape == (T, F), f"Expected {(T, F)}, got {raw.shape}"
    mean, std = load_standardization(DEFAULT_STATS_PATH)
    model_in = preprocess_features_for_model(raw, mean, std)
    assert model_in.shape == (1, T, F * 2), f"Expected {(1, T, F * 2)}, got {model_in.shape}"
    print(f"  PASSED: raw={raw.shape}, model_input={model_in.shape}.")


def test_d_video_inference() -> list[dict]:
    print("[Test D] Real video inference (1 train, 1 val, 1 test)...")
    df = pd.read_csv("split_date.csv", encoding="utf-8-sig")
    label_map = load_label_map(DEFAULT_LABELS_CSV)

    samples = [
        ("train", df[df["split"] == "train"].iloc[0]),
        ("val", df[df["split"] == "val"].iloc[0]),
        ("test", df[df["split"] == "test"].iloc[0]),
    ]

    records = []
    for split_name, row in samples:
        gt_idx = int(row["y"])
        gt_info = label_map[gt_idx]
        res = predict_video(row["path"])
        correct = res["class_index"] == gt_idx
        records.append({
            "split": split_name,
            "path": row["path"],
            "gt_idx": gt_idx,
            "gt_label": f"{gt_info['english_label']} ({gt_info['arabic_label']})",
            "pred_idx": res["class_index"],
            "pred_label": f"{res['english_label']} ({res['arabic_label']})",
            "confidence": res["confidence"],
            "correct": correct,
            "time_s": res["inference_time_seconds"],
        })
        print(
            f"  [{split_name:5s}] GT={gt_idx} ({gt_info['english_label']}) | "
            f"Pred={res['class_index']} ({res['english_label']}) | "
            f"Conf={res['confidence']*100:.2f}% | Correct={correct} | "
            f"Time={res['inference_time_seconds']:.2f}s"
        )
    print("  PASSED: All 3 real videos inferred without error.")
    return records


def test_e_reproducibility() -> None:
    print("[Test E] Reproducibility...")
    df = pd.read_csv("split_date.csv", encoding="utf-8-sig")
    sample_path = df[df["split"] == "val"].iloc[0]["path"]
    r1 = predict_video(sample_path)
    r2 = predict_video(sample_path)
    assert r1["class_index"] == r2["class_index"]
    assert np.isclose(r1["confidence"], r2["confidence"], atol=1e-7)
    assert np.allclose(r1["probabilities"], r2["probabilities"], atol=1e-7)
    print("  PASSED: Repeated inference produced identical predictions and probabilities.")


def _expect_exception(fn, exc_type, desc: str) -> None:
    try:
        fn()
    except exc_type as exc:
        print(f"  PASSED ({desc}): raised {exc_type.__name__}: {exc}")
        return
    raise AssertionError(f"Expected {exc_type.__name__} for {desc}, but no exception was raised.")


def test_f_error_handling() -> None:
    print("[Test F] Error handling...")
    df = pd.read_csv("split_date.csv", encoding="utf-8-sig")
    valid_video = df.iloc[0]["path"]

    _expect_exception(
        lambda: predict_video("nonexistent_video_12345.mp4"),
        FileNotFoundError,
        "missing video",
    )
    _expect_exception(
        lambda: predict_video(valid_video, checkpoint_path="missing_ckpt.pt"),
        FileNotFoundError,
        "missing checkpoint",
    )
    _expect_exception(
        lambda: predict_video(valid_video, stats_path="missing_stats.npz"),
        FileNotFoundError,
        "missing standardization file",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Corrupt / zero-frame file
        corrupt_path = tmp / "corrupt.mp4"
        corrupt_path.write_bytes(b"not a real mp4")
        _expect_exception(
            lambda: predict_video(corrupt_path),
            ValueError,
            "corrupt/zero-frame video",
        )

        # Valid blank MP4 with no hands -> failed MediaPipe hand detection
        blank_path = tmp / "blank.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(blank_path), fourcc, 30.0, (320, 180))
        blank_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for _ in range(16):
            writer.write(blank_frame)
        writer.release()

        _expect_exception(
            lambda: predict_video(blank_path),
            RuntimeError,
            "failed MediaPipe hand detection",
        )


def main() -> None:
    test_a_checkpoint_loading()
    test_b_standardization()
    test_c_feature_dimensions()
    test_d_video_inference()
    test_e_reproducibility()
    test_f_error_handling()
    print("\nALL INFERENCE TESTS (A-F) PASSED!")


if __name__ == "__main__":
    main()
