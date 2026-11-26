"""
SignBridge AI — Real-Time Webcam Sign Language Recognition (BiLSTM V2)

Reuses the exact V2 MediaPipe Holistic feature extraction, standardization,
masking, temporal velocity, and BiLSTM model from `mediapipe_lstm.py` and `infer.py`.
"""

from __future__ import annotations

import argparse
from collections import deque
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch

from infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LABELS_CSV,
    DEFAULT_STATS_PATH,
    _create_landmarker,
    load_label_map,
    load_model,
    load_standardization,
    preprocess_features_for_model,
)
from mediapipe_lstm import FRAME_WIDTH, MODEL_PATH, T, frame_features


class WebcamSignRecognizer:
    """Stateful CPU recognizer maintaining a 16-frame rolling buffer and probability smoother."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        stats_path: str | Path = DEFAULT_STATS_PATH,
        labels_csv: str | Path = DEFAULT_LABELS_CSV,
        device: str = "cpu",
        confidence_threshold: float = 0.40,
        smooth_window: int = 5,
    ) -> None:
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.smooth_window = max(1, int(smooth_window))

        self.mean, self.std = load_standardization(stats_path)
        self.label_map = load_label_map(labels_csv)
        self.model = load_model(checkpoint_path, device=device)
        self.landmarker = _create_landmarker(MODEL_PATH)

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=T)
        self.prob_history: deque[np.ndarray] = deque(maxlen=self.smooth_window)
        self.last_infer_ms: float = 0.0
        self.last_result: Any = None

    def reset(self) -> None:
        """Clear rolling 16-frame feature buffer and prediction smoothing history."""
        self.frame_buffer.clear()
        self.prob_history.clear()
        self.last_result = None

    def close(self) -> None:
        """Release MediaPipe HolisticLandmarker resources."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def extract_frame_features(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Resize and convert BGR frame identically to `mediapipe_lstm.read_video_frames`,
        then extract 82 V2 features via `mediapipe_lstm.frame_features`.
        Returns (features_82, any_hand_detected).
        """
        h, w = bgr_frame.shape[:2]
        aspect_ratio = w / float(h)

        if w > FRAME_WIDTH:
            new_h = int(h * FRAME_WIDTH / w)
            resized = cv2.resize(bgr_frame, (FRAME_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = bgr_frame

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect(mp_image)
        self.last_result = result
        feats = frame_features(result, aspect_ratio)
        hand_detected = bool(feats[-2] > 0.5 or feats[-1] > 0.5)
        return feats, hand_detected

    def process_frame(self, bgr_frame: np.ndarray) -> dict[str, Any]:
        """
        Process one BGR frame, update the 16-frame rolling buffer, and return live state.
        """
        t0 = time.perf_counter()
        feats, hand_detected = self.extract_frame_features(bgr_frame)
        self.frame_buffer.append(feats)

        buf_len = len(self.frame_buffer)
        if buf_len < T:
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "ready": False,
                "buffer_count": buf_len,
                "buffer_target": T,
                "hand_detected": hand_detected,
                "status": f"Collecting frames ({buf_len}/{T})",
                "display_label": "Warming up...",
                "class_index": None,
                "confidence": 0.0,
                "uncertain": True,
                "probabilities": None,
                "latency_ms": self.last_infer_ms,
            }

        raw_seq = np.stack(list(self.frame_buffer), axis=0).astype(np.float32)
        any_hand_in_seq = bool(raw_seq[:, -2:].sum() > 0.0)

        if not any_hand_in_seq:
            self.prob_history.clear()
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "ready": True,
                "buffer_count": T,
                "buffer_target": T,
                "hand_detected": False,
                "status": "No hand detected",
                "display_label": "Uncertain (No hand)",
                "class_index": None,
                "confidence": 0.0,
                "uncertain": True,
                "probabilities": None,
                "latency_ms": self.last_infer_ms,
            }

        model_input = preprocess_features_for_model(raw_seq, self.mean, self.std)
        tensor_x = torch.from_numpy(model_input).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        self.prob_history.append(probs)
        smoothed_probs = np.mean(list(self.prob_history), axis=0)

        pred_idx = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[pred_idx])
        info = self.label_map[pred_idx]
        uncertain = confidence < self.confidence_threshold

        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        if uncertain:
            display_label = f"Uncertain ({info['english_label']}?)"
            status = f"Low confidence (<{self.confidence_threshold * 100:.0f}%)"
        else:
            display_label = f"{info['english_label']} (ID {info['sign_id']})"
            status = "Recognizing"

        return {
            "ready": True,
            "buffer_count": T,
            "buffer_target": T,
            "hand_detected": hand_detected,
            "status": status,
            "display_label": display_label,
            "class_index": pred_idx,
            "sign_id": info["sign_id"],
            "english_label": info["english_label"],
            "arabic_label": info["arabic_label"],
            "confidence": confidence,
            "uncertain": uncertain,
            "probabilities": smoothed_probs,
            "latency_ms": self.last_infer_ms,
        }


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
)

FINGERTIP_IDS = {4, 8, 12, 16, 20}


def draw_hand_skeleton(
    frame: np.ndarray,
    landmarks: Any,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
) -> None:
    """Draw 21 hand landmarks and skeleton connections onto frame in-place."""
    if not landmarks:
        return

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, 2, cv2.LINE_AA)

    for idx, pt in enumerate(pts):
        radius = 5 if (idx == 0 or idx in FINGERTIP_IDS) else 4
        cv2.circle(frame, pt, radius, point_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius, (20, 20, 20), 1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    state: dict[str, Any],
    fps: float,
    landmarker_result: Any = None,
) -> np.ndarray:
    """Render hand skeletons, status, predicted sign, confidence, FPS, and controls."""
    out = frame.copy()
    h, w = out.shape[:2]

    if landmarker_result is not None:
        # Left hand: cyan/blue | Right hand: green
        draw_hand_skeleton(out, landmarker_result.left_hand_landmarks, (255, 190, 0), (255, 255, 0))
        draw_hand_skeleton(out, landmarker_result.right_hand_landmarks, (0, 210, 80), (100, 255, 140))

    # Semi-transparent top banner
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    if not state["ready"]:
        color = (0, 200, 255)
        pred_text = f"Sign: Collecting frames ({state['buffer_count']}/{state['buffer_target']})"
        conf_text = "Confidence: --"
    elif state["uncertain"]:
        color = (0, 165, 255)
        pred_text = f"Sign: {state['display_label']}"
        conf_text = (
            f"Confidence: {state['confidence'] * 100:.1f}%"
            if state["class_index"] is not None
            else "Confidence: 0.0%"
        )
    else:
        color = (80, 230, 120)
        pred_text = f"Sign: {state['display_label']} | Class {state['class_index']}"
        conf_text = f"Confidence: {state['confidence'] * 100:.1f}%"

    cv2.putText(out, pred_text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(out, conf_text, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"Status: {state['status']} | FPS: {fps:.1f} | {state['latency_ms']:.0f} ms",
        (16, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "Keys: [Q] Quit   [R] Reset Buffer",
        (16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def run_webcam(
    camera_id: int = 0,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    stats_path: str | Path = DEFAULT_STATS_PATH,
    labels_csv: str | Path = DEFAULT_LABELS_CSV,
    confidence_threshold: float = 0.40,
    smooth_window: int = 5,
    frame_skip: int = 1,
    device: str = "cpu",
) -> None:
    """Open default webcam, run live sign recognition, and release all resources on exit."""
    print("Initializing SignBridge AI (BiLSTM V2, CPU)...")
    recognizer = WebcamSignRecognizer(
        checkpoint_path=checkpoint_path,
        stats_path=stats_path,
        labels_csv=labels_csv,
        device=device,
        confidence_threshold=confidence_threshold,
        smooth_window=smooth_window,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        recognizer.close()
        raise RuntimeError(
            f"Could not open webcam (camera_id={camera_id}). "
            "Verify a camera is connected and not in use by another application."
        )

    window_name = "SignBridge AI - Live Sign Recognition (BiLSTM V2)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_state: dict[str, Any] = {
        "ready": False,
        "buffer_count": 0,
        "buffer_target": T,
        "hand_detected": False,
        "status": "Starting...",
        "display_label": "Warming up...",
        "class_index": None,
        "confidence": 0.0,
        "uncertain": True,
        "probabilities": None,
        "latency_ms": 0.0,
    }

    frame_idx = 0
    fps_ema = 0.0
    prev_t = time.perf_counter()

    print("Webcam started. Press 'Q' to quit, 'R' to reset buffer.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: Failed to read frame from webcam.")
                break

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

            if frame_idx % max(1, frame_skip) == 0:
                last_state = recognizer.process_frame(frame)
            frame_idx += 1

            display = draw_overlay(frame, last_state, fps_ema, landmarker_result=recognizer.last_result)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                recognizer.reset()
                last_state["ready"] = False
                last_state["buffer_count"] = 0
                last_state["status"] = "Buffer reset"

            # Exit cleanly if user closes the window via [X]
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        print("Webcam and MediaPipe resources released.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge AI — Live Webcam Recognition")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Checkpoint path")
    parser.add_argument("--stats", type=str, default=str(DEFAULT_STATS_PATH), help="Standardization NPZ path")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS_CSV), help="Label CSV path")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--smooth", type=int, default=5, help="Prediction smoothing window (default: 5)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args()

    run_webcam(
        camera_id=args.camera_id,
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        labels_csv=args.labels,
        confidence_threshold=args.threshold,
        smooth_window=args.smooth,
        frame_skip=args.frame_skip,
        device=args.device,
    )


if __name__ == "__main__":
    main()
