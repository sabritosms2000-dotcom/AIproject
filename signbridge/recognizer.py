"""
SignBridge AI — Real-Time Sign Recognizer & Sentence Builder (Stages 2 & 3).

Wraps `WebcamSignRecognizer` from `webcam_app.py` without altering the V2 BiLSTM
feature pipeline, and adds duplicate-hold suppression and sentence assembly.
"""

from __future__ import annotations

import time
from typing import Any

from signbridge.config import (
    CHECKPOINT_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_HOLD_FRAMES_TO_COMMIT,
    DEFAULT_SMOOTH_WINDOW,
    LABELS_CSV,
    STANDARDIZATION_PATH,
)
from webcam_app import WebcamSignRecognizer, draw_overlay


class SignSentenceRecogniser:
    """
    Combines `WebcamSignRecognizer` with stable sign commitment:
    - Requires `hold_frames` consecutive confident predictions of the same sign.
    - Prevents repeatedly adding the same sign while the user holds the gesture.
    - Supports pause/resume and sentence clearing.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        hold_frames: int = DEFAULT_HOLD_FRAMES_TO_COMMIT,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self.core = WebcamSignRecognizer(
            checkpoint_path=CHECKPOINT_PATH,
            stats_path=STANDARDIZATION_PATH,
            labels_csv=LABELS_CSV,
            device="cpu",
            confidence_threshold=confidence_threshold,
            smooth_window=smooth_window,
        )
        self.hold_frames = max(1, int(hold_frames))
        self.cooldown_seconds = float(cooldown_seconds)
        self.paused = False

        self.sentence_tokens: list[dict[str, Any]] = []
        self._candidate_class: int | None = None
        self._candidate_streak: int = 0
        self._last_committed_class: int | None = None
        self._last_commit_time: float = 0.0

    @property
    def confidence_threshold(self) -> float:
        return self.core.confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self.core.confidence_threshold = float(value)

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def clear_sentence(self) -> None:
        self.sentence_tokens.clear()
        self._candidate_class = None
        self._candidate_streak = 0
        self._last_committed_class = None

    def reset_buffer(self) -> None:
        self.core.reset()
        self._candidate_class = None
        self._candidate_streak = 0

    def get_arabic_sentence(self) -> str:
        return " ".join(tok["arabic_label"] for tok in self.sentence_tokens)

    def close(self) -> None:
        self.core.close()

    def process_frame(self, bgr_frame: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        Process one BGR frame. Returns `(state, newly_committed_token_or_None)`.
        """
        if self.paused:
            return {
                "ready": len(self.core.frame_buffer) >= 16,
                "buffer_count": len(self.core.frame_buffer),
                "buffer_target": 16,
                "hand_detected": False,
                "status": "Paused",
                "display_label": "Paused",
                "class_index": None,
                "confidence": 0.0,
                "uncertain": True,
                "probabilities": None,
                "latency_ms": 0.0,
            }, None

        state = self.core.process_frame(bgr_frame)
        committed: dict[str, Any] | None = None

        if not state["ready"] or state["uncertain"] or state["class_index"] is None:
            # Break hold streak when hands disappear or confidence drops
            if not state["hand_detected"]:
                self._candidate_class = None
                self._candidate_streak = 0
                self._last_committed_class = None
            return state, None

        cls_idx = int(state["class_index"])
        if cls_idx == self._candidate_class:
            self._candidate_streak += 1
        else:
            self._candidate_class = cls_idx
            self._candidate_streak = 1

        now = time.monotonic()
        if self._candidate_streak >= self.hold_frames:
            is_duplicate_hold = (
                cls_idx == self._last_committed_class
                and (now - self._last_commit_time) < self.cooldown_seconds
            )
            if not is_duplicate_hold and cls_idx != self._last_committed_class:
                token = {
                    "class_index": cls_idx,
                    "sign_id": state["sign_id"],
                    "arabic_label": state["arabic_label"],
                    "english_label": state["english_label"],
                    "confidence": float(state["confidence"]),
                }
                self.sentence_tokens.append(token)
                self._last_committed_class = cls_idx
                self._last_commit_time = now
                committed = token

        return state, committed

    def render_frame(self, bgr_frame: Any, state: dict[str, Any], fps: float) -> Any:
        return draw_overlay(bgr_frame, state, fps, landmarker_result=self.core.last_result)
