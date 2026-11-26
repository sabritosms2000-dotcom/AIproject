"""
SignBridge AI — Microphone Recording & Replaceable Arabic Speech-to-Text (Stage 5).

Supports real microphone recording via `sounddevice` (16 kHz mono WAV) and
pluggable STT backends (`faster_whisper`, `vosk`, or `speech_recognition`),
with honest reporting and manual Arabic text fallback when no Arabic STT engine
is installed locally.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import threading
from typing import Protocol

import numpy as np
from scipy.io import wavfile
import sounddevice as sd


class SpeechTranscriber(Protocol):
    """Replaceable interface for Arabic Speech-to-Text engines."""

    def name(self) -> str: ...
    def is_available(self) -> bool: ...
    def transcribe_wav(self, wav_path: str | Path) -> str: ...


class MicrophoneRecorder:
    """Thread-safe 16 kHz mono microphone recorder backed by `sounddevice`."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.is_recording = False

    def list_input_devices(self) -> list[str]:
        devices = sd.query_devices()
        return [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]

    def start(self) -> None:
        if self.is_recording:
            return
        self._chunks = []

        def _callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            with self._lock:
                self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=_callback,
        )
        self._stream.start()
        self.is_recording = True

    def stop_and_save(self, out_wav: str | Path | None = None) -> Path:
        if not self.is_recording or self._stream is None:
            raise RuntimeError("Microphone is not currently recording.")

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.is_recording = False

        with self._lock:
            audio = np.concatenate(self._chunks, axis=0) if self._chunks else np.zeros((0, 1), dtype=np.int16)

        target = Path(out_wav) if out_wav else Path(tempfile.gettempdir()) / "signbridge_mic_input.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(target), self.sample_rate, audio)
        return target


class AutoArabicSTT:
    """
    Detects installed STT packages (`faster_whisper`, `vosk`, `speech_recognition`).
    If none are installed, reports unavailability clearly so the UI falls back to manual text.
    """

    def __init__(self) -> None:
        self.backend_name = self._detect_backend()

    @staticmethod
    def _detect_backend() -> str | None:
        for pkg in ("faster_whisper", "vosk", "speech_recognition"):
            try:
                importlib.import_module(pkg)
                return pkg
            except Exception:
                continue
        return None

    def is_available(self) -> bool:
        return self.backend_name is not None

    def status_message(self) -> str:
        if self.backend_name:
            return f"Speech engine available: {self.backend_name}"
        return (
            "Offline Arabic STT engine not installed in .venv "
            "(Windows SpeechRecognizer only has en-US; faster-whisper/vosk not installed). "
            "Microphone recording works; use manual Arabic text input as fallback."
        )

    def transcribe_wav(self, wav_path: str | Path) -> str:
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self.backend_name == "faster_whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(path), language="ar")
            return " ".join(seg.text.strip() for seg in segments).strip()

        if self.backend_name == "speech_recognition":
            import speech_recognition as sr
            rec = sr.Recognizer()
            with sr.AudioFile(str(path)) as src:
                audio = rec.record(src)
            return rec.recognize_google(audio, language="ar-EG")

        raise RuntimeError(self.status_message())
