"""
SignBridge AI — Stage 9 Comprehensive Automated Test Suite

Clearly separates automated offline tests (model, sentence builder, Arabic TTS
WAV synthesis, Text-to-Sign video lookup, SQLite history, Person 3 adapters,
and Tkinter UI launch/navigation) from physical webcam/microphone hardware tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from app import SignBridgeApp
from signbridge.history import TranslationHistoryStore
from signbridge.person3_adapter import LocalAnalyticsAdapter, PlaceholderAssistantAdapter
from signbridge.recognizer import SignSentenceRecogniser
from signbridge.stt import AutoArabicSTT, MicrophoneRecorder
from signbridge.text_to_sign import TextToSignMapper, normalize_arabic
from signbridge.tts import ArabicTTS


def test_history_store() -> None:
    print("[1/7] Testing TranslationHistoryStore (SQLite + corruption recovery)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "hist.db"
        store = TranslationHistoryStore(db_path)
        row_id = store.add_entry("Sign -> Speech", "0 (ID 0001)", "صفر", 0.6545, True)
        assert row_id >= 1
        recent = store.get_recent(10)
        assert len(recent) == 1 and recent[0]["output_text"] == "صفر"
        stats = store.get_summary_stats()
        assert stats["total_translations"] == 1
        store.clear_history()
        assert len(store.get_recent(10)) == 0

        # Test corrupt file recovery
        db_path.write_bytes(b"corrupt sqlite bytes")
        recovered = TranslationHistoryStore(db_path)
        assert recovered.get_recent(10) == []
    print("  PASSED: History add, reload, clear, and corruption recovery.")


def test_text_to_sign() -> None:
    print("[2/7] Testing TextToSignMapper (Arabic normalization, phrase match, video lookup)...")
    mapper = TextToSignMapper()
    assert len(mapper.vocab) == 70

    # Phrase-first match: "عشرة ملايين" must match SignID 0031 (class 30), not "عشرة" (class 10)
    res_phrase = mapper.translate("عشرة ملايين")
    assert len(res_phrase["matched_signs"]) == 1
    assert res_phrase["matched_signs"][0]["class_index"] == 30
    assert res_phrase["unsupported_tokens"] == []

    # Mixed supported + unsupported + diacritics
    res_mix = mapper.translate("صِفْر وَاحِد كلمة_غير_موجودة")
    assert [m["arabic_label"] for m in res_mix["matched_signs"]] == ["صفر", "واحد"]
    assert res_mix["unsupported_tokens"] == ["كلمة_غير_موجودة"]
    assert len(res_mix["playlist"]) == 2

    # Verify ordered MP4 playback readability
    for item in res_mix["playlist"]:
        cap = cv2.VideoCapture(item["video_path"])
        ok, frame = cap.read()
        cap.release()
        assert ok and frame is not None, f"Could not read sign video: {item['video_path']}"
    print("  PASSED: 70-class lookup, phrase priority, unsupported detection, and MP4 readability.")


def test_arabic_tts() -> None:
    print("[3/7] Testing offline Arabic TTS (Microsoft Hoda ar-EG)...")
    tts = ArabicTTS()
    avail, info = tts.check_available()
    assert avail, f"Expected offline Arabic TTS voice, got: {info}"
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "out.wav"
        tts.synthesize_to_wav("صفر واحد اثنان", wav_path)
        data = wav_path.read_bytes()
        assert len(data) > 10000 and data[:4] == b"RIFF", "Invalid WAV output"
    print(f"  PASSED: Offline Arabic TTS synthesized {len(data)} bytes via {info}.")


def test_stt_and_mic_status() -> None:
    print("[4/7] Testing MicrophoneRecorder & AutoArabicSTT status...")
    mic = MicrophoneRecorder()
    devices = mic.list_input_devices()
    stt = AutoArabicSTT()
    msg = stt.status_message()
    assert isinstance(msg, str) and len(msg) > 0
    print(f"  PASSED: Input devices found={len(devices)}; STT available={stt.is_available()} ({msg})")


def test_recognizer_and_duplicate_suppression() -> None:
    print("[5/7] Testing SignSentenceRecogniser (real video frames + duplicate hold suppression)...")
    rec = SignSentenceRecogniser(confidence_threshold=0.40, smooth_window=5, hold_frames=3, cooldown_seconds=2.0)
    try:
        df = pd.read_csv("split_date.csv", encoding="utf-8-sig")
        vid = df[df["split"] == "test"].iloc[0]["path"].replace("\\", "/")
        cap = cv2.VideoCapture(vid)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()

        idx = np.linspace(0, len(frames) - 1, 16).round().astype(int)
        # Feed 16 frames + 5 extra repeats of last frame (hold_frames=3)
        feed = [frames[i] for i in idx] + [frames[idx[-1]]] * 5
        committed_list = []
        for f in feed:
            state, committed = rec.process_frame(f)
            if committed is not None:
                committed_list.append(committed)

        assert len(committed_list) == 1, f"Expected exactly 1 committed sign (duplicate hold suppressed), got {len(committed_list)}"
        assert committed_list[0]["arabic_label"] == "صفر"
        assert rec.get_arabic_sentence() == "صفر"

        # Pause test
        rec.set_paused(True)
        p_state, p_comm = rec.process_frame(frames[0])
        assert p_state["status"] == "Paused" and p_comm is None
        rec.set_paused(False)

        # No-hand blank frame test
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(16):
            b_state, _ = rec.process_frame(blank)
        assert b_state["uncertain"] is True and b_state["status"] == "No hand detected"

        rec.clear_sentence()
        assert rec.get_arabic_sentence() == ""
    finally:
        rec.close()
    print("  PASSED: 16-frame recognition, duplicate-hold suppression, pause, and no-hand safety.")


def test_person3_adapters() -> None:
    print("[6/7] Testing Person 3 integration adapters...")
    with tempfile.TemporaryDirectory() as tmpdir:
        hist = TranslationHistoryStore(Path(tmpdir) / "h.db")
        asst = PlaceholderAssistantAdapter()
        assert asst.is_connected() is False
        reply = asst.ask("Hello", {"test": 1})
        assert "Placeholder" in reply

        ana = LocalAnalyticsAdapter(hist)
        dash = ana.get_dashboard_data()
        assert dash["person3_connected"] is False
        assert len(dash["model_results"]) == 3
    print("  PASSED: Person 3 adapters return structured data without fabrication.")


def test_tkinter_app_launch_and_navigation() -> None:
    print("[7/7] Testing Tkinter SignBridgeApp launch, 6-tab navigation, and clean exit...")
    root = tk.Tk()
    root.withdraw()
    app = SignBridgeApp(root)
    assert app.notebook.index("end") == 6
    for i in range(6):
        app.notebook.select(i)
        root.update_idletasks()
    app.t2s_entry.delete(0, tk.END)
    app.t2s_entry.insert(0, "صفر")
    app.run_text_to_sign()
    assert "صفر" in app.t2s_match_var.get()
    app._stop_player()
    app.on_close()
    print("  PASSED: 6-tab Tkinter UI launched, navigated, ran Text-to-Sign, and closed cleanly.")


def main() -> None:
    test_history_store()
    test_text_to_sign()
    test_arabic_tts()
    test_stt_and_mic_status()
    test_recognizer_and_duplicate_suppression()
    test_person3_adapters()
    test_tkinter_app_launch_and_navigation()
    print("\nALL 7 APPLICATION SUITE TESTS PASSED!")


if __name__ == "__main__":
    main()
