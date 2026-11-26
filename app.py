"""
SignBridge AI — Main Desktop Application (Person 2 Deliverable)

6-Tab Tkinter Application:
  1. Sign to Speech (Live Webcam + MediaPipe + BiLSTM V2 + Arabic TTS)
  2. Speech to Sign (Microphone Recorder + STT Adapter + Manual Fallback)
  3. Text to Sign (Arabic Normalization + 70-Sign Lookup + Ordered MP4 Player)
  4. Translation History (Local SQLite Persistence)
  5. AI Assistant (Person 3 Integration Adapter)
  6. Analytics Dashboard (Verified Model Metrics + History Stats + Person 3 Adapter)
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from signbridge.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODEL_ACCURACY_LIMITATION,
    VOCABULARY_LIMITATION,
)
from signbridge.history import TranslationHistoryStore
from signbridge.person3_adapter import (
    LocalAnalyticsAdapter,
    PlaceholderAssistantAdapter,
)
from signbridge.recognizer import SignSentenceRecogniser
from signbridge.stt import AutoArabicSTT, MicrophoneRecorder
from signbridge.text_to_sign import TextToSignMapper
from signbridge.tts import ArabicTTS


class SignBridgeApp:
    """Main Tkinter desktop application for SignBridge AI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SignBridge AI — Arabic Sign Language Translator (BiLSTM V2)")
        self.root.geometry("1120x740")
        self.root.minsize(960, 640)

        # Core services
        self.history = TranslationHistoryStore()
        self.tts = ArabicTTS()
        self.stt_engine = AutoArabicSTT()
        self.mic = MicrophoneRecorder()
        self.t2s = TextToSignMapper()
        self.assistant = PlaceholderAssistantAdapter()
        self.analytics = LocalAnalyticsAdapter(self.history)

        # Lazy-loaded recognizer (loaded on first camera start or explicit load)
        self.recognizer: SignSentenceRecogniser | None = None
        self._cap: cv2.VideoCapture | None = None
        self._cam_running = False
        self._fps_ema = 0.0
        self._prev_frame_t = time.perf_counter()
        self._cam_photo: ImageTk.PhotoImage | None = None

        # Text-to-Sign playback state
        self._playlist: list[dict[str, Any]] = []
        self._playlist_idx = 0
        self._player_cap: cv2.VideoCapture | None = None
        self._player_playing = False
        self._player_photo: ImageTk.PhotoImage | None = None

        self._configure_styles()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("ArabicBig.TLabel", font=("Segoe UI", 20, "bold"), foreground="#0b5394")
        style.configure("Warn.TLabel", font=("Segoe UI", 9), foreground="#994400")

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(fill=tk.X)
        ttk.Label(top, text="SignBridge AI", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            top,
            text="  |  70-Class KArSL-502 BiLSTM V2 (CPU)  —  " + VOCABULARY_LIMITATION,
            style="Sub.TLabel",
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.tab_s2s = ttk.Frame(self.notebook, padding=10)
        self.tab_stt = ttk.Frame(self.notebook, padding=10)
        self.tab_t2s = ttk.Frame(self.notebook, padding=10)
        self.tab_hist = ttk.Frame(self.notebook, padding=10)
        self.tab_ai = ttk.Frame(self.notebook, padding=10)
        self.tab_dash = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_s2s, text="1. Sign to Speech")
        self.notebook.add(self.tab_stt, text="2. Speech to Sign")
        self.notebook.add(self.tab_t2s, text="3. Text to Sign")
        self.notebook.add(self.tab_hist, text="4. Translation History")
        self.notebook.add(self.tab_ai, text="5. AI Assistant (P3)")
        self.notebook.add(self.tab_dash, text="6. Analytics Dashboard")

        self._build_sign_to_speech_tab()
        self._build_speech_to_sign_tab()
        self._build_text_to_sign_tab()
        self._build_history_tab()
        self._build_assistant_tab()
        self._build_analytics_tab()

    # ------------------------------------------------------------------
    # Tab 1: Sign to Speech
    # ------------------------------------------------------------------
    def _build_sign_to_speech_tab(self) -> None:
        left = ttk.Frame(self.tab_s2s)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cam_label = ttk.Label(left, text="Camera stopped. Click 'Start Camera' to begin.", anchor=tk.CENTER)
        self.cam_label.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        cam_btns = ttk.Frame(left)
        cam_btns.pack(fill=tk.X)
        self.btn_start_cam = ttk.Button(cam_btns, text="Start Camera", command=self.start_camera)
        self.btn_start_cam.pack(side=tk.LEFT, padx=4)
        self.btn_stop_cam = ttk.Button(cam_btns, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop_cam.pack(side=tk.LEFT, padx=4)
        self.btn_pause_rec = ttk.Button(cam_btns, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause_rec.pack(side=tk.LEFT, padx=4)
        ttk.Button(cam_btns, text="Reset Buffer", command=self.reset_buffer).pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(self.tab_s2s, width=360, padding=(12, 0, 0, 0))
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        ttk.Label(right, text="Live Recognition", style="Header.TLabel").pack(anchor=tk.W)
        self.s2s_status_var = tk.StringVar(value="Idle")
        self.s2s_pred_var = tk.StringVar(value="—")
        self.s2s_conf_var = tk.StringVar(value="Confidence: —")
        self.s2s_fps_var = tk.StringVar(value="FPS: 0.0")

        ttk.Label(right, textvariable=self.s2s_pred_var, style="ArabicBig.TLabel", anchor=tk.E).pack(fill=tk.X, pady=(8, 2))
        ttk.Label(right, textvariable=self.s2s_conf_var).pack(anchor=tk.W)
        ttk.Label(right, textvariable=self.s2s_status_var, style="Sub.TLabel").pack(anchor=tk.W)
        ttk.Label(right, textvariable=self.s2s_fps_var, style="Sub.TLabel").pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(right, text="Confidence Threshold (uncalibrated softmax):").pack(anchor=tk.W)
        self.thresh_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        thresh_row = ttk.Frame(right)
        thresh_row.pack(fill=tk.X, pady=(2, 10))
        ttk.Scale(
            thresh_row, from_=0.10, to=0.90, variable=self.thresh_var, command=self._on_threshold_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.thresh_lbl = ttk.Label(thresh_row, text=f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}", width=5)
        self.thresh_lbl.pack(side=tk.RIGHT)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Arabic Sentence (RTL):", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.sentence_text = tk.Text(right, height=4, font=("Segoe UI", 14), wrap=tk.WORD)
        self.sentence_text.tag_configure("rtl", justify="right")
        self.sentence_text.pack(fill=tk.X, pady=6)

        tts_btns = ttk.Frame(right)
        tts_btns.pack(fill=tk.X)
        ttk.Button(tts_btns, text="Speak Arabic", command=self.speak_sentence).pack(side=tk.LEFT, padx=2)
        ttk.Button(tts_btns, text="Repeat", command=self.repeat_speech).pack(side=tk.LEFT, padx=2)
        ttk.Button(tts_btns, text="Clear", command=self.clear_sentence).pack(side=tk.LEFT, padx=2)

        tts_ok, tts_info = self.tts.check_available()
        tts_msg = f"TTS Ready: {tts_info}" if tts_ok else f"TTS Unavailable: {tts_info}"
        ttk.Label(right, text=tts_msg, style="Sub.TLabel", wraplength=340).pack(anchor=tk.W, pady=(8, 4))
        ttk.Label(right, text=MODEL_ACCURACY_LIMITATION, style="Warn.TLabel", wraplength=340).pack(anchor=tk.W, pady=(6, 0))

    def _on_threshold_change(self, _val: str) -> None:
        v = round(self.thresh_var.get(), 2)
        self.thresh_lbl.config(text=f"{v:.2f}")
        if self.recognizer is not None:
            self.recognizer.confidence_threshold = v

    def _ensure_recognizer(self) -> SignSentenceRecogniser:
        if self.recognizer is None:
            self.recognizer = SignSentenceRecogniser(confidence_threshold=self.thresh_var.get())
        return self.recognizer

    def start_camera(self) -> None:
        if self._cam_running:
            return
        self.s2s_status_var.set("Loading BiLSTM V2 & MediaPipe...")
        self.root.update_idletasks()
        try:
            self._ensure_recognizer()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open default webcam (camera_id=0).")
        except Exception as exc:
            self.s2s_status_var.set("Camera error")
            messagebox.showerror("Camera Error", str(exc))
            return

        self._cap = cap
        self._cam_running = True
        self._prev_frame_t = time.perf_counter()
        self.btn_start_cam.config(state=tk.DISABLED)
        self.btn_stop_cam.config(state=tk.NORMAL)
        self.btn_pause_rec.config(state=tk.NORMAL)
        self._poll_camera()

    def stop_camera(self) -> None:
        self._cam_running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.btn_start_cam.config(state=tk.NORMAL)
        self.btn_stop_cam.config(state=tk.DISABLED)
        self.btn_pause_rec.config(state=tk.DISABLED, text="Pause")
        self.s2s_status_var.set("Stopped")

    def toggle_pause(self) -> None:
        if self.recognizer is None:
            return
        new_state = not self.recognizer.paused
        self.recognizer.set_paused(new_state)
        self.btn_pause_rec.config(text="Resume" if new_state else "Pause")

    def reset_buffer(self) -> None:
        if self.recognizer is not None:
            self.recognizer.reset_buffer()
        self.s2s_status_var.set("Buffer reset")

    def _poll_camera(self) -> None:
        if not self._cam_running or self._cap is None or self.recognizer is None:
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.stop_camera()
            messagebox.showwarning("Webcam", "Lost connection to webcam.")
            return

        now = time.perf_counter()
        dt = max(now - self._prev_frame_t, 1e-6)
        self._prev_frame_t = now
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema == 0.0 else (0.85 * self._fps_ema + 0.15 * inst_fps)

        state, committed = self.recognizer.process_frame(frame)
        vis_bgr = self.recognizer.render_frame(frame, state, self._fps_ema)

        # Update UI labels
        self.s2s_status_var.set(state["status"])
        self.s2s_fps_var.set(f"FPS: {self._fps_ema:.1f} ({state['latency_ms']:.0f} ms)")
        if state["ready"] and state["class_index"] is not None:
            ar = state.get("arabic_label", "")
            en = state.get("english_label", "")
            prefix = "Uncertain: " if state["uncertain"] else ""
            self.s2s_pred_var.set(f"{prefix}{ar} ({en})")
            self.s2s_conf_var.set(f"Confidence: {state['confidence'] * 100:.1f}%")
        elif state["ready"]:
            self.s2s_pred_var.set(state["display_label"])
            self.s2s_conf_var.set("Confidence: 0.0%")
        else:
            self.s2s_pred_var.set(f"Collecting ({state['buffer_count']}/16)")
            self.s2s_conf_var.set("Confidence: --")

        if committed is not None:
            self._update_sentence_box()
            self.history.add_entry(
                direction="Sign -> Speech",
                input_text=f"{committed['english_label']} (ID {committed['sign_id']})",
                output_text=committed["arabic_label"],
                confidence=committed["confidence"],
                success=True,
            )

        # Render BGR -> Tkinter ImageTk
        rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((680, 480))
        self._cam_photo = ImageTk.PhotoImage(img)
        self.cam_label.config(image=self._cam_photo, text="")

        self.root.after(15, self._poll_camera)

    def _update_sentence_box(self) -> None:
        if self.recognizer is None:
            return
        text = self.recognizer.get_arabic_sentence()
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert("1.0", text, "rtl")

    def speak_sentence(self) -> None:
        text = self.sentence_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Speak Arabic", "No Arabic text in sentence box.")
            return
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text: str) -> None:
        try:
            self.tts.speak(text, blocking=False)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("TTS Error", str(exc)))

    def repeat_speech(self) -> None:
        if not self.tts.repeat_last(blocking=False):
            self.speak_sentence()

    def clear_sentence(self) -> None:
        if self.recognizer is not None:
            self.recognizer.clear_sentence()
        self.sentence_text.delete("1.0", tk.END)

    # ------------------------------------------------------------------
    # Tab 2: Speech to Sign
    # ------------------------------------------------------------------
    def _build_speech_to_sign_tab(self) -> None:
        ttk.Label(self.tab_stt, text="Arabic Speech to Sign", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            self.tab_stt,
            text=self.stt_engine.status_message(),
            style="Warn.TLabel",
            wraplength=900,
        ).pack(anchor=tk.W, pady=(4, 10))

        mic_row = ttk.Frame(self.tab_stt)
        mic_row.pack(fill=tk.X, pady=4)
        self.btn_rec_start = ttk.Button(mic_row, text="Start Microphone Recording", command=self.start_mic)
        self.btn_rec_start.pack(side=tk.LEFT, padx=4)
        self.btn_rec_stop = ttk.Button(mic_row, text="Stop & Transcribe", command=self.stop_mic, state=tk.DISABLED)
        self.btn_rec_stop.pack(side=tk.LEFT, padx=4)
        self.stt_status_var = tk.StringVar(value="Microphone idle")
        ttk.Label(mic_row, textvariable=self.stt_status_var).pack(side=tk.LEFT, padx=12)

        ttk.Label(self.tab_stt, text="Recognized / Manual Arabic Text (RTL):", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W, pady=(12, 4)
        )
        self.stt_text = tk.Text(self.tab_stt, height=3, font=("Segoe UI", 14))
        self.stt_text.tag_configure("rtl", justify="right")
        self.stt_text.insert("1.0", "واحد اثنان ثلاثة", "rtl")
        self.stt_text.pack(fill=tk.X, pady=4)

        ttk.Button(
            self.tab_stt,
            text="Translate Arabic Text to Sign Videos ->",
            command=self.send_stt_to_t2s,
        ).pack(anchor=tk.W, pady=8)

    def start_mic(self) -> None:
        try:
            self.mic.start()
            self.btn_rec_start.config(state=tk.DISABLED)
            self.btn_rec_stop.config(state=tk.NORMAL)
            self.stt_status_var.set("Recording 16 kHz mono audio...")
        except Exception as exc:
            messagebox.showerror("Microphone Error", str(exc))

    def stop_mic(self) -> None:
        try:
            wav_path = self.mic.stop_and_save()
            self.btn_rec_start.config(state=tk.NORMAL)
            self.btn_rec_stop.config(state=tk.DISABLED)
            self.stt_status_var.set(f"Saved audio: {wav_path.name}")
            if self.stt_engine.is_available():
                text = self.stt_engine.transcribe_wav(wav_path)
                self.stt_text.delete("1.0", tk.END)
                self.stt_text.insert("1.0", text, "rtl")
                self.history.add_entry("Speech -> Text", wav_path.name, text, None, True)
            else:
                messagebox.showinfo(
                    "STT Fallback",
                    "Audio recorded to WAV, but no Arabic STT engine is installed in .venv.\n"
                    "Use the manual Arabic text box below to translate to Sign.",
                )
        except Exception as exc:
            self.btn_rec_start.config(state=tk.NORMAL)
            self.btn_rec_stop.config(state=tk.DISABLED)
            messagebox.showerror("Recording Error", str(exc))

    def send_stt_to_t2s(self) -> None:
        text = self.stt_text.get("1.0", tk.END).strip()
        self.t2s_entry.delete(0, tk.END)
        self.t2s_entry.insert(0, text)
        self.notebook.select(self.tab_t2s)
        self.run_text_to_sign()

    # ------------------------------------------------------------------
    # Tab 3: Text to Sign
    # ------------------------------------------------------------------
    def _build_text_to_sign_tab(self) -> None:
        ttk.Label(self.tab_t2s, text="Arabic Text to Sign Language Playback", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(self.tab_t2s, text=VOCABULARY_LIMITATION, style="Warn.TLabel").pack(anchor=tk.W, pady=(2, 8))

        inp_row = ttk.Frame(self.tab_t2s)
        inp_row.pack(fill=tk.X, pady=4)
        self.t2s_entry = ttk.Entry(inp_row, font=("Segoe UI", 14), justify=tk.RIGHT)
        self.t2s_entry.insert(0, "صفر واحد عشرة ملايين")
        self.t2s_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(inp_row, text="Translate & Play", command=self.run_text_to_sign).pack(side=tk.RIGHT)

        samples = ttk.Frame(self.tab_t2s)
        samples.pack(fill=tk.X, pady=4)
        ttk.Label(samples, text="Quick samples:").pack(side=tk.LEFT, padx=(0, 6))
        for sample in ["صفر واحد اثنان", "عشرة ملايين", "ا ب ت ث", "مرحبا واحد"]:
            ttk.Button(
                samples,
                text=sample,
                command=lambda s=sample: (self.t2s_entry.delete(0, tk.END), self.t2s_entry.insert(0, s), self.run_text_to_sign()),
            ).pack(side=tk.LEFT, padx=3)

        self.t2s_match_var = tk.StringVar(value="Matched: —")
        self.t2s_unsup_var = tk.StringVar(value="Unsupported: —")
        ttk.Label(self.tab_t2s, textvariable=self.t2s_match_var, font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(6, 2))
        ttk.Label(self.tab_t2s, textvariable=self.t2s_unsup_var, style="Warn.TLabel").pack(anchor=tk.W, pady=(0, 8))

        self.player_label = ttk.Label(self.tab_t2s, text="No sign video loaded.", anchor=tk.CENTER)
        self.player_label.pack(fill=tk.BOTH, expand=True)

        p_ctrl = ttk.Frame(self.tab_t2s)
        p_ctrl.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(p_ctrl, text="Play / Pause", command=self.toggle_player).pack(side=tk.LEFT, padx=4)
        ttk.Button(p_ctrl, text="Replay All", command=self.replay_playlist).pack(side=tk.LEFT, padx=4)
        ttk.Button(p_ctrl, text="Prev", command=self.prev_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(p_ctrl, text="Next", command=self.next_video).pack(side=tk.LEFT, padx=4)
        self.player_info_var = tk.StringVar(value="Playlist: 0/0")
        ttk.Label(p_ctrl, textvariable=self.player_info_var).pack(side=tk.LEFT, padx=12)

    def run_text_to_sign(self) -> None:
        raw = self.t2s_entry.get().strip()
        res = self.t2s.translate(raw)
        matched = [f"{m['arabic_label']} ({m['english_label']})" for m in res["matched_signs"]]
        self.t2s_match_var.set("Matched: " + (", ".join(matched) if matched else "None"))
        self.t2s_unsup_var.set(
            "Unsupported words: " + (", ".join(res["unsupported_tokens"]) if res["unsupported_tokens"] else "None")
        )

        self.history.add_entry(
            direction="Text -> Sign",
            input_text=raw,
            output_text=", ".join(matched) if matched else "Unsupported",
            confidence=None,
            success=len(res["playlist"]) > 0,
        )

        self._playlist = res["playlist"]
        self._playlist_idx = 0
        if self._playlist:
            self._open_playlist_item(0)
            self._player_playing = True
            self._poll_player()
        else:
            self._stop_player()
            self.player_label.config(image="", text="No matching sign videos found for input.")
            self.player_info_var.set("Playlist: 0/0")

    def _open_playlist_item(self, idx: int) -> None:
        if self._player_cap is not None:
            self._player_cap.release()
            self._player_cap = None
        if not self._playlist:
            return
        self._playlist_idx = idx % len(self._playlist)
        item = self._playlist[self._playlist_idx]
        self._player_cap = cv2.VideoCapture(item["video_path"])
        self.player_info_var.set(
            f"Playing {self._playlist_idx + 1}/{len(self._playlist)}: "
            f"{item['arabic_label']} ({item['english_label']}, ID {item['sign_id']})"
        )

    def _stop_player(self) -> None:
        self._player_playing = False
        if self._player_cap is not None:
            self._player_cap.release()
            self._player_cap = None

    def toggle_player(self) -> None:
        if not self._playlist:
            return
        self._player_playing = not self._player_playing
        if self._player_playing:
            self._poll_player()

    def replay_playlist(self) -> None:
        if not self._playlist:
            return
        self._open_playlist_item(0)
        self._player_playing = True
        self._poll_player()

    def prev_video(self) -> None:
        if not self._playlist:
            return
        self._open_playlist_item(self._playlist_idx - 1)

    def next_video(self) -> None:
        if not self._playlist:
            return
        self._open_playlist_item(self._playlist_idx + 1)

    def _poll_player(self) -> None:
        if not self._player_playing or self._player_cap is None:
            return
        ok, frame = self._player_cap.read()
        if not ok or frame is None:
            # Advance to next video in sequence
            if self._playlist_idx + 1 < len(self._playlist):
                self._open_playlist_item(self._playlist_idx + 1)
                self.root.after(40, self._poll_player)
            else:
                self._player_playing = False
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((640, 360))
        self._player_photo = ImageTk.PhotoImage(img)
        self.player_label.config(image=self._player_photo, text="")
        self.root.after(35, self._poll_player)

    # ------------------------------------------------------------------
    # Tab 4: Translation History
    # ------------------------------------------------------------------
    def _build_history_tab(self) -> None:
        ctrl = ttk.Frame(self.tab_hist)
        ctrl.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ctrl, text="Translation History", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Clear History", command=self.clear_history).pack(side=tk.RIGHT, padx=4)
        ttk.Button(ctrl, text="Refresh", command=self.refresh_history).pack(side=tk.RIGHT, padx=4)

        cols = ("id", "timestamp", "direction", "input", "output", "confidence", "success")
        self.hist_tree = ttk.Treeview(self.tab_hist, columns=cols, show="headings")
        for c, w in [("id", 50), ("timestamp", 150), ("direction", 130), ("input", 220), ("output", 240), ("confidence", 90), ("success", 70)]:
            self.hist_tree.heading(c, text=c.title())
            self.hist_tree.column(c, width=w)
        self.hist_tree.pack(fill=tk.BOTH, expand=True)
        self.refresh_history()

    def refresh_history(self) -> None:
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)
        for row in self.history.get_recent(limit=200):
            conf_str = f"{row['confidence'] * 100:.1f}%" if row["confidence"] is not None else "—"
            self.hist_tree.insert(
                "",
                tk.END,
                values=(
                    row["id"],
                    row["timestamp"],
                    row["direction"],
                    row["input_text"],
                    row["output_text"],
                    conf_str,
                    "OK" if row["success"] else "Failed",
                ),
            )

    def clear_history(self) -> None:
        self.history.clear_history()
        self.refresh_history()

    # ------------------------------------------------------------------
    # Tab 5: AI Assistant (Person 3)
    # ------------------------------------------------------------------
    def _build_assistant_tab(self) -> None:
        ttk.Label(self.tab_ai, text="AI Assistant (Person 3 Integration Point)", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(self.tab_ai, text=self.assistant.status_message(), style="Warn.TLabel").pack(anchor=tk.W, pady=(4, 10))

        self.ai_log = tk.Text(self.tab_ai, height=16, font=("Segoe UI", 10), state=tk.DISABLED)
        self.ai_log.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        row = ttk.Frame(self.tab_ai)
        row.pack(fill=tk.X)
        self.ai_entry = ttk.Entry(row, font=("Segoe UI", 11))
        self.ai_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row, text="Send Query", command=self.send_ai_query).pack(side=tk.RIGHT)

    def send_ai_query(self) -> None:
        q = self.ai_entry.get().strip()
        if not q:
            return
        ctx = {"recent_history_count": len(self.history.get_recent(10))}
        reply = self.assistant.ask(q, context=ctx)
        self.ai_log.config(state=tk.NORMAL)
        self.ai_log.insert(tk.END, f"User: {q}\n{reply}\n\n")
        self.ai_log.config(state=tk.DISABLED)
        self.ai_entry.delete(0, tk.END)

    # ------------------------------------------------------------------
    # Tab 6: Analytics Dashboard
    # ------------------------------------------------------------------
    def _build_analytics_tab(self) -> None:
        ctrl = ttk.Frame(self.tab_dash)
        ctrl.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ctrl, text="Analytics Dashboard", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Refresh Metrics", command=self.refresh_analytics).pack(side=tk.RIGHT)

        self.dash_text = tk.Text(self.tab_dash, font=("Consolas", 10), state=tk.DISABLED)
        self.dash_text.pack(fill=tk.BOTH, expand=True)
        self.refresh_analytics()

    def refresh_analytics(self) -> None:
        data = self.analytics.get_dashboard_data()
        lines = [
            f"Person 3 Custom Analytics Connected: {data['person3_connected']}",
            "",
            "=== Verified BiLSTM V2 Model Results (lstm_v2_results.csv) ===",
        ]
        for r in data["model_results"]:
            lines.append(
                f"  Seed {int(r['seed']):4d} | Best Val Acc: {r['best_val_accuracy']*100:6.2f}% | "
                f"Cross-Day Test Acc: {r['test_accuracy']*100:6.2f}%"
            )
        hs = data["history_stats"]
        lines.extend([
            "",
            "=== Local Translation History Stats ===",
            f"  Total Logged Translations : {hs['total_translations']}",
            f"  Successful Translations   : {hs['successful_translations']}",
            f"  By Direction              : {hs['by_direction']}",
            f"  Mean Logged Confidence    : {hs['mean_logged_confidence']}",
        ])
        self.dash_text.config(state=tk.NORMAL)
        self.dash_text.delete("1.0", tk.END)
        self.dash_text.insert("1.0", "\n".join(lines))
        self.dash_text.config(state=tk.DISABLED)

    def on_close(self) -> None:
        self.stop_camera()
        self._stop_player()
        if self.recognizer is not None:
            self.recognizer.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SignBridgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
