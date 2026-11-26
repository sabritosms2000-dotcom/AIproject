"""
SignBridge AI — Offline Arabic Text-to-Speech (Stage 4).

Uses Windows 11 built-in WinRT SpeechSynthesizer (`Microsoft Hoda - ar-EG`)
with `winsound` playback — 100% offline, zero paid APIs, zero extra packages.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import threading
import winsound


_WINRT_TTS_PS = r"""
param([string]$TextPath, [string]$OutWav)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType=WindowsRuntime]

$text = [System.IO.File]::ReadAllText($TextPath, [System.Text.Encoding]::UTF8)
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$arVoice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -like "ar*" } | Select-Object -First 1
if ($null -eq $arVoice) {
    Write-Error "No Arabic WinRT voice installed."
    exit 2
}
$synth.Voice = $arVoice

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]
Function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

$stream = Await ($synth.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$size = [uint32]$stream.Size
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
$null = Await ($reader.LoadAsync($size)) ([uint32])
$bytes = New-Object byte[] $size
$reader.ReadBytes($bytes)
[System.IO.File]::WriteAllBytes($OutWav, $bytes)
"""


class ArabicTTS:
    """Offline Arabic (ar-EG) speech synthesizer using Windows WinRT SpeechSynthesizer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_wav: Path | None = None

    def check_available(self) -> tuple[bool, str]:
        """Verify an Arabic WinRT voice is available on this Windows machine."""
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
                        "$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime]; "
                        "$v = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -like 'ar*' } | Select-Object -First 1; "
                        "if ($v) { Write-Output ($v.DisplayName + ' (' + $v.Language + ')') } else { exit 2 }"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return True, proc.stdout.strip()
            return False, "No Arabic WinRT voice found."
        except Exception as exc:
            return False, f"TTS check failed: {exc}"

    def synthesize_to_wav(self, text: str, out_wav: str | Path) -> Path:
        """Synthesize Arabic text into a WAV file synchronously."""
        clean = text.strip()
        if not clean:
            raise ValueError("Cannot synthesize empty text.")

        out_path = Path(out_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ps_file = tmp / "tts.ps1"
            txt_file = tmp / "input.txt"
            ps_file.write_text(_WINRT_TTS_PS, encoding="utf-8")
            txt_file.write_text(clean, encoding="utf-8")

            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ps_file),
                    "-TextPath",
                    str(txt_file),
                    "-OutWav",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                err = proc.stderr.strip() or proc.stdout.strip() or "Unknown TTS error"
                raise RuntimeError(f"Arabic TTS synthesis failed: {err}")

        return out_path

    def speak(self, text: str, blocking: bool = False) -> Path:
        """Synthesize Arabic text and play it via Windows audio output."""
        wav_path = Path(tempfile.gettempdir()) / "signbridge_tts_last.wav"
        with self._lock:
            self.synthesize_to_wav(text, wav_path)
            self._last_wav = wav_path
            flags = winsound.SND_FILENAME | (0 if blocking else winsound.SND_ASYNC)
            winsound.PlaySound(str(wav_path), flags)
        return wav_path

    def repeat_last(self, blocking: bool = False) -> bool:
        """Replay the most recently synthesized Arabic speech WAV if present."""
        if self._last_wav is None or not self._last_wav.exists():
            return False
        flags = winsound.SND_FILENAME | (0 if blocking else winsound.SND_ASYNC)
        winsound.PlaySound(str(self._last_wav), flags)
        return True
