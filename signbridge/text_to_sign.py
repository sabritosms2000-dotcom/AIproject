"""
SignBridge AI — Arabic Text-to-Sign Vocabulary & Real Video Lookup (Stage 6).

Matches normalized Arabic text against the 70 verified KArSL-502 signs in
`dataset_clean.csv` and resolves real MP4 videos from `sample_download/extracted/`.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import pandas as pd

from signbridge.config import LABELS_CSV, PROJECT_ROOT

_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06FF]+")


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text without collapsing distinct hamza/alif-maqsoura/taa-marbuuta
    letters that are separate classes in the 70-sign vocabulary.
    """
    cleaned = _ARABIC_DIACRITICS_RE.sub("", text.strip())
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class TextToSignMapper:
    """Phrase-first Arabic text-to-sign matcher backed by real dataset videos."""

    def __init__(self, labels_csv: str | Path = LABELS_CSV, root_dir: str | Path = PROJECT_ROOT) -> None:
        self.labels_csv = Path(labels_csv)
        self.root_dir = Path(root_dir)
        self.vocab: list[dict[str, Any]] = []
        self._lookup: dict[str, dict[str, Any]] = {}
        self._phrases_sorted: list[tuple[str, dict[str, Any]]] = []
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Vocabulary CSV not found: {self.labels_csv}")

        df = pd.read_csv(self.labels_csv, encoding="utf-8-sig")
        for sign_id, grp in df.groupby("SignID", sort=True):
            row0 = grp.iloc[0]
            arabic_label = str(row0["Arabic_Label"]).strip()
            english_label = str(row0["English_Label"]).strip()
            class_index = int(sign_id) - 1

            # Pick first existing MP4 video on disk
            video_path: Path | None = None
            for raw_p in grp["Video_Path"].astype(str):
                cand = self.root_dir / raw_p.replace("\\", "/")
                if cand.exists() and cand.is_file():
                    video_path = cand
                    break

            entry = {
                "class_index": class_index,
                "sign_id": f"{int(sign_id):04d}",
                "arabic_label": arabic_label,
                "english_label": english_label,
                "video_path": str(video_path) if video_path else None,
                "verified": video_path is not None,
            }
            self.vocab.append(entry)

            norm_ar = normalize_arabic(arabic_label)
            self._lookup[norm_ar] = entry
            # Also allow matching numeric digits (e.g. "0", "1000") for number signs
            if english_label.isdigit():
                self._lookup[english_label] = entry

        # Longest phrases first so "عشرة ملايين" matches before "عشرة"
        self._phrases_sorted = sorted(
            self._lookup.items(),
            key=lambda kv: len(kv[0].split()),
            reverse=True,
        )

    def translate(self, text: str) -> dict[str, Any]:
        """
        Match Arabic text into an ordered sequence of verified sign entries and
        explicitly flag unsupported tokens.
        """
        norm = normalize_arabic(text)
        if not norm:
            return {
                "normalized_text": "",
                "matched_signs": [],
                "unsupported_tokens": [],
                "playlist": [],
            }

        tokens = norm.split()
        matched_signs: list[dict[str, Any]] = []
        unsupported_tokens: list[str] = []

        i = 0
        while i < len(tokens):
            matched_entry = None
            matched_len = 0

            for phrase, entry in self._phrases_sorted:
                p_tokens = phrase.split()
                plen = len(p_tokens)
                if i + plen <= len(tokens) and tokens[i : i + plen] == p_tokens:
                    matched_entry = entry
                    matched_len = plen
                    break

            if matched_entry is not None:
                matched_signs.append(matched_entry)
                i += matched_len
            else:
                unsupported_tokens.append(tokens[i])
                i += 1

        playlist = [
            m for m in matched_signs if m["verified"] and m["video_path"] is not None
        ]

        return {
            "normalized_text": norm,
            "matched_signs": matched_signs,
            "unsupported_tokens": unsupported_tokens,
            "playlist": playlist,
        }
