"""Whisper transcription via MLX.

Whisper is a speech-to-text model, not an instruction-following one -- it has no
refusal behaviour to remove, so there is no "abliterated Whisper" and no need
for one. It transcribes whatever it hears.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path

# Roughly 1.6 GB and ~5x faster than large-v3 at close to the same accuracy.
WHISPER_MODEL = os.environ.get(
    "ABLIT_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
)


@dataclass
class Transcript:
    text: str
    segments: list[dict]
    language: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def as_timestamped(self, max_chars: int = 20_000) -> str:
        """[mm:ss] prefixed lines, so the LLM can tie narration to timeline."""
        lines = []
        for seg in self.segments:
            start = int(seg.get("start", 0))
            body = (seg.get("text") or "").strip()
            if body:
                lines.append(f"[{start // 60:02d}:{start % 60:02d}] {body}")
        joined = "\n".join(lines)
        if len(joined) > max_chars:
            joined = joined[:max_chars] + "\n… (transcript truncated)"
        return joined


def transcribe(wav_path: Path, language: str | None = None) -> Transcript:
    import mlx.core as mx
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=WHISPER_MODEL,
        language=language,
        # Silence in a screen recording is the classic Whisper hallucination
        # trigger; this suppresses the invented "thanks for watching" endings.
        hallucination_silence_threshold=2.0,
        condition_on_previous_text=False,
    )

    transcript = Transcript(
        text=(result.get("text") or "").strip(),
        segments=result.get("segments") or [],
        language=result.get("language") or "unknown",
    )

    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    return transcript
