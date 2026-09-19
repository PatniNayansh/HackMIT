"""Synthesize spoken narration for a silent slide.

Decks in this app are silent PDFs (backend/sightline/ingest.py has no audio path), but
TRIBE needs narrated input -- a silent slide gives it almost nothing (spec 5). This
reads a slide's own extracted text aloud via gTTS (already a TRIBE v2 dependency, so no
extra service to stand up) and produces the (audio, transcript) pair the precompute
pipeline needs.

This is SYNTHETIC narration, not a real presenter reading the deck. Whatever a real talk
would emphasize, pause on, or skip, this cannot capture -- it is flat, uninflected TTS of
the extracted slide text. Disclose that in the Methods panel wherever neural output built
from a silent deck is shown, not just here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Narration:
    audio_path: Path
    transcript: str
    duration_s: float


def _clean_for_speech(slide_text: str) -> str:
    """Slide text carries `[title] ...` / `[body] ...` layout-role prefixes (see
    ingest.py); strip them so gTTS reads the words, not the markup."""
    lines = []
    for line in slide_text.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            line = line.split("]", 1)[1].strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def synthesize(slide_text: str, out_path: Path, *, lang: str = "en") -> Narration:
    """slide_text -> an mp3 at out_path + the transcript actually spoken. Raises ValueError
    on a slide with no extractable text: there is nothing to narrate, and TRIBE should not
    be run on it (an image-only slide needs a human-written caption, not a fabricated one)."""
    transcript = _clean_for_speech(slide_text)
    if not transcript:
        raise ValueError("slide has no extractable text to narrate (image-only slide)")

    from gtts import gTTS
    from mutagen.mp3 import MP3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=transcript, lang=lang).save(str(out_path))
    duration_s = MP3(out_path).info.length
    return Narration(audio_path=out_path, transcript=transcript, duration_s=duration_s)
