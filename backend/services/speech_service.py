"""Speech-to-text (faster-whisper) and optional text-to-speech (gTTS).

Voice is optional everywhere: any failure raises :class:`SpeechUnavailable` with a message the
UI shows so the user can fall back to typing (e.g. microphone denied, model not downloaded).
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from functools import lru_cache

from backend.core.config import get_settings

log = logging.getLogger("prepwise.speech")
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}


class SpeechUnavailable(RuntimeError):
    """Voice feature cannot be used right now (user-facing message)."""


@lru_cache(maxsize=1)
def _whisper():
    s = get_settings()
    try:
        from faster_whisper import WhisperModel

        return WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type)
    except ImportError as exc:
        raise SpeechUnavailable("faster-whisper is not installed on the server. Please type your answer.") from exc
    except Exception as exc:  # model download blocked, out of memory, ...
        raise SpeechUnavailable(
            f"The Whisper model could not be loaded ({type(exc).__name__}). Please type your answer.") from exc


def transcribe(audio: bytes, filename: str = "answer.wav", language: str | None = None) -> dict:
    """Transcribe audio. Returns text, duration and per-segment timestamps (for pause analysis)."""
    s = get_settings()
    if not audio:
        raise SpeechUnavailable("No audio was received. Check that your microphone permission is allowed in the browser.")
    if len(audio) > s.max_audio_mb * 1024 * 1024:
        raise SpeechUnavailable(f"Audio is larger than {s.max_audio_mb} MB. Please record a shorter answer.")
    ext = os.path.splitext(filename.lower())[1] or ".wav"
    if ext not in ALLOWED_EXT:
        raise SpeechUnavailable(f"Unsupported audio type '{ext}'. Use one of {sorted(ALLOWED_EXT)}.")
    model = _whisper()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(audio)
        tmp.flush()
        try:
            segs, info = model.transcribe(tmp.name, language=language, vad_filter=True, beam_size=1)
            segments = [{"start": float(x.start), "end": float(x.end), "text": x.text.strip()} for x in segs]
        except Exception as exc:
            raise SpeechUnavailable(f"Could not decode this audio ({type(exc).__name__}).") from exc
    text = " ".join(x["text"] for x in segments).strip()
    if not text:
        raise SpeechUnavailable("No speech was detected in the recording. Please try again closer to the microphone.")
    speech_sec = segments[-1]["end"] - segments[0]["start"] if segments else 0.0
    return {"text": text, "duration_sec": round(float(info.duration or speech_sec), 2),
            "speech_sec": round(speech_sec, 2), "segments": segments, "language": info.language}


def synthesize(text: str, lang: str = "en") -> bytes:
    """Optional gTTS text-to-speech (needs internet). Returns MP3 bytes."""
    if not get_settings().tts_enabled:
        raise SpeechUnavailable("Text-to-speech is disabled.")
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text[:1500], lang={"en": "en", "ta": "ta", "hi": "hi"}.get(lang, "en")).write_to_fp(buf)
        return buf.getvalue()
    except ImportError as exc:
        raise SpeechUnavailable("gTTS is not installed.") from exc
    except Exception as exc:
        raise SpeechUnavailable(f"Text-to-speech failed ({type(exc).__name__}); it needs internet access.") from exc
