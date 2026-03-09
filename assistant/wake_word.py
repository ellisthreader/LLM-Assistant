"""Wake word detection utilities.

Primary mode uses Porcupine (low CPU). If Porcupine is unavailable,
a fallback mode uses short STT snippets to detect the wake phrase.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import sounddevice as sd
from openai import OpenAI

from audio_input import record_audio
from config import (
    PORCUPINE_ACCESS_KEY,
    PORCUPINE_KEYWORD_PATH,
    WAKE_WORD_ENGINE,
    WAKE_WORD_PHRASE,
    WAKE_WORD_LISTEN_SECONDS,
)
from stt import transcribe_audio

try:
    import pvporcupine
except Exception:  # pragma: no cover - optional dependency path
    pvporcupine = None


def _wait_for_porcupine() -> bool:
    if pvporcupine is None or not PORCUPINE_ACCESS_KEY:
        return False

    porcupine = None
    stream = None
    try:
        if PORCUPINE_KEYWORD_PATH:
            porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=[PORCUPINE_KEYWORD_PATH],
            )
        else:
            # Built-in keyword fallback if no custom "Hey Luna" model is configured.
            porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keywords=["porcupine"],
            )

        stream = sd.InputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=porcupine.frame_length,
        )
        stream.start()

        while True:
            pcm, _ = stream.read(porcupine.frame_length)
            frame = np.asarray(pcm[:, 0], dtype=np.int16)
            keyword_index = porcupine.process(frame)
            if keyword_index >= 0:
                return True
    except Exception as exc:
        print(f"[WakeWord] Porcupine mode failed: {exc}")
        return False
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        if porcupine is not None:
            porcupine.delete()


def _wait_for_stt_phrase(client: OpenAI | None = None) -> bool:
    phrase = WAKE_WORD_PHRASE.lower().strip()
    print(f"Wake phrase mode active. Say '{WAKE_WORD_PHRASE}' to activate.")
    consecutive_stt_failures = 0

    while True:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            record_audio(tmp_path, duration_seconds=WAKE_WORD_LISTEN_SECONDS)
            text = transcribe_audio(
                tmp_path,
                client=client,
                max_retries=1,
                log_errors=False,
            ).lower().strip()
            if not text:
                consecutive_stt_failures += 1
                if consecutive_stt_failures % 10 == 0:
                    print(
                        "[WakeWord] STT wake check still failing (network/API issue). "
                        "Continuing to listen..."
                    )
                continue
            consecutive_stt_failures = 0
            if phrase in text:
                return True
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            time.sleep(0.05)


def wait_for_wake_word(client: OpenAI | None = None) -> bool:
    """Block until wake word is detected."""
    engine = WAKE_WORD_ENGINE.lower().strip()

    if engine == "porcupine":
        detected = _wait_for_porcupine()
        if detected:
            return True
        print("[WakeWord] Falling back to STT wake phrase detection.")

    return _wait_for_stt_phrase(client=client)
