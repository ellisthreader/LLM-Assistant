"""Speech-to-text (Whisper API) integration."""

from __future__ import annotations

import time

from openai import OpenAI

from config import MAX_RETRIES, OPENAI_API_KEY, RETRY_DELAY_SECONDS, WHISPER_MODEL


def transcribe_audio(
    audio_path: str,
    client: OpenAI | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Transcribe speech from a WAV file using OpenAI Whisper."""
    api_client = client or OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(1, max_retries + 1):
        try:
            with open(audio_path, "rb") as audio_file:
                result = api_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                )

            text = (getattr(result, "text", "") or "").strip()
            if text:
                return text

            raise ValueError("Whisper returned an empty transcription")
        except Exception as exc:
            print(f"[STT] Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY_SECONDS)

    return ""
