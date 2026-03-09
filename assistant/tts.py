"""Text-to-speech synthesis and playback utilities."""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd
from openai import OpenAI
from scipy.io.wavfile import read as read_wav

from config import (
    MAX_RETRIES,
    OPENAI_API_KEY,
    RETRY_DELAY_SECONDS,
    TTS_MODEL,
    TTS_VOICE,
)


def synthesize_speech(
    text: str,
    output_path: str,
    client: OpenAI | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Generate speech audio from text with OpenAI TTS and save as WAV."""
    api_client = client or OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(1, max_retries + 1):
        try:
            response = api_client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                format="wav",
            )
            response.stream_to_file(output_path)
            return output_path
        except Exception as exc:
            print(f"[TTS] Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY_SECONDS)

    return ""


def play_audio(file_path: str) -> bool:
    """Play a WAV file through the default speaker output."""
    try:
        sample_rate, audio_data = read_wav(file_path)

        # Normalize integer PCM to float32 for consistent playback across devices.
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / float(max_val)

        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
        return True
    except Exception as exc:
        print(f"[Playback] Failed: {exc}")
        return False
