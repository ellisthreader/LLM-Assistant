"""Microphone recording utilities."""

from __future__ import annotations

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

from config import AUDIO_DTYPE, CHANNELS, RECORD_SECONDS, SAMPLE_RATE


def record_audio(
    output_path: str,
    duration_seconds: int = RECORD_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Record microphone audio and save it as a 16 kHz mono WAV file."""
    frames = int(duration_seconds * sample_rate)

    try:
        audio: np.ndarray = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
        )
        sd.wait()
        write_wav(output_path, sample_rate, audio)
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc
