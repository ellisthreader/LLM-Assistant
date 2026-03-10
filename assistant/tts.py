"""Text-to-speech synthesis and playback utilities."""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

from config import (
    MAX_RETRIES,
    OPENAI_API_KEY,
    RETRY_DELAY_SECONDS,
    TTS_MODEL,
    TTS_VOICE,
)

# TTS is spoken sentence-by-sentence; treat very large/long WAVs as corrupted output.
_MAX_TTS_WAV_BYTES = 25_000_000  # 25 MB
_MAX_TTS_SECONDS = 90.0


def _pcm_bytes_to_float32(
    raw: bytes, *, channels: int, sample_width_bytes: int
) -> np.ndarray:
    """Convert PCM frames (little-endian) into float32 in range ~[-1, 1]."""
    if channels <= 0:
        raise ValueError(f"Invalid channel count: {channels}")

    if sample_width_bytes == 1:
        # 8-bit WAV PCM is unsigned.
        audio_u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio_u8 - 128.0) / 128.0
    elif sample_width_bytes == 2:
        audio_i16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio = audio_i16 / 32768.0
    elif sample_width_bytes == 3:
        # 24-bit PCM stored as little-endian packed bytes.
        data_u8 = np.frombuffer(raw, dtype=np.uint8)
        if data_u8.size % 3 != 0:
            raise ValueError("Invalid 24-bit PCM byte length")
        triples = data_u8.reshape(-1, 3).astype(np.int32)
        vals = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        # Sign-extend 24-bit to 32-bit.
        sign_bit = 1 << 23
        vals = (vals ^ sign_bit) - sign_bit
        audio = vals.astype(np.float32) / float(1 << 23)
    elif sample_width_bytes == 4:
        audio_i32 = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio = audio_i32 / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width_bytes} bytes")

    if channels == 1:
        return audio.reshape(-1, 1)
    if audio.size % channels != 0:
        raise ValueError("PCM frame count is not divisible by channel count")
    return audio.reshape(-1, channels)


def _read_wav_safe(file_path: str) -> tuple[int, np.ndarray]:
    """Read WAV using the stdlib wave module with sanity checks."""
    file_size = os.path.getsize(file_path)
    if file_size <= 44:
        raise ValueError(f"WAV too small ({file_size} bytes)")
    if file_size > _MAX_TTS_WAV_BYTES:
        raise ValueError(f"WAV too large ({file_size} bytes)")

    with contextlib.closing(wave.open(file_path, "rb")) as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        header_desc = (
            f"sr={sample_rate}, ch={channels}, sw={sample_width}, frames={frame_count}"
        )

        if channels < 1 or channels > 2:
            raise ValueError(f"Unexpected channel count: {channels} ({header_desc})")
        if sample_rate <= 0 or sample_rate > 192000:
            raise ValueError(f"Unexpected sample rate: {sample_rate} ({header_desc})")
        if frame_count <= 0:
            raise ValueError(f"WAV has no frames ({header_desc})")

        duration = frame_count / float(sample_rate)
        if duration > _MAX_TTS_SECONDS:
            raise ValueError(f"WAV duration too long ({duration:.1f}s) ({header_desc})")

        expected_pcm_bytes = frame_count * channels * sample_width
        # Header lies or truncated downloads can make alloc sizes explode; fail early.
        if expected_pcm_bytes > file_size:
            raise ValueError(
                f"WAV header/data mismatch (expected {expected_pcm_bytes} bytes, file {file_size} bytes) ({header_desc})"
            )

        raw = wf.readframes(frame_count)
        audio = _pcm_bytes_to_float32(
            raw, channels=channels, sample_width_bytes=sample_width
        )

    return sample_rate, audio


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
            # Newer SDKs use `response_format`; older ones used `format`.
            try:
                response = api_client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=text,
                    response_format="wav",
                )
            except TypeError:
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
        sample_rate, audio_data = _read_wav_safe(file_path)
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
        return True
    except Exception as exc:
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            file_size = -1
        print(f"[Playback] Failed: {exc} (file={file_path}, size={file_size} bytes)")
        return False


def synthesize_and_play(
    text: str,
    client: OpenAI | None = None,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """Synthesize speech for text and play it immediately."""
    chunk = text.strip()
    if not chunk:
        return False

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        audio_path = synthesize_speech(
            chunk,
            output_path=tmp_path,
            client=client,
            max_retries=max_retries,
        )
        if not audio_path:
            return False
        return play_audio(audio_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
