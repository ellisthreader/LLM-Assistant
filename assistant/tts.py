"""Text-to-speech synthesis and playback utilities."""

from __future__ import annotations

import os
import struct
import tempfile
import time

import numpy as np
import sounddevice as sd
from openai import OpenAI

from config import (
    AUDIO_OUTPUT_DEVICE,
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
    """Read WAV with sanity checks.

    Some streaming WAV writers emit a placeholder 0xFFFFFFFF data chunk size, which
    makes naive readers believe the file is ~4GB long. We clamp reads to the actual
    file size and derive frame count from real bytes.
    """
    file_size = os.path.getsize(file_path)
    if file_size <= 44:
        raise ValueError(f"WAV too small ({file_size} bytes)")
    if file_size > _MAX_TTS_WAV_BYTES:
        raise ValueError(f"WAV too large ({file_size} bytes)")

    with open(file_path, "rb") as f:
        riff_header = f.read(12)
        if len(riff_header) != 12:
            raise ValueError("Failed to read WAV header")

        riff, _riff_size, wave_id = struct.unpack("<4sI4s", riff_header)
        if riff != b"RIFF" or wave_id != b"WAVE":
            head = riff_header + f.read(20)
            raise ValueError(f"Not a RIFF/WAVE file (head={head[:32].hex()})")

        channels: int | None = None
        sample_rate: int | None = None
        bits_per_sample: int | None = None
        audio_format: int | None = None
        block_align: int | None = None

        data_offset: int | None = None
        data_size: int | None = None

        # Walk chunks until we find fmt/data.
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) != 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_start = f.tell()

            if chunk_id == b"fmt ":
                fmt = f.read(chunk_size)
                if len(fmt) < 16:
                    raise ValueError("Invalid fmt chunk")
                (
                    audio_format,
                    channels,
                    sample_rate,
                    _byte_rate,
                    block_align,
                    bits_per_sample,
                ) = struct.unpack("<HHIIHH", fmt[:16])
            elif chunk_id == b"data":
                data_offset = f.tell()
                data_size = chunk_size
                # Don't seek past data; we'll read after loop.
                f.seek(chunk_start + chunk_size)
            else:
                # Skip unknown chunk.
                f.seek(chunk_start + chunk_size)

            # Chunks are padded to even sizes.
            if chunk_size % 2 == 1:
                f.seek(1, os.SEEK_CUR)

            if (channels is not None) and (sample_rate is not None) and (data_offset is not None):
                break

        if (
            channels is None
            or sample_rate is None
            or bits_per_sample is None
            or audio_format is None
            or block_align is None
            or data_offset is None
            or data_size is None
        ):
            raise ValueError("Missing required WAV chunks (fmt/data)")

        header_desc = (
            f"sr={sample_rate}, ch={channels}, bps={bits_per_sample}, fmt={audio_format}, align={block_align}, data={data_size}"
        )

        if channels < 1 or channels > 2:
            raise ValueError(f"Unexpected channel count: {channels} ({header_desc})")
        if sample_rate <= 0 or sample_rate > 192000:
            raise ValueError(f"Unexpected sample rate: {sample_rate} ({header_desc})")
        if block_align <= 0:
            raise ValueError(f"Unexpected block align: {block_align} ({header_desc})")

        # Clamp to actual remaining bytes in the file (handles 0xFFFFFFFF placeholder sizes).
        remaining = max(0, file_size - data_offset)
        clamped_data_size = min(int(data_size), int(remaining))
        if clamped_data_size <= 0:
            raise ValueError(f"WAV has no data bytes ({header_desc})")

        f.seek(data_offset)
        raw = f.read(clamped_data_size)
        if not raw:
            raise ValueError(f"Failed to read WAV data ({header_desc})")

    # Trim to whole frames.
    frame_bytes = block_align
    frame_count = len(raw) // frame_bytes
    if frame_count <= 0:
        raise ValueError(f"WAV has no complete frames ({header_desc})")
    raw = raw[: frame_count * frame_bytes]

    duration = frame_count / float(sample_rate)
    if duration > _MAX_TTS_SECONDS:
        raise ValueError(f"WAV duration too long ({duration:.1f}s) ({header_desc})")

    if audio_format == 1:  # PCM
        sample_width = int(bits_per_sample) // 8
        if sample_width <= 0:
            raise ValueError(f"Unexpected bits-per-sample: {bits_per_sample} ({header_desc})")
        audio = _pcm_bytes_to_float32(
            raw, channels=int(channels), sample_width_bytes=sample_width
        )
    elif audio_format == 3 and bits_per_sample == 32:  # IEEE float
        audio = np.frombuffer(raw, dtype=np.float32)
        if audio.size % int(channels) != 0:
            raise ValueError(f"Float PCM not divisible by channels ({header_desc})")
        audio = audio.reshape(-1, int(channels))
    else:
        raise ValueError(f"Unsupported WAV format ({header_desc})")

    return int(sample_rate), audio.astype(np.float32, copy=False)


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
        device: int | str | None = None
        if AUDIO_OUTPUT_DEVICE:
            device = (
                int(AUDIO_OUTPUT_DEVICE)
                if AUDIO_OUTPUT_DEVICE.isdigit()
                else AUDIO_OUTPUT_DEVICE
            )
        sd.play(audio_data, samplerate=sample_rate, device=device)
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
