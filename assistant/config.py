"""Configuration for the Raspberry Pi voice assistant."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
AUDIO_DTYPE = "int16"

INPUT_WAV_PATH = "input.wav"
OUTPUT_WAV_PATH = "response.wav"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5

SYSTEM_PROMPT = (
    "You are a concise, helpful Raspberry Pi voice assistant. "
    "Provide clear spoken-style responses."
)
