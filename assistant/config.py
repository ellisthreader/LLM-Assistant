"""Configuration for the Raspberry Pi voice assistant."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "echo")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))

WAKE_WORD_ENABLED = os.getenv("WAKE_WORD_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WAKE_WORD_ENGINE = os.getenv("WAKE_WORD_ENGINE", "porcupine")
WAKE_WORD_PHRASE = os.getenv("WAKE_WORD_PHRASE", "hey luna")
WAKE_WORD_LISTEN_SECONDS = float(os.getenv("WAKE_WORD_LISTEN_SECONDS", "1.6"))
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "")
PORCUPINE_KEYWORD_PATH = os.getenv("PORCUPINE_KEYWORD_PATH", "")

REMINDERS_FILE = os.getenv("REMINDERS_FILE", "reminders.json")
REMINDER_POLL_SECONDS = float(os.getenv("REMINDER_POLL_SECONDS", "1.0"))

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
    "Provide clear spoken-style responses. "
    "For time-sensitive questions, use available tools to fetch current information."
)
