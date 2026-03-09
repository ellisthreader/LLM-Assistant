"""Main loop for Raspberry Pi voice assistant pipeline.

Pipeline:
Wake Word -> Microphone -> STT -> LLM -> TTS -> Speaker
"""

from __future__ import annotations

import re
import threading
from queue import Queue
from threading import Thread

from openai import OpenAI

from audio_input import record_audio
from config import (
    INPUT_WAV_PATH,
    MAX_HISTORY_MESSAGES,
    OPENAI_API_KEY,
    REMINDER_POLL_SECONDS,
    REMINDERS_FILE,
    WAKE_WORD_ENABLED,
)
from llm import generate_reply, stream_reply_tokens
from reminders import Reminder, ReminderScheduler, ReminderStore, try_schedule_reminder
from stt import transcribe_audio
from tts import synthesize_and_play
from wake_word import wait_for_wake_word
from weather import (
    build_weather_context,
    get_current_weather,
    get_ip_location,
    is_weather_query,
)


def _extract_ready_chunks(buffer: str) -> tuple[list[str], str]:
    """Split buffered streamed text into speakable sentence chunks."""
    ready: list[str] = []
    last_idx = 0
    for match in re.finditer(r"[.!?](?:\s+|$)|\n", buffer):
        end_idx = match.end()
        chunk = buffer[last_idx:end_idx].strip()
        if chunk:
            ready.append(chunk)
        last_idx = end_idx
    remaining = buffer[last_idx:]
    return ready, remaining


def _tts_worker(
    text_queue: Queue[str | None],
    client: OpenAI,
    audio_lock: threading.Lock,
) -> None:
    """Consume text chunks and play them as they are synthesized."""
    while True:
        text = text_queue.get()
        try:
            if text is None:
                return
            with audio_lock:
                synthesize_and_play(text, client=client)
        finally:
            text_queue.task_done()


def run_assistant() -> None:
    """Run the continuous voice assistant loop."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_KEY":
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set it in config.py or via .env."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    history: list[dict[str, str]] = []
    audio_lock = threading.Lock()

    reminder_store = ReminderStore(REMINDERS_FILE)

    def _on_due_reminder(reminder: Reminder) -> None:
        reminder_text = f"Reminder: {reminder.text}."
        print(f"\n[Reminder] {reminder_text}")
        with audio_lock:
            synthesize_and_play(reminder_text, client=client)

    reminder_scheduler = ReminderScheduler(
        store=reminder_store,
        on_due=_on_due_reminder,
        poll_seconds=REMINDER_POLL_SECONDS,
    )
    reminder_scheduler.start()

    cached_location = get_ip_location()
    if cached_location:
        print(
            f"Location detected: {cached_location.city}, {cached_location.region}, {cached_location.country}"
        )
    else:
        print("Location detection failed. Weather answers may be less accurate.")

    print("Voice assistant started. Press Ctrl+C to stop.")

    try:
        while True:
            if WAKE_WORD_ENABLED:
                print("\nWaiting for wake word...")
                wait_for_wake_word(client=client)
                print("Wake word detected.")

            print("Listening...")
            audio_path = record_audio(INPUT_WAV_PATH)

            transcription = transcribe_audio(audio_path, client=client)
            if not transcription:
                print("No transcription detected. Retrying...")
                continue

            print(f"You: {transcription}")

            reminder_confirmation = try_schedule_reminder(transcription, reminder_store)
            if reminder_confirmation:
                print(f"Assistant: {reminder_confirmation}")
                with audio_lock:
                    synthesize_and_play(reminder_confirmation, client=client)
                history.append({"role": "user", "content": transcription})
                history.append({"role": "assistant", "content": reminder_confirmation})
                history = history[-MAX_HISTORY_MESSAGES:]
                continue

            enriched_transcription = transcription
            if is_weather_query(transcription):
                if not cached_location:
                    cached_location = get_ip_location()
                if cached_location:
                    weather = get_current_weather(cached_location)
                    if weather:
                        weather_context = build_weather_context(cached_location, weather)
                        enriched_transcription = f"{transcription}\n\n{weather_context}"

            print("Assistant: ", end="", flush=True)
            reply_parts: list[str] = []
            buffered_text = ""

            tts_queue: Queue[str | None] = Queue()
            worker = Thread(
                target=_tts_worker,
                args=(tts_queue, client, audio_lock),
                daemon=True,
            )
            worker.start()

            for token in stream_reply_tokens(
                enriched_transcription,
                conversation_history=history,
                client=client,
            ):
                print(token, end="", flush=True)
                reply_parts.append(token)
                buffered_text += token
                ready_chunks, buffered_text = _extract_ready_chunks(buffered_text)
                for chunk in ready_chunks:
                    tts_queue.put(chunk)

            if buffered_text.strip():
                tts_queue.put(buffered_text.strip())

            tts_queue.put(None)
            tts_queue.join()
            worker.join(timeout=1.0)
            print()

            reply = "".join(reply_parts).strip()
            if not reply:
                # Fallback path if streaming yields nothing.
                reply = generate_reply(
                    enriched_transcription,
                    conversation_history=history,
                    client=client,
                ).strip()
                if not reply:
                    print("LLM returned an empty response. Retrying...")
                    continue
                print(f"Assistant: {reply}")
                with audio_lock:
                    synthesize_and_play(reply, client=client)

            history.append({"role": "user", "content": transcription})
            history.append({"role": "assistant", "content": reply})
            history = history[-MAX_HISTORY_MESSAGES:]

    except KeyboardInterrupt:
        print("\nStopping assistant.")
    finally:
        reminder_scheduler.stop()
        reminder_scheduler.join(timeout=1.0)


def main() -> None:
    run_assistant()


if __name__ == "__main__":
    main()
