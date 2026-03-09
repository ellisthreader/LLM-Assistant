"""Main loop for Raspberry Pi voice assistant pipeline.

Pipeline:
Microphone -> STT -> LLM -> TTS -> Speaker
"""

from __future__ import annotations

from openai import OpenAI

from audio_input import record_audio
from config import INPUT_WAV_PATH, OPENAI_API_KEY, OUTPUT_WAV_PATH
from llm import generate_reply
from stt import transcribe_audio
from tts import play_audio, synthesize_speech


def run_assistant() -> None:
    """Run the continuous voice assistant loop."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_KEY":
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set it in config.py or via .env."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    history: list[dict[str, str]] = []

    print("Voice assistant started. Press Ctrl+C to stop.")

    while True:
        try:
            print("\nListening...")
            audio_path = record_audio(INPUT_WAV_PATH)

            transcription = transcribe_audio(audio_path, client=client)
            if not transcription:
                print("No transcription detected. Retrying...")
                continue

            print(f"You: {transcription}")

            reply = generate_reply(transcription, conversation_history=history, client=client)
            if not reply:
                print("LLM returned an empty response. Retrying...")
                continue

            print(f"Assistant: {reply}")

            history.append({"role": "user", "content": transcription})
            history.append({"role": "assistant", "content": reply})
            history = history[-10:]

            response_audio = synthesize_speech(reply, OUTPUT_WAV_PATH, client=client)
            if not response_audio:
                print("TTS generation failed. Skipping playback.")
                continue

            play_audio(response_audio)

        except KeyboardInterrupt:
            print("\nStopping assistant.")
            break
        except Exception as exc:
            print(f"[Main] Unexpected error: {exc}")


def main() -> None:
    run_assistant()


if __name__ == "__main__":
    main()
