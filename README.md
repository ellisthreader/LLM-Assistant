# LLM Assistant

An embedded voice-assistant platform that connects Raspberry Pi audio hardware to a real LLM pipeline for natural, spoken interaction.

The system is designed for a physical assistant device: it listens for a wake word, records microphone input, transcribes speech, generates context-aware LLM responses, and streams synthesized speech back through connected speakers. It also includes practical assistant features such as local reminders, alarms, conversational history, and live weather lookup.

## Highlights

- Hardware-ready Raspberry Pi voice pipeline: wake word, microphone input, STT, LLM response, TTS, and speaker playback.
- Conversational context management so the assistant can maintain recent dialogue and respond more naturally.
- Low-latency streamed responses, with text-to-speech beginning before the full LLM reply has finished.
- Local reminder and alarm scheduling with spoken notifications.
- Weather support using approximate IP location and live Open-Meteo conditions.
- Modular Python structure for audio input, transcription, language-model interaction, speech synthesis, wake-word detection, reminders, and weather.

## Technical Stack

- Python
- Raspberry Pi / Linux audio
- OpenAI APIs for speech-to-text, LLM responses, and text-to-speech
- Porcupine wake-word support with STT fallback
- Open-Meteo weather data

## Repository Structure

```text
assistant/
├── main.py
├── audio_input.py
├── stt.py
├── llm.py
├── tts.py
├── wake_word.py
├── reminders.py
├── weather.py
├── config.py
├── requirements.txt
└── README.md
```

See [assistant/README.md](assistant/README.md) for setup, configuration, and Raspberry Pi run instructions.
