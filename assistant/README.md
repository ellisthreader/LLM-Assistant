# Raspberry Pi Voice Assistant (OpenAI Pipeline)

A Python voice assistant for Raspberry Pi with this pipeline:

`Wake Word -> Microphone -> Whisper STT -> GPT LLM -> OpenAI TTS -> Speaker`

## Project Structure

```text
assistant/
├── main.py
├── audio_input.py
├── stt.py
├── llm.py
├── tts.py
├── weather.py
├── wake_word.py
├── reminders.py
├── config.py
├── requirements.txt
└── README.md
```

## Requirements

- Raspberry Pi (Linux)
- Python 3.10+
- USB microphone
- Audio output device (speaker/headphones)

Install OS audio dependencies:

```bash
sudo apt update
sudo apt install portaudio19-dev python3-pyaudio
```

Create and activate virtualenv:

```bash
cd assistant
sudo apt install python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy your existing `.env` into `assistant/.env`.

Required keys (already supported by code):
- `OPENAI_API_KEY`
- `TTS_VOICE=echo`
- `WAKE_WORD_ENABLED`
- `WAKE_WORD_ENGINE`
- `PORCUPINE_ACCESS_KEY`
- `PORCUPINE_KEYWORD_PATH` (if using custom wakeword model)

Notes:
- Default TTS voice is `echo`.
- For exact phrase `Hey Luna` with Porcupine, create a custom `.ppn` keyword model in Picovoice Console and set `PORCUPINE_KEYWORD_PATH`.
- If Porcupine is unavailable, assistant falls back to STT-based wake phrase detection.
- Place custom wakeword files in a stable path, for example:
  `/home/pi/assistant/wakewords/hello-assistant/Hello-Assistant_en_linux_v4_0_0.ppn`

## Features

1. Wake Word Detection
- Waits for wake word before running full assistant flow.
- Porcupine-first (low CPU), fallback wake phrase mode available.

2. Context Memory
- Maintains recent conversation history.
- Controlled by `MAX_HISTORY_MESSAGES` to prevent context growth.

3. Reminders & Alarms
- Commands like:
  - `Remind me to water the plants at 6 PM`
  - `Remind me to stand up in 30 minutes`
  - `Set alarm at 6 PM`
- Stored in local JSON file (`REMINDERS_FILE`).
- Background scheduler checks timestamps and speaks due reminders.

4. Real-time News/Weather
- `ENABLE_WEB_SEARCH=true` enables web lookup for current topics.
- Weather queries use approximate IP location + Open-Meteo live conditions.

5. Low-Latency Speech
- LLM output is streamed.
- TTS speaks sentence chunks while generation is still ongoing.

## Run

From inside `assistant/`:

```bash
python main.py
```

From repo root:

```bash
source assistant/.venv/bin/activate
python assistant/main.py
```

## Raspberry Pi Quick Setup

```bash
git clone <your-repo-url>
cd <repo>/assistant
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Copy your working .env into this folder
python main.py
```

## Usage Tips

- Ask weather naturally: `What's the temperature outside?`
- Set reminders naturally: `Remind me to call John at 7:30 PM`
- Press `Ctrl+C` to stop.
