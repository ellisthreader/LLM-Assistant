# Raspberry Pi Voice Assistant (OpenAI Pipeline)

A clean Python prototype for Raspberry Pi that runs this loop continuously:

`Microphone -> Whisper STT -> GPT LLM -> OpenAI TTS -> Speaker`

## Project Structure

```text
assistant/
├── main.py
├── audio_input.py
├── stt.py
├── llm.py
├── tts.py
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

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set your OpenAI key in one of these ways:

1. Edit `config.py` directly:

```python
OPENAI_API_KEY = "YOUR_KEY"
```

2. Or create a `.env` file in the `assistant/` directory:

```bash
OPENAI_API_KEY=your_real_key
MODEL_NAME=gpt-4.1-mini
```

Default audio settings in `config.py`:

- `SAMPLE_RATE = 16000`
- `RECORD_SECONDS = 5`

## Run

From inside `assistant/`:

```bash
python main.py
```

## Notes

- Recording uses mono 16 kHz WAV.
- STT, LLM, and TTS calls have retry logic.
- Empty transcriptions are skipped safely.
- Press `Ctrl+C` to stop the loop.
