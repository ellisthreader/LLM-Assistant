# Desk Assistant

A wake-word voice assistant for a Raspberry Pi with a touchscreen, written in Rust.
Say the wake phrase (or tap the orb) and talk; it answers out loud with a natural
OpenAI voice, shows a live transcript, and runs real tools: weather, timers, and
volume control. It is honest about what it can and cannot do.

- **Wake word** — [rustpotter], fully offline and free; you train it on your own voice
- **Speech-to-text** — OpenAI `gpt-4o-mini-transcribe`
- **Brain** — OpenAI `gpt-4o-mini` (cheapest effective mini model; configurable)
- **Text-to-speech** — OpenAI `gpt-4o-mini-tts`, streamed sentence-by-sentence so it
  starts talking fast
- **UI** — fullscreen egui app: animated orb that reacts to your voice, clock,
  timers, live transcript

Rough running cost: a typical spoken exchange uses ~$0.001–0.003 of API credit
(mostly TTS). Heavy daily use lands around a few dollars a month.

## 1. Set up on the Raspberry Pi

Raspberry Pi 4 or 5 with Raspberry Pi OS (64-bit, desktop), a microphone
(USB mic or ReSpeaker HAT), and speakers/screen connected.

```bash
sudo apt update
sudo apt install -y git build-essential pkg-config libasound2-dev \
    libx11-dev libxi-dev libxcursor-dev libxrandr-dev libxkbcommon-dev \
    libxkbcommon-x11-dev libgl1-mesa-dev libwayland-dev

# Rust toolchain (rustup picks up this repo's pinned version automatically)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Copy this project onto the Pi (from your PC):
#   scp -r VoiceAssistant pi@raspberrypi.local:~/
cd ~/VoiceAssistant
cargo build --release        # first build takes a while on a Pi — get a coffee
```

Or just run `./setup-pi.sh`, which does all of the above.

## 2. Configure

```bash
export OPENAI_API_KEY=sk-...     # or put it in config.toml → [openai] api_key
```

Run once to generate `config.toml`, then edit it:

- `assistant.name` / `assistant.wake_phrase` — call it whatever you like
- `assistant.location` — your town, used for weather
- `openai.chat_model` — `gpt-4o-mini` (default, cheapest), or `gpt-5-mini` for a
  smarter brain, `gpt-4.1-nano` / `gpt-5-nano` for even cheaper
- `openai.tts_voice` — try `nova`, `alloy`, `ash`, `coral`, `sage`, `shimmer`
- `ui.fullscreen` — `true` for the desk display

## 3. Train your wake word

```bash
./target/release/desk-assistant train-wake
```

It records 6 short samples of you saying the phrase and builds `wake-word.rpw`.
Do this on the Pi with the mic you'll actually use, at your normal
speaking distance. Until a model exists, tapping the orb (or Space) wakes it.

If it triggers too easily / not easily enough, adjust `[wake] threshold`
(higher = stricter) in `config.toml`.

## 4. Run

```bash
./target/release/desk-assistant            # run the assistant
./target/release/desk-assistant devices    # list microphones (put a name substring in [audio] input_device)
```

- Say the wake phrase or tap the orb → chime → speak
- It keeps listening for a few seconds after each answer — just keep talking
- Tap while it's speaking to interrupt it
- `Esc` quits

## 5. Start on boot

```bash
mkdir -p ~/.config/systemd/user
cp desk-assistant.service ~/.config/systemd/user/
# edit the service file if your project path or API key differ
systemctl --user daemon-reload
systemctl --user enable --now desk-assistant
loginctl enable-linger $USER      # keep it running without an open session
```

Logs: `journalctl --user -u desk-assistant -f`

## What it can do

Ask it anything conversational, plus:

- **Weather** — "What's the weather like?", "Will it rain in Manchester tomorrow?"
- **Timers** — "Set a 10 minute pasta timer", "How long left?", "Cancel the timer"
  (chimes and announces itself when done; countdown shows on screen)
- **Volume** — "Set your volume to 40 percent"
- **Time & date** — always knows the current local time

It will tell you straight when something is beyond it (browsing, music,
smart-home control, remembering after a restart).

## Troubleshooting

- **It can't hear you** — `desk-assistant devices`, then set `[audio] input_device`
  to a substring of your mic's name. Check `alsamixer` capture level.
- **Wake word flaky** — retrain in the room you use it in; lower `[wake] threshold`
  to ~0.45, or raise `mfcc_size` to 16 for better accuracy (more CPU).
- **No sound out** — pick the right output in `raspi-config` / desktop audio menu;
  the app uses the system default output.
- **Window doesn't open over SSH** — run it from the Pi's desktop session, or set
  `DISPLAY=:0` (see the systemd service).

[rustpotter]: https://github.com/GiviMAD/rustpotter
