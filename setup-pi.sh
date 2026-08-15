#!/usr/bin/env bash
# One-shot setup for Raspberry Pi OS (64-bit). Run from the project directory.
set -euo pipefail

echo "==> Installing system packages…"
sudo apt update
sudo apt install -y git build-essential pkg-config libasound2-dev \
    libx11-dev libxi-dev libxcursor-dev libxrandr-dev libxkbcommon-dev \
    libxkbcommon-x11-dev libgl1-mesa-dev libwayland-dev

if ! command -v cargo >/dev/null 2>&1; then
    echo "==> Installing Rust…"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

echo "==> Building (release) — the first build takes a while on a Pi…"
cargo build --release

echo
echo "Done. Next steps:"
echo "  1. export OPENAI_API_KEY=sk-...   (or edit config.toml after first run)"
echo "  2. ./target/release/desk-assistant train-wake"
echo "  3. ./target/release/desk-assistant"
