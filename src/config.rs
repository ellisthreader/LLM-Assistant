use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub openai: OpenAiConfig,
    pub assistant: AssistantConfig,
    pub wake: WakeConfig,
    pub audio: AudioConfig,
    pub ui: UiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiConfig {
    /// Leave empty to use the OPENAI_API_KEY environment variable.
    pub api_key: String,
    pub chat_model: String,
    pub stt_model: String,
    pub tts_model: String,
    pub tts_voice: String,
    pub tts_instructions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AssistantConfig {
    pub name: String,
    pub wake_phrase: String,
    /// Default town/city for weather lookups.
    pub location: String,
    /// Seconds to keep listening for a follow-up after the assistant replies.
    pub follow_up_window_secs: f32,
    /// Hard cap on a single spoken request.
    pub max_utterance_secs: f32,
    /// Silence that ends an utterance, in milliseconds.
    pub end_silence_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WakeConfig {
    pub model_path: String,
    pub sample_dir: String,
    /// 0..1 – higher = fewer false triggers, harder to wake.
    pub threshold: f32,
    pub avg_threshold: f32,
    /// MFCC coefficients per frame (5 is fast, 16 is more accurate).
    pub mfcc_size: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Substring of the input device name; empty = system default.
    pub input_device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub fullscreen: bool,
    pub width: f32,
    pub height: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            openai: OpenAiConfig::default(),
            assistant: AssistantConfig::default(),
            wake: WakeConfig::default(),
            audio: AudioConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            chat_model: "gpt-4o-mini".into(),
            stt_model: "gpt-4o-mini-transcribe".into(),
            tts_model: "gpt-4o-mini-tts".into(),
            tts_voice: "nova".into(),
            tts_instructions: "Speak warmly and naturally, like a helpful personal assistant on a desk. \
                               Brisk pace, friendly tone, never robotic."
                .into(),
        }
    }
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            name: "Pixel".into(),
            wake_phrase: "hey pixel".into(),
            location: "London".into(),
            follow_up_window_secs: 6.0,
            max_utterance_secs: 20.0,
            end_silence_ms: 900,
        }
    }
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            model_path: "wake-word.rpw".into(),
            sample_dir: "wake-samples".into(),
            threshold: 0.5,
            avg_threshold: 0.2,
            mfcc_size: 5,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: String::new(),
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            fullscreen: true,
            width: 800.0,
            height: 480.0,
        }
    }
}

impl Config {
    pub fn load_or_create(path: &str) -> Result<Self> {
        if Path::new(path).exists() {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("reading config file {path}"))?;
            let cfg: Config = toml::from_str(&raw).with_context(|| format!("parsing {path}"))?;
            Ok(cfg)
        } else {
            let cfg = Config::default();
            let raw = toml::to_string_pretty(&cfg)?;
            std::fs::write(path, raw).with_context(|| format!("writing default config {path}"))?;
            log::info!("wrote default config to {path}");
            Ok(cfg)
        }
    }

    pub fn api_key(&self) -> Option<String> {
        if !self.openai.api_key.trim().is_empty() {
            return Some(self.openai.api_key.trim().to_string());
        }
        std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.trim().is_empty())
    }
}
