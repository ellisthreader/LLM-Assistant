pub mod chat;
pub mod stt;
pub mod tts;

use crate::config::OpenAiConfig;

pub const API_BASE: &str = "https://api.openai.com/v1";

/// Thin client over the OpenAI HTTP API. Cheap to clone.
#[derive(Clone)]
pub struct OpenAi {
    pub http: reqwest::Client,
    pub cfg: OpenAiConfig,
    key: String,
}

impl OpenAi {
    pub fn new(cfg: OpenAiConfig, key: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("http client");
        Self { http, cfg, key }
    }

    pub fn key(&self) -> &str {
        &self.key
    }
}

/// Turn an error body from the API into something short enough to speak/show.
pub fn short_api_error(status: reqwest::StatusCode, body: &str) -> String {
    let msg = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v["error"]["message"]
                .as_str()
                .map(|s| s.chars().take(140).collect::<String>())
        })
        .unwrap_or_else(|| body.chars().take(140).collect());
    format!("OpenAI API error {status}: {msg}")
}
