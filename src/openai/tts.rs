use super::{short_api_error, OpenAi, API_BASE};
use anyhow::{anyhow, Result};
use serde_json::json;

impl OpenAi {
    /// Synthesize speech for one chunk of text. Returns encoded WAV bytes.
    pub async fn tts(&self, text: &str) -> Result<Vec<u8>> {
        let mut body = json!({
            "model": self.cfg.tts_model,
            "voice": self.cfg.tts_voice,
            "input": text,
            "response_format": "wav",
        });
        // `instructions` is only supported by the gpt-4o-* TTS family.
        if !self.cfg.tts_instructions.trim().is_empty() && !self.cfg.tts_model.starts_with("tts-1")
        {
            body["instructions"] = json!(self.cfg.tts_instructions);
        }

        let resp = self
            .http
            .post(format!("{API_BASE}/audio/speech"))
            .bearer_auth(self.key())
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!(short_api_error(status, &text)));
        }
        Ok(resp.bytes().await?.to_vec())
    }
}
