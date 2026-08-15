use super::{short_api_error, OpenAi, API_BASE};
use anyhow::{anyhow, Result};

impl OpenAi {
    /// Transcribe a WAV recording to text.
    pub async fn transcribe(&self, wav: Vec<u8>) -> Result<String> {
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("speech.wav")
            .mime_str("audio/wav")?;
        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.cfg.stt_model.clone())
            .text("response_format", "json");

        let resp = self
            .http
            .post(format!("{API_BASE}/audio/transcriptions"))
            .bearer_auth(self.key())
            .multipart(form)
            .send()
            .await?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(anyhow!(short_api_error(status, &body)));
        }
        let v: serde_json::Value = serde_json::from_str(&body)?;
        Ok(v["text"].as_str().unwrap_or("").trim().to_string())
    }
}
