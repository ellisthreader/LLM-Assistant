use super::{short_api_error, OpenAi, API_BASE};
use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct ToolCallReq {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Stream a chat completion. `on_delta` receives text as it is generated.
/// Returns the full assistant text plus any tool calls the model requested.
/// Setting `cancel` aborts the stream early (used for barge-in).
impl OpenAi {
    pub async fn chat_stream(
        &self,
        messages: &[Value],
        tools: &[Value],
        cancel: &Arc<AtomicBool>,
        mut on_delta: impl FnMut(&str),
    ) -> Result<(String, Vec<ToolCallReq>)> {
        let mut body = json!({
            "model": self.cfg.chat_model,
            "messages": messages,
            "stream": true,
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
        }

        let resp = self
            .http
            .post(format!("{API_BASE}/chat/completions"))
            .bearer_auth(self.key())
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!(short_api_error(status, &text)));
        }

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCallReq> = Vec::new();
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();

        'outer: while let Some(chunk) = stream.next().await {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let chunk = chunk?;
            buf.extend_from_slice(&chunk);
            // SSE frames are newline-delimited; \n never splits a UTF-8 char.
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    break 'outer;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                let delta = &v["choices"][0]["delta"];
                if let Some(t) = delta["content"].as_str() {
                    if !t.is_empty() {
                        content.push_str(t);
                        on_delta(t);
                    }
                }
                if let Some(tcs) = delta["tool_calls"].as_array() {
                    for tc in tcs {
                        let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                        while tool_calls.len() <= idx {
                            tool_calls.push(ToolCallReq::default());
                        }
                        let slot = &mut tool_calls[idx];
                        if let Some(id) = tc["id"].as_str() {
                            slot.id = id.to_string();
                        }
                        if let Some(n) = tc["function"]["name"].as_str() {
                            slot.name.push_str(n);
                        }
                        if let Some(a) = tc["function"]["arguments"].as_str() {
                            slot.arguments.push_str(a);
                        }
                    }
                }
            }
        }
        Ok((content, tool_calls))
    }
}
