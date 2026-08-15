use crate::state::{AssistantEvent, Shared, TimerInfo};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ─── Timers ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TimerManager {
    inner: Arc<Mutex<TimerStore>>,
    event_tx: crossbeam_channel::Sender<AssistantEvent>,
    shared: Shared,
}

struct TimerStore {
    next_id: u64,
    timers: Vec<TimerInfo>,
}

impl TimerManager {
    pub fn new(event_tx: crossbeam_channel::Sender<AssistantEvent>, shared: Shared) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TimerStore {
                next_id: 1,
                timers: Vec::new(),
            })),
            event_tx,
            shared,
        }
    }

    fn sync_ui(&self) {
        let timers = self.inner.lock().unwrap().timers.clone();
        if let Ok(mut s) = self.shared.lock() {
            s.timers = timers;
        }
    }

    /// Must be called from within a tokio runtime.
    pub fn set(&self, secs: u64, label: &str) -> String {
        let label = if label.trim().is_empty() {
            "timer".to_string()
        } else {
            label.trim().to_string()
        };
        let id = {
            let mut store = self.inner.lock().unwrap();
            let id = store.next_id;
            store.next_id += 1;
            store.timers.push(TimerInfo {
                id,
                label: label.clone(),
                deadline: Instant::now() + Duration::from_secs(secs),
            });
            id
        };
        self.sync_ui();

        let mgr = self.clone();
        let fired_label = label.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(secs)).await;
            let still_there = {
                let mut store = mgr.inner.lock().unwrap();
                let before = store.timers.len();
                store.timers.retain(|t| t.id != id);
                store.timers.len() != before
            };
            if still_there {
                mgr.sync_ui();
                let _ = mgr
                    .event_tx
                    .send(AssistantEvent::TimerFired { label: fired_label });
            }
        });
        format!("Timer #{id} '{label}' set for {}.", human_duration(secs))
    }

    pub fn cancel(&self, query: &str) -> String {
        let q = query.trim().to_lowercase();
        let mut store = self.inner.lock().unwrap();
        let before = store.timers.len();
        if q.is_empty() || q == "all" {
            store.timers.clear();
        } else {
            store
                .timers
                .retain(|t| !t.label.to_lowercase().contains(&q) && t.id.to_string() != q);
        }
        let removed = before - store.timers.len();
        drop(store);
        self.sync_ui();
        if removed == 0 {
            format!("No timer matching '{query}' was found.")
        } else {
            format!("Cancelled {removed} timer(s).")
        }
    }

    pub fn list(&self) -> String {
        let store = self.inner.lock().unwrap();
        if store.timers.is_empty() {
            return "No timers are currently running.".into();
        }
        store
            .timers
            .iter()
            .map(|t| {
                let left = t.deadline.saturating_duration_since(Instant::now()).as_secs();
                format!("#{} '{}' — {} left", t.id, t.label, human_duration(left))
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

pub fn human_duration(total_secs: u64) -> String {
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    let mut parts = Vec::new();
    if h > 0 {
        parts.push(format!("{h} hour{}", if h == 1 { "" } else { "s" }));
    }
    if m > 0 {
        parts.push(format!("{m} minute{}", if m == 1 { "" } else { "s" }));
    }
    if s > 0 && h == 0 {
        parts.push(format!("{s} second{}", if s == 1 { "" } else { "s" }));
    }
    if parts.is_empty() {
        "0 seconds".into()
    } else {
        parts.join(" ")
    }
}

// ─── Tool definitions ──────────────────────────────────────────────────────

pub fn tool_defs() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather and a 3-day forecast for a town or city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "Town or city name. Omit to use the user's configured home location." }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Start a countdown timer that will chime and announce itself when done.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": { "type": "integer", "description": "Total duration in seconds." },
                        "label": { "type": "string", "description": "Short name, e.g. 'pasta'." }
                    },
                    "required": ["seconds"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "cancel_timer",
                "description": "Cancel a running timer by label or id, or all timers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "which": { "type": "string", "description": "Label, id, or 'all'." }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_timers",
                "description": "List running timers and their remaining time.",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "set_volume",
                "description": "Set the speaker volume as a percentage from 0 to 100.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "percent": { "type": "integer", "description": "0 = mute, 100 = max." }
                    },
                    "required": ["percent"]
                }
            }
        }),
    ]
}

// ─── Execution ─────────────────────────────────────────────────────────────

pub struct ToolCtx {
    pub http: reqwest::Client,
    pub home_location: String,
    pub timers: TimerManager,
}

pub async fn execute(ctx: &ToolCtx, name: &str, args_json: &str) -> String {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| json!({}));
    let result = match name {
        "get_weather" => {
            let city = args["city"]
                .as_str()
                .filter(|c| !c.trim().is_empty())
                .unwrap_or(&ctx.home_location)
                .to_string();
            weather(&ctx.http, &city).await
        }
        "set_timer" => {
            let secs = args["seconds"].as_u64().unwrap_or(0);
            if secs == 0 || secs > 24 * 3600 {
                Ok("Invalid duration: must be between 1 second and 24 hours.".to_string())
            } else {
                Ok(ctx.timers.set(secs, args["label"].as_str().unwrap_or("")))
            }
        }
        "cancel_timer" => Ok(ctx.timers.cancel(args["which"].as_str().unwrap_or("all"))),
        "list_timers" => Ok(ctx.timers.list()),
        "set_volume" => {
            let pct = args["percent"].as_u64().unwrap_or(50).min(100);
            Ok(set_volume(pct))
        }
        other => Ok(format!("Unknown tool '{other}'.")),
    };
    match result {
        Ok(s) => s,
        Err(e) => format!("Tool failed: {e}"),
    }
}

fn set_volume(pct: u64) -> String {
    let amixer = std::process::Command::new("amixer")
        .args(["-M", "sset", "Master", &format!("{pct}%")])
        .output();
    if matches!(&amixer, Ok(o) if o.status.success()) {
        return format!("Volume set to {pct} percent.");
    }
    let pactl = std::process::Command::new("pactl")
        .args(["set-sink-volume", "@DEFAULT_SINK@", &format!("{pct}%")])
        .output();
    if matches!(&pactl, Ok(o) if o.status.success()) {
        return format!("Volume set to {pct} percent.");
    }
    "Could not change the volume on this system (no amixer or pactl control found).".into()
}

// ─── Weather (Open-Meteo, free, no API key) ────────────────────────────────

async fn weather(http: &reqwest::Client, city: &str) -> Result<String> {
    let geo: Value = http
        .get("https://geocoding-api.open-meteo.com/v1/search")
        .query(&[("name", city), ("count", "1"), ("format", "json")])
        .send()
        .await?
        .json()
        .await?;
    let hit = geo["results"][0].clone();
    if hit.is_null() {
        return Ok(format!("Could not find a place called '{city}'."));
    }
    let lat = hit["latitude"].as_f64().ok_or_else(|| anyhow!("bad geo data"))?;
    let lon = hit["longitude"].as_f64().ok_or_else(|| anyhow!("bad geo data"))?;
    let place = hit["name"].as_str().unwrap_or(city).to_string();
    let country = hit["country"].as_str().unwrap_or("").to_string();

    let fc: Value = http
        .get("https://api.open-meteo.com/v1/forecast")
        .query(&[
            ("latitude", lat.to_string().as_str()),
            ("longitude", lon.to_string().as_str()),
            (
                "current",
                "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
            ),
            (
                "daily",
                "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            ),
            ("timezone", "auto"),
            ("forecast_days", "3"),
        ])
        .send()
        .await?
        .json()
        .await?;

    let cur = &fc["current"];
    let daily = &fc["daily"];
    let mut out = format!(
        "Weather for {place}{}{}: now {}°C (feels like {}°C), {}, wind {} km/h, humidity {}%.",
        if country.is_empty() { "" } else { ", " },
        country,
        cur["temperature_2m"].as_f64().map(|v| v.round()).unwrap_or(0.0),
        cur["apparent_temperature"].as_f64().map(|v| v.round()).unwrap_or(0.0),
        weather_code_text(cur["weather_code"].as_u64().unwrap_or(0)),
        cur["wind_speed_10m"].as_f64().map(|v| v.round()).unwrap_or(0.0),
        cur["relative_humidity_2m"].as_u64().unwrap_or(0),
    );
    let names = ["Today", "Tomorrow", "Day after"];
    for i in 0..3 {
        if daily["time"][i].is_null() {
            break;
        }
        out.push_str(&format!(
            " {}: {} to {}°C, {}, {}% chance of rain.",
            names[i],
            daily["temperature_2m_min"][i].as_f64().map(|v| v.round()).unwrap_or(0.0),
            daily["temperature_2m_max"][i].as_f64().map(|v| v.round()).unwrap_or(0.0),
            weather_code_text(daily["weather_code"][i].as_u64().unwrap_or(0)),
            daily["precipitation_probability_max"][i].as_u64().unwrap_or(0),
        ));
    }
    Ok(out)
}

fn weather_code_text(code: u64) -> &'static str {
    match code {
        0 => "clear sky",
        1 => "mostly clear",
        2 => "partly cloudy",
        3 => "overcast",
        45 | 48 => "foggy",
        51 | 53 | 55 => "drizzle",
        56 | 57 => "freezing drizzle",
        61 | 63 | 65 => "rain",
        66 | 67 => "freezing rain",
        71 | 73 | 75 => "snow",
        77 => "snow grains",
        80 | 81 | 82 => "rain showers",
        85 | 86 => "snow showers",
        95 => "thunderstorm",
        96 | 99 => "thunderstorm with hail",
        _ => "mixed conditions",
    }
}
