use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What the assistant is doing right now — drives the orb animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Boot,
    Idle,
    Listening,
    Thinking,
    Speaking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub role: Role,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct TimerInfo {
    pub id: u64,
    pub label: String,
    pub deadline: Instant,
}

/// State shared between the assistant thread and the UI thread.
pub struct UiState {
    pub phase: Phase,
    pub status: String,
    /// Smoothed microphone level, 0..1.
    pub mic_level: f32,
    /// Recent levels, newest at the back — feeds the ring visualiser.
    pub levels: VecDeque<f32>,
    pub transcript: Vec<TranscriptEntry>,
    pub timers: Vec<TimerInfo>,
    pub error: Option<String>,
    pub wake_ready: bool,
}

pub type Shared = Arc<Mutex<UiState>>;

pub const LEVEL_HISTORY: usize = 96;

impl UiState {
    pub fn new() -> Shared {
        Arc::new(Mutex::new(UiState {
            phase: Phase::Boot,
            status: "Starting up…".into(),
            mic_level: 0.0,
            levels: VecDeque::from(vec![0.0; LEVEL_HISTORY]),
            transcript: Vec::new(),
            timers: Vec::new(),
            error: None,
            wake_ready: false,
        }))
    }
}

/// Small helpers so the assistant loop stays readable.
pub fn set_phase(shared: &Shared, phase: Phase, status: &str) {
    if let Ok(mut s) = shared.lock() {
        s.phase = phase;
        s.status = status.to_string();
    }
}

pub fn set_error(shared: &Shared, err: Option<String>) {
    if let Ok(mut s) = shared.lock() {
        s.error = err;
    }
}

pub fn push_transcript(shared: &Shared, role: Role, text: &str) {
    if let Ok(mut s) = shared.lock() {
        s.transcript.push(TranscriptEntry {
            role,
            text: text.to_string(),
        });
        let extra = s.transcript.len().saturating_sub(40);
        if extra > 0 {
            s.transcript.drain(..extra);
        }
    }
}

/// Append streamed text to the most recent assistant transcript entry
/// (creating it if the last entry is not an assistant one).
pub fn stream_to_transcript(shared: &Shared, delta: &str) {
    if let Ok(mut s) = shared.lock() {
        match s.transcript.last_mut() {
            Some(e) if e.role == Role::Assistant => e.text.push_str(delta),
            _ => s.transcript.push(TranscriptEntry {
                role: Role::Assistant,
                text: delta.to_string(),
            }),
        }
    }
}

/// Commands sent from the UI to the assistant loop.
#[derive(Debug, Clone, Copy)]
pub enum UiCommand {
    /// The user tapped the orb (or pressed space): wake / interrupt.
    Tap,
}

/// Events sent from background helpers to the assistant loop.
#[derive(Debug, Clone)]
pub enum AssistantEvent {
    TimerFired { label: String },
}
