use crate::audio::capture::{self, CaptureHandle};
use crate::audio::playback::{ChimeKind, Player};
use crate::audio::{vad, wav_from_samples};
use crate::config::Config;
use crate::openai::OpenAi;
use crate::state::{
    push_transcript, set_error, set_phase, stream_to_transcript, AssistantEvent, Phase, Role,
    Shared, UiCommand,
};
use crate::tools::{self, TimerManager, ToolCtx};
use crate::wake::WakeDetector;
use anyhow::Result;
use crossbeam_channel::Receiver;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

enum Wake {
    Voice,
    Tap,
    Timer(String),
}

enum TurnEnd {
    Spoke,
    Interrupted,
    NoInput,
}

pub fn spawn(
    cfg: Config,
    shared: Shared,
    capture: CaptureHandle,
    player: Player,
    ui_cmd_rx: Receiver<UiCommand>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("assistant".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("tokio runtime");

            let (event_tx, event_rx) = crossbeam_channel::unbounded::<AssistantEvent>();
            let timers = TimerManager::new(event_tx, shared.clone());

            let openai = cfg.api_key().map(|key| OpenAi::new(cfg.openai.clone(), key));
            if openai.is_none() {
                set_error(
                    &shared,
                    Some(
                        "No OpenAI API key. Set OPENAI_API_KEY or add it to config.toml, then restart."
                            .into(),
                    ),
                );
            }

            let wake = match WakeDetector::new(&cfg) {
                Ok(w) => {
                    if w.is_none() {
                        log::warn!(
                            "no wake model at {} — run `desk-assistant train-wake` (tap-to-talk still works)",
                            cfg.wake.model_path
                        );
                    }
                    w
                }
                Err(e) => {
                    log::error!("wake detector failed: {e}");
                    set_error(&shared, Some(format!("Wake word disabled: {e}")));
                    None
                }
            };
            if let Ok(mut s) = shared.lock() {
                s.wake_ready = wake.is_some();
            }

            let tool_ctx = ToolCtx {
                http: reqwest::Client::new(),
                home_location: cfg.assistant.location.clone(),
                timers,
            };

            let mut assistant = Assistant {
                cfg,
                shared,
                capture,
                player,
                ui_cmd_rx,
                event_rx,
                openai,
                tool_ctx,
                wake,
                history: Vec::new(),
            };
            assistant.run(&rt);
        })
        .expect("spawning assistant thread")
}

struct Assistant {
    cfg: Config,
    shared: Shared,
    capture: CaptureHandle,
    player: Player,
    ui_cmd_rx: Receiver<UiCommand>,
    event_rx: Receiver<AssistantEvent>,
    openai: Option<OpenAi>,
    tool_ctx: ToolCtx,
    wake: Option<WakeDetector>,
    history: Vec<Value>,
}

impl Assistant {
    fn run(&mut self, rt: &tokio::runtime::Runtime) {
        log::info!("assistant loop running");
        loop {
            match self.wait_for_wake() {
                Wake::Voice | Wake::Tap => self.conversation(rt),
                Wake::Timer(label) => self.announce_timer(rt, &label),
            }
        }
    }

    fn idle_status(&self) -> String {
        if self.wake.is_some() {
            format!("Say “{}” — or tap", self.cfg.assistant.wake_phrase)
        } else {
            "Tap the orb to talk".into()
        }
    }

    fn wait_for_wake(&mut self) -> Wake {
        set_phase(&self.shared, Phase::Idle, &self.idle_status());
        if let Some(w) = &mut self.wake {
            w.reset();
        }
        capture::flush(&self.capture.rx);
        loop {
            while let Ok(cmd) = self.ui_cmd_rx.try_recv() {
                match cmd {
                    UiCommand::Tap => return Wake::Tap,
                }
            }
            while let Ok(ev) = self.event_rx.try_recv() {
                match ev {
                    AssistantEvent::TimerFired { label } => return Wake::Timer(label),
                }
            }
            match self.capture.rx.recv_timeout(Duration::from_millis(150)) {
                Ok(block) => {
                    if let Some(w) = &mut self.wake {
                        if w.feed(&block).is_some() {
                            return Wake::Voice;
                        }
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    std::thread::sleep(Duration::from_millis(150));
                }
            }
        }
    }

    /// True if a tap is waiting; consumes it.
    fn tap_pending(&self) -> bool {
        matches!(self.ui_cmd_rx.try_recv(), Ok(UiCommand::Tap))
    }

    fn conversation(&mut self, rt: &tokio::runtime::Runtime) {
        let mut first = true;
        loop {
            self.player.stop();
            self.player.chime(ChimeKind::Listen);
            std::thread::sleep(Duration::from_millis(350));
            capture::flush(&self.capture.rx);
            set_phase(&self.shared, Phase::Listening, "Listening…");

            let wait = if first {
                Duration::from_secs(8)
            } else {
                Duration::from_secs_f32(self.cfg.assistant.follow_up_window_secs)
            };
            let ui_rx = self.ui_cmd_rx.clone();
            let outcome = vad::record_utterance(
                &self.capture.rx,
                wait,
                Duration::from_secs_f32(self.cfg.assistant.max_utterance_secs),
                Duration::from_millis(self.cfg.assistant.end_silence_ms),
                move || matches!(ui_rx.try_recv(), Ok(UiCommand::Tap)),
            );

            match outcome {
                vad::ListenOutcome::Utterance(samples)
                    if samples.len() >= capture::TARGET_RATE as usize / 2 =>
                {
                    match self.turn(rt, samples) {
                        TurnEnd::Interrupted => {
                            // The user cut in — listen again right away.
                            first = true;
                            continue;
                        }
                        TurnEnd::Spoke | TurnEnd::NoInput => {
                            first = false;
                            continue; // follow-up window
                        }
                    }
                }
                vad::ListenOutcome::Utterance(_) | vad::ListenOutcome::NoSpeech => {
                    if first {
                        set_phase(&self.shared, Phase::Idle, "Didn't hear anything");
                        std::thread::sleep(Duration::from_millis(600));
                    }
                    return;
                }
                vad::ListenOutcome::Aborted => return,
            }
        }
    }

    fn turn(&mut self, rt: &tokio::runtime::Runtime, samples: Vec<i16>) -> TurnEnd {
        let Some(openai) = self.openai.clone() else {
            set_phase(&self.shared, Phase::Idle, "No API key configured");
            return TurnEnd::NoInput;
        };

        set_phase(&self.shared, Phase::Thinking, "Transcribing…");
        let wav = wav_from_samples(&samples);
        let text = match rt.block_on(openai.transcribe(wav)) {
            Ok(t) => t,
            Err(e) => {
                log::error!("stt: {e}");
                set_error(&self.shared, Some(format!("{e}")));
                set_phase(&self.shared, Phase::Idle, "Couldn't reach OpenAI");
                return TurnEnd::NoInput;
            }
        };
        if text.is_empty() {
            set_phase(&self.shared, Phase::Idle, "Didn't catch that");
            return TurnEnd::NoInput;
        }
        set_error(&self.shared, None);
        log::info!("user said: {text}");
        push_transcript(&self.shared, Role::User, &text);
        self.history.push(json!({ "role": "user", "content": text }));
        self.trim_history();

        set_phase(&self.shared, Phase::Thinking, "Thinking…");
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        let spoke_anything = rt.block_on(async {
            // Watches for taps for the whole generate + speak window (barge-in).
            let poller = {
                let cancel = cancel.clone();
                let done = done.clone();
                let player = self.player.clone();
                let ui_rx = self.ui_cmd_rx.clone();
                tokio::spawn(async move {
                    while !done.load(Ordering::SeqCst) && !cancel.load(Ordering::SeqCst) {
                        if let Ok(UiCommand::Tap) = ui_rx.try_recv() {
                            cancel.store(true, Ordering::SeqCst);
                            player.stop();
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(80)).await;
                    }
                })
            };

            let spoke = match self.agent_loop(&openai, &cancel).await {
                Ok(spoke) => spoke,
                Err(e) => {
                    log::error!("chat: {e}");
                    set_error(&self.shared, Some(format!("{e}")));
                    self.say_canned(&openai, &cancel, "Sorry, I hit a problem talking to OpenAI.")
                        .await;
                    true
                }
            };

            while self.player.is_busy() && !cancel.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            done.store(true, Ordering::SeqCst);
            let _ = poller.await;
            spoke
        });

        if cancel.load(Ordering::SeqCst) {
            return TurnEnd::Interrupted;
        }
        if spoke_anything {
            TurnEnd::Spoke
        } else {
            set_phase(&self.shared, Phase::Idle, "…");
            TurnEnd::NoInput
        }
    }

    /// Runs the model (with tool rounds), streaming sentences to TTS as they form.
    /// Returns whether any speech was queued.
    async fn agent_loop(&mut self, openai: &OpenAi, cancel: &Arc<AtomicBool>) -> Result<bool> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let worker = tokio::spawn(tts_worker(
            openai.clone(),
            self.player.clone(),
            rx,
            cancel.clone(),
            self.shared.clone(),
        ));

        let mut splitter = SentenceSplitter::default();
        let tool_defs = tools::tool_defs();
        let mut sentences_sent = 0usize;
        let mut rounds = 0usize;

        loop {
            let messages = self.build_messages();
            let shared = self.shared.clone();
            let tx2 = tx.clone();
            let (content, tool_calls) = openai
                .chat_stream(&messages, &tool_defs, cancel, |delta| {
                    stream_to_transcript(&shared, delta);
                    for sentence in splitter.push(delta) {
                        sentences_sent += 1;
                        let _ = tx2.send(sentence);
                    }
                })
                .await?;

            if cancel.load(Ordering::SeqCst) {
                break;
            }
            if tool_calls.is_empty() {
                if !content.trim().is_empty() {
                    self.history
                        .push(json!({ "role": "assistant", "content": content }));
                }
                break;
            }

            let tc_json: Vec<Value> = tool_calls
                .iter()
                .map(|t| {
                    json!({
                        "id": t.id,
                        "type": "function",
                        "function": { "name": t.name, "arguments": t.arguments }
                    })
                })
                .collect();
            let mut amsg = json!({ "role": "assistant", "tool_calls": tc_json });
            if !content.trim().is_empty() {
                amsg["content"] = json!(content);
            }
            self.history.push(amsg);

            for call in &tool_calls {
                set_phase(&self.shared, Phase::Thinking, friendly_tool_status(&call.name));
                log::info!("tool call: {}({})", call.name, call.arguments);
                let output = tools::execute(&self.tool_ctx, &call.name, &call.arguments).await;
                log::info!("tool result: {output}");
                self.history.push(json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output
                }));
            }

            rounds += 1;
            if rounds >= 4 {
                log::warn!("too many tool rounds, stopping");
                break;
            }
        }

        if let Some(rest) = splitter.finish() {
            sentences_sent += 1;
            let _ = tx.send(rest);
        }
        if sentences_sent == 0 && !cancel.load(Ordering::SeqCst) {
            // The model produced nothing speakable — stay honest and audible.
            let _ = tx.send("Sorry, I got stuck on that one. Mind trying again?".to_string());
            sentences_sent += 1;
        }
        drop(tx);
        let _ = worker.await;
        Ok(sentences_sent > 0)
    }

    async fn say_canned(&self, openai: &OpenAi, cancel: &Arc<AtomicBool>, text: &str) {
        push_transcript(&self.shared, Role::Assistant, text);
        if cancel.load(Ordering::SeqCst) {
            return;
        }
        if let Ok(bytes) = openai.tts(text).await {
            set_phase(&self.shared, Phase::Speaking, "Speaking…");
            self.player.play(bytes);
        }
    }

    fn announce_timer(&mut self, rt: &tokio::runtime::Runtime, label: &str) {
        let text = format!("Your {label} is done!");
        log::info!("{text}");
        self.player.chime(ChimeKind::Timer);
        self.player.chime(ChimeKind::Timer);
        push_transcript(&self.shared, Role::Assistant, &text);
        set_phase(&self.shared, Phase::Speaking, "Timer finished");
        if let Some(openai) = self.openai.clone() {
            if let Ok(bytes) = rt.block_on(openai.tts(&text)) {
                self.player.play(bytes);
            }
        }
        self.player.wait_idle(|| self.tap_pending());
        self.player.stop();
    }

    fn build_messages(&self) -> Vec<Value> {
        let now = chrono::Local::now();
        let a = &self.cfg.assistant;
        let system = format!(
            "You are {name}, a voice assistant living on a small screen on the user's desk, \
             running on a Raspberry Pi and powered by OpenAI models.\n\
             Current date and time: {now}. The user's home location is {loc}.\n\
             \n\
             Everything you write is spoken aloud by a text-to-speech voice, so:\n\
             - Plain conversational sentences only. No markdown, no bullet points, no emojis, \
               no stage directions, no URLs.\n\
             - Be brief: one to three short sentences unless the user asks for detail.\n\
             - Read numbers, dates and units the way a person would say them.\n\
             \n\
             Things you can genuinely do, through your tools: report current weather and a \
             three-day forecast for any town; set, cancel and list countdown timers that chime; \
             change your speaker volume; and answer general-knowledge questions from your training.\n\
             Things you cannot do: browse the internet or fetch live news, play music, control \
             smart-home devices, send messages or emails, see through a camera, or remember \
             conversations after a restart. When asked for one of these, say plainly that you \
             can't do it yet, in one sentence, and offer the nearest thing you can do.\n\
             Be honest about uncertainty: if you might be wrong or your knowledge may be out of \
             date, say so briefly. Never invent facts, prices, or news.\n\
             You have a warm, capable, lightly witty personality — an excellent personal assistant, \
             never robotic, never sycophantic.",
            name = a.name,
            now = now.format("%A %e %B %Y, %H:%M"),
            loc = a.location,
        );
        let mut messages = vec![json!({ "role": "system", "content": system })];
        messages.extend(self.history.iter().cloned());
        messages
    }

    fn trim_history(&mut self) {
        const MAX: usize = 24;
        if self.history.len() <= MAX {
            return;
        }
        let mut drop = self.history.len() - MAX;
        // Never cut between an assistant tool call and its tool results.
        while drop < self.history.len() && self.history[drop]["role"] != "user" {
            drop += 1;
        }
        self.history.drain(..drop);
    }
}

fn friendly_tool_status(tool: &str) -> &'static str {
    match tool {
        "get_weather" => "Checking the weather…",
        "set_timer" => "Setting a timer…",
        "cancel_timer" | "list_timers" => "Checking timers…",
        "set_volume" => "Adjusting volume…",
        _ => "Working on it…",
    }
}

/// Fetches TTS audio for each queued sentence, in order, and hands it to the player.
async fn tts_worker(
    openai: OpenAi,
    player: Player,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    cancel: Arc<AtomicBool>,
    shared: Shared,
) {
    let mut first = true;
    while let Some(text) = rx.recv().await {
        if cancel.load(Ordering::SeqCst) {
            continue; // keep draining so the sender never blocks
        }
        match openai.tts(&text).await {
            Ok(bytes) => {
                if cancel.load(Ordering::SeqCst) {
                    continue;
                }
                if first {
                    set_phase(&shared, Phase::Speaking, "Speaking…");
                    first = false;
                }
                player.play(bytes);
            }
            Err(e) => {
                log::error!("tts: {e}");
                set_error(&shared, Some(format!("{e}")));
            }
        }
    }
}

/// Accumulates streamed text and emits it one speakable sentence at a time.
#[derive(Default)]
struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        loop {
            let mut cut_at: Option<usize> = None;
            let mut prev_len = 0usize;
            for (i, ch) in self.buf.char_indices() {
                let end = i + ch.len_utf8();
                if ch == '\n' {
                    if self.buf[..i].trim().len() >= 2 {
                        cut_at = Some(end);
                        break;
                    }
                } else if matches!(ch, '.' | '!' | '?') && prev_len >= 20 {
                    // Only cut when the terminator is followed by whitespace —
                    // keeps "3.14" and "Dr. Smith"-style tokens intact.
                    if self.buf[end..].chars().next().is_some_and(|c| c.is_whitespace()) {
                        cut_at = Some(end);
                        break;
                    }
                }
                prev_len += 1;
            }
            match cut_at {
                Some(end) => {
                    let sentence: String = self.buf.drain(..end).collect();
                    let clean = sanitize_for_tts(&sentence);
                    if !clean.is_empty() {
                        out.push(clean);
                    }
                }
                None => break,
            }
        }
        out
    }

    fn finish(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buf);
        let clean = sanitize_for_tts(&rest);
        (!clean.is_empty()).then_some(clean)
    }
}

fn sanitize_for_tts(text: &str) -> String {
    text.chars()
        .filter(|c| !matches!(c, '*' | '#' | '`' | '_' | '~'))
        .collect::<String>()
        .trim()
        .to_string()
}
