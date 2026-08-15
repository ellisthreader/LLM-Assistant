use crate::audio::capture::BLOCK_SAMPLES;
use crossbeam_channel::Receiver;
use std::time::{Duration, Instant};

const BLOCK_MS: u64 = (BLOCK_SAMPLES as u64) * 1000 / 16_000; // 20 ms
/// Audio kept from just before speech is detected.
const PREROLL_BLOCKS: usize = 12; // 240 ms
/// Consecutive loud blocks needed to count as speech onset.
const ONSET_BLOCKS: usize = 3; // 60 ms

pub enum ListenOutcome {
    Utterance(Vec<i16>),
    NoSpeech,
    Aborted,
}

fn block_rms(block: &[i16]) -> f32 {
    let sum: f64 = block.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum();
    ((sum / block.len().max(1) as f64) as f32).sqrt()
}

/// Record one utterance using an adaptive energy gate.
///
/// * waits up to `wait_for_speech` for the user to start talking
/// * stops after `end_silence` of quiet, or at `max_len`
/// * `should_abort` is polled so a UI tap can cancel listening
pub fn record_utterance(
    rx: &Receiver<Vec<i16>>,
    wait_for_speech: Duration,
    max_len: Duration,
    end_silence: Duration,
    mut should_abort: impl FnMut() -> bool,
) -> ListenOutcome {
    let mut noise_floor: f32 = 0.006;
    let mut preroll: Vec<Vec<i16>> = Vec::with_capacity(PREROLL_BLOCKS);
    let mut utterance: Vec<i16> = Vec::new();
    let mut in_speech = false;
    let mut onset_run = 0usize;
    let mut silence_run_ms: u64 = 0;
    let started = Instant::now();
    let mut speech_started_at: Option<Instant> = None;

    loop {
        if should_abort() {
            return ListenOutcome::Aborted;
        }
        if !in_speech && started.elapsed() > wait_for_speech {
            return ListenOutcome::NoSpeech;
        }
        if let Some(t0) = speech_started_at {
            if t0.elapsed() > max_len {
                return ListenOutcome::Utterance(utterance);
            }
        }

        let block = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(b) => b,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return ListenOutcome::Aborted
            }
        };

        let rms = block_rms(&block);
        let threshold = (noise_floor * 2.5).max(0.012);
        let loud = rms > threshold;

        if !in_speech {
            // Track the room's noise floor while nobody is talking.
            if !loud {
                noise_floor = noise_floor * 0.95 + rms * 0.05;
            }
            preroll.push(block);
            if preroll.len() > PREROLL_BLOCKS {
                preroll.remove(0);
            }
            onset_run = if loud { onset_run + 1 } else { 0 };
            if onset_run >= ONSET_BLOCKS {
                in_speech = true;
                speech_started_at = Some(Instant::now());
                for b in preroll.drain(..) {
                    utterance.extend_from_slice(&b);
                }
            }
        } else {
            utterance.extend_from_slice(&block);
            if loud {
                silence_run_ms = 0;
            } else {
                silence_run_ms += BLOCK_MS;
                if silence_run_ms >= end_silence.as_millis() as u64 {
                    // Trim most of the trailing silence, keep ~200 ms.
                    let keep_ms = 200u64.min(silence_run_ms);
                    let trim_blocks = ((silence_run_ms - keep_ms) / BLOCK_MS) as usize;
                    let trim_samples = (trim_blocks * BLOCK_SAMPLES).min(utterance.len());
                    utterance.truncate(utterance.len() - trim_samples);
                    return ListenOutcome::Utterance(utterance);
                }
            }
        }
    }
}
