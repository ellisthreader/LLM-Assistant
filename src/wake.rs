use crate::audio::capture::{self, BLOCK_SAMPLES};
use crate::config::Config;
use crate::state::UiState;
use anyhow::{anyhow, Context, Result};
use rustpotter::{
    Rustpotter, RustpotterConfig, SampleFormat, ScoreMode, WakewordRef, WakewordRefBuildFromFiles,
    WakewordSave,
};
use std::io::Write;
use std::path::Path;
use std::time::Duration;

pub struct WakeDetector {
    rp: Rustpotter,
    frame: usize,
    buf: Vec<i16>,
}

impl WakeDetector {
    /// Returns Ok(None) when no trained model exists yet — tap-to-wake still works.
    pub fn new(cfg: &Config) -> Result<Option<Self>> {
        if !Path::new(&cfg.wake.model_path).exists() {
            return Ok(None);
        }
        let mut rp_cfg = RustpotterConfig::default();
        rp_cfg.fmt.sample_rate = capture::TARGET_RATE as usize;
        rp_cfg.fmt.sample_format = SampleFormat::I16;
        rp_cfg.fmt.channels = 1;
        rp_cfg.detector.threshold = cfg.wake.threshold;
        rp_cfg.detector.avg_threshold = cfg.wake.avg_threshold;
        rp_cfg.detector.score_mode = ScoreMode::Max;
        rp_cfg.filters.gain_normalizer.enabled = true;

        let mut rp = Rustpotter::new(&rp_cfg).map_err(|e| anyhow!("rustpotter init: {e}"))?;
        rp.add_wakeword_from_file("wake", &cfg.wake.model_path)
            .map_err(|e| anyhow!("loading wake model {}: {e}", cfg.wake.model_path))?;
        let frame = rp.get_samples_per_frame();
        Ok(Some(Self {
            rp,
            frame,
            buf: Vec::with_capacity(frame * 2),
        }))
    }

    /// Feed captured samples; returns the detection score when the wake word fires.
    pub fn feed(&mut self, samples: &[i16]) -> Option<f32> {
        self.buf.extend_from_slice(samples);
        let mut hit = None;
        while self.buf.len() >= self.frame {
            let chunk: Vec<i16> = self.buf.drain(..self.frame).collect();
            if let Some(detection) = self.rp.process_samples(chunk) {
                log::info!(
                    "wake word detected (score {:.2}, avg {:.2})",
                    detection.score,
                    detection.avg_score
                );
                hit = Some(detection.score);
            }
        }
        hit
    }

    pub fn reset(&mut self) {
        self.rp.reset();
        self.buf.clear();
    }
}

/// Interactive CLI: record samples of the wake phrase and build the model.
pub fn train(cfg: &Config) -> Result<()> {
    let phrase = cfg.assistant.wake_phrase.clone();
    let n_samples = 6usize;
    println!();
    println!("── Wake word training ──────────────────────────────");
    println!("Phrase: “{phrase}”   Samples needed: {n_samples}");
    println!("Speak at the distance you'll normally use the assistant from.");
    println!("Vary your tone slightly between takes for a more robust model.");
    println!();

    let shared = UiState::new();
    let capture = capture::start(&cfg.audio, shared).context("starting microphone")?;
    std::fs::create_dir_all(&cfg.wake.sample_dir)?;

    let record_secs = 2.5f32;
    let blocks_needed = (record_secs * 1000.0 / 20.0) as usize;
    let mut files: Vec<String> = Vec::new();

    for i in 1..=n_samples {
        print!("[{i}/{n_samples}] Press ENTER, then say “{phrase}” … ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;

        capture::flush(&capture.rx);
        println!("    ● recording ({record_secs:.1}s)…");
        let mut samples: Vec<i16> = Vec::with_capacity(blocks_needed * BLOCK_SAMPLES);
        let mut got = 0usize;
        while got < blocks_needed {
            match capture.rx.recv_timeout(Duration::from_secs(2)) {
                Ok(block) => {
                    samples.extend_from_slice(&block);
                    got += 1;
                }
                Err(_) => return Err(anyhow!("microphone stopped delivering audio")),
            }
        }

        let trimmed = trim_silence(&samples);
        if trimmed.len() < BLOCK_SAMPLES * 10 {
            println!("    ✗ that take sounded silent — try again");
            return train_retry(cfg, i);
        }
        let path = format!("{}/sample_{i}.wav", cfg.wake.sample_dir);
        std::fs::write(&path, crate::audio::wav_from_samples(&trimmed))?;
        println!("    ✓ saved {path}");
        files.push(path);
    }

    println!();
    println!("Building model…");
    let wakeword = WakewordRef::new_from_sample_files(
        phrase.clone(),
        None,
        None,
        files,
        cfg.wake.mfcc_size,
    )
    .map_err(|e| anyhow!("building wake model: {e}"))?;
    wakeword
        .save_to_file(&cfg.wake.model_path)
        .map_err(|e| anyhow!("saving wake model: {e}"))?;
    println!("✓ Wake word model saved to {}", cfg.wake.model_path);
    println!("Run the assistant and say “{phrase}” to wake it.");
    Ok(())
}

fn train_retry(cfg: &Config, _failed_at: usize) -> Result<()> {
    println!("Restarting training from the top.\n");
    train(cfg)
}

/// Cut leading/trailing silence, keeping ~150 ms of padding.
fn trim_silence(samples: &[i16]) -> Vec<i16> {
    let block = BLOCK_SAMPLES;
    let levels: Vec<f32> = samples
        .chunks(block)
        .map(|c| {
            let sum: f64 = c.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum();
            ((sum / c.len().max(1) as f64) as f32).sqrt()
        })
        .collect();
    let peak = levels.iter().cloned().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return samples.to_vec();
    }
    let gate = (peak * 0.15).max(0.008);
    let first = levels.iter().position(|&l| l > gate);
    let last = levels.iter().rposition(|&l| l > gate);
    match (first, last) {
        (Some(f), Some(l)) if l >= f => {
            let pad = 8; // blocks ≈ 160 ms
            let start = f.saturating_sub(pad) * block;
            let end = ((l + 1 + pad) * block).min(samples.len());
            samples[start..end].to_vec()
        }
        _ => samples.to_vec(),
    }
}
