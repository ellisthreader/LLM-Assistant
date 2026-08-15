use crate::config::AudioConfig;
use crate::state::{Shared, LEVEL_HISTORY};
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};

pub const TARGET_RATE: u32 = 16_000;
/// 20 ms of audio at 16 kHz — the block size handed to consumers.
pub const BLOCK_SAMPLES: usize = 320;

pub struct CaptureHandle {
    pub rx: Receiver<Vec<i16>>,
}

/// Drain anything queued on the capture channel (e.g. audio recorded while
/// the assistant was speaking, so it doesn't hear itself).
pub fn flush(rx: &Receiver<Vec<i16>>) {
    while rx.try_recv().is_ok() {}
}

pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices.filter_map(|d| d.name().ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// Linear-interpolation resampler that carries state across blocks.
struct Resampler {
    step: f64,
    pos: f64,
    last: f32,
}

impl Resampler {
    fn new(src_rate: u32) -> Self {
        Self {
            step: src_rate as f64 / TARGET_RATE as f64,
            pos: 0.0,
            last: 0.0,
        }
    }

    /// `input` is mono f32 at the source rate; appends 16 kHz samples to `out`.
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let n = input.len();
        if n == 0 {
            return;
        }
        // Virtual sample array: v[0] = last sample of previous block, v[1..=n] = input.
        let get = |i: usize| -> f32 {
            if i == 0 {
                self.last
            } else {
                input[i - 1]
            }
        };
        let mut pos = self.pos;
        while pos < n as f64 {
            let i = pos.floor() as usize;
            let frac = (pos - i as f64) as f32;
            out.push(get(i) * (1.0 - frac) + get(i + 1) * frac);
            pos += self.step;
        }
        self.pos = pos - n as f64;
        self.last = input[n - 1];
    }
}

pub fn start(cfg: &AudioConfig, shared: Shared) -> Result<CaptureHandle> {
    let (tx, rx) = bounded::<Vec<i16>>(256);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String>>();
    let device_filter = cfg.input_device.clone();

    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || run_capture_thread(device_filter, tx, shared, ready_tx))
        .context("spawning audio capture thread")?;

    match ready_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(Ok(name)) => {
            log::info!("capturing from input device: {name}");
            Ok(CaptureHandle { rx })
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(anyhow!("audio capture thread did not start in time")),
    }
}

fn run_capture_thread(
    device_filter: String,
    tx: Sender<Vec<i16>>,
    shared: Shared,
    ready_tx: std::sync::mpsc::Sender<Result<String>>,
) {
    let result = open_stream(&device_filter, tx, shared);
    match result {
        Ok((stream, name)) => {
            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(anyhow!("failed to start input stream: {e}")));
                return;
            }
            let _ = ready_tx.send(Ok(name));
            // Keep the stream alive forever; callbacks run on the audio backend thread.
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
        }
    }
}

fn open_stream(
    device_filter: &str,
    tx: Sender<Vec<i16>>,
    shared: Shared,
) -> Result<(cpal::Stream, String)> {
    let host = cpal::default_host();
    let device = if device_filter.trim().is_empty() {
        host.default_input_device()
            .ok_or_else(|| anyhow!("no default input device — is a microphone connected?"))?
    } else {
        let wanted = device_filter.to_lowercase();
        host.input_devices()
            .context("enumerating input devices")?
            .find(|d| {
                d.name()
                    .map(|n| n.to_lowercase().contains(&wanted))
                    .unwrap_or(false)
            })
            .ok_or_else(|| anyhow!("no input device matching '{device_filter}'"))?
    };
    let name = device.name().unwrap_or_else(|_| "unknown".into());

    let supported = device
        .default_input_config()
        .context("querying default input config")?;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    let channels = stream_config.channels as usize;
    let src_rate = stream_config.sample_rate.0;

    log::info!("input: {name} @ {src_rate} Hz, {channels} ch, {sample_format:?}");

    let mut ctx = CallbackCtx {
        resampler: Resampler::new(src_rate),
        mono: Vec::with_capacity(4096),
        resampled: Vec::with_capacity(4096),
        pending: Vec::with_capacity(BLOCK_SAMPLES * 2),
        tx,
        shared,
        smoothed: 0.0,
    };

    let err_fn = |e| log::error!("audio input error: {e}");
    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| ctx.handle(data, channels, |s| s),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| ctx.handle(data, channels, |s| s as f32 / 32768.0),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| ctx.handle(data, channels, |s| (s as f32 - 32768.0) / 32768.0),
            err_fn,
            None,
        )?,
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };
    Ok((stream, name))
}

struct CallbackCtx {
    resampler: Resampler,
    mono: Vec<f32>,
    resampled: Vec<f32>,
    pending: Vec<i16>,
    tx: Sender<Vec<i16>>,
    shared: Shared,
    smoothed: f32,
}

impl CallbackCtx {
    fn handle<T: Copy>(&mut self, data: &[T], channels: usize, to_f32: impl Fn(T) -> f32) {
        // Downmix to mono.
        self.mono.clear();
        for frame in data.chunks_exact(channels.max(1)) {
            let sum: f32 = frame.iter().map(|s| to_f32(*s)).sum();
            self.mono.push(sum / channels.max(1) as f32);
        }

        // Level meter (RMS of this callback, smoothed).
        let rms = (self.mono.iter().map(|s| s * s).sum::<f32>() / self.mono.len().max(1) as f32)
            .sqrt();
        let level = (rms * 8.0).min(1.0);
        self.smoothed = self.smoothed * 0.75 + level * 0.25;
        if let Ok(mut s) = self.shared.try_lock() {
            s.mic_level = self.smoothed;
            s.levels.push_back(self.smoothed);
            while s.levels.len() > LEVEL_HISTORY {
                s.levels.pop_front();
            }
        }

        // Resample to 16 kHz and ship fixed-size blocks.
        self.resampled.clear();
        self.resampler.process(&self.mono, &mut self.resampled);
        self.pending.extend(
            self.resampled
                .iter()
                .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16),
        );
        while self.pending.len() >= BLOCK_SAMPLES {
            let block: Vec<i16> = self.pending.drain(..BLOCK_SAMPLES).collect();
            // If the consumer is behind, drop the block — live audio must not back up.
            let _ = self.tx.try_send(block);
        }
    }
}
