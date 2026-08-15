use anyhow::{anyhow, Context, Result};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

enum PlayerCmd {
    /// Queue an encoded audio clip (wav/mp3) for playback.
    Play(Vec<u8>),
    /// Short UI sound.
    Chime(ChimeKind),
    /// Stop everything and clear the queue.
    Stop,
}

#[derive(Clone, Copy)]
pub enum ChimeKind {
    Listen,
    Timer,
}

/// Handle to the audio output thread. Cheap to clone.
#[derive(Clone)]
pub struct Player {
    tx: crossbeam_channel::Sender<PlayerCmd>,
    busy: Arc<AtomicBool>,
}

impl Player {
    pub fn new() -> Result<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<PlayerCmd>();
        let busy = Arc::new(AtomicBool::new(false));
        let busy2 = busy.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

        std::thread::Builder::new()
            .name("audio-playback".into())
            .spawn(move || playback_thread(rx, busy2, ready_tx))
            .context("spawning playback thread")?;

        match ready_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(Self { tx, busy }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow!("audio output thread did not start in time")),
        }
    }

    pub fn play(&self, encoded: Vec<u8>) {
        self.busy.store(true, Ordering::SeqCst);
        let _ = self.tx.send(PlayerCmd::Play(encoded));
    }

    pub fn chime(&self, kind: ChimeKind) {
        let _ = self.tx.send(PlayerCmd::Chime(kind));
    }

    pub fn stop(&self) {
        let _ = self.tx.send(PlayerCmd::Stop);
    }

    /// True while queued speech is still playing.
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Block until all queued audio has finished (checks a caller-supplied
    /// cancel flag so interruptions stay responsive).
    pub fn wait_idle(&self, cancelled: impl Fn() -> bool) {
        loop {
            if !self.is_busy() || cancelled() {
                return;
            }
            std::thread::sleep(Duration::from_millis(40));
        }
    }
}

fn playback_thread(
    rx: crossbeam_channel::Receiver<PlayerCmd>,
    busy: Arc<AtomicBool>,
    ready_tx: std::sync::mpsc::Sender<Result<()>>,
) {
    let stream = match rodio::OutputStream::try_default() {
        Ok(pair) => pair,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow!("no audio output device: {e}")));
            return;
        }
    };
    let (_stream, handle) = stream;
    let mut sink = match rodio::Sink::try_new(&handle) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow!("failed to create audio sink: {e}")));
            return;
        }
    };
    let _ = ready_tx.send(Ok(()));

    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(PlayerCmd::Play(bytes)) => match rodio::Decoder::new(Cursor::new(bytes)) {
                Ok(source) => {
                    sink.append(source);
                    busy.store(true, Ordering::SeqCst);
                }
                Err(e) => log::error!("could not decode audio clip: {e}"),
            },
            Ok(PlayerCmd::Chime(kind)) => {
                use rodio::source::{SineWave, Source};
                let (f1, f2) = match kind {
                    ChimeKind::Listen => (660.0, 880.0),
                    ChimeKind::Timer => (880.0, 660.0),
                };
                let a = SineWave::new(f1)
                    .take_duration(Duration::from_millis(90))
                    .amplify(0.20)
                    .fade_in(Duration::from_millis(10));
                let b = SineWave::new(f2)
                    .take_duration(Duration::from_millis(120))
                    .amplify(0.20)
                    .fade_in(Duration::from_millis(5));
                sink.append(a);
                sink.append(b);
                busy.store(true, Ordering::SeqCst);
            }
            Ok(PlayerCmd::Stop) => {
                sink.stop();
                // Recreate the sink — safest across rodio versions.
                if let Ok(new_sink) = rodio::Sink::try_new(&handle) {
                    sink = new_sink;
                }
                busy.store(false, Ordering::SeqCst);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
        }
        if sink.empty() {
            busy.store(false, Ordering::SeqCst);
        }
    }
}
