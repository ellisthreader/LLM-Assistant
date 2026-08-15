mod assistant;
mod audio;
mod config;
mod openai;
mod state;
mod tools;
mod ui;
mod wake;

use anyhow::Result;
use config::Config;
use state::UiState;

const CONFIG_PATH: &str = "config.toml";

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "info,zbus=warn,tracing=warn,winit=warn,calloop=warn,polling=warn,async_io=warn,arboard=warn",
    ))
    .init();

    let arg = std::env::args().nth(1).unwrap_or_default();
    match arg.as_str() {
        "devices" => {
            println!("Input devices:");
            for name in audio::capture::list_input_devices() {
                println!("  - {name}");
            }
            Ok(())
        }
        "train-wake" => {
            let cfg = Config::load_or_create(CONFIG_PATH)?;
            wake::train(&cfg)
        }
        "help" | "--help" | "-h" => {
            println!("desk-assistant            run the assistant (default)");
            println!("desk-assistant train-wake record your wake word and build the model");
            println!("desk-assistant devices    list microphone devices");
            Ok(())
        }
        _ => run(),
    }
}

fn run() -> Result<()> {
    let cfg = Config::load_or_create(CONFIG_PATH)?;
    let shared = UiState::new();

    let player = audio::playback::Player::new()?;
    let capture = audio::capture::start(&cfg.audio, shared.clone())?;
    let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<state::UiCommand>();

    assistant::spawn(cfg.clone(), shared.clone(), capture, player, ui_rx);

    let mut viewport = egui::ViewportBuilder::default()
        .with_title(cfg.assistant.name.clone())
        .with_inner_size([cfg.ui.width, cfg.ui.height]);
    if cfg.ui.fullscreen {
        viewport = viewport.with_fullscreen(true);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "desk-assistant",
        options,
        Box::new(move |_cc| Ok(Box::new(ui::AssistantApp::new(shared, ui_tx)))),
    )
    .map_err(|e| anyhow::anyhow!("UI error: {e}"))
}
