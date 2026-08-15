use crate::state::{Phase, Role, Shared, UiCommand};
use egui::{
    Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2,
};
use std::time::Instant;

pub struct AssistantApp {
    shared: Shared,
    ui_tx: crossbeam_channel::Sender<UiCommand>,
    start: Instant,
}

struct Snapshot {
    phase: Phase,
    status: String,
    mic_level: f32,
    levels: Vec<f32>,
    last_user: Option<String>,
    last_assistant: Option<String>,
    timers: Vec<(String, u64)>,
    error: Option<String>,
}

impl AssistantApp {
    pub fn new(shared: Shared, ui_tx: crossbeam_channel::Sender<UiCommand>) -> Self {
        Self {
            shared,
            ui_tx,
            start: Instant::now(),
        }
    }

    fn snapshot(&self) -> Snapshot {
        let s = self.shared.lock().unwrap();
        let mut last_user = None;
        let mut last_assistant = None;
        for e in s.transcript.iter().rev() {
            match e.role {
                Role::Assistant if last_assistant.is_none() && last_user.is_none() => {
                    last_assistant = Some(e.text.clone())
                }
                Role::User if last_user.is_none() => last_user = Some(e.text.clone()),
                _ => {}
            }
            if last_user.is_some() {
                break;
            }
        }
        Snapshot {
            phase: s.phase,
            status: s.status.clone(),
            mic_level: s.mic_level,
            levels: s.levels.iter().cloned().collect(),
            last_user,
            last_assistant,
            timers: s
                .timers
                .iter()
                .map(|t| {
                    (
                        t.label.clone(),
                        t.deadline
                            .saturating_duration_since(Instant::now())
                            .as_secs(),
                    )
                })
                .collect(),
            error: s.error.clone(),
        }
    }
}

fn phase_color(phase: Phase) -> Color32 {
    match phase {
        Phase::Boot => Color32::from_rgb(120, 130, 150),
        Phase::Idle => Color32::from_rgb(56, 152, 224),
        Phase::Listening => Color32::from_rgb(64, 200, 120),
        Phase::Thinking => Color32::from_rgb(160, 130, 250),
        Phase::Speaking => Color32::from_rgb(245, 166, 70),
    }
}

fn with_alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

impl eframe::App for AssistantApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
        ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::None);

        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            let _ = self.ui_tx.send(UiCommand::Tap);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        let snap = self.snapshot();
        let t = self.start.elapsed().as_secs_f32();

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::from_rgb(9, 11, 19)))
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                // Whole screen is the tap target.
                let resp = ui.interact(rect, egui::Id::new("tap-anywhere"), Sense::click());
                if resp.clicked() {
                    let _ = self.ui_tx.send(UiCommand::Tap);
                }
                self.draw(ui, rect, &snap, t);
            });
    }
}

impl AssistantApp {
    fn draw(&self, ui: &mut egui::Ui, rect: Rect, snap: &Snapshot, t: f32) {
        let painter = ui.painter().clone();
        let w = rect.width();
        let h = rect.height();
        let color = phase_color(snap.phase);

        // ── Clock ──
        let now = chrono::Local::now();
        painter.text(
            Pos2::new(rect.center().x, rect.top() + h * 0.02),
            Align2::CENTER_TOP,
            now.format("%H:%M").to_string(),
            FontId::proportional(h * 0.085),
            Color32::from_rgb(214, 222, 235),
        );
        painter.text(
            Pos2::new(rect.center().x, rect.top() + h * 0.115),
            Align2::CENTER_TOP,
            now.format("%A %e %B").to_string(),
            FontId::proportional(h * 0.032),
            Color32::from_rgb(120, 132, 155),
        );

        // ── Timer chips (top right) ──
        let mut chip_y = rect.top() + h * 0.03;
        for (label, secs) in snap.timers.iter().take(4) {
            let text = format!("⏱ {}  {:02}:{:02}", label, secs / 60, secs % 60);
            let galley = painter.layout_no_wrap(
                text,
                FontId::proportional(h * 0.03),
                Color32::from_rgb(220, 226, 240),
            );
            let pad = Vec2::new(10.0, 6.0);
            let size = galley.size() + pad * 2.0;
            let chip = Rect::from_min_size(
                Pos2::new(rect.right() - size.x - h * 0.03, chip_y),
                size,
            );
            painter.rect_filled(chip, 8.0, Color32::from_rgba_unmultiplied(255, 255, 255, 14));
            painter.galley(chip.min + pad, galley, Color32::WHITE);
            chip_y += size.y + 6.0;
        }

        // ── Orb ──
        let center = Pos2::new(rect.center().x, rect.top() + h * 0.50);
        let base_r = h * 0.145;
        let breath = (t * 1.3).sin() * 0.5 + 0.5;
        let activity = match snap.phase {
            Phase::Listening => snap.mic_level,
            Phase::Speaking => {
                ((t * 7.3).sin() * 0.5 + 0.5) * 0.6 + ((t * 13.1).sin() * 0.5 + 0.5) * 0.4
            }
            _ => 0.0,
        };
        let r = base_r * (0.94 + 0.05 * breath) + base_r * 0.22 * activity;

        // Level ring: bars around the orb driven by recent audio levels.
        if matches!(snap.phase, Phase::Listening | Phase::Speaking) {
            let n = 64usize;
            let len = snap.levels.len();
            for k in 0..n {
                let idx = len.saturating_sub(n) + k;
                let lvl = snap.levels.get(idx).copied().unwrap_or(0.0);
                let ang = k as f32 / n as f32 * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
                let dir = Vec2::new(ang.cos(), ang.sin());
                let r0 = r * 1.22;
                let bar = 3.0 + lvl * base_r * 0.45;
                painter.line_segment(
                    [center + dir * r0, center + dir * (r0 + bar)],
                    Stroke::new(2.5, with_alpha(color, 90)),
                );
            }
        }

        // Glow layers, outer → inner (many low-alpha layers = smooth falloff).
        for i in (0..14).rev() {
            let k = i as f32 / 13.0;
            let radius = r * (1.0 + 0.48 * k);
            let alpha = (6.0 + (1.0 - k) * (1.0 - k) * 22.0) as u8;
            painter.circle_filled(center, radius, with_alpha(color, alpha));
        }
        painter.circle_filled(center, r * 0.82, with_alpha(color, 200));
        painter.circle_filled(
            center,
            r * 0.40,
            Color32::from_rgba_unmultiplied(255, 255, 255, 90),
        );

        // Thinking: three orbiting dots.
        if snap.phase == Phase::Thinking {
            for k in 0..3 {
                let ang = t * 2.8 + k as f32 * std::f32::consts::TAU / 3.0;
                let pos = center + Vec2::new(ang.cos(), ang.sin()) * r * 1.28;
                painter.circle_filled(pos, 5.0, with_alpha(Color32::WHITE, 170));
            }
        }

        // ── Status line ──
        painter.text(
            Pos2::new(rect.center().x, center.y + base_r * 1.75),
            Align2::CENTER_TOP,
            &snap.status,
            FontId::proportional(h * 0.042),
            Color32::from_rgb(168, 180, 200),
        );

        // ── Transcript (bottom) ──
        let transcript_rect = Rect::from_min_max(
            Pos2::new(rect.left() + w * 0.07, rect.bottom() - h * 0.20),
            Pos2::new(rect.right() - w * 0.07, rect.bottom() - h * 0.02),
        );
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(transcript_rect), |ui| {
            ui.vertical_centered(|ui| {
                if let Some(user) = &snap.last_user {
                    ui.label(
                        RichText::new(format!("“{}”", truncate(user, 120)))
                            .font(FontId::proportional(h * 0.032))
                            .italics()
                            .color(Color32::from_rgb(122, 134, 158)),
                    );
                }
                if let Some(assistant) = &snap.last_assistant {
                    ui.label(
                        RichText::new(truncate(assistant, 260))
                            .font(FontId::proportional(h * 0.038))
                            .color(Color32::from_rgb(222, 230, 243)),
                    );
                }
            });
        });

        // ── Error banner ──
        if let Some(err) = &snap.error {
            let galley = painter.layout(
                err.clone(),
                FontId::proportional(h * 0.028),
                Color32::from_rgb(255, 214, 214),
                w * 0.8,
            );
            let pad = Vec2::new(12.0, 8.0);
            let size = galley.size() + pad * 2.0;
            let banner = Rect::from_min_size(
                Pos2::new(
                    rect.center().x - size.x / 2.0,
                    rect.bottom() - size.y - h * 0.015,
                ),
                size,
            );
            painter.rect_filled(banner, 10.0, Color32::from_rgba_unmultiplied(120, 30, 30, 235));
            painter.galley(banner.min + pad, galley, Color32::WHITE);
        }
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let tail: String = text
        .chars()
        .rev()
        .take(max_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}
