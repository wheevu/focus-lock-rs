use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

use fancam_core::{
    detection::{draw_boxes, Detector, FaceIdentifier},
    rendering::{crop_fancam, letterbox_passthrough},
    tracking::BiasTracker,
    video::{to_grayscale, transcode, RgbFrame},
};

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "focus-lock",
    version,
    about = "High-performance automated fancam generator",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Phase 1 smoke-test: read a video, convert to grayscale, save.
    Gray {
        /// Input video path
        #[arg(short, long)]
        input: PathBuf,

        /// Output video path
        #[arg(short, long, default_value = "gray.mp4")]
        output: PathBuf,
    },

    /// Phase 2: draw bounding boxes around all detected persons.
    Detect {
        /// Input video path
        #[arg(short, long)]
        input: PathBuf,

        /// YOLOv8n ONNX model path
        #[arg(long, default_value = "yolov8n.onnx")]
        model: PathBuf,

        /// Output video path
        #[arg(short, long, default_value = "detected.mp4")]
        output: PathBuf,
    },

    /// Phase 3 + 4: generate a stabilised 9:16 fancam for the target identity.
    Fancam {
        /// Input video path
        #[arg(short, long)]
        video: PathBuf,

        /// Reference face image (your bias)
        #[arg(short, long)]
        bias: PathBuf,

        /// Output fancam path
        #[arg(short, long, default_value = "fancam.mp4")]
        output: PathBuf,

        /// YOLOv8n ONNX model path
        #[arg(long, default_value = "yolov8n.onnx")]
        yolo_model: PathBuf,

        /// ArcFace ONNX model path
        #[arg(long, default_value = "arcface.onnx")]
        face_model: PathBuf,

        /// Cosine similarity threshold (0–1)
        #[arg(long, default_value_t = 0.6)]
        threshold: f32,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Respect RUST_LOG; default to info
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Gray { input, output } => cmd_gray(input, output),
        Commands::Detect {
            input,
            model,
            output,
        } => cmd_detect(input, model, output),
        Commands::Fancam {
            video,
            bias,
            output,
            yolo_model,
            face_model,
            threshold,
        } => cmd_fancam(video, bias, output, yolo_model, face_model, threshold),
    }
}

// ── Phase 1: grayscale ────────────────────────────────────────────────────────

fn cmd_gray(input: PathBuf, output: PathBuf) -> Result<()> {
    info!("Phase 1 — grayscale conversion");
    info!("  input  : {}", input.display());
    info!("  output : {}", output.display());

    let pb = spinner("Converting to grayscale…");
    let pb2 = pb.clone();

    transcode(input, &output, move |frame: &mut RgbFrame| {
        to_grayscale(frame);
        pb2.tick();
    })
    .context("grayscale transcode failed")?;

    pb.finish_with_message("Done.");
    Ok(())
}

// ── Phase 2: person detection ─────────────────────────────────────────────────

fn cmd_detect(input: PathBuf, model: PathBuf, output: PathBuf) -> Result<()> {
    info!("Phase 2 — person detection");

    let mut detector = Detector::load(&model)
        .with_context(|| format!("failed to load model: {}", model.display()))?;

    let pb = spinner("Detecting persons…");
    let pb2 = pb.clone();

    transcode(input, &output, move |frame: &mut RgbFrame| {
        pb2.tick();
        match detector.detect(frame) {
            Ok(boxes) => {
                draw_boxes(frame, &boxes, [0, 255, 0]);
            }
            Err(e) => tracing::warn!("detection error: {e}"),
        }
    })
    .context("detection transcode failed")?;

    pb.finish_with_message("Done.");
    Ok(())
}

// ── Phase 3 + 4: full fancam pipeline ────────────────────────────────────────

fn cmd_fancam(
    video: PathBuf,
    bias: PathBuf,
    output: PathBuf,
    yolo_model: PathBuf,
    face_model: PathBuf,
    _threshold: f32,
) -> Result<()> {
    info!("Fancam pipeline");
    info!("  video      : {}", video.display());
    info!("  bias image : {}", bias.display());
    info!("  output     : {}", output.display());

    let mut detector = Detector::load(&yolo_model)
        .with_context(|| format!("failed to load YOLO model: {}", yolo_model.display()))?;

    let mut identifier = FaceIdentifier::load(&face_model, &bias).with_context(|| {
        format!(
            "failed to load face model or embed reference: {}",
            bias.display()
        )
    })?;

    let mut tracker = BiasTracker::new();
    let pb = spinner("Generating fancam…");
    let pb2 = pb.clone();

    transcode(video, &output, move |frame: &mut RgbFrame| {
        pb2.tick();

        // Throttle recognition when locked on target
        let detection = if tracker.should_run_recognition() {
            match detector.detect(frame) {
                Ok(persons) => match identifier.identify(frame, &persons) {
                    Ok(found) => found,
                    Err(e) => {
                        tracing::warn!("face ID error: {e}");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("detection error: {e}");
                    None
                }
            }
        } else {
            // Predict-only frame — pass None so tracker runs Kalman predict
            None
        };

        let camera = tracker.update(detection);

        // Render the 9:16 crop (or letterbox fallback if target is lost)
        let cropped = match camera {
            Some(ref state) => crop_fancam(frame, state),
            None => letterbox_passthrough(frame),
        };

        match cropped {
            Ok(out) => {
                frame.data = out.data;
                frame.width = out.width;
                frame.height = out.height;
            }
            Err(e) => tracing::warn!("render error: {e}"),
        }
    })
    .context("fancam transcode failed")?;

    pb.finish_with_message("Fancam saved.");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg} [{elapsed_precise}]")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}
