//! CLI for focus-lock-rs
//!
//! Command-line interface for the high-performance automated fancam generator.

#![warn(
    clippy::all,
    clippy::pedantic,
    missing_docs,
    rust_2018_idioms,
    unused_qualifications
)]

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

use fancam_core::{
    detection::{Detector, FaceIdentifier, draw_boxes},
    mode::ProcessingMode,
    pipeline::{Analyzer, OfflinePrepassProgress, Pipeline, Renderer},
    plan::CropPlanV1,
    runtime::OrtConfig,
    video::{
        RgbFrame, for_each_rgb_frame, total_frames, transcode,
        transcode_with_progress_staged_mode_fallible,
    },
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
    /// Phase 2: draw bounding boxes around all detected persons.
    #[command(hide = true)]
    Detect {
        /// Input video path
        #[arg(short, long)]
        input: PathBuf,

        /// `YOLOv8n` ONNX model path
        #[arg(long, default_value = "models/yolov8n.onnx")]
        model: PathBuf,

        /// Output video path
        #[arg(short, long, default_value = "detected.mp4")]
        output: PathBuf,
    },

    /// Run environment diagnostics to help debug setup issues.
    Doctor,

    /// Inspect identity match potential: sample frames, detect persons,
    /// compute similarity scores against a reference face image.
    InspectIdentity {
        /// Input video path
        #[arg(short, long)]
        video: PathBuf,

        /// Reference face image (your bias)
        #[arg(short, long)]
        bias: PathBuf,

        /// `YOLOv8n` ONNX model path
        #[arg(long, default_value = "models/yolov8n.onnx")]
        yolo_model: PathBuf,

        /// `ArcFace` ONNX model path (e.g. `models/w600k_mbf.onnx`)
        #[arg(long, default_value = "models/w600k_mbf.onnx")]
        face_model: PathBuf,

        /// Cosine similarity threshold (0–1)
        #[arg(long, default_value_t = 0.6)]
        threshold: f32,

        /// Face detection model (SCRFD, e.g. `models/det_500m.onnx`).
        /// When provided, candidate face crops use detected faces instead of heuristic head regions.
        #[arg(long)]
        face_det_model: Option<PathBuf>,

        /// Sample every N frames
        #[arg(
            long,
            default_value_t = 30,
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        sample_every: u64,

        /// Maximum frames to process
        #[arg(long, default_value_t = 300)]
        max_frames: u64,
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

        /// `YOLOv8n` ONNX model path
        #[arg(long, default_value = "models/yolov8n.onnx")]
        yolo_model: PathBuf,

        /// `ArcFace` ONNX model path (e.g. `models/w600k_mbf.onnx`)
        #[arg(long, default_value = "models/w600k_mbf.onnx")]
        face_model: PathBuf,

        /// Optional explicit identity model path (overrides --face-model)
        #[arg(long)]
        identity_model: Option<PathBuf>,

        /// Cosine similarity threshold (0–1)
        #[arg(long, default_value_t = 0.6)]
        threshold: f32,

        /// Processing mode: fast, balanced, or quality
        #[arg(long, value_name = "fast|balanced|quality", default_value = "balanced", value_parser = parse_processing_mode)]
        mode: ProcessingMode,

        /// Optional JSON sidecar path for the generated crop plan.
        #[arg(long)]
        plan_output: Option<PathBuf>,

        /// Optional existing crop plan whose manual keyframes should override
        /// the newly generated camera path.
        #[arg(long)]
        plan_input: Option<PathBuf>,
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
        Commands::Detect {
            input,
            model,
            output,
        } => {
            OrtConfig::ensure_initialized().context("failed to initialize ONNX Runtime")?;
            cmd_detect(input, &model, &output)
        }
        Commands::Doctor => cmd_doctor(),
        Commands::InspectIdentity {
            video,
            bias,
            yolo_model,
            face_model,
            threshold,
            face_det_model,
            sample_every,
            max_frames,
        } => {
            OrtConfig::ensure_initialized().context("failed to initialize ONNX Runtime")?;
            cmd_inspect_identity(IdentityInspectOptions {
                video,
                bias,
                yolo_model,
                face_model,
                threshold,
                face_det_model,
                sample_every,
                max_frames,
            })
        }
        Commands::Fancam {
            video,
            bias,
            output,
            yolo_model,
            face_model,
            identity_model,
            threshold,
            mode,
            plan_output,
            plan_input,
        } => cmd_fancam(FancamOptions {
            video,
            bias,
            output,
            yolo_model,
            face_model,
            identity_model,
            threshold,
            mode,
            plan_output,
            plan_input,
        }),
    }
}

fn parse_processing_mode(value: &str) -> std::result::Result<ProcessingMode, String> {
    match value {
        "fast" => Ok(ProcessingMode::Fast),
        "balanced" => Ok(ProcessingMode::Balanced),
        "quality" => Ok(ProcessingMode::Quality),
        _ => Err(format!(
            "invalid processing mode '{value}'; expected fast, balanced, or quality"
        )),
    }
}

fn validate_fancam_inputs(
    video: &Path,
    bias: &Path,
    output: &Path,
    yolo_model: &Path,
    face_model: &Path,
    identity_model: Option<&Path>,
    threshold: f32,
) -> Result<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        bail!("threshold must be a finite number between 0.0 and 1.0");
    }

    let identity_model = identity_model.unwrap_or(face_model);
    let sources = [
        ("video", video),
        ("bias image", bias),
        ("YOLO model", yolo_model),
        ("identity model", identity_model),
    ];
    let canonical_sources = sources
        .into_iter()
        .map(|(label, path)| canonical_input_file(label, path).map(|canonical| (label, canonical)))
        .collect::<Result<Vec<_>>>()?;

    let canonical_output = canonical_output_file(output)?;
    for (label, canonical_source) in canonical_sources {
        if canonical_source == canonical_output {
            bail!(
                "output path {} collides with the {label} at {}",
                output.display(),
                canonical_source.display()
            );
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Default)]
struct PlanPaths<'a> {
    input: Option<&'a Path>,
    output: Option<&'a Path>,
}

fn validate_plan_paths(
    video: &Path,
    bias: &Path,
    output: &Path,
    yolo_model: &Path,
    face_model: &Path,
    identity_model: Option<&Path>,
    plan_paths: PlanPaths<'_>,
) -> Result<()> {
    let output_canonical = canonical_output_file(output)?;
    let sources = [
        ("video", video),
        ("bias image", bias),
        ("YOLO model", yolo_model),
        ("identity model", identity_model.unwrap_or(face_model)),
    ];
    let canonical_sources = sources
        .into_iter()
        .map(|(label, path)| canonical_input_file(label, path))
        .collect::<Result<Vec<_>>>()?;

    let input_canonical = plan_paths
        .input
        .map(|path| canonical_input_file("plan input", path))
        .transpose()?;
    if let Some(input_canonical) = input_canonical.as_ref()
        && input_canonical == &output_canonical
    {
        bail!("plan input must be different from the video output");
    }

    let plan_output_canonical = plan_paths.output.map(canonical_output_file).transpose()?;
    if let Some(plan_output_canonical) = plan_output_canonical.as_ref() {
        if plan_output_canonical == &output_canonical {
            bail!("plan output must be different from the video output");
        }
        if input_canonical
            .as_ref()
            .is_some_and(|input| input == plan_output_canonical)
        {
            bail!("plan input and plan output must be different files");
        }
        if canonical_sources
            .iter()
            .any(|source| source == plan_output_canonical)
        {
            bail!("plan output collides with an input or model file");
        }
    }
    Ok(())
}

fn canonical_input_file(label: &str, path: &Path) -> Result<PathBuf> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("{label} path does not exist: {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{label} path is not a regular file: {}", path.display());
    }
    fs::canonicalize(path)
        .with_context(|| format!("failed to resolve {label} path: {}", path.display()))
}

fn canonical_output_file(path: &Path) -> Result<PathBuf> {
    let file_name = path.file_name().context("output path must name a file")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent_metadata = fs::metadata(parent)
        .with_context(|| format!("output parent does not exist: {}", parent.display()))?;
    if !parent_metadata.is_dir() {
        bail!("output parent is not a directory: {}", parent.display());
    }
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("failed to resolve output parent: {}", parent.display()))?;

    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                bail!("output path must not be a symlink: {}", path.display());
            }
            if !metadata.is_file() {
                bail!("output path is not a regular file: {}", path.display());
            }
            fs::canonicalize(path)
                .with_context(|| format!("failed to resolve output path: {}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(canonical_parent.join(file_name))
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to inspect output path: {}", path.display()))
        }
    }
}

fn install_cancellation_handler(cancel_flag: &Arc<AtomicBool>) -> Result<()> {
    let cancel_flag = Arc::clone(cancel_flag);
    ctrlc::set_handler(move || {
        if cancel_flag.swap(true, Ordering::SeqCst) {
            eprintln!("Cancellation is already in progress.");
        } else {
            eprintln!("Cancellation requested; stopping after the current frame…");
        }
    })
    .context("failed to install Ctrl-C handler")
}

// ── Phase 2: person detection ─────────────────────────────────────────────────

fn cmd_detect(input: PathBuf, model: &Path, output: &Path) -> Result<()> {
    info!("Phase 2 — person detection");

    let mut detector = Detector::load(model)
        .with_context(|| format!("failed to load model: {}", model.display()))?;

    let pb = spinner("Detecting persons…");
    let pb2 = pb.clone();

    transcode(input, output, move |frame: &mut RgbFrame| {
        pb2.tick();
        match detector.detect(frame) {
            Ok(boxes) => {
                if let Err(e) = draw_boxes(frame, &boxes, [0, 255, 0]) {
                    tracing::warn!("draw boxes error: {e}");
                }
            }
            Err(e) => tracing::warn!("detection error: {e}"),
        }
    })
    .context("detection transcode failed")?;

    pb.finish_with_message("Done.");
    Ok(())
}

// ── Phase 3 + 4: full fancam pipeline ────────────────────────────────────────

struct FancamOptions {
    video: PathBuf,
    bias: PathBuf,
    output: PathBuf,
    yolo_model: PathBuf,
    face_model: PathBuf,
    identity_model: Option<PathBuf>,
    threshold: f32,
    mode: ProcessingMode,
    plan_output: Option<PathBuf>,
    plan_input: Option<PathBuf>,
}

fn cmd_fancam(options: FancamOptions) -> Result<()> {
    let FancamOptions {
        video,
        bias,
        output,
        yolo_model,
        face_model,
        identity_model,
        threshold,
        mode,
        plan_output,
        plan_input,
    } = options;
    validate_fancam_inputs(
        &video,
        &bias,
        &output,
        &yolo_model,
        &face_model,
        identity_model.as_deref(),
        threshold,
    )?;
    validate_plan_paths(
        &video,
        &bias,
        &output,
        &yolo_model,
        &face_model,
        identity_model.as_deref(),
        PlanPaths {
            input: plan_input.as_deref(),
            output: plan_output.as_deref(),
        },
    )?;

    let input_plan = read_input_plan(plan_input.as_deref())?;

    let cancel_flag = Arc::new(AtomicBool::new(false));
    install_cancellation_handler(&cancel_flag)?;
    OrtConfig::ensure_initialized().context("failed to initialize ONNX Runtime")?;

    info!("Fancam pipeline");
    info!("  video      : {}", video.display());
    info!("  bias image : {}", bias.display());
    info!("  output     : {}", output.display());
    info!("  mode       : {}", mode.as_str());

    let identity_model = identity_model.unwrap_or(face_model);

    let pipeline =
        Pipeline::load_with_hint_mode(&yolo_model, &identity_model, &bias, threshold, None, mode)
            .with_context(|| {
            format!(
                "failed to load models or embed reference: {}",
                bias.display()
            )
        })?;

    let total = total_frames(&video);
    let needs_plan = plan_output.is_some() || input_plan.is_some();
    let (mut analyzer, renderer, generated_plan) = run_offline_prepass(
        pipeline,
        &video,
        total,
        needs_plan,
        &cancel_flag,
        &frame_progress(total, "Prepass"),
    )?;

    if let Some(mut plan) = generated_plan {
        apply_manual_plan(&mut plan, input_plan, &mut analyzer)?;
        if let Some(plan_output) = plan_output {
            plan.write_to_path(&plan_output)
                .with_context(|| format!("failed to write crop plan {}", plan_output.display()))?;
        }
    }

    render_with_progress(
        video,
        &output,
        total,
        &cancel_flag,
        (analyzer, renderer),
        mode,
        &frame_progress(total, "Render"),
    )
}

fn read_input_plan(path: Option<&Path>) -> Result<Option<CropPlanV1>> {
    path.map(CropPlanV1::read_from_path)
        .transpose()
        .with_context(|| {
            path.map_or_else(
                || "failed to read crop plan".to_string(),
                |path| format!("failed to read crop plan {}", path.display()),
            )
        })
}

fn prepass_progress_hook(pb: &ProgressBar, total: u64, progress: OfflinePrepassProgress) {
    if total > 0 {
        pb.set_position(progress.decoded_frames.min(total));
        pb.set_message(format!("Prepass: {} sampled", progress.sampled_frames));
    } else {
        pb.set_message(format!(
            "Prepass: {} decoded, {} sampled",
            progress.decoded_frames, progress.sampled_frames
        ));
    }
    pb.tick();
}

/// Run the offline prepass that builds tracklets and solves the camera path,
/// reporting progress and honouring the cancellation flag.
fn run_offline_prepass(
    pipeline: Pipeline,
    video: &Path,
    total: u64,
    needs_plan: bool,
    cancel: &Arc<AtomicBool>,
    pb: &ProgressBar,
) -> Result<(Analyzer, Renderer, Option<CropPlanV1>)> {
    let pb_for_hook = pb.clone();
    let cancel_prepass = Arc::clone(cancel);
    let parts = if needs_plan {
        pipeline
            .into_parts_with_offline_solution_and_plan_with_hooks(
                video,
                move |progress| prepass_progress_hook(&pb_for_hook, total, progress),
                move || cancel_prepass.load(Ordering::Relaxed),
            )
            .map(|(analyzer, renderer, plan)| (analyzer, renderer, Some(plan)))
    } else {
        pipeline
            .into_parts_with_offline_solution_with_hooks(
                video,
                move |progress| prepass_progress_hook(&pb_for_hook, total, progress),
                move || cancel_prepass.load(Ordering::Relaxed),
            )
            .map(|(analyzer, renderer)| (analyzer, renderer, None))
    };
    match parts {
        Ok(parts) => {
            pb.finish_with_message("Prepass complete.");
            Ok(parts)
        }
        Err(error) => {
            pb.abandon_with_message(if cancel.load(Ordering::Relaxed) {
                "Prepass cancelled."
            } else {
                "Prepass failed."
            });
            Err(error)
        }
    }
}

/// Overlay manual keyframes from an existing plan onto the freshly generated
/// plan and re-validate the result.
fn apply_manual_plan(
    generated: &mut CropPlanV1,
    input: Option<CropPlanV1>,
    analyzer: &mut Analyzer,
) -> Result<()> {
    let Some(input_plan) = input else {
        return Ok(());
    };
    generated
        .ensure_source_fingerprint_matches(&input_plan.source_fingerprint)
        .context("crop plan input belongs to a different source video")?;
    generated.manual_keyframes = input_plan.manual_keyframes;
    generated
        .validate()
        .context("manual keyframes made crop plan invalid")?;
    analyzer.enable_offline_from_plan(generated);
    Ok(())
}

/// Render the final fancam using the solved camera path, with progress and
/// cancellation handling.
fn render_with_progress(
    video: PathBuf,
    output: &Path,
    total: u64,
    cancel: &Arc<AtomicBool>,
    parts: (Analyzer, Renderer),
    mode: ProcessingMode,
    pb: &ProgressBar,
) -> Result<()> {
    let mut parts: (Analyzer, Renderer) = parts;
    let pb_for_progress = pb.clone();
    let transcode_result = transcode_with_progress_staged_mode_fallible(
        video,
        output,
        total,
        Arc::clone(cancel),
        move |frame| parts.0.analyze(frame),
        move |frame: &mut RgbFrame, camera| {
            parts.1.render_checked(frame, camera)?;
            Ok(())
        },
        mode,
        move |current, reported_total| {
            if total > 0 {
                pb_for_progress.set_position(current.min(total));
                pb_for_progress.set_message(format!("Render: {current}/{total} frames"));
            } else if reported_total > 0 {
                pb_for_progress.set_message(format!("Render: {current}/{reported_total} frames"));
            } else {
                pb_for_progress.set_message(format!("Render: {current} frames"));
            }
            pb_for_progress.tick();
        },
    );

    match transcode_result {
        Ok(()) => {
            pb.finish_with_message("Fancam saved.");
            Ok(())
        }
        Err(error) => {
            pb.abandon_with_message(if cancel.load(Ordering::Relaxed) {
                "Fancam cancelled."
            } else {
                "Fancam failed."
            });
            Err(error).context("fancam transcode failed")
        }
    }
}

// ── Identity inspection ───────────────────────────────────────────────────────

struct IdentityInspectSummary {
    frames_decoded: u64,
    frames_sampled: u64,
    frames_with_detections: u64,
    total_detections: u64,
    candidates_checked: u64,
    candidates_with_faces: u64,
    candidates_heuristic: u64,
    candidates_skipped_no_face: u64,
    best_similarity: f32,
    best_similarity_mode: ScoringMode,
    second_best_similarity: Option<f32>,
    accepted_matches: u64,
    rejected_matches: u64,
    best_frame: u64,
    best_bbox: (f32, f32, f32, f32),
    threshold: f32,
    reference_warning: Option<String>,
    face_detector_loaded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScoringMode {
    FaceDetected,
    HeuristicFallback,
    NoScore,
}

struct IdentityInspectOptions {
    video: PathBuf,
    bias: PathBuf,
    yolo_model: PathBuf,
    face_model: PathBuf,
    threshold: f32,
    face_det_model: Option<PathBuf>,
    sample_every: u64,
    max_frames: u64,
}

fn cmd_inspect_identity(options: IdentityInspectOptions) -> Result<()> {
    let IdentityInspectOptions {
        video,
        bias,
        yolo_model,
        face_model,
        threshold,
        face_det_model,
        sample_every,
        max_frames,
    } = options;
    if sample_every == 0 {
        bail!("sample_every must be greater than zero");
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        bail!("threshold must be a finite number between 0.0 and 1.0");
    }
    println!("═══ Identity Inspection ═══\n");

    // Validate reference image
    let reference_warning = validate_reference_image(&bias);
    if let Some(ref warn) = reference_warning {
        println!("⚠ Bias image warning: {warn}");
    }

    // Load YOLO detector
    println!("Loading YOLO model...");
    let mut detector = Detector::load(&yolo_model)
        .with_context(|| format!("failed to load YOLO model: {}", yolo_model.display()))?;
    println!("  ✓ YOLO loaded\n");

    // Optionally load face detector
    let mut face_detector = load_face_detector(face_det_model.as_deref());

    // Load identity model and embed reference
    // If face detector is available, first find a face in the reference image
    print!("Loading ArcFace model and embedding reference...");
    let identifier = embed_reference_identity(&mut face_detector, &face_model, &bias, threshold)?;
    println!(
        "  ✓ Reference embedded (threshold={:.2})\n",
        identifier.similarity_threshold()
    );

    // Sample frames and score identity candidates
    println!("Scanning video: {}", video.display());
    println!("  Sample every {sample_every} frame(s), max {max_frames} frames\n");

    let mut summary = initial_inspect_summary(
        identifier.similarity_threshold(),
        reference_warning,
        face_detector.is_some(),
    );

    let pb = spinner("Inspecting identity...");
    let pb2 = pb.clone();

    for_each_rgb_frame(&video, |frame_idx, frame| {
        if summary.frames_decoded >= max_frames {
            return Ok(true);
        }
        summary.frames_decoded += 1;

        if !(summary.frames_decoded - 1).is_multiple_of(sample_every) {
            return Ok(false);
        }
        summary.frames_sampled += 1;
        pb2.tick();

        // Detect persons
        let persons = match detector.detect(frame) {
            Ok(boxes) => boxes,
            Err(e) => {
                tracing::warn!("detection error at frame {}: {e}", frame_idx);
                return Ok(false);
            }
        };
        if persons.is_empty() {
            return Ok(false);
        }
        summary.frames_with_detections += 1;
        summary.total_detections += persons.len() as u64;

        // If face detector is available, filter persons to only those with detected faces
        let persons_to_score =
            filter_persons_by_face(frame, &persons, &mut face_detector, &mut summary);

        // Score identity candidates
        let observations = match identifier.observations(frame, &persons_to_score, None) {
            Ok(obs) => obs,
            Err(e) => {
                tracing::warn!("identity error at frame {}: {e}", frame_idx);
                return Ok(false);
            }
        };

        summary.candidates_checked += observations.len() as u64;
        score_observations(&mut summary, &identifier, &observations);

        Ok(false)
    })
    .context("failed to read video for identity inspection")?;

    pb.finish_with_message("Done.");
    println!();
    print_identity_summary(&summary);
    println!();
    print_identity_verdict(&summary, threshold);

    Ok(())
}

/// Start an empty inspection summary; `threshold` mirrors the identifier's
/// effective similarity threshold.
fn initial_inspect_summary(
    threshold: f32,
    reference_warning: Option<String>,
    face_detector_loaded: bool,
) -> IdentityInspectSummary {
    IdentityInspectSummary {
        frames_decoded: 0,
        frames_sampled: 0,
        frames_with_detections: 0,
        total_detections: 0,
        candidates_checked: 0,
        candidates_with_faces: 0,
        candidates_heuristic: 0,
        candidates_skipped_no_face: 0,
        best_similarity: f32::NEG_INFINITY,
        best_similarity_mode: ScoringMode::NoScore,
        second_best_similarity: None,
        accepted_matches: 0,
        rejected_matches: 0,
        best_frame: 0,
        best_bbox: (0.0, 0.0, 0.0, 0.0),
        threshold,
        reference_warning,
        face_detector_loaded,
    }
}

/// Print post-scan guidance: missed matches, missing detections, and a
/// threshold suggestion when the best similarity falls short.
fn print_identity_verdict(summary: &IdentityInspectSummary, threshold: f32) {
    if summary.accepted_matches == 0 && summary.frames_with_detections > 0 {
        let best = if summary.best_similarity.is_finite() {
            format!("{:.3}", summary.best_similarity)
        } else {
            "N/A".to_string()
        };
        println!(
            "  ⚠ No identity matches at threshold {threshold:.2} (best similarity was {best})"
        );
        if let Some(second) = summary.second_best_similarity {
            let margin = summary.best_similarity - second;
            println!("    Margin over next candidate: {margin:.3}");
        }
        if summary.reference_warning.is_some() {
            println!("    Note: Reference image may not contain a clear face.");
        }
    }

    if summary.frames_with_detections == 0 {
        println!("  ⚠ No person detections found in sampled frames.");
        println!("    Check that the video contains visible people.");
    }

    if summary.best_similarity.is_finite() && summary.best_similarity < threshold {
        println!(
            "  Suggestion: try --threshold {:.2} to lower the bar,",
            (summary.best_similarity - 0.05).max(0.1)
        );
        println!("    or use a better reference image.");
    }
}

/// Load the optional SCRFD face detector; a load failure degrades to `None`
/// (heuristic crops) instead of aborting the inspection.
fn load_face_detector(path: Option<&Path>) -> Option<fancam_core::face::FaceDetector> {
    let path = path?;
    println!("Loading face detector model...");
    match fancam_core::face::FaceDetector::load(path) {
        Ok(fd) => {
            println!("  ✓ Face detector loaded\n");
            Some(fd)
        }
        Err(e) => {
            println!("  ⚠ Face detector load failed: {e} (proceeding without)\n");
            None
        }
    }
}

/// Embed the reference identity, preferring a face-detected crop of the
/// reference image when a face detector is available.
fn embed_reference_identity(
    face_detector: &mut Option<fancam_core::face::FaceDetector>,
    face_model: &Path,
    bias: &Path,
    threshold: f32,
) -> Result<FaceIdentifier> {
    let threshold = threshold.clamp(0.0, 1.0);
    if let Some(fd) = face_detector {
        let ref_rgb = load_reference_as_rgb_frame(bias)?;
        let ref_faces = fd.detect(&ref_rgb)?;
        if let Some(best_face) = ref_faces.into_iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let cropped_reference = crop_face_from_frame(&ref_rgb, &best_face)?;
            println!(" ✓ face detected (conf={:.2})", best_face.confidence);
            FaceIdentifier::load_from_rgb_image(face_model, &cropped_reference, threshold)
                .with_context(|| format!("failed to load ArcFace model: {}", face_model.display()))
        } else {
            println!(" ⚠ no face detected in reference image, using full image as fallback");
            FaceIdentifier::load(face_model, bias, threshold)
                .with_context(|| format!("failed to load ArcFace model: {}", face_model.display()))
        }
    } else {
        FaceIdentifier::load(face_model, bias, threshold).with_context(|| {
            format!(
                "failed to load ArcFace model or embed reference: {}",
                bias.display()
            )
        })
    }
}

/// Restrict candidates to persons with a detected face; falls back to all
/// persons (heuristic crops) when no face is found in the frame.
fn filter_persons_by_face(
    frame: &RgbFrame,
    persons: &[fancam_core::detection::BBox],
    face_detector: &mut Option<fancam_core::face::FaceDetector>,
    summary: &mut IdentityInspectSummary,
) -> Vec<fancam_core::detection::BBox> {
    let Some(fd) = face_detector else {
        summary.candidates_heuristic += persons.len() as u64;
        return persons.to_vec();
    };
    let mut face_persons = Vec::new();
    let mut no_face_count = 0u64;
    for person in persons {
        match fd.best_face_in_person_bbox(frame, *person, 0.05) {
            Ok(Some(_face)) => {
                face_persons.push(*person);
                summary.candidates_with_faces += 1;
            }
            Ok(None) => {
                no_face_count += 1;
            }
            Err(e) => {
                tracing::warn!("face detection error: {e}");
                no_face_count += 1;
            }
        }
    }
    summary.candidates_skipped_no_face += no_face_count;
    if face_persons.is_empty() {
        // Fallback: if no face detected for any person, use all persons with heuristic
        summary.candidates_skipped_no_face = summary
            .candidates_skipped_no_face
            .saturating_sub(persons.len() as u64);
        summary.candidates_heuristic += persons.len() as u64;
        persons.to_vec()
    } else {
        face_persons
    }
}

/// Fold scored observations into the running summary (best match, margin, and
/// accept/reject tallies).
fn score_observations(
    summary: &mut IdentityInspectSummary,
    identifier: &FaceIdentifier,
    observations: &[fancam_core::detection::FaceObservation],
) {
    for obs in observations {
        if obs.similarity >= identifier.similarity_threshold()
            && obs.margin >= identifier.margin_threshold()
        {
            summary.accepted_matches += 1;
        } else {
            summary.rejected_matches += 1;
        }

        if obs.similarity > summary.best_similarity {
            summary.second_best_similarity = Some(summary.best_similarity);
            summary.best_similarity = obs.similarity;
            summary.best_frame = summary.frames_decoded;
            summary.best_bbox = (obs.bbox.x1, obs.bbox.y1, obs.bbox.x2, obs.bbox.y2);
            summary.best_similarity_mode = if summary.face_detector_loaded {
                ScoringMode::FaceDetected
            } else {
                ScoringMode::HeuristicFallback
            };
        } else if summary.best_similarity.is_finite() {
            let second = summary.second_best_similarity.get_or_insert(obs.similarity);
            if obs.similarity > *second {
                *second = obs.similarity;
            }
        }
    }
}

/// Load an image as an [`RgbFrame`] for face detection.
fn load_reference_as_rgb_frame(path: &Path) -> Result<RgbFrame> {
    let img = image::ImageReader::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open image: {e}"))?
        .decode()
        .map_err(|e| anyhow::anyhow!("cannot decode image: {e}"))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok(RgbFrame {
        data: img.into_raw(),
        width: w,
        height: h,
        pts: 0,
    })
}

/// Crop a face region from a frame based on a `FaceBox` (expands slightly around the bbox).
///
/// All float coordinates are clamped to the valid pixel range before the `u32`
/// conversion, so the casts cannot truncate or lose the sign.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn crop_face_from_frame(
    frame: &RgbFrame,
    face_box: &fancam_core::face::FaceBox,
) -> Result<image::RgbImage> {
    frame
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid reference frame: {e}"))?;
    let margin = 0.20; // expand 20% around face bbox for context
    let bw = face_box.bbox.width();
    let bh = face_box.bbox.height();
    if !bw.is_finite() || !bh.is_finite() || bw <= 0.0 || bh <= 0.0 {
        return Err(anyhow::anyhow!(
            "face detection returned an invalid bounding box"
        ));
    }
    let x1 = (face_box.bbox.x1 - bw * margin)
        .floor()
        .clamp(0.0, frame.width.saturating_sub(1) as f32) as u32;
    let y1 = (face_box.bbox.y1 - bh * margin)
        .floor()
        .clamp(0.0, frame.height.saturating_sub(1) as f32) as u32;
    let x2 = (face_box.bbox.x2 + bw * margin)
        .ceil()
        .clamp((x1 + 1) as f32, frame.width as f32) as u32;
    let y2 = (face_box.bbox.y2 + bh * margin)
        .ceil()
        .clamp((y1 + 1) as f32, frame.height as f32) as u32;
    let cw = x2 - x1;
    let ch = y2 - y1;

    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (cw * 3) as usize;
    let mut data = vec![0u8; dst_stride * ch as usize];
    for row in 0..ch as usize {
        let src_start = (y1 as usize + row) * src_stride + x1 as usize * 3;
        let dst_start = row * dst_stride;
        let len = dst_stride.min(frame.data.len().saturating_sub(src_start));
        data[dst_start..dst_start + len].copy_from_slice(&frame.data[src_start..src_start + len]);
    }
    image::RgbImage::from_raw(cw, ch, data)
        .ok_or_else(|| anyhow::anyhow!("cropped face buffer size mismatch"))
}

/// Heuristic check that the reference image is not blank or too small.
///
/// Pixel counts are small enough that converting the count to `f32` cannot
/// lose meaningful precision for the variance heuristic.
#[allow(clippy::cast_precision_loss)]
fn validate_reference_image(path: &Path) -> Option<String> {
    if !path.is_file() {
        return Some(format!("reference image not found: {}", path.display()));
    }

    let img = match image::ImageReader::open(path) {
        Ok(reader) => match reader.decode() {
            Ok(img) => img,
            Err(e) => return Some(format!("cannot decode reference image: {e}")),
        },
        Err(e) => return Some(format!("cannot open reference image: {e}")),
    };

    if img.width() < 32 || img.height() < 32 {
        return Some(format!(
            "reference image too small: {}x{} (minimum 32x32)",
            img.width(),
            img.height()
        ));
    }

    // Check for near-blank image (low variance)
    let rgb = img.to_rgb8();
    let pixels = rgb.as_raw();
    let n = pixels.len() / 3;
    if n == 0 {
        return Some("reference image has no pixels".to_string());
    }
    let mean_r = pixels.iter().step_by(3).map(|&v| f32::from(v)).sum::<f32>() / n as f32;
    let mean_g = pixels
        .iter()
        .skip(1)
        .step_by(3)
        .map(|&v| f32::from(v))
        .sum::<f32>()
        / n as f32;
    let mean_b = pixels
        .iter()
        .skip(2)
        .step_by(3)
        .map(|&v| f32::from(v))
        .sum::<f32>()
        / n as f32;
    let variance = pixels
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            let dr = f32::from(c[0]) - mean_r;
            let dg = f32::from(c[1]) - mean_g;
            let db = f32::from(c[2]) - mean_b;
            dr * dr + dg * dg + db * db
        })
        .sum::<f32>()
        / (n as f32).max(1.0);
    let std_dev = variance.sqrt();

    if std_dev < 5.0 {
        return Some(format!(
            "reference image appears near-blank (std_dev={std_dev:.1}). ArcFace requires a recognizable face crop."
        ));
    }

    None
}

fn print_identity_summary(s: &IdentityInspectSummary) {
    println!("═══ Identity Inspection Results ═══");
    println!(
        "  Mode:                 {}",
        if s.face_detector_loaded {
            "face-detected"
        } else {
            "heuristic crop"
        }
    );
    println!("  Frames decoded:       {}", s.frames_decoded);
    println!("  Frames sampled:       {}", s.frames_sampled);
    println!("  Frames with persons:  {}", s.frames_with_detections);
    println!("  Total detections:     {}", s.total_detections);
    println!("  Candidates checked:   {}", s.candidates_checked);
    if s.face_detector_loaded {
        println!("  Candidates with face: {}", s.candidates_with_faces);
        println!("  Heuristic fallback:   {}", s.candidates_heuristic);
        println!("  Skipped (no face):    {}", s.candidates_skipped_no_face);
    }
    println!("  Best similarity:      {:.4}", s.best_similarity);
    println!("  Best similarity mode: {:?}", s.best_similarity_mode);
    if let Some(second) = s.second_best_similarity {
        println!("  Second best:          {second:.4}");
        println!("  Margin:               {:.4}", s.best_similarity - second);
    }
    println!(
        "  Accepted matches:     {} (threshold {:.2})",
        s.accepted_matches, s.threshold
    );
    println!("  Rejected matches:     {}", s.rejected_matches);
    println!("  Best frame index:     {}", s.best_frame);
    println!(
        "  Best bbox:            ({:.0},{:.0}) ({:.0},{:.0})",
        s.best_bbox.0, s.best_bbox.1, s.best_bbox.2, s.best_bbox.3
    );
    if s.best_similarity.is_finite() {
        let verdict = if s.best_similarity >= s.threshold {
            "✓ ACCEPTED"
        } else {
            "✗ REJECTED"
        };
        println!(
            "  Verdict:              {} (sim={:.3}, threshold={:.2})",
            verdict, s.best_similarity, s.threshold
        );
    } else {
        println!("  Verdict:              ✗ NO SIMILARITY DATA");
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

fn cmd_doctor() -> Result<()> {
    println!("╔═══ focus-lock diagnostics ═══╗");
    println!();

    let mut all_ok = true;
    all_ok &= doctor_platform();
    all_ok &= doctor_ffmpeg();
    all_ok &= doctor_ort_library();
    all_ok &= doctor_ort_init();
    all_ok &= doctor_model_files();
    all_ok &= doctor_output_directory();

    println!(
        "╚═══ {}",
        if all_ok {
            "all checks passed"
        } else {
            "some checks failed — see above"
        }
    );

    // Use process exit to avoid ORT C++ runtime cleanup crash during global dtor
    std::process::exit(i32::from(!all_ok));
}

fn doctor_platform() -> bool {
    println!("[1/6] Platform");
    println!(
        "  OS:   {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("  Rust: {}", rustc_version());
    println!();
    true
}

fn doctor_ffmpeg() -> bool {
    println!("[2/6] FFmpeg");
    match ffmpeg_next::init() {
        Ok(()) => println!("  ✓ FFmpeg initialized"),
        Err(e) => {
            println!("  ✗ FFmpeg init failed: {e}");
            return false;
        }
    }
    println!();
    true
}

fn doctor_ort_library() -> bool {
    println!("[3/6] ONNX Runtime library");
    let found_ort = if let Ok(ref path) = std::env::var("ORT_DYLIB_PATH") {
        let exists = Path::new(path).is_file();
        println!("  ORT_DYLIB_PATH = {path}");
        if exists {
            println!("  ✓ file exists");
            true
        } else {
            println!("  ✗ file does not exist");
            false
        }
    } else {
        println!("  ORT_DYLIB_PATH not set (will auto-discover)");
        false
    };
    if !found_ort {
        let found: Vec<_> = OrtConfig::candidates()
            .into_iter()
            .filter(|cand| cand.is_file())
            .collect();
        if found.is_empty() {
            println!("  ✗ no ORT library found in search paths");
            println!("    Expected libonnxruntime.dylib in:");
            println!("    - models/onnxruntime/lib/");
            println!("    - /opt/homebrew/lib/");
            println!("    - ORT_DYLIB_PATH environment variable");
            return false;
        }
        for cand in &found {
            println!("  ✓ found: {}", cand.display());
        }
    }
    println!();
    true
}

fn doctor_ort_init() -> bool {
    // Full session init is skipped to avoid C++ runtime cleanup crash on exit.
    // Run `focus-lock detect` or `focus-lock fancam` to validate model loading.
    println!("[4/6] ONNX Runtime initialization (light check)");
    match OrtConfig::discover() {
        Ok(config) => {
            println!("  ✓ ORT library found: {}", config.path().display());
        }
        Err(e) => {
            println!("  ✗ {e}");
            return false;
        }
    }
    println!();
    true
}

fn doctor_model_files() -> bool {
    println!("[5/6] Model files");
    let expected_models = &[
        ("YOLO", "models/yolov8n.onnx"),
        ("Face (ArcFace)", "models/w600k_mbf.onnx"),
        ("Body ReID", "models/osnet_x0_25_msmt17.onnx"),
        ("Face Detector (SCRFD)", "models/det_500m.onnx"),
    ];
    let mut ok = true;
    for (label, rel_path) in expected_models {
        let path = Path::new(rel_path);
        let mark = if path.is_file() { "✓" } else { "✗" };
        let size = if path.is_file() {
            match fs::metadata(path) {
                Ok(m) => format!(" ({})", human_size(m.len())),
                Err(_) => String::new(),
            }
        } else {
            ok = false;
            String::new()
        };
        println!("  [{mark}] {label}: {rel_path}{size}");
    }
    println!();
    ok
}

fn doctor_output_directory() -> bool {
    println!("[6/6] Output directory");
    let cwd = std::env::current_dir().unwrap_or_default();
    let writable = fs::metadata(&cwd).is_ok_and(|m| !m.permissions().readonly());
    if writable {
        println!("  ✓ current directory writable: {}", cwd.display());
    } else {
        println!("  ✗ current directory not writable: {}", cwd.display());
    }
    println!();
    writable
}

fn rustc_version() -> String {
    let v = option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown");
    v.to_string()
}

/// Format a byte count for display; `f64` is exact well beyond any realistic
/// model file size.
#[allow(clippy::cast_precision_loss)]
fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    format!("{size:.1} {}", UNITS[unit_idx])
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    let style = ProgressStyle::with_template("{spinner:.cyan} {msg} [{elapsed_precise}]")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]);
    pb.set_style(style);
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

fn frame_progress(total: u64, phase: &str) -> ProgressBar {
    if total == 0 {
        return spinner(phase);
    }

    let pb = ProgressBar::new(total);
    let style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} frames {msg} [{elapsed_precise}]",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar());
    pb.set_style(style);
    pb.set_message(phase.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let suffix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after the Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "focus-lock-cli-test-{}-{suffix}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create temporary test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> (TempDir, PathBuf, PathBuf, PathBuf, PathBuf) {
        let dir = TempDir::new();
        let video = dir.path().join("video.mp4");
        let bias = dir.path().join("bias.jpg");
        let yolo_model = dir.path().join("yolo.onnx");
        let face_model = dir.path().join("face.onnx");
        for path in [&video, &bias, &yolo_model, &face_model] {
            fs::write(path, b"fixture").expect("write fixture file");
        }
        (dir, video, bias, yolo_model, face_model)
    }

    #[test]
    fn processing_mode_parser_accepts_only_documented_values() {
        assert_eq!(parse_processing_mode("fast"), Ok(ProcessingMode::Fast));
        assert_eq!(
            parse_processing_mode("balanced"),
            Ok(ProcessingMode::Balanced)
        );
        assert_eq!(
            parse_processing_mode("quality"),
            Ok(ProcessingMode::Quality)
        );
        assert!(parse_processing_mode("normal").is_err());
    }

    #[test]
    fn fancam_defaults_to_balanced_mode() {
        let cli = Cli::try_parse_from([
            "focus-lock",
            "fancam",
            "--video",
            "video.mp4",
            "--bias",
            "bias.jpg",
        ])
        .expect("parse default Fancam arguments");

        let Commands::Fancam { mode, .. } = cli.command else {
            panic!("expected Fancam command");
        };
        assert_eq!(mode, ProcessingMode::Balanced);
    }

    #[test]
    fn fancam_accepts_optional_plan_paths() {
        let cli = Cli::try_parse_from([
            "focus-lock",
            "fancam",
            "--video",
            "video.mp4",
            "--bias",
            "bias.jpg",
            "--plan-output",
            "crop-plan.json",
            "--plan-input",
            "reviewed-plan.json",
        ])
        .expect("parse optional crop plan arguments");

        let Commands::Fancam {
            plan_output,
            plan_input,
            ..
        } = cli.command
        else {
            panic!("expected Fancam command");
        };
        assert_eq!(plan_output, Some(PathBuf::from("crop-plan.json")));
        assert_eq!(plan_input, Some(PathBuf::from("reviewed-plan.json")));
    }

    #[test]
    fn inspect_identity_rejects_zero_sample_every() {
        let error = Cli::try_parse_from([
            "focus-lock",
            "inspect-identity",
            "--video",
            "video.mp4",
            "--bias",
            "bias.jpg",
            "--sample-every",
            "0",
        ])
        .err()
        .expect("zero sample stride should be rejected");
        assert!(error.to_string().contains("sample-every"));
    }

    #[test]
    fn preflight_rejects_invalid_thresholds() {
        assert!(
            validate_fancam_inputs(
                Path::new("video.mp4"),
                Path::new("bias.jpg"),
                Path::new("fancam.mp4"),
                Path::new("yolo.onnx"),
                Path::new("face.onnx"),
                None,
                f32::NAN,
            )
            .is_err()
        );
        assert!(
            validate_fancam_inputs(
                Path::new("video.mp4"),
                Path::new("bias.jpg"),
                Path::new("fancam.mp4"),
                Path::new("yolo.onnx"),
                Path::new("face.onnx"),
                None,
                -0.01,
            )
            .is_err()
        );
        assert!(
            validate_fancam_inputs(
                Path::new("video.mp4"),
                Path::new("bias.jpg"),
                Path::new("fancam.mp4"),
                Path::new("yolo.onnx"),
                Path::new("face.onnx"),
                None,
                1.01,
            )
            .is_err()
        );
    }

    #[test]
    fn preflight_rejects_source_collisions_and_missing_output_parent() {
        let (_dir, video, bias, yolo_model, face_model) = fixture();
        assert!(
            validate_fancam_inputs(&video, &bias, &video, &yolo_model, &face_model, None, 0.6,)
                .is_err()
        );

        let missing_video = video
            .parent()
            .expect("fixture has a parent")
            .join("missing.mp4");
        assert!(
            validate_fancam_inputs(
                &missing_video,
                &bias,
                &video,
                &yolo_model,
                &face_model,
                None,
                0.6,
            )
            .is_err()
        );

        let output = video
            .parent()
            .expect("fixture has a parent")
            .join("missing")
            .join("fancam.mp4");
        assert!(
            validate_fancam_inputs(&video, &bias, &output, &yolo_model, &face_model, None, 0.6,)
                .is_err()
        );
    }

    #[test]
    fn preflight_allows_overwriting_a_distinct_output_file() {
        let (dir, video, bias, yolo_model, face_model) = fixture();
        let output = dir.path().join("fancam.mp4");
        fs::write(&output, b"previous output").expect("write existing output");

        validate_fancam_inputs(&video, &bias, &output, &yolo_model, &face_model, None, 0.6)
            .expect("distinct output should be valid");
    }

    #[test]
    fn plan_preflight_rejects_collisions_and_allows_distinct_sidecars() {
        let (dir, video, bias, yolo_model, face_model) = fixture();
        let plan_input = dir.path().join("reviewed-plan.json");
        fs::write(&plan_input, b"{}").expect("write plan input");
        let plan_output = dir.path().join("crop-plan.json");

        validate_plan_paths(
            &video,
            &bias,
            &dir.path().join("fancam.mp4"),
            &yolo_model,
            &face_model,
            None,
            PlanPaths {
                input: Some(&plan_input),
                output: Some(&plan_output),
            },
        )
        .expect("distinct plan paths should be valid");

        assert!(
            validate_plan_paths(
                &video,
                &bias,
                &dir.path().join("fancam.mp4"),
                &yolo_model,
                &face_model,
                None,
                PlanPaths {
                    input: Some(&plan_input),
                    output: Some(&dir.path().join("fancam.mp4")),
                },
            )
            .is_err()
        );
        assert!(
            validate_plan_paths(
                &video,
                &bias,
                &dir.path().join("fancam.mp4"),
                &yolo_model,
                &face_model,
                None,
                PlanPaths {
                    input: Some(&plan_input),
                    output: Some(&plan_input),
                },
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn preflight_rejects_dangling_output_symlinks() {
        let (dir, video, bias, yolo_model, face_model) = fixture();
        let output = dir.path().join("fancam.mp4");
        std::os::unix::fs::symlink(dir.path().join("outside.mp4"), &output)
            .expect("create dangling output symlink");

        assert!(
            validate_fancam_inputs(&video, &bias, &output, &yolo_model, &face_model, None, 0.6)
                .is_err()
        );
    }

    #[test]
    fn human_size_uses_next_unit_at_exact_1024_boundaries() {
        assert_eq!(human_size(1023), "1023.0 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1025), "1.0 KB");
        assert_eq!(human_size(1_048_576), "1.0 MB");
        assert_eq!(human_size(12_300_000), "11.7 MB");
    }
}
