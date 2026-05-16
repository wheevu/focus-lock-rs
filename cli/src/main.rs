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

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use tracing::info;
use tracing_subscriber::EnvFilter;

use fancam_core::{
    detection::{Detector, FaceIdentifier, draw_boxes},
    pipeline::Pipeline,
    runtime::OrtConfig,
    video::{
        RgbFrame, for_each_rgb_frame, to_grayscale, transcode, transcode_with_progress_staged,
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

        /// Face detection model (SCRFD, e.g. models/det_500m.onnx).
        /// When provided, candidate face crops use detected faces instead of heuristic head regions.
        #[arg(long)]
        face_det_model: Option<PathBuf>,

        /// Sample every N frames
        #[arg(long, default_value_t = 30)]
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
        } => {
            OrtConfig::ensure_initialized().context("failed to initialize ONNX Runtime")?;
            cmd_detect(input, model, output)
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
            cmd_inspect_identity(
                video,
                bias,
                yolo_model,
                face_model,
                threshold,
                face_det_model,
                sample_every,
                max_frames,
            )
        }
        Commands::Fancam {
            video,
            bias,
            output,
            yolo_model,
            face_model,
            identity_model,
            threshold,
        } => cmd_fancam(
            video,
            bias,
            output,
            yolo_model,
            face_model,
            identity_model,
            threshold,
        ),
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

fn cmd_fancam(
    video: PathBuf,
    bias: PathBuf,
    output: PathBuf,
    yolo_model: PathBuf,
    face_model: PathBuf,
    identity_model: Option<PathBuf>,
    threshold: f32,
) -> Result<()> {
    info!("Fancam pipeline");
    info!("  video      : {}", video.display());
    info!("  bias image : {}", bias.display());
    info!("  output     : {}", output.display());

    let identity_model = identity_model.unwrap_or(face_model);

    let pipeline = Pipeline::load(
        &yolo_model,
        &identity_model,
        &bias,
        threshold.clamp(0.0, 1.0),
    )
    .with_context(|| {
        format!(
            "failed to load models or embed reference: {}",
            bias.display()
        )
    })?;

    let pb = spinner("Building offline prepass…");
    let pb_prepass = pb.clone();
    let (mut analyzer, mut renderer) = pipeline
        .into_parts_with_offline_solution_with_hooks(
            &video,
            |_| {
                pb_prepass.tick();
            },
            || false,
        )
        .context("failed to build offline tracklet/camera solution")?;

    pb.set_message("Generating fancam…".to_string());
    let pb_render = pb.clone();

    transcode_with_progress_staged(
        video,
        &output,
        0,
        move |frame| analyzer.analyze(frame),
        move |frame: &mut RgbFrame, camera| {
            renderer.render(frame, camera);
            pb_render.tick();
        },
        |_, _| {},
    )
    .context("fancam transcode failed")?;

    pb.finish_with_message("Fancam saved.");
    Ok(())
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

fn cmd_inspect_identity(
    video: PathBuf,
    bias: PathBuf,
    yolo_model: PathBuf,
    face_model: PathBuf,
    threshold: f32,
    face_det_model: Option<PathBuf>,
    sample_every: u64,
    max_frames: u64,
) -> Result<()> {
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
    let mut face_detector: Option<fancam_core::face::FaceDetector> = None;
    if let Some(ref fd_path) = face_det_model {
        println!("Loading face detector model...");
        match fancam_core::face::FaceDetector::load(fd_path) {
            Ok(fd) => {
                face_detector = Some(fd);
                println!("  ✓ Face detector loaded\n");
            }
            Err(e) => {
                println!("  ⚠ Face detector load failed: {e} (proceeding without)\n");
            }
        }
    }

    // Load identity model and embed reference
    // If face detector is available, first find a face in the reference image
    print!("Loading ArcFace model and embedding reference...");
    let identifier = if let Some(ref mut fd) = face_detector {
        // Load the reference image and detect a face in it
        let ref_rgb = load_reference_as_rgb_frame(&bias)?;
        let ref_faces = fd.detect(&ref_rgb)?;
        if ref_faces.is_empty() {
            println!(" ⚠ no face detected in reference image, using full image as fallback");
            FaceIdentifier::load(&face_model, &bias, threshold.clamp(0.0, 1.0)).with_context(
                || format!("failed to load ArcFace model: {}", face_model.display()),
            )?
        } else {
            let best_face = ref_faces
                .into_iter()
                .max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            println!(" ✓ face detected (conf={:.2})", best_face.confidence);
            FaceIdentifier::load(&face_model, &bias, threshold.clamp(0.0, 1.0)).with_context(
                || format!("failed to load ArcFace model: {}", face_model.display()),
            )?
        }
    } else {
        FaceIdentifier::load(&face_model, &bias, threshold.clamp(0.0, 1.0)).with_context(|| {
            format!(
                "failed to load ArcFace model or embed reference: {}",
                bias.display()
            )
        })?
    };
    println!(
        "  ✓ Reference embedded (threshold={:.2})\n",
        identifier.similarity_threshold()
    );

    // Sample frames and score identity candidates
    println!("Scanning video: {}", video.display());
    println!(
        "  Sample every {} frame(s), max {} frames\n",
        sample_every, max_frames
    );

    let mut summary = IdentityInspectSummary {
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
        threshold: identifier.similarity_threshold(),
        reference_warning,
        face_detector_loaded: face_detector.is_some(),
    };

    let pb = spinner("Inspecting identity...");
    let pb2 = pb.clone();

    for_each_rgb_frame(&video, |frame_idx, frame| {
        if summary.frames_decoded >= max_frames {
            return Ok(true);
        }
        summary.frames_decoded += 1;

        if (summary.frames_decoded - 1) % sample_every != 0 {
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
        let persons_to_score = if let Some(ref mut fd) = face_detector {
            let mut face_persons = Vec::new();
            let mut no_face_count = 0u64;
            for person in &persons {
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
                persons
            } else {
                face_persons
            }
        } else {
            summary.candidates_heuristic += persons.len() as u64;
            persons
        };

        // Score identity candidates
        let observations = match identifier.observations(frame, &persons_to_score, None) {
            Ok(obs) => obs,
            Err(e) => {
                tracing::warn!("identity error at frame {}: {e}", frame_idx);
                return Ok(false);
            }
        };

        summary.candidates_checked += observations.len() as u64;

        for obs in &observations {
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

        Ok(false)
    })
    .context("failed to read video for identity inspection")?;

    pb.finish_with_message("Done.");
    println!();
    print_identity_summary(&summary);
    println!();

    if summary.accepted_matches == 0 && summary.frames_with_detections > 0 {
        let best = if summary.best_similarity.is_finite() {
            format!("{:.3}", summary.best_similarity)
        } else {
            "N/A".to_string()
        };
        println!(
            "  ⚠ No identity matches at threshold {:.2} (best similarity was {})",
            threshold, best
        );
        if let Some(second) = summary.second_best_similarity {
            let margin = summary.best_similarity - second;
            println!("    Margin over next candidate: {:.3}", margin);
        }
        if !summary.reference_warning.is_none() {
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

    Ok(())
}

/// Load an image as an RgbFrame for face detection.
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

/// Crop a face region from a frame based on a FaceBox (expands slightly around the bbox).
#[allow(dead_code)]
fn crop_face_from_frame(
    frame: &RgbFrame,
    face_box: &fancam_core::face::FaceBox,
) -> Result<RgbFrame> {
    let margin = 0.20; // expand 20% around face bbox for context
    let bw = face_box.bbox.width();
    let bh = face_box.bbox.height();
    let x1 = (face_box.bbox.x1 - bw * margin).max(0.0) as u32;
    let y1 = (face_box.bbox.y1 - bh * margin).max(0.0) as u32;
    let x2 = (face_box.bbox.x2 + bw * margin).min(frame.width as f32) as u32;
    let y2 = (face_box.bbox.y2 + bh * margin).min(frame.height as f32) as u32;
    let cw = (x2 - x1).max(1);
    let ch = (y2 - y1).max(1);

    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (cw * 3) as usize;
    let mut data = vec![0u8; dst_stride * ch as usize];
    for row in 0..ch as usize {
        let src_start = (y1 as usize + row) * src_stride + x1 as usize * 3;
        let dst_start = row * dst_stride;
        let len = dst_stride.min(frame.data.len().saturating_sub(src_start));
        data[dst_start..dst_start + len].copy_from_slice(&frame.data[src_start..src_start + len]);
    }
    Ok(RgbFrame {
        data,
        width: cw,
        height: ch,
        pts: frame.pts,
    })
}

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
    let mean_r = pixels.iter().step_by(3).map(|&v| v as f32).sum::<f32>() / n as f32;
    let mean_g = pixels
        .iter()
        .skip(1)
        .step_by(3)
        .map(|&v| v as f32)
        .sum::<f32>()
        / n as f32;
    let mean_b = pixels
        .iter()
        .skip(2)
        .step_by(3)
        .map(|&v| v as f32)
        .sum::<f32>()
        / n as f32;
    let variance = pixels
        .chunks_exact(3)
        .map(|c| {
            let dr = c[0] as f32 - mean_r;
            let dg = c[1] as f32 - mean_g;
            let db = c[2] as f32 - mean_b;
            dr * dr + dg * dg + db * db
        })
        .sum::<f32>()
        / (n as f32).max(1.0);
    let std_dev = variance.sqrt();

    if std_dev < 5.0 {
        return Some(format!(
            "reference image appears near-blank (std_dev={:.1}). ArcFace requires a recognizable face crop.",
            std_dev
        ));
    }

    // Warn that no face detection is performed on reference
    Some("no face detector available — ensure reference is a cropped face image".to_string())
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
        println!("  Second best:          {:.4}", second);
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

    // 1. Platform
    println!("[1/6] Platform");
    println!(
        "  OS:   {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("  Rust: {}", rustc_version());
    println!();

    // 2. FFmpeg
    println!("[2/6] FFmpeg");
    match ffmpeg_next::init() {
        Ok(()) => println!("  ✓ FFmpeg initialized"),
        Err(e) => {
            println!("  ✗ FFmpeg init failed: {e}");
            all_ok = false;
        }
    }
    println!();

    // 3. ONNX Runtime dylib
    println!("[3/6] ONNX Runtime library");
    let found_ort = match std::env::var("ORT_DYLIB_PATH") {
        Ok(ref path) => {
            let exists = Path::new(path).is_file();
            println!("  ORT_DYLIB_PATH = {path}");
            if exists {
                println!("  ✓ file exists");
                true
            } else {
                println!("  ✗ file does not exist");
                all_ok = false;
                false
            }
        }
        Err(_) => {
            println!("  ORT_DYLIB_PATH not set (will auto-discover)");
            false
        }
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
            all_ok = false;
        } else {
            for cand in &found {
                println!("  ✓ found: {}", cand.display());
            }
        }
    }
    println!();

    // 4. ONNX Runtime init (dylib presence check only)
    // Full session init is skipped to avoid C++ runtime cleanup crash on exit.
    // Run `focus-lock detect` or `focus-lock fancam` to validate model loading.
    println!("[4/6] ONNX Runtime initialization (light check)");
    match OrtConfig::discover() {
        Ok(config) => {
            println!("  ✓ ORT library found: {}", config.path().display());
        }
        Err(e) => {
            println!("  ✗ {e}");
            all_ok = false;
        }
    }
    println!();

    // 5. Model files
    println!("[5/6] Model files");
    let expected_models = &[
        ("YOLO", "models/yolov8n.onnx"),
        ("Face (ArcFace)", "models/w600k_mbf.onnx"),
        ("Body ReID", "models/osnet_x0_25_msmt17.onnx"),
        ("Face Detector (SCRFD)", "models/det_500m.onnx"),
    ];
    for (label, rel_path) in expected_models {
        let path = Path::new(rel_path);
        let mark = if path.is_file() { "✓" } else { "✗" };
        let size = if path.is_file() {
            match std::fs::metadata(path) {
                Ok(m) => format!(" ({})", human_size(m.len())),
                Err(_) => String::new(),
            }
        } else {
            all_ok = false;
            String::new()
        };
        println!("  [{mark}] {label}: {rel_path}{size}");
    }
    println!();

    // 6. Output directory
    println!("[6/6] Output directory");
    let cwd = std::env::current_dir().unwrap_or_default();
    let writable = std::fs::metadata(&cwd).is_ok_and(|m| !m.permissions().readonly());
    if writable {
        println!("  ✓ current directory writable: {}", cwd.display());
    } else {
        println!("  ✗ current directory not writable: {}", cwd.display());
        all_ok = false;
    }
    println!();

    println!(
        "╚═══ {}",
        if all_ok {
            "all checks passed"
        } else {
            "some checks failed — see above"
        }
    );

    // Use process exit to avoid ORT C++ runtime cleanup crash during global dtor
    std::process::exit(if all_ok { 0 } else { 1 });
}

fn rustc_version() -> String {
    let v = option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown");
    v.to_string()
}

fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size > 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    format!("{size:.1} {}", UNITS[unit_idx])
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
