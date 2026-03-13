//! pipeline — high-level video processing pipeline
//!
//! This module provides the main processing pipeline that coordinates detection,
//! identification, tracking, and rendering into a cohesive video processing flow.
//!
//! The pipeline consists of two main components:
//! - [`Analyzer`]: Runs detection and identification to find the target person
//! - [`Renderer`]: Crops and renders the output frames
//!
//! # Example
//!
//! ```rust,no_run
//! use fancam_core::pipeline::Pipeline;
//!
//! let pipeline = Pipeline::load(
//!     "yolov8n.onnx",
//!     "arcface.onnx",
//!     "reference_face.jpg",
//!     0.6, // similarity threshold
//! ).expect("Failed to load pipeline");
//!
//! let (mut analyzer, mut renderer) = pipeline.into_parts();
//! // Use analyzer and renderer in your processing loop...
//! ```

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::detection::{Detector, FaceIdentifier};
use crate::mode::ProcessingMode;
use crate::reid::BodyReidentifier;
use crate::rendering::FrameRenderer;
use crate::tracking::{CameraState, TargetTracker, TrackingState};
use crate::video::RgbFrame;

/// Analyzes video frames to detect and identify the target person.
///
/// The analyzer runs YOLO detection and ArcFace identification on each frame,
/// then feeds the results to the tracker for smooth camera movement.
///
/// Profiling metrics are logged every 300 frames to help diagnose performance.
#[derive(Debug)]
pub struct Analyzer {
    detector: Detector,
    identifier: FaceIdentifier,
    body_reidentifier: Option<BodyReidentifier>,
    body_gallery: Vec<Vec<f32>>,
    tracker: TargetTracker,
    prof_frames: u64,
    prof_detect: Duration,
    prof_identify: Duration,
    prof_reid: Duration,
}

impl Analyzer {
    /// Creates a new analyzer with the given detector, identifier, and tracker.
    pub fn new(
        detector: Detector,
        identifier: FaceIdentifier,
        body_reidentifier: Option<BodyReidentifier>,
        body_gallery: Vec<Vec<f32>>,
        tracker: TargetTracker,
    ) -> Self {
        Self {
            detector,
            identifier,
            body_reidentifier,
            body_gallery,
            tracker,
            prof_frames: 0,
            prof_detect: Duration::ZERO,
            prof_identify: Duration::ZERO,
            prof_reid: Duration::ZERO,
        }
    }

    /// Analyzes a single frame and returns the camera state if the target is found.
    ///
    /// This method runs detection and identification (throttled based on tracker state),
    /// updates the tracker with the results, and returns the smoothed camera position.
    ///
    /// Profiling metrics are logged every 300 frames.
    pub fn analyze(&mut self, frame: &RgbFrame) -> Option<CameraState> {
        let run_recognition = self.tracker.should_run_recognition();

        let detect_start = Instant::now();
        let persons = match self.detector.detect(frame) {
            Ok(persons) => persons,
            Err(e) => {
                tracing::warn!("detection error: {e}");
                Vec::new()
            }
        };
        self.prof_detect += detect_start.elapsed();

        let camera = if run_recognition {
            let identify_start = Instant::now();
            let mut observations = match if matches!(
                self.tracker.state(),
                TrackingState::Recovering | TrackingState::Lost
            ) {
                self.identifier
                    .recovery_observations(frame, &persons, self.tracker.search_hint())
            } else {
                self.identifier
                    .observations(frame, &persons, self.tracker.search_hint())
            } {
                Ok(rows) => rows,
                Err(e) => {
                    tracing::warn!("face ID error: {e}");
                    Vec::new()
                }
            };

            if matches!(
                self.tracker.state(),
                TrackingState::Recovering | TrackingState::Lost
            ) && let Some(last_bbox) = self.tracker.last_confirmed_bbox()
            {
                observations.sort_unstable_by(|a, b| {
                    let ascore = recovery_priority(*a, last_bbox);
                    let bscore = recovery_priority(*b, last_bbox);
                    bscore
                        .partial_cmp(&ascore)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            b.similarity
                                .partial_cmp(&a.similarity)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                });
            }

            if !observations.is_empty()
                && !self.body_gallery.is_empty()
                && let Some(body_reid) = self.body_reidentifier.as_ref()
            {
                let reid_start = Instant::now();
                if let Err(err) = body_reid.annotate_observations_with_gallery(
                    frame,
                    &mut observations,
                    &self.body_gallery,
                ) {
                    tracing::warn!("body reid error: {err}");
                }
                self.prof_reid += reid_start.elapsed();
            }

            self.prof_identify += identify_start.elapsed();
            self.tracker.update(
                &observations,
                self.identifier.similarity_threshold(),
                self.identifier.margin_threshold(),
            )
        } else {
            self.tracker.update_from_person_detections(&persons)
        };

        self.prof_frames += 1;
        if self.prof_frames.is_multiple_of(300) {
            tracing::info!(
                frames = self.prof_frames,
                detect_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_detect.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                identify_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_identify.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                reid_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_reid.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                "pipeline analyze timings"
            );
        }

        camera
    }
}

fn recovery_priority(
    obs: crate::detection::FaceObservation,
    last_bbox: crate::detection::BBox,
) -> f32 {
    let iou = obs.bbox.iou(&last_bbox).clamp(0.0, 1.0);
    let dx = obs.bbox.center_x() - last_bbox.center_x();
    let dy = obs.bbox.center_y() - last_bbox.center_y();
    let distance = (dx * dx + dy * dy).sqrt();
    let norm = (last_bbox.width().max(last_bbox.height()) * 6.5).max(1.0);
    let proximity = 1.0 - (distance / norm).clamp(0.0, 1.0);
    let body = obs
        .body_similarity
        .map(|sim| ((sim + 1.0) * 0.5).clamp(0.0, 1.0))
        .unwrap_or(0.0);

    obs.similarity * 0.55 + obs.margin * 0.15 + iou * 0.16 + proximity * 0.10 + body * 0.04
}

/// Renders output frames by cropping and scaling to the target resolution.
///
/// The renderer applies the camera state from the tracker to produce the final
/// 9:16 vertical output. When the target is lost, it renders a letterboxed
/// passthrough instead.
///
/// Profiling metrics are logged every 300 frames.
#[derive(Debug)]
pub struct Renderer {
    renderer: FrameRenderer,
    prof_frames: u64,
    prof_render: Duration,
}

impl Renderer {
    /// Creates a new renderer wrapping the given frame renderer.
    pub fn new(renderer: FrameRenderer) -> Self {
        Self {
            renderer,
            prof_frames: 0,
            prof_render: Duration::ZERO,
        }
    }

    /// Renders a frame based on the camera state.
    ///
    /// If `camera` is `Some`, crops to the target position. If `None`, renders
    /// a letterboxed passthrough.
    ///
    /// # Arguments
    ///
    /// * `frame` - The input frame to modify in-place
    /// * `camera` - The camera state from the tracker, or `None` if target lost
    pub fn render(&mut self, frame: &mut RgbFrame, camera: Option<CameraState>) {
        let render_start = Instant::now();
        let result = match camera {
            Some(ref state) => self.renderer.crop_fancam_inplace(frame, state),
            None => self.renderer.letterbox_passthrough_inplace(frame),
        };
        self.prof_render += render_start.elapsed();
        self.prof_frames += 1;

        if let Err(e) = result {
            tracing::warn!("render error: {e}");
        }

        if self.prof_frames.is_multiple_of(300) {
            tracing::info!(
                frames = self.prof_frames,
                render_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_render.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                "pipeline render timings"
            );
        }
    }
}

/// Complete processing pipeline combining analysis and rendering.
///
/// The pipeline loads the ML models and reference image, then provides
/// an [`Analyzer`] and [`Renderer`] that work together to process video frames.
///
/// Use [`Pipeline::load`] or [`Pipeline::load_with_hint`] to create a pipeline,
/// then call [`into_parts`](Self::into_parts) to get the analyzer and renderer.
#[derive(Debug)]
pub struct Pipeline {
    analyzer: Analyzer,
    renderer: Renderer,
}

impl Pipeline {
    fn build(
        detector: Detector,
        identifier: FaceIdentifier,
        body_reidentifier: Option<BodyReidentifier>,
        body_gallery: Vec<Vec<f32>>,
        initial_search_hint: Option<(f32, f32)>,
        mode: ProcessingMode,
    ) -> Self {
        let tracker = TargetTracker::new_with_hint(initial_search_hint);
        let renderer = FrameRenderer::new_with_mode(mode);
        Self {
            analyzer: Analyzer::new(
                detector,
                identifier,
                body_reidentifier,
                body_gallery,
                tracker,
            ),
            renderer: Renderer::new(renderer),
        }
    }

    /// Loads the pipeline with the given model paths and reference image.
    ///
    /// # Arguments
    ///
    /// * `yolo_model_path` - Path to the YOLOv8 ONNX model for person detection
    /// * `face_model_path` - Path to the ArcFace ONNX model for face identification
    /// * `reference_image_path` - Path to the reference face image of the target person
    /// * `similarity_threshold` - Cosine similarity threshold (0.0-1.0) for matching
    ///
    /// # Errors
    ///
    /// Returns an error if the models cannot be loaded or the reference image
    /// cannot be processed.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        reference_image_path: R,
        similarity_threshold: f32,
    ) -> Result<Self> {
        Self::load_with_hint(
            yolo_model_path,
            face_model_path,
            reference_image_path,
            similarity_threshold,
            None,
        )
    }

    /// Loads the pipeline with an optional initial search hint.
    ///
    /// The search hint provides a starting position (x, y) for the tracker
    /// before the first detection, which can improve initial lock-on speed.
    ///
    /// # Arguments
    ///
    /// * `yolo_model_path` - Path to the YOLOv8 ONNX model
    /// * `face_model_path` - Path to the ArcFace ONNX model
    /// * `reference_image_path` - Path to the reference face image
    /// * `similarity_threshold` - Cosine similarity threshold (0.0-1.0)
    /// * `initial_search_hint` - Optional (x, y) starting position hint
    ///
    /// # Errors
    ///
    /// Returns an error if models cannot be loaded.
    pub fn load_with_hint<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        reference_image_path: R,
        similarity_threshold: f32,
        initial_search_hint: Option<(f32, f32)>,
    ) -> Result<Self> {
        Self::load_with_hint_mode(
            yolo_model_path,
            face_model_path,
            reference_image_path,
            similarity_threshold,
            initial_search_hint,
            ProcessingMode::default(),
        )
    }

    /// Loads the pipeline with an explicit processing mode.
    pub fn load_with_hint_mode<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        reference_image_path: R,
        similarity_threshold: f32,
        initial_search_hint: Option<(f32, f32)>,
        mode: ProcessingMode,
    ) -> Result<Self> {
        let detector = Detector::load(yolo_model_path)?;
        let identifier = FaceIdentifier::load(
            face_model_path,
            reference_image_path,
            similarity_threshold.clamp(0.0, 1.0),
        )?;
        Ok(Self::build(
            detector,
            identifier,
            None,
            Vec::new(),
            initial_search_hint,
            mode,
        ))
    }

    /// Loads the pipeline using a precomputed reference embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the models cannot be loaded.
    pub fn load_with_hint_embedding<P: AsRef<Path>, Q: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        reference_embedding: Vec<f32>,
        similarity_threshold: f32,
        initial_search_hint: Option<(f32, f32)>,
        mode: ProcessingMode,
    ) -> Result<Self> {
        let detector = Detector::load(yolo_model_path)?;
        let identifier = FaceIdentifier::load_from_embedding(
            face_model_path,
            reference_embedding,
            similarity_threshold.clamp(0.0, 1.0),
        )?;
        Ok(Self::build(
            detector,
            identifier,
            None,
            Vec::new(),
            initial_search_hint,
            mode,
        ))
    }

    /// Loads the pipeline with explicit positive and negative embedding galleries.
    ///
    /// # Errors
    ///
    /// Returns an error if models cannot be loaded or target gallery is invalid.
    pub fn load_with_hint_galleries<P: AsRef<Path>, Q: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        body_reid_model_path: Option<&str>,
        target_embeddings: Vec<Vec<f32>>,
        body_target_embeddings: Vec<Vec<f32>>,
        negative_embeddings: Vec<Vec<f32>>,
        similarity_threshold: f32,
        margin_threshold: f32,
        initial_search_hint: Option<(f32, f32)>,
        mode: ProcessingMode,
    ) -> Result<Self> {
        let detector = Detector::load(yolo_model_path)?;
        let body_reidentifier = body_reid_model_path
            .and_then(|path| {
                let trimmed = path.trim();
                (!trimmed.is_empty()).then_some(trimmed)
            })
            .map(BodyReidentifier::load)
            .transpose()?;
        let body_gallery = body_target_embeddings;
        let identifier = FaceIdentifier::load_from_galleries(
            face_model_path,
            target_embeddings,
            negative_embeddings,
            similarity_threshold.clamp(0.0, 1.0),
            margin_threshold,
        )?;
        Ok(Self::build(
            detector,
            identifier,
            body_reidentifier,
            body_gallery,
            initial_search_hint,
            mode,
        ))
    }

    /// Consumes the pipeline and returns its analyzer and renderer components.
    ///
    /// This allows direct access to the components for advanced use cases
    /// where the standard pipeline flow needs to be customized.
    #[must_use]
    pub fn into_parts(self) -> (Analyzer, Renderer) {
        (self.analyzer, self.renderer)
    }
}
