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
use crate::rendering::FrameRenderer;
use crate::tracking::{BiasTracker, CameraState};
use crate::video::RgbFrame;

/// Analyzes video frames to detect and identify the target person.
///
/// The analyzer runs YOLO detection and ArcFace identification on each frame,
/// then feeds the results to the tracker for smooth camera movement.
///
/// Profiling metrics are logged every 300 frames to help diagnose performance.
pub struct Analyzer {
    detector: Detector,
    identifier: FaceIdentifier,
    tracker: BiasTracker,
    prof_frames: u64,
    prof_detect: Duration,
    prof_identify: Duration,
}

impl Analyzer {
    /// Creates a new analyzer with the given detector, identifier, and tracker.
    pub fn new(detector: Detector, identifier: FaceIdentifier, tracker: BiasTracker) -> Self {
        Self {
            detector,
            identifier,
            tracker,
            prof_frames: 0,
            prof_detect: Duration::ZERO,
            prof_identify: Duration::ZERO,
        }
    }

    /// Analyzes a single frame and returns the camera state if the target is found.
    ///
    /// This method runs detection and identification (throttled based on tracker state),
    /// updates the tracker with the results, and returns the smoothed camera position.
    ///
    /// Profiling metrics are logged every 300 frames.
    pub fn analyze(&mut self, frame: &RgbFrame) -> Option<CameraState> {
        let detection = if self.tracker.should_run_recognition() {
            let detect_start = Instant::now();
            match self.detector.detect(frame) {
                Ok(persons) => {
                    self.prof_detect += detect_start.elapsed();
                    let identify_start = Instant::now();
                    match self
                        .identifier
                        .identify(frame, &persons, self.tracker.search_hint())
                    {
                        Ok(found) => {
                            self.prof_identify += identify_start.elapsed();
                            found
                        }
                        Err(e) => {
                            self.prof_identify += identify_start.elapsed();
                            tracing::warn!("face ID error: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    self.prof_detect += detect_start.elapsed();
                    tracing::warn!("detection error: {e}");
                    None
                }
            }
        } else {
            None
        };

        self.tracker
            .note_similarity(detection.as_ref().map(|m| m.similarity));
        let camera = self.tracker.update(detection.map(|m| m.bbox));

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
                "pipeline analyze timings"
            );
        }

        camera
    }
}

/// Renders output frames by cropping and scaling to the target resolution.
///
/// The renderer applies the camera state from the tracker to produce the final
/// 9:16 vertical output. When the target is lost, it renders a letterboxed
/// passthrough instead.
///
/// Profiling metrics are logged every 300 frames.
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
pub struct Pipeline {
    analyzer: Analyzer,
    renderer: Renderer,
}

impl Pipeline {
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

    pub fn load_with_hint<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        reference_image_path: R,
        similarity_threshold: f32,
        initial_search_hint: Option<(f32, f32)>,
    ) -> Result<Self> {
        let detector = Detector::load(yolo_model_path)?;
        let identifier = FaceIdentifier::load(
            face_model_path,
            reference_image_path,
            similarity_threshold.clamp(0.0, 1.0),
        )?;
        let tracker = BiasTracker::new_with_hint(initial_search_hint);
        let renderer = FrameRenderer::new();

        Ok(Self {
            analyzer: Analyzer::new(detector, identifier, tracker),
            renderer: Renderer::new(renderer),
        })
    }

    pub fn into_parts(self) -> (Analyzer, Renderer) {
        (self.analyzer, self.renderer)
    }
}
