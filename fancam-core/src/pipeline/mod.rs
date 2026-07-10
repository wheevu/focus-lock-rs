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

use crate::camera::CameraCursor;
use crate::detection::{BBox, Detector, FaceIdentifier};
use crate::mode::ProcessingMode;
use crate::reid::BodyReidentifier;
use crate::rendering::FrameRenderer;
use crate::solver::{self, SolverResult};
use crate::tracking::{CameraState, TargetTracker, TrackingState};
use crate::tracklet::Tracklet;
use crate::video;
use crate::video::RgbFrame;

const TRACKLET_MAX_GAP_FRAMES: u64 = 6;
const TRACKLET_MIN_IOU: f32 = 0.12;
const TRACKLET_MAX_CENTER_DISTANCE_NORM: f32 = 0.24;
const OBS_MATCH_MIN_SCORE: f32 = 0.56;

/// Progress updates emitted while building offline prepass tracklets.
#[derive(Debug, Clone, Copy)]
pub struct OfflinePrepassProgress {
    /// Number of decoded frames seen in prepass.
    pub decoded_frames: u64,
    /// Number of sampled frames actually processed for identity cues.
    pub sampled_frames: u64,
}

/// Analyzes video frames to detect and identify the target person.
///
/// The analyzer runs YOLO detection and `ArcFace` identification on each frame,
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
    offline_cursor: Option<CameraCursor>,
    offline_frame_index: u64,
    mode: ProcessingMode,
}

impl Analyzer {
    /// Creates a new analyzer with the given detector, identifier, and tracker.
    #[must_use]
    pub const fn new(
        detector: Detector,
        identifier: FaceIdentifier,
        body_reidentifier: Option<BodyReidentifier>,
        body_gallery: Vec<Vec<f32>>,
        tracker: TargetTracker,
        mode: ProcessingMode,
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
            offline_cursor: None,
            offline_frame_index: 0,
            mode,
        }
    }

    /// Analyzes a single frame and returns the camera state if the target is found.
    ///
    /// This method runs detection and identification (throttled based on tracker state),
    /// updates the tracker with the results, and returns the smoothed camera position.
    ///
    /// Profiling metrics are logged every 300 frames.
    pub fn analyze(&mut self, frame: &RgbFrame) -> Option<CameraState> {
        if let Some(cursor) = self.offline_cursor.as_mut() {
            self.offline_frame_index = self.offline_frame_index.saturating_add(1);
            return cursor.camera_for_frame(self.offline_frame_index);
        }

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

    /// Enable offline clustered camera mode from prebuilt tracklets.
    pub fn enable_offline_from_tracklets(&mut self, tracklets: &[Tracklet]) {
        let solved = solver::solve(tracklets);
        self.enable_offline_from_solver_result(solved);
    }

    /// Enable offline clustered camera mode from an existing solver result.
    pub fn enable_offline_from_solver_result(&mut self, solved: SolverResult) {
        if solved.camera_path.is_empty() {
            self.offline_cursor = None;
            self.offline_frame_index = 0;
            return;
        }
        if let Some(identity_id) = solved.selected_identity_id {
            tracing::info!(identity_id, "offline identity selected for render");
        }
        self.offline_cursor = Some(CameraCursor::from_path(solved.camera_path));
        self.offline_frame_index = 0;
    }

    /// Disable offline mode and return to online tracking.
    pub fn disable_offline_mode(&mut self) {
        self.offline_cursor = None;
        self.offline_frame_index = 0;
    }

    /// Pass-1 helper: build short-term tracklets from frame stream.
    pub fn build_tracklets<F>(&mut self, frame_source: F) -> Result<Vec<Tracklet>>
    where
        F: FnOnce() -> Result<Vec<(u64, RgbFrame)>>,
    {
        let frames = frame_source()?;
        let mut next_tracklet_id = 0usize;
        let mut tracklets = Vec::<Tracklet>::new();

        for (frame_index, frame) in frames {
            let persons = self.detector.detect(&frame)?;
            if persons.is_empty() {
                continue;
            }

            let mut observations = self
                .identifier
                .observations(&frame, &persons, self.tracker.search_hint())?
                .into_iter()
                .collect::<Vec<crate::detection::FaceObservation>>();

            if !observations.is_empty()
                && !self.body_gallery.is_empty()
                && let Some(body_reid) = self.body_reidentifier.as_ref()
            {
                let _ = body_reid.annotate_observations_with_gallery(
                    &frame,
                    &mut observations,
                    &self.body_gallery,
                );
            }

            let mut identity_rows = observations
                .into_iter()
                .map(Into::into)
                .collect::<Vec<crate::observation::IdentityObservation>>();

            for bbox in persons {
                let identity = take_best_identity_for_bbox(&mut identity_rows, bbox)
                    .unwrap_or_else(|| {
                        crate::observation::IdentityObservation::from_face_scores(
                            bbox, 0.0, -1.0, 0.0, None,
                        )
                    });

                let best_idx = find_tracklet_assignment(
                    &tracklets,
                    frame_index,
                    bbox,
                    &identity,
                    frame.width,
                    frame.height,
                );
                if let Some(tracklet_idx) = best_idx {
                    if let Some(tracklet) = tracklets.get_mut(tracklet_idx) {
                        tracklet.push(frame_index, bbox, identity);
                        continue;
                    }
                }

                let mut tracklet = Tracklet::new(next_tracklet_id);
                next_tracklet_id = next_tracklet_id.saturating_add(1);
                tracklet.push(frame_index, bbox, identity);
                tracklets.push(tracklet);
            }
        }

        // Discard one-off fragments unless they are high-confidence.
        Ok(tracklets
            .into_iter()
            .filter(|tracklet| tracklet.len() > 1 || tracklet.best_composite_score() >= 0.66)
            .collect())
    }

    /// Pass-1 helper: build short-term tracklets directly from a source video.
    ///
    /// This decodes frames incrementally and does not retain full-frame RGB
    /// buffers, so it is suitable for long videos.
    ///
    /// # Errors
    ///
    /// Returns an error if decode, detection, or identity inference fails.
    pub fn build_tracklets_from_video<P: AsRef<Path>>(
        &mut self,
        video_path: P,
    ) -> Result<Vec<Tracklet>> {
        self.build_tracklets_from_video_with_hooks(video_path, |_| {}, || false)
    }

    /// Pass-1 helper: build short-term tracklets from a source video with
    /// progress and cancellation hooks.
    ///
    /// # Errors
    ///
    /// Returns an error if decode, detection, identity inference fails, or the
    /// cancellation hook requests stop.
    pub fn build_tracklets_from_video_with_hooks<P, F, C>(
        &mut self,
        video_path: P,
        mut on_progress: F,
        mut should_cancel: C,
    ) -> Result<Vec<Tracklet>>
    where
        P: AsRef<Path>,
        F: FnMut(OfflinePrepassProgress),
        C: FnMut() -> bool,
    {
        let mut next_tracklet_id = 0usize;
        let mut tracklets = Vec::<Tracklet>::new();
        let mut sampled_frames = 0u64;
        let sample_stride = self.offline_sample_stride().max(1);

        video::for_each_rgb_frame(video_path, |frame_index, frame| {
            if should_cancel() {
                anyhow::bail!("offline prepass cancelled");
            }

            if sample_stride > 1 && !frame_index.is_multiple_of(sample_stride) {
                on_progress(OfflinePrepassProgress {
                    decoded_frames: frame_index,
                    sampled_frames,
                });
                return Ok(false);
            }

            sampled_frames = sampled_frames.saturating_add(1);
            let persons = self.detector.detect(frame)?;
            if persons.is_empty() {
                on_progress(OfflinePrepassProgress {
                    decoded_frames: frame_index,
                    sampled_frames,
                });
                return Ok(false);
            }

            let mut observations = self
                .identifier
                .observations(frame, &persons, self.tracker.search_hint())?
                .into_iter()
                .collect::<Vec<crate::detection::FaceObservation>>();

            if !observations.is_empty()
                && !self.body_gallery.is_empty()
                && let Some(body_reid) = self.body_reidentifier.as_ref()
            {
                let _ = body_reid.annotate_observations_with_gallery(
                    frame,
                    &mut observations,
                    &self.body_gallery,
                );
            }

            let mut identity_rows = observations
                .into_iter()
                .map(Into::into)
                .collect::<Vec<crate::observation::IdentityObservation>>();

            for bbox in persons {
                let identity = take_best_identity_for_bbox(&mut identity_rows, bbox)
                    .unwrap_or_else(|| {
                        crate::observation::IdentityObservation::from_face_scores(
                            bbox, 0.0, -1.0, 0.0, None,
                        )
                    });

                let best_idx = find_tracklet_assignment(
                    &tracklets,
                    frame_index,
                    bbox,
                    &identity,
                    frame.width,
                    frame.height,
                );
                if let Some(tracklet_idx) = best_idx {
                    if let Some(tracklet) = tracklets.get_mut(tracklet_idx) {
                        tracklet.push(frame_index, bbox, identity);
                        continue;
                    }
                }

                let mut tracklet = Tracklet::new(next_tracklet_id);
                next_tracklet_id = next_tracklet_id.saturating_add(1);
                tracklet.push(frame_index, bbox, identity);
                tracklets.push(tracklet);
            }

            on_progress(OfflinePrepassProgress {
                decoded_frames: frame_index,
                sampled_frames,
            });

            Ok(false)
        })?;

        Ok(tracklets
            .into_iter()
            .filter(|tracklet| tracklet.len() > 1 || tracklet.best_composite_score() >= 0.66)
            .collect())
    }

    const fn offline_sample_stride(&self) -> u64 {
        match self.mode {
            ProcessingMode::Fast => 3,
            ProcessingMode::Balanced => 2,
            ProcessingMode::Quality => 1,
        }
    }

    /// Pass-2 helper: run heuristic identity clustering on tracklets.
    #[must_use]
    pub fn solve_tracklets(&self, tracklets: &[Tracklet]) -> SolverResult {
        solver::solve(tracklets)
    }
}

fn take_best_identity_for_bbox(
    rows: &mut Vec<crate::observation::IdentityObservation>,
    bbox: BBox,
) -> Option<crate::observation::IdentityObservation> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, row) in rows.iter().enumerate() {
        let iou = row.bbox.iou(&bbox).clamp(0.0, 1.0);
        let dx = row.bbox.center_x() - bbox.center_x();
        let dy = row.bbox.center_y() - bbox.center_y();
        let center_distance = dx.hypot(dy);
        let scale = bbox.width().max(bbox.height()).max(1.0);
        let center_term = (1.0 - (center_distance / scale).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let score = iou * 0.85 + center_term * 0.15;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((idx, score));
        }
    }

    let (idx, score) = best?;
    (score >= OBS_MATCH_MIN_SCORE).then_some(rows.swap_remove(idx))
}

fn find_tracklet_assignment(
    tracklets: &[Tracklet],
    frame_index: u64,
    bbox: BBox,
    identity: &crate::observation::IdentityObservation,
    frame_width: u32,
    frame_height: u32,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;

    for (idx, tracklet) in tracklets.iter().enumerate() {
        let Some(last_obs) = tracklet.last_observation() else {
            continue;
        };

        let frame_gap = frame_index.saturating_sub(last_obs.frame_index);
        if frame_gap == 0 || frame_gap > TRACKLET_MAX_GAP_FRAMES {
            continue;
        }

        let iou = last_obs.bbox.iou(&bbox).clamp(0.0, 1.0);
        let center_distance = normalized_distance(
            last_obs.bbox.center_x(),
            last_obs.bbox.center_y(),
            bbox.center_x(),
            bbox.center_y(),
            frame_width,
            frame_height,
        );
        if iou < TRACKLET_MIN_IOU && center_distance > TRACKLET_MAX_CENTER_DISTANCE_NORM {
            continue;
        }

        let appearance = cosine_like(
            last_obs.observation.similarity,
            last_obs.observation.impostor_similarity,
            identity.similarity,
            identity.impostor_similarity,
        );

        let score = identity.composite_score().clamp(0.0, 1.0).mul_add(
            0.05,
            ((appearance + 1.0) * 0.5).clamp(0.0, 1.0).mul_add(
                0.20,
                iou * 0.45
                    + (1.0 - (center_distance / TRACKLET_MAX_CENTER_DISTANCE_NORM).clamp(0.0, 1.0))
                        * 0.30,
            ),
        ) - ((frame_gap.saturating_sub(1)) as f32 * 0.03).clamp(0.0, 0.15);

        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((idx, score));
        }
    }

    best.and_then(|(idx, score)| (score > 0.0).then_some(idx))
}

fn normalized_distance(
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    frame_width: u32,
    frame_height: u32,
) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    let distance = dx.hypot(dy);
    let norm = (frame_width.max(1) as f32)
        .hypot(frame_height.max(1) as f32)
        .max(1.0);
    (distance / norm).clamp(0.0, 1.0)
}

fn cosine_like(a_sim: f32, a_imp: f32, b_sim: f32, b_imp: f32) -> f32 {
    // Heuristic compatibility score from ranked similarities/margins.
    let a_margin = a_sim - a_imp;
    let b_margin = b_sim - b_imp;
    let dot = a_sim.mul_add(b_sim, a_margin * b_margin);
    let an = a_sim.hypot(a_margin).max(1e-5);
    let bn = b_sim.hypot(b_margin).max(1e-5);
    (dot / (an * bn)).clamp(-1.0, 1.0)
}

fn recovery_priority(obs: crate::detection::FaceObservation, last_bbox: BBox) -> f32 {
    let iou = obs.bbox.iou(&last_bbox).clamp(0.0, 1.0);
    let dx = obs.bbox.center_x() - last_bbox.center_x();
    let dy = obs.bbox.center_y() - last_bbox.center_y();
    let distance = dx.hypot(dy);
    let norm = (last_bbox.width().max(last_bbox.height()) * 6.5).max(1.0);
    let proximity = 1.0 - (distance / norm).clamp(0.0, 1.0);
    let body = obs
        .body_similarity
        .map_or(0.0, |sim| ((sim + 1.0) * 0.5).clamp(0.0, 1.0));

    obs.similarity.mul_add(0.50, obs.margin * 0.13) + iou * 0.15 + proximity * 0.08 + body * 0.14
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
    #[must_use]
    pub const fn new(renderer: FrameRenderer) -> Self {
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
                mode,
            ),
            renderer: Renderer::new(renderer),
        }
    }

    /// Loads the pipeline with the given model paths and reference image.
    ///
    /// # Arguments
    ///
    /// * `yolo_model_path` - Path to the `YOLOv8` ONNX model for person detection
    /// * `face_model_path` - Path to the `ArcFace` ONNX model for face identification
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
    /// * `yolo_model_path` - Path to the `YOLOv8` ONNX model
    /// * `face_model_path` - Path to the `ArcFace` ONNX model
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

    /// Solve an offline camera path and consume the pipeline into parts.
    ///
    /// This runs a first pass to build short-term tracklets from the video,
    /// then a second pass uses clustered camera states for render-time framing.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or inference fails during tracklet build.
    pub fn into_parts_with_offline_solution<P: AsRef<Path>>(
        self,
        video_path: P,
    ) -> Result<(Analyzer, Renderer)> {
        self.into_parts_with_offline_solution_with_hooks(video_path, |_| {}, || false)
    }

    /// Solve an offline camera path and consume the pipeline into parts,
    /// forwarding prepass progress and cancellation hooks.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or inference fails during tracklet build,
    /// or if cancellation is requested.
    pub fn into_parts_with_offline_solution_with_hooks<P, F, C>(
        mut self,
        video_path: P,
        on_progress: F,
        should_cancel: C,
    ) -> Result<(Analyzer, Renderer)>
    where
        P: AsRef<Path>,
        F: FnMut(OfflinePrepassProgress),
        C: FnMut() -> bool,
    {
        let tracklets = self.analyzer.build_tracklets_from_video_with_hooks(
            video_path,
            on_progress,
            should_cancel,
        )?;
        let solved = self.analyzer.solve_tracklets(&tracklets);
        self.analyzer.enable_offline_from_solver_result(solved);
        Ok((self.analyzer, self.renderer))
    }
}
