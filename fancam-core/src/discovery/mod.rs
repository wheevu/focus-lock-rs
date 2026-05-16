//! discovery — identity discovery for multi-person videos
//!
//! This module provides functionality to scan a video and discover distinct
//! identities present in it. This is useful for group performances where you
//! need to select which member to track.
//!
//! The discovery process:
//! 1. Samples frames from the video at regular intervals
//! 2. Detects all persons in each sampled frame
//! 3. Extracts face embeddings for each detected person
//! 4. Clusters embeddings to group observations of the same identity
//! 5. Generates thumbnails and confidence scores for each candidate

use std::cmp::Ordering;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::Error as FfmpegError;
use ffmpeg_next::{frame, media, software::scaling};
use image::codecs::jpeg::JpegEncoder;
use image::imageops;
use image::{ImageBuffer, Rgb, RgbImage};

use crate::detection::{
    BBox, Detector, FaceEmbedder, embedding_cosine_similarity, face_crop_region_for_bbox,
    face_presence_score, face_preview_score,
};
use crate::mode::ProcessingMode;
use crate::reid::BodyReidentifier;
use crate::video::{RgbFrame, open_input_with_hwaccel};

/// Default frame sampling stride (analyze every 12th frame).
const DEFAULT_SAMPLE_STRIDE: u64 = 12;
const DEFAULT_MAX_SAMPLED_FRAMES: usize = 900;
const DEFAULT_MAX_FACES_PER_FRAME: usize = 8;
const DEFAULT_CLUSTER_SIMILARITY: f32 = 0.76;
const DEFAULT_DUPLICATE_SIMILARITY: f32 = 0.86;
const DEFAULT_MIN_OBSERVATIONS: u32 = 2;
const DEFAULT_MIN_EMBEDDING_SIMILARITY: f32 = 0.68;
const DEFAULT_MIN_FACE_CROP_EDGE: u32 = 42;
const DEFAULT_MAX_CANDIDATES: usize = 18;
const DEFAULT_MAX_DUPLICATES: usize = 36;
const DEFAULT_MIN_THUMBNAIL_QUALITY: f32 = 0.54;
const DEFAULT_MIN_FACE_PRESENCE: f32 = 0.42;
const DEFAULT_MIN_FACE_PREVIEW: f32 = 0.58;
const DEFAULT_MIN_PREVIEW_SCORE: f32 = 0.46;
const DEFAULT_TRACKLET_MAX_GAP_FRAMES: u64 = 4;
const DEFAULT_TRACKLET_MIN_IOU: f32 = 0.12;
const DEFAULT_TRACKLET_MAX_CENTER_DISTANCE: f32 = 0.22;
const PROGRESS_EMIT_FRAME_INTERVAL: u64 = 24;

const MERGE_STRONG_SIMILARITY: f32 = 0.84;
const MERGE_SOFT_SIMILARITY: f32 = 0.80;
const MERGE_SOFT_ANCHOR_DISTANCE: f32 = 0.16;

/// Duplicate score floor used for embedding-driven review rows.
const DUPLICATE_EMBEDDING_FLOOR: f32 = 0.70;
/// Maximum anchor distance considered for duplicate scoring.
const DUPLICATE_ANCHOR_DISTANCE_MAX: f32 = 0.28;
/// Maximum confidence gap considered for duplicate scoring.
const DUPLICATE_CONFIDENCE_GAP_MAX: f32 = 0.30;

/// Configuration for identity discovery.
///
/// Controls how frames are sampled and how faces are clustered into identities.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Process every Nth frame (higher = faster but may miss brief appearances).
    pub sample_stride: u64,
    /// Maximum number of frames to sample (limits processing time for long videos).
    pub max_sampled_frames: usize,
    /// Maximum faces to process per frame (prioritizes highest confidence detections).
    pub max_faces_per_frame: usize,
    /// Minimum cosine similarity to merge faces into the same cluster (0.0-1.0).
    pub cluster_similarity: f32,
    /// Minimum similarity to flag clusters as potential duplicates (0.0-1.0).
    pub duplicate_similarity: f32,
    /// Minimum observations required for a cluster to become a candidate.
    pub min_observations: u32,
    /// Minimum average embedding consistency for a candidate.
    pub min_embedding_similarity: f32,
    /// Minimum face crop edge in pixels required to run `ArcFace` embedding.
    pub min_face_crop_edge: u32,
    /// Maximum number of candidates returned after ranking.
    pub max_candidates: usize,
    /// Maximum number of duplicate rows emitted for review.
    pub max_duplicates: usize,
    /// Minimum thumbnail quality score for candidate emission.
    pub min_thumbnail_quality: f32,
    /// Minimum face-presence confidence required for sampled observations.
    pub min_face_presence: f32,
    /// Minimum strict preview score required for candidate thumbnails.
    pub min_face_preview: f32,
    /// Minimum average preview score required for candidate emission.
    pub min_preview_score: f32,
    /// Maximum sampled-frame gap that still allows tracklet continuation.
    pub tracklet_max_gap_frames: u64,
    /// Minimum `IoU` with last tracklet box for spatial continuation.
    pub tracklet_min_iou: f32,
    /// Maximum normalized center distance (0..1 over image diagonal) for continuation.
    pub tracklet_max_center_distance: f32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            sample_stride: DEFAULT_SAMPLE_STRIDE,
            max_sampled_frames: DEFAULT_MAX_SAMPLED_FRAMES,
            max_faces_per_frame: DEFAULT_MAX_FACES_PER_FRAME,
            cluster_similarity: DEFAULT_CLUSTER_SIMILARITY,
            duplicate_similarity: DEFAULT_DUPLICATE_SIMILARITY,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            min_embedding_similarity: DEFAULT_MIN_EMBEDDING_SIMILARITY,
            min_face_crop_edge: DEFAULT_MIN_FACE_CROP_EDGE,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            max_duplicates: DEFAULT_MAX_DUPLICATES,
            min_thumbnail_quality: DEFAULT_MIN_THUMBNAIL_QUALITY,
            min_face_presence: DEFAULT_MIN_FACE_PRESENCE,
            min_face_preview: DEFAULT_MIN_FACE_PREVIEW,
            min_preview_score: DEFAULT_MIN_PREVIEW_SCORE,
            tracklet_max_gap_frames: DEFAULT_TRACKLET_MAX_GAP_FRAMES,
            tracklet_min_iou: DEFAULT_TRACKLET_MIN_IOU,
            tracklet_max_center_distance: DEFAULT_TRACKLET_MAX_CENTER_DISTANCE,
        }
    }
}

impl DiscoveryConfig {
    /// Build a mode-tuned config for discovery.
    #[must_use]
    pub const fn for_mode(mode: ProcessingMode) -> Self {
        match mode {
            ProcessingMode::Fast => Self {
                sample_stride: 18,
                max_sampled_frames: 520,
                max_faces_per_frame: 4,
                cluster_similarity: 0.79,
                duplicate_similarity: 0.88,
                min_observations: 3,
                min_embedding_similarity: 0.72,
                min_face_crop_edge: 52,
                max_candidates: 12,
                max_duplicates: 24,
                min_thumbnail_quality: 0.60,
                min_face_presence: 0.52,
                min_face_preview: 0.62,
                min_preview_score: 0.50,
                tracklet_max_gap_frames: 3,
                tracklet_min_iou: 0.15,
                tracklet_max_center_distance: 0.20,
            },
            ProcessingMode::Balanced => Self {
                sample_stride: 12,
                max_sampled_frames: 900,
                max_faces_per_frame: 8,
                cluster_similarity: 0.77,
                duplicate_similarity: 0.87,
                min_observations: 2,
                min_embedding_similarity: 0.69,
                min_face_crop_edge: 46,
                max_candidates: 18,
                max_duplicates: 36,
                min_thumbnail_quality: 0.54,
                min_face_presence: 0.44,
                min_face_preview: 0.58,
                min_preview_score: 0.46,
                tracklet_max_gap_frames: 4,
                tracklet_min_iou: 0.12,
                tracklet_max_center_distance: 0.22,
            },
            ProcessingMode::Quality => Self {
                sample_stride: 8,
                max_sampled_frames: 1400,
                max_faces_per_frame: 10,
                cluster_similarity: 0.75,
                duplicate_similarity: 0.85,
                min_observations: 2,
                min_embedding_similarity: 0.66,
                min_face_crop_edge: 38,
                max_candidates: 28,
                max_duplicates: 60,
                min_thumbnail_quality: 0.48,
                min_face_presence: 0.36,
                min_face_preview: 0.54,
                min_preview_score: 0.42,
                tracklet_max_gap_frames: 5,
                tracklet_min_iou: 0.10,
                tracklet_max_center_distance: 0.25,
            },
        }
    }

    /// Returns a tighter pass intended to improve under-counted scans.
    #[must_use]
    pub fn informed_under_count_pass(&self) -> Self {
        Self {
            sample_stride: self.sample_stride.saturating_div(2).max(4),
            max_sampled_frames: self.max_sampled_frames.saturating_mul(2),
            max_faces_per_frame: (self.max_faces_per_frame + 2).min(10),
            cluster_similarity: (self.cluster_similarity - 0.02).max(0.70),
            duplicate_similarity: (self.duplicate_similarity - 0.01).max(0.82),
            min_observations: self.min_observations.max(2),
            min_embedding_similarity: (self.min_embedding_similarity - 0.02).max(0.64),
            min_face_crop_edge: self.min_face_crop_edge.saturating_sub(6).max(34),
            max_candidates: self.max_candidates.saturating_add(10).min(42),
            max_duplicates: self.max_duplicates.saturating_add(24).min(96),
            min_thumbnail_quality: (self.min_thumbnail_quality - 0.04).max(0.40),
            min_face_presence: (self.min_face_presence - 0.04).max(0.28),
            min_face_preview: (self.min_face_preview - 0.03).max(0.44),
            min_preview_score: (self.min_preview_score - 0.03).max(0.36),
            tracklet_max_gap_frames: self.tracklet_max_gap_frames.saturating_add(1).min(7),
            tracklet_min_iou: (self.tracklet_min_iou - 0.02).max(0.08),
            tracklet_max_center_distance: (self.tracklet_max_center_distance + 0.03).min(0.30),
        }
    }
}

/// A pair of identity candidates flagged as potential duplicates.
#[derive(Debug, Clone)]
pub struct DuplicatePair {
    /// ID of the first candidate.
    pub a: usize,
    /// ID of the second candidate.
    pub b: usize,
    /// Cosine similarity between their centroids (0.0-1.0).
    pub similarity: f32,
}

/// A discovered identity candidate.
///
/// Represents a distinct person found in the video, with metadata about
/// their appearance frequency and a representative thumbnail.
#[derive(Debug, Clone)]
pub struct IdentityCandidate {
    /// Unique identifier for this candidate (index in the candidates list).
    pub id: usize,
    /// Confidence score (0.0-1.0) based on observation count and detection confidence.
    pub confidence: f32,
    /// Number of times this identity was observed across sampled frames.
    pub observations: u32,
    /// First frame index where this identity was seen.
    pub first_frame: u64,
    /// Last frame index where this identity was seen.
    pub last_frame: u64,
    /// Average X position in source-frame pixel coordinates for initial search hint.
    pub anchor_x: f32,
    /// Average Y position in source-frame pixel coordinates for initial search hint.
    pub anchor_y: f32,
    /// Average normalized X position in source-frame coordinates (0.0-1.0).
    pub anchor_x_norm: f32,
    /// Average normalized Y position in source-frame coordinates (0.0-1.0).
    pub anchor_y_norm: f32,
    /// JPEG-encoded thumbnail image for UI display.
    pub thumbnail_jpeg: Vec<u8>,
    /// Centroid embedding used for tracking handoff.
    pub embedding: Vec<f32>,
    /// Optional body embedding centroid used for runtime body `ReID` gallery.
    pub body_embedding: Option<Vec<f32>>,
    /// Average strict preview quality over preview-eligible observations.
    pub preview_score: f32,
    /// Number of preview-eligible observations contributing to this identity.
    pub preview_observations: u32,
}

/// Report generated by identity discovery.
///
/// Contains all discovered candidates and any duplicate pairs that may
/// need user review.
#[derive(Debug, Clone)]
pub struct DiscoveryReport {
    /// Number of frames that were sampled and analyzed.
    pub sampled_frames: u64,
    /// Total number of frames decoded (including skipped frames).
    pub total_decoded_frames: u64,
    /// List of discovered identity candidates.
    pub candidates: Vec<IdentityCandidate>,
    /// Pairs of candidates flagged as potential duplicates.
    pub duplicates: Vec<DuplicatePair>,
    /// Number of sampled embeddings skipped due to tiny/invalid face crops.
    pub rejected_embeddings: u64,
    /// Number of clusters suppressed by precision filtering.
    pub suppressed_clusters: usize,
    /// Number of automatic merges applied to reduce fragmentation.
    pub merged_clusters: usize,
    /// Number of provisional tracklets built from sampled detections.
    pub provisional_tracklets: usize,
}

/// Engine for discovering identities in videos.
///
/// Loads the required ML models and provides methods to scan videos
/// and build a list of distinct identities present.
#[derive(Debug)]
pub struct DiscoveryEngine {
    detector: Detector,
    embedder: Arc<FaceEmbedder>,
    body_reidentifier: Option<BodyReidentifier>,
}

impl DiscoveryEngine {
    /// Loads the discovery engine with the given model paths.
    ///
    /// # Arguments
    ///
    /// * `yolo_model_path` - Path to the `YOLOv8` ONNX model for person detection
    /// * `face_model_path` - Path to the `ArcFace` ONNX model for face embedding
    ///
    /// # Errors
    ///
    /// Returns an error if either model cannot be loaded.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
    ) -> Result<Self> {
        Self::load_with_body_reid(yolo_model_path, face_model_path, None::<&str>)
    }

    /// Loads the discovery engine with optional body `ReID` support.
    ///
    /// # Errors
    ///
    /// Returns an error if required models cannot be loaded.
    pub fn load_with_body_reid<P, Q, R>(
        yolo_model_path: P,
        face_model_path: Q,
        body_reid_model_path: Option<R>,
    ) -> Result<Self>
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
        R: AsRef<Path>,
    {
        Ok(Self {
            detector: Detector::load(yolo_model_path)?,
            embedder: Arc::new(FaceEmbedder::load(face_model_path)?),
            body_reidentifier: body_reid_model_path
                .map(BodyReidentifier::load)
                .transpose()?,
        })
    }

    /// Scans a video to discover distinct identities.
    ///
    /// Samples frames from the video according to the config, detects persons,
    /// extracts face embeddings, and clusters them into distinct identities.
    ///
    /// # Arguments
    ///
    /// * `video_path` - Path to the video file to scan
    /// * `config` - Configuration controlling sampling and clustering behavior
    ///
    /// # Errors
    ///
    /// Returns an error if the video cannot be opened or processed.
    pub fn scan_video<P: AsRef<Path>>(
        &mut self,
        video_path: P,
        config: &DiscoveryConfig,
    ) -> Result<DiscoveryReport> {
        self.scan_video_with_hooks(video_path, config, |_, _| {}, || false)
    }

    /// Scans a video to discover distinct identities, with progress and cancellation hooks.
    pub fn scan_video_with_hooks<P, F, C>(
        &mut self,
        video_path: P,
        config: &DiscoveryConfig,
        mut on_progress: F,
        mut should_cancel: C,
    ) -> Result<DiscoveryReport>
    where
        P: AsRef<Path>,
        F: FnMut(u64, u64),
        C: FnMut() -> bool,
    {
        ffmpeg::init().context("failed to initialize ffmpeg for identity discovery")?;

        let mut ictx =
            open_input_with_hwaccel(&video_path).context("failed to open input video")?;
        let (stream_index, codecpar) = {
            let stream = ictx
                .streams()
                .best(media::Type::Video)
                .context("no video stream found")?;
            (stream.index(), stream.parameters())
        };
        let mut decoder = ffmpeg::codec::Context::from_parameters(codecpar)
            .context("failed to create decoder context")?
            .decoder()
            .video()
            .context("failed to open decoder")?;

        let mut to_rgb = scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            scaling::Flags::BILINEAR,
        )
        .context("failed to create rgb scaler")?;

        let mut decoded = frame::Video::empty();
        let mut rgb_frame = frame::Video::empty();

        let mut frame_index = 0u64;
        let mut sampled_frames = 0u64;
        let mut rejected_embeddings = 0u64;
        let mut tracklets: Vec<ProvisionalTracklet> = Vec::with_capacity(24);

        let frame_width = decoder.width().max(1) as f32;
        let frame_height = decoder.height().max(1) as f32;

        let mut process = |src: &frame::Video| -> Result<bool> {
            frame_index += 1;
            if should_cancel() {
                anyhow::bail!("identity scan cancelled");
            }
            if frame_index.is_multiple_of(PROGRESS_EMIT_FRAME_INTERVAL) {
                on_progress(sampled_frames, frame_index);
            }
            if config.sample_stride > 1 && !frame_index.is_multiple_of(config.sample_stride) {
                return Ok(false);
            }
            if sampled_frames as usize >= config.max_sampled_frames {
                return Ok(true);
            }

            to_rgb
                .run(src, &mut rgb_frame)
                .context("failed to convert frame to rgb")?;
            let rgb = copy_rgb_frame(&rgb_frame, frame_index);
            sampled_frames += 1;
            on_progress(sampled_frames, frame_index);

            let mut persons = self
                .detector
                .detect(&rgb)
                .context("person detection failed during identity discovery")?;
            persons.sort_unstable_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(Ordering::Equal)
            });
            persons.truncate(config.max_faces_per_frame.max(1));

            let face_presence_scores = persons
                .iter()
                .map(|bbox| face_presence_score(&rgb, *bbox))
                .collect::<Vec<_>>();
            let face_preview_scores = persons
                .iter()
                .map(|bbox| face_preview_score(&rgb, *bbox))
                .collect::<Vec<_>>();

            let embedding_inputs = persons
                .iter()
                .copied()
                .zip(face_presence_scores.iter().copied())
                .filter_map(|(bbox, face_presence)| {
                    (face_presence >= config.min_face_presence).then_some(bbox)
                })
                .collect::<Vec<_>>();

            rejected_embeddings += persons.len().saturating_sub(embedding_inputs.len()) as u64;

            let mut face_rows = self.embedder.embed_many_from_bboxes(
                &rgb,
                &embedding_inputs,
                config.min_face_crop_edge,
            )?;
            rejected_embeddings += embedding_inputs.len().saturating_sub(face_rows.len()) as u64;

            let mut body_rows = if let Some(body_reidentifier) = self.body_reidentifier.as_ref() {
                body_reidentifier
                    .embed_many_from_bboxes(&rgb, &persons)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let mut observations = Vec::with_capacity(persons.len());
            for (index, bbox) in persons.iter().copied().enumerate() {
                if should_cancel() {
                    anyhow::bail!("identity scan cancelled");
                }

                let face_presence = face_presence_scores.get(index).copied().unwrap_or_default();
                let face_preview = face_preview_scores.get(index).copied().unwrap_or_default();
                let face_embedding = if face_presence >= config.min_face_presence {
                    take_embedding_for_bbox(&mut face_rows, bbox)
                } else {
                    None
                };
                let body_embedding = take_embedding_for_bbox(&mut body_rows, bbox);

                let anchor_x = bbox.center_x();
                let anchor_y = bbox.center_y();
                observations.push(FrameDetectionObservation {
                    frame_index,
                    sampled_index: sampled_frames,
                    bbox,
                    anchor_x,
                    anchor_y,
                    anchor_x_norm: (anchor_x / frame_width).clamp(0.0, 1.0),
                    anchor_y_norm: (anchor_y / frame_height).clamp(0.0, 1.0),
                    face_presence,
                    face_preview,
                    preview_eligible: face_preview >= config.min_face_preview,
                    face_embedding,
                    body_embedding,
                });
            }

            assign_observations_to_tracklets(&mut tracklets, observations, &rgb, config);
            Ok(sampled_frames as usize >= config.max_sampled_frames)
        };

        let mut limit_reached = false;
        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder
                .send_packet(&packet)
                .context("failed to send packet to decoder")?;
            loop {
                match decoder.receive_frame(&mut decoded) {
                    Ok(()) => {
                        if process(&decoded)? {
                            limit_reached = true;
                            break;
                        }
                    }
                    Err(FfmpegError::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                    Err(FfmpegError::Eof) => break,
                    Err(err) => {
                        return Err(anyhow::anyhow!(
                            "decoder receive_frame failed during scan: {err}"
                        ));
                    }
                }
            }
            if limit_reached {
                break;
            }
        }

        if !limit_reached {
            decoder.send_eof().ok();
            loop {
                match decoder.receive_frame(&mut decoded) {
                    Ok(()) => {
                        if process(&decoded)? {
                            break;
                        }
                    }
                    Err(FfmpegError::Eof) => break,
                    Err(FfmpegError::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                    Err(err) => {
                        return Err(anyhow::anyhow!(
                            "decoder drain receive_frame failed during scan: {err}"
                        ));
                    }
                }
            }
        }

        let provisional_tracklets = tracklets.len();

        let mut clusters = tracklets
            .into_iter()
            .filter_map(tracklet_to_cluster_seed)
            .map(Cluster::new)
            .collect::<Vec<_>>();

        let merged_clusters = merge_clusters(&mut clusters);

        let cluster_count_before_filter = clusters.len();
        let mut candidates: Vec<IdentityCandidate> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(id, cluster)| cluster.into_candidate(id, config))
            .collect();

        let suppressed_clusters = cluster_count_before_filter.saturating_sub(candidates.len());

        candidates.sort_unstable_by(|a, b| {
            b.preview_score
                .partial_cmp(&a.preview_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| b.preview_observations.cmp(&a.preview_observations))
                .then_with(|| b.observations.cmp(&a.observations))
        });

        if candidates.len() > config.max_candidates {
            candidates.truncate(config.max_candidates);
        }

        let duplicates = collect_duplicates(&candidates, config);

        Ok(DiscoveryReport {
            sampled_frames,
            total_decoded_frames: frame_index,
            candidates,
            duplicates,
            rejected_embeddings,
            suppressed_clusters,
            merged_clusters,
            provisional_tracklets,
        })
    }
}

#[derive(Clone)]
struct FrameDetectionObservation {
    frame_index: u64,
    sampled_index: u64,
    bbox: BBox,
    anchor_x: f32,
    anchor_y: f32,
    anchor_x_norm: f32,
    anchor_y_norm: f32,
    face_presence: f32,
    face_preview: f32,
    preview_eligible: bool,
    face_embedding: Option<Vec<f32>>,
    body_embedding: Option<Vec<f32>>,
}

#[derive(Clone)]
struct TrackletObservation {
    frame_index: u64,
    bbox: BBox,
    anchor_x: f32,
    anchor_y: f32,
    anchor_x_norm: f32,
    anchor_y_norm: f32,
    face_presence: f32,
    face_preview: f32,
    preview_eligible: bool,
    face_embedding: Option<Vec<f32>>,
    body_embedding: Option<Vec<f32>>,
}

struct ProvisionalTracklet {
    last_sampled_index: u64,
    observations: Vec<TrackletObservation>,
    best_preview_score: f32,
    best_preview_jpeg: Vec<u8>,
    last_face_embedding: Option<Vec<f32>>,
    last_body_embedding: Option<Vec<f32>>,
    preview_observations: u32,
    body_support_observations: u32,
}

struct ClusterSeed {
    centroid: Vec<f32>,
    body_centroid: Option<Vec<f32>>,
    confidence_sum: f32,
    observations: u32,
    first_frame: u64,
    last_frame: u64,
    anchor_x_acc: f32,
    anchor_y_acc: f32,
    anchor_x_norm_acc: f32,
    anchor_y_norm_acc: f32,
    embedding_sim_sum: f32,
    face_presence_sum: f32,
    preview_sum: f32,
    preview_observations: u32,
    body_support_observations: u32,
    thumbnail_score: f32,
    thumbnail_jpeg: Vec<u8>,
    strong_face_observations: u32,
}

struct Cluster {
    centroid: Vec<f32>,
    body_centroid: Option<Vec<f32>>,
    confidence_sum: f32,
    observations: u32,
    first_frame: u64,
    last_frame: u64,
    anchor_x_acc: f32,
    anchor_y_acc: f32,
    anchor_x_norm_acc: f32,
    anchor_y_norm_acc: f32,
    embedding_sim_sum: f32,
    face_presence_sum: f32,
    preview_sum: f32,
    preview_observations: u32,
    body_support_observations: u32,
    thumbnail_score: f32,
    thumbnail_jpeg: Vec<u8>,
    strong_face_observations: u32,
}

impl Cluster {
    fn new(seed: ClusterSeed) -> Self {
        let thumbnail_score = if seed.thumbnail_jpeg.is_empty() {
            0.0
        } else {
            seed.thumbnail_score
        };
        Self {
            centroid: seed.centroid,
            body_centroid: seed.body_centroid,
            confidence_sum: seed.confidence_sum,
            observations: seed.observations,
            first_frame: seed.first_frame,
            last_frame: seed.last_frame,
            anchor_x_acc: seed.anchor_x_acc,
            anchor_y_acc: seed.anchor_y_acc,
            anchor_x_norm_acc: seed.anchor_x_norm_acc,
            anchor_y_norm_acc: seed.anchor_y_norm_acc,
            embedding_sim_sum: seed.embedding_sim_sum,
            face_presence_sum: seed.face_presence_sum,
            preview_sum: seed.preview_sum,
            preview_observations: seed.preview_observations,
            body_support_observations: seed.body_support_observations,
            thumbnail_score,
            thumbnail_jpeg: seed.thumbnail_jpeg,
            strong_face_observations: seed.strong_face_observations,
        }
    }

    fn into_candidate(self, id: usize, config: &DiscoveryConfig) -> Option<IdentityCandidate> {
        if self.observations < config.min_observations {
            return None;
        }
        if self.preview_observations == 0 {
            return None;
        }
        if self.centroid.is_empty() {
            return None;
        }
        let avg_embedding_similarity = self.embedding_sim_sum / self.observations as f32;
        if avg_embedding_similarity < config.min_embedding_similarity {
            return None;
        }
        if self.thumbnail_score < config.min_thumbnail_quality {
            return None;
        }
        let avg_face_presence = self.face_presence_sum / self.observations as f32;
        if avg_face_presence < config.min_face_presence {
            return None;
        }
        if self.strong_face_observations < (config.min_observations / 2).max(1) {
            return None;
        }
        if self.thumbnail_jpeg.is_empty() {
            return None;
        }
        let avg_preview_score = self.preview_sum / self.preview_observations as f32;
        if avg_preview_score < config.min_preview_score {
            return None;
        }
        let body_support_ratio =
            self.body_support_observations as f32 / self.observations.max(1) as f32;
        let avg_score = self.confidence_sum / self.observations as f32;
        let confidence = (self.thumbnail_score.mul_add(
            0.10,
            avg_preview_score.mul_add(
                0.28,
                avg_face_presence.mul_add(
                    0.08,
                    avg_embedding_similarity.mul_add(
                        0.20,
                        avg_score.mul_add(0.12, 0.24 + (self.observations as f32 * 0.03).min(0.16)),
                    ),
                ),
            ),
        ) + body_support_ratio * 0.04)
            .clamp(0.0, 0.995);
        Some(IdentityCandidate {
            id,
            confidence,
            observations: self.observations,
            first_frame: self.first_frame,
            last_frame: self.last_frame,
            anchor_x: self.anchor_x_acc / self.observations as f32,
            anchor_y: self.anchor_y_acc / self.observations as f32,
            anchor_x_norm: self.anchor_x_norm_acc / self.observations as f32,
            anchor_y_norm: self.anchor_y_norm_acc / self.observations as f32,
            thumbnail_jpeg: self.thumbnail_jpeg,
            embedding: self.centroid,
            body_embedding: self.body_centroid,
            preview_score: avg_preview_score,
            preview_observations: self.preview_observations,
        })
    }
}

fn collect_duplicates(
    candidates: &[IdentityCandidate],
    config: &DiscoveryConfig,
) -> Vec<DuplicatePair> {
    let mut pairs = Vec::new();
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let score = duplicate_similarity_score(&candidates[i], &candidates[j]);
            if score >= config.duplicate_similarity {
                pairs.push(DuplicatePair {
                    a: candidates[i].id,
                    b: candidates[j].id,
                    similarity: score,
                });
            }
        }
    }
    pairs.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(Ordering::Equal)
    });
    if pairs.len() > config.max_duplicates {
        pairs.truncate(config.max_duplicates);
    }
    pairs
}

fn anchor_similarity(a: &IdentityCandidate, b: &IdentityCandidate) -> f32 {
    let dx = a.anchor_x_norm - b.anchor_x_norm;
    let dy = a.anchor_y_norm - b.anchor_y_norm;
    let distance = dx.hypot(dy);
    (1.0 - (distance / 0.75).clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn duplicate_similarity_score(a: &IdentityCandidate, b: &IdentityCandidate) -> f32 {
    let embedding = embedding_cosine_similarity(&a.embedding, &b.embedding).clamp(0.0, 1.0);
    if embedding < DUPLICATE_EMBEDDING_FLOOR {
        return 0.0;
    }

    let anchor_score = anchor_similarity(a, b);
    let dx = a.anchor_x_norm - b.anchor_x_norm;
    let dy = a.anchor_y_norm - b.anchor_y_norm;
    let anchor_distance = dx.hypot(dy);
    if anchor_distance > DUPLICATE_ANCHOR_DISTANCE_MAX {
        return 0.0;
    }

    let time_overlap = temporal_overlap_score(a, b);
    let confidence_gap = (a.confidence - b.confidence).abs();
    let confidence_score = (1.0 - confidence_gap / DUPLICATE_CONFIDENCE_GAP_MAX).clamp(0.0, 1.0);

    (time_overlap.mul_add(0.08, embedding * 0.78 + anchor_score * 0.10) + confidence_score * 0.04)
        .clamp(0.0, 1.0)
}

fn temporal_overlap_score(a: &IdentityCandidate, b: &IdentityCandidate) -> f32 {
    let overlap_start = a.first_frame.max(b.first_frame);
    let overlap_end = a.last_frame.min(b.last_frame);
    if overlap_end < overlap_start {
        return 0.0;
    }

    let overlap = overlap_end - overlap_start + 1;
    let span_a = a.last_frame.saturating_sub(a.first_frame) + 1;
    let span_b = b.last_frame.saturating_sub(b.first_frame) + 1;
    let denom = span_a.min(span_b).max(1);
    (overlap as f32 / denom as f32).clamp(0.0, 1.0)
}

fn merge_clusters(clusters: &mut Vec<Cluster>) -> usize {
    let mut merges = 0usize;
    loop {
        let mut best_pair: Option<(usize, usize, f32)> = None;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let sim = embedding_cosine_similarity(&clusters[i].centroid, &clusters[j].centroid);
                let anchor_dist = cluster_anchor_distance(&clusters[i], &clusters[j]);
                let observations = clusters[i].observations + clusters[j].observations;
                let mergeable = sim >= MERGE_STRONG_SIMILARITY
                    || (sim >= MERGE_SOFT_SIMILARITY
                        && anchor_dist <= MERGE_SOFT_ANCHOR_DISTANCE
                        && observations >= 5);
                if !mergeable {
                    continue;
                }
                if best_pair.is_none_or(|(_, _, best)| sim > best) {
                    best_pair = Some((i, j, sim));
                }
            }
        }

        let Some((left, right, _)) = best_pair else {
            break;
        };
        let right_cluster = clusters.swap_remove(right);
        merge_two_clusters(&mut clusters[left], right_cluster);
        merges += 1;
    }
    merges
}

fn merge_two_clusters(target: &mut Cluster, other: Cluster) {
    let left_obs = target.observations as f32;
    let right_obs = other.observations as f32;
    let denom = (left_obs + right_obs).max(1.0);

    for (t, o) in target.centroid.iter_mut().zip(other.centroid.iter()) {
        *t = (*t).mul_add(left_obs, *o * right_obs) / denom;
    }
    target.centroid = l2_normalize(&target.centroid);
    match (&mut target.body_centroid, other.body_centroid) {
        (Some(target_body), Some(other_body)) => {
            for (t, o) in target_body.iter_mut().zip(other_body.iter()) {
                *t = (*t).mul_add(left_obs, *o * right_obs) / denom;
            }
            *target_body = l2_normalize(target_body);
        }
        (None, Some(other_body)) => {
            target.body_centroid = Some(other_body);
        }
        _ => {}
    }
    target.confidence_sum += other.confidence_sum;
    target.embedding_sim_sum += other.embedding_sim_sum;
    target.face_presence_sum += other.face_presence_sum;
    target.preview_sum += other.preview_sum;
    target.preview_observations = target
        .preview_observations
        .saturating_add(other.preview_observations);
    target.body_support_observations = target
        .body_support_observations
        .saturating_add(other.body_support_observations);
    target.observations += other.observations;
    target.strong_face_observations = target
        .strong_face_observations
        .saturating_add(other.strong_face_observations);
    target.first_frame = target.first_frame.min(other.first_frame);
    target.last_frame = target.last_frame.max(other.last_frame);
    target.anchor_x_acc += other.anchor_x_acc;
    target.anchor_y_acc += other.anchor_y_acc;
    target.anchor_x_norm_acc += other.anchor_x_norm_acc;
    target.anchor_y_norm_acc += other.anchor_y_norm_acc;
    if other.thumbnail_score > target.thumbnail_score {
        target.thumbnail_score = other.thumbnail_score;
        target.thumbnail_jpeg = other.thumbnail_jpeg;
    }
}

fn cluster_anchor_distance(a: &Cluster, b: &Cluster) -> f32 {
    let ax = a.anchor_x_norm_acc / a.observations.max(1) as f32;
    let ay = a.anchor_y_norm_acc / a.observations.max(1) as f32;
    let bx = b.anchor_x_norm_acc / b.observations.max(1) as f32;
    let by = b.anchor_y_norm_acc / b.observations.max(1) as f32;
    let dx = ax - bx;
    let dy = ay - by;
    dx.hypot(dy)
}

impl ProvisionalTracklet {
    fn new(obs: FrameDetectionObservation, frame: &RgbFrame) -> Self {
        let quality = thumbnail_quality(frame, obs.bbox, 1.0, obs.face_presence);
        let thumbnail_jpeg = if obs.preview_eligible {
            thumbnail_from_bbox(frame, obs.bbox, true).unwrap_or_default()
        } else {
            Vec::new()
        };
        let tracklet_obs = TrackletObservation {
            frame_index: obs.frame_index,
            bbox: obs.bbox,
            anchor_x: obs.anchor_x,
            anchor_y: obs.anchor_y,
            anchor_x_norm: obs.anchor_x_norm,
            anchor_y_norm: obs.anchor_y_norm,
            face_presence: obs.face_presence,
            face_preview: obs.face_preview,
            preview_eligible: obs.preview_eligible,
            face_embedding: obs.face_embedding.clone(),
            body_embedding: obs.body_embedding.clone(),
        };
        let body_support_observations = u32::from(tracklet_obs.body_embedding.is_some());

        Self {
            last_sampled_index: obs.sampled_index,
            observations: vec![tracklet_obs],
            best_preview_score: if thumbnail_jpeg.is_empty() {
                0.0
            } else {
                obs.face_preview.max(quality)
            },
            best_preview_jpeg: thumbnail_jpeg,
            last_face_embedding: obs.face_embedding,
            last_body_embedding: obs.body_embedding,
            preview_observations: u32::from(obs.preview_eligible),
            body_support_observations,
        }
    }

    fn push_observation(
        &mut self,
        obs: FrameDetectionObservation,
        frame: &RgbFrame,
        appearance_similarity: f32,
    ) {
        let quality = thumbnail_quality(
            frame,
            obs.bbox,
            appearance_similarity.clamp(0.0, 1.0),
            obs.face_presence,
        );
        if obs.preview_eligible {
            if obs.face_preview >= self.best_preview_score
                && let Ok(thumb) = thumbnail_from_bbox(frame, obs.bbox, true)
                && !thumb.is_empty()
            {
                self.best_preview_score = obs.face_preview.max(quality);
                self.best_preview_jpeg = thumb;
            }
            self.preview_observations = self.preview_observations.saturating_add(1);
        }
        if obs.body_embedding.is_some() {
            self.body_support_observations = self.body_support_observations.saturating_add(1);
        }

        self.last_sampled_index = obs.sampled_index;
        self.last_face_embedding = obs.face_embedding.clone();
        self.last_body_embedding = obs.body_embedding.clone();
        self.observations.push(TrackletObservation {
            frame_index: obs.frame_index,
            bbox: obs.bbox,
            anchor_x: obs.anchor_x,
            anchor_y: obs.anchor_y,
            anchor_x_norm: obs.anchor_x_norm,
            anchor_y_norm: obs.anchor_y_norm,
            face_presence: obs.face_presence,
            face_preview: obs.face_preview,
            preview_eligible: obs.preview_eligible,
            face_embedding: obs.face_embedding,
            body_embedding: obs.body_embedding,
        });
    }

    fn preview_embedding(&self) -> Option<&[f32]> {
        self.observations
            .iter()
            .rev()
            .find(|obs| obs.preview_eligible)
            .and_then(|obs| obs.face_embedding.as_deref())
            .or(self.last_face_embedding.as_deref())
    }

    fn last_observation(&self) -> Option<&TrackletObservation> {
        self.observations.last()
    }
}

fn assign_observations_to_tracklets(
    tracklets: &mut Vec<ProvisionalTracklet>,
    mut observations: Vec<FrameDetectionObservation>,
    frame: &RgbFrame,
    config: &DiscoveryConfig,
) {
    observations.sort_unstable_by(|a, b| {
        b.bbox
            .confidence
            .partial_cmp(&a.bbox.confidence)
            .unwrap_or(Ordering::Equal)
    });

    for obs in observations {
        let best_assignment = best_tracklet_assignment(tracklets, &obs, config);
        if let Some((idx, appearance_similarity)) = best_assignment {
            if let Some(tracklet) = tracklets.get_mut(idx) {
                tracklet.push_observation(obs, frame, appearance_similarity);
                continue;
            }
        }
        tracklets.push(ProvisionalTracklet::new(obs, frame));
    }
}

fn best_tracklet_assignment(
    tracklets: &[ProvisionalTracklet],
    obs: &FrameDetectionObservation,
    config: &DiscoveryConfig,
) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32, f32)> = None;

    for (idx, tracklet) in tracklets.iter().enumerate() {
        let Some(last_obs) = tracklet.last_observation() else {
            continue;
        };

        let gap = obs
            .sampled_index
            .saturating_sub(tracklet.last_sampled_index);
        if gap == 0 || gap > config.tracklet_max_gap_frames {
            continue;
        }

        let iou = last_obs.bbox.iou(&obs.bbox).clamp(0.0, 1.0);
        let center_distance = normalized_anchor_distance(
            last_obs.anchor_x_norm,
            last_obs.anchor_y_norm,
            obs.anchor_x_norm,
            obs.anchor_y_norm,
        );
        if iou < config.tracklet_min_iou && center_distance > config.tracklet_max_center_distance {
            continue;
        }

        let face_similarity = tracklet
            .preview_embedding()
            .zip(obs.face_embedding.as_deref())
            .map(|(a, b)| embedding_cosine_similarity(a, b));
        if let Some(sim) = face_similarity
            && sim < 0.30
        {
            continue;
        }

        let body_similarity = tracklet
            .last_body_embedding
            .as_ref()
            .zip(obs.body_embedding.as_ref())
            .map(|(a, b)| embedding_cosine_similarity(a, b));
        if face_similarity.is_none()
            && let Some(sim) = body_similarity
            && sim < -0.20
        {
            continue;
        }

        let proximity = (1.0 - (center_distance / config.tracklet_max_center_distance.max(1e-4)))
            .clamp(0.0, 1.0);
        let face_term = face_similarity.unwrap_or(0.0).clamp(0.0, 1.0);
        let body_term = body_similarity.unwrap_or(0.0).clamp(0.0, 1.0);
        let gap_penalty = ((gap.saturating_sub(1)) as f32 * 0.03).clamp(0.0, 0.18);
        let score = obs.bbox.confidence.clamp(0.0, 1.0).mul_add(
            0.02,
            iou * 0.45 + proximity * 0.25 + face_term * 0.20 + body_term * 0.08,
        ) - gap_penalty;

        if best.is_none_or(|(_, best_score, _)| score > best_score) {
            best = Some((idx, score, face_similarity.unwrap_or(1.0).clamp(0.0, 1.0)));
        }
    }

    best.and_then(|(idx, score, similarity)| (score > 0.0).then_some((idx, similarity)))
}

fn tracklet_to_cluster_seed(tracklet: ProvisionalTracklet) -> Option<ClusterSeed> {
    let preview_face_rows = tracklet
        .observations
        .iter()
        .filter_map(|obs| {
            obs.preview_eligible
                .then_some(obs)
                .and_then(|preview_obs| preview_obs.face_embedding.as_ref())
                .filter(|emb| !emb.is_empty())
                .map(|emb| (obs, emb))
        })
        .collect::<Vec<_>>();

    let mut face_rows = if preview_face_rows.is_empty() {
        tracklet
            .observations
            .iter()
            .filter_map(|obs| {
                obs.face_embedding
                    .as_ref()
                    .filter(|emb| !emb.is_empty())
                    .map(|emb| (obs, emb))
            })
            .collect::<Vec<_>>()
    } else {
        preview_face_rows
    };

    if face_rows.is_empty() {
        return None;
    }

    let mut preview_sum = 0.0f32;
    let mut preview_observations = 0u32;

    face_rows.sort_unstable_by(|(a, _), (b, _)| a.frame_index.cmp(&b.frame_index));
    let mut centroid = face_rows[0].1.clone();
    let mut confidence_sum = face_rows[0].0.bbox.confidence;
    let mut embedding_sim_sum = 1.0f32;
    let mut observations = 1u32;
    let mut first_frame = face_rows[0].0.frame_index;
    let mut last_frame = face_rows[0].0.frame_index;
    let mut anchor_x_acc = face_rows[0].0.anchor_x;
    let mut anchor_y_acc = face_rows[0].0.anchor_y;
    let mut anchor_x_norm_acc = face_rows[0].0.anchor_x_norm;
    let mut anchor_y_norm_acc = face_rows[0].0.anchor_y_norm;
    let mut face_presence_sum = face_rows[0].0.face_presence;
    let mut strong_face_observations = u32::from(face_rows[0].0.face_presence >= 0.58);

    for (obs, face_embedding) in face_rows.into_iter().skip(1) {
        let similarity = embedding_cosine_similarity(&centroid, face_embedding).clamp(0.0, 1.0);
        let n = observations as f32;
        for (c, e) in centroid.iter_mut().zip(face_embedding.iter()) {
            *c = (*c).mul_add(n, *e) / (n + 1.0);
        }
        centroid = l2_normalize(&centroid);

        observations += 1;
        confidence_sum += obs.bbox.confidence * similarity.max(0.1);
        embedding_sim_sum += similarity;
        first_frame = first_frame.min(obs.frame_index);
        last_frame = last_frame.max(obs.frame_index);
        anchor_x_acc += obs.anchor_x;
        anchor_y_acc += obs.anchor_y;
        anchor_x_norm_acc += obs.anchor_x_norm;
        anchor_y_norm_acc += obs.anchor_y_norm;
        face_presence_sum += obs.face_presence;
        if obs.face_presence >= 0.58 {
            strong_face_observations = strong_face_observations.saturating_add(1);
        }
    }

    let mut body_centroid = None::<Vec<f32>>;
    let mut body_count = 0f32;
    for obs in &tracklet.observations {
        if obs.preview_eligible {
            preview_sum += obs.face_preview;
            preview_observations = preview_observations.saturating_add(1);
        }
        let Some(body_embedding) = obs.body_embedding.as_ref() else {
            continue;
        };
        if body_embedding.is_empty() {
            continue;
        }
        if let Some(acc) = body_centroid.as_mut() {
            for (a, b) in acc.iter_mut().zip(body_embedding.iter()) {
                *a = (*a).mul_add(body_count, *b) / (body_count + 1.0);
            }
            *acc = l2_normalize(acc);
        } else {
            body_centroid = Some(body_embedding.clone());
        }
        body_count += 1.0;
    }

    Some(ClusterSeed {
        centroid,
        body_centroid,
        confidence_sum,
        observations,
        first_frame,
        last_frame,
        anchor_x_acc,
        anchor_y_acc,
        anchor_x_norm_acc,
        anchor_y_norm_acc,
        embedding_sim_sum,
        face_presence_sum,
        preview_sum,
        preview_observations,
        body_support_observations: tracklet.body_support_observations,
        thumbnail_score: tracklet.best_preview_score,
        thumbnail_jpeg: tracklet.best_preview_jpeg,
        strong_face_observations,
    })
}

fn normalized_anchor_distance(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    dx.hypot(dy)
}

fn take_embedding_for_bbox(rows: &mut Vec<(BBox, Vec<f32>)>, bbox: BBox) -> Option<Vec<f32>> {
    let mut best: Option<(usize, f32, f32)> = None;

    for (idx, (candidate_bbox, _)) in rows.iter().enumerate() {
        let iou = candidate_bbox.iou(&bbox).clamp(0.0, 1.0);
        let center_distance = {
            let dx = candidate_bbox.center_x() - bbox.center_x();
            let dy = candidate_bbox.center_y() - bbox.center_y();
            dx.hypot(dy)
        };
        let scale = bbox.width().max(bbox.height()).max(1.0);
        let normalized_distance = (center_distance / scale).clamp(0.0, 1.0);
        let score = iou * 0.85 + (1.0 - normalized_distance) * 0.15;

        if best.is_none_or(|(_, best_score, _)| score > best_score) {
            best = Some((idx, score, normalized_distance));
        }
    }

    let (idx, score, normalized_distance) = best?;
    if score < 0.70 && normalized_distance > 0.20 {
        return None;
    }
    Some(rows.swap_remove(idx).1)
}

fn copy_rgb_frame(frame: &frame::Video, pts: u64) -> RgbFrame {
    let w = frame.width();
    let h = frame.height();
    let stride = frame.stride(0);
    let data = frame.data(0);
    let row_len = (w as usize) * 3;

    let mut rgb = vec![0u8; row_len * h as usize];
    for row in 0..h as usize {
        let src_start = row * stride;
        let dst_start = row * row_len;
        rgb[dst_start..dst_start + row_len].copy_from_slice(&data[src_start..src_start + row_len]);
    }

    RgbFrame {
        data: rgb,
        width: w,
        height: h,
        pts: pts as i64,
    }
}

fn thumbnail_from_bbox(frame: &RgbFrame, bbox: BBox, prefer_face_crop: bool) -> Result<Vec<u8>> {
    let (x1, y1, w, h) = if prefer_face_crop {
        face_crop_region_for_bbox(frame, bbox)
    } else {
        bbox_region(frame, bbox)
    };
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }

    let image = crop_region(frame, x1, y1, w, h)?;
    let resized = imageops::thumbnail(&image, 132, 198);

    let mut canvas = ImageBuffer::from_pixel(132, 198, Rgb([15u8, 15u8, 19u8]));
    let offset_x = (132u32.saturating_sub(resized.width())) / 2;
    let offset_y = (198u32.saturating_sub(resized.height())) / 2;
    imageops::overlay(
        &mut canvas,
        &resized,
        i64::from(offset_x),
        i64::from(offset_y),
    );

    let mut out = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut out, 88);
    encoder
        .encode_image(&image::DynamicImage::ImageRgb8(canvas))
        .context("failed to encode thumbnail")?;
    Ok(out.into_inner())
}

fn thumbnail_quality(frame: &RgbFrame, bbox: BBox, similarity: f32, face_presence: f32) -> f32 {
    let (_, _, fw, fh) = face_crop_region_for_bbox(frame, bbox);
    let min_dim = fw.min(fh) as f32;
    let size_score = (min_dim / 92.0).clamp(0.0, 1.0);
    let conf_score = bbox.confidence.clamp(0.0, 1.0);
    let sim_score = similarity.clamp(0.0, 1.0);
    let face_score = face_presence.clamp(0.0, 1.0);
    (size_score * 0.40 + conf_score * 0.20 + sim_score * 0.16 + face_score * 0.24).clamp(0.0, 1.0)
}

fn bbox_region(frame: &RgbFrame, bbox: BBox) -> (u32, u32, u32, u32) {
    let x1 = bbox.x1.max(0.0).floor() as u32;
    let y1 = bbox.y1.max(0.0).floor() as u32;
    if x1 >= frame.width || y1 >= frame.height {
        return (0, 0, 0, 0);
    }
    let w = (bbox.width().max(1.0).round() as u32).min(frame.width.saturating_sub(x1));
    let h = (bbox.height().max(1.0).round() as u32).min(frame.height.saturating_sub(y1));
    if w == 0 || h == 0 {
        return (0, 0, 0, 0);
    }
    (x1, y1, w, h)
}

fn crop_region(frame: &RgbFrame, x1: u32, y1: u32, w: u32, h: u32) -> Result<RgbImage> {
    let mut crop = vec![0u8; (w * h * 3) as usize];
    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (w * 3) as usize;
    for row in 0..h as usize {
        let src_start = (y1 as usize + row) * src_stride + x1 as usize * 3;
        let dst_start = row * dst_stride;
        crop[dst_start..dst_start + dst_stride]
            .copy_from_slice(&frame.data[src_start..src_start + dst_stride]);
    }
    ImageBuffer::from_raw(w, h, crop).context("invalid thumbnail crop")
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_frame() -> RgbFrame {
        RgbFrame {
            data: vec![127u8; 320 * 240 * 3],
            width: 320,
            height: 240,
            pts: 0,
        }
    }

    fn frame_obs(
        frame_index: u64,
        sampled_index: u64,
        bbox: BBox,
        face_embedding: Option<Vec<f32>>,
    ) -> FrameDetectionObservation {
        FrameDetectionObservation {
            frame_index,
            sampled_index,
            bbox,
            anchor_x: bbox.center_x(),
            anchor_y: bbox.center_y(),
            anchor_x_norm: bbox.center_x() / 320.0,
            anchor_y_norm: bbox.center_y() / 240.0,
            face_presence: 0.8,
            face_preview: 0.7,
            preview_eligible: true,
            face_embedding,
            body_embedding: None,
        }
    }

    #[test]
    fn tracklet_assignment_merges_consistent_observations() {
        let mut tracklets = Vec::new();
        let frame = rgb_frame();
        let config = DiscoveryConfig::default();
        let bbox = BBox {
            x1: 40.0,
            y1: 30.0,
            x2: 140.0,
            y2: 210.0,
            confidence: 0.9,
        };

        assign_observations_to_tracklets(
            &mut tracklets,
            vec![frame_obs(
                1,
                1,
                bbox,
                Some(l2_normalize(&[1.0, 0.0, 0.0, 0.0])),
            )],
            &frame,
            &config,
        );
        assign_observations_to_tracklets(
            &mut tracklets,
            vec![frame_obs(
                2,
                2,
                bbox,
                Some(l2_normalize(&[0.98, 0.01, 0.0, 0.0])),
            )],
            &frame,
            &config,
        );

        assert_eq!(tracklets.len(), 1);
        assert_eq!(tracklets[0].observations.len(), 2);
    }

    #[test]
    fn duplicate_similarity_uses_normalized_anchor_distance() {
        let a = IdentityCandidate {
            id: 0,
            confidence: 0.9,
            observations: 4,
            first_frame: 10,
            last_frame: 30,
            anchor_x: 100.0,
            anchor_y: 120.0,
            anchor_x_norm: 0.30,
            anchor_y_norm: 0.50,
            thumbnail_jpeg: vec![1, 2, 3],
            embedding: l2_normalize(&[1.0, 0.0, 0.0, 0.0]),
            body_embedding: None,
            preview_score: 0.8,
            preview_observations: 3,
        };
        let mut b = a.clone();
        b.id = 1;
        b.anchor_x = 1800.0;
        b.anchor_y = 850.0;
        b.anchor_x_norm = 0.32;
        b.anchor_y_norm = 0.53;

        let score = duplicate_similarity_score(&a, &b);
        assert!(score > 0.70);
    }

    #[test]
    fn previewless_tracklet_is_suppressed_from_candidates() {
        let bbox = BBox {
            x1: 32.0,
            y1: 24.0,
            x2: 132.0,
            y2: 210.0,
            confidence: 0.88,
        };
        let obs = FrameDetectionObservation {
            frame_index: 1,
            sampled_index: 1,
            bbox,
            anchor_x: bbox.center_x(),
            anchor_y: bbox.center_y(),
            anchor_x_norm: bbox.center_x() / 320.0,
            anchor_y_norm: bbox.center_y() / 240.0,
            face_presence: 0.82,
            face_preview: 0.30,
            preview_eligible: false,
            face_embedding: Some(l2_normalize(&[1.0, 0.0, 0.0, 0.0])),
            body_embedding: Some(l2_normalize(&[0.2, 0.8, 0.0, 0.0])),
        };

        let tracklet = ProvisionalTracklet {
            last_sampled_index: obs.sampled_index,
            observations: vec![TrackletObservation {
                frame_index: obs.frame_index,
                bbox: obs.bbox,
                anchor_x: obs.anchor_x,
                anchor_y: obs.anchor_y,
                anchor_x_norm: obs.anchor_x_norm,
                anchor_y_norm: obs.anchor_y_norm,
                face_presence: obs.face_presence,
                face_preview: obs.face_preview,
                preview_eligible: obs.preview_eligible,
                face_embedding: obs.face_embedding.clone(),
                body_embedding: obs.body_embedding.clone(),
            }],
            best_preview_score: 0.0,
            best_preview_jpeg: Vec::new(),
            last_face_embedding: obs.face_embedding,
            last_body_embedding: obs.body_embedding,
            preview_observations: 0,
            body_support_observations: 1,
        };

        let Some(seed) = tracklet_to_cluster_seed(tracklet) else {
            panic!("expected seed")
        };
        let cluster = Cluster::new(seed);
        assert!(
            cluster
                .into_candidate(0, &DiscoveryConfig::default())
                .is_none()
        );
    }

    #[test]
    fn candidate_ranking_prefers_preview_quality() {
        let mk =
            |id: usize, conf: f32, obs: u32, preview: f32, preview_obs: u32| IdentityCandidate {
                id,
                confidence: conf,
                observations: obs,
                first_frame: 0,
                last_frame: u64::from(obs),
                anchor_x: 100.0,
                anchor_y: 80.0,
                anchor_x_norm: 0.3,
                anchor_y_norm: 0.3,
                thumbnail_jpeg: vec![1, 2, 3],
                embedding: l2_normalize(&[1.0, 0.0, 0.0, 0.0]),
                body_embedding: None,
                preview_score: preview,
                preview_observations: preview_obs,
            };

        let mut candidates = [
            mk(0, 0.90, 12, 0.44, 10),
            mk(1, 0.78, 6, 0.79, 4),
            mk(2, 0.70, 8, 0.62, 7),
        ];

        candidates.sort_unstable_by(|a, b| {
            b.preview_score
                .partial_cmp(&a.preview_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| b.preview_observations.cmp(&a.preview_observations))
                .then_with(|| b.observations.cmp(&a.observations))
        });

        assert_eq!(candidates[0].id, 1);
        assert_eq!(candidates[1].id, 2);
        assert_eq!(candidates[2].id, 0);
    }
}
