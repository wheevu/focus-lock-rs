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
    face_presence_score,
};
use crate::mode::ProcessingMode;
use crate::video::{RgbFrame, open_input_with_hwaccel};

/// Default frame sampling stride (analyze every 12th frame).
const DEFAULT_SAMPLE_STRIDE: u64 = 12;
const DEFAULT_MAX_SAMPLED_FRAMES: usize = 900;
const DEFAULT_MAX_FACES_PER_FRAME: usize = 6;
const DEFAULT_CLUSTER_SIMILARITY: f32 = 0.76;
const DEFAULT_DUPLICATE_SIMILARITY: f32 = 0.86;
const DEFAULT_MIN_OBSERVATIONS: u32 = 2;
const DEFAULT_MIN_EMBEDDING_SIMILARITY: f32 = 0.68;
const DEFAULT_MIN_FACE_CROP_EDGE: u32 = 42;
const DEFAULT_MAX_CANDIDATES: usize = 18;
const DEFAULT_MAX_DUPLICATES: usize = 36;
const DEFAULT_MIN_THUMBNAIL_QUALITY: f32 = 0.54;
const DEFAULT_MIN_FACE_PRESENCE: f32 = 0.42;
const PROGRESS_EMIT_FRAME_INTERVAL: u64 = 24;

const MERGE_STRONG_SIMILARITY: f32 = 0.84;
const MERGE_SOFT_SIMILARITY: f32 = 0.80;
const MERGE_SOFT_ANCHOR_DISTANCE: f32 = 0.11;

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
    /// Minimum face crop edge in pixels required to run ArcFace embedding.
    pub min_face_crop_edge: u32,
    /// Maximum number of candidates returned after ranking.
    pub max_candidates: usize,
    /// Maximum number of duplicate rows emitted for review.
    pub max_duplicates: usize,
    /// Minimum thumbnail quality score for candidate emission.
    pub min_thumbnail_quality: f32,
    /// Minimum face-presence confidence required for sampled observations.
    pub min_face_presence: f32,
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
        }
    }
}

impl DiscoveryConfig {
    /// Build a mode-tuned config for discovery.
    #[must_use]
    pub fn for_mode(mode: ProcessingMode) -> Self {
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
            },
            ProcessingMode::Balanced => Self {
                sample_stride: 12,
                max_sampled_frames: 900,
                max_faces_per_frame: 6,
                cluster_similarity: 0.77,
                duplicate_similarity: 0.87,
                min_observations: 2,
                min_embedding_similarity: 0.69,
                min_face_crop_edge: 46,
                max_candidates: 18,
                max_duplicates: 36,
                min_thumbnail_quality: 0.54,
                min_face_presence: 0.44,
            },
            ProcessingMode::Quality => Self {
                sample_stride: 8,
                max_sampled_frames: 1400,
                max_faces_per_frame: 8,
                cluster_similarity: 0.75,
                duplicate_similarity: 0.85,
                min_observations: 2,
                min_embedding_similarity: 0.66,
                min_face_crop_edge: 38,
                max_candidates: 28,
                max_duplicates: 60,
                min_thumbnail_quality: 0.48,
                min_face_presence: 0.36,
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
    /// JPEG-encoded thumbnail image for UI display.
    pub thumbnail_jpeg: Vec<u8>,
    /// Centroid embedding used for tracking handoff.
    pub embedding: Vec<f32>,
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
}

/// Engine for discovering identities in videos.
///
/// Loads the required ML models and provides methods to scan videos
/// and build a list of distinct identities present.
#[derive(Debug)]
pub struct DiscoveryEngine {
    detector: Detector,
    embedder: Arc<FaceEmbedder>,
}

impl DiscoveryEngine {
    /// Loads the discovery engine with the given model paths.
    ///
    /// # Arguments
    ///
    /// * `yolo_model_path` - Path to the YOLOv8 ONNX model for person detection
    /// * `face_model_path` - Path to the ArcFace ONNX model for face embedding
    ///
    /// # Errors
    ///
    /// Returns an error if either model cannot be loaded.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
    ) -> Result<Self> {
        Ok(Self {
            detector: Detector::load(yolo_model_path)?,
            embedder: Arc::new(FaceEmbedder::load(face_model_path)?),
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
        // Pre-allocate clusters with reasonable capacity to avoid reallocations.
        // Most videos have 1-10 distinct faces, so 16 is a good starting point.
        let mut clusters: Vec<Cluster> = Vec::with_capacity(16);

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

            let embeddings =
                self.embedder
                    .embed_many_from_bboxes(&rgb, &persons, config.min_face_crop_edge)?;
            rejected_embeddings += persons.len().saturating_sub(embeddings.len()) as u64;

            for (bbox, embedding) in embeddings {
                if should_cancel() {
                    anyhow::bail!("identity scan cancelled");
                }
                let face_presence = face_presence_score(&rgb, bbox);
                if face_presence < config.min_face_presence {
                    rejected_embeddings = rejected_embeddings.saturating_add(1);
                    continue;
                }
                let (anchor_x, anchor_y) = (bbox.center_x(), bbox.center_y());
                update_clusters(
                    &mut clusters,
                    ClusterObservation {
                        frame_index,
                        bbox,
                        anchor_x,
                        anchor_y,
                        embedding,
                        face_presence,
                    },
                    config.cluster_similarity,
                    &rgb,
                );
            }
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

        let merged_clusters = merge_clusters(&mut clusters);

        let cluster_count_before_filter = clusters.len();
        let mut candidates: Vec<IdentityCandidate> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(id, cluster)| cluster.into_candidate(id, config))
            .collect();

        let suppressed_clusters = cluster_count_before_filter.saturating_sub(candidates.len());

        candidates.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
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
        })
    }
}

#[derive(Clone)]
struct ClusterObservation {
    frame_index: u64,
    bbox: BBox,
    anchor_x: f32,
    anchor_y: f32,
    embedding: Vec<f32>,
    face_presence: f32,
}

struct Cluster {
    centroid: Vec<f32>,
    confidence_sum: f32,
    observations: u32,
    first_frame: u64,
    last_frame: u64,
    anchor_x_acc: f32,
    anchor_y_acc: f32,
    embedding_sim_sum: f32,
    face_presence_sum: f32,
    thumbnail_score: f32,
    thumbnail_jpeg: Vec<u8>,
    strong_face_observations: u32,
}

impl Cluster {
    fn new(obs: &ClusterObservation, quality: f32, thumbnail_jpeg: Vec<u8>) -> Self {
        let thumbnail_score = if thumbnail_jpeg.is_empty() {
            0.0
        } else {
            quality
        };
        let strong_face_observations = u32::from(obs.face_presence >= 0.58);
        Self {
            centroid: obs.embedding.clone(),
            confidence_sum: obs.bbox.confidence,
            observations: 1,
            first_frame: obs.frame_index,
            last_frame: obs.frame_index,
            anchor_x_acc: obs.anchor_x,
            anchor_y_acc: obs.anchor_y,
            embedding_sim_sum: 1.0,
            face_presence_sum: obs.face_presence,
            thumbnail_score,
            thumbnail_jpeg,
            strong_face_observations,
        }
    }

    fn absorb(
        &mut self,
        obs: &ClusterObservation,
        similarity: f32,
        quality: f32,
        thumbnail_jpeg: Option<Vec<u8>>,
    ) {
        let n = self.observations as f32;
        for (c, e) in self.centroid.iter_mut().zip(obs.embedding.iter()) {
            *c = (*c * n + *e) / (n + 1.0);
        }
        self.centroid = l2_normalize(&self.centroid);
        self.confidence_sum += obs.bbox.confidence * similarity.max(0.1);
        self.observations += 1;
        self.last_frame = obs.frame_index;
        self.anchor_x_acc += obs.anchor_x;
        self.anchor_y_acc += obs.anchor_y;
        self.embedding_sim_sum += similarity.clamp(0.0, 1.0);
        self.face_presence_sum += obs.face_presence;
        if obs.face_presence >= 0.58 {
            self.strong_face_observations = self.strong_face_observations.saturating_add(1);
        }

        if quality > self.thumbnail_score {
            self.thumbnail_score = quality;
            if let Some(thumbnail_jpeg) = thumbnail_jpeg {
                self.thumbnail_jpeg = thumbnail_jpeg;
            }
        }
    }

    fn into_candidate(self, id: usize, config: &DiscoveryConfig) -> Option<IdentityCandidate> {
        if self.observations < config.min_observations {
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
        let avg_score = self.confidence_sum / self.observations as f32;
        let confidence = (0.36
            + (self.observations as f32 * 0.08).min(0.44)
            + avg_score * 0.16
            + avg_embedding_similarity * 0.16
            + avg_face_presence * 0.10
            + self.thumbnail_score * 0.12)
            .clamp(0.0, 0.995);
        Some(IdentityCandidate {
            id,
            confidence,
            observations: self.observations,
            first_frame: self.first_frame,
            last_frame: self.last_frame,
            anchor_x: self.anchor_x_acc / self.observations as f32,
            anchor_y: self.anchor_y_acc / self.observations as f32,
            thumbnail_jpeg: self.thumbnail_jpeg,
            embedding: self.centroid,
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
    let dx = a.anchor_x - b.anchor_x;
    let dy = a.anchor_y - b.anchor_y;
    let distance = (dx * dx + dy * dy).sqrt();
    (1.0 - distance.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn duplicate_similarity_score(a: &IdentityCandidate, b: &IdentityCandidate) -> f32 {
    let embedding = embedding_cosine_similarity(&a.embedding, &b.embedding).clamp(0.0, 1.0);
    if embedding < DUPLICATE_EMBEDDING_FLOOR {
        return 0.0;
    }

    let anchor_score = anchor_similarity(a, b);
    let dx = a.anchor_x - b.anchor_x;
    let dy = a.anchor_y - b.anchor_y;
    let anchor_distance = (dx * dx + dy * dy).sqrt();
    if anchor_distance > DUPLICATE_ANCHOR_DISTANCE_MAX {
        return 0.0;
    }

    let time_overlap = temporal_overlap_score(a, b);
    let confidence_gap = (a.confidence - b.confidence).abs();
    let confidence_score = (1.0 - confidence_gap / DUPLICATE_CONFIDENCE_GAP_MAX).clamp(0.0, 1.0);

    (embedding * 0.78 + anchor_score * 0.10 + time_overlap * 0.08 + confidence_score * 0.04)
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
        *t = (*t * left_obs + *o * right_obs) / denom;
    }
    target.centroid = l2_normalize(&target.centroid);
    target.confidence_sum += other.confidence_sum;
    target.embedding_sim_sum += other.embedding_sim_sum;
    target.face_presence_sum += other.face_presence_sum;
    target.observations += other.observations;
    target.strong_face_observations = target
        .strong_face_observations
        .saturating_add(other.strong_face_observations);
    target.first_frame = target.first_frame.min(other.first_frame);
    target.last_frame = target.last_frame.max(other.last_frame);
    target.anchor_x_acc += other.anchor_x_acc;
    target.anchor_y_acc += other.anchor_y_acc;
    if other.thumbnail_score > target.thumbnail_score {
        target.thumbnail_score = other.thumbnail_score;
        target.thumbnail_jpeg = other.thumbnail_jpeg;
    }
}

fn cluster_anchor_distance(a: &Cluster, b: &Cluster) -> f32 {
    let ax = a.anchor_x_acc / a.observations.max(1) as f32;
    let ay = a.anchor_y_acc / a.observations.max(1) as f32;
    let bx = b.anchor_x_acc / b.observations.max(1) as f32;
    let by = b.anchor_y_acc / b.observations.max(1) as f32;
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

fn update_clusters(
    clusters: &mut Vec<Cluster>,
    obs: ClusterObservation,
    similarity_gate: f32,
    frame: &RgbFrame,
) {
    let mut best_idx = None;
    let mut best_sim = -1.0f32;
    for (idx, c) in clusters.iter().enumerate() {
        let sim = embedding_cosine_similarity(&c.centroid, &obs.embedding);
        if sim > best_sim {
            best_sim = sim;
            best_idx = Some(idx);
        }
    }

    if let Some(idx) = best_idx
        && best_sim >= similarity_gate
    {
        let quality = thumbnail_quality(frame, obs.bbox, best_sim, obs.face_presence);
        let thumbnail_jpeg = if quality > clusters[idx].thumbnail_score {
            thumbnail_from_bbox(frame, obs.bbox, true).ok()
        } else {
            None
        };
        clusters[idx].absorb(&obs, best_sim, quality, thumbnail_jpeg);
    } else {
        let quality = thumbnail_quality(frame, obs.bbox, 1.0, obs.face_presence);
        let thumbnail_jpeg = thumbnail_from_bbox(frame, obs.bbox, true).unwrap_or_default();
        clusters.push(Cluster::new(&obs, quality, thumbnail_jpeg));
    }
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

    fn obs(id: u64, emb: Vec<f32>, bbox: BBox) -> ClusterObservation {
        ClusterObservation {
            frame_index: id,
            bbox,
            anchor_x: bbox.center_x(),
            anchor_y: bbox.center_y(),
            embedding: emb,
            face_presence: 0.8,
        }
    }

    #[test]
    fn cluster_update_merges_similar_embeddings() {
        let mut clusters = Vec::new();
        let frame = rgb_frame();
        let bbox = BBox {
            x1: 40.0,
            y1: 30.0,
            x2: 140.0,
            y2: 210.0,
            confidence: 0.9,
        };
        update_clusters(
            &mut clusters,
            obs(1, l2_normalize(&[1.0, 0.0, 0.0, 0.0]), bbox),
            0.75,
            &frame,
        );
        update_clusters(
            &mut clusters,
            obs(2, l2_normalize(&[0.98, 0.01, 0.0, 0.0]), bbox),
            0.75,
            &frame,
        );
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].observations, 2);
    }

    #[test]
    fn cluster_update_splits_dissimilar_embeddings() {
        let mut clusters = Vec::new();
        let frame = rgb_frame();
        let bbox = BBox {
            x1: 40.0,
            y1: 30.0,
            x2: 140.0,
            y2: 210.0,
            confidence: 0.85,
        };
        update_clusters(
            &mut clusters,
            obs(1, l2_normalize(&[1.0, 0.0, 0.0, 0.0]), bbox),
            0.85,
            &frame,
        );
        update_clusters(
            &mut clusters,
            obs(2, l2_normalize(&[0.0, 1.0, 0.0, 0.0]), bbox),
            0.85,
            &frame,
        );
        assert_eq!(clusters.len(), 2);
    }
}
