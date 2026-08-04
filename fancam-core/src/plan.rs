//! Versioned, privacy-safe crop plans for offline renders.
//!
//! A crop plan is deliberately made from geometry and aggregate quality
//! signals.  It never contains model embeddings, thumbnails, decoded pixels,
//! or source paths, so it is safe to keep beside a rendered video and share
//! for debugging or manual framing review.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    camera::{CameraKeyframe, CameraPath},
    rendering::{OUT_HEIGHT, OUT_WIDTH},
    solver::SolverResult,
    tracking::{CameraSource, CameraState},
    tracklet::Tracklet,
};

/// JSON schema identifier for crop sidecars.
pub const CROP_PLAN_SCHEMA: &str = "focus-lock.crop-plan";
/// Current crop sidecar version.
pub const CROP_PLAN_VERSION: u32 = 1;
/// Version of the bounded source-content fingerprint algorithm.
pub const SOURCE_FINGERPRINT_VERSION: u32 = 1;
/// Algorithm identifier for the bounded source-content fingerprint.
pub const SOURCE_FINGERPRINT_ALGORITHM: &str = "sha256-sampled-container-v1";

static PLAN_TEMP_SEQ: AtomicU64 = AtomicU64::new(1);

/// Source video metadata that does not identify the user's filesystem path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VideoMetadata {
    /// Source width in pixels.
    pub width: u32,
    /// Source height in pixels.
    pub height: u32,
    /// Exact number of frames decoded during the offline prepass.
    pub frame_count: u64,
    /// Average frame-rate numerator, or zero when unavailable.
    pub frame_rate_num: u32,
    /// Average frame-rate denominator, or zero when unavailable.
    pub frame_rate_den: u32,
    /// Source duration in milliseconds when available.
    pub duration_ms: Option<u64>,
}

/// A privacy-safe, versioned identity for the selected source video.
///
/// The digest covers canonical video metadata, file length, and at most sixteen
/// evenly spaced 64 KiB windows from the raw container bytes. It is deliberately
/// bounded and does not store a path, decoded pixels, thumbnails, or embeddings;
/// edits outside the sampled windows can therefore go undetected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceVideoFingerprint {
    /// Fingerprint algorithm version.
    pub version: u32,
    /// Stable algorithm identifier.
    pub algorithm: String,
    /// Lowercase SHA-256 digest of the canonical metadata and samples.
    pub digest: String,
    /// Source container length in bytes.
    pub file_size: u64,
    /// Number of raw container bytes included in the digest.
    pub sampled_bytes: u64,
}

/// Fixed output geometry used by the renderer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CropOutput {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

/// Reason for a timeline boundary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShotBoundaryKind {
    /// A large change in a small luma signature suggests a hard cut.
    HardCut,
}

/// A detected boundary in the source timeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShotBoundary {
    /// First frame belonging to the new shot.
    pub frame_index: u64,
    /// Confidence derived from the frame-signature change.
    pub confidence: f32,
    /// Detection reason.
    pub kind: ShotBoundaryKind,
}

/// Aggregate quality information for one short-term tracklet.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackQuality {
    /// Local tracklet id.
    pub track_id: usize,
    /// Clustered identity id, when the solver assigned one.
    pub identity_id: Option<usize>,
    /// First observed frame.
    pub first_frame: u64,
    /// Last observed frame.
    pub last_frame: u64,
    /// Number of observations retained for this tracklet.
    pub observation_count: u32,
    /// Mean bounded identity/camera confidence.
    pub mean_confidence: f32,
    /// Best bounded identity/camera confidence.
    pub best_confidence: f32,
    /// Combined confidence and continuity score.
    pub quality_score: f32,
    /// Frames skipped inside this tracklet or while re-entering the same identity.
    pub occlusion_frames: u64,
    /// Number of non-contiguous re-entry gaps associated with this identity.
    pub reentry_count: u32,
}

/// Stable source label for a generated crop keyframe.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanKeyframeSource {
    /// A state backed by a target observation.
    Observed,
    /// A short gap filled by interpolation.
    Predicted,
    /// A state held while the target is missing.
    Held,
}

/// A generated camera keyframe with a confidence estimate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanKeyframe {
    /// Source frame index.
    pub frame_index: u64,
    /// Crop center X in source pixels.
    pub cx: f32,
    /// Crop center Y in source pixels.
    pub cy: f32,
    /// Half-size of the tracked subject box in source pixels.
    pub half_size: f32,
    /// Confidence estimate in the inclusive range 0–1.
    pub confidence: f32,
    /// How this state was obtained.
    pub source: PlanKeyframeSource,
}

/// A user-authored framing correction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManualKeyframe {
    /// Source frame index.
    pub frame_index: u64,
    /// Crop center X in source pixels.
    pub cx: f32,
    /// Crop center Y in source pixels.
    pub cy: f32,
    /// Half-size of the tracked subject box in source pixels.
    pub half_size: f32,
}

/// Aggregate metrics suitable for a small confidence timeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanQualityMetrics {
    /// Number of generated keyframes backed by observations.
    pub observed_keyframes: u64,
    /// Number of generated keyframes filled by interpolation.
    pub predicted_keyframes: u64,
    /// Number of held/frozen keyframes.
    pub held_keyframes: u64,
    /// Mean confidence across generated keyframes.
    pub mean_confidence: f32,
    /// Lowest confidence across generated keyframes.
    pub min_confidence: f32,
    /// Fraction of source frames covered by the selected camera path.
    pub path_coverage: f32,
    /// Longest observed gap in the selected identity timeline.
    pub max_gap_frames: u64,
    /// Number of detected hard cuts.
    pub shot_boundary_count: u64,
}

/// Versioned sidecar emitted by the CLI or Tauri renderer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CropPlanV1 {
    /// Schema identifier.
    pub schema: String,
    /// Schema version.
    pub version: u32,
    /// Versioned content identity for the selected source video.
    pub source_fingerprint: SourceVideoFingerprint,
    /// Source video metadata without a source path.
    pub video: VideoMetadata,
    /// Target render geometry.
    pub output: CropOutput,
    /// Detected source timeline boundaries.
    pub shots: Vec<ShotBoundary>,
    /// Aggregate track quality rows.
    pub tracks: Vec<TrackQuality>,
    /// Generated camera path.
    pub keyframes: Vec<PlanKeyframe>,
    /// User corrections; empty for a newly generated plan.
    pub manual_keyframes: Vec<ManualKeyframe>,
    /// Aggregate quality metrics.
    pub quality: PlanQualityMetrics,
}

impl CropPlanV1 {
    /// Build a plan from the existing offline solver output.
    #[must_use]
    pub fn from_offline_solution(
        video: VideoMetadata,
        source_fingerprint: SourceVideoFingerprint,
        tracklets: &[Tracklet],
        solved: &SolverResult,
        mut shots: Vec<ShotBoundary>,
    ) -> Self {
        let assignment_by_track = solved
            .assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.tracklet_id,
                    (assignment.identity_id, clamp_score(assignment.confidence)),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();

        let mut sorted_tracklets = tracklets.iter().collect::<Vec<_>>();
        sorted_tracklets.sort_unstable_by_key(|tracklet| tracklet.id);

        let mut tracks = sorted_tracklets
            .iter()
            .map(|tracklet| {
                let assignment = assignment_by_track.get(&tracklet.id).copied();
                let (mean_confidence, best_confidence, internal_gap_frames) =
                    tracklet_confidence(tracklet);
                let continuity = continuity_score(tracklet, internal_gap_frames);
                let assignment_confidence = assignment.map_or(0.0, |(_, value)| value);
                let quality_score = clamp_score(
                    mean_confidence
                        .mul_add(0.65, assignment_confidence.mul_add(0.2, continuity * 0.15)),
                );
                TrackQuality {
                    track_id: tracklet.id,
                    identity_id: assignment.map(|(identity_id, _)| identity_id),
                    first_frame: tracklet.first_frame().unwrap_or(0),
                    last_frame: tracklet.last_frame().unwrap_or(0),
                    observation_count: tracklet.len().min(u32::MAX as usize) as u32,
                    mean_confidence,
                    best_confidence,
                    quality_score,
                    occlusion_frames: internal_gap_frames,
                    reentry_count: 0,
                }
            })
            .collect::<Vec<_>>();

        let mut by_identity = std::collections::HashMap::<usize, Vec<(usize, u64, u64)>>::new();
        for track in &tracks {
            if let Some(identity_id) = track.identity_id {
                by_identity.entry(identity_id).or_default().push((
                    track.track_id,
                    track.first_frame,
                    track.last_frame,
                ));
            }
        }
        for rows in by_identity.values_mut() {
            rows.sort_unstable_by_key(|(_, first, _)| *first);
            let mut previous_last = None;
            for (track_id, first, last) in rows.iter().copied() {
                let gap = previous_last.map_or(0, |previous| first.saturating_sub(previous + 1));
                if gap > 0
                    && let Some(track) = tracks.iter_mut().find(|track| track.track_id == track_id)
                {
                    track.occlusion_frames = track.occlusion_frames.saturating_add(gap);
                    track.reentry_count = 1;
                }
                previous_last = Some(last);
            }
        }

        shots.sort_unstable_by(|a, b| {
            a.frame_index
                .cmp(&b.frame_index)
                .then_with(|| a.confidence.total_cmp(&b.confidence))
        });
        shots.dedup_by_key(|shot| shot.frame_index);

        let selected_track_ids = solved
            .selected_identity_id
            .map(|identity_id| {
                solved
                    .assignments
                    .iter()
                    .filter(|assignment| assignment.identity_id == identity_id)
                    .map(|assignment| assignment.tracklet_id)
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();

        let mut keyframes = solved
            .camera_path
            .keyframes
            .iter()
            .filter_map(|keyframe| {
                if !valid_camera_state(keyframe.state) {
                    return None;
                }
                let confidence = camera_confidence(keyframe, tracklets, &selected_track_ids);
                let (cx, cy, half_size) = normalize_geometry(
                    keyframe.state.cx,
                    keyframe.state.cy,
                    keyframe.state.half_size,
                    video.width,
                    video.height,
                );
                Some(PlanKeyframe {
                    frame_index: keyframe.frame_index,
                    cx,
                    cy,
                    half_size,
                    confidence,
                    source: plan_source(keyframe.state.source),
                })
            })
            .collect::<Vec<_>>();
        keyframes.sort_unstable_by_key(|keyframe| keyframe.frame_index);
        keyframes.dedup_by_key(|keyframe| keyframe.frame_index);

        let observed_keyframes = keyframes
            .iter()
            .filter(|keyframe| keyframe.source == PlanKeyframeSource::Observed)
            .count() as u64;
        let predicted_keyframes = keyframes
            .iter()
            .filter(|keyframe| keyframe.source == PlanKeyframeSource::Predicted)
            .count() as u64;
        let held_keyframes = keyframes
            .iter()
            .filter(|keyframe| keyframe.source == PlanKeyframeSource::Held)
            .count() as u64;
        let (mean_confidence, min_confidence) = confidence_metrics(&keyframes);
        let max_gap_frames = selected_max_gap(tracklets, &selected_track_ids);
        let path_coverage = match (video.frame_count, keyframes.first(), keyframes.last()) {
            (frame_count, Some(first), Some(last)) if frame_count > 0 => {
                ((last.frame_index.saturating_sub(first.frame_index) + 1) as f32
                    / frame_count as f32)
                    .clamp(0.0, 1.0)
            }
            _ => 0.0,
        };

        let shot_boundary_count = shots.len() as u64;
        Self {
            schema: CROP_PLAN_SCHEMA.to_string(),
            version: CROP_PLAN_VERSION,
            source_fingerprint,
            video,
            output: CropOutput {
                width: OUT_WIDTH,
                height: OUT_HEIGHT,
            },
            shots,
            tracks,
            keyframes,
            manual_keyframes: Vec::new(),
            quality: PlanQualityMetrics {
                observed_keyframes,
                predicted_keyframes,
                held_keyframes,
                mean_confidence,
                min_confidence,
                path_coverage,
                max_gap_frames,
                shot_boundary_count,
            },
        }
    }

    /// Validate the sidecar contract and all geometry before it is persisted or used.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CROP_PLAN_SCHEMA,
            "unsupported crop plan schema"
        );
        ensure!(
            self.version == CROP_PLAN_VERSION,
            "unsupported crop plan version"
        );
        validate_source_fingerprint(&self.source_fingerprint)?;
        ensure!(self.video.width > 0, "video width must be positive");
        ensure!(self.video.height > 0, "video height must be positive");
        ensure!(
            (self.video.frame_rate_num == 0) == (self.video.frame_rate_den == 0),
            "frame rate numerator and denominator must be both known or both zero"
        );
        ensure!(
            self.video.frame_rate_den == 0 || self.video.frame_rate_num > 0,
            "frame rate must be positive"
        );
        ensure!(
            self.output.width == OUT_WIDTH && self.output.height == OUT_HEIGHT,
            "crop plan V1 output must match the renderer's fixed {OUT_WIDTH}x{OUT_HEIGHT} geometry"
        );

        validate_sorted_unique(
            self.shots.iter().map(|shot| shot.frame_index),
            "shot boundaries",
        )?;
        for shot in &self.shots {
            validate_frame(shot.frame_index, self.video.frame_count, "shot boundary")?;
            validate_score(shot.confidence, "shot confidence")?;
        }

        for track in &self.tracks {
            ensure!(
                track.first_frame <= track.last_frame,
                "track frame range is inverted"
            );
            ensure!(
                track.observation_count > 0,
                "track observation count is zero"
            );
            validate_frame(track.first_frame, self.video.frame_count, "track start")?;
            validate_frame(track.last_frame, self.video.frame_count, "track end")?;
            validate_score(track.mean_confidence, "track mean confidence")?;
            validate_score(track.best_confidence, "track best confidence")?;
            validate_score(track.quality_score, "track quality")?;
        }

        validate_keyframes(
            &self.keyframes,
            self.video.frame_count,
            self.video.width,
            self.video.height,
            "generated keyframes",
        )?;
        validate_manual_keyframes(
            &self.manual_keyframes,
            self.video.frame_count,
            self.video.width,
            self.video.height,
        )?;
        validate_score(self.quality.mean_confidence, "mean confidence")?;
        validate_score(self.quality.min_confidence, "minimum confidence")?;
        validate_score(self.quality.path_coverage, "path coverage")?;
        Ok(())
    }

    /// Add or replace one manual correction and keep the timeline deterministic.
    pub fn upsert_manual_keyframe(&mut self, keyframe: ManualKeyframe) -> Result<()> {
        let keyframe = normalize_manual_keyframe(keyframe, &self.video)?;
        if let Some(existing) = self
            .manual_keyframes
            .iter_mut()
            .find(|existing| existing.frame_index == keyframe.frame_index)
        {
            *existing = keyframe;
        } else {
            self.manual_keyframes.push(keyframe);
        }
        self.manual_keyframes
            .sort_unstable_by_key(|keyframe| keyframe.frame_index);
        self.validate()
    }

    /// Reject a sidecar created for a different source video.
    pub fn ensure_source_fingerprint_matches(
        &self,
        expected: &SourceVideoFingerprint,
    ) -> Result<()> {
        ensure!(
            &self.source_fingerprint == expected,
            "crop plan source fingerprint does not match the selected video"
        );
        Ok(())
    }

    /// Remove one manual correction. Missing frames are harmless.
    pub fn remove_manual_keyframe(&mut self, frame_index: u64) {
        self.manual_keyframes
            .retain(|keyframe| keyframe.frame_index != frame_index);
    }

    /// Convert generated and manual keyframes into a renderable camera path.
    #[must_use]
    pub fn camera_path(&self) -> CameraPath {
        let mut keyframes = self
            .keyframes
            .iter()
            .map(|keyframe| CameraKeyframe {
                frame_index: keyframe.frame_index,
                state: CameraState {
                    cx: keyframe.cx,
                    cy: keyframe.cy,
                    half_size: keyframe.half_size,
                    source: camera_source(keyframe.source),
                    miss_count: 0,
                },
            })
            .collect::<Vec<_>>();
        for keyframe in &self.manual_keyframes {
            keyframes.retain(|existing| existing.frame_index != keyframe.frame_index);
            keyframes.push(CameraKeyframe {
                frame_index: keyframe.frame_index,
                state: CameraState {
                    cx: keyframe.cx,
                    cy: keyframe.cy,
                    half_size: keyframe.half_size,
                    source: CameraSource::Manual,
                    miss_count: 0,
                },
            });
        }
        keyframes.sort_unstable_by_key(|keyframe| keyframe.frame_index);
        CameraPath { keyframes }
    }

    /// Read, validate, and deserialize a crop plan from JSON.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read crop plan {}", path.display()))?;
        let plan: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse crop plan {}", path.display()))?;
        plan.validate()?;
        Ok(plan)
    }

    /// Validate and atomically write a crop plan as pretty JSON.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure!(parent.is_dir(), "crop plan parent is not a directory");
        if fs::symlink_metadata(path)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            bail!("crop plan path must not be a symlink");
        }

        let mut bytes = serde_json::to_vec_pretty(self).context("failed to serialize crop plan")?;
        bytes.push(b'\n');
        let temp_path = temporary_plan_path(path)?;
        let write_result = (|| -> Result<()> {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&temp_path)
                .with_context(|| {
                    format!("failed to open temporary crop plan {}", temp_path.display())
                })?;
            file.write_all(&bytes)
                .context("failed to write crop plan")?;
            file.sync_all().context("failed to flush crop plan")?;
            fs::rename(&temp_path, path)
                .with_context(|| format!("failed to commit crop plan {}", path.display()))?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        write_result
    }
}

/// Read a crop plan from a path.
pub fn read_crop_plan(path: impl AsRef<Path>) -> Result<CropPlanV1> {
    CropPlanV1::read_from_path(path)
}

/// Write a crop plan to a path.
pub fn write_crop_plan(path: impl AsRef<Path>, plan: &CropPlanV1) -> Result<()> {
    plan.write_to_path(path)
}

fn temporary_plan_path(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("crop-plan.json");
    for _ in 0..100 {
        let sequence = PLAN_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{file_name}.focus-lock-{}-{sequence}.partial",
            std::process::id()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context("failed to reserve crop plan temporary file"),
        }
    }
    bail!("could not allocate a crop plan temporary file")
}

fn validate_keyframes(
    keyframes: &[PlanKeyframe],
    frame_count: u64,
    width: u32,
    height: u32,
    label: &str,
) -> Result<()> {
    validate_sorted_unique(keyframes.iter().map(|keyframe| keyframe.frame_index), label)?;
    for keyframe in keyframes {
        validate_frame(keyframe.frame_index, frame_count, label)?;
        validate_geometry(keyframe.cx, keyframe.cy, keyframe.half_size, width, height)?;
        validate_score(keyframe.confidence, "keyframe confidence")?;
    }
    Ok(())
}

fn validate_manual_keyframes(
    keyframes: &[ManualKeyframe],
    frame_count: u64,
    width: u32,
    height: u32,
) -> Result<()> {
    validate_sorted_unique(
        keyframes.iter().map(|keyframe| keyframe.frame_index),
        "manual keyframes",
    )?;
    for keyframe in keyframes {
        validate_frame(keyframe.frame_index, frame_count, "manual keyframe")?;
        validate_geometry(keyframe.cx, keyframe.cy, keyframe.half_size, width, height)?;
    }
    Ok(())
}

fn validate_sorted_unique<I>(values: I, label: &str) -> Result<()>
where
    I: IntoIterator<Item = u64>,
{
    let mut previous = None;
    for value in values {
        if previous.is_some_and(|previous| value <= previous) {
            bail!("{label} must be sorted and unique");
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_frame(frame_index: u64, frame_count: u64, label: &str) -> Result<()> {
    ensure!(frame_index > 0, "{label} frame index must be 1-based");
    if frame_count > 0 {
        ensure!(
            frame_index <= frame_count,
            "{label} is outside video frame range"
        );
    }
    Ok(())
}

fn validate_geometry(cx: f32, cy: f32, half_size: f32, width: u32, height: u32) -> Result<()> {
    ensure!(
        cx.is_finite() && cy.is_finite(),
        "crop keyframe coordinates must be finite"
    );
    ensure!(
        half_size.is_finite() && half_size > 0.0,
        "crop keyframe half-size must be positive"
    );
    let (normalized_cx, normalized_cy, normalized_half_size) =
        normalize_geometry(cx, cy, half_size, width, height);
    ensure!(
        (cx - normalized_cx).abs() <= 1e-3
            && (cy - normalized_cy).abs() <= 1e-3
            && (half_size - normalized_half_size).abs() <= 1e-3,
        "crop geometry must be normalized to the source frame"
    );
    Ok(())
}

fn normalize_manual_keyframe(
    keyframe: ManualKeyframe,
    video: &VideoMetadata,
) -> Result<ManualKeyframe> {
    validate_frame(keyframe.frame_index, video.frame_count, "manual keyframe")?;
    ensure!(
        keyframe.cx.is_finite()
            && keyframe.cy.is_finite()
            && keyframe.half_size.is_finite()
            && keyframe.half_size > 0.0,
        "crop keyframe geometry must be finite and positive"
    );
    let (cx, cy, half_size) = normalize_geometry(
        keyframe.cx,
        keyframe.cy,
        keyframe.half_size,
        video.width,
        video.height,
    );
    Ok(ManualKeyframe {
        frame_index: keyframe.frame_index,
        cx,
        cy,
        half_size,
    })
}

/// Normalize camera geometry with the same crop-window math as the renderer.
#[must_use]
pub fn normalize_geometry(
    cx: f32,
    cy: f32,
    half_size: f32,
    width: u32,
    height: u32,
) -> (f32, f32, f32) {
    let width_f = width.max(1) as f32;
    let height_f = height.max(1) as f32;
    let safe_half_size = if half_size.is_finite() {
        half_size.clamp(1.0, width_f.max(height_f))
    } else {
        1.0
    };
    let aspect = OUT_WIDTH as f32 / OUT_HEIGHT as f32;
    let crop_w = (safe_half_size * 2.5).max(OUT_WIDTH as f32).min(width_f);
    let crop_h = (crop_w / aspect).min(height_f);
    let crop_w = crop_h * aspect;
    let normalized_cx = if cx.is_finite() {
        cx.clamp(crop_w / 2.0, (width_f - crop_w / 2.0).max(crop_w / 2.0))
    } else {
        width_f / 2.0
    };
    let normalized_cy = if cy.is_finite() {
        cy.clamp(crop_h / 2.0, (height_f - crop_h / 2.0).max(crop_h / 2.0))
    } else {
        height_f / 2.0
    };
    (normalized_cx, normalized_cy, safe_half_size)
}

fn validate_source_fingerprint(fingerprint: &SourceVideoFingerprint) -> Result<()> {
    ensure!(
        fingerprint.version == SOURCE_FINGERPRINT_VERSION,
        "unsupported source fingerprint version"
    );
    ensure!(
        fingerprint.algorithm == SOURCE_FINGERPRINT_ALGORITHM,
        "unsupported source fingerprint algorithm"
    );
    ensure!(
        fingerprint.digest.len() == 64
            && fingerprint
                .digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "source fingerprint digest must be a 64-character hexadecimal SHA-256 value"
    );
    ensure!(
        fingerprint.sampled_bytes <= fingerprint.file_size,
        "source fingerprint sampled byte count exceeds file size"
    );
    Ok(())
}

fn validate_score(value: f32, label: &str) -> Result<()> {
    ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "{label} must be finite in 0..=1"
    );
    Ok(())
}

fn clamp_score(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn valid_camera_state(state: CameraState) -> bool {
    state.cx.is_finite()
        && state.cy.is_finite()
        && state.half_size.is_finite()
        && state.half_size > 0.0
}

fn plan_source(source: CameraSource) -> PlanKeyframeSource {
    match source {
        CameraSource::Observed | CameraSource::Manual => PlanKeyframeSource::Observed,
        CameraSource::Predicted => PlanKeyframeSource::Predicted,
        CameraSource::Held => PlanKeyframeSource::Held,
    }
}

fn camera_source(source: PlanKeyframeSource) -> CameraSource {
    match source {
        PlanKeyframeSource::Observed => CameraSource::Observed,
        PlanKeyframeSource::Predicted => CameraSource::Predicted,
        PlanKeyframeSource::Held => CameraSource::Held,
    }
}

fn tracklet_confidence(tracklet: &Tracklet) -> (f32, f32, u64) {
    let mut sum = 0.0;
    let mut best: f32 = 0.0;
    let mut count = 0u64;
    let mut gaps = 0u64;
    for observation in &tracklet.observations {
        let score = clamp_score(observation.observation.composite_score());
        sum += score;
        best = best.max(score);
        count = count.saturating_add(1);
    }
    for window in tracklet.observations.windows(2) {
        gaps = gaps.saturating_add(
            window[1]
                .frame_index
                .saturating_sub(window[0].frame_index + 1),
        );
    }
    let mean = if count > 0 {
        clamp_score(sum / count as f32)
    } else {
        0.0
    };
    (mean, best, gaps)
}

fn continuity_score(tracklet: &Tracklet, gaps: u64) -> f32 {
    let span = tracklet
        .first_frame()
        .zip(tracklet.last_frame())
        .map_or(0, |(first, last)| last.saturating_sub(first) + 1);
    if span == 0 {
        return 0.0;
    }
    clamp_score(1.0 - (gaps as f32 / span as f32).clamp(0.0, 1.0))
}

fn camera_confidence(
    keyframe: &CameraKeyframe,
    tracklets: &[Tracklet],
    selected_track_ids: &std::collections::HashSet<usize>,
) -> f32 {
    if keyframe.state.source != CameraSource::Observed {
        return match keyframe.state.source {
            CameraSource::Predicted => 0.55,
            CameraSource::Held => 0.4,
            CameraSource::Manual => 1.0,
            CameraSource::Observed => 0.0,
        };
    }
    let mut best: f32 = 0.0;
    for tracklet in tracklets {
        if !selected_track_ids.contains(&tracklet.id) {
            continue;
        }
        for observation in &tracklet.observations {
            if observation.frame_index == keyframe.frame_index {
                best = best.max(clamp_score(observation.observation.composite_score()));
            }
        }
    }
    best
}

fn confidence_metrics(keyframes: &[PlanKeyframe]) -> (f32, f32) {
    if keyframes.is_empty() {
        return (0.0, 0.0);
    }
    let sum = keyframes
        .iter()
        .map(|keyframe| keyframe.confidence)
        .sum::<f32>();
    let min = keyframes
        .iter()
        .map(|keyframe| keyframe.confidence)
        .fold(1.0, f32::min);
    (clamp_score(sum / keyframes.len() as f32), clamp_score(min))
}

fn selected_max_gap(
    tracklets: &[Tracklet],
    selected_track_ids: &std::collections::HashSet<usize>,
) -> u64 {
    let mut max_gap = 0;
    for tracklet in tracklets {
        if !selected_track_ids.contains(&tracklet.id) {
            continue;
        }
        for window in tracklet.observations.windows(2) {
            max_gap = max_gap.max(
                window[1]
                    .frame_index
                    .saturating_sub(window[0].frame_index + 1),
            );
        }
    }
    max_gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        detection::BBox,
        observation::IdentityObservation,
        solver::{SolverResult, TrackletAssignment},
    };

    fn bbox(x: f32, y: f32) -> BBox {
        BBox {
            x1: x,
            y1: y,
            x2: x + 40.0,
            y2: y + 80.0,
            confidence: 0.9,
        }
    }

    fn fixture_plan() -> CropPlanV1 {
        let mut tracklet = Tracklet::new(4);
        tracklet.push(
            1,
            bbox(10.0, 20.0),
            IdentityObservation::from_face_scores(bbox(10.0, 20.0), 0.9, 0.1, 0.8, None),
        );
        tracklet.push(
            3,
            bbox(12.0, 20.0),
            IdentityObservation::from_face_scores(bbox(12.0, 20.0), 0.8, 0.1, 0.7, None),
        );
        let solved = SolverResult {
            assignments: vec![TrackletAssignment {
                tracklet_id: 4,
                identity_id: 0,
                confidence: 0.9,
            }],
            selected_identity_id: Some(0),
            camera_path: CameraPath {
                keyframes: vec![
                    CameraKeyframe {
                        frame_index: 1,
                        state: CameraState {
                            cx: 30.0,
                            cy: 60.0,
                            half_size: 40.0,
                            source: CameraSource::Observed,
                            miss_count: 0,
                        },
                    },
                    CameraKeyframe {
                        frame_index: 2,
                        state: CameraState {
                            cx: 31.0,
                            cy: 60.0,
                            half_size: 40.0,
                            source: CameraSource::Predicted,
                            miss_count: 1,
                        },
                    },
                ],
            },
        };
        CropPlanV1::from_offline_solution(
            VideoMetadata {
                width: 1920,
                height: 1080,
                frame_count: 4,
                frame_rate_num: 30,
                frame_rate_den: 1,
                duration_ms: Some(133),
            },
            SourceVideoFingerprint {
                version: SOURCE_FINGERPRINT_VERSION,
                algorithm: SOURCE_FINGERPRINT_ALGORITHM.to_string(),
                digest: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .to_string(),
                file_size: 128,
                sampled_bytes: 128,
            },
            &[tracklet],
            &solved,
            vec![ShotBoundary {
                frame_index: 2,
                confidence: 0.8,
                kind: ShotBoundaryKind::HardCut,
            }],
        )
    }

    #[test]
    fn crop_plan_round_trips_without_sensitive_fields() {
        let plan = fixture_plan();
        plan.validate().expect("fixture plan is valid");
        let json = serde_json::to_string_pretty(&plan).expect("serialize plan");
        let decoded: CropPlanV1 = serde_json::from_str(&json).expect("deserialize plan");
        assert_eq!(decoded, plan);
        assert!(!json.contains("embedding"));
        assert!(!json.contains("thumbnail"));
        assert!(!json.contains("/Users/"));
    }

    #[test]
    fn plan_is_deterministic_for_same_solver_output() {
        let left = fixture_plan();
        let right = fixture_plan();
        assert_eq!(
            serde_json::to_vec(&left).expect("serialize left"),
            serde_json::to_vec(&right).expect("serialize right")
        );
    }

    #[test]
    fn invalid_manual_geometry_and_nan_are_rejected() {
        let mut plan = fixture_plan();
        let error = plan
            .upsert_manual_keyframe(ManualKeyframe {
                frame_index: 1,
                cx: f32::NAN,
                cy: 2.0,
                half_size: 10.0,
            })
            .expect_err("NaN must be rejected");
        assert!(error.to_string().contains("finite"));
        let error = plan
            .upsert_manual_keyframe(ManualKeyframe {
                frame_index: 1,
                cx: 2.0,
                cy: 2.0,
                half_size: 0.0,
            })
            .expect_err("zero geometry must be rejected");
        assert!(error.to_string().contains("positive"));
    }

    #[test]
    fn manual_keyframe_overrides_generated_camera_state() {
        let mut plan = fixture_plan();
        plan.upsert_manual_keyframe(ManualKeyframe {
            frame_index: 1,
            cx: 99.0,
            cy: 88.0,
            half_size: 42.0,
        })
        .expect("valid manual keyframe");
        let path = plan.camera_path();
        let keyframe = path
            .keyframes
            .iter()
            .find(|keyframe| keyframe.frame_index == 1)
            .expect("manual frame");
        assert_eq!(keyframe.state.source, CameraSource::Manual);
        assert!(keyframe.state.cx.is_finite());
        assert!(keyframe.state.cy.is_finite());
    }

    #[test]
    fn frame_indices_are_one_based_and_geometry_is_normalized() {
        let mut plan = fixture_plan();
        assert!(plan.validate().is_ok());
        let error = plan
            .upsert_manual_keyframe(ManualKeyframe {
                frame_index: 0,
                cx: 30.0,
                cy: 60.0,
                half_size: 40.0,
            })
            .expect_err("frame zero must be rejected");
        assert!(error.to_string().contains("1-based"));
        plan.upsert_manual_keyframe(ManualKeyframe {
            frame_index: 3,
            cx: 0.0,
            cy: 0.0,
            half_size: 40.0,
        })
        .expect("edge geometry is normalized before persistence");
        let edge = plan
            .manual_keyframes
            .iter()
            .find(|keyframe| keyframe.frame_index == 3)
            .expect("normalized edge keyframe");
        assert!(edge.cx > 0.0);
        assert!(edge.cy > 0.0);
    }

    #[test]
    fn source_fingerprint_mismatch_is_rejected() {
        let plan = fixture_plan();
        let mut other = plan.source_fingerprint.clone();
        other.digest.replace_range(0..1, "f");
        let error = plan
            .ensure_source_fingerprint_matches(&other)
            .expect_err("different source must be rejected");
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn version_one_rejects_output_geometry_the_renderer_cannot_honor() {
        let mut plan = fixture_plan();
        plan.output.width = OUT_WIDTH + 1;

        let error = plan
            .validate()
            .expect_err("V1 output must match the fixed renderer");

        assert!(error.to_string().contains("1080x1920"));
    }
}
