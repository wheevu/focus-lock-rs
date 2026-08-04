use std::{
    collections::HashMap,
    collections::HashSet,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use fancam_core::{
    detection::DEFAULT_IDENTITY_MARGIN_THRESHOLD,
    discovery::{DiscoveryConfig, DiscoveryEngine},
    mode::ProcessingMode,
    pipeline::{OfflinePrepassProgress, Pipeline},
    plan::{
        CropPlanV1, ManualKeyframe, read_crop_plan as read_crop_plan_file,
        write_crop_plan as write_crop_plan_file,
    },
    runtime::OrtConfig,
    video::{
        fingerprint_video, probe_video_metadata, total_frames,
        transcode_with_progress_staged_mode_fallible,
    },
};
use image::ImageReader;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tokio::task;

use crate::{
    CancelFlag, IdentityScanState, IdentityScanStore, QueueStore, QueueWorkerStore, RenderJobStore,
    ScanJobState, ScanJobStore, StorageWorkerStore, queue, storage,
};

static RUN_ID_SEQ: AtomicU64 = AtomicU64::new(1);

const MAX_PREVIEW_DIMENSION: u32 = 4096;
const MAX_PREVIEW_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_PREVIEW_ALLOC: u64 = 64 * 1024 * 1024;
const MAX_CROP_PLAN_BYTES: u64 = 4 * 1024 * 1024;

// ─── DTO types ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FancamArgs {
    pub video: String,
    pub bias: String,
    pub output: String,
    pub yolo_model: String,
    pub face_model: String,
    #[serde(default)]
    pub identity_model: Option<String>,
    pub threshold: f32,
    pub scan_id: Option<String>,
    pub selected_identity_id: Option<usize>,
    pub target_anchor_x: Option<f32>,
    pub target_anchor_y: Option<f32>,
    pub processing_mode: Option<String>,
    #[serde(default)]
    pub body_reid_model: Option<String>,
    #[serde(default)]
    pub target_embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub target_embeddings: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub body_target_embeddings: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub negative_embeddings: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub identity_margin_threshold: Option<f32>,
    pub expected_member_count: Option<u32>,
    #[serde(default)]
    pub excluded_identity_ids: Vec<usize>,
    #[serde(default)]
    pub accepted_low_confidence_ids: Vec<usize>,
    #[serde(default)]
    pub resolved_duplicates: Vec<ReviewDuplicateResolution>,
    #[serde(default)]
    pub pending_split_ids: Vec<usize>,
    #[serde(default)]
    pub client_run_id: Option<String>,
    #[serde(default)]
    pub plan_input: Option<String>,
    #[serde(default)]
    pub plan_output: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IdentityScanArgs {
    pub video: String,
    pub yolo_model: String,
    pub face_model: String,
    #[serde(default)]
    pub identity_model: Option<String>,
    #[serde(default)]
    pub body_reid_model: Option<String>,
    pub expected_member_count: Option<u32>,
    pub processing_mode: Option<String>,
    pub client_run_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CropPlanPathArgs {
    pub path: String,
    #[serde(default)]
    pub video: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CropPlanWriteArgs {
    pub path: String,
    pub plan: CropPlanV1,
}

#[derive(Debug, Deserialize)]
pub struct UpdateCropPlanKeyframeArgs {
    pub path: String,
    pub keyframe: ManualKeyframe,
}

#[derive(Debug, Serialize)]
pub struct CropPlanWriteResult {
    pub path: String,
    pub bytes: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IdentityCandidatePayload {
    pub id: usize,
    pub confidence: f32,
    pub observations: u32,
    pub first_frame: u64,
    pub last_frame: u64,
    pub anchor_x: f32,
    pub anchor_y: f32,
    #[serde(default)]
    pub anchor_x_norm: Option<f32>,
    #[serde(default)]
    pub anchor_y_norm: Option<f32>,
    pub thumbnail_data_url: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub body_embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub preview_score: Option<f32>,
    #[serde(default)]
    pub preview_observations: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DuplicatePairPayload {
    pub a: usize,
    pub b: usize,
    pub similarity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IdentityScanCache {
    pub video: String,
    pub yolo_model: String,
    pub face_model: String,
    #[serde(default)]
    pub identity_model: Option<String>,
    pub processing_mode: String,
    pub expected_count: Option<u32>,
    pub candidates: Vec<IdentityCandidatePayload>,
    pub duplicates: Vec<DuplicatePairPayload>,
    pub review_ready: bool,
    pub selected_identity_id: Option<usize>,
    pub selected_anchor_x: Option<f32>,
    pub selected_anchor_y: Option<f32>,
    pub validated_threshold: Option<f32>,
    pub last_blockers: Vec<String>,
    pub updated_at_ms: u64,
    pub status: ScanSessionStatus,
    pub events: Vec<ScanSessionEvent>,
    pub excluded_identity_ids: Vec<usize>,
    pub accepted_low_confidence_ids: Vec<usize>,
    pub resolved_duplicates: Vec<ReviewDuplicateResolution>,
    pub pending_split_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScanSessionStatus {
    Proposed,
    Validated,
    Tracking,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSessionEvent {
    pub at_ms: u64,
    pub action: String,
    pub details: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ScanSessionSummary {
    pub scan_id: String,
    pub video: String,
    pub status: ScanSessionStatus,
    pub review_ready: bool,
    pub selected_identity_id: Option<usize>,
    pub pending_split_count: usize,
    pub event_count: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Serialize, Clone)]
pub struct ScanSessionDetail {
    pub scan_id: String,
    pub video: String,
    pub yolo_model: String,
    pub identity_model: String,
    pub status: ScanSessionStatus,
    pub expected_count: Option<u32>,
    pub processing_mode: String,
    pub review_ready: bool,
    pub selected_identity_id: Option<usize>,
    pub selected_anchor_x: Option<f32>,
    pub selected_anchor_y: Option<f32>,
    pub validated_threshold: Option<f32>,
    pub last_blockers: Vec<String>,
    pub candidates: Vec<IdentityCandidatePayload>,
    pub duplicates: Vec<DuplicatePairPayload>,
    pub excluded_identity_ids: Vec<usize>,
    pub accepted_low_confidence_ids: Vec<usize>,
    pub resolved_duplicates: Vec<ReviewDuplicateResolution>,
    pub pending_split_ids: Vec<usize>,
    pub updated_at_ms: u64,
    pub event_count: usize,
    pub recent_events: Vec<ScanSessionEvent>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedScanState {
    next_id: u64,
    scans: HashMap<String, IdentityScanCache>,
}

#[derive(Debug, Serialize, Clone)]
pub struct IdentityScanResult {
    pub scan_id: String,
    pub ok: bool,
    pub message: String,
    pub video: String,
    pub processing_mode: String,
    pub sampled_frames: u64,
    pub total_decoded_frames: u64,
    pub proposed_count: usize,
    pub expected_count: Option<u32>,
    pub rescan_performed: bool,
    pub needs_review: bool,
    pub rejected_embeddings: u64,
    pub suppressed_clusters: usize,
    pub merged_clusters: usize,
    pub provisional_tracklets: usize,
    pub candidates: Vec<IdentityCandidatePayload>,
    pub duplicates: Vec<DuplicatePairPayload>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ReviewDuplicateResolution {
    pub a: usize,
    pub b: usize,
    pub keep: usize,
}

#[derive(Debug, Deserialize)]
pub struct ValidateIdentityReviewArgs {
    pub scan_id: String,
    pub selected_identity_id: Option<usize>,
    pub threshold: f32,
    pub excluded_identity_ids: Vec<usize>,
    pub accepted_low_confidence_ids: Vec<usize>,
    pub resolved_duplicates: Vec<ReviewDuplicateResolution>,
    pub pending_split_ids: Vec<usize>,
    pub expected_member_count: Option<u32>,
}

#[derive(Debug, Serialize, Clone)]
pub struct IdentityReviewResult {
    pub ok: bool,
    pub ready: bool,
    pub blockers: Vec<String>,
    pub active_count: usize,
    pub selected_identity_id: Option<usize>,
    pub selected_anchor_x: Option<f32>,
    pub selected_anchor_y: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct EnqueueDiscoveryJobArgs {
    pub scan_id: String,
    pub video: String,
    pub yolo_model: String,
    pub face_model: String,
    #[serde(default)]
    pub identity_model: Option<String>,
    pub expected_member_count: Option<u32>,
    pub processing_mode: Option<String>,
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EnqueueSplitRescanArgs {
    pub scan_id: String,
    pub processing_mode: Option<String>,
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProcessNextDiscoveryJobArgs {
    pub max_attempts_before_dlq: Option<u32>,
    pub client_run_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct QueuePeekArgs {
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct QueuePeekResult {
    pub attempts: Vec<u32>,
}

#[derive(Debug, Deserialize)]
pub struct QueueDlqListArgs {
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct QueueDlqReplayArgs {
    pub message_id: String,
}

#[derive(Debug, Deserialize)]
pub struct QueryIdentityScansArgs {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub status: Option<String>,
    pub cursor_updated_at_ms: Option<u64>,
    pub cursor_scan_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct QueryIdentityScansResult {
    pub rows: Vec<ScanSessionSummary>,
    pub next_cursor_updated_at_ms: Option<u64>,
    pub next_cursor_scan_id: Option<String>,
    pub offset_ignored: bool,
}

#[derive(Debug, Deserialize)]
pub struct QueryScanEventsArgs {
    pub scan_id: String,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub action_contains: Option<String>,
    pub since_ms: Option<u64>,
    pub until_ms: Option<u64>,
    pub cursor_event_id: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct QueryScanEventsResult {
    pub rows: Vec<ScanSessionEvent>,
    pub next_cursor_event_id: Option<u64>,
    pub offset_ignored: bool,
}

#[derive(Debug, Serialize)]
pub struct ScanStorageStats {
    pub schema_version: i64,
    pub session_count: u64,
    pub event_count: u64,
    pub db_path: String,
}

#[derive(Debug, Deserialize)]
pub struct ScanStorageMaintenanceArgs {
    pub max_session_age_ms: Option<u64>,
    pub max_events_per_scan: Option<u32>,
    pub vacuum: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ScanStorageMaintenanceResult {
    pub deleted_sessions: u64,
    pub deleted_events: u64,
    pub vacuum_ran: bool,
    pub stats: ScanStorageStats,
}

#[derive(Debug, Deserialize)]
pub struct ExportDiagnosticsArgs {
    pub scan_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExportDiagnosticsResult {
    pub path: String,
    pub bytes: usize,
}

#[derive(Debug, Deserialize)]
pub struct ListDiagnosticsBundlesArgs {
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct DiagnosticsBundleInfo {
    pub file_name: String,
    pub path: String,
    pub bytes: u64,
    pub modified_at_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ListDiagnosticsBundlesResult {
    pub bundles: Vec<DiagnosticsBundleInfo>,
}

#[derive(Debug, Deserialize)]
pub struct StorageWorkerStartArgs {
    pub poll_interval_ms: Option<u64>,
    pub max_session_age_ms: Option<u64>,
    pub max_events_per_scan: Option<u32>,
    pub vacuum: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct StorageWorkerStatus {
    pub running: bool,
    pub stop_requested: bool,
    pub poll_interval_ms: u64,
    pub max_session_age_ms: u64,
    pub max_events_per_scan: u32,
    pub vacuum: bool,
    pub runs_total: u64,
    pub last_run_ms: Option<u64>,
    pub last_error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct QueueWorkerStartArgs {
    pub poll_interval_ms: Option<u64>,
    pub max_attempts_before_dlq: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct QueueWorkerStatus {
    pub running: bool,
    pub stop_requested: bool,
    pub poll_interval_ms: u64,
    pub max_attempts_before_dlq: u32,
    pub processed_total: u64,
    pub last_error: Option<String>,
    pub recent_events: Vec<QueueWorkerEventPayload>,
}

#[derive(Debug, Serialize, Clone)]
pub struct QueueWorkerEventPayload {
    pub at_ms: u64,
    pub queue: String,
    pub message_id: Option<String>,
    pub job_id: Option<String>,
    pub attempt: Option<u32>,
    pub moved_to_dlq: bool,
    pub requeued: bool,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ProgressPayload {
    pub run_id: String,
    pub current: u64,
    pub total: u64,
    pub fraction: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct ScanProgressPayload {
    pub run_id: String,
    pub sampled_frames: u64,
    pub total_decoded_frames: u64,
    pub estimated_total_samples: u64,
    pub pass_fraction: f64,
    pub overall_fraction: f64,
    pub phase: String,
    pub pass_index: u8,
    pub pass_total: u8,
}

#[derive(Debug, Serialize, Clone)]
pub struct ScanDonePayload {
    pub run_id: String,
    pub ok: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct JobResult {
    pub ok: bool,
    pub message: String,
    pub output_path: Option<String>,
    pub run_id: Option<String>,
}

fn emit_render_done(app: &AppHandle, result: &JobResult) {
    let _ = app.emit("fancam://done", result.clone());
}

// ─── Commands ─────────────────────────────────────────────────────────────────

/// Return the absolute path to the `models/` directory sitting next to the
/// running binary (works both in `cargo run` dev builds and release bundles).
#[tauri::command]
pub fn model_dir() -> String {
    // std::env::current_exe() → <repo>/target/debug/focus-lock
    // Walking up to find a sibling `models/` directory.
    if let Ok(exe) = std::env::current_exe() {
        // Walk up from the binary looking for a models/ directory
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        while let Some(d) = dir {
            let candidate = d.join("models");
            if candidate.is_dir() {
                return candidate.to_string_lossy().into_owned();
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
    }
    // Fallback: models/ relative to CWD
    std::env::current_dir()
        .map(|d| d.join("models").to_string_lossy().into_owned())
        .unwrap_or_else(|_| "models".to_string())
}

fn validate_crop_plan_path(path: &str, for_write: bool) -> Result<PathBuf, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("crop plan path is empty".to_string());
    }
    let path = PathBuf::from(trimmed);
    if for_write {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(format!(
                "crop plan parent is not a directory: {}",
                parent.display()
            ));
        }
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if metadata.file_type().is_symlink() {
                return Err("crop plan path must not be a symlink".to_string());
            }
            if !metadata.is_file() {
                return Err("crop plan path is not a regular file".to_string());
            }
        }
        return Ok(path);
    }

    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("failed to inspect crop plan {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err("crop plan path must not be a symlink".to_string());
    }
    if !metadata.is_file() {
        return Err("crop plan path is not a regular file".to_string());
    }
    if metadata.len() > MAX_CROP_PLAN_BYTES {
        return Err(format!(
            "crop plan is larger than the {} byte limit",
            MAX_CROP_PLAN_BYTES
        ));
    }
    Ok(path)
}

#[tauri::command]
pub fn read_crop_plan(args: CropPlanPathArgs) -> Result<CropPlanV1, String> {
    let path = validate_crop_plan_path(&args.path, false)?;
    let plan = read_crop_plan_file(&path).map_err(|error| error.to_string())?;
    let video = args
        .video
        .as_deref()
        .map(str::trim)
        .filter(|video| !video.is_empty())
        .ok_or_else(|| "a source video is required to load a crop plan".to_string())?;
    ensure_crop_plan_matches_video(&plan, video)?;
    Ok(plan)
}

fn ensure_crop_plan_matches_video(plan: &CropPlanV1, video: &str) -> Result<(), String> {
    let probed = probe_video_metadata(video).map_err(|error| error.to_string())?;
    ensure_crop_plan_metadata_matches(plan, &probed)?;

    // Container frame totals may be estimates. The plan's count came from the
    // decoded prepass, so use it when rebuilding the fingerprint contract.
    let mut fingerprint_metadata = probed;
    fingerprint_metadata.frame_count = plan.video.frame_count;
    let fingerprint =
        fingerprint_video(video, &fingerprint_metadata).map_err(|error| error.to_string())?;
    plan.ensure_source_fingerprint_matches(&fingerprint)
        .map_err(|error| error.to_string())
}

fn ensure_crop_plan_metadata_matches(
    plan: &CropPlanV1,
    probed: &fancam_core::plan::VideoMetadata,
) -> Result<(), String> {
    if plan.video.width != probed.width
        || plan.video.height != probed.height
        || plan.video.frame_rate_num != probed.frame_rate_num
        || plan.video.frame_rate_den != probed.frame_rate_den
        || plan.video.duration_ms != probed.duration_ms
    {
        return Err("crop plan video metadata does not match the selected video".to_string());
    }
    Ok(())
}

#[tauri::command]
pub fn write_crop_plan(args: CropPlanWriteArgs) -> Result<CropPlanWriteResult, String> {
    let path = validate_crop_plan_path(&args.path, true)?;
    let serialized = serde_json::to_vec(&args.plan).map_err(|error| error.to_string())?;
    if serialized.len() as u64 > MAX_CROP_PLAN_BYTES {
        return Err(format!(
            "crop plan is larger than the {} byte limit",
            MAX_CROP_PLAN_BYTES
        ));
    }
    write_crop_plan_file(&path, &args.plan).map_err(|error| error.to_string())?;
    let bytes = fs::metadata(&path)
        .map_err(|error| error.to_string())?
        .len();
    Ok(CropPlanWriteResult {
        path: path.to_string_lossy().into_owned(),
        bytes,
    })
}

#[tauri::command]
pub fn update_crop_plan_keyframe(args: UpdateCropPlanKeyframeArgs) -> Result<CropPlanV1, String> {
    let path = validate_crop_plan_path(&args.path, false)?;
    let mut plan = read_crop_plan_file(&path).map_err(|error| error.to_string())?;
    plan.upsert_manual_keyframe(args.keyframe)
        .map_err(|error| error.to_string())?;
    let serialized = serde_json::to_vec(&plan).map_err(|error| error.to_string())?;
    if serialized.len() as u64 > MAX_CROP_PLAN_BYTES {
        return Err(format!(
            "crop plan is larger than the {} byte limit",
            MAX_CROP_PLAN_BYTES
        ));
    }
    write_crop_plan_file(&path, &plan).map_err(|error| error.to_string())?;
    Ok(plan)
}

#[tauri::command]
pub async fn probe_video(path: String) -> Result<u64, String> {
    let path = validate_preview_path(&path)?;
    task::spawn_blocking(move || {
        let frames = total_frames(&path);
        if frames == 0 {
            Err(format!(
                "could not determine a video frame count for {}",
                path.display()
            ))
        } else {
            Ok(frames)
        }
    })
    .await
    .map_err(|e| format!("video probe task failed: {e}"))?
}

/// Read an image file and return a small JPEG data-URL suitable for preview.
/// For video files we decode the first frame via ffmpeg; for images we just
/// read + transcode.  The result fits in an `<img src="...">` attribute.
#[tauri::command]
pub async fn read_thumbnail(path: String) -> Result<String, String> {
    let path = validate_preview_path(&path)?;
    task::spawn_blocking(move || make_thumbnail(&path.to_string_lossy()))
        .await
        .map_err(|e| format!("thumbnail task failed: {e}"))?
}

/// Guard that marks the active scan job and exposes a per-scan cancel flag.
/// Always resets the running state on drop, even on mutex poisoning.
#[derive(Debug)]
struct ScanJobGuard<'a> {
    store: &'a ScanJobStore,
}

impl<'a> ScanJobGuard<'a> {
    fn acquire(store: &'a ScanJobStore) -> Result<(Self, Arc<AtomicBool>), String> {
        let mut job = match store.0.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                let mut g = poisoned.into_inner();
                g.running = false;
                g.cancelling = false;
                g.cancel.store(false, Ordering::Relaxed);
                drop(g);
                store.0.clear_poison();
                store.0.lock().map_err(|e| e.to_string())?
            }
        };
        if job.running {
            let message = if job.cancelling {
                "scan cancellation is in progress; wait for stop to finish".to_string()
            } else {
                "an identity scan is already running".to_string()
            };
            return Err(message);
        }
        job.running = true;
        job.cancelling = false;
        job.cancel.store(false, Ordering::Relaxed);
        let cancel = Arc::clone(&job.cancel);
        Ok((Self { store }, cancel))
    }
}

impl<'a> Drop for ScanJobGuard<'a> {
    fn drop(&mut self) {
        match self.store.0.lock() {
            Ok(mut job) => {
                job.running = false;
                job.cancelling = false;
            }
            Err(poisoned) => {
                let mut job = poisoned.into_inner();
                job.running = false;
                job.cancelling = false;
                drop(job);
                self.store.0.clear_poison();
            }
        }
    }
}

#[tauri::command]
pub async fn scan_identities(
    app: AppHandle,
    state: State<'_, IdentityScanStore>,
    scan_job_state: State<'_, ScanJobStore>,
    args: IdentityScanArgs,
) -> Result<IdentityScanResult, String> {
    validate_identity_scan_paths(&args)?;
    let run_id = args
        .client_run_id
        .clone()
        .unwrap_or_else(|| next_run_id("scan"));

    let (_guard, cancel_flag) = ScanJobGuard::acquire(&scan_job_state)?;

    let app_for_scan = app.clone();
    let yolo_model = args.yolo_model.clone();
    let identity_model =
        effective_identity_model(&args.face_model, args.identity_model.as_deref()).to_string();
    let progress_run_id = run_id.clone();
    let scan_result = task::spawn_blocking(move || {
        run_identity_scan_with_hooks(
            IdentityScanArgs {
                identity_model: args.identity_model.clone(),
                processing_mode: sanitize_processing_mode(args.processing_mode.as_deref()),
                body_reid_model: args.body_reid_model.clone(),
                ..args
            },
            move |progress| {
                let _ = app_for_scan.emit(
                    "scan://progress",
                    ScanProgressPayload {
                        run_id: progress_run_id.clone(),
                        sampled_frames: progress.sampled_frames,
                        total_decoded_frames: progress.total_decoded_frames,
                        estimated_total_samples: progress.estimated_total_samples,
                        pass_fraction: progress.pass_fraction,
                        overall_fraction: progress.overall_fraction,
                        phase: progress.phase,
                        pass_index: progress.pass_index,
                        pass_total: progress.pass_total,
                    },
                );
            },
            move || cancel_flag.load(Ordering::Relaxed),
        )
    })
    .await
    .map_err(|e| e.to_string())?;

    let scan_result = match scan_result {
        Ok(result) => result,
        Err(err) => {
            let _ = app.emit(
                "scan://done",
                ScanDonePayload {
                    run_id: run_id.clone(),
                    ok: false,
                    message: err.clone(),
                },
            );
            return Err(err);
        }
    };

    tracing::info!(
        proposed = scan_result.proposed_count,
        rejected_embeddings = scan_result.rejected_embeddings,
        suppressed_clusters = scan_result.suppressed_clusters,
        merged_clusters = scan_result.merged_clusters,
        provisional_tracklets = scan_result.provisional_tracklets,
        mode = %scan_result.processing_mode,
        "identity discovery stats"
    );

    let (scan_id, snapshot) = {
        let mut lock = state.0.lock().map_err(|e| e.to_string())?;
        ensure_scan_store_loaded(&mut lock);
        lock.next_id += 1;
        let scan_id = format!("scan-{}", lock.next_id);
        upsert_scan_cache(
            &mut lock.scans,
            &scan_id,
            &scan_result,
            &yolo_model,
            &identity_model,
        );
        let snapshot = snapshot_scan_entry(&lock, &scan_id)?;
        (scan_id, snapshot)
    };
    if let Some(snapshot) = snapshot.as_ref() {
        persist_scan_entry_snapshot(snapshot)?;
    }

    let _ = app.emit(
        "scan://done",
        ScanDonePayload {
            run_id,
            ok: true,
            message: "Identity scan complete".to_string(),
        },
    );

    // Keep full candidate payload (including embeddings) in backend storage,
    // but return a lighter client payload to avoid UI thread stalls at scan
    // completion.
    let mut response = scan_result;
    response.scan_id = scan_id;
    response.candidates = strip_candidate_embeddings(&response.candidates);
    Ok(response)
}

fn strip_candidate_embeddings(
    candidates: &[IdentityCandidatePayload],
) -> Vec<IdentityCandidatePayload> {
    candidates
        .iter()
        .cloned()
        .map(|mut candidate| {
            candidate.embedding = None;
            candidate.body_embedding = None;
            candidate
        })
        .collect()
}

fn upsert_scan_cache(
    scans: &mut std::collections::HashMap<String, IdentityScanCache>,
    scan_id: &str,
    scan_result: &IdentityScanResult,
    yolo_model: &str,
    identity_model: &str,
) {
    let now = epoch_ms();
    let mut events = Vec::new();
    events.push(ScanSessionEvent {
        at_ms: now,
        action: "scan_created".to_string(),
        details: format!(
            "proposed={} needs_review={} expected_count={}",
            scan_result.proposed_count,
            scan_result.needs_review,
            scan_result
                .expected_count
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string())
        ),
    });
    scans.insert(
        scan_id.to_string(),
        IdentityScanCache {
            video: scan_result.video.clone(),
            yolo_model: yolo_model.to_string(),
            face_model: identity_model.to_string(),
            identity_model: Some(identity_model.to_string()),
            processing_mode: scan_result.processing_mode.clone(),
            expected_count: scan_result.expected_count,
            candidates: scan_result.candidates.clone(),
            duplicates: scan_result.duplicates.clone(),
            review_ready: false,
            selected_identity_id: None,
            selected_anchor_x: None,
            selected_anchor_y: None,
            validated_threshold: None,
            last_blockers: Vec::new(),
            updated_at_ms: now,
            status: ScanSessionStatus::Proposed,
            events,
            excluded_identity_ids: Vec::new(),
            accepted_low_confidence_ids: Vec::new(),
            resolved_duplicates: Vec::new(),
            pending_split_ids: Vec::new(),
        },
    );
}

fn append_scan_event(scan: &mut IdentityScanCache, action: &str, details: String) {
    scan.events.push(ScanSessionEvent {
        at_ms: epoch_ms(),
        action: action.to_string(),
        details,
    });
    if scan.events.len() > 200 {
        let keep_from = scan.events.len().saturating_sub(200);
        scan.events.drain(0..keep_from);
    }
}

fn can_transition_status(from: &ScanSessionStatus, to: &ScanSessionStatus) -> bool {
    if from == to {
        return true;
    }
    matches!(
        (from, to),
        (ScanSessionStatus::Proposed, ScanSessionStatus::Validated)
            | (ScanSessionStatus::Validated, ScanSessionStatus::Proposed)
            | (ScanSessionStatus::Validated, ScanSessionStatus::Tracking)
            | (ScanSessionStatus::Tracking, ScanSessionStatus::Validated)
            | (ScanSessionStatus::Tracking, ScanSessionStatus::Completed)
            | (ScanSessionStatus::Tracking, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Proposed, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Validated, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Completed, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Failed, ScanSessionStatus::Proposed)
    )
}

fn set_scan_status(scan: &mut IdentityScanCache, to: ScanSessionStatus) -> Result<(), String> {
    if !can_transition_status(&scan.status, &to) {
        return Err(format!(
            "invalid scan session transition: {} -> {}",
            status_to_db(&scan.status),
            status_to_db(&to)
        ));
    }
    scan.status = to;
    Ok(())
}

fn status_to_db(status: &ScanSessionStatus) -> &'static str {
    match status {
        ScanSessionStatus::Proposed => "proposed",
        ScanSessionStatus::Validated => "validated",
        ScanSessionStatus::Tracking => "tracking",
        ScanSessionStatus::Completed => "completed",
        ScanSessionStatus::Failed => "failed",
    }
}

fn status_from_db(value: &str) -> ScanSessionStatus {
    match value {
        "validated" => ScanSessionStatus::Validated,
        "tracking" => ScanSessionStatus::Tracking,
        "completed" => ScanSessionStatus::Completed,
        "failed" => ScanSessionStatus::Failed,
        _ => ScanSessionStatus::Proposed,
    }
}

fn scan_to_row(scan_id: &str, scan: &IdentityScanCache) -> Result<storage::ScanSessionRow, String> {
    let identity_model = scan
        .identity_model
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&scan.face_model)
        .to_string();
    Ok(storage::ScanSessionRow {
        scan_id: scan_id.to_string(),
        video: scan.video.clone(),
        yolo_model: scan.yolo_model.clone(),
        identity_model,
        status: status_to_db(&scan.status).to_string(),
        processing_mode: processing_mode_string(Some(&scan.processing_mode)),
        expected_count: scan.expected_count.map(|v| v as i64),
        review_ready: scan.review_ready,
        selected_identity_id: scan.selected_identity_id.map(|v| v as i64),
        selected_anchor_x: scan.selected_anchor_x,
        selected_anchor_y: scan.selected_anchor_y,
        validated_threshold: scan.validated_threshold,
        updated_at_ms: scan.updated_at_ms,
        candidates_json: serde_json::to_string(&scan.candidates)
            .map_err(|e| format!("failed to serialize candidates: {e}"))?,
        duplicates_json: serde_json::to_string(&scan.duplicates)
            .map_err(|e| format!("failed to serialize duplicates: {e}"))?,
        excluded_identity_ids_json: serde_json::to_string(&scan.excluded_identity_ids)
            .map_err(|e| format!("failed to serialize excluded ids: {e}"))?,
        accepted_low_confidence_ids_json: serde_json::to_string(&scan.accepted_low_confidence_ids)
            .map_err(|e| format!("failed to serialize accepted low-confidence ids: {e}"))?,
        resolved_duplicate_keys_json: serde_json::to_string(&scan.resolved_duplicates)
            .map_err(|e| format!("failed to serialize resolved duplicate decisions: {e}"))?,
        pending_split_ids_json: serde_json::to_string(&scan.pending_split_ids)
            .map_err(|e| format!("failed to serialize pending split ids: {e}"))?,
        pending_split_count: scan.pending_split_ids.len() as i64,
        last_blockers_json: serde_json::to_string(&scan.last_blockers)
            .map_err(|e| format!("failed to serialize blockers: {e}"))?,
    })
}

fn row_to_scan(
    row: &storage::ScanSessionRow,
    events: Vec<ScanSessionEvent>,
) -> Result<IdentityScanCache, String> {
    let resolved_duplicates = serde_json::from_str(&row.resolved_duplicate_keys_json)
        .or_else(|_| {
            serde_json::from_str::<Vec<(usize, usize)>>(&row.resolved_duplicate_keys_json).map(
                |pairs| {
                    pairs
                        .into_iter()
                        .map(|(a, b)| ReviewDuplicateResolution { a, b, keep: a })
                        .collect::<Vec<_>>()
                },
            )
        })
        .map_err(|e| format!("failed to deserialize resolved duplicate decisions: {e}"))?;

    Ok(IdentityScanCache {
        video: row.video.clone(),
        yolo_model: row.yolo_model.clone(),
        face_model: row.identity_model.clone(),
        identity_model: Some(row.identity_model.clone()),
        processing_mode: processing_mode_string(Some(&row.processing_mode)),
        expected_count: row.expected_count.map(|v| v as u32),
        candidates: serde_json::from_str(&row.candidates_json)
            .map_err(|e| format!("failed to deserialize candidates: {e}"))?,
        duplicates: serde_json::from_str(&row.duplicates_json)
            .map_err(|e| format!("failed to deserialize duplicates: {e}"))?,
        review_ready: row.review_ready,
        selected_identity_id: row.selected_identity_id.map(|v| v as usize),
        selected_anchor_x: row.selected_anchor_x,
        selected_anchor_y: row.selected_anchor_y,
        validated_threshold: row.validated_threshold,
        last_blockers: serde_json::from_str(&row.last_blockers_json)
            .map_err(|e| format!("failed to deserialize blockers: {e}"))?,
        updated_at_ms: row.updated_at_ms,
        status: status_from_db(&row.status),
        events,
        excluded_identity_ids: serde_json::from_str(&row.excluded_identity_ids_json)
            .map_err(|e| format!("failed to deserialize excluded ids: {e}"))?,
        accepted_low_confidence_ids: serde_json::from_str(&row.accepted_low_confidence_ids_json)
            .map_err(|e| format!("failed to deserialize accepted low-confidence ids: {e}"))?,
        resolved_duplicates,
        pending_split_ids: serde_json::from_str(&row.pending_split_ids_json)
            .map_err(|e| format!("failed to deserialize pending split ids: {e}"))?,
    })
}

fn ensure_scan_store_loaded(state: &mut IdentityScanState) {
    if state.loaded_from_disk {
        return;
    }
    let db_path = storage::scan_store_db_path();
    if let Ok(Some(rows)) = storage::load_scan_rows(&db_path) {
        let mut events_by_scan = std::collections::HashMap::<String, Vec<ScanSessionEvent>>::new();
        for event in rows.events {
            events_by_scan
                .entry(event.scan_id)
                .or_default()
                .push(ScanSessionEvent {
                    at_ms: event.at_ms,
                    action: event.action,
                    details: event.details,
                });
        }

        let mut scans = std::collections::HashMap::new();
        for session in rows.sessions {
            let events = events_by_scan.remove(&session.scan_id).unwrap_or_default();
            if let Ok(scan) = row_to_scan(&session, events) {
                scans.insert(session.scan_id, scan);
            }
        }

        let mut recovered_scan = false;
        for scan in scans.values_mut() {
            if scan.status == ScanSessionStatus::Tracking {
                scan.status = ScanSessionStatus::Validated;
                scan.updated_at_ms = epoch_ms();
                append_scan_event(
                    scan,
                    "tracking_recovered",
                    "application restarted while tracking; session returned to validated"
                        .to_string(),
                );
                recovered_scan = true;
            }
        }

        state.next_id = rows.next_id;
        state.scans = scans;
        state.loaded_from_disk = true;
        if recovered_scan && let Err(error) = persist_scan_store(state) {
            tracing::warn!(%error, "failed to persist recovered scan sessions");
        }
        return;
    }

    let legacy_json_path = storage::scan_store_json_path();
    if let Ok(bytes) = fs::read(&legacy_json_path)
        && let Ok(persisted) = serde_json::from_slice::<PersistedScanState>(&bytes)
    {
        state.next_id = persisted.next_id;
        state.scans = persisted.scans;
        let mut recovered_scan = false;
        for scan in state.scans.values_mut() {
            if scan.status == ScanSessionStatus::Tracking {
                scan.status = ScanSessionStatus::Validated;
                scan.updated_at_ms = epoch_ms();
                append_scan_event(
                    scan,
                    "tracking_recovered",
                    "application restarted while tracking; session returned to validated"
                        .to_string(),
                );
                recovered_scan = true;
            }
        }
        let _ = persist_scan_store(state);
        if recovered_scan {
            tracing::info!("recovered interrupted legacy tracking session");
        }
    }
    state.loaded_from_disk = true;
}

fn persist_scan_store(state: &IdentityScanState) -> Result<(), String> {
    let mut sessions = Vec::new();
    let mut events = Vec::new();
    for (scan_id, scan) in &state.scans {
        sessions.push(scan_to_row(scan_id, scan)?);
        for event in &scan.events {
            events.push(storage::ScanSessionEventRow {
                scan_id: scan_id.clone(),
                at_ms: event.at_ms,
                action: event.action.clone(),
                details: event.details.clone(),
            });
        }
    }
    storage::save_scan_rows(
        &storage::scan_store_db_path(),
        &storage::ScanStoreRows {
            next_id: state.next_id,
            sessions,
            events,
        },
    )
}

struct ScanEntrySnapshot {
    next_id: u64,
    session: storage::ScanSessionRow,
    events: Vec<storage::ScanSessionEventRow>,
}

fn snapshot_scan_entry(
    state: &IdentityScanState,
    scan_id: &str,
) -> Result<Option<ScanEntrySnapshot>, String> {
    let Some(scan) = state.scans.get(scan_id) else {
        return Ok(None);
    };
    let session = scan_to_row(scan_id, scan)?;
    let events = scan
        .events
        .iter()
        .map(|event| storage::ScanSessionEventRow {
            scan_id: scan_id.to_string(),
            at_ms: event.at_ms,
            action: event.action.clone(),
            details: event.details.clone(),
        })
        .collect::<Vec<_>>();
    Ok(Some(ScanEntrySnapshot {
        next_id: state.next_id,
        session,
        events,
    }))
}

fn persist_scan_entry_snapshot(snapshot: &ScanEntrySnapshot) -> Result<(), String> {
    storage::save_scan_row(
        &storage::scan_store_db_path(),
        snapshot.next_id,
        &snapshot.session,
        &snapshot.events,
    )
}

fn delete_scan_entries(scan_ids: &[String]) -> Result<(), String> {
    storage::delete_scan_rows(&storage::scan_store_db_path(), scan_ids)
}

fn validate_preview_path(path: &str) -> Result<PathBuf, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("preview path is empty".to_string());
    }
    let path = PathBuf::from(trimmed);
    let metadata = fs::metadata(&path)
        .map_err(|e| format!("cannot access preview input {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("preview input is not a file: {}", path.display()));
    }
    Ok(path)
}

fn validate_preview_dimensions(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("preview input has an empty frame".to_string());
    }
    if width > MAX_PREVIEW_DIMENSION || height > MAX_PREVIEW_DIMENSION {
        return Err(format!(
            "preview input exceeds the {}px dimension limit",
            MAX_PREVIEW_DIMENSION
        ));
    }
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels > MAX_PREVIEW_PIXELS {
        return Err("preview input exceeds the pixel limit".to_string());
    }
    Ok(())
}

fn make_thumbnail(path: &str) -> Result<String, String> {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();

    let is_video = matches!(
        ext.as_str(),
        "mp4" | "mov" | "mkv" | "avi" | "webm" | "ts" | "flv"
    );

    let rgb_image = if is_video {
        extract_video_frame(path).map_err(|e| format!("video frame extraction: {e}"))?
    } else {
        let mut reader = ImageReader::open(path).map_err(|e| format!("open image: {e}"))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_PREVIEW_DIMENSION);
        limits.max_image_height = Some(MAX_PREVIEW_DIMENSION);
        limits.max_alloc = Some(MAX_PREVIEW_ALLOC);
        reader.limits(limits);
        reader
            .decode()
            .map_err(|e| format!("decode image: {e}"))?
            .to_rgb8()
    };

    validate_preview_dimensions(rgb_image.width(), rgb_image.height())?;

    // Resize preserving aspect ratio — fit within 280px on the longest edge
    let (src_w, src_h) = (rgb_image.width() as f64, rgb_image.height() as f64);
    let max_dim = 280.0;
    let scale = (max_dim / src_w).min(max_dim / src_h).min(1.0);
    let dst_w = (src_w * scale).round().max(1.0) as u32;
    let dst_h = (src_h * scale).round().max(1.0) as u32;

    let thumb = image::imageops::resize(
        &rgb_image,
        dst_w,
        dst_h,
        image::imageops::FilterType::Triangle,
    );

    let mut jpeg_buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(thumb.width(), thumb.height(), thumb.into_raw())
            .ok_or("thumbnail conversion failed")?,
    )
    .write_to(&mut jpeg_buf, image::ImageFormat::Jpeg)
    .map_err(|e| format!("encode jpeg: {e}"))?;

    let b64 = B64.encode(jpeg_buf.into_inner());
    Ok(format!("data:image/jpeg;base64,{b64}"))
}

/// Decode the very first frame of a video file using ffmpeg and return it as
/// an `image::RgbImage`.
fn extract_video_frame(path: &str) -> Result<image::RgbImage, String> {
    // We borrow the ffmpeg plumbing that fancam_core already links.
    extern crate ffmpeg_next as ffmpeg;
    use ffmpeg::{format, media, software::scaling};

    ffmpeg::init().map_err(|e| e.to_string())?;

    let mut ictx = format::input(&path).map_err(|e| format!("open: {e}"))?;

    let stream = ictx
        .streams()
        .best(media::Type::Video)
        .ok_or("no video stream")?;
    let stream_index = stream.index();

    let codecpar = stream.parameters();
    let mut decoder = ffmpeg::codec::Context::from_parameters(codecpar)
        .map_err(|e| format!("codec ctx: {e}"))?
        .decoder()
        .video()
        .map_err(|e| format!("decoder: {e}"))?;

    validate_preview_dimensions(decoder.width(), decoder.height())?;

    let mut scaler = scaling::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        scaling::Flags::BILINEAR,
    )
    .map_err(|e| format!("scaler: {e}"))?;

    let mut decoded = ffmpeg::frame::Video::empty();
    let mut rgb_frame = ffmpeg::frame::Video::empty();

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| format!("send: {e}"))?;
        if decoder.receive_frame(&mut decoded).is_ok() {
            scaler
                .run(&decoded, &mut rgb_frame)
                .map_err(|e| format!("scale: {e}"))?;

            let w = rgb_frame.width();
            let h = rgb_frame.height();
            let stride = rgb_frame.stride(0);
            let data = rgb_frame.data(0);

            // ffmpeg rows may be padded — copy row-by-row
            let mut buf = Vec::with_capacity((w * h * 3) as usize);
            for row in 0..h as usize {
                let start = row * stride;
                buf.extend_from_slice(&data[start..start + (w as usize) * 3]);
            }

            return image::RgbImage::from_raw(w, h, buf)
                .ok_or_else(|| "frame buffer size mismatch".to_string());
        }
    }

    Err("could not decode any frame".to_string())
}

fn run_identity_scan_for_queue(
    args: IdentityScanArgs,
    app: Option<AppHandle>,
    queue_phase: &'static str,
    cancel: Arc<AtomicBool>,
) -> Result<IdentityScanResult, String> {
    let run_id = args
        .client_run_id
        .clone()
        .unwrap_or_else(|| next_run_id(queue_phase));
    run_identity_scan_with_hooks(
        args,
        move |progress| {
            if let Some(app_handle) = app.as_ref() {
                let _ = app_handle.emit(
                    "scan://progress",
                    ScanProgressPayload {
                        run_id: run_id.clone(),
                        sampled_frames: progress.sampled_frames,
                        total_decoded_frames: progress.total_decoded_frames,
                        estimated_total_samples: progress.estimated_total_samples,
                        pass_fraction: progress.pass_fraction,
                        overall_fraction: progress.overall_fraction,
                        phase: format!("{queue_phase}: {}", progress.phase),
                        pass_index: progress.pass_index,
                        pass_total: progress.pass_total,
                    },
                );
            }
        },
        || cancel.load(Ordering::Relaxed),
    )
}

fn run_identity_scan_with_hooks<F, C>(
    args: IdentityScanArgs,
    mut on_progress: F,
    mut should_cancel: C,
) -> Result<IdentityScanResult, String>
where
    F: FnMut(ScanProgressUpdate),
    C: FnMut() -> bool,
{
    OrtConfig::ensure_initialized()
        .map_err(|e| format!("failed to initialize ONNX Runtime: {e}"))?;

    let identity_model =
        effective_identity_model(&args.face_model, args.identity_model.as_deref()).to_string();

    let body_reid_model_path = args
        .body_reid_model
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| resolve_default_body_reid_model(&identity_model));

    let mut engine = DiscoveryEngine::load_with_body_reid(
        &args.yolo_model,
        &identity_model,
        body_reid_model_path.as_deref(),
    )
    .map_err(|e| format!("failed to initialize discovery engine: {e}"))?;

    let mode = processing_mode_from_option(args.processing_mode.as_deref());
    let mode_label = mode.as_str().to_string();
    let base = DiscoveryConfig::for_mode(mode);
    let mut pass_total = 1u8;
    let mut should_under_count_rescan = false;

    let estimated_initial_samples = estimate_samples_for_pass(&args.video, &base);

    let mut report = engine
        .scan_video_with_hooks(
            &args.video,
            &base,
            &mut |sampled_frames, total_decoded_frames| {
                on_progress(ScanProgressUpdate {
                    sampled_frames,
                    total_decoded_frames,
                    estimated_total_samples: estimated_initial_samples,
                    pass_fraction: fraction(sampled_frames, estimated_initial_samples),
                    overall_fraction: fraction(sampled_frames, estimated_initial_samples),
                    phase: "initial scan".to_string(),
                    pass_index: 1,
                    pass_total,
                });
            },
            &mut should_cancel,
        )
        .map_err(|e| format!("identity discovery failed: {e}"))?;
    let mut rescan_performed = false;

    let has_duplicates = !report.duplicates.is_empty();
    let low_confidence_or_weak_preview = report
        .candidates
        .iter()
        .any(|c| c.confidence < 0.6 || c.preview_score < 0.58);

    if let Some(expected) = args.expected_member_count
        && report.candidates.len() as u32 > expected
    {
        report = tighten_to_expected(report, expected as usize);
    }

    if let Some(expected) = args.expected_member_count
        && report.candidates.len() as u32 + 1 < expected
    {
        should_under_count_rescan = true;
    }

    if should_under_count_rescan
        || (has_duplicates && mode != ProcessingMode::Fast)
        || low_confidence_or_weak_preview
    {
        rescan_performed = true;
        pass_total = 2;
        let informed = base.informed_under_count_pass();
        let estimated_rescan_samples = estimate_samples_for_pass(&args.video, &informed);
        on_progress(ScanProgressUpdate {
            sampled_frames: 0,
            total_decoded_frames: 0,
            estimated_total_samples: estimated_rescan_samples,
            pass_fraction: 0.0,
            overall_fraction: 0.5,
            phase: "informed rescan".to_string(),
            pass_index: 2,
            pass_total,
        });
        report = engine
            .scan_video_with_hooks(
                &args.video,
                &informed,
                &mut |sampled_frames, total_decoded_frames| {
                    let pass_fraction = fraction(sampled_frames, estimated_rescan_samples);
                    on_progress(ScanProgressUpdate {
                        sampled_frames,
                        total_decoded_frames,
                        estimated_total_samples: estimated_rescan_samples,
                        pass_fraction,
                        overall_fraction: 0.5 + pass_fraction * 0.5,
                        phase: "informed rescan".to_string(),
                        pass_index: 2,
                        pass_total,
                    });
                },
                &mut should_cancel,
            )
            .map_err(|e| format!("informed identity rescan failed: {e}"))?;

        if let Some(expected) = args.expected_member_count
            && report.candidates.len() as u32 > expected
        {
            report = tighten_to_expected(report, expected as usize);
        }
    }

    let count_blocker = args
        .expected_member_count
        .is_some_and(|k| report.candidates.len() as u32 != k);
    let duplicate_blocker = !report.duplicates.is_empty();
    let confidence_blocker = report.candidates.iter().any(|c| c.confidence < 0.55);
    let preview_blocker = report
        .candidates
        .iter()
        .any(|c| c.preview_score < 0.58 || c.preview_observations < 2);
    let needs_review = count_blocker || duplicate_blocker || confidence_blocker || preview_blocker;

    let sampled_frames = report.sampled_frames;
    let total_decoded_frames = report.total_decoded_frames;
    let rejected_embeddings = report.rejected_embeddings;
    let suppressed_clusters = report.suppressed_clusters;
    let merged_clusters = report.merged_clusters;
    let provisional_tracklets = report.provisional_tracklets;
    let discovered = report.candidates;
    let duplicate_rows = report.duplicates;

    let candidates = discovered
        .into_iter()
        .map(|c| IdentityCandidatePayload {
            id: c.id,
            confidence: c.confidence,
            observations: c.observations,
            first_frame: c.first_frame,
            last_frame: c.last_frame,
            anchor_x: c.anchor_x,
            anchor_y: c.anchor_y,
            anchor_x_norm: Some(c.anchor_x_norm),
            anchor_y_norm: Some(c.anchor_y_norm),
            thumbnail_data_url: format!("data:image/jpeg;base64,{}", B64.encode(c.thumbnail_jpeg)),
            embedding: Some(c.embedding),
            body_embedding: c.body_embedding,
            preview_score: Some(c.preview_score),
            preview_observations: Some(c.preview_observations),
        })
        .collect::<Vec<_>>();

    let duplicates = duplicate_rows
        .into_iter()
        .map(|d| DuplicatePairPayload {
            a: d.a,
            b: d.b,
            similarity: d.similarity,
        })
        .collect::<Vec<_>>();

    let message = if needs_review {
        "Identity scan complete: review suggestions before tracking".to_string()
    } else {
        "Identity scan complete".to_string()
    };

    on_progress(ScanProgressUpdate {
        sampled_frames,
        total_decoded_frames,
        estimated_total_samples: sampled_frames.max(1),
        pass_fraction: 1.0,
        overall_fraction: 1.0,
        phase: "scan complete".to_string(),
        pass_index: pass_total,
        pass_total,
    });

    Ok(IdentityScanResult {
        scan_id: String::new(),
        ok: true,
        message,
        video: args.video,
        processing_mode: mode_label,
        sampled_frames,
        total_decoded_frames,
        proposed_count: candidates.len(),
        expected_count: args.expected_member_count,
        rescan_performed,
        needs_review,
        rejected_embeddings,
        suppressed_clusters,
        merged_clusters,
        provisional_tracklets,
        candidates,
        duplicates,
    })
}

#[derive(Debug, Clone)]
struct ScanProgressUpdate {
    sampled_frames: u64,
    total_decoded_frames: u64,
    estimated_total_samples: u64,
    pass_fraction: f64,
    overall_fraction: f64,
    phase: String,
    pass_index: u8,
    pass_total: u8,
}

fn fraction(current: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    (current as f64 / total as f64).clamp(0.0, 1.0)
}

fn processing_mode_from_option(value: Option<&str>) -> ProcessingMode {
    value
        .and_then(|value| value.parse::<ProcessingMode>().ok())
        .unwrap_or_default()
}

fn processing_mode_string(value: Option<&str>) -> String {
    processing_mode_from_option(value).as_str().to_string()
}

fn estimate_samples_for_pass(video: &str, config: &DiscoveryConfig) -> u64 {
    let frame_count = total_frames(video);
    if frame_count == 0 {
        return config.max_sampled_frames as u64;
    }
    let stride = config.sample_stride.max(1);
    let sampled = (frame_count / stride).max(1);
    sampled.min(config.max_sampled_frames as u64)
}

fn tighten_to_expected(
    mut report: fancam_core::discovery::DiscoveryReport,
    expected: usize,
) -> fancam_core::discovery::DiscoveryReport {
    if expected == 0 || report.candidates.len() <= expected {
        return report;
    }

    report.candidates.sort_unstable_by(|a, b| {
        b.preview_score
            .partial_cmp(&a.preview_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.preview_observations.cmp(&a.preview_observations))
            .then_with(|| b.observations.cmp(&a.observations))
            .then_with(|| {
                let a_span = a.last_frame.saturating_sub(a.first_frame);
                let b_span = b.last_frame.saturating_sub(b.first_frame);
                b_span.cmp(&a_span)
            })
    });

    let keep = expected.saturating_add(1);
    report.candidates.truncate(keep);
    let keep_ids = report
        .candidates
        .iter()
        .map(|candidate| candidate.id)
        .collect::<HashSet<_>>();
    report
        .duplicates
        .retain(|pair| keep_ids.contains(&pair.a) && keep_ids.contains(&pair.b));
    report
}

#[derive(Debug, Clone)]
struct ReviewComputation {
    ready: bool,
    blockers: Vec<String>,
    active_count: usize,
    expected_count: Option<u32>,
    selected_identity_id: Option<usize>,
    selected_anchor_x: Option<f32>,
    selected_anchor_y: Option<f32>,
    excluded_identity_ids: Vec<usize>,
    accepted_low_confidence_ids: Vec<usize>,
    resolved_duplicates: Vec<ReviewDuplicateResolution>,
    pending_split_ids: Vec<usize>,
}

type ReviewDuplicateNormalization = (
    Vec<ReviewDuplicateResolution>,
    HashSet<(usize, usize)>,
    HashSet<usize>,
    usize,
);

fn normalize_review_duplicates(
    resolved_duplicates: &[ReviewDuplicateResolution],
) -> ReviewDuplicateNormalization {
    let mut normalized = Vec::with_capacity(resolved_duplicates.len());
    let mut resolved_pairs = HashSet::<(usize, usize)>::new();
    let mut forced_excluded = HashSet::<usize>::new();
    let mut invalid_resolution = 0usize;

    for pair in resolved_duplicates {
        let a = pair.a.min(pair.b);
        let b = pair.a.max(pair.b);
        let keep = pair.keep;
        if keep != a && keep != b {
            invalid_resolution += 1;
            continue;
        }
        if !resolved_pairs.insert((a, b)) {
            continue;
        }
        let drop = if keep == a { b } else { a };
        forced_excluded.insert(drop);
        normalized.push(ReviewDuplicateResolution { a, b, keep });
    }

    (
        normalized,
        resolved_pairs,
        forced_excluded,
        invalid_resolution,
    )
}

fn compute_review(
    scan: &IdentityScanCache,
    args: &ValidateIdentityReviewArgs,
) -> ReviewComputation {
    #[derive(Clone, Copy)]
    struct ActiveCandidate {
        id: usize,
        confidence: f32,
        preview_score: f32,
        preview_observations: u32,
        anchor_x: f32,
        anchor_y: f32,
    }

    let excluded: HashSet<usize> = args.excluded_identity_ids.iter().copied().collect();
    let accepted_low_confidence: HashSet<usize> =
        args.accepted_low_confidence_ids.iter().copied().collect();
    let (resolved_duplicates, resolved_pairs, forced_excluded, invalid_resolution) =
        normalize_review_duplicates(&args.resolved_duplicates);

    let mut effective_excluded = excluded;
    effective_excluded.extend(forced_excluded.iter().copied());

    let active_candidates = scan
        .candidates
        .iter()
        .filter(|candidate| !effective_excluded.contains(&candidate.id))
        .map(|candidate| ActiveCandidate {
            id: candidate.id,
            confidence: candidate.confidence,
            preview_score: candidate.preview_score.unwrap_or_default(),
            preview_observations: candidate.preview_observations.unwrap_or_default(),
            anchor_x: candidate.anchor_x,
            anchor_y: candidate.anchor_y,
        })
        .collect::<Vec<_>>();
    let active_count = active_candidates.len();

    let mut blockers = Vec::new();
    if invalid_resolution > 0 {
        blockers.push(format!(
            "invalid duplicate resolution entries: {invalid_resolution}"
        ));
    }

    let expected_count = args.expected_member_count.or(scan.expected_count);
    if let Some(expected_count) = expected_count
        && active_count as u32 != expected_count
    {
        blockers.push(format!(
            "member count mismatch: expected {expected_count}, active {active_count}"
        ));
    }

    let unresolved_duplicates = scan
        .duplicates
        .iter()
        .filter(|duplicate| {
            !effective_excluded.contains(&duplicate.a) && !effective_excluded.contains(&duplicate.b)
        })
        .filter(|duplicate| {
            let key = (duplicate.a.min(duplicate.b), duplicate.a.max(duplicate.b));
            !resolved_pairs.contains(&key)
        })
        .count();
    if unresolved_duplicates > 0 {
        blockers.push(format!(
            "unresolved duplicate pairs: {unresolved_duplicates}"
        ));
    }

    let unresolved_low_confidence = active_candidates
        .iter()
        .filter(|candidate| {
            candidate.confidence < 0.55 && !accepted_low_confidence.contains(&candidate.id)
        })
        .count();
    if unresolved_low_confidence > 0 {
        blockers.push(format!(
            "unconfirmed low-confidence identities: {unresolved_low_confidence}"
        ));
    }

    let unresolved_weak_preview = active_candidates
        .iter()
        .filter(|candidate| {
            (candidate.preview_score < 0.58 || candidate.preview_observations < 2)
                && !accepted_low_confidence.contains(&candidate.id)
        })
        .count();
    if unresolved_weak_preview > 0 {
        blockers.push(format!(
            "unconfirmed weak-preview identities: {unresolved_weak_preview}"
        ));
    }

    let pending_split_ids = args.pending_split_ids.clone();
    let pending_split: HashSet<usize> = pending_split_ids.iter().copied().collect();
    let unresolved_split_count = active_candidates
        .iter()
        .filter(|candidate| pending_split.contains(&candidate.id))
        .count();
    if unresolved_split_count > 0 {
        blockers.push(format!(
            "pending split review identities: {unresolved_split_count}"
        ));
    }

    let selected = args.selected_identity_id.and_then(|id| {
        active_candidates
            .iter()
            .find(|candidate| candidate.id == id)
            .copied()
    });
    if selected.is_none() {
        blockers.push("no valid selected target identity".to_string());
    }

    let selected_identity_id = selected.map(|candidate| candidate.id);
    let selected_anchor_x = selected.map(|candidate| candidate.anchor_x);
    let selected_anchor_y = selected.map(|candidate| candidate.anchor_y);
    let mut excluded_identity_ids = effective_excluded.into_iter().collect::<Vec<_>>();
    excluded_identity_ids.sort_unstable();

    ReviewComputation {
        ready: blockers.is_empty(),
        blockers,
        active_count,
        expected_count,
        selected_identity_id,
        selected_anchor_x,
        selected_anchor_y,
        excluded_identity_ids,
        accepted_low_confidence_ids: args.accepted_low_confidence_ids.clone(),
        resolved_duplicates,
        pending_split_ids,
    }
}

fn apply_review_to_scan(
    scan: &mut IdentityScanCache,
    review: &ReviewComputation,
    threshold: f32,
) -> Result<(), String> {
    scan.expected_count = review.expected_count;
    scan.review_ready = review.ready;
    scan.selected_identity_id = review.selected_identity_id;
    scan.selected_anchor_x = review.selected_anchor_x;
    scan.selected_anchor_y = review.selected_anchor_y;
    scan.validated_threshold = Some(threshold.clamp(0.0, 1.0));
    scan.last_blockers = review.blockers.clone();
    scan.updated_at_ms = epoch_ms();
    scan.excluded_identity_ids = review.excluded_identity_ids.clone();
    scan.accepted_low_confidence_ids = review.accepted_low_confidence_ids.clone();
    scan.resolved_duplicates = review.resolved_duplicates.clone();
    scan.pending_split_ids = review.pending_split_ids.clone();
    set_scan_status(
        scan,
        if review.ready {
            ScanSessionStatus::Validated
        } else {
            ScanSessionStatus::Proposed
        },
    )
}

fn build_runtime_embedding_galleries(
    scan: &IdentityScanCache,
    selected_id: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut selected_aliases = HashSet::from([selected_id]);
    for resolution in &scan.resolved_duplicates {
        if resolution.keep == selected_id {
            let alias = if resolution.a == selected_id {
                resolution.b
            } else {
                resolution.a
            };
            selected_aliases.insert(alias);
        }
    }

    let mut target_embeddings = Vec::new();
    let mut negative_embeddings = Vec::new();
    for candidate in &scan.candidates {
        let Some(embedding) = candidate.embedding.clone() else {
            continue;
        };
        if embedding.is_empty() {
            continue;
        }
        if selected_aliases.contains(&candidate.id) {
            target_embeddings.push(embedding);
        } else {
            negative_embeddings.push(embedding);
        }
    }
    (target_embeddings, negative_embeddings)
}

fn build_runtime_body_reid_gallery(scan: &IdentityScanCache, selected_id: usize) -> Vec<Vec<f32>> {
    let mut selected_aliases = HashSet::from([selected_id]);
    for resolution in &scan.resolved_duplicates {
        if resolution.keep == selected_id {
            let alias = if resolution.a == selected_id {
                resolution.b
            } else {
                resolution.a
            };
            selected_aliases.insert(alias);
        }
    }

    let mut gallery = scan
        .candidates
        .iter()
        .filter(|candidate| selected_aliases.contains(&candidate.id))
        .filter_map(|candidate| candidate.body_embedding.as_ref())
        .filter(|embedding| !embedding.is_empty())
        .map(|embedding| l2_normalize(embedding))
        .filter(|embedding| !embedding.is_empty())
        .collect::<Vec<_>>();

    if !gallery.is_empty() {
        gallery.sort_by_key(|embedding| std::cmp::Reverse(embedding.len()));
        gallery.truncate(6);
    }

    gallery
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    if v.is_empty() || v.iter().any(|value| !value.is_finite()) {
        return Vec::new();
    }
    let norm_squared = v.iter().map(|x| x * x).sum::<f32>();
    if !norm_squared.is_finite() || norm_squared <= f32::EPSILON {
        return Vec::new();
    }
    let norm = norm_squared.sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn resolve_default_body_reid_model(face_model_path: &str) -> Option<String> {
    let face_path = PathBuf::from(face_model_path);
    let models_dir = face_path.parent()?;
    let preferred = models_dir.join("osnet_x0_25_msmt17.onnx");
    if preferred.is_file() {
        return Some(preferred.to_string_lossy().into_owned());
    }
    None
}

fn effective_identity_model<'a>(face_model: &'a str, identity_model: Option<&'a str>) -> &'a str {
    identity_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(face_model)
}

#[tauri::command]
pub fn validate_identity_review(
    state: State<'_, IdentityScanStore>,
    args: ValidateIdentityReviewArgs,
) -> Result<IdentityReviewResult, String> {
    if !args.threshold.is_finite() {
        return Err("identity threshold must be finite".to_string());
    }
    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    ensure_scan_store_loaded(&mut lock);
    let Some(scan) = lock.scans.get(&args.scan_id) else {
        return Err("identity scan session not found; rerun scan".to_string());
    };
    let review = compute_review(scan, &args);

    Ok(IdentityReviewResult {
        ok: true,
        ready: review.ready,
        blockers: review.blockers,
        active_count: review.active_count,
        selected_identity_id: review.selected_identity_id,
        selected_anchor_x: review.selected_anchor_x,
        selected_anchor_y: review.selected_anchor_y,
    })
}

#[tauri::command]
pub fn list_identity_scans(
    state: State<'_, IdentityScanStore>,
) -> Result<Vec<ScanSessionSummary>, String> {
    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    ensure_scan_store_loaded(&mut lock);
    let mut rows = lock
        .scans
        .iter()
        .map(|(scan_id, scan)| ScanSessionSummary {
            scan_id: scan_id.clone(),
            video: scan.video.clone(),
            status: scan.status.clone(),
            review_ready: scan.review_ready,
            selected_identity_id: scan.selected_identity_id,
            pending_split_count: scan.pending_split_ids.len(),
            event_count: scan.events.len() as u64,
            updated_at_ms: scan.updated_at_ms,
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| std::cmp::Reverse(row.updated_at_ms));
    Ok(rows)
}

#[tauri::command]
pub fn get_identity_scan(
    state: State<'_, IdentityScanStore>,
    scan_id: String,
) -> Result<ScanSessionDetail, String> {
    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    ensure_scan_store_loaded(&mut lock);
    let Some(scan) = lock.scans.get(&scan_id) else {
        return Err("scan session not found".to_string());
    };
    let recent_events = scan
        .events
        .iter()
        .rev()
        .take(25)
        .cloned()
        .collect::<Vec<_>>();
    Ok(ScanSessionDetail {
        scan_id,
        video: scan.video.clone(),
        yolo_model: scan.yolo_model.clone(),
        identity_model: scan
            .identity_model
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&scan.face_model)
            .to_string(),
        status: scan.status.clone(),
        expected_count: scan.expected_count,
        processing_mode: scan.processing_mode.clone(),
        review_ready: scan.review_ready,
        selected_identity_id: scan.selected_identity_id,
        selected_anchor_x: scan.selected_anchor_x,
        selected_anchor_y: scan.selected_anchor_y,
        validated_threshold: scan.validated_threshold,
        last_blockers: scan.last_blockers.clone(),
        candidates: strip_candidate_embeddings(&scan.candidates),
        duplicates: scan.duplicates.clone(),
        excluded_identity_ids: scan.excluded_identity_ids.clone(),
        accepted_low_confidence_ids: scan.accepted_low_confidence_ids.clone(),
        resolved_duplicates: scan.resolved_duplicates.clone(),
        pending_split_ids: scan.pending_split_ids.clone(),
        updated_at_ms: scan.updated_at_ms,
        event_count: scan.events.len(),
        recent_events,
    })
}

#[tauri::command]
pub fn cleanup_identity_scans(
    state: State<'_, IdentityScanStore>,
    max_age_ms: Option<u64>,
) -> Result<usize, String> {
    let ttl = max_age_ms.unwrap_or(86_400_000).max(60_000);
    let cutoff = epoch_ms().saturating_sub(ttl);
    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    ensure_scan_store_loaded(&mut lock);
    let removed_ids = lock
        .scans
        .iter()
        .filter_map(|(scan_id, scan)| (scan.updated_at_ms < cutoff).then_some(scan_id.clone()))
        .collect::<Vec<_>>();
    for scan_id in &removed_ids {
        lock.scans.remove(scan_id);
    }
    let removed = removed_ids.len();
    if removed > 0 {
        delete_scan_entries(&removed_ids)?;
    }
    Ok(removed)
}

#[tauri::command]
pub fn query_identity_scans(
    args: Option<QueryIdentityScansArgs>,
) -> Result<QueryIdentityScansResult, String> {
    let has_cursor = args
        .as_ref()
        .is_some_and(|a| a.cursor_updated_at_ms.is_some() && a.cursor_scan_id.is_some());
    let limit = args
        .as_ref()
        .and_then(|a| a.limit)
        .unwrap_or(25)
        .clamp(1, 200);
    let offset = if has_cursor {
        0
    } else {
        args.as_ref().and_then(|a| a.offset).unwrap_or(0)
    };
    let offset_ignored = has_cursor && args.as_ref().and_then(|a| a.offset).unwrap_or(0) > 0;
    let status = args
        .as_ref()
        .and_then(|a| a.status.as_ref().map(|s| s.trim().to_ascii_lowercase()))
        .filter(|s| !s.is_empty());
    let cursor_updated_at_ms = args.as_ref().and_then(|a| a.cursor_updated_at_ms);
    let cursor_scan_id = args.as_ref().and_then(|a| a.cursor_scan_id.as_deref());
    let rows = storage::query_scan_summaries(
        &storage::scan_store_db_path(),
        limit,
        offset,
        status.as_deref(),
        cursor_updated_at_ms,
        cursor_scan_id,
    )?
    .into_iter()
    .map(|row| ScanSessionSummary {
        scan_id: row.scan_id,
        video: row.video,
        status: status_from_db(&row.status),
        review_ready: row.review_ready,
        selected_identity_id: row.selected_identity_id.map(|v| v as usize),
        pending_split_count: row.pending_split_count as usize,
        event_count: row.event_count,
        updated_at_ms: row.updated_at_ms,
    })
    .collect::<Vec<_>>();

    let next_cursor_updated_at_ms = rows.last().map(|r| r.updated_at_ms);
    let next_cursor_scan_id = rows.last().map(|r| r.scan_id.clone());

    Ok(QueryIdentityScansResult {
        rows,
        next_cursor_updated_at_ms,
        next_cursor_scan_id,
        offset_ignored,
    })
}

#[tauri::command]
pub fn query_scan_events(args: QueryScanEventsArgs) -> Result<QueryScanEventsResult, String> {
    let limit = args.limit.unwrap_or(40).clamp(1, 200);
    let offset = if args.cursor_event_id.is_some() {
        0
    } else {
        args.offset.unwrap_or(0)
    };
    let offset_ignored = args.cursor_event_id.is_some() && args.offset.unwrap_or(0) > 0;
    let action_contains = args
        .action_contains
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());
    let rows = storage::query_scan_events(
        &storage::scan_store_db_path(),
        storage::ScanEventQuery {
            scan_id: &args.scan_id,
            limit,
            offset,
            action_contains,
            since_ms: args.since_ms,
            until_ms: args.until_ms,
            cursor_event_id: args.cursor_event_id,
        },
    )?
    .into_iter()
    .collect::<Vec<_>>();
    let next_cursor_event_id = rows.last().map(|r| r.event_id);
    let mapped = rows
        .into_iter()
        .map(|row| ScanSessionEvent {
            at_ms: row.at_ms,
            action: row.action,
            details: row.details,
        })
        .collect();
    Ok(QueryScanEventsResult {
        rows: mapped,
        next_cursor_event_id,
        offset_ignored,
    })
}

#[tauri::command]
pub fn scan_storage_stats() -> Result<ScanStorageStats, String> {
    let stats = storage::get_storage_stats(&storage::scan_store_db_path())?;
    Ok(ScanStorageStats {
        schema_version: stats.schema_version,
        session_count: stats.session_count,
        event_count: stats.event_count,
        db_path: storage::scan_store_db_path().to_string_lossy().into_owned(),
    })
}

#[tauri::command]
pub fn run_scan_storage_maintenance(
    args: Option<ScanStorageMaintenanceArgs>,
) -> Result<ScanStorageMaintenanceResult, String> {
    let max_session_age_ms = args
        .as_ref()
        .and_then(|a| a.max_session_age_ms)
        .unwrap_or(7 * 86_400_000)
        .max(60_000);
    let max_events_per_scan = args
        .as_ref()
        .and_then(|a| a.max_events_per_scan)
        .unwrap_or(120)
        .max(10);
    let vacuum = args.as_ref().and_then(|a| a.vacuum).unwrap_or(false);

    let maintenance = storage::run_storage_maintenance(
        &storage::scan_store_db_path(),
        max_session_age_ms,
        max_events_per_scan,
        vacuum,
    )?;
    let stats = scan_storage_stats()?;

    Ok(ScanStorageMaintenanceResult {
        deleted_sessions: maintenance.deleted_sessions,
        deleted_events: maintenance.deleted_events,
        vacuum_ran: maintenance.vacuum_ran,
        stats,
    })
}

#[tauri::command]
pub fn export_diagnostics_bundle(
    queue_state: State<'_, QueueStore>,
    queue_worker_state: State<'_, QueueWorkerStore>,
    args: Option<ExportDiagnosticsArgs>,
) -> Result<ExportDiagnosticsResult, String> {
    let scan_id = args.and_then(|a| a.scan_id);
    let stats = scan_storage_stats()?;
    let sessions = query_identity_scans(Some(QueryIdentityScansArgs {
        limit: Some(50),
        offset: Some(0),
        status: None,
        cursor_updated_at_ms: None,
        cursor_scan_id: None,
    }))?
    .rows;

    let events = if let Some(id) = scan_id.clone() {
        query_scan_events(QueryScanEventsArgs {
            scan_id: id,
            limit: Some(80),
            offset: Some(0),
            action_contains: None,
            since_ms: None,
            until_ms: None,
            cursor_event_id: None,
        })?
        .rows
    } else {
        Vec::new()
    };

    let queue_health = {
        let q = queue_state.0.lock().map_err(|e| e.to_string())?;
        q.health()
    };
    let queue_worker = {
        let w = queue_worker_state.0.lock().map_err(|e| e.to_string())?;
        QueueWorkerStatus {
            running: w.running,
            stop_requested: w.stop_requested,
            poll_interval_ms: w.poll_interval_ms,
            max_attempts_before_dlq: w.max_attempts_before_dlq,
            processed_total: w.processed_total,
            last_error: w.last_error.clone(),
            recent_events: w
                .recent_events
                .iter()
                .rev()
                .take(40)
                .map(worker_event_payload)
                .collect(),
        }
    };

    #[derive(Serialize)]
    struct Bundle {
        created_at_ms: u64,
        scan_id: Option<String>,
        storage: ScanStorageStats,
        sessions: Vec<ScanSessionSummary>,
        events: Vec<ScanSessionEvent>,
        queue_health: queue::QueueHealth,
        queue_worker: QueueWorkerStatus,
    }

    let bundle = Bundle {
        created_at_ms: epoch_ms(),
        scan_id,
        storage: stats,
        sessions,
        events,
        queue_health,
        queue_worker,
    };
    let json = serde_json::to_vec_pretty(&bundle)
        .map_err(|e| format!("failed to serialize diagnostics bundle: {e}"))?;

    let mut out_path = diagnostics_dir_path();
    fs::create_dir_all(&out_path).map_err(|e| format!("failed to create diagnostics dir: {e}"))?;
    out_path.push(format!(
        "bundle-{}-{}.json",
        epoch_ms(),
        RUN_ID_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary_path = out_path.with_extension("json.partial");
    fs::write(&temporary_path, &json)
        .map_err(|e| format!("failed to write diagnostics bundle: {e}"))?;
    if let Err(error) = fs::rename(&temporary_path, &out_path) {
        let _ = fs::remove_file(&temporary_path);
        return Err(format!("failed to commit diagnostics bundle: {error}"));
    }
    let out_path_str = out_path.to_string_lossy().into_owned();

    Ok(ExportDiagnosticsResult {
        path: out_path_str,
        bytes: json.len(),
    })
}

#[tauri::command]
pub fn list_diagnostics_bundles(
    args: Option<ListDiagnosticsBundlesArgs>,
) -> Result<ListDiagnosticsBundlesResult, String> {
    let limit = args.and_then(|a| a.limit).unwrap_or(30).clamp(1, 500);
    let dir = diagnostics_dir_path();
    if !dir.exists() {
        return Ok(ListDiagnosticsBundlesResult {
            bundles: Vec::new(),
        });
    }

    let mut bundles = Vec::new();
    let entries = fs::read_dir(&dir).map_err(|e| format!("failed to read diagnostics dir: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read diagnostics entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
        else {
            continue;
        };
        if file_name == "manifest.json" {
            continue;
        }
        let path_str = path.to_string_lossy().into_owned();
        let meta = entry
            .metadata()
            .map_err(|e| format!("failed to read diagnostics metadata: {e}"))?;
        bundles.push(DiagnosticsBundleInfo {
            file_name,
            path: path_str.clone(),
            bytes: meta.len(),
            modified_at_ms: file_modified_ms(&meta),
        });
    }

    bundles.sort_by(|a, b| {
        b.modified_at_ms
            .cmp(&a.modified_at_ms)
            .then_with(|| b.file_name.cmp(&a.file_name))
    });
    bundles.truncate(limit);

    Ok(ListDiagnosticsBundlesResult { bundles })
}

#[tauri::command]
pub fn storage_worker_status(
    state: State<'_, StorageWorkerStore>,
) -> Result<StorageWorkerStatus, String> {
    let s = state.0.lock().map_err(|e| e.to_string())?;
    Ok(StorageWorkerStatus {
        running: s.running,
        stop_requested: s.stop_requested,
        poll_interval_ms: s.poll_interval_ms,
        max_session_age_ms: s.max_session_age_ms,
        max_events_per_scan: s.max_events_per_scan,
        vacuum: s.vacuum,
        runs_total: s.runs_total,
        last_run_ms: s.last_run_ms,
        last_error: s.last_error.clone(),
    })
}

#[tauri::command]
pub fn storage_worker_stop(
    state: State<'_, StorageWorkerStore>,
) -> Result<StorageWorkerStatus, String> {
    {
        let mut s = state.0.lock().map_err(|e| e.to_string())?;
        s.stop_requested = true;
    }
    state.1.notify_one();
    storage_worker_status(state)
}

#[tauri::command]
pub fn storage_worker_start(
    state: State<'_, StorageWorkerStore>,
    args: Option<StorageWorkerStartArgs>,
) -> Result<StorageWorkerStatus, String> {
    let poll_interval_ms = args
        .as_ref()
        .and_then(|a| a.poll_interval_ms)
        .unwrap_or(300_000)
        .max(10_000);
    let max_session_age_ms = args
        .as_ref()
        .and_then(|a| a.max_session_age_ms)
        .unwrap_or(7 * 86_400_000)
        .max(60_000);
    let max_events_per_scan = args
        .as_ref()
        .and_then(|a| a.max_events_per_scan)
        .unwrap_or(120)
        .max(10);
    let vacuum = args.as_ref().and_then(|a| a.vacuum).unwrap_or(false);

    let already_running = {
        let mut s = state.0.lock().map_err(|e| e.to_string())?;
        if s.running {
            true
        } else {
            s.running = true;
            s.stop_requested = false;
            s.poll_interval_ms = poll_interval_ms;
            s.max_session_age_ms = max_session_age_ms;
            s.max_events_per_scan = max_events_per_scan;
            s.vacuum = vacuum;
            s.last_error = None;
            false
        }
    };

    if already_running {
        return storage_worker_status(state);
    }

    let worker_arc = state.0.clone();
    let stop_notify = state.1.clone();
    tokio::spawn(async move {
        loop {
            let (should_stop, poll_ms, age_ms, max_events, vacuum_flag) = match worker_arc.lock() {
                Ok(s) => (
                    s.stop_requested,
                    s.poll_interval_ms,
                    s.max_session_age_ms,
                    s.max_events_per_scan,
                    s.vacuum,
                ),
                Err(_) => (true, 60_000, 7 * 86_400_000, 120, false),
            };
            if should_stop {
                break;
            }

            let run = storage::run_storage_maintenance(
                &storage::scan_store_db_path(),
                age_ms,
                max_events,
                vacuum_flag,
            );
            if let Ok(mut s) = worker_arc.lock() {
                match run {
                    Ok(_) => {
                        s.runs_total = s.runs_total.saturating_add(1);
                        s.last_run_ms = Some(epoch_ms());
                        s.last_error = None;
                    }
                    Err(err) => {
                        s.runs_total = s.runs_total.saturating_add(1);
                        s.last_run_ms = Some(epoch_ms());
                        s.last_error = Some(err);
                    }
                }
            }
            tokio::select! {
                () = tokio::time::sleep(std::time::Duration::from_millis(poll_ms)) => {},
                () = stop_notify.notified() => break,
            }
        }

        if let Ok(mut s) = worker_arc.lock() {
            s.running = false;
            s.stop_requested = false;
        }
    });

    storage_worker_status(state)
}

#[tauri::command]
pub fn queue_health(state: State<'_, QueueStore>) -> Result<queue::QueueHealth, String> {
    let lock = state.0.lock().map_err(|e| e.to_string())?;
    Ok(lock.health())
}

#[tauri::command]
pub fn queue_dlq_list(
    state: State<'_, QueueStore>,
    args: Option<QueueDlqListArgs>,
) -> Result<Vec<queue::QueueDlqItemSummary>, String> {
    let limit = args
        .and_then(|value| value.limit)
        .unwrap_or(20)
        .clamp(1, 100);
    let lock = state.0.lock().map_err(|e| e.to_string())?;
    lock.list_dlq(limit)
}

#[tauri::command]
pub fn queue_dlq_replay(
    state: State<'_, QueueStore>,
    args: QueueDlqReplayArgs,
) -> Result<queue::QueueReplayResult, String> {
    if args.message_id.trim().is_empty() {
        return Err("message_id is required".to_string());
    }
    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    lock.replay_dlq(args.message_id.trim())
}

#[tauri::command]
pub fn enqueue_discovery_job(
    state: State<'_, QueueStore>,
    args: EnqueueDiscoveryJobArgs,
) -> Result<queue::QueueEnqueueResult, String> {
    let idempotency_key = args.idempotency_key.unwrap_or_else(|| {
        format!(
            "discovery:{}:{}:{}",
            args.scan_id,
            args.video,
            args.expected_member_count.unwrap_or_default()
        )
    });
    let payload = queue::DiscoveryJobPayload {
        scan_id: args.scan_id,
        video: args.video,
        yolo_model: args.yolo_model,
        identity_model: effective_identity_model(&args.face_model, args.identity_model.as_deref())
            .to_string(),
        expected_member_count: args.expected_member_count,
        processing_mode: sanitize_processing_mode(args.processing_mode.as_deref()),
    };

    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    lock.enqueue_discovery(payload, idempotency_key)
}

#[tauri::command]
pub fn enqueue_split_rescan_job(
    queue_state: State<'_, QueueStore>,
    scan_state: State<'_, IdentityScanStore>,
    args: EnqueueSplitRescanArgs,
) -> Result<queue::QueueEnqueueResult, String> {
    let scan_id = args.scan_id.clone();
    let scan_snapshot = {
        let mut scans = scan_state.0.lock().map_err(|e| e.to_string())?;
        ensure_scan_store_loaded(&mut scans);
        let Some(scan) = scans.scans.get(&scan_id) else {
            return Err("scan session not found".to_string());
        };
        let identity_model = scan
            .identity_model
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&scan.face_model)
            .to_string();
        (
            scan.video.clone(),
            scan.yolo_model.clone(),
            identity_model,
            scan.pending_split_ids.clone(),
            scan.selected_identity_id,
            Some(scan.processing_mode.clone()),
        )
    };

    if scan_snapshot.3.is_empty() {
        return Err("no pending split identities to rescan".to_string());
    }

    let idempotency_key = args.idempotency_key.unwrap_or_else(|| {
        format!(
            "rescan:{}:{}:{}",
            scan_id,
            scan_snapshot
                .4
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            scan_snapshot
                .3
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("-")
        )
    });

    let payload = queue::RescanJobPayload {
        scan_id: scan_id.clone(),
        video: scan_snapshot.0,
        yolo_model: scan_snapshot.1,
        identity_model: scan_snapshot.2,
        split_identity_ids: scan_snapshot.3,
        // A persisted scan owns its processing mode; do not let a stale UI
        // payload change the mode of an existing session.
        processing_mode: scan_snapshot.5,
    };

    let enqueue_result = {
        let mut q = queue_state.0.lock().map_err(|e| e.to_string())?;
        q.enqueue_rescan(payload, idempotency_key)?
    };

    if !enqueue_result.accepted {
        return Ok(enqueue_result);
    }

    if let Ok(mut scans) = scan_state.0.lock() {
        ensure_scan_store_loaded(&mut scans);
        if let Some(scan) = scans.scans.get_mut(&scan_id) {
            set_scan_status(scan, ScanSessionStatus::Proposed)?;
            scan.updated_at_ms = epoch_ms();
            append_scan_event(
                scan,
                "split_rescan_enqueued",
                format!(
                    "pending_splits={} queue={} deduplicated={}",
                    scan.pending_split_ids.len(),
                    enqueue_result.queue,
                    enqueue_result.deduplicated
                ),
            );
        }
        if let Ok(Some(snapshot)) = snapshot_scan_entry(&scans, &scan_id) {
            let _ = persist_scan_entry_snapshot(&snapshot);
        }
    }

    Ok(enqueue_result)
}

#[tauri::command]
pub async fn process_next_discovery_job(
    app: AppHandle,
    queue_state: State<'_, QueueStore>,
    scan_state: State<'_, IdentityScanStore>,
    scan_job_state: State<'_, ScanJobStore>,
    args: Option<ProcessNextDiscoveryJobArgs>,
) -> Result<queue::QueueProcessResult, String> {
    let args = args.unwrap_or(ProcessNextDiscoveryJobArgs {
        max_attempts_before_dlq: None,
        client_run_id: None,
    });
    let max_attempts_before_dlq = args.max_attempts_before_dlq.unwrap_or(3).max(1);

    process_next_discovery_job_core(
        Some(app),
        queue_state.0.clone(),
        scan_state.0.clone(),
        scan_job_state.0.clone(),
        max_attempts_before_dlq,
        args.client_run_id,
    )
    .await
}

async fn process_next_discovery_job_core(
    app: Option<AppHandle>,
    queue_store: std::sync::Arc<std::sync::Mutex<queue::QueueRuntime>>,
    scan_store: std::sync::Arc<std::sync::Mutex<IdentityScanState>>,
    scan_job_store: std::sync::Arc<std::sync::Mutex<ScanJobState>>,
    max_attempts_before_dlq: u32,
    client_run_id: Option<String>,
) -> Result<queue::QueueProcessResult, String> {
    let max_attempts_before_dlq = max_attempts_before_dlq.max(1);

    let dequeued = {
        let mut queue = queue_store.lock().map_err(|e| e.to_string())?;

        match queue.dequeue_discovery() {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                return Ok(queue::QueueProcessResult {
                    processed: false,
                    cancelled: false,
                    queue: "discovery".to_string(),
                    message_id: None,
                    job_id: None,
                    moved_to_dlq: false,
                    requeued: false,
                    attempt: None,
                    error: None,
                    remaining_depth: queue.health().depths.discovery,
                });
            }
            Err(err) => {
                return Ok(queue::QueueProcessResult {
                    processed: true,
                    cancelled: false,
                    queue: "discovery".to_string(),
                    message_id: None,
                    job_id: None,
                    moved_to_dlq: true,
                    requeued: false,
                    attempt: None,
                    error: Some(err),
                    remaining_depth: queue.health().depths.discovery,
                });
            }
        }
    };

    let mut envelope = dequeued.envelope;
    let payload = envelope.payload.clone();
    let yolo_model = payload.yolo_model.clone();
    let identity_model = payload.identity_model.clone();
    let processing_mode = {
        let mut scans = scan_store.lock().map_err(|e| e.to_string())?;
        ensure_scan_store_loaded(&mut scans);
        scans
            .scans
            .get(&payload.scan_id)
            .map(|scan| scan.processing_mode.clone())
            .or_else(|| sanitize_processing_mode(payload.processing_mode.as_deref()))
    };
    let app_for_scan = app.clone();
    let queue_run_id = client_run_id.clone();
    let run_result = task::spawn_blocking(move || {
        let scan_job = ScanJobStore(scan_job_store);
        let (_guard, cancel) = ScanJobGuard::acquire(&scan_job)?;
        let result = run_identity_scan_for_queue(
            IdentityScanArgs {
                video: payload.video,
                yolo_model: payload.yolo_model,
                face_model: identity_model.clone(),
                identity_model: Some(identity_model),
                body_reid_model: None,
                expected_member_count: payload.expected_member_count,
                processing_mode,
                client_run_id: queue_run_id,
            },
            app_for_scan,
            "queued discovery",
            Arc::clone(&cancel),
        );
        Ok::<_, String>((result, cancel.load(Ordering::Relaxed)))
    })
    .await
    .map_err(|e| e.to_string())??;

    let (run_result, cancelled) = run_result;

    match run_result {
        Ok(scan_result) => {
            let snapshot = {
                let mut scans = scan_store.lock().map_err(|e| e.to_string())?;
                ensure_scan_store_loaded(&mut scans);
                upsert_scan_cache(
                    &mut scans.scans,
                    &payload.scan_id,
                    &scan_result,
                    &yolo_model,
                    &payload.identity_model,
                );
                snapshot_scan_entry(&scans, &payload.scan_id)?
            };
            if let Some(snapshot) = snapshot.as_ref() {
                persist_scan_entry_snapshot(snapshot)?;
            }

            let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
            queue.acknowledge(&envelope.message_id)?;
            Ok(queue::QueueProcessResult {
                processed: true,
                cancelled: false,
                queue: "discovery".to_string(),
                message_id: Some(envelope.message_id),
                job_id: Some(envelope.job_id),
                moved_to_dlq: false,
                requeued: false,
                attempt: Some(envelope.attempt),
                error: None,
                remaining_depth: queue.health().depths.discovery,
            })
        }
        Err(err) => {
            let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
            if cancelled {
                let remaining_depth = queue.requeue_discovery_raw_result(dequeued.raw)?;
                return Ok(queue::QueueProcessResult {
                    processed: false,
                    cancelled: true,
                    queue: "discovery".to_string(),
                    message_id: Some(envelope.message_id),
                    job_id: Some(envelope.job_id),
                    moved_to_dlq: false,
                    requeued: true,
                    attempt: Some(envelope.attempt),
                    error: None,
                    remaining_depth,
                });
            }
            let mut moved_to_dlq = false;
            let mut requeued = false;
            if envelope.attempt + 1 >= max_attempts_before_dlq {
                queue.move_discovery_to_dlq_with_error(dequeued.raw, Some(&err))?;
                moved_to_dlq = true;
            } else {
                queue.requeue_discovery_retry_with_error(&mut envelope, Some(&err))?;
                requeued = true;
            }
            Ok(queue::QueueProcessResult {
                processed: true,
                cancelled: false,
                queue: "discovery".to_string(),
                message_id: Some(envelope.message_id),
                job_id: Some(envelope.job_id),
                moved_to_dlq,
                requeued,
                attempt: Some(envelope.attempt),
                error: Some(err),
                remaining_depth: queue.health().depths.discovery,
            })
        }
    }
}

#[tauri::command]
pub async fn process_next_rescan_job(
    app: AppHandle,
    queue_state: State<'_, QueueStore>,
    scan_state: State<'_, IdentityScanStore>,
    scan_job_state: State<'_, ScanJobStore>,
    args: Option<ProcessNextDiscoveryJobArgs>,
) -> Result<queue::QueueProcessResult, String> {
    let args = args.unwrap_or(ProcessNextDiscoveryJobArgs {
        max_attempts_before_dlq: None,
        client_run_id: None,
    });
    let max_attempts_before_dlq = args.max_attempts_before_dlq.unwrap_or(3).max(1);
    process_next_rescan_job_core(
        Some(app),
        queue_state.0.clone(),
        scan_state.0.clone(),
        scan_job_state.0.clone(),
        max_attempts_before_dlq,
        args.client_run_id,
    )
    .await
}

async fn process_next_rescan_job_core(
    app: Option<AppHandle>,
    queue_store: std::sync::Arc<std::sync::Mutex<queue::QueueRuntime>>,
    scan_store: std::sync::Arc<std::sync::Mutex<IdentityScanState>>,
    scan_job_store: std::sync::Arc<std::sync::Mutex<ScanJobState>>,
    max_attempts_before_dlq: u32,
    client_run_id: Option<String>,
) -> Result<queue::QueueProcessResult, String> {
    let max_attempts_before_dlq = max_attempts_before_dlq.max(1);
    let dequeued = {
        let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
        match queue.dequeue_rescan() {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                return Ok(queue::QueueProcessResult {
                    processed: false,
                    cancelled: false,
                    queue: "rescan".to_string(),
                    message_id: None,
                    job_id: None,
                    moved_to_dlq: false,
                    requeued: false,
                    attempt: None,
                    error: None,
                    remaining_depth: queue.health().depths.rescan,
                });
            }
            Err(err) => {
                return Ok(queue::QueueProcessResult {
                    processed: true,
                    cancelled: false,
                    queue: "rescan".to_string(),
                    message_id: None,
                    job_id: None,
                    moved_to_dlq: true,
                    requeued: false,
                    attempt: None,
                    error: Some(err),
                    remaining_depth: queue.health().depths.rescan,
                });
            }
        }
    };

    let mut envelope = dequeued.envelope;
    let payload = envelope.payload.clone();
    let processing_mode = {
        let mut scans = scan_store.lock().map_err(|e| e.to_string())?;
        ensure_scan_store_loaded(&mut scans);
        scans
            .scans
            .get(&payload.scan_id)
            .map(|scan| scan.processing_mode.clone())
            .or_else(|| sanitize_processing_mode(payload.processing_mode.as_deref()))
    };
    let app_for_scan = app.clone();
    let queue_run_id = client_run_id.clone();
    let run_result = task::spawn_blocking(move || {
        let scan_job = ScanJobStore(scan_job_store);
        let (_guard, cancel) = ScanJobGuard::acquire(&scan_job)?;
        let result = run_identity_scan_for_queue(
            IdentityScanArgs {
                video: payload.video,
                yolo_model: payload.yolo_model,
                face_model: payload.identity_model.clone(),
                identity_model: Some(payload.identity_model),
                body_reid_model: None,
                expected_member_count: None,
                processing_mode,
                client_run_id: queue_run_id,
            },
            app_for_scan,
            "queued rescan",
            Arc::clone(&cancel),
        );
        Ok::<_, String>((result, cancel.load(Ordering::Relaxed)))
    });
    let run_result = run_result.await.map_err(|e| e.to_string())??;

    let (run_result, cancelled) = run_result;

    match run_result {
        Ok(scan_result) => {
            let snapshot = {
                let mut scans = scan_store.lock().map_err(|e| e.to_string())?;
                ensure_scan_store_loaded(&mut scans);
                if let Some(scan) = scans.scans.get_mut(&payload.scan_id) {
                    scan.candidates = scan_result.candidates;
                    scan.duplicates = scan_result.duplicates;
                    scan.pending_split_ids.clear();
                    scan.review_ready = false;
                    scan.selected_identity_id = None;
                    scan.selected_anchor_x = None;
                    scan.selected_anchor_y = None;
                    scan.validated_threshold = None;
                    set_scan_status(scan, ScanSessionStatus::Proposed)?;
                    scan.last_blockers =
                        vec!["split rescan complete: please validate again".to_string()];
                    scan.updated_at_ms = epoch_ms();
                    append_scan_event(
                        scan,
                        "split_rescan_processed",
                        "candidates refreshed and review reset".to_string(),
                    );
                }
                snapshot_scan_entry(&scans, &payload.scan_id)?
            };
            if let Some(snapshot) = snapshot.as_ref() {
                persist_scan_entry_snapshot(snapshot)?;
            }
            let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
            queue.acknowledge(&envelope.message_id)?;
            Ok(queue::QueueProcessResult {
                processed: true,
                cancelled: false,
                queue: "rescan".to_string(),
                message_id: Some(envelope.message_id),
                job_id: Some(envelope.job_id),
                moved_to_dlq: false,
                requeued: false,
                attempt: Some(envelope.attempt),
                error: None,
                remaining_depth: queue.health().depths.rescan,
            })
        }
        Err(err) => {
            let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
            if cancelled {
                let remaining_depth = queue.requeue_rescan_raw_result(dequeued.raw)?;
                return Ok(queue::QueueProcessResult {
                    processed: false,
                    cancelled: true,
                    queue: "rescan".to_string(),
                    message_id: Some(envelope.message_id),
                    job_id: Some(envelope.job_id),
                    moved_to_dlq: false,
                    requeued: true,
                    attempt: Some(envelope.attempt),
                    error: None,
                    remaining_depth,
                });
            }
            let mut moved_to_dlq = false;
            let mut requeued = false;
            if envelope.attempt + 1 >= max_attempts_before_dlq {
                queue.move_rescan_to_dlq_with_error(dequeued.raw, Some(&err))?;
                moved_to_dlq = true;
            } else {
                queue.requeue_rescan_retry_with_error(&mut envelope, Some(&err))?;
                requeued = true;
            }
            Ok(queue::QueueProcessResult {
                processed: true,
                cancelled: false,
                queue: "rescan".to_string(),
                message_id: Some(envelope.message_id),
                job_id: Some(envelope.job_id),
                moved_to_dlq,
                requeued,
                attempt: Some(envelope.attempt),
                error: Some(err),
                remaining_depth: queue.health().depths.rescan,
            })
        }
    }
}

#[tauri::command]
pub fn queue_worker_start(
    app: AppHandle,
    queue_state: State<'_, QueueStore>,
    scan_state: State<'_, IdentityScanStore>,
    scan_job_state: State<'_, ScanJobStore>,
    worker_state: State<'_, QueueWorkerStore>,
    args: Option<QueueWorkerStartArgs>,
) -> Result<QueueWorkerStatus, String> {
    let poll_interval_ms = args
        .as_ref()
        .and_then(|a| a.poll_interval_ms)
        .unwrap_or(1200)
        .max(200);
    let max_attempts_before_dlq = args
        .as_ref()
        .and_then(|a| a.max_attempts_before_dlq)
        .unwrap_or(3)
        .max(1);

    {
        let mut worker = worker_state.0.lock().map_err(|e| e.to_string())?;
        if worker.running {
            return Ok(QueueWorkerStatus {
                running: worker.running,
                stop_requested: worker.stop_requested,
                poll_interval_ms: worker.poll_interval_ms,
                max_attempts_before_dlq: worker.max_attempts_before_dlq,
                processed_total: worker.processed_total,
                last_error: worker.last_error.clone(),
                recent_events: worker
                    .recent_events
                    .iter()
                    .rev()
                    .take(20)
                    .map(worker_event_payload)
                    .collect(),
            });
        }
        worker.running = true;
        worker.stop_requested = false;
        worker.poll_interval_ms = poll_interval_ms;
        worker.max_attempts_before_dlq = max_attempts_before_dlq;
        worker.last_error = None;
    }

    let queue_arc = queue_state.0.clone();
    let scan_arc = scan_state.0.clone();
    let scan_job_arc = scan_job_state.0.clone();
    let worker_arc = worker_state.0.clone();
    let stop_notify = worker_state.1.clone();
    let app_for_worker = app.clone();

    tokio::spawn(async move {
        loop {
            let should_stop = match worker_arc.lock() {
                Ok(worker) => worker.stop_requested,
                Err(_) => true,
            };
            if should_stop {
                break;
            }

            let discovery_result = process_next_discovery_job_core(
                Some(app_for_worker.clone()),
                queue_arc.clone(),
                scan_arc.clone(),
                scan_job_arc.clone(),
                max_attempts_before_dlq,
                None,
            )
            .await;

            let result = match discovery_result {
                Ok(res) if !res.processed => {
                    process_next_rescan_job_core(
                        Some(app_for_worker.clone()),
                        queue_arc.clone(),
                        scan_arc.clone(),
                        scan_job_arc.clone(),
                        max_attempts_before_dlq,
                        None,
                    )
                    .await
                }
                other => other,
            };

            let mut sleep_ms = poll_interval_ms;
            if let Ok(mut worker) = worker_arc.lock() {
                match result {
                    Ok(res) => {
                        if res.processed {
                            worker.processed_total = worker.processed_total.saturating_add(1);
                            sleep_ms = 150;
                        }
                        worker.last_error = res.error.clone();
                        if res.processed || res.error.is_some() {
                            push_worker_event(
                                &mut worker,
                                crate::QueueWorkerEvent {
                                    at_ms: epoch_ms(),
                                    queue: res.queue.clone(),
                                    message_id: res.message_id.clone(),
                                    job_id: res.job_id.clone(),
                                    attempt: res.attempt,
                                    moved_to_dlq: res.moved_to_dlq,
                                    requeued: res.requeued,
                                    error: res.error.clone(),
                                },
                            );
                        }
                    }
                    Err(err) => {
                        worker.last_error = Some(err.clone());
                        push_worker_event(
                            &mut worker,
                            crate::QueueWorkerEvent {
                                at_ms: epoch_ms(),
                                queue: "discovery".to_string(),
                                message_id: None,
                                job_id: None,
                                attempt: None,
                                moved_to_dlq: false,
                                requeued: false,
                                error: Some(err),
                            },
                        );
                    }
                }
            }

            tokio::select! {
                () = tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)) => {},
                () = stop_notify.notified() => break,
            }
        }

        if let Ok(mut worker) = worker_arc.lock() {
            worker.running = false;
            worker.stop_requested = false;
        }
    });

    queue_worker_status_internal(worker_state.0.clone())
}

#[tauri::command]
pub fn queue_worker_stop(
    worker_state: State<'_, QueueWorkerStore>,
) -> Result<QueueWorkerStatus, String> {
    {
        let mut worker = worker_state.0.lock().map_err(|e| e.to_string())?;
        worker.stop_requested = true;
    }
    worker_state.1.notify_one();
    queue_worker_status_internal(worker_state.0.clone())
}

#[tauri::command]
pub fn queue_worker_status(
    worker_state: State<'_, QueueWorkerStore>,
) -> Result<QueueWorkerStatus, String> {
    queue_worker_status_internal(worker_state.0.clone())
}

#[tauri::command]
pub fn queue_worker_clear_events(
    worker_state: State<'_, QueueWorkerStore>,
) -> Result<QueueWorkerStatus, String> {
    {
        let mut worker = worker_state.0.lock().map_err(|e| e.to_string())?;
        worker.recent_events.clear();
    }
    queue_worker_status_internal(worker_state.0.clone())
}

fn queue_worker_status_internal(
    worker_arc: std::sync::Arc<std::sync::Mutex<crate::QueueWorkerState>>,
) -> Result<QueueWorkerStatus, String> {
    let worker = worker_arc.lock().map_err(|e| e.to_string())?;
    Ok(QueueWorkerStatus {
        running: worker.running,
        stop_requested: worker.stop_requested,
        poll_interval_ms: worker.poll_interval_ms,
        max_attempts_before_dlq: worker.max_attempts_before_dlq,
        processed_total: worker.processed_total,
        last_error: worker.last_error.clone(),
        recent_events: worker
            .recent_events
            .iter()
            .rev()
            .take(20)
            .map(worker_event_payload)
            .collect(),
    })
}

fn worker_event_payload(event: &crate::QueueWorkerEvent) -> QueueWorkerEventPayload {
    QueueWorkerEventPayload {
        at_ms: event.at_ms,
        queue: event.queue.clone(),
        message_id: event.message_id.clone(),
        job_id: event.job_id.clone(),
        attempt: event.attempt,
        moved_to_dlq: event.moved_to_dlq,
        requeued: event.requeued,
        error: event.error.clone(),
    }
}

fn push_worker_event(worker: &mut crate::QueueWorkerState, event: crate::QueueWorkerEvent) {
    worker.recent_events.push_back(event);
    while worker.recent_events.len() > worker.max_events.max(1) {
        worker.recent_events.pop_front();
    }
}

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn next_run_id(prefix: &str) -> String {
    format!(
        "{prefix}-{}-{}",
        epoch_ms(),
        RUN_ID_SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

fn diagnostics_dir_path() -> PathBuf {
    storage::diagnostics_dir_path()
}

fn file_modified_ms(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
}

#[tauri::command]
pub async fn queue_peek_discovery_attempts(
    queue_state: State<'_, QueueStore>,
    args: Option<QueuePeekArgs>,
) -> Result<QueuePeekResult, String> {
    let limit = args.and_then(|a| a.limit).unwrap_or(10);
    let queue = queue_state.0.lock().map_err(|e| e.to_string())?;
    Ok(QueuePeekResult {
        attempts: queue.peek_discovery_attempts(limit)?,
    })
}

#[tauri::command]
pub async fn cancel_job(
    state: State<'_, CancelFlag>,
    render_state: State<'_, RenderJobStore>,
) -> Result<(), String> {
    state.0.store(true, Ordering::Relaxed);
    if let Ok(mut status) = render_state.0.lock() {
        status.cancelling = true;
    }
    Ok(())
}

#[tauri::command]
pub async fn cancel_scan(state: State<'_, ScanJobStore>) -> Result<(), String> {
    let mut job = state.0.lock().map_err(|e| e.to_string())?;
    if !job.running {
        return Ok(());
    }
    job.cancelling = true;
    job.cancel.store(true, Ordering::Relaxed);
    Ok(())
}

/// Guard that marks the render job as running and always resets it on drop,
/// even if the backing mutex has been poisoned.
#[derive(Debug)]
struct RenderJobGuard<'a> {
    store: &'a RenderJobStore,
}

impl<'a> RenderJobGuard<'a> {
    fn acquire(store: &'a RenderJobStore) -> Result<Self, String> {
        let mut job = match store.0.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                let mut g = poisoned.into_inner();
                g.running = false;
                g.cancelling = false;
                drop(g);
                store.0.clear_poison();
                store.0.lock().map_err(|e| e.to_string())?
            }
        };
        if job.running {
            let message = if job.cancelling {
                "render cancellation is in progress; wait for stop to finish".to_string()
            } else {
                "a render job is already running".to_string()
            };
            return Err(message);
        }
        job.running = true;
        job.cancelling = false;
        Ok(Self { store })
    }
}

impl<'a> Drop for RenderJobGuard<'a> {
    fn drop(&mut self) {
        match self.store.0.lock() {
            Ok(mut job) => {
                job.running = false;
                job.cancelling = false;
            }
            Err(poisoned) => {
                let mut job = poisoned.into_inner();
                job.running = false;
                job.cancelling = false;
                drop(job);
                self.store.0.clear_poison();
            }
        }
    }
}

#[tauri::command]
pub async fn run_fancam(
    app: AppHandle,
    state: State<'_, CancelFlag>,
    render_state: State<'_, RenderJobStore>,
    scan_state: State<'_, IdentityScanStore>,
    mut args: FancamArgs,
) -> Result<JobResult, String> {
    let render_run_id = args
        .client_run_id
        .clone()
        .unwrap_or_else(|| next_run_id("render"));
    let _guard = match RenderJobGuard::acquire(&render_state) {
        Ok(g) => g,
        Err(message) => {
            let result = JobResult {
                ok: false,
                message,
                output_path: None,
                run_id: Some(render_run_id.clone()),
            };
            emit_render_done(&app, &result);
            return Ok(result);
        }
    };
    state.0.store(false, Ordering::Relaxed);

    let scan_id_for_state = args.scan_id.clone();

    if let Err(message) = validate_fancam_paths(&args) {
        let result = JobResult {
            ok: false,
            message,
            output_path: None,
            run_id: Some(render_run_id.clone()),
        };
        emit_render_done(&app, &result);
        return Ok(result);
    }

    if let Some(scan_id) = &scan_id_for_state {
        let review_args = ValidateIdentityReviewArgs {
            scan_id: scan_id.clone(),
            selected_identity_id: args.selected_identity_id,
            threshold: args.threshold,
            excluded_identity_ids: args.excluded_identity_ids.clone(),
            accepted_low_confidence_ids: args.accepted_low_confidence_ids.clone(),
            resolved_duplicates: args.resolved_duplicates.clone(),
            pending_split_ids: args.pending_split_ids.clone(),
            expected_member_count: args.expected_member_count,
        };
        let snapshot = {
            let mut lock = scan_state.0.lock().map_err(|e| e.to_string())?;
            ensure_scan_store_loaded(&mut lock);
            let Some(scan) = lock.scans.get_mut(scan_id) else {
                let result = JobResult {
                    ok: false,
                    message: "identity validation session not found; rerun scan".to_string(),
                    output_path: None,
                    run_id: Some(render_run_id.clone()),
                };
                emit_render_done(&app, &result);
                return Ok(result);
            };

            // The persisted session is authoritative.  This prevents an old
            // webview request from silently changing encoder behavior.
            args.processing_mode = Some(scan.processing_mode.clone());

            let review = compute_review(scan, &review_args);
            if !review.ready {
                let why = if review.blockers.is_empty() {
                    "identity review not complete".to_string()
                } else {
                    format!(
                        "identity review not complete: {}",
                        review.blockers.join("; ")
                    )
                };
                let result = JobResult {
                    ok: false,
                    message: why,
                    output_path: None,
                    run_id: Some(render_run_id.clone()),
                };
                emit_render_done(&app, &result);
                return Ok(result);
            }

            apply_review_to_scan(scan, &review, args.threshold)?;

            args.threshold = scan
                .validated_threshold
                .unwrap_or(args.threshold)
                .clamp(0.0, 1.0);
            args.identity_model = scan
                .identity_model
                .clone()
                .or_else(|| Some(scan.face_model.clone()));
            args.selected_identity_id = scan.selected_identity_id;
            args.target_anchor_x = scan.selected_anchor_x;
            args.target_anchor_y = scan.selected_anchor_y;
            args.expected_member_count = scan.expected_count;
            args.target_embedding = scan
                .selected_identity_id
                .and_then(|id| scan.candidates.iter().find(|candidate| candidate.id == id))
                .and_then(|candidate| candidate.embedding.clone());
            if let Some(selected_id) = scan.selected_identity_id {
                let (target_embeddings, negative_embeddings) =
                    build_runtime_embedding_galleries(scan, selected_id);
                if !target_embeddings.is_empty() {
                    args.target_embeddings = Some(target_embeddings);
                    args.negative_embeddings = Some(negative_embeddings);
                    args.body_target_embeddings =
                        Some(build_runtime_body_reid_gallery(scan, selected_id));
                    args.identity_margin_threshold
                        .get_or_insert(DEFAULT_IDENTITY_MARGIN_THRESHOLD);
                    if args.body_reid_model.is_none() {
                        let identity_model = scan
                            .identity_model
                            .as_deref()
                            .filter(|value| !value.trim().is_empty())
                            .unwrap_or(&scan.face_model);
                        args.body_reid_model = resolve_default_body_reid_model(identity_model);
                    }
                }
            }

            set_scan_status(scan, ScanSessionStatus::Tracking)?;
            scan.updated_at_ms = epoch_ms();
            append_scan_event(
                scan,
                "tracking_started",
                format!(
                    "identity={} output={}",
                    scan.selected_identity_id
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    args.output
                ),
            );

            snapshot_scan_entry(&lock, scan_id)?
        };
        if let Some(snapshot) = snapshot.as_ref()
            && let Err(err) = persist_scan_entry_snapshot(snapshot)
        {
            let result = JobResult {
                ok: false,
                message: format!("failed to persist tracking state: {err}"),
                output_path: None,
                run_id: Some(render_run_id.clone()),
            };
            emit_render_done(&app, &result);
            return Ok(result);
        }
    }

    let cancel = Arc::clone(&state.0);
    let app2 = app.clone();

    let result = match task::spawn_blocking(move || run_pipeline(app2, cancel, args)).await {
        Ok(result) => result,
        Err(e) => {
            let result = JobResult {
                ok: false,
                message: format!("render task failed: {e}"),
                output_path: None,
                run_id: Some(render_run_id.clone()),
            };
            emit_render_done(&app, &result);
            return Ok(result);
        }
    };

    match result {
        Ok(path) => {
            if let Some(scan_id) = &scan_id_for_state
                && let Ok(mut lock) = scan_state.0.lock()
            {
                ensure_scan_store_loaded(&mut lock);
                let snapshot = if let Some(scan) = lock.scans.get_mut(scan_id) {
                    if let Err(error) = set_scan_status(scan, ScanSessionStatus::Completed) {
                        tracing::error!(%error, scan_id, "failed to complete scan session");
                        None
                    } else {
                        scan.updated_at_ms = epoch_ms();
                        append_scan_event(scan, "tracking_completed", format!("output={path}"));
                        snapshot_scan_entry(&lock, scan_id).ok().flatten()
                    }
                } else {
                    None
                };
                if let Some(snapshot) = snapshot.as_ref() {
                    let _ = persist_scan_entry_snapshot(snapshot);
                }
            }
            let result = JobResult {
                ok: true,
                message: "Done".into(),
                output_path: Some(path),
                run_id: Some(render_run_id),
            };
            emit_render_done(&app, &result);
            Ok(result)
        }
        Err(e) => {
            let cancelled = state.0.load(Ordering::Relaxed);
            if let Some(scan_id) = &scan_id_for_state
                && let Ok(mut lock) = scan_state.0.lock()
            {
                ensure_scan_store_loaded(&mut lock);
                let snapshot = if let Some(scan) = lock.scans.get_mut(scan_id) {
                    let (status, action, details) = if cancelled {
                        (
                            ScanSessionStatus::Validated,
                            "tracking_cancelled",
                            "render cancelled by user".to_string(),
                        )
                    } else {
                        (ScanSessionStatus::Failed, "tracking_failed", e.to_string())
                    };
                    if let Err(error) = set_scan_status(scan, status) {
                        tracing::error!(%error, scan_id, "failed to finalize scan session");
                        None
                    } else {
                        scan.updated_at_ms = epoch_ms();
                        append_scan_event(scan, action, details);
                        snapshot_scan_entry(&lock, scan_id).ok().flatten()
                    }
                } else {
                    None
                };
                if let Some(snapshot) = snapshot.as_ref() {
                    let _ = persist_scan_entry_snapshot(snapshot);
                }
            }
            let result = JobResult {
                ok: false,
                message: if cancelled {
                    "Render cancelled".to_string()
                } else {
                    e.to_string()
                },
                output_path: None,
                run_id: Some(render_run_id),
            };
            emit_render_done(&app, &result);
            Ok(result)
        }
    }
}

// ─── Pipeline (blocking) ─────────────────────────────────────────────────────

fn canonical_for_compare(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn validate_fancam_paths(args: &FancamArgs) -> Result<(), String> {
    let video_path = PathBuf::from(args.video.trim());
    let output_path = PathBuf::from(args.output.trim());
    let yolo_model = PathBuf::from(args.yolo_model.trim());
    let identity_model = effective_identity_model(&args.face_model, args.identity_model.as_deref());
    let face_model = PathBuf::from(identity_model);

    if !args.threshold.is_finite() {
        return Err("identity threshold must be finite".to_string());
    }
    if args
        .target_anchor_x
        .into_iter()
        .chain(args.target_anchor_y)
        .any(|value| !value.is_finite())
    {
        return Err("target anchor coordinates must be finite".to_string());
    }

    if !video_path.is_file() {
        return Err(format!(
            "input video not found: {}",
            video_path.to_string_lossy()
        ));
    }
    let has_embedding = args
        .target_embedding
        .as_ref()
        .is_some_and(|emb| !emb.is_empty())
        || args
            .target_embeddings
            .as_ref()
            .is_some_and(|gallery| gallery.iter().any(|emb| !emb.is_empty()));
    let bias_trimmed = args.bias.trim();
    if !has_embedding {
        let bias_path = PathBuf::from(bias_trimmed);
        if !bias_path.is_file() {
            return Err(format!(
                "bias image not found: {}",
                bias_path.to_string_lossy()
            ));
        }
    }

    if let Some(margin) = args.identity_margin_threshold
        && !margin.is_finite()
    {
        return Err("identity margin threshold must be finite".to_string());
    }
    if args
        .target_embedding
        .as_ref()
        .is_some_and(|embedding| embedding.iter().any(|value| !value.is_finite()))
        || args.target_embeddings.as_ref().is_some_and(|gallery| {
            gallery
                .iter()
                .any(|embedding| embedding.iter().any(|value| !value.is_finite()))
        })
        || args.negative_embeddings.as_ref().is_some_and(|gallery| {
            gallery
                .iter()
                .any(|embedding| embedding.iter().any(|value| !value.is_finite()))
        })
        || args.body_target_embeddings.as_ref().is_some_and(|gallery| {
            gallery
                .iter()
                .any(|embedding| embedding.iter().any(|value| !value.is_finite()))
        })
    {
        return Err("identity embeddings must contain only finite values".to_string());
    }
    if let Some(reid_model) = args.body_reid_model.as_ref().map(|s| s.trim())
        && !reid_model.is_empty()
    {
        let path = PathBuf::from(reid_model);
        if !path.is_file() {
            return Err(format!(
                "body reid model not found: {}",
                path.to_string_lossy()
            ));
        }
    }
    if !yolo_model.is_file() {
        return Err(format!(
            "YOLO model not found: {}",
            yolo_model.to_string_lossy()
        ));
    }
    if !face_model.is_file() {
        return Err(format!(
            "face model not found: {}",
            face_model.to_string_lossy()
        ));
    }

    if output_path.as_os_str().is_empty() {
        return Err("output path is empty".to_string());
    }

    let input_cmp = canonical_for_compare(&video_path);
    let output_cmp = canonical_for_compare(&output_path);
    if input_cmp == output_cmp {
        return Err("output path must be different from input video path".to_string());
    }

    let parent = output_path
        .parent()
        .ok_or_else(|| "output path has no parent directory".to_string())?;
    if !parent.exists() {
        return Err(format!(
            "output directory does not exist: {}",
            parent.to_string_lossy()
        ));
    }
    if !parent.is_dir() {
        return Err(format!(
            "output parent is not a directory: {}",
            parent.to_string_lossy()
        ));
    }

    validate_optional_plan_paths(args)?;

    Ok(())
}

fn validate_optional_plan_paths(args: &FancamArgs) -> Result<(), String> {
    let video = PathBuf::from(args.video.trim());
    let bias = PathBuf::from(args.bias.trim());
    let output = PathBuf::from(args.output.trim());
    let yolo = PathBuf::from(args.yolo_model.trim());
    let face = PathBuf::from(effective_identity_model(
        &args.face_model,
        args.identity_model.as_deref(),
    ));
    let source_paths = [&video, &bias, &output, &yolo, &face];

    let plan_input = args
        .plan_input
        .as_deref()
        .map(|path| validate_crop_plan_path(path, false))
        .transpose()?;
    let plan_output = args
        .plan_output
        .as_deref()
        .map(|path| validate_crop_plan_path(path, true))
        .transpose()?;

    if let Some(plan_input) = plan_input.as_ref() {
        let plan_input_cmp = canonical_for_compare(plan_input);
        if source_paths
            .iter()
            .any(|source| canonical_for_compare(source) == plan_input_cmp)
        {
            return Err("plan input must not replace a video, output, or model file".to_string());
        }
    }
    if let Some(plan_output) = plan_output.as_ref() {
        let plan_output_cmp = canonical_for_compare(plan_output);
        if source_paths
            .iter()
            .any(|source| canonical_for_compare(source) == plan_output_cmp)
        {
            return Err("plan output must not replace a video, output, or model file".to_string());
        }
        if plan_input
            .as_ref()
            .is_some_and(|input| canonical_for_compare(input) == plan_output_cmp)
        {
            return Err("plan input and plan output must be different files".to_string());
        }
    }
    Ok(())
}

fn validate_identity_scan_paths(args: &IdentityScanArgs) -> Result<(), String> {
    let video_path = PathBuf::from(args.video.trim());
    let yolo_model = PathBuf::from(args.yolo_model.trim());
    let identity_model = effective_identity_model(&args.face_model, args.identity_model.as_deref());
    let face_model = PathBuf::from(identity_model);

    if !video_path.is_file() {
        return Err(format!(
            "input video not found: {}",
            video_path.to_string_lossy()
        ));
    }
    if !yolo_model.is_file() {
        return Err(format!(
            "YOLO model not found: {}",
            yolo_model.to_string_lossy()
        ));
    }
    if !face_model.is_file() {
        return Err(format!(
            "face model not found: {}",
            face_model.to_string_lossy()
        ));
    }
    if let Some(body_reid_model) = args
        .body_reid_model
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let body_model = PathBuf::from(body_reid_model);
        if !body_model.is_file() {
            return Err(format!(
                "body reid model not found: {}",
                body_model.to_string_lossy()
            ));
        }
    }
    OrtConfig::ensure_initialized()
        .map_err(|e| format!("failed to initialize ONNX Runtime: {e}"))?;

    Ok(())
}

fn run_pipeline(
    app: AppHandle,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    args: FancamArgs,
) -> anyhow::Result<String> {
    let video_path = PathBuf::from(&args.video);
    let output_path = PathBuf::from(&args.output);
    let total = total_frames(&video_path);
    let threshold = args.threshold.clamp(0.0, 1.0);
    let mode = processing_mode_from_option(args.processing_mode.as_deref());
    let run_id = args
        .client_run_id
        .clone()
        .unwrap_or_else(|| next_run_id("render"));

    OrtConfig::ensure_initialized().map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let identity_model =
        effective_identity_model(&args.face_model, args.identity_model.as_deref()).to_string();

    let initial_hint = match (args.target_anchor_x, args.target_anchor_y) {
        (Some(x), Some(y)) => Some((x.max(0.0), y.max(0.0))),
        _ => None,
    };
    let margin_threshold = args
        .identity_margin_threshold
        .unwrap_or(DEFAULT_IDENTITY_MARGIN_THRESHOLD)
        .max(0.0);
    let body_reid_model = args
        .body_reid_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| resolve_default_body_reid_model(&identity_model));

    let target_gallery = args
        .target_embeddings
        .as_ref()
        .map(|rows| {
            rows.iter()
                .filter(|emb| !emb.is_empty())
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let negative_gallery = args
        .negative_embeddings
        .as_ref()
        .map(|rows| {
            rows.iter()
                .filter(|emb| !emb.is_empty())
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let body_target_gallery = args
        .body_target_embeddings
        .as_ref()
        .map(|rows| {
            rows.iter()
                .filter(|emb| !emb.is_empty())
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let input_plan = args
        .plan_input
        .as_deref()
        .map(read_crop_plan_file)
        .transpose()
        .context("failed to read crop plan input")?;

    let pipeline = if !target_gallery.is_empty() {
        Pipeline::load_with_hint_galleries(
            &args.yolo_model,
            &identity_model,
            body_reid_model.as_deref(),
            target_gallery,
            body_target_gallery,
            negative_gallery,
            threshold,
            margin_threshold,
            initial_hint,
            mode,
        )?
    } else if let Some(embedding) = args.target_embedding.as_ref().filter(|v| !v.is_empty()) {
        Pipeline::load_with_hint_embedding(
            &args.yolo_model,
            &identity_model,
            embedding.clone(),
            threshold,
            initial_hint,
            mode,
        )?
    } else {
        Pipeline::load_with_hint_mode(
            &args.yolo_model,
            &identity_model,
            &args.bias,
            threshold,
            initial_hint,
            mode,
        )?
    };
    let total_for_prepass = total.max(1);
    let prepass_run_id = run_id.clone();
    let mut prepass_last_emit = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);
    let needs_plan = args.plan_input.is_some() || args.plan_output.is_some();
    let offline_parts = if needs_plan {
        pipeline
            .into_parts_with_offline_solution_and_plan_with_hooks(
                &video_path,
                |progress: OfflinePrepassProgress| {
                    let now = Instant::now();
                    let should_emit = progress.decoded_frames <= 1
                        || now.duration_since(prepass_last_emit) >= Duration::from_millis(180);
                    if !should_emit {
                        return;
                    }
                    prepass_last_emit = now;

                    let decoded = progress.decoded_frames.min(total_for_prepass);
                    let fraction = 0.5 * (decoded as f64 / total_for_prepass as f64);
                    let _ = app.emit(
                        "fancam://progress",
                        ProgressPayload {
                            run_id: prepass_run_id.clone(),
                            current: decoded,
                            total: total_for_prepass,
                            fraction,
                        },
                    );
                },
                || cancel.load(Ordering::Relaxed),
            )
            .map(|(analyzer, renderer, plan)| (analyzer, renderer, Some(plan)))
    } else {
        pipeline
            .into_parts_with_offline_solution_with_hooks(
                &video_path,
                |progress: OfflinePrepassProgress| {
                    let now = Instant::now();
                    let should_emit = progress.decoded_frames <= 1
                        || now.duration_since(prepass_last_emit) >= Duration::from_millis(180);
                    if !should_emit {
                        return;
                    }
                    prepass_last_emit = now;

                    let decoded = progress.decoded_frames.min(total_for_prepass);
                    let fraction = 0.5 * (decoded as f64 / total_for_prepass as f64);
                    let _ = app.emit(
                        "fancam://progress",
                        ProgressPayload {
                            run_id: prepass_run_id.clone(),
                            current: decoded,
                            total: total_for_prepass,
                            fraction,
                        },
                    );
                },
                || cancel.load(Ordering::Relaxed),
            )
            .map(|(analyzer, renderer)| (analyzer, renderer, None))
    }?;
    let (mut analyzer, mut renderer, generated_plan) = offline_parts;

    if let Some(mut plan) = generated_plan {
        if let Some(input_plan) = input_plan {
            plan.ensure_source_fingerprint_matches(&input_plan.source_fingerprint)
                .context("crop plan input belongs to a different source video")?;
            plan.manual_keyframes = input_plan.manual_keyframes;
            plan.validate()
                .context("manual keyframes made crop plan invalid")?;
            analyzer.enable_offline_from_plan(&plan);
        }
        if let Some(plan_output) = args.plan_output.as_deref() {
            write_crop_plan_file(plan_output, &plan)
                .with_context(|| format!("failed to write crop plan {plan_output}"))?;
        }
    }

    let cancel_analyze = Arc::clone(&cancel);
    let cancel_render = Arc::clone(&cancel);
    let progress_run_id = run_id.clone();
    let mut last_progress_emit = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);

    transcode_with_progress_staged_mode_fallible(
        video_path,
        &output_path,
        total,
        Arc::clone(&cancel),
        move |frame| {
            if cancel_analyze.load(Ordering::Relaxed) {
                None
            } else {
                analyzer.analyze(frame)
            }
        },
        move |frame, camera| {
            if cancel_render.load(Ordering::Relaxed) {
                return Ok(());
            }
            renderer.render_checked(frame, camera)
        },
        mode,
        |current, total| {
            let now = Instant::now();
            let is_last = total > 0 && current >= total;
            let should_emit = current <= 1
                || is_last
                || now.duration_since(last_progress_emit) >= Duration::from_millis(180);
            if !should_emit {
                return;
            }
            last_progress_emit = now;
            let fraction = if total > 0 {
                0.5 + 0.5 * (current as f64 / total as f64)
            } else {
                0.5
            };
            let _ = app.emit(
                "fancam://progress",
                ProgressPayload {
                    run_id: progress_run_id.clone(),
                    current,
                    total,
                    fraction,
                },
            );
        },
    )?;

    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("render cancelled");
    }

    Ok(args.output)
}

fn sanitize_processing_mode(value: Option<&str>) -> Option<String> {
    value.map(|v| processing_mode_string(Some(v)))
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::atomic::Ordering,
        sync::{Mutex, OnceLock},
    };

    use super::{
        CropPlanPathArgs, CropPlanWriteArgs, FancamArgs, QueryIdentityScansArgs,
        QueryScanEventsArgs, RenderJobGuard, RenderJobStore, ScanJobGuard, ScanJobStore,
        UpdateCropPlanKeyframeArgs, ensure_crop_plan_metadata_matches, query_identity_scans,
        query_scan_events, read_crop_plan, update_crop_plan_keyframe, validate_fancam_paths,
        write_crop_plan,
    };
    use crate::storage;
    use fancam_core::plan::{
        CROP_PLAN_SCHEMA, CROP_PLAN_VERSION, CropOutput, CropPlanV1, PlanKeyframe,
        PlanKeyframeSource, PlanQualityMetrics, SOURCE_FINGERPRINT_ALGORITHM,
        SOURCE_FINGERPRINT_VERSION, SourceVideoFingerprint, VideoMetadata,
    };

    fn diagnostics_test_mutex() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_temp_workspace<T>(f: impl FnOnce(PathBuf) -> T) -> T {
        let _guard = diagnostics_test_mutex()
            .lock()
            .expect("test mutex poisoned");
        let previous = std::env::current_dir().expect("cwd");
        let mut dir = std::env::temp_dir();
        dir.push(format!("focus-lock-workspace-test-{}", super::epoch_ms()));
        std::fs::create_dir_all(&dir).expect("create test workspace");
        std::env::set_current_dir(&dir).expect("set test cwd");
        let result = f(dir.clone());
        std::env::set_current_dir(previous).expect("restore cwd");
        let _ = std::fs::remove_dir_all(dir);
        result
    }

    #[test]
    fn query_identity_scans_ignores_offset_when_cursor_present() {
        with_temp_workspace(|_| {
            let now = super::epoch_ms();
            let rows = storage::ScanStoreRows {
                next_id: 3,
                sessions: vec![
                    storage::ScanSessionRow {
                        scan_id: "scan-a".to_string(),
                        video: "a.mp4".to_string(),
                        yolo_model: "y.onnx".to_string(),
                        identity_model: "f.onnx".to_string(),
                        status: "validated".to_string(),
                        processing_mode: "fast".to_string(),
                        expected_count: Some(3),
                        review_ready: true,
                        selected_identity_id: Some(1),
                        selected_anchor_x: None,
                        selected_anchor_y: None,
                        validated_threshold: Some(0.65),
                        updated_at_ms: now,
                        candidates_json: "[]".to_string(),
                        duplicates_json: "[]".to_string(),
                        excluded_identity_ids_json: "[]".to_string(),
                        accepted_low_confidence_ids_json: "[]".to_string(),
                        resolved_duplicate_keys_json: "[]".to_string(),
                        pending_split_ids_json: "[]".to_string(),
                        pending_split_count: 0,
                        last_blockers_json: "[]".to_string(),
                    },
                    storage::ScanSessionRow {
                        scan_id: "scan-b".to_string(),
                        video: "b.mp4".to_string(),
                        yolo_model: "y.onnx".to_string(),
                        identity_model: "f.onnx".to_string(),
                        status: "proposed".to_string(),
                        processing_mode: "balanced".to_string(),
                        expected_count: None,
                        review_ready: false,
                        selected_identity_id: None,
                        selected_anchor_x: None,
                        selected_anchor_y: None,
                        validated_threshold: None,
                        updated_at_ms: now.saturating_sub(1),
                        candidates_json: "[]".to_string(),
                        duplicates_json: "[]".to_string(),
                        excluded_identity_ids_json: "[]".to_string(),
                        accepted_low_confidence_ids_json: "[]".to_string(),
                        resolved_duplicate_keys_json: "[]".to_string(),
                        pending_split_ids_json: "[]".to_string(),
                        pending_split_count: 0,
                        last_blockers_json: "[]".to_string(),
                    },
                    storage::ScanSessionRow {
                        scan_id: "scan-c".to_string(),
                        video: "c.mp4".to_string(),
                        yolo_model: "y.onnx".to_string(),
                        identity_model: "f.onnx".to_string(),
                        status: "failed".to_string(),
                        processing_mode: "quality".to_string(),
                        expected_count: None,
                        review_ready: false,
                        selected_identity_id: None,
                        selected_anchor_x: None,
                        selected_anchor_y: None,
                        validated_threshold: None,
                        updated_at_ms: now.saturating_sub(2),
                        candidates_json: "[]".to_string(),
                        duplicates_json: "[]".to_string(),
                        excluded_identity_ids_json: "[]".to_string(),
                        accepted_low_confidence_ids_json: "[]".to_string(),
                        resolved_duplicate_keys_json: "[]".to_string(),
                        pending_split_ids_json: "[]".to_string(),
                        pending_split_count: 0,
                        last_blockers_json: "[]".to_string(),
                    },
                ],
                events: vec![],
            };
            storage::save_scan_rows(&storage::scan_store_db_path(), &rows).expect("seed rows");

            let first = query_identity_scans(Some(QueryIdentityScansArgs {
                limit: Some(1),
                offset: Some(0),
                status: None,
                cursor_updated_at_ms: None,
                cursor_scan_id: None,
            }))
            .expect("query first page");
            assert_eq!(first.rows.len(), 1);
            let cursor_ms = first.next_cursor_updated_at_ms.expect("cursor ms");
            let cursor_id = first.next_cursor_scan_id.clone().expect("cursor id");

            let with_zero_offset = query_identity_scans(Some(QueryIdentityScansArgs {
                limit: Some(2),
                offset: Some(0),
                status: None,
                cursor_updated_at_ms: Some(cursor_ms),
                cursor_scan_id: Some(cursor_id.clone()),
            }))
            .expect("query cursor page offset zero");
            let with_large_offset = query_identity_scans(Some(QueryIdentityScansArgs {
                limit: Some(2),
                offset: Some(99),
                status: None,
                cursor_updated_at_ms: Some(cursor_ms),
                cursor_scan_id: Some(cursor_id),
            }))
            .expect("query cursor page with ignored offset");

            assert_eq!(with_zero_offset.rows.len(), with_large_offset.rows.len());
            assert_eq!(
                with_zero_offset
                    .rows
                    .iter()
                    .map(|r| r.scan_id.clone())
                    .collect::<Vec<_>>(),
                with_large_offset
                    .rows
                    .iter()
                    .map(|r| r.scan_id.clone())
                    .collect::<Vec<_>>()
            );
            assert!(!with_zero_offset.offset_ignored);
            assert!(with_large_offset.offset_ignored);
        });
    }

    #[test]
    fn query_scan_events_ignores_offset_when_cursor_present() {
        with_temp_workspace(|_| {
            let now = super::epoch_ms();
            let rows = storage::ScanStoreRows {
                next_id: 1,
                sessions: vec![storage::ScanSessionRow {
                    scan_id: "scan-events".to_string(),
                    video: "events.mp4".to_string(),
                    yolo_model: "y.onnx".to_string(),
                    identity_model: "f.onnx".to_string(),
                    status: "tracking".to_string(),
                    processing_mode: "fast".to_string(),
                    expected_count: None,
                    review_ready: true,
                    selected_identity_id: Some(1),
                    selected_anchor_x: None,
                    selected_anchor_y: None,
                    validated_threshold: Some(0.6),
                    updated_at_ms: now,
                    candidates_json: "[]".to_string(),
                    duplicates_json: "[]".to_string(),
                    excluded_identity_ids_json: "[]".to_string(),
                    accepted_low_confidence_ids_json: "[]".to_string(),
                    resolved_duplicate_keys_json: "[]".to_string(),
                    pending_split_ids_json: "[]".to_string(),
                    pending_split_count: 0,
                    last_blockers_json: "[]".to_string(),
                }],
                events: vec![
                    storage::ScanSessionEventRow {
                        scan_id: "scan-events".to_string(),
                        at_ms: now.saturating_sub(3),
                        action: "a".to_string(),
                        details: "one".to_string(),
                    },
                    storage::ScanSessionEventRow {
                        scan_id: "scan-events".to_string(),
                        at_ms: now.saturating_sub(2),
                        action: "b".to_string(),
                        details: "two".to_string(),
                    },
                    storage::ScanSessionEventRow {
                        scan_id: "scan-events".to_string(),
                        at_ms: now.saturating_sub(1),
                        action: "c".to_string(),
                        details: "three".to_string(),
                    },
                ],
            };
            storage::save_scan_rows(&storage::scan_store_db_path(), &rows).expect("seed rows");

            let first = query_scan_events(QueryScanEventsArgs {
                scan_id: "scan-events".to_string(),
                limit: Some(1),
                offset: Some(0),
                action_contains: None,
                since_ms: None,
                until_ms: None,
                cursor_event_id: None,
            })
            .expect("query first event page");
            assert_eq!(first.rows.len(), 1);
            let cursor = first.next_cursor_event_id.expect("cursor id");

            let with_zero_offset = query_scan_events(QueryScanEventsArgs {
                scan_id: "scan-events".to_string(),
                limit: Some(2),
                offset: Some(0),
                action_contains: None,
                since_ms: None,
                until_ms: None,
                cursor_event_id: Some(cursor),
            })
            .expect("query events with offset 0");
            let with_large_offset = query_scan_events(QueryScanEventsArgs {
                scan_id: "scan-events".to_string(),
                limit: Some(2),
                offset: Some(99),
                action_contains: None,
                since_ms: None,
                until_ms: None,
                cursor_event_id: Some(cursor),
            })
            .expect("query events with ignored offset");

            assert_eq!(with_zero_offset.rows.len(), with_large_offset.rows.len());
            assert_eq!(
                with_zero_offset
                    .rows
                    .iter()
                    .map(|r| format!("{}:{}", r.action, r.details))
                    .collect::<Vec<_>>(),
                with_large_offset
                    .rows
                    .iter()
                    .map(|r| format!("{}:{}", r.action, r.details))
                    .collect::<Vec<_>>()
            );
            assert!(!with_zero_offset.offset_ignored);
            assert!(with_large_offset.offset_ignored);
        });
    }

    #[test]
    fn interrupted_tracking_sessions_recover_and_persist_their_mode() {
        with_temp_workspace(|_| {
            let now = super::epoch_ms();
            let rows = storage::ScanStoreRows {
                next_id: 2,
                sessions: vec![storage::ScanSessionRow {
                    scan_id: "scan-recover".to_string(),
                    video: "recover.mp4".to_string(),
                    yolo_model: "y.onnx".to_string(),
                    identity_model: "f.onnx".to_string(),
                    status: "tracking".to_string(),
                    processing_mode: "quality".to_string(),
                    expected_count: Some(1),
                    review_ready: true,
                    selected_identity_id: Some(1),
                    selected_anchor_x: Some(10.0),
                    selected_anchor_y: Some(20.0),
                    validated_threshold: Some(0.7),
                    updated_at_ms: now,
                    candidates_json: "[]".to_string(),
                    duplicates_json: "[]".to_string(),
                    excluded_identity_ids_json: "[]".to_string(),
                    accepted_low_confidence_ids_json: "[]".to_string(),
                    resolved_duplicate_keys_json: "[]".to_string(),
                    pending_split_ids_json: "[]".to_string(),
                    pending_split_count: 0,
                    last_blockers_json: "[]".to_string(),
                }],
                events: vec![],
            };
            storage::save_scan_rows(&storage::scan_store_db_path(), &rows).expect("seed scan");

            let mut state = super::IdentityScanState::default();
            super::ensure_scan_store_loaded(&mut state);
            let recovered = state.scans.get("scan-recover").expect("recovered scan");
            assert_eq!(recovered.status, super::ScanSessionStatus::Validated);
            assert_eq!(recovered.processing_mode, "quality");
            assert!(
                recovered
                    .events
                    .iter()
                    .any(|event| event.action == "tracking_recovered")
            );

            let persisted = storage::load_scan_rows(&storage::scan_store_db_path())
                .expect("load persisted scan")
                .expect("persisted scan rows");
            assert_eq!(persisted.sessions[0].status, "validated");
            assert_eq!(persisted.sessions[0].processing_mode, "quality");
        });
    }

    #[test]
    fn preview_dimension_limits_reject_empty_and_oversized_inputs() {
        assert!(super::validate_preview_dimensions(0, 100).is_err());
        assert!(super::validate_preview_dimensions(4096, 4096).is_ok());
        assert!(super::validate_preview_dimensions(4097, 1).is_err());
        assert!(super::validate_preview_dimensions(4096, 4097).is_err());
    }

    #[test]
    fn fancam_preflight_rejects_same_input_output() {
        with_temp_workspace(|dir| {
            let video = dir.join("input.mp4");
            let bias = dir.join("bias.jpg");
            let yolo = dir.join("yolo.onnx");
            let face = dir.join("face.onnx");
            std::fs::write(&video, b"v").expect("write video");
            std::fs::write(&bias, b"b").expect("write bias");
            std::fs::write(&yolo, b"y").expect("write yolo");
            std::fs::write(&face, b"f").expect("write face");

            let args = FancamArgs {
                video: video.to_string_lossy().into_owned(),
                bias: bias.to_string_lossy().into_owned(),
                output: video.to_string_lossy().into_owned(),
                yolo_model: yolo.to_string_lossy().into_owned(),
                face_model: face.to_string_lossy().into_owned(),
                identity_model: None,
                threshold: 0.6,
                processing_mode: None,
                body_reid_model: None,
                target_embedding: None,
                target_embeddings: None,
                body_target_embeddings: None,
                negative_embeddings: None,
                identity_margin_threshold: None,
                expected_member_count: None,
                excluded_identity_ids: Vec::new(),
                accepted_low_confidence_ids: Vec::new(),
                resolved_duplicates: Vec::new(),
                pending_split_ids: Vec::new(),
                client_run_id: None,
                plan_input: None,
                plan_output: None,
                scan_id: None,
                selected_identity_id: None,
                target_anchor_x: None,
                target_anchor_y: None,
            };
            let result = validate_fancam_paths(&args);
            assert!(result.is_err());
        });
    }

    #[test]
    fn fancam_preflight_accepts_valid_paths() {
        with_temp_workspace(|dir| {
            let video = dir.join("input.mp4");
            let bias = dir.join("bias.jpg");
            let yolo = dir.join("yolo.onnx");
            let face = dir.join("face.onnx");
            let output = dir.join("out.mp4");
            std::fs::write(&video, b"v").expect("write video");
            std::fs::write(&bias, b"b").expect("write bias");
            std::fs::write(&yolo, b"y").expect("write yolo");
            std::fs::write(&face, b"f").expect("write face");

            let args = FancamArgs {
                video: video.to_string_lossy().into_owned(),
                bias: bias.to_string_lossy().into_owned(),
                output: output.to_string_lossy().into_owned(),
                yolo_model: yolo.to_string_lossy().into_owned(),
                face_model: face.to_string_lossy().into_owned(),
                identity_model: None,
                threshold: 0.6,
                processing_mode: None,
                body_reid_model: None,
                target_embedding: None,
                target_embeddings: None,
                body_target_embeddings: None,
                negative_embeddings: None,
                identity_margin_threshold: None,
                expected_member_count: None,
                excluded_identity_ids: Vec::new(),
                accepted_low_confidence_ids: Vec::new(),
                resolved_duplicates: Vec::new(),
                pending_split_ids: Vec::new(),
                client_run_id: None,
                plan_input: None,
                plan_output: None,
                scan_id: None,
                selected_identity_id: None,
                target_anchor_x: None,
                target_anchor_y: None,
            };
            let result = validate_fancam_paths(&args);
            assert!(result.is_ok());
        });
    }

    #[test]
    fn crop_plan_commands_round_trip_and_update_only_manual_geometry() {
        with_temp_workspace(|dir| {
            let path = dir.join("crop-plan.json");
            let plan = CropPlanV1 {
                schema: CROP_PLAN_SCHEMA.to_string(),
                version: CROP_PLAN_VERSION,
                source_fingerprint: SourceVideoFingerprint {
                    version: SOURCE_FINGERPRINT_VERSION,
                    algorithm: SOURCE_FINGERPRINT_ALGORITHM.to_string(),
                    digest: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                        .to_string(),
                    file_size: 128,
                    sampled_bytes: 128,
                },
                video: VideoMetadata {
                    width: 1920,
                    height: 1080,
                    frame_count: 10,
                    frame_rate_num: 30,
                    frame_rate_den: 1,
                    duration_ms: Some(333),
                },
                output: CropOutput {
                    width: 1080,
                    height: 1920,
                },
                shots: Vec::new(),
                tracks: Vec::new(),
                keyframes: vec![PlanKeyframe {
                    frame_index: 1,
                    cx: 800.0,
                    cy: 540.0,
                    half_size: 40.0,
                    confidence: 0.8,
                    source: PlanKeyframeSource::Observed,
                }],
                manual_keyframes: Vec::new(),
                quality: PlanQualityMetrics {
                    observed_keyframes: 1,
                    predicted_keyframes: 0,
                    held_keyframes: 0,
                    mean_confidence: 0.8,
                    min_confidence: 0.8,
                    path_coverage: 0.1,
                    max_gap_frames: 0,
                    shot_boundary_count: 0,
                },
            };

            let written = write_crop_plan(CropPlanWriteArgs {
                path: path.to_string_lossy().into_owned(),
                plan,
            })
            .expect("write crop plan");
            assert!(written.bytes > 0);

            let json = std::fs::read_to_string(&path).expect("read crop plan bytes");
            assert!(!json.contains("embedding"));
            assert!(!json.contains("thumbnail"));

            let error = read_crop_plan(CropPlanPathArgs {
                path: path.to_string_lossy().into_owned(),
                video: None,
            })
            .expect_err("source binding is required");
            assert!(error.contains("source video is required"));
            let loaded = fancam_core::plan::read_crop_plan(&path).expect("read crop plan file");
            assert!(loaded.manual_keyframes.is_empty());

            let updated = update_crop_plan_keyframe(UpdateCropPlanKeyframeArgs {
                path: path.to_string_lossy().into_owned(),
                keyframe: fancam_core::plan::ManualKeyframe {
                    frame_index: 1,
                    cx: 900.0,
                    cy: 540.0,
                    half_size: 44.0,
                },
            })
            .expect("update manual keyframe");
            assert_eq!(updated.manual_keyframes.len(), 1);
            assert_eq!(updated.manual_keyframes[0].cx, 900.0);

            let invalid = dir.join("invalid.json");
            std::fs::write(&invalid, b"{\"schema\":\"wrong\"}").expect("write invalid plan");
            assert!(
                read_crop_plan(CropPlanPathArgs {
                    path: invalid.to_string_lossy().into_owned(),
                    video: None,
                })
                .is_err()
            );
        });
    }

    #[test]
    fn crop_plan_source_check_ignores_only_the_probed_frame_estimate() {
        let plan = CropPlanV1 {
            schema: CROP_PLAN_SCHEMA.to_string(),
            version: CROP_PLAN_VERSION,
            source_fingerprint: SourceVideoFingerprint {
                version: SOURCE_FINGERPRINT_VERSION,
                algorithm: SOURCE_FINGERPRINT_ALGORITHM.to_string(),
                digest: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .to_string(),
                file_size: 128,
                sampled_bytes: 128,
            },
            video: VideoMetadata {
                width: 1920,
                height: 1080,
                frame_count: 117,
                frame_rate_num: 30,
                frame_rate_den: 1,
                duration_ms: Some(4_000),
            },
            output: CropOutput {
                width: 1080,
                height: 1920,
            },
            shots: Vec::new(),
            tracks: Vec::new(),
            keyframes: Vec::new(),
            manual_keyframes: Vec::new(),
            quality: PlanQualityMetrics {
                observed_keyframes: 0,
                predicted_keyframes: 0,
                held_keyframes: 0,
                mean_confidence: 0.0,
                min_confidence: 0.0,
                path_coverage: 0.0,
                max_gap_frames: 0,
                shot_boundary_count: 0,
            },
        };
        let mut probe = plan.video.clone();
        probe.frame_count = 120;

        assert!(ensure_crop_plan_metadata_matches(&plan, &probe).is_ok());
        probe.width = 1280;
        assert!(ensure_crop_plan_metadata_matches(&plan, &probe).is_err());
    }

    #[test]
    fn render_guard_acquire_marks_running_and_resets_on_drop() {
        let store = RenderJobStore::default();
        {
            let guard = RenderJobGuard::acquire(&store).expect("acquire render guard");
            assert!(store.0.lock().expect("lock").running);
            drop(guard);
        }
        let state = store.0.lock().expect("lock");
        assert!(!state.running);
        assert!(!state.cancelling);
    }

    #[test]
    fn render_guard_second_acquire_fails_while_running() {
        let store = RenderJobStore::default();
        let guard = RenderJobGuard::acquire(&store).expect("first acquire");
        let err = RenderJobGuard::acquire(&store).expect_err("second acquire should fail");
        assert!(err.contains("already running"));
        drop(guard);
    }

    #[test]
    fn render_guard_recovers_from_poisoned_mutex() {
        let store = RenderJobStore::default();
        // Poison the mutex by panicking while holding the lock.
        let result = std::panic::catch_unwind(|| {
            let _guard = store.0.lock().expect("lock");
            panic!("intentional test panic");
        });
        assert!(result.is_err());
        assert!(store.0.is_poisoned());

        // The guard should recover and still allow a new render to start.
        let guard = RenderJobGuard::acquire(&store).expect("acquire after poison");
        assert!(store.0.lock().expect("lock after recovery").running);
        drop(guard);

        let state = store.0.lock().expect("final lock");
        assert!(!state.running);
        assert!(!store.0.is_poisoned());
    }

    #[test]
    fn scan_guard_acquire_marks_running_and_resets_on_drop() {
        let store = ScanJobStore::default();
        let (_guard, cancel) = ScanJobGuard::acquire(&store).expect("acquire scan guard");
        assert!(store.0.lock().expect("lock").running);
        assert!(!cancel.load(Ordering::Relaxed));
        drop(_guard);

        let state = store.0.lock().expect("final lock");
        assert!(!state.running);
        assert!(!state.cancelling);
    }

    #[test]
    fn scan_guard_second_acquire_fails_while_running() {
        let store = ScanJobStore::default();
        let (guard, _cancel) = ScanJobGuard::acquire(&store).expect("first acquire");
        let err = ScanJobGuard::acquire(&store).expect_err("second acquire should fail");
        assert!(err.contains("already running"));
        drop(guard);
    }

    #[test]
    fn scan_guard_per_scan_cancel_flag_is_isolated() {
        let store = ScanJobStore::default();
        let (guard_a, cancel_a) = ScanJobGuard::acquire(&store).expect("first scan");
        cancel_a.store(true, Ordering::Relaxed);
        assert!(cancel_a.load(Ordering::Relaxed));
        drop(guard_a);

        // A subsequent scan gets a fresh, unset cancel flag.
        let (guard_b, cancel_b) = ScanJobGuard::acquire(&store).expect("second scan");
        assert!(!cancel_b.load(Ordering::Relaxed));
        drop(guard_b);
    }
}
