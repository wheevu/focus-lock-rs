use std::{
    collections::HashMap,
    collections::HashSet,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use fancam_core::{
    detection::DEFAULT_IDENTITY_MARGIN_THRESHOLD,
    discovery::{DiscoveryConfig, DiscoveryEngine},
    mode::ProcessingMode,
    pipeline::Pipeline,
    runtime::OrtConfig,
    video::{total_frames, transcode_with_progress_staged_mode},
};
use image::ImageReader;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, State};
use tokio::task;

use crate::{
    CancelFlag, IdentityScanState, IdentityScanStore, QueueStore, QueueWorkerStore, RenderJobStore,
    ScanCancelFlag, StorageWorkerStore, queue, storage,
};

static RUN_ID_SEQ: AtomicU64 = AtomicU64::new(1);

// ─── DTO types ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FancamArgs {
    pub video: String,
    pub bias: String,
    pub output: String,
    pub yolo_model: String,
    pub face_model: String,
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
}

#[derive(Debug, Deserialize)]
pub struct IdentityScanArgs {
    pub video: String,
    pub yolo_model: String,
    pub face_model: String,
    pub expected_member_count: Option<u32>,
    pub processing_mode: Option<String>,
    pub client_run_id: Option<String>,
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
    pub thumbnail_data_url: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
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
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListDiagnosticsBundlesResult {
    pub bundles: Vec<DiagnosticsBundleInfo>,
}

#[derive(Debug, Deserialize)]
pub struct PruneDiagnosticsBundlesArgs {
    pub keep_latest: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct PruneDiagnosticsBundlesResult {
    pub deleted: usize,
    pub kept: usize,
}

#[derive(Debug, Deserialize)]
pub struct DeleteDiagnosticsBundleArgs {
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct DeleteDiagnosticsBundleResult {
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct VerifyDiagnosticsBundleArgs {
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct VerifyDiagnosticsBundleResult {
    pub path: String,
    pub expected_sha256: Option<String>,
    pub actual_sha256: String,
    pub matches: bool,
}

#[derive(Debug, Deserialize)]
pub struct ReadDiagnosticsBundleArgs {
    pub path: String,
    pub max_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ReadDiagnosticsBundleResult {
    pub path: String,
    pub bytes: usize,
    pub content: String,
    pub truncated: bool,
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

#[tauri::command]
pub async fn probe_video(path: String) -> u64 {
    let p = PathBuf::from(path);
    task::spawn_blocking(move || total_frames(&p))
        .await
        .unwrap_or(0)
}

/// Read an image file and return a small JPEG data-URL suitable for preview.
/// For video files we decode the first frame via ffmpeg; for images we just
/// read + transcode.  The result fits in an `<img src="...">` attribute.
#[tauri::command]
pub async fn read_thumbnail(path: String) -> Result<String, String> {
    task::spawn_blocking(move || make_thumbnail(&path))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn scan_identities(
    app: AppHandle,
    state: State<'_, IdentityScanStore>,
    scan_cancel: State<'_, ScanCancelFlag>,
    args: IdentityScanArgs,
) -> Result<IdentityScanResult, String> {
    validate_identity_scan_paths(&args)?;
    let run_id = args
        .client_run_id
        .clone()
        .unwrap_or_else(|| next_run_id("scan"));

    {
        scan_cancel.0.store(false, Ordering::Relaxed);
    }

    let app_for_scan = app.clone();
    let cancel_flag = Arc::clone(&scan_cancel.0);
    let yolo_model = args.yolo_model.clone();
    let face_model = args.face_model.clone();
    let progress_run_id = run_id.clone();
    let scan_result = task::spawn_blocking(move || {
        run_identity_scan_with_hooks(
            IdentityScanArgs {
                processing_mode: sanitize_processing_mode(args.processing_mode.as_deref()),
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
            &face_model,
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
            candidate
        })
        .collect()
}

fn upsert_scan_cache(
    scans: &mut std::collections::HashMap<String, IdentityScanCache>,
    scan_id: &str,
    scan_result: &IdentityScanResult,
    yolo_model: &str,
    face_model: &str,
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
            face_model: face_model.to_string(),
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
            | (ScanSessionStatus::Tracking, ScanSessionStatus::Completed)
            | (ScanSessionStatus::Tracking, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Proposed, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Validated, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Completed, ScanSessionStatus::Failed)
            | (ScanSessionStatus::Failed, ScanSessionStatus::Proposed)
    )
}

fn set_scan_status(scan: &mut IdentityScanCache, to: ScanSessionStatus) {
    if can_transition_status(&scan.status, &to) {
        scan.status = to;
    }
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
    Ok(storage::ScanSessionRow {
        scan_id: scan_id.to_string(),
        video: scan.video.clone(),
        yolo_model: scan.yolo_model.clone(),
        face_model: scan.face_model.clone(),
        status: status_to_db(&scan.status).to_string(),
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
        face_model: row.face_model.clone(),
        processing_mode: "fast".to_string(),
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

        state.next_id = rows.next_id;
        state.scans = scans;
        state.loaded_from_disk = true;
        return;
    }

    let legacy_json_path = storage::scan_store_json_path();
    if let Ok(bytes) = fs::read(&legacy_json_path)
        && let Ok(persisted) = serde_json::from_slice::<PersistedScanState>(&bytes)
    {
        state.next_id = persisted.next_id;
        state.scans = persisted.scans;
        let _ = persist_scan_store(state);
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

fn make_thumbnail(path: &str) -> Result<String, String> {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();

    let is_video = matches!(
        ext.as_str(),
        "mp4" | "mov" | "mkv" | "avi" | "webm" | "ts" | "flv"
    );

    let rgb_image = if is_video {
        extract_video_frame(path).map_err(|e| format!("video frame extraction: {e}"))?
    } else {
        ImageReader::open(path)
            .map_err(|e| format!("open image: {e}"))?
            .decode()
            .map_err(|e| format!("decode image: {e}"))?
            .to_rgb8()
    };

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
        || false,
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

    let mut engine = DiscoveryEngine::load(&args.yolo_model, &args.face_model)
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
    let low_confidence = report.candidates.iter().any(|c| c.confidence < 0.6);

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
        || low_confidence
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
    let needs_review = count_blocker || duplicate_blocker || confidence_blocker;

    let sampled_frames = report.sampled_frames;
    let total_decoded_frames = report.total_decoded_frames;
    let rejected_embeddings = report.rejected_embeddings;
    let suppressed_clusters = report.suppressed_clusters;
    let merged_clusters = report.merged_clusters;
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
            thumbnail_data_url: format!("data:image/jpeg;base64,{}", B64.encode(c.thumbnail_jpeg)),
            embedding: Some(c.embedding),
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
        .map(ProcessingMode::from_str)
        .unwrap_or_else(ProcessingMode::default)
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
        b.observations.cmp(&a.observations).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    report.candidates.truncate(expected.saturating_add(2));
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

fn normalize_review_duplicates(
    resolved_duplicates: &[ReviewDuplicateResolution],
) -> (
    Vec<ReviewDuplicateResolution>,
    HashSet<(usize, usize)>,
    HashSet<usize>,
    usize,
) {
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

fn apply_review_to_scan(scan: &mut IdentityScanCache, review: &ReviewComputation, threshold: f32) {
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
    );
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
    let selected = scan
        .candidates
        .iter()
        .find(|candidate| candidate.id == selected_id)
        .and_then(|candidate| candidate.embedding.clone())
        .filter(|emb| !emb.is_empty());
    let Some(selected_embedding) = selected else {
        return Vec::new();
    };

    let normalized_target = l2_normalize(&selected_embedding);
    if normalized_target.is_empty() {
        return Vec::new();
    }

    let mut distances = scan
        .candidates
        .iter()
        .filter(|candidate| candidate.id != selected_id)
        .filter_map(|candidate| {
            let emb = candidate.embedding.as_ref()?;
            if emb.len() != normalized_target.len() || emb.is_empty() {
                return None;
            }
            let normalized = l2_normalize(emb);
            let similarity = embedding_cosine_similarity(&normalized_target, &normalized);
            Some((similarity, normalized))
        })
        .collect::<Vec<_>>();

    distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut gallery = vec![normalized_target];
    for (_, emb) in distances.into_iter().take(2) {
        gallery.push(emb);
    }
    gallery
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-8 {
        return Vec::new();
    }
    v.iter().map(|x| x / norm).collect()
}

fn embedding_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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

#[tauri::command]
pub fn validate_identity_review(
    state: State<'_, IdentityScanStore>,
    args: ValidateIdentityReviewArgs,
) -> Result<IdentityReviewResult, String> {
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
    rows.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
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
        &args.scan_id,
        limit,
        offset,
        action_contains,
        args.since_ms,
        args.until_ms,
        args.cursor_event_id,
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
    out_path.push(format!("bundle-{}.json", epoch_ms()));
    fs::write(&out_path, &json).map_err(|e| format!("failed to write diagnostics bundle: {e}"))?;
    let out_path_str = out_path.to_string_lossy().into_owned();
    let sha256 = diagnostics_hash_hex(&json);
    upsert_manifest_entry(&out_path_str, json.len() as u64, sha256)?;

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

    let manifest = load_diagnostics_manifest();
    let manifest_sha = manifest
        .entries
        .into_iter()
        .map(|entry| (entry.path, entry.sha256))
        .collect::<HashMap<String, String>>();

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
            sha256: manifest_sha.get(&path_str).cloned(),
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
pub fn prune_diagnostics_bundles(
    args: Option<PruneDiagnosticsBundlesArgs>,
) -> Result<PruneDiagnosticsBundlesResult, String> {
    let keep_latest = args.and_then(|a| a.keep_latest).unwrap_or(20);
    let list = list_diagnostics_bundles(Some(ListDiagnosticsBundlesArgs { limit: Some(1000) }))?;
    if list.bundles.len() <= keep_latest {
        return Ok(PruneDiagnosticsBundlesResult {
            deleted: 0,
            kept: list.bundles.len(),
        });
    }

    let mut deleted = 0usize;
    for bundle in list.bundles.iter().skip(keep_latest) {
        if fs::remove_file(&bundle.path).is_ok() {
            let _ = remove_manifest_entry(&bundle.path);
            deleted += 1;
        }
    }
    let kept = list.bundles.len().saturating_sub(deleted);
    Ok(PruneDiagnosticsBundlesResult { deleted, kept })
}

#[tauri::command]
pub fn read_diagnostics_bundle(
    args: ReadDiagnosticsBundleArgs,
) -> Result<ReadDiagnosticsBundleResult, String> {
    let max_bytes = args
        .max_bytes
        .unwrap_or(256 * 1024)
        .clamp(1024, 2 * 1024 * 1024);
    let path = PathBuf::from(&args.path);
    let diagnostics_dir = diagnostics_dir_path();
    if !path.starts_with(&diagnostics_dir) {
        return Err("path must be inside .focus-lock/diagnostics".to_string());
    }
    let bytes = fs::read(&path).map_err(|e| format!("failed to read bundle: {e}"))?;
    let truncated = bytes.len() > max_bytes;
    let shown = if truncated {
        &bytes[..max_bytes]
    } else {
        &bytes[..]
    };
    let content = String::from_utf8_lossy(shown).to_string();
    Ok(ReadDiagnosticsBundleResult {
        path: path.to_string_lossy().into_owned(),
        bytes: bytes.len(),
        content,
        truncated,
    })
}

#[tauri::command]
pub fn verify_diagnostics_bundle(
    args: VerifyDiagnosticsBundleArgs,
) -> Result<VerifyDiagnosticsBundleResult, String> {
    let path = PathBuf::from(&args.path);
    let diagnostics_dir = diagnostics_dir_path();
    if !path.starts_with(&diagnostics_dir) {
        return Err("path must be inside diagnostics directory".to_string());
    }
    let bytes = fs::read(&path).map_err(|e| format!("failed to read diagnostics bundle: {e}"))?;
    let actual_sha256 = diagnostics_hash_hex(&bytes);
    let path_str = path.to_string_lossy().into_owned();
    let expected_sha256 = manifest_sha_for(&path_str);
    let matches = expected_sha256
        .as_ref()
        .is_some_and(|expected| expected == &actual_sha256);
    Ok(VerifyDiagnosticsBundleResult {
        path: path_str,
        expected_sha256,
        actual_sha256,
        matches,
    })
}

#[tauri::command]
pub fn delete_diagnostics_bundle(
    args: DeleteDiagnosticsBundleArgs,
) -> Result<DeleteDiagnosticsBundleResult, String> {
    let path = PathBuf::from(&args.path);
    let diagnostics_dir = diagnostics_dir_path();
    if !path.starts_with(&diagnostics_dir) {
        return Err("path must be inside diagnostics directory".to_string());
    }
    if !path.exists() {
        let _ = remove_manifest_entry(&args.path);
        return Ok(DeleteDiagnosticsBundleResult { deleted: false });
    }
    fs::remove_file(&path).map_err(|e| format!("failed to delete diagnostics bundle: {e}"))?;
    let path_str = path.to_string_lossy().into_owned();
    let _ = remove_manifest_entry(&path_str);
    Ok(DeleteDiagnosticsBundleResult { deleted: true })
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
            tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
        }

        if let Ok(mut s) = worker_arc.lock() {
            s.running = false;
            s.stop_requested = false;
        }
    });

    storage_worker_status(state)
}

#[tauri::command]
pub async fn queue_health(state: State<'_, QueueStore>) -> Result<queue::QueueHealth, String> {
    let config = {
        let lock = state.0.lock().map_err(|e| e.to_string())?;
        lock.config.clone()
    };
    if config.sqs_enabled {
        queue::sqs_health(&config).await
    } else {
        let lock = state.0.lock().map_err(|e| e.to_string())?;
        Ok(lock.health())
    }
}

#[tauri::command]
pub async fn enqueue_discovery_job(
    state: State<'_, QueueStore>,
    args: EnqueueDiscoveryJobArgs,
) -> Result<queue::QueueEnqueueResult, String> {
    let config = {
        let lock = state.0.lock().map_err(|e| e.to_string())?;
        lock.config.clone()
    };
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
        face_model: args.face_model,
        expected_member_count: args.expected_member_count,
        processing_mode: sanitize_processing_mode(args.processing_mode.as_deref()),
    };

    if config.sqs_enabled {
        queue::sqs_enqueue_discovery(&config, payload, idempotency_key).await
    } else {
        let mut lock = state.0.lock().map_err(|e| e.to_string())?;
        lock.enqueue_discovery(payload, idempotency_key)
    }
}

#[tauri::command]
pub async fn enqueue_split_rescan_job(
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
        (
            scan.video.clone(),
            scan.yolo_model.clone(),
            scan.face_model.clone(),
            scan.pending_split_ids.clone(),
            scan.selected_identity_id,
            Some(scan.processing_mode.clone()),
        )
    };

    if scan_snapshot.3.is_empty() {
        return Err("no pending split identities to rescan".to_string());
    }

    let config = {
        let q = queue_state.0.lock().map_err(|e| e.to_string())?;
        q.config.clone()
    };

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
        face_model: scan_snapshot.2,
        split_identity_ids: scan_snapshot.3,
        processing_mode: sanitize_processing_mode(args.processing_mode.as_deref())
            .or(scan_snapshot.5),
    };

    let enqueue_result = if config.sqs_enabled {
        queue::sqs_enqueue_rescan(&config, payload, idempotency_key).await?
    } else {
        let mut q = queue_state.0.lock().map_err(|e| e.to_string())?;
        q.enqueue_rescan(payload, idempotency_key)?
    };

    if let Ok(mut scans) = scan_state.0.lock() {
        ensure_scan_store_loaded(&mut scans);
        if let Some(scan) = scans.scans.get_mut(&scan_id) {
            set_scan_status(scan, ScanSessionStatus::Proposed);
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
        max_attempts_before_dlq,
        args.client_run_id,
    )
    .await
}

async fn process_next_discovery_job_core(
    app: Option<AppHandle>,
    queue_store: std::sync::Arc<std::sync::Mutex<queue::QueueRuntime>>,
    scan_store: std::sync::Arc<std::sync::Mutex<IdentityScanState>>,
    max_attempts_before_dlq: u32,
    client_run_id: Option<String>,
) -> Result<queue::QueueProcessResult, String> {
    let max_attempts_before_dlq = max_attempts_before_dlq.max(1);

    let config = {
        let queue = queue_store.lock().map_err(|e| e.to_string())?;
        queue.config.clone()
    };

    if config.sqs_enabled {
        let Some(msg) = queue::sqs_receive_discovery(&config).await? else {
            return Ok(queue::QueueProcessResult {
                processed: false,
                queue: config.discovery_queue.clone(),
                message_id: None,
                job_id: None,
                moved_to_dlq: false,
                requeued: false,
                attempt: None,
                error: None,
                remaining_depth: queue::sqs_health(&config).await?.depths.discovery,
            });
        };

        let envelope = msg.envelope;
        let payload = envelope.payload.clone();
        let yolo_model = payload.yolo_model.clone();
        let face_model = payload.face_model.clone();
        let app_for_scan = app.clone();
        let queue_run_id = client_run_id.clone();
        let run_result = task::spawn_blocking(move || {
            run_identity_scan_for_queue(
                IdentityScanArgs {
                    video: payload.video,
                    yolo_model: payload.yolo_model,
                    face_model: payload.face_model,
                    expected_member_count: payload.expected_member_count,
                    processing_mode: sanitize_processing_mode(payload.processing_mode.as_deref()),
                    client_run_id: queue_run_id,
                },
                app_for_scan,
                "queued discovery",
            )
        })
        .await
        .map_err(|e| e.to_string())?;

        match run_result {
            Ok(scan_result) => {
                queue::sqs_ack(&msg.queue_url, &msg.receipt_handle).await?;
                let snapshot = {
                    let mut scans = scan_store.lock().map_err(|e| e.to_string())?;
                    ensure_scan_store_loaded(&mut scans);
                    upsert_scan_cache(
                        &mut scans.scans,
                        &envelope.payload.scan_id,
                        &scan_result,
                        &yolo_model,
                        &face_model,
                    );
                    snapshot_scan_entry(&scans, &envelope.payload.scan_id)?
                };
                if let Some(snapshot) = snapshot.as_ref() {
                    persist_scan_entry_snapshot(snapshot)?;
                }
                let health = queue::sqs_health(&config).await?;
                return Ok(queue::QueueProcessResult {
                    processed: true,
                    queue: config.discovery_queue.clone(),
                    message_id: Some(envelope.message_id),
                    job_id: Some(envelope.job_id),
                    moved_to_dlq: false,
                    requeued: false,
                    attempt: Some(envelope.attempt),
                    error: None,
                    remaining_depth: health.depths.discovery,
                });
            }
            Err(err) => {
                let (moved_to_dlq, requeued) = queue::sqs_retry_or_dlq(
                    &config,
                    &msg.queue_url,
                    &msg.receipt_handle,
                    envelope.clone(),
                    max_attempts_before_dlq,
                )
                .await?;
                let health = queue::sqs_health(&config).await?;
                return Ok(queue::QueueProcessResult {
                    processed: true,
                    queue: config.discovery_queue.clone(),
                    message_id: Some(envelope.message_id),
                    job_id: Some(envelope.job_id),
                    moved_to_dlq,
                    requeued,
                    attempt: Some(envelope.attempt),
                    error: Some(err),
                    remaining_depth: health.depths.discovery,
                });
            }
        }
    }

    let dequeued = {
        let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
        match queue.dequeue_discovery() {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                return Ok(queue::QueueProcessResult {
                    processed: false,
                    queue: queue.config.discovery_queue.clone(),
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
                    processed: false,
                    queue: queue.config.discovery_queue.clone(),
                    message_id: None,
                    job_id: None,
                    moved_to_dlq: false,
                    requeued: false,
                    attempt: None,
                    error: Some(err),
                    remaining_depth: queue.health().depths.discovery,
                });
            }
        }
    };

    let envelope = dequeued.envelope;
    let payload = envelope.payload.clone();
    let yolo_model = payload.yolo_model.clone();
    let face_model = payload.face_model.clone();
    let app_for_scan = app.clone();
    let queue_run_id = client_run_id.clone();
    let run_result = task::spawn_blocking(move || {
        run_identity_scan_for_queue(
            IdentityScanArgs {
                video: payload.video,
                yolo_model: payload.yolo_model,
                face_model: payload.face_model,
                expected_member_count: payload.expected_member_count,
                processing_mode: sanitize_processing_mode(payload.processing_mode.as_deref()),
                client_run_id: queue_run_id,
            },
            app_for_scan,
            "queued discovery",
        )
    })
    .await
    .map_err(|e| e.to_string())?;

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
                    &face_model,
                );
                snapshot_scan_entry(&scans, &payload.scan_id)?
            };
            if let Some(snapshot) = snapshot.as_ref() {
                persist_scan_entry_snapshot(snapshot)?;
            }

            let queue = queue_store.lock().map_err(|e| e.to_string())?;
            Ok(queue::QueueProcessResult {
                processed: true,
                queue: queue.config.discovery_queue.clone(),
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
            let mut moved_to_dlq = false;
            let mut requeued = false;
            if envelope.attempt + 1 >= max_attempts_before_dlq {
                queue.move_discovery_to_dlq(dequeued.raw);
                moved_to_dlq = true;
            } else {
                queue.requeue_discovery_retry(envelope.clone())?;
                requeued = true;
            }
            Ok(queue::QueueProcessResult {
                processed: true,
                queue: queue.config.discovery_queue.clone(),
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
        max_attempts_before_dlq,
        args.client_run_id,
    )
    .await
}

async fn process_next_rescan_job_core(
    app: Option<AppHandle>,
    queue_store: std::sync::Arc<std::sync::Mutex<queue::QueueRuntime>>,
    scan_store: std::sync::Arc<std::sync::Mutex<IdentityScanState>>,
    max_attempts_before_dlq: u32,
    client_run_id: Option<String>,
) -> Result<queue::QueueProcessResult, String> {
    let max_attempts_before_dlq = max_attempts_before_dlq.max(1);
    let config = {
        let queue = queue_store.lock().map_err(|e| e.to_string())?;
        queue.config.clone()
    };

    if config.sqs_enabled {
        let Some(msg) = queue::sqs_receive_rescan(&config).await? else {
            return Ok(queue::QueueProcessResult {
                processed: false,
                queue: config.rescan_queue.clone(),
                message_id: None,
                job_id: None,
                moved_to_dlq: false,
                requeued: false,
                attempt: None,
                error: None,
                remaining_depth: queue::sqs_health(&config).await?.depths.rescan,
            });
        };

        let envelope = msg.envelope;
        let payload = envelope.payload.clone();
        let yolo_model = payload.yolo_model.clone();
        let face_model = payload.face_model.clone();
        let video = payload.video.clone();
        let app_for_scan = app.clone();
        let queue_run_id = client_run_id.clone();
        let run_result = task::spawn_blocking(move || {
            run_identity_scan_for_queue(
                IdentityScanArgs {
                    video,
                    yolo_model,
                    face_model,
                    expected_member_count: None,
                    processing_mode: sanitize_processing_mode(payload.processing_mode.as_deref()),
                    client_run_id: queue_run_id,
                },
                app_for_scan,
                "queued rescan",
            )
        })
        .await
        .map_err(|e| e.to_string())?;

        match run_result {
            Ok(scan_result) => {
                queue::sqs_ack(&msg.queue_url, &msg.receipt_handle).await?;
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
                        set_scan_status(scan, ScanSessionStatus::Proposed);
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
                let health = queue::sqs_health(&config).await?;
                Ok(queue::QueueProcessResult {
                    processed: true,
                    queue: config.rescan_queue.clone(),
                    message_id: Some(envelope.message_id),
                    job_id: Some(envelope.job_id),
                    moved_to_dlq: false,
                    requeued: false,
                    attempt: Some(envelope.attempt),
                    error: None,
                    remaining_depth: health.depths.rescan,
                })
            }
            Err(err) => {
                let (moved_to_dlq, requeued) = queue::sqs_retry_or_dlq_rescan(
                    &config,
                    &msg.queue_url,
                    &msg.receipt_handle,
                    envelope.clone(),
                    max_attempts_before_dlq,
                )
                .await?;
                let health = queue::sqs_health(&config).await?;
                Ok(queue::QueueProcessResult {
                    processed: true,
                    queue: config.rescan_queue.clone(),
                    message_id: Some(envelope.message_id),
                    job_id: Some(envelope.job_id),
                    moved_to_dlq,
                    requeued,
                    attempt: Some(envelope.attempt),
                    error: Some(err),
                    remaining_depth: health.depths.rescan,
                })
            }
        }
    } else {
        let dequeued = {
            let mut queue = queue_store.lock().map_err(|e| e.to_string())?;
            match queue.dequeue_rescan() {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    return Ok(queue::QueueProcessResult {
                        processed: false,
                        queue: queue.config.rescan_queue.clone(),
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
                        processed: false,
                        queue: queue.config.rescan_queue.clone(),
                        message_id: None,
                        job_id: None,
                        moved_to_dlq: false,
                        requeued: false,
                        attempt: None,
                        error: Some(err),
                        remaining_depth: queue.health().depths.rescan,
                    });
                }
            }
        };

        let envelope = dequeued.envelope;
        let payload = envelope.payload.clone();
        let app_for_scan = app.clone();
        let queue_run_id = client_run_id.clone();
        let run_result = task::spawn_blocking(move || {
            run_identity_scan_for_queue(
                IdentityScanArgs {
                    video: payload.video,
                    yolo_model: payload.yolo_model,
                    face_model: payload.face_model,
                    expected_member_count: None,
                    processing_mode: sanitize_processing_mode(payload.processing_mode.as_deref()),
                    client_run_id: queue_run_id,
                },
                app_for_scan,
                "queued rescan",
            )
        });
        let run_result = run_result.await.map_err(|e| e.to_string())?;

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
                        set_scan_status(scan, ScanSessionStatus::Proposed);
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
                let queue = queue_store.lock().map_err(|e| e.to_string())?;
                Ok(queue::QueueProcessResult {
                    processed: true,
                    queue: queue.config.rescan_queue.clone(),
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
                let mut moved_to_dlq = false;
                let mut requeued = false;
                if envelope.attempt + 1 >= max_attempts_before_dlq {
                    queue.move_rescan_to_dlq(dequeued.raw);
                    moved_to_dlq = true;
                } else {
                    queue.requeue_rescan_retry(envelope.clone())?;
                    requeued = true;
                }
                Ok(queue::QueueProcessResult {
                    processed: true,
                    queue: queue.config.rescan_queue.clone(),
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
}

#[tauri::command]
pub fn queue_worker_start(
    app: AppHandle,
    queue_state: State<'_, QueueStore>,
    scan_state: State<'_, IdentityScanStore>,
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
    let worker_arc = worker_state.0.clone();
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

            tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;
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
    if let Ok(custom) = std::env::var("FOCUS_LOCK_DIAGNOSTICS_DIR") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let mut path = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    path.push(".focus-lock");
    path.push("diagnostics");
    path
}

fn file_modified_ms(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DiagnosticsManifestEntry {
    path: String,
    bytes: u64,
    sha256: String,
    created_at_ms: u64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct DiagnosticsManifest {
    entries: Vec<DiagnosticsManifestEntry>,
}

fn diagnostics_manifest_path() -> PathBuf {
    let mut path = diagnostics_dir_path();
    path.push("manifest.json");
    path
}

fn diagnostics_hash_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn load_diagnostics_manifest() -> DiagnosticsManifest {
    let path = diagnostics_manifest_path();
    let Ok(bytes) = fs::read(path) else {
        return DiagnosticsManifest::default();
    };
    serde_json::from_slice::<DiagnosticsManifest>(&bytes).unwrap_or_default()
}

fn save_diagnostics_manifest(manifest: &DiagnosticsManifest) -> Result<(), String> {
    let path = diagnostics_manifest_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed to create diagnostics dir: {e}"))?;
    }
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| format!("failed to serialize diagnostics manifest: {e}"))?;
    fs::write(path, bytes).map_err(|e| format!("failed to write diagnostics manifest: {e}"))
}

fn upsert_manifest_entry(path: &str, bytes: u64, sha256: String) -> Result<(), String> {
    let mut manifest = load_diagnostics_manifest();
    let now = epoch_ms();
    if let Some(entry) = manifest.entries.iter_mut().find(|e| e.path == path) {
        entry.bytes = bytes;
        entry.sha256 = sha256;
        entry.created_at_ms = now;
    } else {
        manifest.entries.push(DiagnosticsManifestEntry {
            path: path.to_string(),
            bytes,
            sha256,
            created_at_ms: now,
        });
    }
    save_diagnostics_manifest(&manifest)
}

fn remove_manifest_entry(path: &str) -> Result<(), String> {
    let mut manifest = load_diagnostics_manifest();
    manifest.entries.retain(|e| e.path != path);
    save_diagnostics_manifest(&manifest)
}

fn manifest_sha_for(path: &str) -> Option<String> {
    let manifest = load_diagnostics_manifest();
    manifest
        .entries
        .into_iter()
        .find(|e| e.path == path)
        .map(|e| e.sha256)
}

#[tauri::command]
pub async fn queue_peek_discovery_attempts(
    queue_state: State<'_, QueueStore>,
    args: Option<QueuePeekArgs>,
) -> Result<QueuePeekResult, String> {
    let limit = args.and_then(|a| a.limit).unwrap_or(10);
    let config = {
        let queue = queue_state.0.lock().map_err(|e| e.to_string())?;
        queue.config.clone()
    };
    if config.sqs_enabled {
        return Ok(QueuePeekResult {
            attempts: Vec::new(),
        });
    }
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
pub async fn cancel_scan(state: State<'_, ScanCancelFlag>) -> Result<(), String> {
    state.0.store(true, Ordering::Relaxed);
    Ok(())
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
    {
        let mut job = render_state.0.lock().map_err(|e| e.to_string())?;
        if job.running {
            return Ok(JobResult {
                ok: false,
                message: if job.cancelling {
                    "render cancellation is in progress; wait for stop to finish".to_string()
                } else {
                    "a render job is already running".to_string()
                },
                output_path: None,
                run_id: Some(render_run_id.clone()),
            });
        }
        job.running = true;
        job.cancelling = false;
    }

    let scan_id_for_state = args.scan_id.clone();

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
                if let Ok(mut job) = render_state.0.lock() {
                    job.running = false;
                    job.cancelling = false;
                }
                return Ok(JobResult {
                    ok: false,
                    message: "identity validation session not found; rerun scan".to_string(),
                    output_path: None,
                    run_id: Some(render_run_id.clone()),
                });
            };

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
                if let Ok(mut job) = render_state.0.lock() {
                    job.running = false;
                    job.cancelling = false;
                }
                return Ok(JobResult {
                    ok: false,
                    message: why,
                    output_path: None,
                    run_id: Some(render_run_id.clone()),
                });
            }

            apply_review_to_scan(scan, &review, args.threshold);

            args.threshold = scan
                .validated_threshold
                .unwrap_or(args.threshold)
                .clamp(0.0, 1.0);
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
                        args.body_reid_model = resolve_default_body_reid_model(&scan.face_model);
                    }
                }
            }

            set_scan_status(scan, ScanSessionStatus::Tracking);
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
        if let Some(snapshot) = snapshot.as_ref() {
            if let Err(err) = persist_scan_entry_snapshot(snapshot) {
                if let Ok(mut job) = render_state.0.lock() {
                    job.running = false;
                    job.cancelling = false;
                }
                return Err(err);
            }
        }
    }

    if let Err(message) = validate_fancam_paths(&args) {
        if let Ok(mut job) = render_state.0.lock() {
            job.running = false;
            job.cancelling = false;
        }
        return Ok(JobResult {
            ok: false,
            message,
            output_path: None,
            run_id: Some(render_run_id.clone()),
        });
    }

    {
        state.0.store(false, Ordering::Relaxed);
    }

    let cancel = Arc::clone(&state.0);
    let app2 = app.clone();

    let result = match task::spawn_blocking(move || run_pipeline(app2, cancel, args)).await {
        Ok(result) => result,
        Err(e) => {
            if let Ok(mut job) = render_state.0.lock() {
                job.running = false;
                job.cancelling = false;
            }
            return Err(e.to_string());
        }
    };

    if let Ok(mut job) = render_state.0.lock() {
        job.running = false;
        job.cancelling = false;
    }

    match result {
        Ok(path) => {
            if let Some(scan_id) = &scan_id_for_state {
                if let Ok(mut lock) = scan_state.0.lock() {
                    ensure_scan_store_loaded(&mut lock);
                    let snapshot = if let Some(scan) = lock.scans.get_mut(scan_id) {
                        set_scan_status(scan, ScanSessionStatus::Completed);
                        scan.updated_at_ms = epoch_ms();
                        append_scan_event(scan, "tracking_completed", format!("output={path}"));
                        snapshot_scan_entry(&lock, scan_id).ok().flatten()
                    } else {
                        None
                    };
                    if let Some(snapshot) = snapshot.as_ref() {
                        let _ = persist_scan_entry_snapshot(snapshot);
                    }
                }
            }
            Ok(JobResult {
                ok: true,
                message: "Done".into(),
                output_path: Some(path),
                run_id: Some(render_run_id),
            })
        }
        Err(e) => {
            if let Some(scan_id) = &scan_id_for_state {
                if let Ok(mut lock) = scan_state.0.lock() {
                    ensure_scan_store_loaded(&mut lock);
                    let snapshot = if let Some(scan) = lock.scans.get_mut(scan_id) {
                        set_scan_status(scan, ScanSessionStatus::Failed);
                        scan.updated_at_ms = epoch_ms();
                        append_scan_event(scan, "tracking_failed", e.to_string());
                        snapshot_scan_entry(&lock, scan_id).ok().flatten()
                    } else {
                        None
                    };
                    if let Some(snapshot) = snapshot.as_ref() {
                        let _ = persist_scan_entry_snapshot(snapshot);
                    }
                }
            }
            Ok(JobResult {
                ok: false,
                message: e.to_string(),
                output_path: None,
                run_id: Some(render_run_id),
            })
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
    let face_model = PathBuf::from(args.face_model.trim());

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
    } else if args
        .target_embeddings
        .as_ref()
        .is_some_and(|gallery| gallery.iter().any(|row| !row.is_empty()))
        && resolve_default_body_reid_model(&args.face_model).is_none()
    {
        return Err(
            "body reid model not found: expected models/osnet_x0_25_msmt17.onnx for identity-locked runs"
                .to_string(),
        );
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

    Ok(())
}

fn validate_identity_scan_paths(args: &IdentityScanArgs) -> Result<(), String> {
    let video_path = PathBuf::from(args.video.trim());
    let yolo_model = PathBuf::from(args.yolo_model.trim());
    let face_model = PathBuf::from(args.face_model.trim());

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
        .clone()
        .or_else(|| resolve_default_body_reid_model(&args.face_model));

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

    let pipeline = if !target_gallery.is_empty() {
        Pipeline::load_with_hint_galleries(
            &args.yolo_model,
            &args.face_model,
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
            &args.face_model,
            embedding.clone(),
            threshold,
            initial_hint,
            mode,
        )?
    } else {
        Pipeline::load_with_hint_mode(
            &args.yolo_model,
            &args.face_model,
            &args.bias,
            threshold,
            initial_hint,
            mode,
        )?
    };
    let (mut analyzer, mut renderer) = pipeline.into_parts();

    let cancel_analyze = Arc::clone(&cancel);
    let cancel_render = Arc::clone(&cancel);
    let progress_run_id = run_id.clone();
    let mut last_progress_emit = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);

    transcode_with_progress_staged_mode(
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
                return;
            }
            renderer.render(frame, camera);
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
                current as f64 / total as f64
            } else {
                0.0
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

    let _ = app.emit(
        "fancam://done",
        JobResult {
            ok: true,
            message: "Done".into(),
            output_path: Some(args.output.clone()),
            run_id: Some(run_id),
        },
    );

    Ok(args.output)
}

fn sanitize_processing_mode(value: Option<&str>) -> Option<String> {
    value.map(|v| processing_mode_string(Some(v)))
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{Mutex, OnceLock},
    };

    use super::{
        DeleteDiagnosticsBundleArgs, FancamArgs, ListDiagnosticsBundlesArgs,
        PruneDiagnosticsBundlesArgs, QueryIdentityScansArgs, QueryScanEventsArgs,
        ReadDiagnosticsBundleArgs, ScanSessionStatus, VerifyDiagnosticsBundleArgs,
        can_transition_status, delete_diagnostics_bundle, diagnostics_hash_hex,
        diagnostics_manifest_path, list_diagnostics_bundles, manifest_sha_for,
        prune_diagnostics_bundles, query_identity_scans, query_scan_events,
        read_diagnostics_bundle, validate_fancam_paths, verify_diagnostics_bundle,
    };
    use crate::storage;

    fn diagnostics_test_mutex() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_diagnostics_test_dir<T>(f: impl FnOnce(PathBuf) -> T) -> T {
        let _guard = diagnostics_test_mutex()
            .lock()
            .expect("test mutex poisoned");
        let mut dir = std::env::temp_dir();
        dir.push(format!("focus-lock-diag-test-{}", super::epoch_ms()));
        let _ = std::fs::create_dir_all(&dir);
        let prev = std::env::var("FOCUS_LOCK_DIAGNOSTICS_DIR").ok();
        unsafe { std::env::set_var("FOCUS_LOCK_DIAGNOSTICS_DIR", &dir) };
        let result = f(dir.clone());
        if let Some(value) = prev {
            unsafe { std::env::set_var("FOCUS_LOCK_DIAGNOSTICS_DIR", value) };
        } else {
            unsafe { std::env::remove_var("FOCUS_LOCK_DIAGNOSTICS_DIR") };
        }
        let _ = std::fs::remove_dir_all(dir);
        result
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
    fn allows_core_lifecycle_transitions() {
        assert!(can_transition_status(
            &ScanSessionStatus::Proposed,
            &ScanSessionStatus::Validated
        ));
        assert!(can_transition_status(
            &ScanSessionStatus::Validated,
            &ScanSessionStatus::Tracking
        ));
        assert!(can_transition_status(
            &ScanSessionStatus::Tracking,
            &ScanSessionStatus::Completed
        ));
    }

    #[test]
    fn blocks_invalid_lifecycle_transitions() {
        assert!(!can_transition_status(
            &ScanSessionStatus::Proposed,
            &ScanSessionStatus::Tracking
        ));
        assert!(!can_transition_status(
            &ScanSessionStatus::Completed,
            &ScanSessionStatus::Tracking
        ));
    }

    #[test]
    fn diagnostics_read_rejects_outside_path() {
        let mut outside = std::env::temp_dir();
        outside.push("focus-lock-outside-bundle.json");
        let _ = std::fs::write(&outside, b"{}");

        let result = read_diagnostics_bundle(ReadDiagnosticsBundleArgs {
            path: outside.to_string_lossy().into_owned(),
            max_bytes: Some(1024),
        });
        assert!(result.is_err());

        let _ = std::fs::remove_file(PathBuf::from(&outside));
    }

    #[test]
    fn diagnostics_manifest_list_and_verify() {
        with_diagnostics_test_dir(|dir| {
            let path = dir.join("bundle-alpha.json");
            let bytes = br#"{"ok":true}"#;
            std::fs::write(&path, bytes).expect("write bundle");
            let path_str = path.to_string_lossy().into_owned();
            let sha = diagnostics_hash_hex(bytes);
            super::upsert_manifest_entry(&path_str, bytes.len() as u64, sha.clone())
                .expect("upsert manifest");

            let list =
                list_diagnostics_bundles(Some(ListDiagnosticsBundlesArgs { limit: Some(5) }))
                    .expect("list bundles");
            assert_eq!(list.bundles.len(), 1);
            assert_eq!(list.bundles[0].path, path_str);
            assert_eq!(list.bundles[0].sha256.as_deref(), Some(sha.as_str()));

            let verify = verify_diagnostics_bundle(VerifyDiagnosticsBundleArgs {
                path: path.to_string_lossy().into_owned(),
            })
            .expect("verify bundle");
            assert!(verify.matches);
            assert_eq!(verify.expected_sha256.as_deref(), Some(sha.as_str()));

            std::fs::write(&path, br#"{"ok":false}"#).expect("rewrite bundle");
            let mismatch = verify_diagnostics_bundle(VerifyDiagnosticsBundleArgs {
                path: path.to_string_lossy().into_owned(),
            })
            .expect("verify mismatched bundle");
            assert!(!mismatch.matches);
            assert!(mismatch.expected_sha256.is_some());
        });
    }

    #[test]
    fn diagnostics_delete_removes_manifest_entry() {
        with_diagnostics_test_dir(|dir| {
            let path = dir.join("bundle-gamma.json");
            let bytes = br#"{"hello":"world"}"#;
            std::fs::write(&path, bytes).expect("write bundle");
            let path_str = path.to_string_lossy().into_owned();
            let sha = diagnostics_hash_hex(bytes);
            super::upsert_manifest_entry(&path_str, bytes.len() as u64, sha).expect("upsert");

            let result = delete_diagnostics_bundle(DeleteDiagnosticsBundleArgs {
                path: path.to_string_lossy().into_owned(),
            })
            .expect("delete bundle");
            assert!(result.deleted);
            assert!(manifest_sha_for(&path_str).is_none());

            let manifest_path = diagnostics_manifest_path();
            assert!(manifest_path.exists());
        });
    }

    #[test]
    fn diagnostics_prune_removes_manifest_for_deleted_bundles() {
        with_diagnostics_test_dir(|dir| {
            let older = dir.join("bundle-older.json");
            let newer = dir.join("bundle-newer.json");
            let older_bytes = br#"{"bundle":"older"}"#;
            let newer_bytes = br#"{"bundle":"newer"}"#;
            std::fs::write(&older, older_bytes).expect("write older bundle");
            std::thread::sleep(std::time::Duration::from_millis(2));
            std::fs::write(&newer, newer_bytes).expect("write newer bundle");

            let older_path = older.to_string_lossy().into_owned();
            let newer_path = newer.to_string_lossy().into_owned();
            super::upsert_manifest_entry(
                &older_path,
                older_bytes.len() as u64,
                diagnostics_hash_hex(older_bytes),
            )
            .expect("upsert older");
            super::upsert_manifest_entry(
                &newer_path,
                newer_bytes.len() as u64,
                diagnostics_hash_hex(newer_bytes),
            )
            .expect("upsert newer");

            let pruned = prune_diagnostics_bundles(Some(PruneDiagnosticsBundlesArgs {
                keep_latest: Some(1),
            }))
            .expect("prune bundles");
            assert_eq!(pruned.deleted, 1);
            assert_eq!(pruned.kept, 1);

            assert!(manifest_sha_for(&older_path).is_none());
            assert!(manifest_sha_for(&newer_path).is_some());
        });
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
                        face_model: "f.onnx".to_string(),
                        status: "validated".to_string(),
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
                        face_model: "f.onnx".to_string(),
                        status: "proposed".to_string(),
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
                        face_model: "f.onnx".to_string(),
                        status: "failed".to_string(),
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
                    face_model: "f.onnx".to_string(),
                    status: "tracking".to_string(),
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
                scan_id: None,
                selected_identity_id: None,
                target_anchor_x: None,
                target_anchor_y: None,
            };
            let result = validate_fancam_paths(&args);
            assert!(result.is_ok());
        });
    }
}
