pub mod commands;
pub mod queue;
pub mod storage;

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, atomic::AtomicBool},
};

use tauri::Manager;
use tokio::sync::Notify;

/// Cancel flag shared with the active fancam job.
pub struct CancelFlag(pub Arc<AtomicBool>);

#[derive(Debug, Default)]
pub struct RenderJobState {
    pub running: bool,
    pub cancelling: bool,
}

#[derive(Debug, Default)]
pub struct RenderJobStore(pub Arc<Mutex<RenderJobState>>);

impl Default for CancelFlag {
    fn default() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
}

#[derive(Debug, Default)]
pub struct ScanJobState {
    pub running: bool,
    pub cancelling: bool,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Default)]
pub struct ScanJobStore(pub Arc<Mutex<ScanJobState>>);

#[derive(Debug, Default)]
pub struct IdentityScanState {
    pub next_id: u64,
    pub scans: HashMap<String, commands::IdentityScanCache>,
    pub loaded_from_disk: bool,
}

#[derive(Default)]
pub struct IdentityScanStore(pub Arc<Mutex<IdentityScanState>>);

pub struct QueueStore(pub Arc<Mutex<queue::QueueRuntime>>);

impl Default for QueueStore {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(queue::QueueRuntime::new())))
    }
}

#[derive(Debug)]
pub struct QueueWorkerState {
    pub running: bool,
    pub stop_requested: bool,
    pub poll_interval_ms: u64,
    pub processed_total: u64,
    pub max_attempts_before_dlq: u32,
    pub last_error: Option<String>,
    pub recent_events: VecDeque<QueueWorkerEvent>,
    pub max_events: usize,
}

#[derive(Debug, Clone)]
pub struct QueueWorkerEvent {
    pub at_ms: u64,
    pub queue: String,
    pub message_id: Option<String>,
    pub job_id: Option<String>,
    pub attempt: Option<u32>,
    pub moved_to_dlq: bool,
    pub requeued: bool,
    pub error: Option<String>,
}

pub struct QueueWorkerStore(pub Arc<Mutex<QueueWorkerState>>, pub Arc<Notify>);

impl Default for QueueWorkerStore {
    fn default() -> Self {
        Self(
            Arc::new(Mutex::new(QueueWorkerState::default())),
            Arc::new(Notify::new()),
        )
    }
}

#[derive(Debug)]
pub struct StorageWorkerState {
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

pub struct StorageWorkerStore(pub Arc<Mutex<StorageWorkerState>>, pub Arc<Notify>);

impl Default for StorageWorkerStore {
    fn default() -> Self {
        Self(
            Arc::new(Mutex::new(StorageWorkerState::default())),
            Arc::new(Notify::new()),
        )
    }
}

impl Default for QueueWorkerState {
    fn default() -> Self {
        Self {
            running: false,
            stop_requested: false,
            poll_interval_ms: 1200,
            processed_total: 0,
            max_attempts_before_dlq: 3,
            last_error: None,
            recent_events: VecDeque::new(),
            max_events: 40,
        }
    }
}

impl Default for StorageWorkerState {
    fn default() -> Self {
        Self {
            running: false,
            stop_requested: false,
            poll_interval_ms: 300_000,
            max_session_age_ms: 7 * 86_400_000,
            max_events_per_scan: 120,
            vacuum: false,
            runs_total: 0,
            last_run_ms: None,
            last_error: None,
        }
    }
}

pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let result = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir().map_err(std::io::Error::other)?;
            storage::initialize_app_data_dir(data_dir).map_err(std::io::Error::other)?;
            Ok(())
        })
        .manage(CancelFlag::default())
        .manage(RenderJobStore::default())
        .manage(ScanJobStore::default())
        .manage(IdentityScanStore::default())
        .manage(QueueStore::default())
        .manage(QueueWorkerStore::default())
        .manage(StorageWorkerStore::default())
        .invoke_handler(tauri::generate_handler![
            commands::model_dir,
            commands::probe_video,
            commands::read_thumbnail,
            commands::scan_identities,
            commands::validate_identity_review,
            commands::list_identity_scans,
            commands::get_identity_scan,
            commands::cleanup_identity_scans,
            commands::query_identity_scans,
            commands::query_scan_events,
            commands::scan_storage_stats,
            commands::run_scan_storage_maintenance,
            commands::export_diagnostics_bundle,
            commands::list_diagnostics_bundles,
            commands::storage_worker_start,
            commands::storage_worker_stop,
            commands::storage_worker_status,
            commands::queue_health,
            commands::enqueue_discovery_job,
            commands::enqueue_split_rescan_job,
            commands::process_next_discovery_job,
            commands::process_next_rescan_job,
            commands::queue_peek_discovery_attempts,
            commands::queue_worker_start,
            commands::queue_worker_stop,
            commands::queue_worker_status,
            commands::queue_worker_clear_events,
            commands::cancel_job,
            commands::cancel_scan,
            commands::run_fancam,
        ])
        .run(tauri::generate_context!());
    if let Err(error) = result {
        eprintln!("error while running focus-lock: {error}");
    }
}
