pub mod commands;

use std::sync::{Arc, Mutex};

/// Cancel flag shared with the active fancam job.
#[derive(Default)]
pub struct CancelFlag(pub Arc<Mutex<bool>>);

pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .manage(CancelFlag::default())
        .invoke_handler(tauri::generate_handler![
            commands::model_dir,
            commands::probe_video,
            commands::cancel_job,
            commands::run_fancam,
        ])
        .run(tauri::generate_context!())
        .expect("error while running focus-lock");
}
