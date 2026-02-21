use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fancam_core::{
    pipeline::Pipeline,
    runtime::configure_ort_dylib,
    video::{total_frames, transcode_with_progress_staged},
};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tokio::task;

use crate::CancelFlag;

// ─── DTO types ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FancamArgs {
    pub video: String,
    pub bias: String,
    pub output: String,
    pub yolo_model: String,
    pub face_model: String,
    pub threshold: f32,
}

#[derive(Debug, Serialize, Clone)]
pub struct ProgressPayload {
    pub current: u64,
    pub total: u64,
    pub fraction: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct JobResult {
    pub ok: bool,
    pub message: String,
    pub output_path: Option<String>,
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

#[tauri::command]
pub async fn cancel_job(state: State<'_, CancelFlag>) -> Result<(), String> {
    let mut flag = state.0.lock().map_err(|e| e.to_string())?;
    *flag = true;
    Ok(())
}

#[tauri::command]
pub async fn run_fancam(
    app: AppHandle,
    state: State<'_, CancelFlag>,
    args: FancamArgs,
) -> Result<JobResult, String> {
    {
        let mut flag = state.0.lock().map_err(|e| e.to_string())?;
        *flag = false;
    }

    let cancel = Arc::clone(&state.0);
    let app2 = app.clone();

    let result = task::spawn_blocking(move || run_pipeline(app2, cancel, args))
        .await
        .map_err(|e| e.to_string())?;

    match result {
        Ok(path) => Ok(JobResult {
            ok: true,
            message: "Done".into(),
            output_path: Some(path),
        }),
        Err(e) => Ok(JobResult {
            ok: false,
            message: e.to_string(),
            output_path: None,
        }),
    }
}

// ─── Pipeline (blocking) ─────────────────────────────────────────────────────

fn run_pipeline(
    app: AppHandle,
    cancel: Arc<Mutex<bool>>,
    args: FancamArgs,
) -> anyhow::Result<String> {
    let video_path = PathBuf::from(&args.video);
    let output_path = PathBuf::from(&args.output);
    let total = total_frames(&video_path);
    let threshold = args.threshold.clamp(0.0, 1.0);

    configure_ort_dylib();

    let pipeline = Pipeline::load(&args.yolo_model, &args.face_model, &args.bias, threshold)?;
    let (mut analyzer, mut renderer) = pipeline.into_parts();

    let cancel_analyze = Arc::clone(&cancel);
    let cancel_render = Arc::clone(&cancel);

    transcode_with_progress_staged(
        video_path,
        &output_path,
        total,
        move |frame| {
            if cancel_analyze.lock().map(|g| *g).unwrap_or(false) {
                None
            } else {
                analyzer.analyze(frame)
            }
        },
        move |frame, camera| {
            if cancel_render.lock().map(|g| *g).unwrap_or(false) {
                return;
            }
            renderer.render(frame, camera);
        },
        |current, total| {
            let fraction = if total > 0 {
                current as f64 / total as f64
            } else {
                0.0
            };
            let _ = app.emit(
                "fancam://progress",
                ProgressPayload {
                    current,
                    total,
                    fraction,
                },
            );
        },
    )?;

    let _ = app.emit(
        "fancam://done",
        JobResult {
            ok: true,
            message: "Done".into(),
            output_path: Some(args.output.clone()),
        },
    );

    Ok(args.output)
}
