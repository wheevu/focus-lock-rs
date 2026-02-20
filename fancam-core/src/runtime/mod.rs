use std::path::{Path, PathBuf};

/// Resolve and set ORT_DYLIB_PATH at runtime when it is missing or invalid.
///
/// Priority order:
/// 1) Existing ORT_DYLIB_PATH (if file exists)
/// 2) models/onnxruntime*/lib/libonnxruntime.dylib near current exe/cwd
/// 3) /opt/homebrew/lib/libonnxruntime.dylib (last-resort fallback)
pub fn configure_ort_dylib() {
    if let Some(existing) = std::env::var_os("ORT_DYLIB_PATH") {
        let existing_path = PathBuf::from(existing);
        if existing_path.is_file() {
            tracing::info!(path = %existing_path.display(), "using ORT_DYLIB_PATH from environment");
            return;
        }
        tracing::warn!(
            path = %existing_path.display(),
            "ORT_DYLIB_PATH is set but file does not exist; attempting auto-discovery"
        );
    }

    for candidate in ort_candidates() {
        if candidate.is_file() {
            // SAFETY: this is called before any ORT sessions are created and
            // from the single job-start thread, so no concurrent env mutation.
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &candidate);
            }
            tracing::info!(path = %candidate.display(), "configured ORT_DYLIB_PATH");
            return;
        }
    }

    tracing::warn!(
        "could not locate libonnxruntime.dylib; set ORT_DYLIB_PATH to an official ONNX Runtime build with CoreML support"
    );
}

fn ort_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }

    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        for _ in 0..7 {
            let Some(d) = dir else {
                break;
            };
            roots.push(d.clone());
            dir = d.parent().map(Path::to_path_buf);
        }
    }

    let mut candidates = Vec::new();
    for root in roots {
        candidates.push(root.join("models/onnxruntime/lib/libonnxruntime.dylib"));
        candidates.push(root.join("models/onnxruntime-osx-arm64/lib/libonnxruntime.dylib"));
        candidates.push(root.join("models/libonnxruntime.dylib"));
    }

    candidates.push(PathBuf::from("/opt/homebrew/lib/libonnxruntime.dylib"));
    candidates
}
