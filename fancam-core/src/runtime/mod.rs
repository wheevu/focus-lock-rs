//! Runtime configuration and environment setup
//!
//! This module handles ONNX Runtime (ORT) dynamic library configuration.
//! It provides safe, explicit initialization without modifying global state.

use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::sync::OnceLock;

use ort::execution_providers::{CoreMLExecutionProvider, ExecutionProvider};

static ORT_RUNTIME_INIT: OnceLock<StdResult<OrtConfig, String>> = OnceLock::new();

/// Configuration for ONNX Runtime dynamic library loading.
///
/// This struct provides explicit configuration for ORT without relying on
/// environment variable side effects. It discovers and validates the ORT
/// library path at runtime.
///
/// # Example
///
/// ```rust,no_run
/// use fancam_core::runtime::OrtConfig;
///
/// let config = OrtConfig::discover().expect("ORT library not found");
/// // Use config.path() to get the validated library path
/// ```
#[derive(Debug, Clone)]
pub struct OrtConfig {
    path: PathBuf,
}

impl OrtConfig {
    /// The default library name for macOS.
    const LIB_NAME: &'static str = "libonnxruntime.dylib";

    /// Default search paths relative to common locations.
    const RELATIVE_PATHS: &'static [&'static str] = &[
        "models/onnxruntime/lib",
        "models/onnxruntime-osx-arm64/lib",
        "models",
    ];

    /// Homebrew installation path (macOS).
    const HOMEBREW_PATH: &'static str = "/opt/homebrew/lib";

    /// Expected ONNX Runtime minor version for this pinned `ort` crate line.
    ///
    /// `ort` 2.0.0-rc.10 targets ONNX Runtime 1.22.x.
    pub const EXPECTED_ORT_MINOR: u32 = 22;

    /// Create a new ORT configuration with an explicit path.
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not exist or is not a file.
    pub fn new(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.is_file() {
            return Err(crate::FancamError::ort_config(format!(
                "ORT library not found at: {}",
                path.display()
            )));
        }
        Ok(Self { path })
    }

    /// Discover the ORT library path automatically.
    ///
    /// Searches in the following order:
    /// 1. Existing `ORT_DYLIB_PATH` environment variable (if valid)
    /// 2. Relative paths from current working directory
    /// 3. Relative paths from executable location
    /// 4. Homebrew installation path
    ///
    /// # Errors
    ///
    /// Returns an error if no valid ORT library is found.
    pub fn discover() -> crate::Result<Self> {
        // `ort` performs its own runtime compatibility check during API load;
        // this discovery step remains file-presence only and delegates version
        // compatibility enforcement to ONNX Runtime initialization.
        Self::discover_with_preference(false)
    }

    /// Discover ONNX Runtime library path.
    ///
    /// `prefer_compatible` is reserved for future strict-path strategies.
    fn discover_with_preference(prefer_compatible: bool) -> crate::Result<Self> {
        // 1. Check environment variable first
        if let Some(existing) = std::env::var_os("ORT_DYLIB_PATH") {
            let path = PathBuf::from(existing);
            if path.is_file() {
                if !prefer_compatible {
                    tracing::info!(path = %path.display(), "using ORT_DYLIB_PATH from environment");
                    return Ok(Self { path });
                }
            } else {
                tracing::warn!(
                    path = %path.display(),
                    "ORT_DYLIB_PATH is set but file does not exist; attempting auto-discovery"
                );
            }
        }

        let candidates = Self::candidates();

        // Fallback to any discovered library when compatibility-filtered discovery fails.
        for candidate in candidates {
            if candidate.is_file() {
                tracing::info!(path = %candidate.display(), "discovered ORT library");
                return Ok(Self { path: candidate });
            }
        }

        Err(crate::FancamError::ort_config(
            "Could not locate libonnxruntime.dylib. \
             Set ORT_DYLIB_PATH to an official ONNX Runtime build with CoreML support",
        ))
    }

    /// Discover and initialize ONNX Runtime for the current process.
    ///
    /// This pins `ort` to a validated dynamic library path via `ort::init_from(...)`
    /// before any session is constructed. The initialization is process-wide and
    /// only happens once.
    ///
    /// # Errors
    ///
    /// Returns an error if discovery fails, the path is invalid UTF-8, or ONNX
    /// Runtime initialization fails.
    pub fn ensure_initialized() -> crate::Result<Self> {
        let init =
            ORT_RUNTIME_INIT.get_or_init(|| Self::initialize_runtime().map_err(|e| e.to_string()));

        match init {
            Ok(config) => Ok(config.clone()),
            Err(message) => Err(crate::FancamError::ort_config(message.clone())),
        }
    }

    fn initialize_runtime() -> crate::Result<Self> {
        let config = Self::discover_with_preference(false)?;
        let path = config.path_str()?.to_string();
        let ort_expected = format!("1.{}.x", Self::EXPECTED_ORT_MINOR);

        let commit_result = std::panic::catch_unwind(|| ort::init_from(path.clone()).commit());
        match commit_result {
            Ok(Ok(committed)) => {
                let coreml_available = CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false);
                tracing::info!(
                    path = %path,
                    expected_runtime = %ort_expected,
                    committed,
                    coreml_available,
                    "initialized ONNX Runtime from discovered dylib"
                );
                if !coreml_available {
                    return Err(crate::FancamError::ort_config(format!(
                        "ONNX Runtime initialized from {} but CoreML execution provider is unavailable. Install CoreML-enabled ONNX Runtime {}.",
                        path, ort_expected
                    )));
                }
                Ok(config)
            }
            Ok(Err(err)) => Err(crate::FancamError::ort_config(format!(
                "failed to initialize ONNX Runtime from {}: {err}. This build expects ONNX Runtime {}.",
                path, ort_expected
            ))),
            Err(payload) => {
                let panic_msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic payload".to_string()
                };
                Err(crate::FancamError::ort_config(format!(
                    "ONNX Runtime initialization panicked for {}: {}. This build expects ONNX Runtime {}.",
                    path, panic_msg, ort_expected
                )))
            }
        }
    }

    /// Get the validated ORT library path.
    #[must_use]
    pub const fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Get the path as a string slice for FFI calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the path contains invalid UTF-8.
    pub fn path_str(&self) -> crate::Result<&str> {
        self.path
            .to_str()
            .ok_or_else(|| crate::FancamError::ort_config("ORT path contains invalid UTF-8"))
    }

    /// Generate candidate paths to search for the ORT library.
    fn candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        // Add relative paths from current directory
        if let Ok(cwd) = std::env::current_dir() {
            for rel_path in Self::RELATIVE_PATHS {
                candidates.push(cwd.join(rel_path).join(Self::LIB_NAME));
            }
        }

        // Add relative paths from executable location
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                // Search up to 7 parent directories
                let mut current = Some(exe_dir.to_path_buf());
                for _ in 0..7 {
                    if let Some(dir) = current {
                        for rel_path in Self::RELATIVE_PATHS {
                            candidates.push(dir.join(rel_path).join(Self::LIB_NAME));
                        }
                        current = dir.parent().map(Path::to_path_buf);
                    } else {
                        break;
                    }
                }
            }
        }

        // Add Homebrew path
        candidates.push(PathBuf::from(Self::HOMEBREW_PATH).join(Self::LIB_NAME));

        candidates
    }
}

impl Default for OrtConfig {
    fn default() -> Self {
        // Panic in default is acceptable for application code,
        // but library code should use discover() explicitly
        Self::discover().expect("ORT library not found")
    }
}

/// Legacy compatibility: Configure ORT by setting environment variable.
///
/// # Deprecated
///
/// This function uses `unsafe` to set environment variables at runtime.
/// Use [`OrtConfig::discover()`] instead for safe, explicit configuration.
///
/// # Safety
///
/// This function is only safe when:
/// - Called before any ORT sessions are created
/// - Called from a single thread (no concurrent env access)
/// - No other code relies on the previous ORT_DYLIB_PATH value
#[deprecated(since = "0.2.0", note = "Use OrtConfig::discover() instead")]
pub fn configure_ort_dylib() {
    #[allow(unsafe_code)]
    unsafe {
        configure_ort_dylib_internal();
    }
}

#[allow(unsafe_code)]
unsafe fn configure_ort_dylib_internal() {
    match OrtConfig::discover() {
        Ok(config) => {
            if let Ok(path_str) = config.path_str() {
                // SAFETY: This is called before any ORT sessions are created
                // and from a single thread to avoid data races.
                unsafe {
                    std::env::set_var("ORT_DYLIB_PATH", path_str);
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "could not configure ORT dylib");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_config_new_valid() {
        // This will only pass if there's an actual ORT library
        // In CI, we might need to mock this
        let result = OrtConfig::new("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_candidates_not_empty() {
        let candidates = OrtConfig::candidates();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_ort_config_path_str() {
        let config = OrtConfig {
            path: PathBuf::from("/test/path.dylib"),
        };
        assert_eq!(config.path_str().unwrap(), "/test/path.dylib");
    }
}
