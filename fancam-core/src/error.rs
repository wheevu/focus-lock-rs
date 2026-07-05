//! Error types for fancam-core
//!
//! This module provides domain-specific error types for programmatic error handling.
//! All errors can be converted to `anyhow::Error` for convenience.

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for fancam-core operations.
///
/// This enum covers all error cases that can occur during video processing,
/// model inference, and pipeline execution.
#[derive(Error, Debug)]
pub enum FancamError {
    /// Error loading an ML model (YOLOv8, ArcFace, etc.)
    #[error("Failed to load model at {path}: {source}")]
    ModelLoad {
        /// Path to the model file
        path: PathBuf,
        /// Source error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error during model inference
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Error during face identification
    #[error("Face identification failed: {0}")]
    FaceIdentification(String),

    /// Invalid frame dimensions or format
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    /// Error during image processing
    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    /// Lock poisoned (mutex)
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    /// ONNX Runtime configuration error
    #[error("ORT configuration error: {0}")]
    OrtConfig(String),

    /// Invalid configuration or parameters
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Catch-all for unexpected errors (used by `From<anyhow::Error>`)
    #[error("Unexpected error: {0}")]
    Unexpected(String),
}

impl FancamError {
    /// Create a model load error
    pub fn model_load<E>(path: impl Into<PathBuf>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ModelLoad {
            path: path.into(),
            source: Box::new(source),
        }
    }

    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    pub fn face_id(msg: impl Into<String>) -> Self {
        Self::FaceIdentification(msg.into())
    }

    pub fn image_processing(msg: impl Into<String>) -> Self {
        Self::ImageProcessing(msg.into())
    }

    pub fn invalid_frame(msg: impl Into<String>) -> Self {
        Self::InvalidFrame(msg.into())
    }

    pub fn lock_poisoned(msg: impl Into<String>) -> Self {
        Self::LockPoisoned(msg.into())
    }

    pub fn ort_config(msg: impl Into<String>) -> Self {
        Self::OrtConfig(msg.into())
    }

    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }
}

/// Result type alias for fancam-core
pub type Result<T> = std::result::Result<T, FancamError>;

/// Convert `anyhow::Error` to `FancamError`
impl From<anyhow::Error> for FancamError {
    fn from(err: anyhow::Error) -> Self {
        Self::Unexpected(err.to_string())
    }
}

/// Helper trait for converting poison errors
pub trait PoisonExt<T> {
    fn to_fancam_err(self, context: &str) -> Result<T>;
}

impl<T, E> PoisonExt<T> for std::result::Result<T, std::sync::PoisonError<E>> {
    fn to_fancam_err(self, context: &str) -> Result<T> {
        self.map_err(|_| FancamError::lock_poisoned(context.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FancamError::inference("test error");
        assert_eq!(err.to_string(), "Inference failed: test error");
    }

    #[test]
    fn test_model_load_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = FancamError::model_load("/path/to/model.onnx", io_err);
        assert!(err.to_string().contains("Failed to load model"));
        assert!(err.to_string().contains("/path/to/model.onnx"));
    }
}
