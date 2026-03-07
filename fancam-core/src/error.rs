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

    /// Error opening or reading a video file
    #[error("Video error for {path}: {source}")]
    Video {
        /// Path to the video file
        path: PathBuf,
        /// Source error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error during video decoding
    #[error("Decode error: {0}")]
    Decode(String),

    /// Error during video encoding
    #[error("Encode error: {0}")]
    Encode(String),

    /// Error during person detection
    #[error("Detection failed: {0}")]
    Detection(String),

    /// Error during face identification
    #[error("Face identification failed: {0}")]
    FaceIdentification(String),

    /// Target identity not found in frame
    #[error("Identity not found")]
    IdentityNotFound,

    /// Invalid frame dimensions or format
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    /// Error during image processing
    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    /// Thread communication error
    #[error("Channel closed: {0}")]
    ChannelClosed(String),

    /// Thread panic or join error
    #[error("Thread error: {0}")]
    Thread(String),

    /// Lock poisoned (mutex/rwlock)
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

    /// Catch-all for unexpected errors
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

    /// Create a video error
    pub fn video<E>(path: impl Into<PathBuf>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Video {
            path: path.into(),
            source: Box::new(source),
        }
    }

    /// Create a decode error
    pub fn decode(msg: impl Into<String>) -> Self {
        Self::Decode(msg.into())
    }

    /// Create an encode error
    pub fn encode(msg: impl Into<String>) -> Self {
        Self::Encode(msg.into())
    }

    /// Create an inference error
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a detection error
    pub fn detection(msg: impl Into<String>) -> Self {
        Self::Detection(msg.into())
    }

    /// Create a face identification error
    pub fn face_id(msg: impl Into<String>) -> Self {
        Self::FaceIdentification(msg.into())
    }

    /// Create an image processing error
    pub fn image_processing(msg: impl Into<String>) -> Self {
        Self::ImageProcessing(msg.into())
    }

    /// Create an invalid frame error
    pub fn invalid_frame(msg: impl Into<String>) -> Self {
        Self::InvalidFrame(msg.into())
    }

    /// Create a channel closed error
    pub fn channel_closed(msg: impl Into<String>) -> Self {
        Self::ChannelClosed(msg.into())
    }

    /// Create a thread error
    pub fn thread(msg: impl Into<String>) -> Self {
        Self::Thread(msg.into())
    }

    /// Create a lock poisoned error
    pub fn lock_poisoned(msg: impl Into<String>) -> Self {
        Self::LockPoisoned(msg.into())
    }

    /// Create an ORT config error
    pub fn ort_config(msg: impl Into<String>) -> Self {
        Self::OrtConfig(msg.into())
    }

    /// Create an invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create an unexpected error
    pub fn unexpected(msg: impl Into<String>) -> Self {
        Self::Unexpected(msg.into())
    }
}

/// Result type alias for fancam-core
pub type Result<T> = std::result::Result<T, FancamError>;

/// Convert anyhow::Error to FancamError
impl From<anyhow::Error> for FancamError {
    fn from(err: anyhow::Error) -> Self {
        Self::Unexpected(err.to_string())
    }
}

/// Helper trait for converting poison errors
pub trait PoisonExt<T> {
    /// Convert a poison error to FancamError
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
        let err = FancamError::decode("test error");
        assert_eq!(err.to_string(), "Decode error: test error");
    }

    #[test]
    fn test_model_load_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = FancamError::model_load("/path/to/model.onnx", io_err);
        assert!(err.to_string().contains("Failed to load model"));
        assert!(err.to_string().contains("/path/to/model.onnx"));
    }
}
