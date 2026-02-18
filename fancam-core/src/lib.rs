pub mod video;
pub mod detection;
pub mod tracking;
pub mod rendering;

// Re-export the top-level pipeline error type so callers only need `fancam_core::Error`
pub use anyhow::Error;
pub use anyhow::Result;
