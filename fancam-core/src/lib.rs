pub mod detection;
pub mod pipeline;
pub mod rendering;
pub mod runtime;
pub mod tracking;
pub mod video;

// Re-export the top-level pipeline error type so callers only need `fancam_core::Error`
pub use anyhow::Error;
pub use anyhow::Result;
