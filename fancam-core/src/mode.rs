//! Runtime processing mode controls.
//!
//! These modes tune discovery and rendering/export behavior for either
//! fast local iteration or higher quality output.

/// Processing mode used across discovery and render/export paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    /// Prioritize iteration speed (default).
    Fast,
    /// Balanced mode between speed and quality.
    Balanced,
    /// Prioritize output quality over throughput.
    Quality,
}

impl ProcessingMode {
    /// Parse from a user-supplied string.
    #[must_use]
    pub fn from_str(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "quality" | "hq" => Self::Quality,
            "balanced" | "balance" | "normal" => Self::Balanced,
            _ => Self::Fast,
        }
    }

    /// Canonical string form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
        }
    }
}

impl Default for ProcessingMode {
    fn default() -> Self {
        Self::Fast
    }
}
