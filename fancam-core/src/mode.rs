//! Runtime processing mode controls.
//!
//! These modes tune discovery and rendering/export behavior for either
//! fast local iteration or higher quality output.

/// Processing mode used across discovery and render/export paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessingMode {
    /// Prioritize iteration speed (default).
    #[default]
    Fast,
    /// Balanced mode between speed and quality.
    Balanced,
    /// Prioritize output quality over throughput.
    Quality,
}

impl ProcessingMode {
    /// Parse from a user-supplied string. Returns `None` for unknown values.
    #[must_use]
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fast" | "faster" => Some(Self::Fast),
            "balanced" | "balance" | "normal" => Some(Self::Balanced),
            "quality" | "hq" | "thorough" => Some(Self::Quality),
            _ => None,
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
