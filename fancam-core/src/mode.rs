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
    /// Return the canonical string form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
        }
    }
}

impl std::str::FromStr for ProcessingMode {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fast" | "faster" => Ok(Self::Fast),
            "balanced" | "balance" | "normal" => Ok(Self::Balanced),
            "quality" | "hq" | "thorough" => Ok(Self::Quality),
            _ => Err(()),
        }
    }
}
