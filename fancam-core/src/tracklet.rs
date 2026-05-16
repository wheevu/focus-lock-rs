//! tracklet — shared short-term identity-agnostic tracklet structures.

use crate::detection::BBox;
use crate::observation::IdentityObservation;

/// Observation attached to a tracklet at one frame.
#[derive(Debug, Clone)]
pub struct TrackletObservation {
    /// Source frame index.
    pub frame_index: u64,
    /// Detection box for this frame.
    pub bbox: BBox,
    /// Generic identity observation for this detection.
    pub observation: IdentityObservation,
}

/// Tracklet built from short-term motion association.
#[derive(Debug, Clone)]
pub struct Tracklet {
    /// Stable local tracklet id.
    pub id: usize,
    /// Contiguous observations associated with this tracklet.
    pub observations: Vec<TrackletObservation>,
}

impl Tracklet {
    /// Start a new tracklet.
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self {
            id,
            observations: Vec::new(),
        }
    }

    /// Append one frame observation to this tracklet.
    pub fn push(&mut self, frame_index: u64, bbox: BBox, observation: IdentityObservation) {
        self.observations.push(TrackletObservation {
            frame_index,
            bbox,
            observation,
        });
    }

    /// First frame observed in this tracklet.
    #[must_use]
    pub fn first_frame(&self) -> Option<u64> {
        self.observations.first().map(|obs| obs.frame_index)
    }

    /// Last frame observed in this tracklet.
    #[must_use]
    pub fn last_frame(&self) -> Option<u64> {
        self.observations.last().map(|obs| obs.frame_index)
    }

    /// Last observation in this tracklet.
    #[must_use]
    pub fn last_observation(&self) -> Option<&TrackletObservation> {
        self.observations.last()
    }

    /// Number of observations.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether this tracklet has no observations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Average composite observation score across this tracklet.
    #[must_use]
    pub fn average_composite_score(&self) -> f32 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let sum = self
            .observations
            .iter()
            .map(|obs| obs.observation.composite_score())
            .sum::<f32>();
        sum / self.observations.len() as f32
    }

    /// Best composite observation score in this tracklet.
    #[must_use]
    pub fn best_composite_score(&self) -> f32 {
        self.observations
            .iter()
            .map(|obs| obs.observation.composite_score())
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0)
    }
}
