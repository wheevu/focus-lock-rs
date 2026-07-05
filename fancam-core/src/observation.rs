//! observation — identity cue observations and scoring primitives.
//!
//! This module introduces architecture-neutral identity observations used by
//! offline two-pass solving and online preview tracking.

use crate::detection::BBox;

/// Type of identity cue contributing to an observation score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueType {
    /// Face embedding similarity.
    Face,
    /// Body re-identification similarity.
    Body,
}

/// Weighted score contribution for one cue type.
#[derive(Debug, Clone, Copy)]
pub struct CueScore {
    /// Cue kind.
    pub cue: CueType,
    /// Cue similarity normalized to [-1, 1] or [0, 1] depending on model.
    pub similarity: f32,
    /// Weight used when fusing this cue with others.
    pub weight: f32,
}

/// Generic identity observation for one detected person candidate.
///
/// `FaceObservation` in legacy online flow maps directly onto this type.
#[derive(Debug, Clone)]
pub struct IdentityObservation {
    /// Candidate detection box in source-frame coordinates.
    pub bbox: BBox,
    /// Best positive gallery similarity.
    pub similarity: f32,
    /// Best negative/impostor gallery similarity.
    pub impostor_similarity: f32,
    /// Positive-vs-negative margin.
    pub margin: f32,
    /// Optional body similarity scored against target body gallery.
    pub body_similarity: Option<f32>,
    /// Additional cue contributions used by offline/global solvers.
    pub cues: Vec<CueScore>,
}

impl IdentityObservation {
    /// Convenience constructor for face-first observations.
    #[must_use]
    pub fn from_face_scores(
        bbox: BBox,
        similarity: f32,
        impostor_similarity: f32,
        margin: f32,
        body_similarity: Option<f32>,
    ) -> Self {
        let mut cues = vec![CueScore {
            cue: CueType::Face,
            similarity,
            weight: 1.0,
        }];
        if let Some(body) = body_similarity {
            cues.push(CueScore {
                cue: CueType::Body,
                similarity: body,
                weight: 0.12,
            });
        }

        Self {
            bbox,
            similarity,
            impostor_similarity,
            margin,
            body_similarity,
            cues,
        }
    }

    /// Composite score used for coarse candidate ranking.
    #[must_use]
    pub fn composite_score(&self) -> f32 {
        let cue_score = self
            .cues
            .iter()
            .map(|cue| cue.similarity * cue.weight)
            .sum::<f32>();
        cue_score.mul_add(0.10, self.similarity.mul_add(0.70, self.margin * 0.20))
    }

    /// Ensure cue list contains updated face/body entries.
    pub fn sync_default_cues(&mut self) {
        self.cues.clear();
        self.cues.push(CueScore {
            cue: CueType::Face,
            similarity: self.similarity,
            weight: 1.0,
        });
        if let Some(body) = self.body_similarity {
            self.cues.push(CueScore {
                cue: CueType::Body,
                similarity: body,
                weight: 0.12,
            });
        }
    }
}
