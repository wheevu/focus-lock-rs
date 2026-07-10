//! camera — offline camera path planning primitives.

use crate::tracking::{CameraSource, CameraState};
use crate::tracklet::Tracklet;
use std::collections::HashMap;

/// Camera state at a specific frame index.
#[derive(Debug, Clone, Copy)]
pub struct CameraKeyframe {
    /// Frame index in source timeline.
    pub frame_index: u64,
    /// Camera state planned for this frame.
    pub state: CameraState,
}

/// Planned camera path for an offline render.
#[derive(Debug, Clone, Default)]
pub struct CameraPath {
    /// Keyframes sorted by frame index.
    pub keyframes: Vec<CameraKeyframe>,
}

impl CameraPath {
    /// Returns true when no keyframes are present.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

/// Stateful cursor that emits camera states for sequential frame indices.
#[derive(Debug, Clone)]
pub struct CameraCursor {
    keyframes: Vec<CameraKeyframe>,
    next_index: usize,
    last_observed_state: Option<CameraState>,
    last_observed_frame: Option<u64>,
}

impl CameraCursor {
    /// Build a cursor from an offline camera path.
    #[must_use]
    pub fn from_path(path: CameraPath) -> Self {
        let mut keyframes = path.keyframes;
        keyframes.sort_unstable_by_key(|keyframe| keyframe.frame_index);
        Self {
            keyframes,
            next_index: 0,
            last_observed_state: None,
            last_observed_frame: None,
        }
    }

    /// Emit camera state for a specific frame index.
    ///
    /// Returns `None` before the first keyframe.
    #[must_use]
    pub fn camera_for_frame(&mut self, frame_index: u64) -> Option<CameraState> {
        while self
            .keyframes
            .get(self.next_index)
            .is_some_and(|keyframe| keyframe.frame_index <= frame_index)
        {
            let keyframe = self.keyframes[self.next_index];
            self.last_observed_state = Some(keyframe.state);
            self.last_observed_frame = Some(keyframe.frame_index);
            self.next_index = self.next_index.saturating_add(1);
        }

        let state = self.last_observed_state?;
        let Some(last_frame) = self.last_observed_frame else {
            return Some(state);
        };
        if frame_index <= last_frame {
            return Some(state);
        }

        Some(CameraState {
            source: CameraSource::Held,
            miss_count: (frame_index - last_frame).min(u64::from(u32::MAX)) as u32,
            ..state
        })
    }
}

/// Build camera path for one clustered identity from tracklets and assignments.
#[must_use]
pub fn plan_camera_for_identity(
    tracklets: &[Tracklet],
    assignments: &[(usize, usize)],
    identity_id: usize,
) -> CameraPath {
    let by_id = tracklets
        .iter()
        .map(|tracklet| (tracklet.id, tracklet))
        .collect::<HashMap<_, _>>();

    let mut scored_keyframes = Vec::<(u64, CameraState, f32)>::new();
    for (tracklet_id, assigned_identity) in assignments {
        if *assigned_identity != identity_id {
            continue;
        }
        let Some(tracklet) = by_id.get(tracklet_id) else {
            continue;
        };
        for obs in &tracklet.observations {
            let hs = (obs.bbox.width().max(obs.bbox.height()) * 0.5).max(1.0);
            scored_keyframes.push((
                obs.frame_index,
                CameraState {
                    cx: obs.bbox.center_x(),
                    cy: obs.bbox.center_y(),
                    half_size: hs,
                    source: CameraSource::Observed,
                    miss_count: 0,
                },
                obs.observation.composite_score(),
            ));
        }
    }

    if scored_keyframes.is_empty() {
        return CameraPath::default();
    }

    scored_keyframes.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Keep only the best-scoring state per frame.
    let mut keyframes = Vec::<CameraKeyframe>::new();
    for (frame_index, state, _) in scored_keyframes {
        if keyframes
            .last()
            .is_some_and(|keyframe| keyframe.frame_index == frame_index)
        {
            continue;
        }
        keyframes.push(CameraKeyframe { frame_index, state });
    }

    // Smooth neighboring keyframes to reduce jitter.
    let mut smoothed = Vec::<CameraKeyframe>::with_capacity(keyframes.len());
    for keyframe in keyframes {
        if let Some(previous) = smoothed.last().copied() {
            let gap = keyframe.frame_index.saturating_sub(previous.frame_index);
            if gap <= 12 {
                smoothed.push(CameraKeyframe {
                    frame_index: keyframe.frame_index,
                    state: CameraState {
                        cx: previous.state.cx.mul_add(0.68, keyframe.state.cx * 0.32),
                        cy: previous.state.cy.mul_add(0.68, keyframe.state.cy * 0.32),
                        half_size: previous
                            .state
                            .half_size
                            .mul_add(0.74, keyframe.state.half_size * 0.26),
                        source: CameraSource::Observed,
                        miss_count: 0,
                    },
                });
            } else {
                smoothed.push(keyframe);
            }
        } else {
            smoothed.push(keyframe);
        }
    }

    // Densify short gaps with linear interpolation.
    let mut keyframes = Vec::new();
    if let Some(first) = smoothed.first().copied() {
        keyframes.push(first);
    }
    for window in smoothed.windows(2) {
        let [left, right] = [window[0], window[1]];
        let gap = right.frame_index.saturating_sub(left.frame_index);
        if gap > 1 && gap <= 12 {
            for offset in 1..gap {
                let t = offset as f32 / gap as f32;
                keyframes.push(CameraKeyframe {
                    frame_index: left.frame_index + offset,
                    state: CameraState {
                        cx: (right.state.cx - left.state.cx).mul_add(t, left.state.cx),
                        cy: (right.state.cy - left.state.cy).mul_add(t, left.state.cy),
                        half_size: (right.state.half_size - left.state.half_size)
                            .mul_add(t, left.state.half_size),
                        source: CameraSource::Predicted,
                        miss_count: offset.min(u64::from(u32::MAX)) as u32,
                    },
                });
            }
        }
        keyframes.push(right);
    }

    keyframes.sort_unstable_by_key(|k| k.frame_index);
    CameraPath { keyframes }
}
