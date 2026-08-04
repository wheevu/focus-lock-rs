//! camera — offline camera path planning primitives.

use crate::tracking::{CameraSource, CameraState};
use crate::tracklet::Tracklet;
use std::cmp::Ordering;
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
        keyframes.sort_unstable_by(|a, b| {
            a.frame_index
                .cmp(&b.frame_index)
                .then_with(|| compare_camera_state(&a.state, &b.state))
        });
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
        if self
            .last_observed_frame
            .is_some_and(|last_frame| frame_index < last_frame)
        {
            self.next_index = 0;
            self.last_observed_state = None;
            self.last_observed_frame = None;
        }

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
            if !obs.bbox.is_valid() || !obs.observation.bbox.is_valid() {
                continue;
            }
            let score = obs.observation.composite_score();
            if !score.is_finite() {
                continue;
            }
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
                score,
            ));
        }
    }

    if scored_keyframes.is_empty() {
        return CameraPath::default();
    }

    scored_keyframes.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.2.total_cmp(&a.2))
            .then_with(|| compare_camera_state(&a.1, &b.1))
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

fn compare_camera_state(a: &CameraState, b: &CameraState) -> Ordering {
    a.cx.total_cmp(&b.cx)
        .then_with(|| a.cy.total_cmp(&b.cy))
        .then_with(|| a.half_size.total_cmp(&b.half_size))
        .then_with(|| camera_source_rank(a.source).cmp(&camera_source_rank(b.source)))
        .then_with(|| a.miss_count.cmp(&b.miss_count))
}

fn camera_source_rank(source: CameraSource) -> u8 {
    match source {
        CameraSource::Observed | CameraSource::Manual => 0,
        CameraSource::Predicted => 1,
        CameraSource::Held => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::IdentityObservation;

    fn bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> crate::detection::BBox {
        crate::detection::BBox {
            x1,
            y1,
            x2,
            y2,
            confidence: 0.9,
        }
    }

    fn state(frame_index: u64, cx: f32) -> CameraKeyframe {
        CameraKeyframe {
            frame_index,
            state: CameraState {
                cx,
                cy: 100.0,
                half_size: 40.0,
                source: CameraSource::Observed,
                miss_count: 0,
            },
        }
    }

    fn tracklet_with_observations(
        id: usize,
        observations: &[(u64, crate::detection::BBox, f32)],
    ) -> Tracklet {
        let mut tracklet = Tracklet::new(id);
        for &(frame_index, bbox, similarity) in observations {
            tracklet.push(
                frame_index,
                bbox,
                IdentityObservation::from_face_scores(bbox, similarity, 0.0, similarity, None),
            );
        }
        tracklet
    }

    #[test]
    fn cursor_rewinds_when_request_precedes_last_consumed_keyframe() {
        let mut cursor = CameraCursor::from_path(CameraPath {
            keyframes: vec![state(10, 10.0), state(20, 20.0)],
        });

        let at_20 = cursor.camera_for_frame(20).expect("frame 20 state");
        assert_eq!(at_20.cx, 20.0);

        let at_10 = cursor.camera_for_frame(10).expect("rewound frame 10 state");
        assert_eq!(at_10.cx, 10.0);
        assert_eq!(at_10.source, CameraSource::Observed);
        assert_eq!(at_10.miss_count, 0);

        let held_at_15 = cursor.camera_for_frame(15).expect("held frame 15 state");
        assert_eq!(held_at_15.cx, 10.0);
        assert_eq!(held_at_15.source, CameraSource::Held);
        assert_eq!(held_at_15.miss_count, 5);
    }

    #[test]
    fn planning_filters_invalid_observations_and_selects_equal_score_deterministically() {
        let left = bbox(10.0, 20.0, 50.0, 100.0);
        let right = bbox(100.0, 20.0, 140.0, 100.0);
        let invalid_geometry = bbox(80.0, 20.0, 70.0, 100.0);

        let first = tracklet_with_observations(
            1,
            &[
                (5, invalid_geometry, 0.95),
                (10, left, 0.8),
                (11, left, f32::NAN),
            ],
        );
        let second = tracklet_with_observations(2, &[(10, right, 0.8)]);

        let first_path =
            plan_camera_for_identity(&[first.clone(), second.clone()], &[(1, 1), (2, 1)], 1);
        let reversed_path = plan_camera_for_identity(&[second, first], &[(2, 1), (1, 1)], 1);

        assert_eq!(first_path.keyframes.len(), 1);
        assert_eq!(first_path.keyframes[0].frame_index, 10);
        assert_eq!(first_path.keyframes[0].state.cx, 30.0);
        assert_eq!(
            first_path.keyframes[0].state.cx,
            reversed_path.keyframes[0].state.cx
        );
    }

    #[test]
    fn cursor_orders_duplicate_frame_keyframes_deterministically() {
        let mut cursor = CameraCursor::from_path(CameraPath {
            keyframes: vec![state(10, 20.0), state(10, 10.0)],
        });

        let selected = cursor.camera_for_frame(10).expect("duplicate frame state");
        assert_eq!(selected.cx, 20.0);
    }
}
