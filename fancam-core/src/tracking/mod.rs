//! tracking — identity-aware camera smoothing and lock state machine.
//!
//! This module owns target lock state (`Searching/Locked/Recovering/Lost`) and
//! smooths accepted detections into camera coordinates.

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};

use crate::detection::{BBox, FaceObservation};

/// Process noise — how much we trust the motion model.
const PROCESS_NOISE: f32 = 4.0;
/// Measurement noise — how much we trust accepted identity observations.
const MEASUREMENT_NOISE: f32 = 16.0;

/// Frames between recognition attempts while searching.
const SEARCHING_RECOGNITION_STRIDE: u64 = 2;
/// Baseline recognition stride while locked.
const LOCKED_RECOGNITION_STRIDE: u64 = 2;
/// Maximum recognition stride when lock confidence is high.
const LOCKED_RECOGNITION_STRIDE_HIGH: u64 = 3;
/// Recognition stride while recovering or lost.
const RECOVERY_RECOGNITION_STRIDE: u64 = 1;

/// Max consecutive misses while recovering before entering lost state.
const MAX_RECOVERING_MISSES: u32 = 18;
/// Max consecutive misses while lost before dropping camera state entirely.
const MAX_LOST_MISSES: u32 = 90;
/// Maximum predicted frames to emit before freezing on last confirmed camera.
const MAX_PREDICTED_MISSES: u32 = 6;
/// Maximum frozen frames to emit before returning `None` while still lost.
const MAX_FROZEN_MISSES: u32 = 24;

/// IoU below this is treated as a hard jump candidate.
const HARD_JUMP_IOU: f32 = 0.03;
/// Minimum score gap required when a hard jump is proposed.
const HARD_JUMP_SCORE_GAP: f32 = 0.04;
/// Consecutive frames needed before accepting a weak hard jump.
const HARD_JUMP_CONFIRM_FRAMES: u32 = 2;
/// Contribution of optional body ReID score to candidate ranking.
const BODY_SIMILARITY_WEIGHT: f32 = 0.12;

/// Origin of the camera state for the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraSource {
    /// Camera derived from a confirmed identity observation in this frame.
    Observed,
    /// Camera derived from short-term motion prediction.
    Predicted,
    /// Camera held at the last confirmed lock to avoid drift while uncertain.
    Held,
}

/// A minimal 2D constant-velocity Kalman filter.
#[derive(Debug)]
struct Kalman2D {
    /// State: [cx, cy, vx, vy]
    x: Vector4<f32>,
    /// State covariance
    p: Matrix4<f32>,
    /// State transition matrix (F)
    f: Matrix4<f32>,
    /// Measurement matrix (H): extracts [cx, cy] from state
    h: Matrix2x4<f32>,
    /// Process noise covariance (Q)
    q: Matrix4<f32>,
    /// Measurement noise covariance (R)
    r: Matrix2<f32>,
}

impl Kalman2D {
    fn new(cx: f32, cy: f32) -> Self {
        let x = Vector4::new(cx, cy, 0.0, 0.0);
        let p = Matrix4::identity() * 100.0;
        let f = Matrix4::new(
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let h = Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let q = Matrix4::identity() * PROCESS_NOISE;
        let r = Matrix2::identity() * MEASUREMENT_NOISE;
        Self { x, p, f, h, q, r }
    }

    fn predict(&mut self) {
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f.transpose() + self.q;
    }

    fn update(&mut self, cx: f32, cy: f32) {
        let z = Vector2::new(cx, cy);
        let y = z - self.h * self.x;
        let s = self.h * self.p * self.h.transpose() + self.r;
        let Some(s_inv) = s.try_inverse() else {
            return;
        };
        let k: Matrix4x2<f32> = self.p * self.h.transpose() * s_inv;
        self.x += k * y;
        self.p = (Matrix4::identity() - k * self.h) * self.p;
    }

    fn cx(&self) -> f32 {
        self.x[0]
    }

    fn cy(&self) -> f32 {
        self.x[1]
    }
}

/// Camera output state in source-frame coordinates.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    /// Center X coordinate of the crop window.
    pub cx: f32,
    /// Center Y coordinate of the crop window.
    pub cy: f32,
    /// Smoothed half-size of the tracked person box.
    pub half_size: f32,
    /// Origin of this camera state.
    pub source: CameraSource,
    /// Consecutive misses since last confirmed identity observation.
    pub miss_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct PendingHardJump {
    bbox: BBox,
    votes: u32,
}

/// Identity lock state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    /// No stable lock yet.
    Searching,
    /// Stable identity lock.
    Locked,
    /// Recently lost observation, trying to recover previous identity.
    Recovering,
    /// Target unavailable for an extended period.
    Lost,
}

/// Identity-aware target tracker with anti-switch behavior.
#[derive(Debug)]
pub struct TargetTracker {
    kalman: Option<Kalman2D>,
    half_size: f32,
    misses: u32,
    stable_hits: u32,
    frame_index: u64,
    similarity_ema: f32,
    bootstrap_hint: Option<(f32, f32)>,
    state: TrackingState,
    last_bbox: Option<BBox>,
    last_confirmed_bbox: Option<BBox>,
    pending_hard_jump: Option<PendingHardJump>,
    last_confirmed_camera: Option<CameraState>,
    predicted_misses: u32,
    frozen_misses: u32,
}

impl TargetTracker {
    /// Create tracker without bootstrap hint.
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_hint(None)
    }

    /// Create tracker with optional initial search hint.
    #[must_use]
    pub fn new_with_hint(bootstrap_hint: Option<(f32, f32)>) -> Self {
        Self {
            kalman: None,
            half_size: 0.0,
            misses: 0,
            stable_hits: 0,
            frame_index: 0,
            similarity_ema: 0.6,
            bootstrap_hint,
            state: TrackingState::Searching,
            last_bbox: None,
            last_confirmed_bbox: None,
            pending_hard_jump: None,
            last_confirmed_camera: None,
            predicted_misses: 0,
            frozen_misses: 0,
        }
    }

    /// Current high-level tracking state.
    #[must_use]
    pub const fn state(&self) -> TrackingState {
        self.state
    }

    /// Whether identity recognition should run on this frame.
    #[must_use]
    pub fn should_run_recognition(&self) -> bool {
        let stride = match self.state {
            TrackingState::Searching => SEARCHING_RECOGNITION_STRIDE,
            TrackingState::Locked => {
                if self.similarity_ema > 0.82 {
                    LOCKED_RECOGNITION_STRIDE_HIGH
                } else {
                    LOCKED_RECOGNITION_STRIDE
                }
            }
            TrackingState::Recovering | TrackingState::Lost => RECOVERY_RECOGNITION_STRIDE,
        };
        self.frame_index.is_multiple_of(stride)
    }

    /// Predicted center to bias candidate ranking.
    #[must_use]
    pub fn search_hint(&self) -> Option<(f32, f32)> {
        self.kalman
            .as_ref()
            .map(|k| (k.cx(), k.cy()))
            .or(self.bootstrap_hint)
    }

    /// Last bbox confirmed by identity observation.
    #[must_use]
    pub fn last_confirmed_bbox(&self) -> Option<BBox> {
        self.last_confirmed_bbox
    }

    /// Update tracker with the scored observations from the current frame.
    ///
    /// The tracker accepts an observation only if it passes threshold/margin
    /// criteria and anti-switch checks.
    #[must_use]
    pub fn update(
        &mut self,
        observations: &[FaceObservation],
        similarity_threshold: f32,
        margin_threshold: f32,
    ) -> Option<CameraState> {
        self.frame_index += 1;

        if let Some(obs) =
            self.select_observation(observations, similarity_threshold, margin_threshold)
        {
            self.on_observation(obs);
            return self.current_camera_with_source(CameraSource::Observed);
        }

        self.on_miss()
    }

    /// Advance tracker using person detections when identity recognition is skipped.
    ///
    /// This keeps motion tethered to nearby person boxes without granting a
    /// confirmed identity hit.
    #[must_use]
    pub fn update_from_person_detections(&mut self, persons: &[BBox]) -> Option<CameraState> {
        self.frame_index += 1;
        let Some(supporting_bbox) = self.select_supporting_bbox(persons) else {
            return self.advance_without_recognition();
        };

        let cx = supporting_bbox.center_x();
        let cy = supporting_bbox.center_y();
        let hs = (supporting_bbox.width().max(supporting_bbox.height())) / 2.0;

        match self.kalman.as_mut() {
            Some(k) => {
                k.predict();
                k.update(cx, cy);
            }
            None => {
                self.kalman = Some(Kalman2D::new(cx, cy));
                self.half_size = hs;
            }
        }

        if self.half_size <= f32::EPSILON {
            self.half_size = hs;
        } else {
            self.half_size = 0.94 * self.half_size + 0.06 * hs;
        }

        self.last_bbox = Some(supporting_bbox);
        self.predicted_misses = 0;
        self.frozen_misses = 0;
        self.current_camera_with_source(CameraSource::Predicted)
    }

    /// Advance one frame when recognition was intentionally skipped.
    ///
    /// This keeps camera motion smooth without treating the frame as an
    /// identity miss.
    #[must_use]
    pub fn advance_without_recognition(&mut self) -> Option<CameraState> {
        self.frame_index += 1;
        if let Some(k) = self.kalman.as_mut() {
            k.predict();
        }
        self.current_camera_with_source(CameraSource::Predicted)
    }

    /// Reset all tracker state.
    pub fn reset(&mut self) {
        self.kalman = None;
        self.half_size = 0.0;
        self.misses = 0;
        self.stable_hits = 0;
        self.similarity_ema = 0.6;
        self.state = TrackingState::Searching;
        self.last_bbox = None;
        self.last_confirmed_bbox = None;
        self.pending_hard_jump = None;
        self.last_confirmed_camera = None;
        self.predicted_misses = 0;
        self.frozen_misses = 0;
    }

    fn select_observation(
        &mut self,
        observations: &[FaceObservation],
        similarity_threshold: f32,
        margin_threshold: f32,
    ) -> Option<FaceObservation> {
        let viable = observations
            .iter()
            .copied()
            .filter(|obs| obs.similarity >= similarity_threshold && obs.margin >= margin_threshold)
            .collect::<Vec<_>>();
        if viable.is_empty() {
            return None;
        }

        let Some((hx, hy)) = self.search_hint() else {
            return viable.first().copied();
        };

        let mut scored = viable
            .iter()
            .map(|obs| {
                let dx = obs.bbox.center_x() - hx;
                let dy = obs.bbox.center_y() - hy;
                let distance = (dx * dx + dy * dy).sqrt();
                let distance_norm = (obs.bbox.width().max(obs.bbox.height()) * 8.0).max(1.0);
                let proximity = 1.0 - (distance / distance_norm).clamp(0.0, 1.0);
                let continuity = self
                    .last_bbox
                    .map(|last| obs.bbox.iou(&last))
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                let body_score = obs
                    .body_similarity
                    .map(|sim| ((sim + 1.0) * 0.5).clamp(0.0, 1.0))
                    .unwrap_or(0.0);
                let score = obs.similarity
                    + obs.margin * 0.20
                    + proximity * 0.16
                    + continuity * 0.16
                    + body_score * BODY_SIMILARITY_WEIGHT;
                (score, distance, *obs)
            })
            .collect::<Vec<_>>();

        scored.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let (best_score, _, best_obs) = *scored.first()?;
        if matches!(
            self.state,
            TrackingState::Locked | TrackingState::Recovering
        ) && let Some(last_bbox) = self.last_bbox
        {
            let iou = best_obs.bbox.iou(&last_bbox);
            let second_score = scored.get(1).map(|row| row.0);
            let hard_jump = iou < HARD_JUMP_IOU;
            let weak_gap = second_score
                .map(|score| best_score - score < HARD_JUMP_SCORE_GAP)
                .unwrap_or(true);
            let body_score = best_obs.body_similarity.map(|sim| ((sim + 1.0) * 0.5).clamp(0.0, 1.0));
            let weak_confidence = best_obs.similarity < similarity_threshold + 0.06
                || best_obs.margin < margin_threshold + 0.02
                || body_score.is_some_and(|score| score < 0.58);
            if hard_jump {
                if !weak_gap && !weak_confidence {
                    self.pending_hard_jump = None;
                } else if !self.confirm_hard_jump(best_obs.bbox) {
                    return None;
                }
            }
        }

        self.pending_hard_jump = None;
        Some(best_obs)
    }

    fn confirm_hard_jump(&mut self, bbox: BBox) -> bool {
        match self.pending_hard_jump {
            Some(mut pending) => {
                let continuity = pending.bbox.iou(&bbox);
                let near_enough = continuity >= 0.20 || {
                    let dx = pending.bbox.center_x() - bbox.center_x();
                    let dy = pending.bbox.center_y() - bbox.center_y();
                    let distance = (dx * dx + dy * dy).sqrt();
                    let scale = pending
                        .bbox
                        .width()
                        .max(pending.bbox.height())
                        .max(bbox.width().max(bbox.height()));
                    distance <= scale * 1.6
                };
                if near_enough {
                    pending.votes = pending.votes.saturating_add(1);
                } else {
                    pending = PendingHardJump { bbox, votes: 1 };
                }
                let confirmed = pending.votes >= HARD_JUMP_CONFIRM_FRAMES;
                self.pending_hard_jump = Some(pending);
                confirmed
            }
            None => {
                self.pending_hard_jump = Some(PendingHardJump { bbox, votes: 1 });
                false
            }
        }
    }

    fn on_observation(&mut self, obs: FaceObservation) {
        self.misses = 0;
        self.stable_hits = self.stable_hits.saturating_add(1);
        self.similarity_ema = 0.85 * self.similarity_ema + 0.15 * obs.similarity.clamp(0.0, 1.0);
        self.predicted_misses = 0;
        self.frozen_misses = 0;
        self.pending_hard_jump = None;

        let bbox = obs.bbox;
        let cx = bbox.center_x();
        let cy = bbox.center_y();
        let hs = (bbox.width().max(bbox.height())) / 2.0;

        match self.kalman.as_mut() {
            Some(k) => {
                k.predict();
                k.update(cx, cy);
            }
            None => {
                self.kalman = Some(Kalman2D::new(cx, cy));
                self.half_size = hs;
            }
        }

        if self.half_size <= f32::EPSILON {
            self.half_size = hs;
        } else {
            self.half_size = 0.88 * self.half_size + 0.12 * hs;
        }

        self.bootstrap_hint = None;
        self.last_bbox = Some(bbox);
        self.last_confirmed_bbox = Some(bbox);

        self.state = if self.stable_hits >= 2 {
            TrackingState::Locked
        } else {
            TrackingState::Recovering
        };

        self.last_confirmed_camera = self.current_camera_with_source(CameraSource::Observed);
    }

    fn select_supporting_bbox(&self, persons: &[BBox]) -> Option<BBox> {
        if persons.is_empty() {
            return None;
        }
        let Some((hx, hy)) = self.search_hint() else {
            return None;
        };

        let mut ranked = persons
            .iter()
            .copied()
            .map(|bbox| {
                let proximity = {
                    let dx = bbox.center_x() - hx;
                    let dy = bbox.center_y() - hy;
                    let distance = (dx * dx + dy * dy).sqrt();
                    let norm = (bbox.width().max(bbox.height()) * 6.0).max(1.0);
                    1.0 - (distance / norm).clamp(0.0, 1.0)
                };
                let continuity = self
                    .last_bbox
                    .map(|last| last.iou(&bbox))
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                let score = proximity * 0.35 + continuity * 0.65;
                (score, bbox, proximity, continuity)
            })
            .collect::<Vec<_>>();

        ranked.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        let (_, best, best_proximity, best_continuity) = *ranked.first()?;

        if matches!(
            self.state,
            TrackingState::Locked | TrackingState::Recovering
        ) && let Some(last_bbox) = self.last_bbox
        {
            let dx = best.center_x() - last_bbox.center_x();
            let dy = best.center_y() - last_bbox.center_y();
            let distance = (dx * dx + dy * dy).sqrt();
            let scale = last_bbox
                .width()
                .max(last_bbox.height())
                .max(best.width().max(best.height()));
            let far_jump = distance > scale * 2.2;
            let weak_support = best_continuity < 0.08 && best_proximity < 0.50;
            if far_jump && weak_support {
                return None;
            }
        }

        Some(best)
    }

    fn on_miss(&mut self) -> Option<CameraState> {
        self.misses = self.misses.saturating_add(1);
        self.stable_hits = 0;

        match self.state {
            TrackingState::Searching => {
                if self.kalman.is_none() {
                    return None;
                }
            }
            TrackingState::Locked => {
                self.state = TrackingState::Recovering;
            }
            TrackingState::Recovering => {
                if self.misses > MAX_RECOVERING_MISSES {
                    self.state = TrackingState::Lost;
                }
            }
            TrackingState::Lost => {
                if self.misses > MAX_LOST_MISSES {
                    self.reset();
                    return None;
                }
            }
        }

        if self.predicted_misses < MAX_PREDICTED_MISSES {
            if let Some(k) = self.kalman.as_mut() {
                k.predict();
            }
            self.predicted_misses = self.predicted_misses.saturating_add(1);
            return self.current_camera_with_source(CameraSource::Predicted);
        }

        if let Some(mut held) = self.last_confirmed_camera {
            if self.frozen_misses < MAX_FROZEN_MISSES {
                self.frozen_misses = self.frozen_misses.saturating_add(1);
                held.source = CameraSource::Held;
                held.miss_count = self.misses;
                return Some(held);
            }
        }

        None
    }

    fn current_camera_with_source(&self, source: CameraSource) -> Option<CameraState> {
        self.kalman.as_ref().map(|k| CameraState {
            cx: k.cx(),
            cy: k.cy(),
            half_size: self.half_size,
            source,
            miss_count: self.misses,
        })
    }
}

impl Default for TargetTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(cx: f32, cy: f32, size: f32) -> BBox {
        BBox {
            x1: cx - size,
            y1: cy - size,
            x2: cx + size,
            y2: cy + size,
            confidence: 0.9,
        }
    }

    fn obs(cx: f32, cy: f32, sim: f32, margin: f32) -> FaceObservation {
        FaceObservation {
            bbox: bbox(cx, cy, 40.0),
            similarity: sim,
            impostor_similarity: sim - margin,
            margin,
            body_similarity: None,
        }
    }

    #[test]
    fn locks_after_two_hits() {
        let mut tracker = TargetTracker::new();
        assert_eq!(tracker.state(), TrackingState::Searching);
        let _ = tracker.update(&[obs(100.0, 100.0, 0.78, 0.12)], 0.6, 0.03);
        assert_eq!(tracker.state(), TrackingState::Recovering);
        let _ = tracker.update(&[obs(102.0, 101.0, 0.79, 0.13)], 0.6, 0.03);
        assert_eq!(tracker.state(), TrackingState::Locked);
    }

    #[test]
    fn rejects_hard_jump_when_ambiguous() {
        let mut tracker = TargetTracker::new();
        let _ = tracker.update(&[obs(200.0, 200.0, 0.82, 0.14)], 0.6, 0.03);
        let _ = tracker.update(&[obs(201.0, 200.0, 0.83, 0.14)], 0.6, 0.03);
        assert_eq!(tracker.state(), TrackingState::Locked);

        let far = obs(900.0, 900.0, 0.67, 0.05);
        let near_competitor = obs(870.0, 880.0, 0.66, 0.05);
        let camera_before = tracker
            .current_camera_with_source(CameraSource::Observed)
            .expect("camera before");
        let _ = tracker.update(&[far, near_competitor], 0.6, 0.03);
        let camera_after = tracker
            .current_camera_with_source(CameraSource::Predicted)
            .expect("camera after");
        assert!((camera_after.cx - camera_before.cx).abs() < 220.0);
    }

    #[test]
    fn freezes_after_prediction_budget_exhausted() {
        let mut tracker = TargetTracker::new();
        let _ = tracker.update(&[obs(300.0, 260.0, 0.81, 0.15)], 0.6, 0.03);
        let _ = tracker.update(&[obs(302.0, 261.0, 0.82, 0.15)], 0.6, 0.03);

        let mut seen_hold = false;
        for _ in 0..(MAX_PREDICTED_MISSES + 4) {
            let camera = tracker.update(&[], 0.6, 0.03);
            if let Some(cam) = camera
                && cam.source == CameraSource::Held
            {
                seen_hold = true;
                break;
            }
        }
        assert!(seen_hold);
    }

    #[test]
    fn hard_jump_requires_repeated_confirmation() {
        let mut tracker = TargetTracker::new();
        let _ = tracker.update(&[obs(120.0, 120.0, 0.79, 0.14)], 0.6, 0.03);
        let _ = tracker.update(&[obs(122.0, 119.0, 0.80, 0.14)], 0.6, 0.03);

        let jump = obs(860.0, 760.0, 0.66, 0.05);
        let first = tracker.update(&[jump], 0.6, 0.03).expect("first camera");
        assert_ne!(first.source, CameraSource::Observed);

        let second = tracker.update(&[jump], 0.6, 0.03).expect("second camera");
        assert_eq!(second.source, CameraSource::Observed);
    }

    #[test]
    fn supporting_bbox_updates_when_recognition_skipped() {
        let mut tracker = TargetTracker::new();
        let _ = tracker.update(&[obs(260.0, 220.0, 0.8, 0.12)], 0.6, 0.03);
        let _ = tracker.update(&[obs(262.0, 222.0, 0.81, 0.12)], 0.6, 0.03);

        let persons = [bbox(264.0, 224.0, 40.0)];
        let camera = tracker
            .update_from_person_detections(&persons)
            .expect("supporting camera");
        assert_eq!(camera.source, CameraSource::Predicted);
        assert!((camera.cx - 264.0).abs() < 80.0);
    }
}
