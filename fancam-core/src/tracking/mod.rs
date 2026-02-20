//! tracking — Kalman filter camera state + Re-ID occlusion handling
//!
//! Phase 4: smooth the raw detection coordinates so the virtual camera pans
//! cinematically rather than snapping frame-to-frame.
//!
//! Model: 2D constant-velocity Kalman filter over (cx, cy) — the centre of
//! the target bounding box.  Zoom (box area) is smoothed with a 1D variant.
//!
//! State vector: [cx, cy, vx, vy]ᵀ  (position + velocity in pixels/frame)
//! Measurement:  [cx, cy]ᵀ

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};
use tracing::debug;

use crate::detection::BBox;

// ── Tuning constants ─────────────────────────────────────────────────────────

/// Process noise — how much we trust the motion model.
const PROCESS_NOISE: f32 = 4.0;
/// Measurement noise — how much we trust the detector.
const MEASUREMENT_NOISE: f32 = 16.0;
/// Number of frames to keep the camera locked on last known position before
/// declaring the target permanently lost.
const MAX_LOST_FRAMES: u32 = 90;

// ── Kalman filter ─────────────────────────────────────────────────────────────

/// A minimal 2D constant-velocity Kalman filter.
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

        // x_{k+1} = F * x_k
        let f = Matrix4::new(
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );

        let h = Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        let q = Matrix4::identity() * PROCESS_NOISE;
        let r = Matrix2::identity() * MEASUREMENT_NOISE;

        Self { x, p, f, h, q, r }
    }

    /// Predict step — advance state one frame.
    fn predict(&mut self) {
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f.transpose() + self.q;
    }

    /// Update step — incorporate a new measurement [cx, cy].
    fn update(&mut self, cx: f32, cy: f32) {
        let z = Vector2::new(cx, cy);
        let y = z - self.h * self.x; // innovation
        let s = self.h * self.p * self.h.transpose() + self.r;
        let Some(s_inv) = s.try_inverse() else {
            return;
        };
        let k: Matrix4x2<f32> = self.p * self.h.transpose() * s_inv;
        self.x = self.x + k * y;
        self.p = (Matrix4::identity() - k * self.h) * self.p;
    }

    fn cx(&self) -> f32 {
        self.x[0]
    }
    fn cy(&self) -> f32 {
        self.x[1]
    }
}

// ── CameraState ───────────────────────────────────────────────────────────────

/// The current state of the virtual camera, expressed in original-frame pixel
/// coordinates.  `(cx, cy)` is the centre of the crop window; `half_size` is
/// half the side length of the (square before aspect conversion) window.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    pub cx: f32,
    pub cy: f32,
    /// Smoothed half-size of the bounding box (used to set crop zoom).
    pub half_size: f32,
}

// ── BiasTracker ───────────────────────────────────────────────────────────────

/// High-level tracker: wraps the Kalman filter and manages the occlusion /
/// re-ID state machine.
pub struct BiasTracker {
    kalman: Option<Kalman2D>,
    /// Smoothed half-size (exponential moving average).
    half_size: f32,
    /// Number of consecutive frames with no detection.
    lost_frames: u32,
    /// Frame index — used to decide when to throttle recognition.
    pub frame_index: u64,
    /// Smoothed identity-match confidence to adapt recognition cadence.
    similarity_ema: f32,
}

/// Baseline frame skip when tracking is stable.
const DEFAULT_RECOGNITION_STRIDE: u64 = 5;
/// Maximum frame skip when identity confidence is very high.
const MAX_RECOGNITION_STRIDE: u64 = 12;
/// Minimum frame skip while trying to recover lock.
const MIN_RECOGNITION_STRIDE: u64 = 2;
/// How many frames to skip recognition before the target is first found.
/// Running YOLO + ArcFace every single frame is extremely expensive on CPU;
/// the target isn't going to appear and vanish between adjacent frames.
const PRE_LOCK_STRIDE: u64 = 3;

impl BiasTracker {
    pub fn new() -> Self {
        Self {
            kalman: None,
            half_size: 0.0,
            lost_frames: 0,
            frame_index: 0,
            similarity_ema: 0.6,
        }
    }

    /// Whether recognition should run on this frame (throttled to every
    /// `PRE_LOCK_STRIDE` frames before lock-on, every `RECOGNITION_STRIDE`
    /// frames after lock-on).
    pub fn should_run_recognition(&self) -> bool {
        let stride = if self.kalman.is_none() {
            PRE_LOCK_STRIDE
        } else if self.lost_frames > 0 || self.similarity_ema < 0.62 {
            MIN_RECOGNITION_STRIDE
        } else if self.similarity_ema > 0.82 {
            MAX_RECOGNITION_STRIDE
        } else if self.similarity_ema > 0.72 {
            8
        } else {
            DEFAULT_RECOGNITION_STRIDE
        };
        self.frame_index % stride == 0
    }

    /// Provide the latest identity similarity so cadence can adapt dynamically.
    pub fn note_similarity(&mut self, similarity: Option<f32>) {
        if let Some(sim) = similarity {
            let sim = sim.clamp(0.0, 1.0);
            self.similarity_ema = 0.85 * self.similarity_ema + 0.15 * sim;
        } else if self.kalman.is_none() {
            self.similarity_ema = 0.6;
        }
    }

    /// Predicted center to bias re-identification search while relocking.
    pub fn search_hint(&self) -> Option<(f32, f32)> {
        self.kalman.as_ref().map(|k| (k.cx(), k.cy()))
    }

    /// Feed in the latest detection result (or `None` if not found this frame).
    /// Returns the smoothed `CameraState` to use for cropping.
    pub fn update(&mut self, detection: Option<BBox>) -> Option<CameraState> {
        self.frame_index += 1;

        match detection {
            Some(bbox) => {
                self.lost_frames = 0;
                let cx = bbox.center_x();
                let cy = bbox.center_y();
                let hs = (bbox.width().max(bbox.height())) / 2.0;

                match self.kalman.as_mut() {
                    Some(k) => {
                        k.predict();
                        k.update(cx, cy);
                    }
                    None => {
                        // First detection — initialise filter
                        self.kalman = Some(Kalman2D::new(cx, cy));
                        self.half_size = hs;
                    }
                }

                // Exponential moving average for box size (α = 0.1)
                self.half_size = 0.9 * self.half_size + 0.1 * hs;

                debug!(cx, cy, half_size = self.half_size, "tracker updated");
            }
            None => {
                self.lost_frames += 1;
                debug!(lost_frames = self.lost_frames, "target not detected");

                if let Some(k) = self.kalman.as_mut() {
                    // Predict-only: camera drifts along last known velocity
                    k.predict();
                }

                if self.lost_frames > MAX_LOST_FRAMES {
                    // Give up — reset the filter so we re-lock on reappearance
                    self.kalman = None;
                    self.lost_frames = 0;
                    return None;
                }
            }
        }

        self.kalman.as_ref().map(|k| CameraState {
            cx: k.cx(),
            cy: k.cy(),
            half_size: self.half_size,
        })
    }

    /// Reset the tracker (e.g. scene cut detected).
    pub fn reset(&mut self) {
        self.kalman = None;
        self.lost_frames = 0;
        self.similarity_ema = 0.6;
    }
}

impl Default for BiasTracker {
    fn default() -> Self {
        Self::new()
    }
}
