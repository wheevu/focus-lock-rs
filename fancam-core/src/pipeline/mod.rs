use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::detection::{Detector, FaceIdentifier};
use crate::rendering::FrameRenderer;
use crate::tracking::{BiasTracker, CameraState};
use crate::video::RgbFrame;

pub struct Analyzer {
    detector: Detector,
    identifier: FaceIdentifier,
    tracker: BiasTracker,
    prof_frames: u64,
    prof_detect: Duration,
    prof_identify: Duration,
}

impl Analyzer {
    pub fn new(detector: Detector, identifier: FaceIdentifier, tracker: BiasTracker) -> Self {
        Self {
            detector,
            identifier,
            tracker,
            prof_frames: 0,
            prof_detect: Duration::ZERO,
            prof_identify: Duration::ZERO,
        }
    }

    pub fn analyze(&mut self, frame: &RgbFrame) -> Option<CameraState> {
        let detection = if self.tracker.should_run_recognition() {
            let detect_start = Instant::now();
            match self.detector.detect(frame) {
                Ok(persons) => {
                    self.prof_detect += detect_start.elapsed();
                    let identify_start = Instant::now();
                    match self
                        .identifier
                        .identify(frame, &persons, self.tracker.search_hint())
                    {
                        Ok(found) => {
                            self.prof_identify += identify_start.elapsed();
                            found
                        }
                        Err(e) => {
                            self.prof_identify += identify_start.elapsed();
                            tracing::warn!("face ID error: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    self.prof_detect += detect_start.elapsed();
                    tracing::warn!("detection error: {e}");
                    None
                }
            }
        } else {
            None
        };

        self.tracker
            .note_similarity(detection.as_ref().map(|m| m.similarity));
        let camera = self.tracker.update(detection.map(|m| m.bbox));

        self.prof_frames += 1;
        if self.prof_frames > 0 && self.prof_frames % 300 == 0 {
            tracing::info!(
                frames = self.prof_frames,
                detect_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_detect.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                identify_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_identify.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                "pipeline analyze timings"
            );
        }

        camera
    }
}

pub struct Renderer {
    renderer: FrameRenderer,
    prof_frames: u64,
    prof_render: Duration,
}

impl Renderer {
    pub fn new(renderer: FrameRenderer) -> Self {
        Self {
            renderer,
            prof_frames: 0,
            prof_render: Duration::ZERO,
        }
    }

    pub fn render(&mut self, frame: &mut RgbFrame, camera: Option<CameraState>) {
        let render_start = Instant::now();
        let result = match camera {
            Some(ref state) => self.renderer.crop_fancam_inplace(frame, state),
            None => self.renderer.letterbox_passthrough_inplace(frame),
        };
        self.prof_render += render_start.elapsed();
        self.prof_frames += 1;

        if let Err(e) = result {
            tracing::warn!("render error: {e}");
        }

        if self.prof_frames > 0 && self.prof_frames % 300 == 0 {
            tracing::info!(
                frames = self.prof_frames,
                render_ms_per_frame = format!(
                    "{:.2}",
                    self.prof_render.as_secs_f64() * 1000.0 / self.prof_frames as f64
                ),
                "pipeline render timings"
            );
        }
    }
}

pub struct Pipeline {
    analyzer: Analyzer,
    renderer: Renderer,
}

impl Pipeline {
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        yolo_model_path: P,
        face_model_path: Q,
        reference_image_path: R,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let detector = Detector::load(yolo_model_path)?;
        let identifier = FaceIdentifier::load(
            face_model_path,
            reference_image_path,
            similarity_threshold.clamp(0.0, 1.0),
        )?;
        let tracker = BiasTracker::new();
        let renderer = FrameRenderer::new();

        Ok(Self {
            analyzer: Analyzer::new(detector, identifier, tracker),
            renderer: Renderer::new(renderer),
        })
    }

    pub fn into_parts(self) -> (Analyzer, Renderer) {
        (self.analyzer, self.renderer)
    }
}
