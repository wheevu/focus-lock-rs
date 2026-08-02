//! face — SCRFD-based face detection
//!
//! Wraps `det_500m.onnx` (a 2.4 MB SCRFD-style face detector) to detect faces,
//! return bounding boxes and 5 facial landmarks for any input image region.
//!
//! # Model details
//!
//! - Input:  `input.1`, dynamic NCHW `[1, 3, H, W]`, RGB, normalized to `[0, 1]`
//! - Outputs: 9 tensors at 3 FPN scales
//!   - scores:   `[N, 1]`
//!   - bboxes:   `[N, 4]`  (cx, cy, w, h)
//!   - landmarks:`[N, 10]` (5 kp × 2 coords: left-eye, right-eye, nose, left-mouth, right-mouth)

use std::path::Path;

use fast_image_resize as fr;
use ort::execution_providers as ep;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use crate::Result;
use crate::detection::BBox;
use crate::video::RgbFrame;

/// Minimum confidence threshold for face detections.
const FACE_CONF_THRESHOLD: f32 = 0.30;
/// IoU threshold for NMS across face candidates.
const FACE_NMS_IOU: f32 = 0.40;
/// Maximum dimension for face detection inference (keep reasonable).
const FACE_DETECT_MAX_DIM: u32 = 640;
/// Maximum candidates per scale before NMS.
const MAX_PRE_NMS: usize = 2000;
/// Maximum faces returned after NMS.
const MAX_POST_NMS: usize = 80;

/// Ordered facial landmark indices.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub enum FaceLandmark {
    LeftEye,
    RightEye,
    Nose,
    LeftMouth,
    RightMouth,
}

/// Five facial landmarks as (x, y) pixel coordinates within the source image.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub struct Landmarks {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub left_mouth: (f32, f32),
    pub right_mouth: (f32, f32),
}

impl Landmarks {
    fn from_slice(values: &[f32]) -> Self {
        Self {
            left_eye: (values[0], values[1]),
            right_eye: (values[2], values[3]),
            nose: (values[4], values[5]),
            left_mouth: (values[6], values[7]),
            right_mouth: (values[8], values[9]),
        }
    }
}

/// Detected face with bounding box, confidence, and optional landmarks.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub struct FaceBox {
    /// Bounding box in source-image coordinates.
    pub bbox: BBox,
    /// Detection confidence.
    pub confidence: f32,
    /// Five facial landmarks if the model provides them.
    pub landmarks: Option<Landmarks>,
}

// ── SCRFD face detector ──────────────────────────────────────────────────────

/// Wraps `det_500m.onnx` session for face detection.
#[derive(Debug)]
pub struct FaceDetector {
    session: Session,
    resizer: fr::Resizer,
    resize_buf: Vec<u8>,
}

/// Anchor stride configuration for SCRFD FPN outputs.
#[derive(Debug, Clone, Copy)]
struct FpnScale {
    /// Number of anchor points at this scale.
    count: usize,
    /// Stride (downsample factor) for this FPN level.
    stride: u32,
}

impl FaceDetector {
    /// Load an SCRFD ONNX model from `model_path`.
    ///
    /// Uses CPU execution provider because the model has dynamic input shapes
    /// that are incompatible with CoreML's static shape compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or the ORT session
    /// cannot be created.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        crate::runtime::OrtConfig::ensure_initialized()
            .map_err(|e| crate::FancamError::ort_config(e.to_string()))?;

        let mk_err = |e: ort::Error| -> crate::FancamError {
            crate::FancamError::inference(format!("face detector session: {e}"))
        };

        let mut builder = Session::builder().map_err(mk_err)?;
        builder = builder.with_intra_threads(1).map_err(mk_err)?;
        builder = builder.with_inter_threads(1).map_err(mk_err)?;
        builder = builder.with_parallel_execution(false).map_err(mk_err)?;
        builder = builder
            .with_execution_providers([ep::CPUExecutionProvider::default().build()])
            .map_err(mk_err)?;
        let session = builder.commit_from_file(path).map_err(mk_err)?;

        Ok(Self {
            session,
            resizer: fr::Resizer::new(),
            resize_buf: Vec::new(),
        })
    }

    /// Detect faces in a full video frame.
    ///
    /// Returns empty vec if no faces are found above the confidence threshold.
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing or inference fails.
    pub fn detect(&mut self, frame: &RgbFrame) -> Result<Vec<FaceBox>> {
        frame
            .validate()
            .map_err(|error| crate::FancamError::invalid_frame(error.to_string()))?;
        self.detect_in_roi(frame, None)
    }

    /// Detect the best (highest confidence) face within a person bounding box.
    ///
    /// Returns `None` if no face is detected in the region.
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing or inference fails.
    pub fn best_face_in_person_bbox(
        &mut self,
        frame: &RgbFrame,
        person_bbox: BBox,
        expand_margin: f32,
    ) -> Result<Option<FaceBox>> {
        frame
            .validate()
            .map_err(|error| crate::FancamError::invalid_frame(error.to_string()))?;
        // Expand the ROI slightly beyond the person bbox to capture slightly-off boxes.
        let margin_x = person_bbox.width() * expand_margin;
        let margin_y = person_bbox.height() * expand_margin;
        let roi = BBox {
            x1: (person_bbox.x1 - margin_x).max(0.0),
            y1: (person_bbox.y1 - margin_y).max(0.0),
            x2: (person_bbox.x2 + margin_x).min(frame.width as f32),
            y2: (person_bbox.y2 + margin_y).min(frame.height as f32),
            confidence: 1.0,
        };

        let faces = self.detect_in_roi(frame, Some(roi))?;
        Ok(faces.into_iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }))
    }

    /// Core detection method with optional ROI restriction.
    fn detect_in_roi(&mut self, frame: &RgbFrame, roi: Option<BBox>) -> Result<Vec<FaceBox>> {
        // Determine effective image dimensions (the whole frame or the ROI).
        let (src_x, src_y, src_w, src_h) = match roi {
            Some(b) => (
                b.x1 as u32,
                b.y1 as u32,
                (b.x2 - b.x1) as u32,
                (b.y2 - b.y1) as u32,
            ),
            None => (0, 0, frame.width, frame.height),
        };

        let src_w = src_w.max(16);
        let src_h = src_h.max(16);

        // Compute downscale ratio to keep the largest dimension <= FACE_DETECT_MAX_DIM
        let max_dim = src_w.max(src_h) as f32;
        let scale = if max_dim > FACE_DETECT_MAX_DIM as f32 {
            FACE_DETECT_MAX_DIM as f32 / max_dim
        } else {
            1.0
        };

        let model_w = (src_w as f32 * scale).round() as u32;
        let model_h = (src_h as f32 * scale).round() as u32;

        // Crop the ROI from the frame and resize to model input size
        let roi_rgb = self.crop_and_resize(frame, src_x, src_y, src_w, src_h, model_w, model_h)?;

        // Build NCHW float tensor [1, 3, model_h, model_w] normalized to [0, 1]
        let tensor_data = preprocess_for_face_detection(&roi_rgb, model_w, model_h);
        let shape = [1usize, 3, model_h as usize, model_w as usize];
        let tensor = Tensor::from_array((shape, tensor_data.into_boxed_slice()))
            .map_err(|e| crate::FancamError::inference(format!("face detector tensor: {e}")))?
            .into_dyn();

        let outputs = self
            .session
            .run(ort::inputs!["input.1" => tensor])
            .map_err(|e| crate::FancamError::inference(format!("face detector inference: {e}")))?;

        // Parse outputs from 3 FPN scales
        let scales = [
            FpnScale {
                count: 12800,
                stride: 8,
            },
            FpnScale {
                count: 3200,
                stride: 16,
            },
            FpnScale {
                count: 800,
                stride: 32,
            },
        ];

        let mut all_faces = Vec::new();

        for (scale_idx, fpn) in scales.iter().enumerate() {
            let score_name = ["443", "468", "493"][scale_idx];
            let bbox_name = ["446", "471", "496"][scale_idx];
            let lm_name = ["449", "474", "499"][scale_idx];

            let (_score_shape, score_data) = outputs[score_name]
                .try_extract_tensor::<f32>()
                .map_err(|e| crate::FancamError::inference(format!("face score extract: {e}")))?;

            let (_bbox_shape, bbox_data) = outputs[bbox_name]
                .try_extract_tensor::<f32>()
                .map_err(|e| crate::FancamError::inference(format!("face bbox extract: {e}")))?;

            let (_lm_shape, lm_data) = outputs[lm_name]
                .try_extract_tensor::<f32>()
                .map_err(|e| crate::FancamError::inference(format!("face lm extract: {e}")))?;

            let count = fpn.count.min(score_data.len());
            let stride = fpn.stride;
            // For dynamic input, recompute grid dimensions
            let grid_h = (model_h + stride / 2) / stride;
            let grid_w = (model_w + stride / 2) / stride;

            let mut candidates: Vec<FaceBox> = (0..count.min(grid_h as usize * grid_w as usize))
                .into_par_iter()
                .filter_map(|i| {
                    let conf = score_data[i];
                    if conf < FACE_CONF_THRESHOLD {
                        return None;
                    }

                    // Decode bbox from (cx, cy, w, h) in model input coordinates
                    let cx = bbox_data[i] + (i % grid_w as usize) as f32 * stride as f32;
                    let cy = bbox_data[i + count] + (i / grid_w as usize) as f32 * stride as f32;
                    let w = bbox_data[i + 2 * count];
                    let h = bbox_data[i + 3 * count];

                    // Map back to source-frame (ROI) coordinates
                    let inv_scale = 1.0 / scale;
                    let x1 = ((cx - w / 2.0) * inv_scale + src_x as f32).max(0.0);
                    let y1 = ((cy - h / 2.0) * inv_scale + src_y as f32).max(0.0);
                    let x2 = ((cx + w / 2.0) * inv_scale + src_x as f32).min(frame.width as f32);
                    let y2 = ((cy + h / 2.0) * inv_scale + src_y as f32).min(frame.height as f32);

                    if x2 <= x1 || y2 <= y1 {
                        return None;
                    }

                    // Decode landmarks
                    let lm_scale = inv_scale;
                    let landmarks = Landmarks::from_slice(&[
                        (lm_data[i] + (i % grid_w as usize) as f32 * stride as f32) * lm_scale
                            + src_x as f32,
                        (lm_data[i + count] + (i / grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_y as f32,
                        (lm_data[i + 2 * count] + (i % grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_x as f32,
                        (lm_data[i + 3 * count] + (i / grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_y as f32,
                        (lm_data[i + 4 * count] + (i % grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_x as f32,
                        (lm_data[i + 5 * count] + (i / grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_y as f32,
                        (lm_data[i + 6 * count] + (i % grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_x as f32,
                        (lm_data[i + 7 * count] + (i / grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_y as f32,
                        (lm_data[i + 8 * count] + (i % grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_x as f32,
                        (lm_data[i + 9 * count] + (i / grid_w as usize) as f32 * stride as f32)
                            * lm_scale
                            + src_y as f32,
                    ]);

                    Some(FaceBox {
                        bbox: BBox {
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence: conf,
                        },
                        confidence: conf,
                        landmarks: Some(landmarks),
                    })
                })
                .collect();

            // Pre-NMS pruning: keep top MAX_PRE_NMS by confidence
            if candidates.len() > MAX_PRE_NMS {
                candidates.select_nth_unstable_by(MAX_PRE_NMS, |a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(MAX_PRE_NMS);
            }

            all_faces.append(&mut candidates);
        }

        // Global NMS across all scales
        Ok(face_nms(all_faces, FACE_NMS_IOU, MAX_POST_NMS))
    }

    /// Crop a rectangular region from the frame and resize to `out_w × out_h`.
    fn crop_and_resize(
        &mut self,
        frame: &RgbFrame,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        out_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>> {
        // Step 1: crop the ROI from the source frame
        let src_stride = (frame.width * 3) as usize;
        let crop_stride = (w * 3) as usize;
        let crop_len = crop_stride * h as usize;
        let mut crop_buf = Vec::with_capacity(crop_len);

        for row in 0..h as usize {
            let src_start = (y as usize + row) * src_stride + x as usize * 3;
            crop_buf.extend_from_slice(&frame.data[src_start..src_start + crop_stride]);
        }

        // Step 2: resize if needed
        if w == out_w && h == out_h {
            return Ok(crop_buf);
        }

        let src_ref = fr::images::ImageRef::new(w, h, &crop_buf, fr::PixelType::U8x3)
            .map_err(|e| crate::FancamError::image_processing(format!("face crop src: {e}")))?;

        let dst_len = (out_w * out_h * 3) as usize;
        if self.resize_buf.len() != dst_len {
            self.resize_buf.resize(dst_len, 0);
        }

        let mut dst = fr::images::Image::from_vec_u8(
            out_w,
            out_h,
            std::mem::take(&mut self.resize_buf),
            fr::PixelType::U8x3,
        )
        .map_err(|e| crate::FancamError::image_processing(format!("face crop dst: {e}")))?;

        let options = fr::ResizeOptions::new()
            .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        self.resizer
            .resize(&src_ref, &mut dst, Some(&options))
            .map_err(|e| crate::FancamError::image_processing(format!("face crop resize: {e}")))?;

        self.resize_buf = dst.into_vec();
        Ok(std::mem::take(&mut self.resize_buf))
    }
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

/// Convert resized RGB bytes to NCHW float tensor with [0, 1] normalization.
fn preprocess_for_face_detection(rgb: &[u8], w: u32, h: u32) -> Vec<f32> {
    let size = (w * h) as usize;
    let mut tensor = vec![0f32; 3 * size];
    let (r, gb) = tensor.split_at_mut(size);
    let (g, b) = gb.split_at_mut(size);

    rayon::join(
        || {
            r.par_iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = rgb[i * 3] as f32 / 255.0);
        },
        || {
            rayon::join(
                || {
                    g.par_iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = rgb[i * 3 + 1] as f32 / 255.0);
                },
                || {
                    b.par_iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = rgb[i * 3 + 2] as f32 / 255.0);
                },
            );
        },
    );

    tensor
}

// ── NMS ──────────────────────────────────────────────────────────────────────

/// Greedy NMS for face boxes, sorted by confidence descending.
fn face_nms(mut boxes: Vec<FaceBox>, iou_thresh: f32, max_out: usize) -> Vec<FaceBox> {
    boxes.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = Vec::with_capacity(max_out.min(boxes.len()));
    for candidate in boxes {
        if result.len() >= max_out {
            break;
        }
        let keep = result
            .iter()
            .all(|selected: &FaceBox| candidate.bbox.iou(&selected.bbox) < iou_thresh);
        if keep {
            result.push(candidate);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_normalization_range() {
        // Verify that preprocess_for_face_detection produces values in [0, 1]
        let rgb = vec![128u8; 3 * 4 * 4]; // 4x4 gray image
        let tensor = preprocess_for_face_detection(&rgb, 4, 4);
        for &v in &tensor {
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0,1] range");
        }
        // Solid 128 should give ~0.502
        assert!((tensor[0] - 128.0 / 255.0).abs() < 0.001);
    }

    #[test]
    fn test_nms_empty_input() {
        let result = face_nms(vec![], 0.4, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_nms_removes_duplicates() {
        let make_face = |x1: f32, y1: f32, x2: f32, y2: f32, conf: f32| FaceBox {
            bbox: BBox {
                x1,
                y1,
                x2,
                y2,
                confidence: conf,
            },
            confidence: conf,
            landmarks: None,
        };
        // Two overlapping faces
        let faces = vec![
            make_face(10.0, 10.0, 100.0, 100.0, 0.9),
            make_face(12.0, 12.0, 98.0, 98.0, 0.85),
        ];
        // IoU of these two boxes: area each = 90*90=8100, intersection = 86*86=7396, union = 8100+8100-7396=8804, iou=0.84
        // Since 0.84 > 0.4 threshold, the second should be suppressed
        let result = face_nms(faces, 0.4, 10);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_nms_keeps_distant_faces() {
        let make_face = |x1: f32, y1: f32, x2: f32, y2: f32, conf: f32| FaceBox {
            bbox: BBox {
                x1,
                y1,
                x2,
                y2,
                confidence: conf,
            },
            confidence: conf,
            landmarks: None,
        };
        let faces = vec![
            make_face(10.0, 10.0, 100.0, 100.0, 0.9),
            make_face(200.0, 200.0, 300.0, 300.0, 0.8),
        ];
        let result = face_nms(faces, 0.4, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_nms_respects_max_out() {
        let make_face = |x1: f32, y1: f32, x2: f32, y2: f32, conf: f32| FaceBox {
            bbox: BBox {
                x1,
                y1,
                x2,
                y2,
                confidence: conf,
            },
            confidence: conf,
            landmarks: None,
        };
        let faces = vec![
            make_face(10.0, 10.0, 50.0, 50.0, 0.9),
            make_face(100.0, 10.0, 140.0, 50.0, 0.8),
            make_face(200.0, 10.0, 240.0, 50.0, 0.7),
        ];
        let result = face_nms(faces, 0.4, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_landmarks_from_slice() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let lm = Landmarks::from_slice(&values);
        assert_eq!(lm.left_eye, (1.0, 2.0));
        assert_eq!(lm.right_eye, (3.0, 4.0));
        assert_eq!(lm.nose, (5.0, 6.0));
        assert_eq!(lm.left_mouth, (7.0, 8.0));
        assert_eq!(lm.right_mouth, (9.0, 10.0));
    }
}
