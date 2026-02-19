//! detection — YOLOv8-Nano person detection + ArcFace face identification
//!
//! Phase 2: load yolov8n.onnx, run inference on a 640×640 frame, return
//!          bounding boxes for the "person" class after NMS.
//!
//! Phase 3: load arcface.onnx, embed a reference face, filter detections by
//!          cosine similarity.

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::rect::Rect;
use ort::execution_providers as ep;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tracing::debug;

use crate::video::RgbFrame;

// ── Constants ────────────────────────────────────────────────────────────────

/// YOLOv8 input size (square).
const YOLO_SIZE: u32 = 640;
/// COCO class index for "person".
const PERSON_CLASS: usize = 0;
/// Confidence threshold for person detection.
const CONF_THRESHOLD: f32 = 0.45;
/// IoU threshold for NMS.
const IOU_THRESHOLD: f32 = 0.45;

/// ArcFace input size.
const FACE_SIZE: u32 = 112;
/// Cosine similarity threshold for identity match.
pub const SIMILARITY_THRESHOLD: f32 = 0.6;
/// Maximum number of person candidates to run ArcFace on per frame.
/// In concert scenes with 10-20 detections, this caps inference cost.
const MAX_FACE_CANDIDATES: usize = 5;
/// Approximate fraction of the person bounding box height occupied by the head.
const HEAD_FRACTION: f32 = 0.22;

// ── Public types ─────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in pixel coordinates of the original frame.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

impl BBox {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }
    pub fn center_x(&self) -> f32 {
        (self.x1 + self.x2) / 2.0
    }
    pub fn center_y(&self) -> f32 {
        (self.y1 + self.y2) / 2.0
    }
    /// IoU (intersection over union) with another box.
    pub fn iou(&self, other: &BBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        if inter == 0.0 {
            return 0.0;
        }
        let union = self.width() * self.height() + other.width() * other.height() - inter;
        inter / union
    }
}

// ── Detector ─────────────────────────────────────────────────────────────────

/// Wraps the YOLOv8-Nano ONNX session.
pub struct Detector {
    session: Session,
}

impl Detector {
    /// Load a YOLOv8n ONNX model from `model_path`.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create ORT session builder")?
            .with_execution_providers([ep::CoreMLExecutionProvider::default()
                .with_compute_units(ep::coreml::CoreMLComputeUnits::CPUAndNeuralEngine)
                .build()])
            .context("failed to register execution providers")?
            .commit_from_file(model_path)
            .context("failed to load YOLOv8 ONNX model")?;
        Ok(Self { session })
    }

    /// Run inference on `frame` and return bounding boxes (in original frame
    /// pixel coordinates) for all persons after NMS.
    pub fn detect(&mut self, frame: &RgbFrame) -> Result<Vec<BBox>> {
        let input_tensor = preprocess_yolo(frame)?;

        let outputs = self
            .session
            .run(ort::inputs!["images" => input_tensor])
            .context("YOLOv8 inference failed")?;

        // YOLOv8 output: [1, 84, 8400]  (84 = 4 box coords + 80 class scores)
        let (_shape, data) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .context("failed to extract YOLOv8 output tensor")?;

        // shape = [1, 84, 8400]
        // num_proposals = 8400, num_classes = 80
        let num_proposals = 8400usize;
        let num_classes = 80usize;

        let scale_x = frame.width as f32 / YOLO_SIZE as f32;
        let scale_y = frame.height as f32 / YOLO_SIZE as f32;

        let mut candidates: Vec<BBox> = Vec::new();

        for i in 0..num_proposals {
            // Data layout: [cx, cy, w, h, cls0_score, cls1_score, ...]
            // Stored column-major across the 84 rows.
            let cx = data[0 * num_proposals + i];
            let cy = data[1 * num_proposals + i];
            let w = data[2 * num_proposals + i];
            let h = data[3 * num_proposals + i];

            // Person score (class 0)
            let person_score = data[(4 + PERSON_CLASS) * num_proposals + i];

            // Best class score across all 80 classes
            let mut max_score = 0f32;
            for c in 0..num_classes {
                let s = data[(4 + c) * num_proposals + i];
                if s > max_score {
                    max_score = s;
                }
            }

            if person_score < CONF_THRESHOLD || person_score < max_score {
                continue;
            }

            // Convert YOLO (cx,cy,w,h) in 640-space → (x1,y1,x2,y2) in original frame
            let x1 = (cx - w / 2.0) * scale_x;
            let y1 = (cy - h / 2.0) * scale_y;
            let x2 = (cx + w / 2.0) * scale_x;
            let y2 = (cy + h / 2.0) * scale_y;

            candidates.push(BBox {
                x1: x1.max(0.0),
                y1: y1.max(0.0),
                x2: x2.min(frame.width as f32),
                y2: y2.min(frame.height as f32),
                confidence: person_score,
            });
        }

        Ok(nms(candidates, IOU_THRESHOLD))
    }
}

// ── Face identifier ──────────────────────────────────────────────────────────

/// Wraps the ArcFace ONNX session and the reference embedding.
pub struct FaceIdentifier {
    session: Session,
    reference_embedding: Vec<f32>,
}

impl FaceIdentifier {
    /// Load an ArcFace ONNX model and embed `reference_image_path` as the
    /// target identity.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        reference_image_path: Q,
    ) -> Result<Self> {
        let mut session = Session::builder()
            .context("failed to create ORT session builder")?
            .with_execution_providers([ep::CoreMLExecutionProvider::default()
                .with_compute_units(ep::coreml::CoreMLComputeUnits::CPUAndNeuralEngine)
                .build()])
            .context("failed to register execution providers")?
            .commit_from_file(model_path)
            .context("failed to load ArcFace ONNX model")?;

        // Load and embed the reference image (assumed to be a face crop)
        let ref_img = image::open(reference_image_path)
            .context("failed to open reference image")?
            .into_rgb8();

        let tensor = preprocess_face(&ref_img)?;
        let reference_embedding = {
            let outputs = session
                .run(ort::inputs!["input.1" => tensor])
                .context("ArcFace reference inference failed")?;
            let embedding = extract_first_embedding(&outputs)?;
            l2_normalize(&embedding)
        };

        debug!(
            dim = reference_embedding.len(),
            "reference face embedding computed"
        );

        Ok(Self {
            session,
            reference_embedding,
        })
    }

    /// Given a full RGB frame and a list of person bounding boxes, return the
    /// bbox of the target identity (highest similarity above threshold), or
    /// `None` if not found.
    ///
    /// To limit per-frame latency in crowded scenes, only the top
    /// `MAX_FACE_CANDIDATES` persons (by detection confidence) are evaluated.
    pub fn identify(&mut self, frame: &RgbFrame, persons: &[BBox]) -> Result<Option<BBox>> {
        let mut best: Option<(f32, BBox)> = None;

        // Sort candidates by confidence descending and cap at MAX_FACE_CANDIDATES.
        let mut candidates: Vec<BBox> = persons.to_vec();
        candidates.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(MAX_FACE_CANDIDATES);

        for &bbox in &candidates {
            let face_crop = crop_face(frame, bbox);
            let tensor = preprocess_face(&face_crop)?;

            let outputs = self
                .session
                .run(ort::inputs!["input.1" => tensor])
                .context("ArcFace inference failed")?;

            let embedding = l2_normalize(&extract_first_embedding(&outputs)?);
            let sim = cosine_similarity(&embedding, &self.reference_embedding);

            debug!(sim, "face similarity");

            if sim >= SIMILARITY_THRESHOLD {
                if best.is_none() || sim > best.unwrap().0 {
                    best = Some((sim, bbox));
                }
            }
        }

        Ok(best.map(|(_, b)| b))
    }
}

// ── Pre/post-processing helpers ──────────────────────────────────────────────

/// Resize the frame to 640×640, convert to NCHW float tensor normalised to
/// [0, 1], return as an ORT `Tensor`.
fn preprocess_yolo(frame: &RgbFrame) -> Result<ort::value::DynValue> {
    use fast_image_resize as fr;

    // Use fast_image_resize with NEON SIMD for the large 4K → 640x640 downscale.
    // ImageRef borrows the frame data immutably — no clone needed.
    let src =
        fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
            .context("failed to create fast_image_resize source")?;

    let mut dst = fr::images::Image::new(YOLO_SIZE, YOLO_SIZE, fr::PixelType::U8x3);

    let mut resizer = fr::Resizer::new();
    let options =
        fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer
        .resize(&src, &mut dst, Some(&options))
        .context("fast_image_resize YOLO downscale failed")?;

    let raw = dst.buffer();

    // NCHW float tensor: [1, 3, 640, 640].
    let size = (YOLO_SIZE * YOLO_SIZE) as usize;
    let mut tensor_data = vec![0f32; 3 * size];

    for idx in 0..size {
        tensor_data[idx] = raw[idx * 3] as f32 / 255.0; // R
        tensor_data[size + idx] = raw[idx * 3 + 1] as f32 / 255.0; // G
        tensor_data[2 * size + idx] = raw[idx * 3 + 2] as f32 / 255.0; // B
    }

    let shape = [1usize, 3, YOLO_SIZE as usize, YOLO_SIZE as usize];
    Ok(Tensor::from_array((shape, tensor_data.into_boxed_slice()))
        .context("failed to create YOLO input tensor")?
        .into_dyn())
}

/// Crop the approximate head region from a person BBox, resize to 112×112,
/// normalise to [-1, 1], return as an ORT `Tensor`.
fn preprocess_face(img: &RgbImage) -> Result<ort::value::DynValue> {
    use image::imageops::FilterType;

    let resized = image::imageops::resize(img, FACE_SIZE, FACE_SIZE, FilterType::Lanczos3);

    // NCHW float tensor: [1, 3, 112, 112].
    // Use flat indexed access over raw bytes — avoids per-pixel Pixel overhead.
    let size = (FACE_SIZE * FACE_SIZE) as usize;
    let mut tensor_data = vec![0f32; 3 * size];
    let raw = resized.as_raw(); // &[u8], packed RGB

    for idx in 0..size {
        tensor_data[0 * size + idx] = (raw[idx * 3] as f32 - 127.5) / 128.0;
        tensor_data[1 * size + idx] = (raw[idx * 3 + 1] as f32 - 127.5) / 128.0;
        tensor_data[2 * size + idx] = (raw[idx * 3 + 2] as f32 - 127.5) / 128.0;
    }

    let shape = [1usize, 3, FACE_SIZE as usize, FACE_SIZE as usize];
    Ok(Tensor::from_array((shape, tensor_data.into_boxed_slice()))
        .context("failed to create face input tensor")?
        .into_dyn())
}

/// Extract the first output tensor's data as a flat `Vec<f32>`.
fn extract_first_embedding(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<f32>> {
    // ArcFace / MobileFaceNet: first output is the embedding vector
    let first_value = outputs
        .iter()
        .next()
        .context("ArcFace produced no outputs")?
        .1;

    let (_shape, data) = first_value
        .try_extract_tensor::<f32>()
        .context("failed to extract ArcFace embedding tensor")?;

    Ok(data.to_vec())
}

/// Crop the estimated face region from a full RGB frame given a person BBox.
fn crop_face(frame: &RgbFrame, bbox: BBox) -> RgbImage {
    use image::imageops::FilterType;

    let bh = bbox.height();
    let face_y1 = (bbox.y1.max(0.0) as u32).min(frame.height.saturating_sub(1));
    let face_h = ((bh * HEAD_FRACTION).max(1.0) as u32).min(frame.height - face_y1);
    let face_x1 = (bbox.x1.max(0.0) as u32).min(frame.width.saturating_sub(1));
    let face_w = (bbox.width().max(1.0) as u32).min(frame.width - face_x1);

    // Copy only the crop rows from the raw slice — O(crop area), no full-frame clone.
    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (face_w * 3) as usize;
    let mut buf = vec![0u8; dst_stride * face_h as usize];
    for row in 0..face_h as usize {
        let src_start = (face_y1 as usize + row) * src_stride + face_x1 as usize * 3;
        let dst_start = row * dst_stride;
        buf[dst_start..dst_start + dst_stride]
            .copy_from_slice(&frame.data[src_start..src_start + dst_stride]);
    }
    let cropped = RgbImage::from_raw(face_w, face_h, buf).expect("valid crop dimensions");

    image::imageops::resize(&cropped, FACE_SIZE, FACE_SIZE, FilterType::Triangle)
}

// ── Math helpers ─────────────────────────────────────────────────────────────

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter().map(|x| x / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Assumes both vectors are already L2-normalised.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Non-Maximum Suppression ──────────────────────────────────────────────────

/// Greedy NMS: sort by confidence descending, suppress overlapping boxes.
fn nms(mut boxes: Vec<BBox>, iou_thresh: f32) -> Vec<BBox> {
    boxes.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut kept: Vec<BBox> = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(boxes[i]);
        for j in (i + 1)..boxes.len() {
            if boxes[i].iou(&boxes[j]) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }

    kept
}

// ── Debug rendering ──────────────────────────────────────────────────────────

/// Draw bounding boxes onto a frame's RGB data in-place (for debug output).
pub fn draw_boxes(frame: &mut RgbFrame, boxes: &[BBox], color: [u8; 3]) {
    // Build the image from the existing buffer — no clone; we write back in-place.
    let mut img: RgbImage =
        ImageBuffer::from_raw(frame.width, frame.height, std::mem::take(&mut frame.data))
            .expect("valid frame dimensions");

    for bbox in boxes {
        let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size(bbox.width() as u32, bbox.height() as u32);
        imageproc::drawing::draw_hollow_rect_mut(&mut img, rect, Rgb(color));
    }

    frame.data = img.into_raw();
}
