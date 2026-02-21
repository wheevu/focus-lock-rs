//! detection — YOLOv8-Nano person detection + ArcFace face identification
//!
//! Phase 2: load yolov8n.onnx, run inference on a 640×640 frame, return
//!          bounding boxes for the "person" class after NMS.
//!
//! Phase 3: load arcface.onnx, embed a reference face, filter detections by
//!          cosine similarity.

use anyhow::{Context, Result};
use fast_image_resize as fr;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::rect::Rect;
use ort::execution_providers as ep;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use std::cell::RefCell;
use std::path::Path;
use std::sync::Mutex;
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
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.6;
/// Maximum number of person candidates to run ArcFace on per frame.
/// In concert scenes with 10-20 detections, this caps inference cost.
const MAX_FACE_CANDIDATES: usize = 8;
/// Approximate fraction range of the person bounding box height occupied by head.
const MIN_HEAD_FRACTION: f32 = 0.18;
const MAX_HEAD_FRACTION: f32 = 0.35;
/// Restrict face crop to the centered width to avoid shoulders/background.
const FACE_WIDTH_FRACTION: f32 = 0.72;
/// For very high-resolution inputs, run person detection on a downscaled frame
/// and map detections back to source coordinates.
const DETECTION_MAX_DIM: u32 = 1920;
/// Number of high-confidence matches to fold into the running identity gallery.
const MAX_GALLERY_SAMPLES: usize = 12;
/// Similarity margin above threshold required to update gallery prototype.
const GALLERY_UPDATE_MARGIN: f32 = 0.08;

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
    yolo_resizer: fr::Resizer,
    yolo_resize_buf: Vec<u8>,
    downscale_resizer: fr::Resizer,
    downscale_buf: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityMatch {
    pub bbox: BBox,
    pub similarity: f32,
}

impl Detector {
    /// Load a YOLOv8n ONNX model from `model_path`.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = build_ort_session(model_path.as_ref(), "failed to load YOLOv8 ONNX model")?;
        Ok(Self {
            session,
            yolo_resizer: fr::Resizer::new(),
            yolo_resize_buf: vec![0u8; (YOLO_SIZE * YOLO_SIZE * 3) as usize],
            downscale_resizer: fr::Resizer::new(),
            downscale_buf: Vec::new(),
        })
    }

    /// Run inference on `frame` and return bounding boxes (in original frame
    /// pixel coordinates) for all persons after NMS.
    pub fn detect(&mut self, frame: &RgbFrame) -> Result<Vec<BBox>> {
        let max_dim = frame.width.max(frame.height);
        if max_dim > DETECTION_MAX_DIM {
            let scale = DETECTION_MAX_DIM as f32 / max_dim as f32;
            let scaled_w = ((frame.width as f32 * scale).round() as u32).max(1);
            let scaled_h = ((frame.height as f32 * scale).round() as u32).max(1);
            let mut scaled = self.downscale_frame(frame, scaled_w, scaled_h)?;
            let mut boxes = self.detect_native(&scaled)?;
            self.downscale_buf = std::mem::take(&mut scaled.data);
            let sx = scaled_w as f32 / frame.width as f32;
            let sy = scaled_h as f32 / frame.height as f32;
            for b in &mut boxes {
                b.x1 /= sx;
                b.x2 /= sx;
                b.y1 /= sy;
                b.y2 /= sy;
            }
            return Ok(boxes);
        }

        self.detect_native(frame)
    }

    fn detect_native(&mut self, frame: &RgbFrame) -> Result<Vec<BBox>> {
        let input_tensor = self.preprocess_yolo(frame)?;

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

        let candidates: Vec<BBox> = (0..num_proposals)
            .into_par_iter()
            .filter_map(|i| {
                // Data layout: [cx, cy, w, h, cls0_score, cls1_score, ...]
                // Stored column-major across the 84 rows.
                let cx = data[i];
                let cy = data[num_proposals + i];
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
                    return None;
                }

                // Convert YOLO (cx,cy,w,h) in 640-space → (x1,y1,x2,y2) in original frame
                let x1 = (cx - w / 2.0) * scale_x;
                let y1 = (cy - h / 2.0) * scale_y;
                let x2 = (cx + w / 2.0) * scale_x;
                let y2 = (cy + h / 2.0) * scale_y;

                Some(BBox {
                    x1: x1.max(0.0),
                    y1: y1.max(0.0),
                    x2: x2.min(frame.width as f32),
                    y2: y2.min(frame.height as f32),
                    confidence: person_score,
                })
            })
            .collect();

        Ok(nms(candidates, IOU_THRESHOLD))
    }

    fn preprocess_yolo(&mut self, frame: &RgbFrame) -> Result<ort::value::DynValue> {
        // Use fast_image_resize with NEON SIMD for the large 4K → 640x640 downscale.
        let src =
            fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
                .context("failed to create fast_image_resize source")?;

        let mut dst = fr::images::Image::from_vec_u8(
            YOLO_SIZE,
            YOLO_SIZE,
            std::mem::take(&mut self.yolo_resize_buf),
            fr::PixelType::U8x3,
        )
        .context("failed to create fast_image_resize destination")?;

        let options = fr::ResizeOptions::new()
            .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        self.yolo_resizer
            .resize(&src, &mut dst, Some(&options))
            .context("fast_image_resize YOLO downscale failed")?;

        self.yolo_resize_buf = dst.into_vec();
        let raw = &self.yolo_resize_buf;

        // NCHW float tensor: [1, 3, 640, 640].
        let size = (YOLO_SIZE * YOLO_SIZE) as usize;
        let mut tensor_data = vec![0f32; 3 * size];

        let (r_plane, gb_plane) = tensor_data.split_at_mut(size);
        let (g_plane, b_plane) = gb_plane.split_at_mut(size);
        rayon::join(
            || {
                r_plane
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, out)| *out = raw[idx * 3] as f32 / 255.0)
            },
            || {
                rayon::join(
                    || {
                        g_plane
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(idx, out)| *out = raw[idx * 3 + 1] as f32 / 255.0)
                    },
                    || {
                        b_plane
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(idx, out)| *out = raw[idx * 3 + 2] as f32 / 255.0)
                    },
                )
            },
        );

        let shape = [1usize, 3, YOLO_SIZE as usize, YOLO_SIZE as usize];
        Ok(Tensor::from_array((shape, tensor_data.into_boxed_slice()))
            .context("failed to create YOLO input tensor")?
            .into_dyn())
    }

    fn downscale_frame(&mut self, frame: &RgbFrame, out_w: u32, out_h: u32) -> Result<RgbFrame> {
        let src =
            fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
                .context("failed to create detection downscale source")?;

        let out_len = (out_w * out_h * 3) as usize;
        if self.downscale_buf.len() != out_len {
            self.downscale_buf.resize(out_len, 0);
        }

        let mut dst = fr::images::Image::from_vec_u8(
            out_w,
            out_h,
            std::mem::take(&mut self.downscale_buf),
            fr::PixelType::U8x3,
        )
        .context("failed to create detection downscale destination")?;

        let options = fr::ResizeOptions::new()
            .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        self.downscale_resizer
            .resize(&src, &mut dst, Some(&options))
            .context("failed to downscale frame for detection")?;

        let scaled_data = dst.into_vec();
        Ok(RgbFrame {
            data: scaled_data,
            width: out_w,
            height: out_h,
            pts: frame.pts,
        })
    }
}

// ── Face identifier ──────────────────────────────────────────────────────────

/// Wraps the ArcFace ONNX session and the reference embedding.
pub struct FaceIdentifier {
    sessions: Vec<Mutex<Session>>,
    reference_embedding: Vec<f32>,
    similarity_threshold: f32,
    gallery_samples: usize,
}

impl FaceIdentifier {
    /// Load an ArcFace ONNX model and embed `reference_image_path` as the
    /// target identity.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        reference_image_path: Q,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let session_count = MAX_FACE_CANDIDATES;
        let mut sessions = Vec::with_capacity(session_count);
        for _ in 0..session_count {
            let session = build_ort_session(&model_path, "failed to load ArcFace ONNX model")?;
            sessions.push(Mutex::new(session));
        }

        // Load and embed the reference image (assumed to be a face crop)
        let ref_img = image::open(reference_image_path)
            .context("failed to open reference image")?
            .into_rgb8();

        let tensor = preprocess_face(&ref_img)?;
        let reference_embedding = {
            let mut session = sessions
                .first()
                .context("ArcFace session pool is empty")?
                .lock()
                .map_err(|_| anyhow::anyhow!("ArcFace session lock poisoned"))?;
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
            sessions,
            reference_embedding,
            similarity_threshold,
            gallery_samples: 1,
        })
    }

    /// Given a full RGB frame and a list of person bounding boxes, return the
    /// bbox of the target identity (highest similarity above threshold), or
    /// `None` if not found.
    ///
    /// To limit per-frame latency in crowded scenes, only the top
    /// `MAX_FACE_CANDIDATES` persons (by detection confidence) are evaluated.
    pub fn identify(
        &mut self,
        frame: &RgbFrame,
        persons: &[BBox],
        search_hint: Option<(f32, f32)>,
    ) -> Result<Option<IdentityMatch>> {
        let mut best: Option<(f32, BBox, Vec<f32>)> = None;

        // Sort candidates by confidence descending and cap at MAX_FACE_CANDIDATES.
        let mut candidates: Vec<BBox> = persons.to_vec();
        candidates.sort_unstable_by(|a, b| {
            rank_candidate(*b, search_hint)
                .partial_cmp(&rank_candidate(*a, search_hint))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(MAX_FACE_CANDIDATES);

        let prepared: Vec<(BBox, Vec<f32>)> = candidates
            .par_iter()
            .filter_map(|&bbox| {
                preprocess_face_from_bbox(frame, bbox)
                    .ok()
                    .map(|data| (bbox, data))
            })
            .collect();

        let sessions = &self.sessions;
        let reference_embedding = &self.reference_embedding;
        let similarity_threshold = self.similarity_threshold;

        let matches: Vec<(f32, BBox, Vec<f32>)> = prepared
            .into_par_iter()
            .enumerate()
            .filter_map(|(i, (bbox, tensor_data))| {
                let tensor = face_tensor_from_data(tensor_data).ok()?;
                let idx = i % sessions.len();
                let embedding = {
                    let mut session = sessions.get(idx)?.lock().ok()?;
                    let outputs = session.run(ort::inputs!["input.1" => tensor]).ok()?;
                    l2_normalize(&extract_first_embedding(&outputs).ok()?)
                };
                let sim = cosine_similarity(&embedding, reference_embedding);
                debug!(sim, "face similarity");
                if sim >= similarity_threshold {
                    Some((sim, bbox, embedding))
                } else {
                    None
                }
            })
            .collect();

        for (sim, bbox, embedding) in matches {
            if best.as_ref().is_none_or(|b| sim > b.0) {
                best = Some((sim, bbox, embedding));
            }
        }

        if let Some((sim, bbox, embedding)) = best {
            if sim >= self.similarity_threshold + GALLERY_UPDATE_MARGIN
                && self.gallery_samples < MAX_GALLERY_SAMPLES
            {
                self.update_reference_gallery(&embedding);
            }
            return Ok(Some(IdentityMatch {
                bbox,
                similarity: sim,
            }));
        }

        Ok(None)
    }

    fn update_reference_gallery(&mut self, embedding: &[f32]) {
        let prev_weight = self.gallery_samples as f32;
        let next_weight = prev_weight + 1.0;
        for (r, e) in self.reference_embedding.iter_mut().zip(embedding.iter()) {
            *r = (*r * prev_weight + *e) / next_weight;
        }
        self.reference_embedding = l2_normalize(&self.reference_embedding);
        self.gallery_samples += 1;
    }
}

// ── Pre/post-processing helpers ──────────────────────────────────────────────

/// Crop the approximate head region from a person BBox, resize to 112×112,
/// normalise to [-1, 1], return as an ORT `Tensor`.
fn preprocess_face(img: &RgbImage) -> Result<ort::value::DynValue> {
    let tensor_data = preprocess_face_data(img)?;
    face_tensor_from_data(tensor_data)
}

thread_local! {
    static FACE_RESIZER: RefCell<fr::Resizer> = RefCell::new(fr::Resizer::new());
    static FACE_RESIZE_BUF: RefCell<Vec<u8>> = RefCell::new(vec![0u8; (FACE_SIZE * FACE_SIZE * 3) as usize]);
    static FACE_CROP_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn preprocess_face_data(img: &RgbImage) -> Result<Vec<f32>> {
    preprocess_face_data_from_raw(img.width(), img.height(), img.as_raw())
}

fn preprocess_face_data_from_raw(width: u32, height: u32, raw_rgb: &[u8]) -> Result<Vec<f32>> {
    let src = fr::images::ImageRef::new(width, height, raw_rgb, fr::PixelType::U8x3)
        .context("failed to create face resize source")?;

    // NCHW float tensor: [1, 3, 112, 112].
    // Use flat indexed access over raw bytes — avoids per-pixel Pixel overhead.
    let size = (FACE_SIZE * FACE_SIZE) as usize;
    let mut tensor_data = vec![0f32; 3 * size];

    FACE_RESIZER.with(|resizer_cell| {
        FACE_RESIZE_BUF.with(|buf_cell| -> Result<()> {
            let mut resize_buf = buf_cell.borrow_mut();
            let mut dst = fr::images::Image::from_vec_u8(
                FACE_SIZE,
                FACE_SIZE,
                std::mem::take(&mut *resize_buf),
                fr::PixelType::U8x3,
            )
            .context("failed to create face resize destination")?;

            let options = fr::ResizeOptions::new()
                .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
            resizer_cell
                .borrow_mut()
                .resize(&src, &mut dst, Some(&options))
                .context("fast_image_resize face resize failed")?;

            *resize_buf = dst.into_vec();
            for idx in 0..size {
                tensor_data[idx] = (resize_buf[idx * 3] as f32 - 127.5) / 128.0;
                tensor_data[size + idx] = (resize_buf[idx * 3 + 1] as f32 - 127.5) / 128.0;
                tensor_data[2 * size + idx] = (resize_buf[idx * 3 + 2] as f32 - 127.5) / 128.0;
            }
            Ok(())
        })
    })?;

    Ok(tensor_data)
}

fn preprocess_face_from_bbox(frame: &RgbFrame, bbox: BBox) -> Result<Vec<f32>> {
    let (face_x1, face_y1, face_w, face_h) = face_crop_region(frame, bbox);
    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (face_w * 3) as usize;
    let crop_len = dst_stride * face_h as usize;

    FACE_CROP_BUF.with(|crop_cell| {
        let mut crop_buf = crop_cell.borrow_mut();
        if crop_buf.len() != crop_len {
            crop_buf.resize(crop_len, 0);
        }

        for row in 0..face_h as usize {
            let src_start = (face_y1 as usize + row) * src_stride + face_x1 as usize * 3;
            let dst_start = row * dst_stride;
            crop_buf[dst_start..dst_start + dst_stride]
                .copy_from_slice(&frame.data[src_start..src_start + dst_stride]);
        }

        preprocess_face_data_from_raw(face_w, face_h, &crop_buf)
    })
}

fn face_tensor_from_data(tensor_data: Vec<f32>) -> Result<ort::value::DynValue> {
    let shape = [1usize, 3, FACE_SIZE as usize, FACE_SIZE as usize];
    Ok(Tensor::from_array((shape, tensor_data.into_boxed_slice()))
        .context("failed to create face input tensor")?
        .into_dyn())
}

fn build_ort_session(model_path: &Path, load_error: &'static str) -> Result<Session> {
    let mut builder = Session::builder().context("failed to create ORT session builder")?;
    builder = builder
        .with_intra_threads(1)
        .context("failed to set ORT intra threads")?;
    builder = builder
        .with_inter_threads(1)
        .context("failed to set ORT inter threads")?;
    builder = builder
        .with_parallel_execution(false)
        .context("failed to set ORT parallel execution")?;
    builder = builder
        .with_execution_providers([ep::CoreMLExecutionProvider::default()
            .with_compute_units(ep::coreml::CoreMLComputeUnits::CPUAndNeuralEngine)
            .build()])
        .context("failed to register execution providers")?;
    builder.commit_from_file(model_path).context(load_error)
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

fn face_crop_region(frame: &RgbFrame, bbox: BBox) -> (u32, u32, u32, u32) {
    let bw = bbox.width().max(1.0);
    let bh = bbox.height().max(1.0);
    let aspect = (bw / bh).clamp(0.2, 2.5);

    // Wider boxes likely include shoulders/background; reduce head fraction.
    let mut head_fraction =
        (0.30 - (aspect - 0.5) * 0.06).clamp(MIN_HEAD_FRACTION, MAX_HEAD_FRACTION);
    if bh < 170.0 {
        // Distant subjects need a taller crop to preserve enough face pixels.
        head_fraction = (head_fraction + 0.04).clamp(MIN_HEAD_FRACTION, MAX_HEAD_FRACTION);
    }

    let face_w_f = (bw * FACE_WIDTH_FRACTION).max(1.0);
    let face_h_f = (bh * head_fraction).max(1.0);
    let face_x1_f = bbox.center_x() - face_w_f / 2.0;
    let face_y1_f = bbox.y1 + bh * 0.02;

    let face_x1 = face_x1_f.max(0.0) as u32;
    let face_y1 = face_y1_f.max(0.0) as u32;
    let face_w = (face_w_f as u32)
        .min(frame.width.saturating_sub(face_x1))
        .max(1);
    let face_h = (face_h_f as u32)
        .min(frame.height.saturating_sub(face_y1))
        .max(1);

    (face_x1, face_y1, face_w, face_h)
}

fn rank_candidate(bbox: BBox, search_hint: Option<(f32, f32)>) -> f32 {
    let mut score = bbox.confidence;
    if let Some((hx, hy)) = search_hint {
        let dx = bbox.center_x() - hx;
        let dy = bbox.center_y() - hy;
        let distance = (dx * dx + dy * dy).sqrt();
        let norm = (bbox.width().max(bbox.height()) * 4.0).max(1.0);
        let proximity = 1.0 - (distance / norm).clamp(0.0, 1.0);
        score += proximity * 0.35;
    }
    score
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
