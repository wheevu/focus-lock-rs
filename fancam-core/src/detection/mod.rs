//! detection — YOLOv8-Nano person detection + ArcFace face identification
//!
//! Phase 2: load yolov8n.onnx, run inference on a 640×640 frame, return
//!          bounding boxes for the "person" class after NMS.
//!
//! Phase 3: load arcface.onnx, embed a reference face, filter detections by
//!          cosine similarity.

use fast_image_resize as fr;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::rect::Rect;
use ort::execution_providers as ep;
use ort::execution_providers::ExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use std::cell::RefCell;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;
use tracing::debug;

use crate::video::RgbFrame;
use crate::{PoisonExt, Result};

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
/// Recovery-mode candidate budget to avoid starvation under occlusion.
const MAX_FACE_CANDIDATES_RECOVERY: usize = 12;
/// Approximate fraction range of the person bounding box height occupied by head.
const MIN_HEAD_FRACTION: f32 = 0.18;
const MAX_HEAD_FRACTION: f32 = 0.35;
/// Restrict face crop to the centered width to avoid shoulders/background.
const FACE_WIDTH_FRACTION: f32 = 0.72;
/// For very high-resolution inputs, run person detection on a downscaled frame
/// and map detections back to source coordinates.
const DETECTION_MAX_DIM: u32 = 1920;
/// Minimum margin between target and impostor similarities for a valid identity.
pub const DEFAULT_IDENTITY_MARGIN_THRESHOLD: f32 = 0.03;
/// Spatial gate (in candidate-box sizes) around the tracker hint.
const SEARCH_GATE_SCALE: f32 = 8.0;
/// Minimum absolute search gate radius in pixels.
const SEARCH_GATE_MIN_RADIUS: f32 = 140.0;
/// Maximum absolute search gate radius in pixels.
const SEARCH_GATE_MAX_RADIUS: f32 = 440.0;
/// Fallback candidates when the search gate rejects all detections.
const SEARCH_GATE_FALLBACK_KEEP: usize = 3;

// ── Public types ─────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in pixel coordinates of the original frame.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    /// Left edge X coordinate (inclusive).
    pub x1: f32,
    /// Top edge Y coordinate (inclusive).
    pub y1: f32,
    /// Right edge X coordinate (exclusive).
    pub x2: f32,
    /// Bottom edge Y coordinate (exclusive).
    pub y2: f32,
    /// Detection confidence score (0.0-1.0).
    pub confidence: f32,
}

impl BBox {
    /// Returns the width of the bounding box.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }
    /// Returns the height of the bounding box.
    #[must_use]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }
    /// Returns the center X coordinate.
    #[must_use]
    pub fn center_x(&self) -> f32 {
        (self.x1 + self.x2) / 2.0
    }
    /// Returns the center Y coordinate.
    #[must_use]
    pub fn center_y(&self) -> f32 {
        (self.y1 + self.y2) / 2.0
    }
    /// IoU (intersection over union) with another box.
    #[must_use]
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
#[derive(Debug)]
pub struct Detector {
    session: Session,
    yolo_resizer: fr::Resizer,
    yolo_resize_buf: Vec<u8>,
    yolo_letterbox_buf: Vec<u8>,
    downscale_resizer: fr::Resizer,
    downscale_buf: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct YoloMap {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

/// A matched identity with bounding box and similarity score.
#[derive(Debug, Clone, Copy)]
pub struct IdentityMatch {
    /// The bounding box of the matched identity.
    pub bbox: BBox,
    /// The cosine similarity score (0.0-1.0).
    pub similarity: f32,
}

/// Scored identity observation for one person candidate in a frame.
#[derive(Debug, Clone, Copy)]
pub struct FaceObservation {
    /// Candidate person bounding box.
    pub bbox: BBox,
    /// Best cosine similarity against positive target gallery.
    pub similarity: f32,
    /// Best cosine similarity against negative/impostor gallery.
    pub impostor_similarity: f32,
    /// Similarity margin (`similarity - impostor_similarity`).
    pub margin: f32,
    /// Optional body re-identification similarity for this candidate.
    pub body_similarity: Option<f32>,
}

impl Detector {
    /// Load a YOLOv8n ONNX model from `model_path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file cannot be loaded or the ORT session
    /// cannot be created.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = build_ort_session(model_path.as_ref())
            .map_err(|e| crate::FancamError::model_load(model_path.as_ref(), e))?;
        Ok(Self {
            session,
            yolo_resizer: fr::Resizer::new(),
            yolo_resize_buf: Vec::new(),
            yolo_letterbox_buf: vec![114u8; (YOLO_SIZE * YOLO_SIZE * 3) as usize],
            downscale_resizer: fr::Resizer::new(),
            downscale_buf: Vec::new(),
        })
    }

    /// Run inference on `frame` and return bounding boxes (in original frame
    /// pixel coordinates) for all persons after NMS.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails or frame preprocessing fails.
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
        let (input_tensor, yolo_map) = self.preprocess_yolo(frame)?;

        let outputs = self
            .session
            .run(ort::inputs!["images" => input_tensor])
            .map_err(|e| crate::FancamError::inference(format!("YOLOv8 inference failed: {e}")))?;

        // YOLOv8 output: [1, 84, 8400]  (84 = 4 box coords + 80 class scores)
        let (_shape, data) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                crate::FancamError::inference(format!(
                    "Failed to extract YOLOv8 output tensor: {e}"
                ))
            })?;

        // shape = [1, 84, 8400]
        // num_proposals = 8400, num_classes = 80
        const NUM_PROPOSALS: usize = 8400;

        let candidates: Vec<BBox> = (0..NUM_PROPOSALS)
            .into_par_iter()
            .filter_map(|i| {
                // Data layout: [cx, cy, w, h, cls0_score, cls1_score, ...]
                // Stored column-major across the 84 rows.
                let cx = data[i];
                let cy = data[NUM_PROPOSALS + i];
                let w = data[2 * NUM_PROPOSALS + i];
                let h = data[3 * NUM_PROPOSALS + i];

                // Person score (class 0)
                let person_score = data[(4 + PERSON_CLASS) * NUM_PROPOSALS + i];

                if person_score < CONF_THRESHOLD {
                    return None;
                }

                // Convert YOLO (cx,cy,w,h) in 640-space back through letterbox map.
                let x1 = (cx - w / 2.0 - yolo_map.pad_x) / yolo_map.scale;
                let y1 = (cy - h / 2.0 - yolo_map.pad_y) / yolo_map.scale;
                let x2 = (cx + w / 2.0 - yolo_map.pad_x) / yolo_map.scale;
                let y2 = (cy + h / 2.0 - yolo_map.pad_y) / yolo_map.scale;

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

    fn preprocess_yolo(&mut self, frame: &RgbFrame) -> Result<(ort::value::DynValue, YoloMap)> {
        // Use fast_image_resize with NEON SIMD for the large 4K → 640x640 downscale.
        let src =
            fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
                .map_err(|e| {
                    crate::FancamError::image_processing(format!(
                        "Failed to create fast_image_resize source: {e}"
                    ))
                })?;

        let scale = (YOLO_SIZE as f32 / frame.width as f32)
            .min(YOLO_SIZE as f32 / frame.height as f32)
            .max(1e-6);
        let resize_w = (frame.width as f32 * scale)
            .round()
            .clamp(1.0, YOLO_SIZE as f32) as u32;
        let resize_h = (frame.height as f32 * scale)
            .round()
            .clamp(1.0, YOLO_SIZE as f32) as u32;

        let resized_len = (resize_w * resize_h * 3) as usize;
        if self.yolo_resize_buf.len() != resized_len {
            self.yolo_resize_buf.resize(resized_len, 0);
        }
        let mut dst = fr::images::Image::from_vec_u8(
            resize_w,
            resize_h,
            std::mem::take(&mut self.yolo_resize_buf),
            fr::PixelType::U8x3,
        )
        .map_err(|e| {
            crate::FancamError::image_processing(format!(
                "Failed to create fast_image_resize destination: {e}"
            ))
        })?;

        let options = fr::ResizeOptions::new()
            .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        self.yolo_resizer
            .resize(&src, &mut dst, Some(&options))
            .map_err(|e| {
                crate::FancamError::image_processing(format!(
                    "fast_image_resize YOLO downscale failed: {e}"
                ))
            })?;

        self.yolo_resize_buf = dst.into_vec();

        let letterbox_len = (YOLO_SIZE * YOLO_SIZE * 3) as usize;
        if self.yolo_letterbox_buf.len() != letterbox_len {
            self.yolo_letterbox_buf.resize(letterbox_len, 114);
        }
        self.yolo_letterbox_buf.fill(114);

        let pad_x = ((YOLO_SIZE - resize_w) / 2) as usize;
        let pad_y = ((YOLO_SIZE - resize_h) / 2) as usize;
        let src_stride = resize_w as usize * 3;
        let dst_stride = YOLO_SIZE as usize * 3;
        for row in 0..resize_h as usize {
            let src_start = row * src_stride;
            let dst_start = (pad_y + row) * dst_stride + pad_x * 3;
            self.yolo_letterbox_buf[dst_start..dst_start + src_stride]
                .copy_from_slice(&self.yolo_resize_buf[src_start..src_start + src_stride]);
        }

        let raw = &self.yolo_letterbox_buf;

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
        Tensor::from_array((shape, tensor_data.into_boxed_slice()))
            .map(|t| t.into_dyn())
            .map_err(|e| {
                crate::FancamError::inference(format!("Failed to create YOLO input tensor: {e}"))
            })
            .map(|tensor| {
                (
                    tensor,
                    YoloMap {
                        scale,
                        pad_x: pad_x as f32,
                        pad_y: pad_y as f32,
                    },
                )
            })
    }

    fn downscale_frame(&mut self, frame: &RgbFrame, out_w: u32, out_h: u32) -> Result<RgbFrame> {
        let src =
            fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
                .map_err(|e| {
                    crate::FancamError::image_processing(format!(
                        "Failed to create detection downscale source: {e}"
                    ))
                })?;

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
        .map_err(|e| {
            crate::FancamError::image_processing(format!(
                "Failed to create detection downscale destination: {e}"
            ))
        })?;

        let options = fr::ResizeOptions::new()
            .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        self.downscale_resizer
            .resize(&src, &mut dst, Some(&options))
            .map_err(|e| {
                crate::FancamError::image_processing(format!(
                    "Failed to downscale frame for detection: {e}"
                ))
            })?;

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
#[derive(Debug)]
pub struct FaceIdentifier {
    sessions: Vec<Mutex<Session>>,
    target_embeddings: Vec<Vec<f32>>,
    negative_embeddings: Vec<Vec<f32>>,
    similarity_threshold: f32,
    margin_threshold: f32,
}

/// ArcFace embedding helper that does not require a reference image.
///
/// Used by identity discovery passes to build candidate clusters before the user
/// chooses which member to track.
#[derive(Debug)]
pub struct FaceEmbedder {
    sessions: Vec<Mutex<Session>>,
}

/// Number of ArcFace sessions used by discovery embedding.
///
/// A small pool improves throughput on crowded sampled frames without creating
/// excessive memory pressure in local development.
const DISCOVERY_EMBEDDER_POOL: usize = 4;

impl FaceEmbedder {
    /// Load an ArcFace model for embedding extraction.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let mut sessions = Vec::with_capacity(DISCOVERY_EMBEDDER_POOL);
        for _ in 0..DISCOVERY_EMBEDDER_POOL {
            let session = build_ort_session(&model_path)
                .map_err(|e| crate::FancamError::model_load(&model_path, e))?;
            sessions.push(Mutex::new(session));
        }
        Ok(Self { sessions })
    }

    /// Extract a face embedding from a bounding box.
    ///
    /// # Errors
    ///
    /// Returns an error if the face cannot be preprocessed or inference fails.
    pub fn embed_from_bbox(&self, frame: &RgbFrame, bbox: BBox) -> Result<Option<Vec<f32>>> {
        self.embed_many_from_bboxes(frame, &[bbox], 1)
            .map(|rows| rows.into_iter().next().map(|(_, emb)| emb))
    }

    /// Extract embeddings for many candidate bounding boxes.
    ///
    /// Work is parallelized across a small ArcFace session pool to improve
    /// discovery throughput on crowded scenes.
    ///
    /// # Errors
    ///
    /// Returns an error if face tensor construction fails.
    pub fn embed_many_from_bboxes(
        &self,
        frame: &RgbFrame,
        bboxes: &[BBox],
        min_face_crop_edge: u32,
    ) -> Result<Vec<(BBox, Vec<f32>)>> {
        if bboxes.is_empty() {
            return Ok(Vec::new());
        }

        let sessions = &self.sessions;
        let rows = bboxes
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &bbox)| {
                let min_edge = NonZeroU32::new(min_face_crop_edge.max(1))?;
                let tensor_data =
                    preprocess_face_from_bbox_with_min(frame, bbox, min_edge).ok()??;
                let tensor = face_tensor_from_data(tensor_data).ok()?;
                let session_idx = idx % sessions.len();
                let embedding = {
                    let mut session = sessions.get(session_idx)?.lock().ok()?;
                    let outputs = session.run(ort::inputs!["input.1" => tensor]).ok()?;
                    l2_normalize(&extract_first_embedding(&outputs).ok()?)
                };
                Some((bbox, embedding))
            })
            .collect::<Vec<_>>();

        Ok(rows)
    }
}

fn preprocess_face_from_bbox_with_min(
    frame: &RgbFrame,
    bbox: BBox,
    min_edge: NonZeroU32,
) -> Result<Option<Vec<f32>>> {
    let (face_x1, face_y1, face_w, face_h) = face_crop_region(frame, bbox);
    if face_w < min_edge.get() || face_h < min_edge.get() {
        return Ok(None);
    }

    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (face_w * 3) as usize;
    let crop_len = dst_stride * face_h as usize;

    FACE_CROP_BUF.with(|crop_cell| {
        let mut crop_buf = crop_cell.borrow_mut();
        let current_cap = crop_buf.capacity();
        if current_cap < crop_len {
            crop_buf.reserve(crop_len - current_cap);
        }
        crop_buf.resize(crop_len, 0);

        for row in 0..face_h as usize {
            let src_start = (face_y1 as usize + row) * src_stride + face_x1 as usize * 3;
            let dst_start = row * dst_stride;
            crop_buf[dst_start..dst_start + dst_stride]
                .copy_from_slice(&frame.data[src_start..src_start + dst_stride]);
        }

        preprocess_face_data_from_raw(face_w, face_h, &crop_buf).map(Some)
    })
}

/// Compute cosine similarity between two embeddings.
#[must_use]
pub fn embedding_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity(a, b)
}

impl FaceIdentifier {
    /// Current minimum similarity required for target acceptance.
    #[must_use]
    pub const fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Current minimum positive-vs-negative margin required for acceptance.
    #[must_use]
    pub const fn margin_threshold(&self) -> f32 {
        self.margin_threshold
    }

    fn create_sessions(model_path: &Path, session_count: usize) -> Result<Vec<Mutex<Session>>> {
        let mut sessions = Vec::with_capacity(session_count.max(1));
        for _ in 0..session_count.max(1) {
            let session = build_ort_session(model_path)
                .map_err(|e| crate::FancamError::model_load(model_path, e))?;
            sessions.push(Mutex::new(session));
        }
        Ok(sessions)
    }

    fn from_reference_embedding<P: AsRef<Path>>(
        model_path: P,
        reference_embedding: Vec<f32>,
        similarity_threshold: f32,
    ) -> Result<Self> {
        if reference_embedding.is_empty() {
            return Err(crate::FancamError::invalid_config(
                "reference embedding is empty",
            ));
        }
        let model_path = model_path.as_ref().to_path_buf();
        let sessions = Self::create_sessions(&model_path, MAX_FACE_CANDIDATES)?;

        Ok(Self {
            sessions,
            target_embeddings: vec![l2_normalize(&reference_embedding)],
            negative_embeddings: Vec::new(),
            similarity_threshold,
            margin_threshold: DEFAULT_IDENTITY_MARGIN_THRESHOLD,
        })
    }

    fn with_galleries(
        mut self,
        target_embeddings: Vec<Vec<f32>>,
        negative_embeddings: Vec<Vec<f32>>,
        margin_threshold: f32,
    ) -> Result<Self> {
        if target_embeddings.is_empty() {
            return Err(crate::FancamError::invalid_config(
                "target embedding gallery is empty",
            ));
        }
        self.target_embeddings = target_embeddings
            .into_iter()
            .map(|emb| l2_normalize(&emb))
            .collect();
        self.negative_embeddings = negative_embeddings
            .into_iter()
            .filter(|emb| !emb.is_empty())
            .map(|emb| l2_normalize(&emb))
            .collect();
        self.margin_threshold = margin_threshold.max(0.0);
        Ok(self)
    }

    /// Load an ArcFace ONNX model and embed `reference_image_path` as the
    /// target identity.
    ///
    /// # Errors
    ///
    /// Returns an error if models cannot be loaded or reference image cannot be processed.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        reference_image_path: Q,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let sessions = Self::create_sessions(&model_path, MAX_FACE_CANDIDATES)?;

        // Load and embed the reference image (assumed to be a face crop)
        let ref_img = image::open(&reference_image_path)
            .map_err(|e| {
                crate::FancamError::image_processing(format!("Failed to open reference image: {e}"))
            })?
            .into_rgb8();

        let tensor = preprocess_face(&ref_img)?;
        let reference_embedding = {
            let mut session = sessions
                .first()
                .ok_or_else(|| crate::FancamError::invalid_config("ArcFace session pool is empty"))?
                .lock()
                .to_fancam_err("ArcFace session lock poisoned")?;
            let outputs = session
                .run(ort::inputs!["input.1" => tensor])
                .map_err(|e| {
                    crate::FancamError::inference(format!(
                        "ArcFace reference inference failed: {e}"
                    ))
                })?;
            let embedding = extract_first_embedding(&outputs)?;
            l2_normalize(&embedding)
        };

        debug!(
            dim = reference_embedding.len(),
            "reference face embedding computed"
        );

        Ok(Self {
            sessions,
            target_embeddings: vec![reference_embedding],
            negative_embeddings: Vec::new(),
            similarity_threshold,
            margin_threshold: DEFAULT_IDENTITY_MARGIN_THRESHOLD,
        })
    }

    /// Load an ArcFace ONNX model with an already-computed reference embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or embedding is invalid.
    pub fn load_from_embedding<P: AsRef<Path>>(
        model_path: P,
        reference_embedding: Vec<f32>,
        similarity_threshold: f32,
    ) -> Result<Self> {
        Self::from_reference_embedding(model_path, reference_embedding, similarity_threshold)
    }

    /// Load ArcFace with explicit positive and negative identity galleries.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails or target gallery is empty.
    pub fn load_from_galleries<P: AsRef<Path>>(
        model_path: P,
        target_embeddings: Vec<Vec<f32>>,
        negative_embeddings: Vec<Vec<f32>>,
        similarity_threshold: f32,
        margin_threshold: f32,
    ) -> Result<Self> {
        let seed = target_embeddings.first().cloned().ok_or_else(|| {
            crate::FancamError::invalid_config("target embedding gallery is empty")
        })?;
        let identifier =
            Self::from_reference_embedding(model_path, seed, similarity_threshold.clamp(0.0, 1.0))?;
        identifier.with_galleries(target_embeddings, negative_embeddings, margin_threshold)
    }

    /// Compute scored face observations for person candidates.
    ///
    /// The output is sorted by target similarity (desc), then by margin (desc).
    ///
    /// # Errors
    ///
    /// Returns an error if face processing or inference fails.
    pub fn observations(
        &self,
        frame: &RgbFrame,
        persons: &[BBox],
        search_hint: Option<(f32, f32)>,
    ) -> Result<Vec<FaceObservation>> {
        self.observations_with_candidate_budget(frame, persons, search_hint, MAX_FACE_CANDIDATES)
    }

    /// Compute scored face observations with an explicit candidate budget.
    ///
    /// This is used by recovery paths to temporarily evaluate more detections
    /// without changing steady-state cost.
    ///
    /// # Errors
    ///
    /// Returns an error if face processing or inference fails.
    pub fn observations_with_candidate_budget(
        &self,
        frame: &RgbFrame,
        persons: &[BBox],
        search_hint: Option<(f32, f32)>,
        max_candidates: usize,
    ) -> Result<Vec<FaceObservation>> {
        let mut candidates: Vec<BBox> = persons.to_vec();
        candidates.sort_unstable_by(|a, b| {
            rank_candidate(*b, search_hint)
                .partial_cmp(&rank_candidate(*a, search_hint))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut candidates = apply_search_gate(candidates, search_hint);
        candidates.truncate(max_candidates.max(1));
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let prepared: Vec<(BBox, Vec<f32>)> = candidates
            .par_iter()
            .filter_map(|&bbox| {
                preprocess_face_from_bbox(frame, bbox)
                    .ok()
                    .map(|data| (bbox, data))
            })
            .collect();

        let sessions = &self.sessions;
        let target_gallery = &self.target_embeddings;
        let negative_gallery = &self.negative_embeddings;

        let mut observations: Vec<FaceObservation> = prepared
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

                let similarity = target_gallery
                    .iter()
                    .map(|target| cosine_similarity(&embedding, target))
                    .fold(f32::NEG_INFINITY, f32::max);
                if !similarity.is_finite() {
                    return None;
                }

                let impostor_similarity = negative_gallery
                    .iter()
                    .map(|neg| cosine_similarity(&embedding, neg))
                    .fold(f32::NEG_INFINITY, f32::max);
                let impostor_similarity = if impostor_similarity.is_finite() {
                    impostor_similarity
                } else {
                    -1.0
                };

                let margin = similarity - impostor_similarity;
                Some(FaceObservation {
                    bbox,
                    similarity,
                    impostor_similarity,
                    margin,
                    body_similarity: None,
                })
            })
            .collect();

        observations.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.margin
                        .partial_cmp(&a.margin)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        Ok(observations)
    }

    /// Best-match helper compatible with legacy call sites.
    ///
    /// A match is accepted only when similarity and positive-vs-negative margin
    /// both clear configured thresholds.
    ///
    /// # Errors
    ///
    /// Returns an error if face processing or inference fails.
    pub fn identify(
        &self,
        frame: &RgbFrame,
        persons: &[BBox],
        search_hint: Option<(f32, f32)>,
    ) -> Result<Option<IdentityMatch>> {
        let observations = self.observations(frame, persons, search_hint)?;
        let Some(best) = observations.first() else {
            return Ok(None);
        };
        if best.similarity < self.similarity_threshold || best.margin < self.margin_threshold {
            return Ok(None);
        }
        debug!(
            similarity = best.similarity,
            impostor_similarity = best.impostor_similarity,
            margin = best.margin,
            "face identity accepted"
        );
        Ok(Some(IdentityMatch {
            bbox: best.bbox,
            similarity: best.similarity,
        }))
    }

    /// Recovery observations helper with wider candidate budget.
    ///
    /// # Errors
    ///
    /// Returns an error if face processing or inference fails.
    pub fn recovery_observations(
        &self,
        frame: &RgbFrame,
        persons: &[BBox],
        search_hint: Option<(f32, f32)>,
    ) -> Result<Vec<FaceObservation>> {
        self.observations_with_candidate_budget(
            frame,
            persons,
            search_hint,
            MAX_FACE_CANDIDATES_RECOVERY,
        )
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
    /// Thread-local crop buffer to avoid per-frame allocations.
    /// Initialized with capacity for a typical face crop (200x200 RGB).
    static FACE_CROP_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn preprocess_face_data(img: &RgbImage) -> Result<Vec<f32>> {
    preprocess_face_data_from_raw(img.width(), img.height(), img.as_raw())
}

fn preprocess_face_data_from_raw(width: u32, height: u32, raw_rgb: &[u8]) -> Result<Vec<f32>> {
    let src =
        fr::images::ImageRef::new(width, height, raw_rgb, fr::PixelType::U8x3).map_err(|e| {
            crate::FancamError::image_processing(format!(
                "Failed to create face resize source: {e}"
            ))
        })?;

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
            .map_err(|e| {
                crate::FancamError::image_processing(format!(
                    "Failed to create face resize destination: {e}"
                ))
            })?;

            let options = fr::ResizeOptions::new()
                .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
            resizer_cell
                .borrow_mut()
                .resize(&src, &mut dst, Some(&options))
                .map_err(|e| {
                    crate::FancamError::image_processing(format!(
                        "fast_image_resize face resize failed: {e}"
                    ))
                })?;

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
        // Ensure capacity and resize to avoid repeated allocations
        let current_cap = crop_buf.capacity();
        if current_cap < crop_len {
            crop_buf.reserve(crop_len - current_cap);
        }
        crop_buf.resize(crop_len, 0);

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
    Tensor::from_array((shape, tensor_data.into_boxed_slice()))
        .map(|t| t.into_dyn())
        .map_err(|e| {
            crate::FancamError::inference(format!("Failed to create face input tensor: {e}"))
        })
}

pub(crate) fn build_ort_session(model_path: &Path) -> std::result::Result<Session, ort::Error> {
    if let Err(err) = crate::runtime::OrtConfig::ensure_initialized() {
        return Err(ort::Error::new(format!(
            "ONNX Runtime init failure (CoreML required): {err}"
        )));
    }
    let base_builder = Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?;

    let coreml = ep::CoreMLExecutionProvider::default()
        .with_subgraphs(true)
        .with_static_input_shapes(false)
        .with_compute_units(ep::coreml::CoreMLComputeUnits::CPUAndNeuralEngine)
        .with_specialization_strategy(ep::coreml::CoreMLSpecializationStrategy::FastPrediction);
    let coreml_available = coreml.is_available().unwrap_or(false);
    if !coreml_available {
        return Err(ort::Error::new(
            "CoreML execution provider is unavailable in the loaded ONNX Runtime.",
        ));
    }

    let builder =
        base_builder.with_execution_providers([ep::CoreMLExecutionProvider::default()
            .with_subgraphs(true)
            .with_static_input_shapes(false)
            .with_compute_units(ep::coreml::CoreMLComputeUnits::CPUAndNeuralEngine)
            .with_specialization_strategy(ep::coreml::CoreMLSpecializationStrategy::FastPrediction)
            .build()])?;
    builder.commit_from_file(model_path)
}

/// Extract the first output tensor's data as a flat `Vec<f32>`.
fn extract_first_embedding(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<f32>> {
    // ArcFace / MobileFaceNet: first output is the embedding vector
    let first_value = outputs
        .iter()
        .next()
        .ok_or_else(|| crate::FancamError::inference("ArcFace produced no outputs"))?
        .1;

    let (_shape, data) = first_value.try_extract_tensor::<f32>().map_err(|e| {
        crate::FancamError::inference(format!("Failed to extract ArcFace embedding tensor: {e}"))
    })?;

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

    let face_x1 = (face_x1_f.max(0.0) as u32).min(frame.width.saturating_sub(1));
    let face_y1 = (face_y1_f.max(0.0) as u32).min(frame.height.saturating_sub(1));
    let face_w = (face_w_f as u32)
        .min(frame.width.saturating_sub(face_x1))
        .max(1);
    let face_h = (face_h_f as u32)
        .min(frame.height.saturating_sub(face_y1))
        .max(1);

    (face_x1, face_y1, face_w, face_h)
}

/// Returns the heuristic face crop region for a person detection box.
#[must_use]
pub fn face_crop_region_for_bbox(frame: &RgbFrame, bbox: BBox) -> (u32, u32, u32, u32) {
    face_crop_region(frame, bbox)
}

/// Estimates whether the upper-body crop contains a valid frontal/near-frontal face.
///
/// This is a lightweight image heuristic used during discovery to suppress
/// obvious non-face clusters (for example backs of heads, shoulders, or torso-only
/// detections) before embedding + clustering.
#[must_use]
pub fn face_presence_score(frame: &RgbFrame, bbox: BBox) -> f32 {
    let (x1, y1, w, h) = face_crop_region(frame, bbox);
    if w < 12 || h < 12 {
        return 0.0;
    }

    let step_x = (w / 26).max(1) as usize;
    let step_y = (h / 26).max(1) as usize;
    let cols = ((w as usize + step_x - 1) / step_x).max(2);
    let rows = ((h as usize + step_y - 1) / step_y).max(2);

    let stride = frame.width as usize * 3;
    let mut luma = vec![0.0f32; rows * cols];

    for row in 0..rows {
        let sample_y = y1 as usize + (row * step_y).min(h as usize - 1);
        for col in 0..cols {
            let sample_x = x1 as usize + (col * step_x).min(w as usize - 1);
            let idx = sample_y * stride + sample_x * 3;
            let r = frame.data[idx] as f32;
            let g = frame.data[idx + 1] as f32;
            let b = frame.data[idx + 2] as f32;
            // Rec.709 luma approximation in [0, 255]
            luma[row * cols + col] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }

    let total = (rows * cols) as f32;
    let mean = luma.iter().sum::<f32>() / total.max(1.0);
    let variance = luma
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / total.max(1.0);
    let std_dev = variance.sqrt();
    let texture_score = ((std_dev - 12.0) / 38.0).clamp(0.0, 1.0);

    let mut edge_sum = 0.0f32;
    let mut edge_count = 0usize;
    let mut center_edge_sum = 0.0f32;
    for row in 0..(rows - 1) {
        for col in 0..(cols - 1) {
            let c = luma[row * cols + col];
            let right = luma[row * cols + (col + 1)];
            let down = luma[(row + 1) * cols + col];
            let edge = (c - right).abs() + (c - down).abs();
            edge_sum += edge;
            edge_count += 1;

            let center_row = row >= rows / 4 && row <= (rows * 3) / 4;
            let center_col = col >= cols / 4 && col <= (cols * 3) / 4;
            if center_row && center_col {
                center_edge_sum += edge;
            }
        }
    }
    let edge_mean = edge_sum / edge_count.max(1) as f32;
    let edge_score = ((edge_mean - 7.0) / 26.0).clamp(0.0, 1.0);
    let outer_edge = (edge_sum - center_edge_sum).max(0.0);
    let center_ratio = center_edge_sum / outer_edge.max(1e-3);
    let center_focus_score = ((center_ratio - 0.75) / 0.85).clamp(0.0, 1.0);

    let mut symmetry_diff = 0.0f32;
    let mut symmetry_count = 0usize;
    for row in 0..rows {
        for col in 0..(cols / 2) {
            let left = luma[row * cols + col];
            let right = luma[row * cols + (cols - 1 - col)];
            symmetry_diff += (left - right).abs();
            symmetry_count += 1;
        }
    }
    let symmetry_mean = symmetry_diff / symmetry_count.max(1) as f32;
    let symmetry_score = (1.0 - symmetry_mean / 58.0).clamp(0.0, 1.0);

    let mut upper_sum = 0.0f32;
    let mut upper_count = 0usize;
    let mut lower_sum = 0.0f32;
    let mut lower_count = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let value = luma[row * cols + col];
            if row < rows / 3 {
                upper_sum += value;
                upper_count += 1;
            } else if row >= (rows * 2) / 3 {
                lower_sum += value;
                lower_count += 1;
            }
        }
    }
    let upper_mean = upper_sum / upper_count.max(1) as f32;
    let lower_mean = lower_sum / lower_count.max(1) as f32;
    let upper_lower_diff = (upper_mean - lower_mean).abs();
    let upper_lower_score = ((upper_lower_diff - 10.0) / 42.0).clamp(0.0, 1.0);

    (texture_score * 0.24
        + edge_score * 0.22
        + symmetry_score * 0.20
        + center_focus_score * 0.22
        + upper_lower_score * 0.12)
        .clamp(0.0, 1.0)
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

fn apply_search_gate(candidates: Vec<BBox>, search_hint: Option<(f32, f32)>) -> Vec<BBox> {
    let Some((hx, hy)) = search_hint else {
        return candidates;
    };

    let gated = candidates
        .iter()
        .copied()
        .filter(|bbox| {
            let dx = bbox.center_x() - hx;
            let dy = bbox.center_y() - hy;
            let distance = (dx * dx + dy * dy).sqrt();
            let scale = bbox.width().max(bbox.height()).max(1.0);
            let gate =
                (scale * SEARCH_GATE_SCALE).clamp(SEARCH_GATE_MIN_RADIUS, SEARCH_GATE_MAX_RADIUS);
            distance <= gate
        })
        .collect::<Vec<_>>();

    if gated.is_empty() {
        // Avoid hard lockout when prediction drifts briefly.
        // Caller already ranked candidates by proximity/confidence.
        return candidates
            .into_iter()
            .take(SEARCH_GATE_FALLBACK_KEEP)
            .collect();
    }
    gated
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
    boxes.retain(|b| b.confidence.is_finite());
    boxes.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
///
/// # Errors
///
/// Returns an error if the frame dimensions are invalid.
pub fn draw_boxes(frame: &mut RgbFrame, boxes: &[BBox], color: [u8; 3]) -> Result<()> {
    // Build the image from the existing buffer — no clone; we write back in-place.
    let mut img: RgbImage =
        ImageBuffer::from_raw(frame.width, frame.height, std::mem::take(&mut frame.data))
            .ok_or_else(|| {
                crate::FancamError::invalid_frame(format!(
                    "Invalid frame dimensions: {}x{}",
                    frame.width, frame.height
                ))
            })?;

    for bbox in boxes {
        let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size(bbox.width() as u32, bbox.height() as u32);
        imageproc::drawing::draw_hollow_rect_mut(&mut img, rect, Rgb(color));
    }

    frame.data = img.into_raw();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32) -> RgbFrame {
        RgbFrame {
            data: vec![0u8; (width * height * 3) as usize],
            width,
            height,
            pts: 0,
        }
    }

    #[test]
    fn nms_suppresses_overlap() {
        let boxes = vec![
            BBox {
                x1: 10.0,
                y1: 10.0,
                x2: 110.0,
                y2: 210.0,
                confidence: 0.95,
            },
            BBox {
                x1: 14.0,
                y1: 15.0,
                x2: 108.0,
                y2: 212.0,
                confidence: 0.91,
            },
            BBox {
                x1: 300.0,
                y1: 20.0,
                x2: 360.0,
                y2: 180.0,
                confidence: 0.70,
            },
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|b| (b.x1 - 10.0).abs() < f32::EPSILON));
        assert!(kept.iter().any(|b| (b.x1 - 300.0).abs() < f32::EPSILON));
    }

    #[test]
    fn face_crop_region_stays_in_bounds() {
        let frame = frame(1920, 1080);
        let bbox = BBox {
            x1: 1810.0,
            y1: 2.0,
            x2: 1918.0,
            y2: 620.0,
            confidence: 0.88,
        };
        let (x, y, w, h) = face_crop_region(&frame, bbox);
        assert!(x < frame.width);
        assert!(y < frame.height);
        assert!(w > 0 && h > 0);
        assert!(x + w <= frame.width);
        assert!(y + h <= frame.height);
    }

    #[test]
    fn rank_candidate_bias_prefers_search_hint() {
        let near = BBox {
            x1: 95.0,
            y1: 95.0,
            x2: 145.0,
            y2: 230.0,
            confidence: 0.50,
        };
        let far = BBox {
            x1: 500.0,
            y1: 500.0,
            x2: 560.0,
            y2: 700.0,
            confidence: 0.60,
        };
        let near_score = rank_candidate(near, Some((120.0, 150.0)));
        let far_score = rank_candidate(far, Some((120.0, 150.0)));
        assert!(near_score > far_score);
    }

    #[test]
    fn face_presence_prefers_face_like_pattern_over_flat_patch() {
        let mut frame = frame(160, 180);

        // Flat patch (very low texture)
        for px in frame.data.chunks_mut(3) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
        }
        let bbox = BBox {
            x1: 40.0,
            y1: 30.0,
            x2: 120.0,
            y2: 170.0,
            confidence: 0.9,
        };
        let flat_score = face_presence_score(&frame, bbox);

        // Paint a rough face-like symmetric pattern in the same region.
        for y in 48..118 {
            for x in 52..108 {
                let idx = (y as usize * frame.width as usize + x as usize) * 3;
                frame.data[idx] = 212;
                frame.data[idx + 1] = 182;
                frame.data[idx + 2] = 160;
            }
        }
        // Eyes
        for y in 68..78 {
            for x in 64..74 {
                let idx = (y as usize * frame.width as usize + x as usize) * 3;
                frame.data[idx] = 32;
                frame.data[idx + 1] = 32;
                frame.data[idx + 2] = 32;
            }
            for x in 86..96 {
                let idx = (y as usize * frame.width as usize + x as usize) * 3;
                frame.data[idx] = 32;
                frame.data[idx + 1] = 32;
                frame.data[idx + 2] = 32;
            }
        }
        // Nose/mouth contrast
        for y in 84..106 {
            let x = 80usize;
            let idx = (y as usize * frame.width as usize + x) * 3;
            frame.data[idx] = 80;
            frame.data[idx + 1] = 60;
            frame.data[idx + 2] = 58;
        }
        for y in 106..112 {
            for x in 68..92 {
                let idx = (y as usize * frame.width as usize + x as usize) * 3;
                frame.data[idx] = 74;
                frame.data[idx + 1] = 40;
                frame.data[idx + 2] = 40;
            }
        }

        let face_like_score = face_presence_score(&frame, bbox);
        assert!(face_like_score > flat_score);
    }
}
