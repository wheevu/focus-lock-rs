//! reid — body re-identification embeddings and scoring.
//!
//! This module provides optional body-level identity cues used to support
//! face-based tracking during difficult frames.

use std::path::Path;
use std::sync::Mutex;

use fast_image_resize as fr;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use crate::detection::{BBox, FaceObservation};
use crate::video::RgbFrame;
use crate::{FancamError, Result};

/// OSNet input width.
const REID_WIDTH: u32 = 128;
/// OSNet input height.
const REID_HEIGHT: u32 = 256;
/// Session pool size for ReID scoring.
const REID_SESSION_POOL: usize = 8;
/// Max detections scored with ReID per frame.
const REID_MAX_CANDIDATES: usize = 16;

/// Body ReID scorer backed by ONNX Runtime sessions.
#[derive(Debug)]
pub struct BodyReidentifier {
    sessions: Vec<Mutex<Session>>,
}

impl BodyReidentifier {
    /// Load a body ReID model.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let mut sessions = Vec::with_capacity(REID_SESSION_POOL);
        for _ in 0..REID_SESSION_POOL {
            let session = super::detection::build_ort_session(&model_path)
                .map_err(|e| FancamError::model_load(&model_path, e))?;
            sessions.push(Mutex::new(session));
        }
        Ok(Self { sessions })
    }

    /// Extract a body embedding for one detection box.
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing or inference fails.
    pub fn embed_from_bbox(&self, frame: &RgbFrame, bbox: BBox) -> Result<Option<Vec<f32>>> {
        self.embed_many_from_bboxes(frame, &[bbox])
            .map(|rows| rows.into_iter().next().map(|(_, emb)| emb))
    }

    /// Extract body embeddings for many detections in one frame.
    ///
    /// # Errors
    ///
    /// Returns an error if no body embedding can be extracted from any input box.
    pub fn embed_many_from_bboxes(
        &self,
        frame: &RgbFrame,
        bboxes: &[BBox],
    ) -> Result<Vec<(BBox, Vec<f32>)>> {
        if bboxes.is_empty() {
            return Ok(Vec::new());
        }

        let rows = bboxes
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &bbox)| {
                let tensor_data = preprocess_body_from_bbox(frame, bbox).ok()?;
                let tensor = body_tensor_from_data(tensor_data).ok()?;
                let session_idx = idx % self.sessions.len();
                let embedding = {
                    let mut session = self.sessions.get(session_idx)?.lock().ok()?;
                    let outputs = run_reid_inference(&mut session, tensor).ok()?;
                    l2_normalize(&extract_first_embedding(&outputs).ok()?)
                };
                Some((bbox, embedding))
            })
            .collect::<Vec<_>>();

        if rows.is_empty() {
            return Err(FancamError::face_id(
                "body reid could not extract any embedding from provided detections",
            ));
        }

        Ok(rows)
    }

    /// Annotate face observations with body similarities against the gallery.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn annotate_observations_with_gallery(
        &self,
        frame: &RgbFrame,
        observations: &mut [FaceObservation],
        target_gallery: &[Vec<f32>],
    ) -> Result<()> {
        if observations.is_empty() || target_gallery.is_empty() {
            return Ok(());
        }

        let normalized_gallery = target_gallery
            .iter()
            .filter(|row| !row.is_empty())
            .map(|row| l2_normalize(row))
            .collect::<Vec<_>>();
        if normalized_gallery.is_empty() {
            return Ok(());
        }

        let mut ranked = observations
            .iter()
            .enumerate()
            .map(|(idx, obs)| (idx, obs.bbox.confidence))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(REID_MAX_CANDIDATES.min(observations.len()));

        let jobs = ranked
            .into_iter()
            .filter_map(|(idx, _)| observations.get(idx).map(|obs| (idx, obs.bbox)))
            .collect::<Vec<_>>();
        if jobs.is_empty() {
            return Ok(());
        }

        let rows = jobs
            .par_iter()
            .enumerate()
            .filter_map(|(n, (obs_index, bbox))| {
                let tensor_data = preprocess_body_from_bbox(frame, *bbox).ok()?;
                let tensor = body_tensor_from_data(tensor_data).ok()?;
                let session_idx = n % self.sessions.len();
                let embedding = {
                    let mut session = self.sessions.get(session_idx)?.lock().ok()?;
                    let outputs = run_reid_inference(&mut session, tensor).ok()?;
                    l2_normalize(&extract_first_embedding(&outputs).ok()?)
                };
                let similarity = cosine_with_gallery(&embedding, &normalized_gallery)?;
                if !similarity.is_finite() {
                    return None;
                }
                Some((*obs_index, similarity.clamp(-1.0, 1.0)))
            })
            .collect::<Vec<_>>();

        for (idx, sim) in rows {
            if let Some(obs) = observations.get_mut(idx) {
                obs.body_similarity = Some(sim);
            }
        }

        Ok(())
    }
}

fn run_reid_inference(
    session: &mut Session,
    tensor: ort::value::DynValue,
) -> Result<ort::session::SessionOutputs<'_>> {
    session
        .run(ort::inputs![tensor])
        .map_err(|e| FancamError::inference(format!("body reid inference failed: {e}")))
}

fn body_tensor_from_data(tensor_data: Vec<f32>) -> Result<ort::value::DynValue> {
    let shape = [1usize, 3, REID_HEIGHT as usize, REID_WIDTH as usize];
    Tensor::from_array((shape, tensor_data.into_boxed_slice()))
        .map(|t| t.into_dyn())
        .map_err(|e| {
            FancamError::inference(format!("failed to create body reid input tensor: {e}"))
        })
}

fn preprocess_body_from_bbox(frame: &RgbFrame, bbox: BBox) -> Result<Vec<f32>> {
    let (x1, y1, w, h) = body_crop_region(frame, bbox);
    let src_stride = (frame.width * 3) as usize;
    let dst_stride = (w * 3) as usize;
    let crop_len = dst_stride * h as usize;

    let mut crop_buf = vec![0u8; crop_len];
    for row in 0..h as usize {
        let src_start = (y1 as usize + row) * src_stride + x1 as usize * 3;
        let dst_start = row * dst_stride;
        crop_buf[dst_start..dst_start + dst_stride]
            .copy_from_slice(&frame.data[src_start..src_start + dst_stride]);
    }

    let src = fr::images::ImageRef::new(w, h, &crop_buf, fr::PixelType::U8x3).map_err(|e| {
        FancamError::image_processing(format!("failed to create body reid source: {e}"))
    })?;
    let mut dst = fr::images::Image::new(REID_WIDTH, REID_HEIGHT, fr::PixelType::U8x3);
    let mut resizer = fr::Resizer::new();
    let options =
        fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer
        .resize(&src, &mut dst, Some(&options))
        .map_err(|e| FancamError::image_processing(format!("failed to resize body crop: {e}")))?;

    let raw = dst.into_vec();
    let size = (REID_WIDTH * REID_HEIGHT) as usize;
    let mut tensor = vec![0f32; 3 * size];
    for idx in 0..size {
        tensor[idx] = raw[idx * 3] as f32 / 255.0;
        tensor[size + idx] = raw[idx * 3 + 1] as f32 / 255.0;
        tensor[2 * size + idx] = raw[idx * 3 + 2] as f32 / 255.0;
    }
    Ok(tensor)
}

fn body_crop_region(frame: &RgbFrame, bbox: BBox) -> (u32, u32, u32, u32) {
    let bh = bbox.height().max(1.0);
    let x1 = bbox.x1.max(0.0).floor() as u32;
    let y1 = (bbox.y1 + bh * 0.10).max(0.0).floor() as u32;
    let x2 = bbox.x2.min(frame.width as f32).ceil() as u32;
    let y2 = bbox.y2.min(frame.height as f32).ceil() as u32;
    let w = x2
        .saturating_sub(x1)
        .max(1)
        .min(frame.width.saturating_sub(x1));
    let h = y2
        .saturating_sub(y1)
        .max(1)
        .min(frame.height.saturating_sub(y1));
    (x1, y1, w, h)
}

fn extract_first_embedding(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<f32>> {
    let first_value = outputs
        .iter()
        .next()
        .ok_or_else(|| FancamError::inference("body reid produced no outputs"))?
        .1;
    let (_shape, data) = first_value.try_extract_tensor::<f32>().map_err(|e| {
        FancamError::inference(format!("failed to extract body reid output tensor: {e}"))
    })?;
    Ok(data.to_vec())
}

fn cosine_with_gallery(embedding: &[f32], gallery: &[Vec<f32>]) -> Option<f32> {
    let dim = embedding.len();
    if dim == 0 {
        return None;
    }
    gallery
        .iter()
        .filter(|row| row.len() == dim)
        .map(|target| cosine_similarity(embedding, target))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter().map(|x| x / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
