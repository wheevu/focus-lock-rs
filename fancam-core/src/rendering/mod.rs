//! rendering — crop a 9:16 vertical slice from a full frame
//!
//! Given a `CameraState` (centre + zoom), extract a `1080×1920` crop from the
//! original frame.  If the subject is small (far from camera), optionally
//! upscale with Lanczos resampling.

use anyhow::{Context, Result};
use fast_image_resize as fr;

use crate::{tracking::CameraState, video::RgbFrame};

/// Output fancam resolution.
pub const OUT_WIDTH: u32 = 1080;
pub const OUT_HEIGHT: u32 = 1920;

/// If the subject occupies less than this fraction of frame height, upscale.
const UPSCALE_THRESHOLD: f32 = 0.25;

/// Reusable rendering context to avoid per-frame allocations.
pub struct FrameRenderer {
    resizer: fr::Resizer,
    crop_buf: Vec<u8>,
    out_buf: Vec<u8>,
    scaled_buf: Vec<u8>,
}

impl FrameRenderer {
    pub fn new() -> Self {
        Self {
            resizer: fr::Resizer::new(),
            crop_buf: Vec::new(),
            out_buf: vec![0u8; (OUT_WIDTH * OUT_HEIGHT * 3) as usize],
            scaled_buf: Vec::new(),
        }
    }

    /// Render a 9:16 fancam crop directly into `frame`.
    pub fn crop_fancam_inplace(&mut self, frame: &mut RgbFrame, state: &CameraState) -> Result<()> {
        // Determine crop dimensions in source-frame pixels.
        // We want OUT_WIDTH × OUT_HEIGHT aspect, scaled so it fits the frame.
        let aspect = OUT_WIDTH as f32 / OUT_HEIGHT as f32; // 9/16 ≈ 0.5625

        // Use the wider of: a fixed 1080-px crop, or 2× the detected half-size.
        // This keeps the subject comfortably framed.
        let crop_w = (state.half_size * 2.5)
            .max(OUT_WIDTH as f32)
            .min(frame.width as f32);
        let crop_h = (crop_w / aspect).min(frame.height as f32);
        let crop_w = crop_h * aspect; // re-derive width after height clamp

        let x1 = (state.cx - crop_w / 2.0)
            .max(0.0)
            .min((frame.width as f32 - crop_w).max(0.0)) as u32;
        let y1 = (state.cy - crop_h / 2.0)
            .max(0.0)
            .min((frame.height as f32 - crop_h).max(0.0)) as u32;

        let crop_w_u = (crop_w as u32).min(frame.width - x1);
        let crop_h_u = (crop_h as u32).min(frame.height - y1);

        // Copy only the crop rows from the raw slice — O(crop area), no full-frame clone.
        let src_stride = (frame.width * 3) as usize;
        let dst_stride = (crop_w_u * 3) as usize;
        let crop_len = dst_stride * crop_h_u as usize;
        if self.crop_buf.len() != crop_len {
            self.crop_buf.resize(crop_len, 0);
        }
        for row in 0..crop_h_u as usize {
            let src_start = (y1 as usize + row) * src_stride + x1 as usize * 3;
            let dst_start = row * dst_stride;
            self.crop_buf[dst_start..dst_start + dst_stride]
                .copy_from_slice(&frame.data[src_start..src_start + dst_stride]);
        }

        // Decide filter quality based on whether we're upscaling significantly
        let subject_height_fraction = state.half_size * 2.0 / frame.height as f32;
        let filter = if subject_height_fraction < UPSCALE_THRESHOLD {
            fr::FilterType::Lanczos3 // subject is small — use high-quality upscale
        } else {
            fr::FilterType::CatmullRom // subject is large — fast enough
        };

        // SIMD-accelerated resize via fast_image_resize (NEON on M1).
        let src =
            fr::images::ImageRef::new(crop_w_u, crop_h_u, &self.crop_buf, fr::PixelType::U8x3)
                .context("failed to create fast_image_resize source for crop")?;

        let mut out_image = fr::images::Image::from_vec_u8(
            OUT_WIDTH,
            OUT_HEIGHT,
            std::mem::take(&mut self.out_buf),
            fr::PixelType::U8x3,
        )
        .context("failed to create fast_image_resize destination for crop")?;

        let options = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(filter));
        self.resizer
            .resize(&src, &mut out_image, Some(&options))
            .context("fast_image_resize crop scale failed")?;

        self.out_buf = out_image.into_vec();
        std::mem::swap(&mut frame.data, &mut self.out_buf);
        frame.width = OUT_WIDTH;
        frame.height = OUT_HEIGHT;
        Ok(())
    }

    /// Render letterbox passthrough directly into `frame`.
    pub fn letterbox_passthrough_inplace(&mut self, frame: &mut RgbFrame) -> Result<()> {
        // Scale uniformly to fit OUT_WIDTH × OUT_HEIGHT, add black bars.
        let src_aspect = frame.width as f32 / frame.height as f32;
        let dst_aspect = OUT_WIDTH as f32 / OUT_HEIGHT as f32;

        let (scaled_w, scaled_h) = if src_aspect > dst_aspect {
            let w = OUT_WIDTH;
            let h = (OUT_WIDTH as f32 / src_aspect) as u32;
            (w, h)
        } else {
            let h = OUT_HEIGHT;
            let w = (OUT_HEIGHT as f32 * src_aspect) as u32;
            (w, h)
        };

        let scaled_w = scaled_w.max(1);
        let scaled_h = scaled_h.max(1);

        let src =
            fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
                .context("failed to create fast_image_resize source for letterbox")?;

        let scaled_len = (scaled_w * scaled_h * 3) as usize;
        if self.scaled_buf.len() != scaled_len {
            self.scaled_buf.resize(scaled_len, 0);
        }
        let mut dst = fr::images::Image::from_vec_u8(
            scaled_w,
            scaled_h,
            std::mem::take(&mut self.scaled_buf),
            fr::PixelType::U8x3,
        )
        .context("failed to create fast_image_resize destination for letterbox")?;

        let options = fr::ResizeOptions::new()
            .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom));
        self.resizer
            .resize(&src, &mut dst, Some(&options))
            .context("fast_image_resize letterbox scale failed")?;

        self.scaled_buf = dst.into_vec();
        let scaled_data = &self.scaled_buf;

        let offset_x = ((OUT_WIDTH - scaled_w) / 2) as usize;
        let offset_y = ((OUT_HEIGHT - scaled_h) / 2) as usize;
        let canvas_stride = (OUT_WIDTH * 3) as usize;
        let scaled_stride = (scaled_w * 3) as usize;

        let out_len = canvas_stride * OUT_HEIGHT as usize;
        self.out_buf.resize(out_len, 0);
        self.out_buf.fill(0);
        for row in 0..scaled_h as usize {
            let dst_start = (offset_y + row) * canvas_stride + offset_x * 3;
            let src_start = row * scaled_stride;
            self.out_buf[dst_start..dst_start + scaled_stride]
                .copy_from_slice(&scaled_data[src_start..src_start + scaled_stride]);
        }

        std::mem::swap(&mut frame.data, &mut self.out_buf);
        frame.width = OUT_WIDTH;
        frame.height = OUT_HEIGHT;
        Ok(())
    }
}

impl Default for FrameRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract and return the 9:16 fancam crop as an `RgbFrame`.
///
/// The crop window is centred on `state.cx`, `state.cy` and has a width of
/// `OUT_WIDTH` pixels in the source frame (clamped to frame boundaries).
/// Height is set to maintain the 9:16 aspect ratio.
pub fn crop_fancam(frame: &RgbFrame, state: &CameraState) -> Result<RgbFrame> {
    let mut out = RgbFrame {
        data: frame.data.clone(),
        width: frame.width,
        height: frame.height,
        pts: frame.pts,
    };
    let mut renderer = FrameRenderer::new();
    renderer.crop_fancam_inplace(&mut out, state)?;
    Ok(out)
}

/// Write a plain full-frame passthrough (no crop) — used when the target is
/// lost and we want a letterboxed placeholder rather than a blank frame.
pub fn letterbox_passthrough(frame: &RgbFrame) -> Result<RgbFrame> {
    let mut out = RgbFrame {
        data: frame.data.clone(),
        width: frame.width,
        height: frame.height,
        pts: frame.pts,
    };
    let mut renderer = FrameRenderer::new();
    renderer.letterbox_passthrough_inplace(&mut out)?;
    Ok(out)
}
