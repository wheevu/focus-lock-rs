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

/// Extract and return the 9:16 fancam crop as an `RgbFrame`.
///
/// The crop window is centred on `state.cx`, `state.cy` and has a width of
/// `OUT_WIDTH` pixels in the source frame (clamped to frame boundaries).
/// Height is set to maintain the 9:16 aspect ratio.
pub fn crop_fancam(frame: &RgbFrame, state: &CameraState) -> Result<RgbFrame> {
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
    let mut buf = vec![0u8; dst_stride * crop_h_u as usize];
    for row in 0..crop_h_u as usize {
        let src_start = (y1 as usize + row) * src_stride + x1 as usize * 3;
        let dst_start = row * dst_stride;
        buf[dst_start..dst_start + dst_stride]
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
    let src = fr::images::Image::from_vec_u8(crop_w_u, crop_h_u, buf, fr::PixelType::U8x3)
        .context("failed to create fast_image_resize source for crop")?;

    let mut dst = fr::images::Image::new(OUT_WIDTH, OUT_HEIGHT, fr::PixelType::U8x3);

    let mut resizer = fr::Resizer::new();
    let options = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(filter));
    resizer
        .resize(&src, &mut dst, Some(&options))
        .context("fast_image_resize crop scale failed")?;

    Ok(RgbFrame {
        data: dst.into_vec(),
        width: OUT_WIDTH,
        height: OUT_HEIGHT,
        pts: frame.pts,
    })
}

/// Write a plain full-frame passthrough (no crop) — used when the target is
/// lost and we want a letterboxed placeholder rather than a blank frame.
pub fn letterbox_passthrough(frame: &RgbFrame) -> Result<RgbFrame> {
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

    // Ensure dimensions are at least 1 for NonZeroU32.
    let scaled_w = scaled_w.max(1);
    let scaled_h = scaled_h.max(1);

    // SIMD-accelerated resize via fast_image_resize (NEON on M1).
    // ImageRef borrows the frame data immutably — no clone needed.
    let src =
        fr::images::ImageRef::new(frame.width, frame.height, &frame.data, fr::PixelType::U8x3)
            .context("failed to create fast_image_resize source for letterbox")?;

    let mut dst = fr::images::Image::new(scaled_w, scaled_h, fr::PixelType::U8x3);

    let mut resizer = fr::Resizer::new();
    let options =
        fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom));
    resizer
        .resize(&src, &mut dst, Some(&options))
        .context("fast_image_resize letterbox scale failed")?;

    let scaled_data = dst.buffer();

    // Composite the scaled image centred on a black canvas.
    let offset_x = ((OUT_WIDTH - scaled_w) / 2) as usize;
    let offset_y = ((OUT_HEIGHT - scaled_h) / 2) as usize;
    let canvas_stride = (OUT_WIDTH * 3) as usize;
    let scaled_stride = (scaled_w * 3) as usize;

    let mut canvas = vec![0u8; canvas_stride * OUT_HEIGHT as usize];
    for row in 0..scaled_h as usize {
        let dst_start = (offset_y + row) * canvas_stride + offset_x * 3;
        let src_start = row * scaled_stride;
        canvas[dst_start..dst_start + scaled_stride]
            .copy_from_slice(&scaled_data[src_start..src_start + scaled_stride]);
    }

    Ok(RgbFrame {
        data: canvas,
        width: OUT_WIDTH,
        height: OUT_HEIGHT,
        pts: frame.pts,
    })
}
