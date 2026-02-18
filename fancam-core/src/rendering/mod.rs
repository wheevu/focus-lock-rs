//! rendering — crop a 9:16 vertical slice from a full frame
//!
//! Given a `CameraState` (centre + zoom), extract a `1080×1920` crop from the
//! original frame.  If the subject is small (far from camera), optionally
//! upscale with Lanczos resampling.

use anyhow::{Context, Result};
use image::{imageops::FilterType, ImageBuffer, RgbImage};

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
    // crop_imm requires an owned RgbImage (image crate needs 'static on
    // the container). Build one from the frame data.
    let img: RgbImage = ImageBuffer::from_raw(frame.width, frame.height, frame.data.clone())
        .context("failed to build image from frame data")?;

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

    let cropped = image::imageops::crop_imm(&img, x1, y1, crop_w_u, crop_h_u).to_image();

    // Decide filter quality based on whether we're upscaling significantly
    let subject_height_fraction = state.half_size * 2.0 / frame.height as f32;
    let filter = if subject_height_fraction < UPSCALE_THRESHOLD {
        FilterType::Lanczos3 // subject is small — use high-quality upscale
    } else {
        FilterType::CatmullRom // subject is large — fast enough
    };

    let output = image::imageops::resize(&cropped, OUT_WIDTH, OUT_HEIGHT, filter);

    Ok(RgbFrame {
        data: output.into_raw(),
        width: OUT_WIDTH,
        height: OUT_HEIGHT,
        pts: frame.pts,
    })
}

/// Write a plain full-frame passthrough (no crop) — used when the target is
/// lost and we want a letterboxed placeholder rather than a blank frame.
pub fn letterbox_passthrough(frame: &RgbFrame) -> Result<RgbFrame> {
    let img = ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
        frame.width,
        frame.height,
        frame.data.as_slice(),
    )
    .context("failed to build image from frame data")?;

    // Scale uniformly to fit OUT_WIDTH × OUT_HEIGHT, add black bars
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

    let scaled = image::imageops::resize(&img, scaled_w, scaled_h, FilterType::CatmullRom);

    let mut canvas: RgbImage = ImageBuffer::new(OUT_WIDTH, OUT_HEIGHT);
    let offset_x = (OUT_WIDTH - scaled_w) / 2;
    let offset_y = (OUT_HEIGHT - scaled_h) / 2;

    image::imageops::overlay(&mut canvas, &scaled, offset_x as i64, offset_y as i64);

    Ok(RgbFrame {
        data: canvas.into_raw(),
        width: OUT_WIDTH,
        height: OUT_HEIGHT,
        pts: frame.pts,
    })
}
