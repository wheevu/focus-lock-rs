//! video — FFmpeg bridge
//!
//! Phase 1 deliverable: open a video, iterate decoded frames, re-encode with
//! an arbitrary per-frame transform applied, and mux the result back to disk.
//!
//! The design intentionally keeps the frame callback generic (`FnMut`) so later
//! phases can slot in detection + crop logic without changing this module.

use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::{
    codec, encoder, format, frame, media, software::scaling, util::rational::Rational,
};
use std::path::Path;
use tracing::{debug, info};

/// Output pixel format for the encoder (YUV420p is universally compatible).
const ENCODE_FORMAT: format::Pixel = format::Pixel::YUV420P;
/// Scaling flags — bilinear is fast and good enough for the decode→encode path.
const SCALE_FLAGS: scaling::Flags = scaling::Flags::BILINEAR;

/// A single decoded video frame in RGB24 format, along with its presentation
/// timestamp (in the source stream's time-base units).
pub struct RgbFrame {
    pub data: Vec<u8>, // packed RGB24, row-major
    pub width: u32,
    pub height: u32,
    pub pts: i64,
}

/// Open `input_path`, apply `frame_fn` to every frame (receives a mutable
/// `RgbFrame` — modify in-place to transform the output), and write the result
/// to `output_path` encoded as H.264.
///
/// Audio is stream-copied without re-encoding.
pub fn transcode<P, Q, F>(input_path: P, output_path: Q, mut frame_fn: F) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
    F: FnMut(&mut RgbFrame),
{
    transcode_inner(input_path, output_path, 0, &mut frame_fn, &mut |_, _| {})
}

fn transcode_inner<P, Q>(
    input_path: P,
    output_path: Q,
    total: u64,
    frame_fn: &mut dyn FnMut(&mut RgbFrame),
    progress_fn: &mut dyn FnMut(u64, u64),
) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    ffmpeg::init().context("failed to initialise FFmpeg")?;

    // ── Input ────────────────────────────────────────────────────────────────
    let mut ictx = format::input(&input_path).context("could not open input file")?;

    let video_stream_index = ictx
        .streams()
        .best(media::Type::Video)
        .context("no video stream found in input")?
        .index();

    let audio_stream_index = ictx.streams().best(media::Type::Audio).map(|s| s.index());

    // Decoder
    let input_video_stream = ictx.stream(video_stream_index).unwrap();
    let video_time_base = input_video_stream.time_base();
    let frame_rate = input_video_stream.avg_frame_rate();

    let decoder_ctx = codec::context::Context::from_parameters(input_video_stream.parameters())
        .context("failed to build decoder context")?;
    let mut decoder = decoder_ctx
        .decoder()
        .video()
        .context("failed to open video decoder")?;

    let src_width = decoder.width();
    let src_height = decoder.height();
    let src_pixel_fmt = decoder.format();

    info!(
        src_width,
        src_height,
        ?src_pixel_fmt,
        "opened input video stream"
    );

    // Scaler: decoded frame → RGB24 for the callback (fixed source size)
    let mut to_rgb = scaling::Context::get(
        src_pixel_fmt,
        src_width,
        src_height,
        format::Pixel::RGB24,
        src_width,
        src_height,
        SCALE_FLAGS,
    )
    .context("failed to create to-RGB scaler")?;

    // ── Output — lazily initialised on first frame ────────────────────────────
    // We defer encoder/muxer setup until after the first frame_fn call so we
    // know the actual output dimensions (the transform may change frame size,
    // e.g. cropping 4K → 1080×1920 for a fancam).
    let mut octx = format::output(&output_path).context("could not create output context")?;

    let global_header = octx
        .format()
        .flags()
        .contains(format::flag::Flags::GLOBAL_HEADER);

    let encoder_codec = encoder::find(codec::Id::H264)
        .context("H.264 encoder not found — is FFmpeg built with libx264?")?;

    // Deferred encoder state
    struct EncoderState {
        video_encoder: encoder::Video,
        to_yuv: scaling::Context,
        out_rgb_frame: frame::Video,
        yuv_frame: frame::Video,
        video_out_index: usize,
        out_width: u32,
        out_height: u32,
    }

    let mut enc_state: Option<EncoderState> = None;

    // Audio output stream (stream copy) — added before write_header
    let audio_out_index: Option<(usize, usize)>;

    // We need to set up audio and write the header before the first video
    // packet is encoded, but we don't know the output video dimensions yet.
    // Strategy: buffer the first RGB frame through frame_fn to learn the
    // output size, then set up the full muxer.
    //
    // To achieve this without seeking, we process all packets in a single
    // loop and lazily open the muxer on the first video frame.
    let mut header_written = false;
    audio_out_index = if let Some(ai) = audio_stream_index {
        let in_stream = ictx.stream(ai).unwrap();
        let mut audio_out = octx.add_stream(ffmpeg_next::codec::Id::None)?;
        audio_out.set_parameters(in_stream.parameters());
        Some((ai, audio_out.index()))
    } else {
        None
    };

    // ── Decode / transform / encode loop ────────────────────────────────────
    let mut decoded_frame = frame::Video::empty();
    let mut rgb_frame = frame::Video::empty();
    // Buffered audio packets collected before header is written
    let mut audio_buffer: Vec<(
        ffmpeg_next::Packet,
        usize,
        usize,
        ffmpeg_next::util::rational::Rational,
        ffmpeg_next::util::rational::Rational,
    )> = Vec::new();
    let mut frame_count = 0u64;

    for (stream, packet) in ictx.packets() {
        let stream_index = stream.index();

        // ── Audio passthrough ────────────────────────────────────────────
        if let Some((ai, ao)) = audio_out_index {
            if stream_index == ai {
                if !header_written {
                    // Buffer until we write the header after the first video frame
                    audio_buffer.push((
                        packet.clone(),
                        ao,
                        ao,
                        stream.time_base(),
                        octx.stream(ao).unwrap().time_base(),
                    ));
                    continue;
                }
                let mut pkt = packet.clone();
                pkt.set_stream(ao);
                pkt.rescale_ts(stream.time_base(), octx.stream(ao).unwrap().time_base());
                pkt.write_interleaved(&mut octx)
                    .context("failed to write audio packet")?;
                continue;
            }
        }

        // ── Video decode ─────────────────────────────────────────────────
        if stream_index != video_stream_index {
            continue;
        }

        decoder
            .send_packet(&packet)
            .context("decoder send_packet")?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            // Convert to RGB24
            to_rgb
                .run(&decoded_frame, &mut rgb_frame)
                .context("to-RGB scaling failed")?;

            // Compact to a plain Vec<u8> (remove stride padding if any)
            let stride = rgb_frame.stride(0);
            let raw = rgb_frame.data(0);
            let mut rgb_data = Vec::with_capacity((src_width * src_height * 3) as usize);
            for row in 0..src_height as usize {
                let start = row * stride;
                rgb_data.extend_from_slice(&raw[start..start + src_width as usize * 3]);
            }

            let pts = decoded_frame.pts().unwrap_or(frame_count as i64);

            let mut rgb = RgbFrame {
                data: rgb_data,
                width: src_width,
                height: src_height,
                pts,
            };

            // ── User transform ───────────────────────────────────────────
            frame_fn(&mut rgb);

            // After the callback, rgb.width/height reflect the output size
            // (may differ from src_width/src_height if the transform crops).
            let out_w = rgb.width;
            let out_h = rgb.height;

            // ── Lazy encoder initialisation on first frame ───────────────
            if enc_state.is_none() {
                let mut video_out_stream = octx.add_stream(encoder_codec)?;
                let encoder_ctx = codec::context::Context::new_with_codec(encoder_codec);
                let mut video_encoder_builder = encoder_ctx.encoder().video()?;

                video_encoder_builder.set_width(out_w);
                video_encoder_builder.set_height(out_h);
                video_encoder_builder.set_format(ENCODE_FORMAT);
                video_encoder_builder.set_time_base(video_time_base);
                video_encoder_builder.set_frame_rate(Some(frame_rate));
                if global_header {
                    video_encoder_builder.set_flags(codec::flag::Flags::GLOBAL_HEADER);
                }

                let video_encoder = video_encoder_builder
                    .open_as_with(
                        encoder_codec,
                        ffmpeg_next::Dictionary::from_iter([("crf", "18"), ("preset", "fast")]),
                    )
                    .context("failed to open H.264 encoder")?;

                video_out_stream.set_parameters(&video_encoder);
                let video_out_index = video_out_stream.index();

                let to_yuv = scaling::Context::get(
                    format::Pixel::RGB24,
                    out_w,
                    out_h,
                    ENCODE_FORMAT,
                    out_w,
                    out_h,
                    SCALE_FLAGS,
                )
                .context("failed to create to-YUV scaler")?;

                info!(out_w, out_h, "output dimensions determined; writing header");
                format::context::output::dump(&octx, 0, output_path.as_ref().to_str());
                octx.write_header()
                    .context("failed to write output header")?;
                header_written = true;

                // Flush buffered audio packets
                for (mut pkt, _, ao, src_tb, dst_tb) in audio_buffer.drain(..) {
                    pkt.set_stream(ao);
                    pkt.rescale_ts(src_tb, dst_tb);
                    pkt.write_interleaved(&mut octx)
                        .context("failed to write buffered audio packet")?;
                }

                enc_state = Some(EncoderState {
                    video_encoder,
                    to_yuv,
                    out_rgb_frame: frame::Video::new(format::Pixel::RGB24, out_w, out_h),
                    yuv_frame: frame::Video::empty(),
                    video_out_index,
                    out_width: out_w,
                    out_height: out_h,
                });
            }

            let state = enc_state.as_mut().unwrap();

            // Write the transformed RGB data into the output AVFrame.
            // The output frame may be a different size than the source, so we
            // use out_w/out_h here (not src_width/src_height).
            let out_stride = state.out_rgb_frame.stride(0);
            let plane_data = state.out_rgb_frame.data_mut(0);
            for row in 0..state.out_height as usize {
                let dst_start = row * out_stride;
                let src_start = row * state.out_width as usize * 3;
                plane_data[dst_start..dst_start + state.out_width as usize * 3].copy_from_slice(
                    &rgb.data[src_start..src_start + state.out_width as usize * 3],
                );
            }

            // Convert RGB24 → YUV420P for encoder
            state
                .to_yuv
                .run(&state.out_rgb_frame, &mut state.yuv_frame)
                .context("to-YUV scaling failed")?;

            state.yuv_frame.set_pts(Some(pts));

            state
                .video_encoder
                .send_frame(&state.yuv_frame)
                .context("encoder send_frame")?;

            flush_encoder(
                &mut state.video_encoder,
                &mut octx,
                state.video_out_index,
                video_time_base,
            )?;

            frame_count += 1;
            progress_fn(frame_count, total);
            if frame_count % 100 == 0 {
                debug!(frame_count, "processed frames");
            }
        }
    }

    let state = enc_state
        .as_mut()
        .context("no video frames were processed")?;

    // Flush decoder
    decoder.send_eof().ok();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        state.video_encoder.send_frame(&decoded_frame).ok();
        flush_encoder(
            &mut state.video_encoder,
            &mut octx,
            state.video_out_index,
            video_time_base,
        )?;
    }

    // Flush encoder
    state.video_encoder.send_eof().ok();
    flush_encoder(
        &mut state.video_encoder,
        &mut octx,
        state.video_out_index,
        video_time_base,
    )?;

    octx.write_trailer()
        .context("failed to write output trailer")?;

    info!(frame_count, "transcode complete");
    Ok(())
}

/// Drain all pending packets from the encoder and write them to the muxer.
fn flush_encoder(
    encoder: &mut encoder::Video,
    octx: &mut format::context::Output,
    stream_index: usize,
    time_base: Rational,
) -> Result<()> {
    let mut encoded = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(stream_index);
        encoded.rescale_ts(time_base, octx.stream(stream_index).unwrap().time_base());
        encoded
            .write_interleaved(octx)
            .context("failed to write encoded packet")?;
    }
    Ok(())
}

/// Return the approximate total frame count for a video file (used for
/// progress reporting).  Falls back to 0 if the count cannot be determined.
pub fn total_frames<P: AsRef<Path>>(input_path: P) -> u64 {
    ffmpeg::init().ok();
    let Ok(ictx) = format::input(&input_path) else {
        return 0;
    };
    let Some(stream) = ictx.streams().best(media::Type::Video) else {
        return 0;
    };
    // nb_frames is set by most muxers; fall back to duration × fps estimate.
    let nb = stream.frames();
    if nb > 0 {
        return nb as u64;
    }
    let dur = stream.duration(); // in stream time-base units
    let tb = stream.time_base();
    let fps = stream.avg_frame_rate();
    if dur > 0 && tb.denominator() > 0 && fps.numerator() > 0 {
        let seconds = dur as f64 * tb.numerator() as f64 / tb.denominator() as f64;
        let fps_f = fps.numerator() as f64 / fps.denominator() as f64;
        return (seconds * fps_f).round() as u64;
    }
    0
}

/// Same as [`transcode`] but calls `progress_fn(current_frame, total_frames)`
/// after every encoded frame, enabling progress reporting to a UI.
pub fn transcode_with_progress<P, Q, F, G>(
    input_path: P,
    output_path: Q,
    total: u64,
    mut frame_fn: F,
    mut progress_fn: G,
) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
    F: FnMut(&mut RgbFrame),
    G: FnMut(u64, u64),
{
    transcode_inner(
        input_path,
        output_path,
        total,
        &mut frame_fn,
        &mut progress_fn,
    )
}

/// Convenience: convert a frame to grayscale in-place (Phase 1 smoke-test).
pub fn to_grayscale(frame: &mut RgbFrame) {
    for chunk in frame.data.chunks_exact_mut(3) {
        // BT.601 luminance
        let luma =
            (0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32) as u8;
        chunk[0] = luma;
        chunk[1] = luma;
        chunk[2] = luma;
    }
}
