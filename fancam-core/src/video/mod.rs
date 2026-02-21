//! video — FFmpeg bridge
//!
//! Phase 1 deliverable: open a video, iterate decoded frames, re-encode with
//! an arbitrary per-frame transform applied, and mux the result back to disk.
//!
//! The design intentionally keeps the frame callback generic (`FnMut`) so later
//! phases can slot in detection + crop logic without changing this module.
//!
//! Performance: uses a 3-thread pipeline —
//!   Thread A  decode:     demux → YUV decode → RGB convert → video_raw channel
//!   Thread B1 analysis:   detect/identify/tracking → analyzed channel
//!   Thread B2 rendering:  apply render closure → video_xfm channel
//!   Main      encode:     receive xfm frames, lazy-init encoder, RGB→YUV, write

use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::{
    codec, encoder, format, frame, media, software::scaling, util::rational::Rational,
};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::tracking::CameraState;

/// Output pixel format for the encoder (YUV420p is universally compatible).
const ENCODE_FORMAT: format::Pixel = format::Pixel::YUV420P;
/// Scaling flags — fast bilinear is sufficient for the decode→encode path.
const SCALE_FLAGS: scaling::Flags = scaling::Flags::FAST_BILINEAR;
/// Decode → inference queue depth.
const VIDEO_RAW_QUEUE: usize = 16;
/// Inference → encode queue depth.
const VIDEO_XFM_QUEUE: usize = 8;
/// Analysis → render queue depth.
const VIDEO_ANALYZED_QUEUE: usize = 8;
/// Reusable RGB buffer pool size.
const VIDEO_BUFFER_POOL: usize = 24;

/// A single decoded video frame in RGB24 format, along with its presentation
/// timestamp (in the source stream's time-base units).
pub struct RgbFrame {
    pub data: Vec<u8>, // packed RGB24, row-major
    pub width: u32,
    pub height: u32,
    pub pts: i64,
}

/// Plain-data audio packet — owns its bytes so it can cross thread boundaries
/// without holding FFmpeg types.
struct AudioPacket {
    data: Vec<u8>,
    pts: Option<i64>,
    dts: Option<i64>,
    /// Raw duration from the source packet; 0 when the source reports
    /// AV_NOPTS_VALUE (i64::MIN) to avoid overflow in rescale_ts.
    duration: i64,
    out_stream_index: usize,
    src_time_base: Rational,
    // dst_time_base is intentionally NOT stored here — the output stream's
    // time-base is not finalised until write_header() is called, so we look
    // it up dynamically inside write_audio_packet (after the header is written).
}

/// Open `input_path`, apply `frame_fn` to every frame (receives a mutable
/// `RgbFrame` — modify in-place to transform the output), and write the result
/// to `output_path` encoded as H.264.
///
/// Audio is stream-copied without re-encoding.
pub fn transcode<P, Q, F>(input_path: P, output_path: Q, frame_fn: F) -> Result<()>
where
    P: AsRef<Path> + Send + 'static,
    Q: AsRef<Path>,
    F: FnMut(&mut RgbFrame) + Send + 'static,
{
    let mut frame_fn = frame_fn;
    transcode_inner(
        input_path,
        output_path,
        0,
        |_frame| None,
        move |frame, _camera| frame_fn(frame),
        |_, _| {},
    )
}

/// Same as [`transcode`] but calls `progress_fn(current_frame, total_frames)`
/// after every encoded frame, enabling progress reporting to a UI.
pub fn transcode_with_progress<P, Q, F, G>(
    input_path: P,
    output_path: Q,
    total: u64,
    frame_fn: F,
    progress_fn: G,
) -> Result<()>
where
    P: AsRef<Path> + Send + 'static,
    Q: AsRef<Path>,
    F: FnMut(&mut RgbFrame) + Send + 'static,
    G: FnMut(u64, u64),
{
    let mut frame_fn = frame_fn;
    transcode_inner(
        input_path,
        output_path,
        total,
        |_frame| None,
        move |frame, _camera| frame_fn(frame),
        progress_fn,
    )
}

/// Staged variant: analysis and rendering run on separate worker threads.
pub fn transcode_with_progress_staged<P, Q, A, R, G>(
    input_path: P,
    output_path: Q,
    total: u64,
    analyze_fn: A,
    render_fn: R,
    progress_fn: G,
) -> Result<()>
where
    P: AsRef<Path> + Send + 'static,
    Q: AsRef<Path>,
    A: FnMut(&RgbFrame) -> Option<CameraState> + Send + 'static,
    R: FnMut(&mut RgbFrame, Option<CameraState>) + Send + 'static,
    G: FnMut(u64, u64),
{
    transcode_inner(
        input_path,
        output_path,
        total,
        analyze_fn,
        render_fn,
        progress_fn,
    )
}

fn transcode_inner<P, Q, A, R, G>(
    input_path: P,
    output_path: Q,
    total: u64,
    analyze_fn: A,
    render_fn: R,
    mut progress_fn: G,
) -> Result<()>
where
    P: AsRef<Path> + Send + 'static,
    Q: AsRef<Path>,
    A: FnMut(&RgbFrame) -> Option<CameraState> + Send + 'static,
    R: FnMut(&mut RgbFrame, Option<CameraState>) + Send + 'static,
    G: FnMut(u64, u64),
{
    ffmpeg::init().context("failed to initialise FFmpeg")?;

    // ── Probe input (on main thread, before spawning) ─────────────────────────
    let mut ictx = open_input_with_hwaccel(&input_path)?;

    let video_stream_index = ictx
        .streams()
        .best(media::Type::Video)
        .context("no video stream found in input")?
        .index();

    let audio_stream_index = ictx.streams().best(media::Type::Audio).map(|s| s.index());

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

    // Capture stream duration for PTS-based progress when total_frames is 0.
    // Try stream-level first, then format-level (converted to stream time-base).
    let stream_duration_pts: i64 = {
        let sd = ictx.stream(video_stream_index).unwrap().duration();
        if sd > 0 {
            sd
        } else {
            // Convert format-level duration (AV_TIME_BASE units) to stream tb.
            let fmt_dur = ictx.duration(); // i64 in AV_TIME_BASE (1/1_000_000)
            if fmt_dur > 0 && video_time_base.denominator() > 0 {
                ffmpeg::Rescale::rescale(&fmt_dur, ffmpeg::rescale::TIME_BASE, video_time_base)
            } else {
                0
            }
        }
    };

    info!(
        src_width,
        src_height,
        ?src_pixel_fmt,
        "opened input video stream"
    );

    // ── Channels ─────────────────────────────────────────────────────────────
    // video_raw:      Thread A  → Thread B1 (decoded RGB frames)
    // video_analyzed: Thread B1 → Thread B2 (camera state + frame)
    // video_xfm:      Thread B2 → Main      (post-transform RGB frames)
    // audio:      Thread A → Main      (plain-data audio packets)
    let (video_raw_tx, video_raw_rx) = mpsc::sync_channel::<RgbFrame>(VIDEO_RAW_QUEUE);
    let (video_analyzed_tx, video_analyzed_rx) =
        mpsc::sync_channel::<(RgbFrame, Option<CameraState>)>(VIDEO_ANALYZED_QUEUE);
    let (video_xfm_tx, video_xfm_rx) = mpsc::sync_channel::<RgbFrame>(VIDEO_XFM_QUEUE);
    let (audio_tx, audio_rx) = mpsc::sync_channel::<AudioPacket>(32);
    let (recycle_tx, recycle_rx) = mpsc::sync_channel::<Vec<u8>>(VIDEO_BUFFER_POOL);

    // ── Output — lazily initialised on first frame ────────────────────────────
    let mut octx = format::output(&output_path).context("could not create output context")?;
    let global_header = octx
        .format()
        .flags()
        .contains(format::flag::Flags::GLOBAL_HEADER);
    let encoder_codec = encoder::find(codec::Id::H264)
        .context("H.264 encoder not found — is FFmpeg built with libx264?")?;

    // Pre-create the audio output stream (stream-copy) so stream indices are
    // stable before write_header.
    let audio_out_index: Option<(usize, usize)> = if let Some(ai) = audio_stream_index {
        let in_stream = ictx.stream(ai).unwrap();
        let mut audio_out = octx.add_stream(ffmpeg_next::codec::Id::None)?;
        audio_out.set_parameters(in_stream.parameters());
        Some((ai, audio_out.index()))
    } else {
        None
    };

    // audio_out src time-base is needed inside Thread A to fill the struct;
    // dst_time_base is resolved lazily at write time (post write_header).
    let audio_src_tb: Option<(usize, Rational)> = if let Some((ai, _ao)) = audio_out_index {
        let src_tb = ictx.stream(ai).unwrap().time_base();
        Some((ai, src_tb))
    } else {
        None
    };

    // ── Thread A: decode ──────────────────────────────────────────────────────
    let thread_a = std::thread::spawn(move || -> Result<()> {
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

        let mut decoded_frame = frame::Video::empty();
        let mut rgb_frame = frame::Video::empty();

        let mut send_video = |dec: &mut ffmpeg_next::decoder::Video| -> Result<()> {
            while dec.receive_frame(&mut decoded_frame).is_ok() {
                to_rgb
                    .run(&decoded_frame, &mut rgb_frame)
                    .context("to-RGB scaling failed")?;

                // De-stride: copy only actual pixel rows (no padding)
                let stride = rgb_frame.stride(0);
                let raw = rgb_frame.data(0);
                let row_bytes = src_width as usize * 3;
                let frame_len = (src_width * src_height * 3) as usize;
                let mut rgb_data = match recycle_rx.try_recv() {
                    Ok(mut buf) => {
                        buf.resize(frame_len, 0);
                        buf
                    }
                    Err(_) => vec![0u8; frame_len],
                };
                for row in 0..src_height as usize {
                    let src_start = row * stride;
                    let dst_start = row * row_bytes;
                    rgb_data[dst_start..dst_start + row_bytes]
                        .copy_from_slice(&raw[src_start..src_start + row_bytes]);
                }

                let pts = decoded_frame.pts().unwrap_or(0);
                video_raw_tx
                    .send(RgbFrame {
                        data: rgb_data,
                        width: src_width,
                        height: src_height,
                        pts,
                    })
                    // Receiver dropped (encode thread exited early due to error)
                    .map_err(|_| anyhow::anyhow!("video_raw channel closed"))?;
            }
            Ok(())
        };

        for (stream, packet) in ictx.packets() {
            let stream_index = stream.index();

            // Route audio packets as plain data structs
            if let Some((ai, src_tb)) = audio_src_tb {
                if stream_index == ai {
                    if let Some(data) = packet.data() {
                        let ap = AudioPacket {
                            data: data.to_vec(),
                            pts: packet.pts(),
                            dts: packet.dts(),
                            // AV_NOPTS_VALUE (i64::MIN) is not a real duration;
                            // treat it as 0 to avoid overflow inside rescale_ts.
                            duration: packet.duration().max(0),
                            out_stream_index: audio_out_index.unwrap().1,
                            src_time_base: src_tb,
                        };
                        // Ignore send errors — main thread may have exited on
                        // error; we don't want to mask the real error.
                        let _ = audio_tx.send(ap);
                    }
                    continue;
                }
            }

            if stream_index != video_stream_index {
                continue;
            }

            decoder
                .send_packet(&packet)
                .context("decoder send_packet")?;
            send_video(&mut decoder)?;
        }

        // Tail-drain: flush decoder before signalling EOF to Thread B
        decoder.send_eof().ok();
        send_video(&mut decoder)?;

        // Dropping video_raw_tx signals EOF to Thread B
        Ok(())
    });

    // ── Thread B1: analysis ───────────────────────────────────────────────────
    let thread_b1 = std::thread::spawn(move || -> Result<()> {
        let mut analyze_fn = analyze_fn;
        for frame in video_raw_rx {
            let camera = analyze_fn(&frame);
            video_analyzed_tx
                .send((frame, camera))
                .map_err(|_| anyhow::anyhow!("video_analyzed channel closed"))?;
        }
        Ok(())
    });

    // ── Thread B2: rendering ──────────────────────────────────────────────────
    let thread_b2 = std::thread::spawn(move || -> Result<()> {
        let mut render_fn = render_fn;
        for (mut frame, camera) in video_analyzed_rx {
            render_fn(&mut frame, camera);
            video_xfm_tx
                .send(frame)
                .map_err(|_| anyhow::anyhow!("video_xfm channel closed"))?;
        }
        Ok(())
    });

    // ── Main thread: encode ───────────────────────────────────────────────────

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
    let mut header_written = false;
    // Audio packets arriving before the header is written are buffered here.
    let mut audio_buffer: Vec<AudioPacket> = Vec::new();
    let mut frame_count = 0u64;
    // Wall-clock start for throughput / ETA logging.
    let encode_start = Instant::now();

    /// Drain all pending audio from the channel into either the buffer (pre-
    /// header) or directly to the muxer (post-header).
    fn drain_audio(
        audio_rx: &mpsc::Receiver<AudioPacket>,
        octx: &mut format::context::Output,
        audio_buffer: &mut Vec<AudioPacket>,
        header_written: bool,
    ) -> Result<()> {
        loop {
            match audio_rx.try_recv() {
                Ok(ap) => {
                    if !header_written {
                        audio_buffer.push(ap);
                    } else {
                        write_audio_packet(octx, &ap)?;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        Ok(())
    }

    fn write_audio_packet(octx: &mut format::context::Output, ap: &AudioPacket) -> Result<()> {
        // Resolve the output stream's time-base at write time — it is not
        // finalised until write_header() has been called, so we must NOT
        // capture it earlier (it would be 0/1 and break rescale_ts).
        let dst_time_base = octx.stream(ap.out_stream_index).unwrap().time_base();
        let mut pkt = ffmpeg_next::Packet::copy(&ap.data);
        pkt.set_stream(ap.out_stream_index);
        pkt.set_pts(ap.pts);
        pkt.set_dts(ap.dts);
        pkt.set_duration(ap.duration);
        pkt.rescale_ts(ap.src_time_base, dst_time_base);
        pkt.write_interleaved(octx)
            .context("failed to write audio packet")?;
        Ok(())
    }

    for mut frame in video_xfm_rx {
        drain_audio(&audio_rx, &mut octx, &mut audio_buffer, header_written)?;

        let out_w = frame.width;
        let out_h = frame.height;
        let pts = frame.pts;

        // Lazy encoder init on first frame (dimensions now known)
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
                    ffmpeg_next::Dictionary::from_iter([("crf", "18"), ("preset", "veryfast")]),
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

            // Flush all audio that arrived before the header
            for ap in audio_buffer.drain(..) {
                write_audio_packet(&mut octx, &ap)?;
            }
            // Drain any audio that arrived during header setup
            drain_audio(&audio_rx, &mut octx, &mut audio_buffer, header_written)?;

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

        // Copy transformed RGB data into the output AVFrame (handling stride)
        let out_stride = state.out_rgb_frame.stride(0);
        let row_bytes = state.out_width as usize * 3;
        let plane_data = state.out_rgb_frame.data_mut(0);
        if out_stride == row_bytes {
            plane_data[..row_bytes * state.out_height as usize]
                .copy_from_slice(&frame.data[..row_bytes * state.out_height as usize]);
        } else {
            for row in 0..state.out_height as usize {
                let dst_start = row * out_stride;
                let src_start = row * row_bytes;
                plane_data[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&frame.data[src_start..src_start + row_bytes]);
            }
        }

        // RGB24 → YUV420P
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

        // Compute progress: prefer frame-count-based, fall back to PTS-based.
        if total > 0 {
            progress_fn(frame_count, total);
        } else if stream_duration_pts > 0 {
            // Estimate total from duration and report PTS-based fraction.
            let frac_pts = pts as f64 / stream_duration_pts as f64;
            let est_total = (frame_count as f64 / frac_pts.max(1e-9)).round() as u64;
            progress_fn(frame_count, est_total.max(frame_count));
        } else {
            // No duration info at all — report 0/0.
            progress_fn(frame_count, 0);
        }

        // Log the very first frame so the user knows encoding has started,
        // then every 100 frames with throughput and ETA.
        let should_log = frame_count == 1 || frame_count % 100 == 0;
        if should_log {
            let elapsed = encode_start.elapsed().as_secs_f64();
            let fps = if elapsed > 0.0 {
                frame_count as f64 / elapsed
            } else {
                0.0
            };
            if total > 0 {
                let remaining = (total - frame_count) as f64;
                let eta_secs = if fps > 0.0 { remaining / fps } else { 0.0 };
                info!(
                    frame_count,
                    total,
                    fps = format!("{fps:.2}"),
                    eta_secs = format!("{eta_secs:.0}"),
                    "encoding progress"
                );
            } else {
                info!(frame_count, fps = format!("{fps:.2}"), "encoding progress");
            }
            debug!(frame_count, "processed frames");
        }

        let mut recycled = std::mem::take(&mut frame.data);
        recycled.clear();
        let _ = recycle_tx.try_send(recycled);
    }

    // Join threads and propagate any errors
    thread_a
        .join()
        .map_err(|_| anyhow::anyhow!("decode thread panicked"))??;
    thread_b1
        .join()
        .map_err(|_| anyhow::anyhow!("analysis thread panicked"))??;
    thread_b2
        .join()
        .map_err(|_| anyhow::anyhow!("render thread panicked"))??;

    let state = enc_state
        .as_mut()
        .context("no video frames were processed")?;

    // Drain any audio packets that arrived after the last video frame was
    // processed (e.g. audio tail at end of file).  Threads are already joined
    // so audio_rx is disconnected; try_recv drains whatever remains.
    drain_audio(&audio_rx, &mut octx, &mut audio_buffer, header_written)?;

    // Flush encoder tail
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
    let Ok(ictx) = open_input_with_hwaccel(&input_path) else {
        return 0;
    };
    let Some(stream) = ictx.streams().best(media::Type::Video) else {
        return 0;
    };
    let fps = stream.avg_frame_rate();
    let fps_f = if fps.numerator() > 0 && fps.denominator() > 0 {
        fps.numerator() as f64 / fps.denominator() as f64
    } else {
        0.0
    };

    // 1) nb_frames is set by most muxers.
    let nb = stream.frames();
    if nb > 0 {
        return nb as u64;
    }

    // 2) Stream-level duration (in stream time-base units).
    let dur = stream.duration();
    let tb = stream.time_base();
    if dur > 0 && tb.denominator() > 0 && fps_f > 0.0 {
        let seconds = dur as f64 * tb.numerator() as f64 / tb.denominator() as f64;
        return (seconds * fps_f).round() as u64;
    }

    // 3) Format-level duration (in AV_TIME_BASE = 1/1_000_000 units).
    //    Many YouTube-ripped files lack stream-level metadata but have this.
    let fmt_dur = ictx.duration(); // i64, in AV_TIME_BASE units
    if fmt_dur > 0 && fps_f > 0.0 {
        let seconds = fmt_dur as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
        return (seconds * fps_f).round() as u64;
    }

    0
}

fn open_input_with_hwaccel<P: AsRef<Path>>(input_path: P) -> Result<format::context::Input> {
    let dict = ffmpeg_next::Dictionary::from_iter([("hwaccel", "videotoolbox")]);
    match format::input_with_dictionary(&input_path, dict) {
        Ok(ctx) => {
            info!("opened input with VideoToolbox decode hint");
            Ok(ctx)
        }
        Err(err) => {
            warn!(
                error = %err,
                "VideoToolbox decode hint unavailable; falling back to default decode"
            );
            format::input(&input_path).context("could not open input file")
        }
    }
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
