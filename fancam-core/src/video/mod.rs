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
//!   Thread B  inference:  apply frame_fn closure → video_xfm channel
//!   Main      encode:     receive xfm frames, lazy-init encoder, RGB→YUV, write

use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::{
    codec, encoder, format, frame, media, software::scaling, util::rational::Rational,
};
use std::path::Path;
use std::sync::mpsc;
use tracing::{debug, info};

/// Output pixel format for the encoder (YUV420p is universally compatible).
const ENCODE_FORMAT: format::Pixel = format::Pixel::YUV420P;
/// Scaling flags — fast bilinear is sufficient for the decode→encode path.
const SCALE_FLAGS: scaling::Flags = scaling::Flags::FAST_BILINEAR;

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
    duration: i64,
    out_stream_index: usize,
    src_time_base: Rational,
    dst_time_base: Rational,
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
    transcode_inner(input_path, output_path, 0, frame_fn, |_, _| {})
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
    transcode_inner(input_path, output_path, total, frame_fn, progress_fn)
}

fn transcode_inner<P, Q, F, G>(
    input_path: P,
    output_path: Q,
    total: u64,
    frame_fn: F,
    mut progress_fn: G,
) -> Result<()>
where
    P: AsRef<Path> + Send + 'static,
    Q: AsRef<Path>,
    F: FnMut(&mut RgbFrame) + Send + 'static,
    G: FnMut(u64, u64),
{
    ffmpeg::init().context("failed to initialise FFmpeg")?;

    // ── Probe input (on main thread, before spawning) ─────────────────────────
    let mut ictx = format::input(&input_path).context("could not open input file")?;

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

    info!(
        src_width,
        src_height,
        ?src_pixel_fmt,
        "opened input video stream"
    );

    // ── Channels ─────────────────────────────────────────────────────────────
    // video_raw:  Thread A → Thread B  (decoded RGB frames)
    // video_xfm:  Thread B → Main      (post-transform RGB frames)
    // audio:      Thread A → Main      (plain-data audio packets)
    let (video_raw_tx, video_raw_rx) = mpsc::sync_channel::<RgbFrame>(4);
    let (video_xfm_tx, video_xfm_rx) = mpsc::sync_channel::<RgbFrame>(4);
    let (audio_tx, audio_rx) = mpsc::sync_channel::<AudioPacket>(32);

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

    // audio_out src/dst time-bases are needed inside Thread A to fill the struct
    let audio_src_tb: Option<(usize, Rational, Rational)> = if let Some((ai, ao)) = audio_out_index
    {
        let src_tb = ictx.stream(ai).unwrap().time_base();
        let dst_tb = octx.stream(ao).unwrap().time_base();
        Some((ai, src_tb, dst_tb))
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
                let mut rgb_data = Vec::with_capacity((src_width * src_height * 3) as usize);
                for row in 0..src_height as usize {
                    let start = row * stride;
                    rgb_data.extend_from_slice(&raw[start..start + src_width as usize * 3]);
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
            if let Some((ai, src_tb, dst_tb)) = audio_src_tb {
                if stream_index == ai {
                    if let Some(data) = packet.data() {
                        let ap = AudioPacket {
                            data: data.to_vec(),
                            pts: packet.pts(),
                            dts: packet.dts(),
                            duration: packet.duration(),
                            out_stream_index: audio_out_index.unwrap().1,
                            src_time_base: src_tb,
                            dst_time_base: dst_tb,
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

    // ── Thread B: inference / transform ──────────────────────────────────────
    let thread_b = std::thread::spawn(move || -> Result<()> {
        let mut frame_fn = frame_fn;
        for mut frame in video_raw_rx {
            frame_fn(&mut frame);
            video_xfm_tx
                .send(frame)
                .map_err(|_| anyhow::anyhow!("video_xfm channel closed"))?;
        }
        // Dropping video_xfm_tx signals EOF to main thread
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
        let mut pkt = ffmpeg_next::Packet::copy(&ap.data);
        pkt.set_stream(ap.out_stream_index);
        pkt.set_pts(ap.pts);
        pkt.set_dts(ap.dts);
        pkt.set_duration(ap.duration);
        pkt.rescale_ts(ap.src_time_base, ap.dst_time_base);
        pkt.write_interleaved(octx)
            .context("failed to write audio packet")?;
        Ok(())
    }

    for frame in video_xfm_rx {
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
        let plane_data = state.out_rgb_frame.data_mut(0);
        for row in 0..state.out_height as usize {
            let dst_start = row * out_stride;
            let src_start = row * state.out_width as usize * 3;
            plane_data[dst_start..dst_start + state.out_width as usize * 3]
                .copy_from_slice(&frame.data[src_start..src_start + state.out_width as usize * 3]);
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
        progress_fn(frame_count, total);
        if frame_count % 100 == 0 {
            debug!(frame_count, "processed frames");
        }
    }

    // Join threads and propagate any errors
    thread_a
        .join()
        .map_err(|_| anyhow::anyhow!("decode thread panicked"))??;
    thread_b
        .join()
        .map_err(|_| anyhow::anyhow!("inference thread panicked"))??;

    let state = enc_state
        .as_mut()
        .context("no video frames were processed")?;

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
