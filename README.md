# focus-lock-rs

![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)
![Tauri](https://img.shields.io/badge/Tauri-2.0-blue?logo=tauri)
![Svelte](https://img.shields.io/badge/Svelte-5-red?logo=svelte)
![License](https://img.shields.io/badge/License-MIT-green)

High-performance automated fancam generator. It takes a standard landscape video and a reference photo of a person (say, your bias), tracks them, and generates a stabilized, vertical (9:16) cropped video locked onto them. 

It features a modular Rust core for high-speed video processing, a CLI for batch operations, and a modern Tauri v2 desktop application for easy usage.

<img src="./src/ui.png">

##  Features

*   **Person Detection**: Uses **YOLOv8-Nano** via ONNX Runtime for fast, accurate person detection.
*   **Identity Locking**: Uses **ArcFace** (cosine similarity) to distinguish the specific target person from others in the frame.
    *   Uses your configured `--threshold` value end-to-end (CLI + GUI).
    *   Adds relock bias from last known position and adaptive recognition stride for better stability under occlusion.
*   **Identity Discovery Pass (GUI)**:
    *   Scans sampled frames first and proposes member thumbnails before tracking begins.
    *   Supports expected-member-count input and automatic informed rescan when duplicates/count mismatch are detected.
    *   Adds manual validation controls (`exclude`, duplicate resolve, low-confidence confirm) before enabling render.
    *   Lets you choose a target member card; the selected anchor is used as an extra tracking prior alongside the bias image.
    *   Persists scan sessions in the Tauri backend and validates review state server-side before allowing render.
    *   `run_fancam` now enforces validated scan session + selected identity match server-side (not only UI-side).
    *   Scan sessions now track lifecycle state (`proposed`, `validated`, `tracking`, `completed`, `failed`) with audit events.
    *   Adds manual split requests per identity and a split-rescan queue path to refresh candidate clustering.
*   **Cinematic Smoothing**: Implements a **2D Kalman Filter** to smooth camera movements, preventing jittery tracking and simulating a professional camera operator.
*   **Performance-First Pipeline**:
    *   3-thread decode/inference/encode pipeline with bounded channels.
    *   Recognition throttling before and after lock-on to avoid CPU stalls (adaptive while locked).
    *   Caps ArcFace identity checks to top-confidence person candidates per frame.
    *   Speeds up large-video processing with detection downscale, parallel tensor prep, and fast SIMD face preprocessing.
    *   Reuses rendering buffers/resizer state to reduce per-frame allocations.
    *   Emits periodic per-stage timing logs (`detect`, `identify`, `render`) for targeted profiling.
*   **Smart Rendering**:
    *   Automated 1080x1920 cropping.
    *   SIMD-accelerated resize path (`fast_image_resize`) for crop and letterbox operations.
    *   Lanczos3 upscaling for distant subjects.
    *   Fallback letterboxing when the target is lost/occluded.
*   **Cross-Platform**: Runs on Windows, macOS, and Linux.

##  Architecture

The project is organized as a Cargo workspace:

*   **`fancam-core/`**: The engine. Handles FFmpeg transcoding, ONNX inference, Kalman tracking, and image processing.
*   **`cli/`**: A command-line interface wrapper for the core engine.
*   **`src-tauri/`** & **`ui/`**: The Desktop application built with Tauri 2 and Svelte 5.

##  Logic Flow

1.  **Decode**: FFmpeg decodes the video stream into RGB frames.
2.  **Detect**: YOLOv8 runs inference on the frame to find all "Person" bounding boxes.
3.  **Identify**: The system crops faces from top-confidence person boxes and compares their embeddings against the reference "bias" image using ArcFace.
4.  **Track**:
    *   If the target is found, the Kalman filter updates position and velocity.
    *   Recognition runs at a stride (before and after lock-on) to reduce CPU load.
    *   If occluded, the filter predicts the position based on previous momentum.
5.  **Render**: The frame is cropped to the smoothed coordinates and re-encoded to H.264.

<img src="./src/process.png">
<img src="./src/output.png">

##  Prerequisites

To build and run this project, you need:

1.  **Rust**: Stable toolchain ([Install](https://rustup.rs/)).
2.  **Node.js**: Required for the UI build steps.
3.  **FFmpeg Libraries**: The project links against FFmpeg native libraries.
    *   **Ubuntu/Debian**: `sudo apt install libavutil-dev libavformat-dev libavcodec-dev libswscale-dev`
    *   **macOS**: `brew install ffmpeg`
    *   **Windows**: Set `FFMPEG_DIR` environment variable to your FFmpeg shared build.

### ONNX Runtime provider note (macOS)

This project requests CoreML execution when available.

- Default lookup path is `models/onnxruntime/lib/libonnxruntime.dylib`.
- If CoreML is unavailable in your local ONNX Runtime build, inference falls back to CPU (works, but much slower on 4K inputs).
- For best Apple Silicon performance, use an official ONNX Runtime macOS build that includes CoreML support.

##  Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/focus-lock-rs.git
    cd focus-lock-rs
    ```

2.  **Download Models:**
    Create a `models/` directory in the root and download the following ONNX models:
    *   `yolov8n.onnx` (YOLOv8 Nano)
    *   `w600k_mbf.onnx` (MobileFaceNet / ArcFace)

3.  **Build the CLI:**
    ```bash
    cargo build --release -p cli
    ```

## Desktop Application (GUI)

The GUI allows you to select files via drag-and-drop and visualize progress.

The recommended GUI flow is now:

1. Select video + bias reference + output.
2. Run **Identity Discovery** and optionally set the expected member count.
3. Select the proposed member thumbnail to track.
4. Render fancam (blocked until identity review has no unresolved warnings).

## Queue Scaffolding (SQS-first)

The Tauri backend now includes a queue abstraction and SQS-oriented message envelope for discovery jobs.

- `queue_health` command reports queue names, depths, and dedupe-key count.
- `enqueue_discovery_job` command enqueues idempotent discovery messages into the in-process queue runtime.
- `process_next_discovery_job` command simulates a worker by consuming one queued discovery message and updating stored scan state.
  - Failed jobs are retried by requeueing with incremented `attempt`, then moved to DLQ after threshold.
- `queue_worker_start`, `queue_worker_stop`, and `queue_worker_status` provide a background worker loop with configurable poll interval.
- `queue_worker_clear_events` clears the in-memory worker event history shown in the UI diagnostics panel.
- The settings drawer auto-refreshes queue + worker telemetry every 2 seconds while open.
- Queue diagnostics panel supports event filtering (`all` vs `issues`) and copying a JSON snapshot for bug reports.
- Message shape includes `message_id`, `job_id`, `idempotency_key`, `attempt`, and `trace_id`.

Environment variables:

- `FOCUS_LOCK_SQS_ENABLED` (`true|false`, default `false`)
- `FOCUS_LOCK_SQS_DISCOVERY_QUEUE` (default `identity-discovery-queue`)
- `FOCUS_LOCK_SQS_RESCAN_QUEUE` (default `identity-rescan-queue`)
- `FOCUS_LOCK_SQS_TRACKING_START_QUEUE` (default `tracking-start-queue`)
- `FOCUS_LOCK_SQS_TRACKING_MONITOR_QUEUE` (default `tracking-monitor-queue`)
- `FOCUS_LOCK_SQS_DLQ` (default `identity-events-dlq`)

This is scaffolding for the distributed phase; current queue processing is in-memory and intended for local integration and contract hardening.

Queue runtime includes unit tests for idempotent enqueue, retry-attempt incrementing, and DLQ movement.

Additional scan-session commands:

- `list_identity_scans`
- `get_identity_scan`
- `cleanup_identity_scans`

Additional queue commands:

- `enqueue_split_rescan_job`
- `process_next_rescan_job`

These support lightweight retention cleanup and lifecycle diagnostics directly from the UI.

Scan-session state is now persisted to disk in SQLite at `.focus-lock/scan_sessions.db` so lifecycle/audit data survives app restarts.

- SQLite schema uses `PRAGMA user_version` migrations (currently version `3`).
- Data is normalized into `scan_sessions` + `scan_session_events` + `app_state` tables.
- Legacy JSON (`.focus-lock/scan_sessions.json`) is read once and migrated into SQLite automatically.

Additional storage query commands:

- `query_identity_scans` (paginated/filtered scan session summaries, cursor-first)
- `query_scan_events` (paginated event history per scan, action/time filters, cursor-first)
- `scan_storage_stats` (schema version + row counts)
- `run_scan_storage_maintenance` (retention trim + optional vacuum)
- `export_diagnostics_bundle` (writes JSON diagnostics to `.focus-lock/diagnostics/`)
- `list_diagnostics_bundles`, `prune_diagnostics_bundles` (bundle inventory + retention)
- `read_diagnostics_bundle` (safe read/preview of bundle content)
- `verify_diagnostics_bundle` (manifest-backed checksum verification)
- `delete_diagnostics_bundle` (remove one diagnostics bundle by path)
- `storage_worker_start`, `storage_worker_stop`, `storage_worker_status` (scheduled maintenance worker)

Diagnostics bundle exports now write SHA-256 entries into `.focus-lock/diagnostics/manifest.json` and list/verify operations use that manifest for integrity checks.

Cursor pagination is the primary mode for scan/session queries. When a cursor is provided, backend query commands ignore `offset` to keep pagination stable and set `offset_ignored=true` in the response. `offset` remains accepted for backward compatibility on non-cursor requests.

The settings diagnostics panel now uses paginated storage queries, status filters (`all/proposed/validated/tracking/completed/failed`), paged event loading, event-action/time-window filtering, diagnostics bundle inventory/pruning/preview/verify, and maintenance worker controls for better scalability.

You can override diagnostics bundle output/read location with `FOCUS_LOCK_DIAGNOSTICS_DIR`.

The UI settings diagnostics panel can now load previous scan sessions from disk and restore their candidate/review state for continued validation.

1.  **Install frontend dependencies:**
    ```bash
    cd ui
    npm install
    ```

2.  **Run in Development Mode:**
    ```bash
    npm run tauri:dev
    ```

    For near-production processing speed while iterating UI, use:
    ```bash
    npm run tauri:dev:release
    ```

3.  **Build for Production:**
    ```bash
    npm run tauri:build
    ```
    The executable will be located in `src-tauri/target/release/bundle/`.

##  CLI Usage

The CLI provides direct access to the pipeline phases.

### Generate a Fancam
The primary command. It performs detection, identification, tracking, and rendering in one pass.

```bash
cargo run --release -p cli -- fancam \
  --video "/path/to/concert.mp4" \
  --bias "/path/to/face_photo.jpg" \
  --output "output_fancam.mp4" \
  --yolo-model "models/yolov8n.onnx" \
  --face-model "models/w600k_mbf.onnx" \
  --threshold 0.6
```

### Other Commands

*   **Smoke Test (Grayscale)**: Verifies FFmpeg linkage and basic video I/O.
    ```bash
    cargo run -p cli -- gray --input video.mp4 --output gray.mp4
    ```

*   **Debug Detection**: Draws bounding boxes around *all* detected people without cropping.
    ```bash
    cargo run -p cli -- detect --input video.mp4 --output boxes.mp4
    ```

## Contributing

Contributions are welcome! Install `rustfmt` and gimme your PRs.

```bash
cargo fmt
cargo test
```
