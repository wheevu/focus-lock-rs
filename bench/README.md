# focus-lock deterministic evaluation

This project needs real model/media benchmarks before publishing performance claims. The committed harness is intentionally smaller: it exercises pure solver/camera planning logic without ONNX models, FFmpeg input, or binary fixtures.

Run:

```bash
cargo run -p fancam-core --example solver_eval --locked
```

What it measures:

- deterministic synthetic tracklets processed per second
- selected identity coverage across the known single-identity fixture
- mean frame-to-frame camera center step in pixels after camera planning

What it does **not** measure:

- detector, face recognition, body ReID, FFmpeg decode/encode, or Tauri UI throughput
- real-world identity accuracy on concert footage
- end-to-end render speed

Use this as a regression/evaluation smoke test only. Publish user-facing speed or quality claims only from separate runs that include the exact media/model fixtures, command, machine, OS, Rust version, and raw results.
