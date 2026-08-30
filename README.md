# focus-lock-rs

Turn landscape concert footage into a tracked vertical fancam.

<p align="center">
  <img src="src/ui.png" width="82%" alt="The focus-lock desktop application">
</p>

A reference photo identifies the person to follow.
The pipeline detects people, builds tracklets, resolves identity, plans a stable camera path, and renders a 1080 by 1920 crop.

<table>
  <tr>
    <td><img src="src/process.png" alt="Landscape source and tracking process"></td>
    <td><img src="src/output.png" alt="Vertical fancam output"></td>
  </tr>
</table>

## What is here

- A Rust tracking and rendering core
- A Tauri and Svelte desktop application
- A CLI for batch work
- Two-pass identity planning with an online fallback
- Crop-plan sidecars for manual correction
- Safe cancellation and a persistent render queue

The current runtime target is macOS with CoreML-enabled ONNX Runtime.

[Setup, models, CLI usage, and implementation details](GUIDE.md).
