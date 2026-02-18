<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { open, save } from '@tauri-apps/plugin-dialog';
  import { onMount, onDestroy } from 'svelte';

  // ── State ─────────────────────────────────────────────────────────────────

  type JobStatus = 'idle' | 'running' | 'done' | 'error';

  let videoPath    = $state('');
  let biasPath     = $state('');
  let outputPath   = $state('');
  let yoloModel    = $state('');
  let faceModel    = $state('');
  let threshold    = $state(0.6);
  let modelDir     = $state('');   // resolved at mount time

  let status:    JobStatus = $state('idle');
  let progress   = $state(0);      // 0–1
  let curFrame   = $state(0);
  let totFrames  = $state(0);
  let errMsg     = $state('');
  let resultPath = $state('');

  let showSettings = $state(false);

  // ── Mount: resolve model dir ──────────────────────────────────────────────

  onMount(async () => {
    try {
      const dir: string = await invoke('model_dir');
      modelDir  = dir;
      yoloModel = dir + '/yolov8n.onnx';
      faceModel = dir + '/w600k_mbf.onnx';
    } catch {
      // fallback — leave as empty string; user can browse
      yoloModel = 'models/yolov8n.onnx';
      faceModel = 'models/w600k_mbf.onnx';
    }
  });

  // ── Event listeners ───────────────────────────────────────────────────────

  let unlistenProgress: (() => void) | null = null;
  let unlistenDone:     (() => void) | null = null;

  async function attachListeners() {
    unlistenProgress = await listen<{ current: number; total: number; fraction: number }>(
      'fancam://progress',
      (e) => {
        curFrame  = e.payload.current;
        totFrames = e.payload.total;
        progress  = e.payload.fraction;
      }
    );
    unlistenDone = await listen<{ ok: boolean; message: string; output_path?: string }>(
      'fancam://done',
      (e) => {
        if (e.payload.ok) {
          status     = 'done';
          resultPath = e.payload.output_path ?? '';
          progress   = 1;
        } else {
          status = 'error';
          errMsg = e.payload.message;
        }
      }
    );
  }

  function detachListeners() {
    unlistenProgress?.();
    unlistenDone?.();
    unlistenProgress = null;
    unlistenDone     = null;
  }

  onDestroy(detachListeners);

  // ── File pickers ──────────────────────────────────────────────────────────

  async function pickVideo() {
    const selected = await open({
      multiple: false,
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm'] }],
    });
    if (typeof selected === 'string') {
      videoPath = selected;
      if (!outputPath) {
        outputPath = selected.replace(/\.[^.]+$/, '_fancam.mp4');
      }
      totFrames = await invoke<number>('probe_video', { path: selected });
    }
  }

  async function pickBias() {
    const selected = await open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
    });
    if (typeof selected === 'string') biasPath = selected;
  }

  async function pickOutput() {
    const selected = await save({
      filters: [{ name: 'Video', extensions: ['mp4'] }],
      defaultPath: outputPath || 'fancam.mp4',
    });
    if (selected) outputPath = selected;
  }

  async function pickYolo() {
    const selected = await open({
      multiple: false,
      defaultPath: modelDir || undefined,
      filters: [{ name: 'ONNX model', extensions: ['onnx'] }],
    });
    if (typeof selected === 'string') yoloModel = selected;
  }

  async function pickFace() {
    const selected = await open({
      multiple: false,
      defaultPath: modelDir || undefined,
      filters: [{ name: 'ONNX model', extensions: ['onnx'] }],
    });
    if (typeof selected === 'string') faceModel = selected;
  }

  // ── Job control ───────────────────────────────────────────────────────────

  async function startJob() {
    if (!videoPath || !biasPath || !outputPath) return;
    status   = 'running';
    progress = 0;
    curFrame = 0;
    errMsg   = '';
    resultPath = '';

    await attachListeners();

    try {
      const result = await invoke<{ ok: boolean; message: string; output_path?: string }>(
        'run_fancam',
        {
          args: {
            video:      videoPath,
            bias:       biasPath,
            output:     outputPath,
            yolo_model: yoloModel,
            face_model: faceModel,
            threshold,
          },
        }
      );
      if (!result.ok) {
        status = 'error';
        errMsg = result.message;
      }
    } catch (e: unknown) {
      status = 'error';
      errMsg = String(e);
    } finally {
      detachListeners();
    }
  }

  async function cancelJob() {
    await invoke('cancel_job');
    status = 'idle';
    detachListeners();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function basename(p: string) {
    return p.split(/[\\/]/).pop() ?? p;
  }

  function pct(v: number) {
    return (v * 100).toFixed(1) + '%';
  }
</script>

<!-- ── Markup ──────────────────────────────────────────────────────────────── -->

<div class="shell">

  <!-- header -->
  <header>
    <span class="logo">focus<span class="accent">-lock</span></span>
    <span class="sub">automated fancam generator</span>
    <button
      class="icon-btn"
      class:active={showSettings}
      onclick={() => (showSettings = !showSettings)}
      title="Settings"
    >⚙</button>
  </header>

  <!-- settings drawer -->
  {#if showSettings}
  <section class="drawer">
    <h3>models</h3>

    <label>
      <span>yolo model</span>
      <div class="file-row">
        <span class="path-badge">{basename(yoloModel) || '—'}</span>
        <button class="ghost-btn" onclick={pickYolo}>browse</button>
      </div>
      {#if yoloModel}
        <span class="path-full">{yoloModel}</span>
      {/if}
    </label>

    <label>
      <span>face model</span>
      <div class="file-row">
        <span class="path-badge">{basename(faceModel) || '—'}</span>
        <button class="ghost-btn" onclick={pickFace}>browse</button>
      </div>
      {#if faceModel}
        <span class="path-full">{faceModel}</span>
      {/if}
    </label>

    <label>
      <span>similarity threshold <em>{threshold.toFixed(2)}</em></span>
      <input type="range" min="0.3" max="0.95" step="0.01" bind:value={threshold} />
    </label>
  </section>
  {/if}

  <!-- main panel -->
  <main>

    <!-- inputs -->
    <section class="inputs">

      <div class="input-block" class:filled={!!videoPath}>
        <div class="input-label">video input</div>
        <button class="drop-zone" onclick={pickVideo}>
          {#if videoPath}
            <span class="filename">{basename(videoPath)}</span>
            {#if totFrames > 0}
              <span class="meta">{totFrames} frames</span>
            {/if}
          {:else}
            <span class="placeholder">click to select video</span>
          {/if}
        </button>
      </div>

      <div class="input-block" class:filled={!!biasPath}>
        <div class="input-label">bias reference</div>
        <button class="drop-zone small" onclick={pickBias}>
          {#if biasPath}
            <span class="filename">{basename(biasPath)}</span>
          {:else}
            <span class="placeholder">click to select face image</span>
          {/if}
        </button>
      </div>

      <div class="input-block" class:filled={!!outputPath}>
        <div class="input-label">output path</div>
        <button class="drop-zone small" onclick={pickOutput}>
          {#if outputPath}
            <span class="filename">{basename(outputPath)}</span>
          {:else}
            <span class="placeholder">click to set output file</span>
          {/if}
        </button>
      </div>

    </section>

    <!-- divider -->
    <div class="divider"></div>

    <!-- progress / status -->
    <section class="status-panel">
      {#if status === 'idle'}
        <div class="ready-msg">
          {videoPath && biasPath && outputPath ? 'ready to render' : 'select inputs above to begin'}
        </div>

      {:else if status === 'running'}
        <div class="prog-label">
          <span>rendering</span>
          <span class="prog-pct">{pct(progress)}</span>
          {#if totFrames > 0}
            <span class="prog-frames">{curFrame} / {totFrames}</span>
          {/if}
        </div>
        <div class="prog-track">
          <div class="prog-fill" style="width:{pct(progress)}"></div>
        </div>

      {:else if status === 'done'}
        <div class="done-msg">
          <span class="check">✓</span>
          fancam saved → <span class="result-path">{basename(resultPath)}</span>
        </div>

      {:else if status === 'error'}
        <div class="err-msg">
          <span class="x">✗</span> {errMsg}
        </div>
      {/if}
    </section>

    <!-- actions -->
    <section class="actions">
      {#if status === 'running'}
        <button class="btn-cancel" onclick={cancelJob}>cancel</button>
      {:else}
        <button
          class="btn-run"
          disabled={!videoPath || !biasPath || !outputPath}
          onclick={startJob}
        >
          {status === 'done' ? 'render again' : 'render fancam'}
        </button>
      {/if}
    </section>

  </main>

  <!-- footer -->
  <footer>
    <span>focus-lock-rs</span>
    <span class="sep">·</span>
    <span>tauri 2 + svelte 5</span>
  </footer>

</div>

<!-- ── Styles ─────────────────────────────────────────────────────────────── -->

<style>
  /* ── Reset & tokens ──────────────────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* Color tokens:
     --bg-base      #0c0c0e  deepest background
     --bg-surface   #111114  card / panel surface
     --bg-raised    #18181c  slightly raised (inputs, badges)
     --bg-hover     #1e1e23  hover state
     --border-dim   #27272d  subtle borders
     --border-mid   #3a3a42  mid-weight borders
     --border-hi    #52525e  highlighted borders
     --text-dim     #4a4a55  muted / disabled text
     --text-mid     #71717d  secondary text
     --text-muted   #8f8f9a  tertiary
     --text-base    #c8c8d2  primary body text  ← grey-shifted from white
     --text-hi      #e2e2ea  headings / filenames
     --accent-dim   #059669  emerald dark
     --accent       #6ee7b7  emerald light
  */

  :global(html, body, #app) {
    height: 100%;
    background: #0c0c0e;
    color: #c8c8d2;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono',
                 ui-monospace, 'Menlo', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Shell ───────────────────────────────────────────────────────────── */
  .shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 760px;
    margin: 0 auto;
    padding: 0 24px;
  }

  /* ── Header ──────────────────────────────────────────────────────────── */
  header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 20px 0 16px;
    border-bottom: 1px solid #27272d;
  }
  .logo {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #e2e2ea;
  }
  .accent { color: #6ee7b7; }
  .sub {
    font-size: 11px;
    color: #4a4a55;
    flex: 1;
  }
  .icon-btn {
    background: none;
    border: 1px solid transparent;
    border-radius: 5px;
    color: #52525e;
    cursor: pointer;
    font-size: 14px;
    padding: 2px 7px;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
  }
  .icon-btn:hover { color: #8f8f9a; border-color: #3a3a42; background: #18181c; }
  .icon-btn.active { color: #c8c8d2; border-color: #52525e; background: #1e1e23; }

  /* ── Settings drawer ─────────────────────────────────────────────────── */
  .drawer {
    background: #111114;
    border: 1px solid #27272d;
    border-radius: 8px;
    margin: 12px 0 0;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .drawer h3 {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a4a55;
  }
  .drawer label {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .drawer label > span {
    font-size: 11px;
    color: #71717d;
  }
  .drawer label > span em {
    color: #8f8f9a;
    font-style: normal;
    margin-left: 6px;
  }
  .file-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .path-badge {
    flex: 1;
    font-size: 11px;
    color: #8f8f9a;
    background: #18181c;
    border: 1px solid #27272d;
    border-radius: 4px;
    padding: 4px 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .path-full {
    font-size: 10px;
    color: #4a4a55;
    padding: 0 2px;
    word-break: break-all;
    line-height: 1.4;
  }
  .ghost-btn {
    background: none;
    border: 1px solid #3a3a42;
    border-radius: 4px;
    color: #71717d;
    cursor: pointer;
    font-family: inherit;
    font-size: 11px;
    padding: 4px 10px;
    white-space: nowrap;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
  }
  .ghost-btn:hover { color: #c8c8d2; border-color: #52525e; background: #1e1e23; }

  input[type="range"] {
    accent-color: #6ee7b7;
    width: 100%;
  }

  /* ── Main ────────────────────────────────────────────────────────────── */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 24px 0;
    overflow-y: auto;
  }

  /* ── Divider ─────────────────────────────────────────────────────────── */
  .divider {
    height: 1px;
    background: #27272d;
  }

  /* ── Inputs ──────────────────────────────────────────────────────────── */
  .inputs {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .input-block {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .input-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4a4a55;
  }
  .drop-zone {
    width: 100%;
    background: #111114;
    border: 1px dashed #3a3a42;
    border-radius: 6px;
    color: inherit;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    padding: 14px 16px;
    text-align: left;
    transition: border-color 0.15s, background 0.15s;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  .drop-zone.small { padding: 10px 16px; }
  .drop-zone:hover { border-color: #52525e; background: #18181c; }
  .input-block.filled .drop-zone {
    border-style: solid;
    border-color: #3a3a42;
    background: #131317;
  }
  .placeholder { color: #3a3a42; }
  .filename { color: #c8c8d2; }
  .meta { font-size: 11px; color: #52525e; }

  /* ── Status panel ────────────────────────────────────────────────────── */
  .status-panel {
    min-height: 52px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 8px;
  }
  .ready-msg {
    font-size: 12px;
    color: #4a4a55;
  }
  .prog-label {
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-size: 12px;
    color: #71717d;
  }
  .prog-pct {
    color: #6ee7b7;
    font-weight: 600;
    font-size: 13px;
  }
  .prog-frames {
    font-size: 11px;
    color: #4a4a55;
    margin-left: auto;
  }
  .prog-track {
    height: 2px;
    background: #27272d;
    border-radius: 2px;
    overflow: hidden;
  }
  .prog-fill {
    height: 100%;
    background: linear-gradient(90deg, #059669, #6ee7b7);
    border-radius: 2px;
    transition: width 0.2s ease;
  }
  .done-msg {
    font-size: 12px;
    color: #8f8f9a;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .check { color: #6ee7b7; font-size: 14px; }
  .result-path { color: #6ee7b7; }
  .err-msg {
    font-size: 12px;
    color: #f87171;
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }
  .x { font-size: 14px; flex-shrink: 0; }

  /* ── Actions ─────────────────────────────────────────────────────────── */
  .actions { display: flex; gap: 10px; }

  .btn-run {
    background: #0a3d2b;
    border: 1px solid #059669;
    border-radius: 6px;
    color: #a7f3d0;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 9px 22px;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
  }
  .btn-run:hover:not(:disabled) {
    background: #0d4f38;
    border-color: #10b981;
    color: #d1fae5;
  }
  .btn-run:disabled { opacity: 0.3; cursor: not-allowed; }

  .btn-cancel {
    background: none;
    border: 1px solid #5c1a1a;
    border-radius: 6px;
    color: #f87171;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    padding: 9px 22px;
    transition: background 0.15s, border-color 0.15s;
  }
  .btn-cancel:hover { background: #1a0808; border-color: #7f1d1d; }

  /* ── Footer ──────────────────────────────────────────────────────────── */
  footer {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 0;
    border-top: 1px solid #18181c;
    font-size: 10px;
    color: #3a3a42;
    letter-spacing: 0.04em;
  }
  .sep { color: #27272d; }
</style>
