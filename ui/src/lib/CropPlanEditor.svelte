<script lang="ts">
  import type { CropPlanV1, ManualKeyframe, PlanKeyframe } from './cropPlan';
  import {
    formatFrame,
    nearestPlanKeyframe,
    removeManualKeyframe,
    sortManualKeyframes,
    timelinePosition,
    upsertManualKeyframe,
    validateManualKeyframe,
  } from './cropPlan';

  type Props = {
    plan: CropPlanV1;
    onManualKeyframesChange?: (manualKeyframes: ManualKeyframe[]) => void;
  };

  type ManualKeyframeDraft = {
    frame_index: string;
    cx: string;
    cy: string;
    half_size: string;
  };

  let { plan, onManualKeyframesChange }: Props = $props();

  let manualKeyframes = $state<ManualKeyframe[]>([]);
  let draft = $state<ManualKeyframeDraft>(emptyDraft(1));
  let editorOpen = $state(false);
  let editingFrame = $state<number | null>(null);
  let draftError = $state('');
  let draftMessage = $state('');

  $effect(() => {
    manualKeyframes = sortManualKeyframes(plan.manual_keyframes);
  });

  const generatedKeyframes = $derived(
    [...plan.keyframes].sort((left, right) => left.frame_index - right.frame_index),
  );
  const shotBoundaries = $derived(
    [...plan.shots].sort((left, right) => left.frame_index - right.frame_index),
  );

  function emptyDraft(frameIndex: number): ManualKeyframeDraft {
    const nearest = nearestPlanKeyframe(plan.keyframes, frameIndex);
    const fallbackHalfSize = Math.max(1, Math.min(plan.video.width, plan.video.height) / 4);
    return {
      frame_index: String(frameIndex),
      cx: formatDraftNumber(nearest?.cx ?? plan.video.width / 2),
      cy: formatDraftNumber(nearest?.cy ?? plan.video.height / 2),
      half_size: formatDraftNumber(nearest?.half_size ?? fallbackHalfSize),
    };
  }

  function formatDraftNumber(value: number): string {
    return String(Math.round(value * 100) / 100);
  }

  function clampFrame(frameIndex: number): number {
    const upperBound = Math.max(1, plan.video.frame_count);
    return Math.max(1, Math.min(upperBound, Math.round(frameIndex)));
  }

  function beginNewCorrection(frameIndex: number) {
    editorOpen = true;
    editingFrame = null;
    draft = emptyDraft(clampFrame(frameIndex));
    draftError = '';
    draftMessage = '';
  }

  function beginEdit(keyframe: ManualKeyframe) {
    editorOpen = true;
    editingFrame = keyframe.frame_index;
    draft = {
      frame_index: String(keyframe.frame_index),
      cx: formatDraftNumber(keyframe.cx),
      cy: formatDraftNumber(keyframe.cy),
      half_size: formatDraftNumber(keyframe.half_size),
    };
    draftError = '';
    draftMessage = '';
  }

  function cancelEdit() {
    editorOpen = false;
    editingFrame = null;
    draftError = '';
    draftMessage = '';
  }

  function updateDraftField(field: keyof ManualKeyframeDraft, event: Event) {
    const input = event.currentTarget;
    if (!(input instanceof HTMLInputElement)) return;
    draft = { ...draft, [field]: input.value };
    draftError = '';
    draftMessage = '';
  }

  function parseDraft(): ManualKeyframe | null {
    if (Object.values(draft).some((value) => value.trim() === '')) {
      draftError = 'all four fields are required';
      return null;
    }

    const candidate: ManualKeyframe = {
      frame_index: Number(draft.frame_index),
      cx: Number(draft.cx),
      cy: Number(draft.cy),
      half_size: Number(draft.half_size),
    };
    const validation = validateManualKeyframe(
      candidate,
      plan.video.frame_count,
      plan.video.width,
      plan.video.height,
    );
    if (!validation.ok) {
      draftError = validation.message;
      return null;
    }
    return validation.value;
  }

  function saveCorrection() {
    const candidate = parseDraft();
    if (!candidate) return;

    let next = manualKeyframes;
    if (editingFrame !== null && editingFrame !== candidate.frame_index) {
      next = removeManualKeyframe(next, editingFrame);
    }
    next = upsertManualKeyframe(next, candidate);
    manualKeyframes = next;
    editingFrame = candidate.frame_index;
    draft = {
      frame_index: String(candidate.frame_index),
      cx: formatDraftNumber(candidate.cx),
      cy: formatDraftNumber(candidate.cy),
      half_size: formatDraftNumber(candidate.half_size),
    };
    draftError = '';
    draftMessage = 'saved in frame order';
    onManualKeyframesChange?.([...next]);
  }

  function deleteCorrection(frameIndex: number) {
    manualKeyframes = removeManualKeyframe(manualKeyframes, frameIndex);
    if (editingFrame === frameIndex) cancelEdit();
    onManualKeyframesChange?.([...manualKeyframes]);
  }

  function handleSubmit(event: SubmitEvent) {
    event.preventDefault();
    saveCorrection();
  }

  function handleTimelineClick(event: MouseEvent) {
    const target = event.currentTarget;
    if (!(target instanceof HTMLElement)) return;
    const bounds = target.getBoundingClientRect();
    const ratio = bounds.width > 0 ? (event.clientX - bounds.left) / bounds.width : 0;
    const lastFrame = Math.max(1, plan.video.frame_count);
    beginNewCorrection(clampFrame(1 + ratio * Math.max(0, lastFrame - 1)));
  }

  function handleTimelineKeydown(event: KeyboardEvent) {
    if (event.key === 'Home') {
      event.preventDefault();
      beginNewCorrection(1);
    } else if (event.key === 'End') {
      event.preventDefault();
      beginNewCorrection(Math.max(1, plan.video.frame_count));
    }
  }

  function markerStyle(frameIndex: number, confidence = 1): string {
    const opacity = Math.max(0.35, Math.min(1, confidence));
    return `left:${timelinePosition(frameIndex, plan.video.frame_count)}%;opacity:${opacity}`;
  }

  function sourceLabel(source: PlanKeyframe['source']): string {
    if (source === 'observed') return 'observed';
    if (source === 'predicted') return 'predicted';
    return 'held';
  }
</script>

<section class="plan-panel" aria-label="Crop plan review">
  <div class="plan-heading">
    <div>
      <div class="eyebrow">crop plan v{plan.version}</div>
      <h2>framing path</h2>
    </div>
    <button type="button" class="ghost-btn" onclick={() => beginNewCorrection(1)}>
      add correction
    </button>
  </div>

  <div class="plan-meta">
    <span>{plan.video.width}×{plan.video.height}</span>
    <span>{plan.video.frame_count} frames</span>
    <span>{plan.tracks.length} tracks</span>
    <span>{shotBoundaries.length} cuts</span>
  </div>

  <div class="timeline-block">
    <div class="timeline-scale" aria-hidden="true">
      <span>{formatFrame(1, plan.video.frame_rate_num, plan.video.frame_rate_den)}</span>
      <span>{formatFrame(Math.max(1, Math.round((plan.video.frame_count + 1) / 2)), plan.video.frame_rate_num, plan.video.frame_rate_den)}</span>
      <span>{formatFrame(Math.max(1, plan.video.frame_count), plan.video.frame_rate_num, plan.video.frame_rate_den)}</span>
    </div>
    <div
      class="timeline"
      role="button"
      tabindex="0"
      aria-label="Confidence timeline. Click to place a manual correction."
      onclick={handleTimelineClick}
      onkeydown={handleTimelineKeydown}
    >
      <div class="timeline-rail"></div>
      {#each shotBoundaries as shot (shot.frame_index)}
        <span
          class="shot-boundary"
          style={`left:${timelinePosition(shot.frame_index, plan.video.frame_count)}%`}
          title={`${shot.kind.replace('_', ' ')} at ${formatFrame(shot.frame_index, plan.video.frame_rate_num, plan.video.frame_rate_den)}`}
          aria-hidden="true"
        ></span>
      {/each}
      {#each generatedKeyframes as keyframe (keyframe.frame_index)}
        <span
          class={`timeline-marker source-${keyframe.source}`}
          style={markerStyle(keyframe.frame_index, keyframe.confidence)}
          title={`${sourceLabel(keyframe.source)} · ${Math.round(keyframe.confidence * 100)}% · ${formatFrame(keyframe.frame_index, plan.video.frame_rate_num, plan.video.frame_rate_den)}`}
          aria-hidden="true"
        ></span>
      {/each}
      {#each manualKeyframes as keyframe (keyframe.frame_index)}
        <button
          type="button"
          class="manual-marker"
          class:active={editingFrame === keyframe.frame_index}
          style={markerStyle(keyframe.frame_index)}
          aria-label={`Edit manual correction at ${formatFrame(keyframe.frame_index, plan.video.frame_rate_num, plan.video.frame_rate_den)}`}
          title={`manual correction · ${formatFrame(keyframe.frame_index, plan.video.frame_rate_num, plan.video.frame_rate_den)}`}
          onclick={(event) => {
            event.stopPropagation();
            beginEdit(keyframe);
          }}
        ></button>
      {/each}
    </div>
    <div class="timeline-hint">click the path to place a correction · home/end jump to the edges</div>
    <div class="legend" aria-label="Timeline legend">
      <span><i class="legend-dot source-observed"></i>observed</span>
      <span><i class="legend-dot source-predicted"></i>predicted</span>
      <span><i class="legend-dot source-held"></i>held</span>
      <span><i class="legend-cut"></i>shot boundary</span>
      <span><i class="legend-manual"></i>manual</span>
    </div>
  </div>

  <div class="plan-columns">
    <section class="corrections" aria-labelledby="manual-corrections-heading">
      <div class="section-heading">
        <h3 id="manual-corrections-heading">manual corrections</h3>
        <span>{manualKeyframes.length} saved</span>
      </div>

      {#if manualKeyframes.length === 0}
        <p class="empty-copy">No overrides yet. Click the timeline where the crop needs a nudge.</p>
      {:else}
        <div class="correction-list">
          {#each manualKeyframes as keyframe (keyframe.frame_index)}
            <div class="correction-row" class:active={editingFrame === keyframe.frame_index}>
              <button type="button" class="frame-button" onclick={() => beginEdit(keyframe)}>
                {formatFrame(keyframe.frame_index, plan.video.frame_rate_num, plan.video.frame_rate_den)}
                <span>frame {keyframe.frame_index}</span>
              </button>
              <span class="correction-values">
                {formatDraftNumber(keyframe.cx)}, {formatDraftNumber(keyframe.cy)} · size {formatDraftNumber(keyframe.half_size)}
              </span>
              <button type="button" class="ghost-btn tiny" onclick={() => beginEdit(keyframe)}>edit</button>
              <button type="button" class="ghost-btn tiny danger" onclick={() => deleteCorrection(keyframe.frame_index)}>remove</button>
            </div>
          {/each}
        </div>
      {/if}

      {#if editorOpen}
        <form class="editor-form" onsubmit={handleSubmit}>
          <div class="form-heading">
            <span>{editingFrame === null ? 'new correction' : `edit frame ${editingFrame}`}</span>
            <button type="button" class="ghost-btn tiny" onclick={cancelEdit}>cancel</button>
          </div>
          <div class="form-grid">
            <label>
              <span>frame</span>
              <input
                type="number"
                min="1"
                max={plan.video.frame_count > 0 ? plan.video.frame_count : undefined}
                step="1"
                value={draft.frame_index}
                oninput={(event) => updateDraftField('frame_index', event)}
              />
            </label>
            <label>
              <span>center x</span>
              <input
                type="number"
                step="0.01"
                value={draft.cx}
                oninput={(event) => updateDraftField('cx', event)}
              />
            </label>
            <label>
              <span>center y</span>
              <input
                type="number"
                step="0.01"
                value={draft.cy}
                oninput={(event) => updateDraftField('cy', event)}
              />
            </label>
            <label>
              <span>half-size</span>
              <input
                type="number"
                min="0.01"
                step="0.01"
                value={draft.half_size}
                oninput={(event) => updateDraftField('half_size', event)}
              />
            </label>
          </div>
          {#if draftError}
            <div class="field-error" role="alert">{draftError}</div>
          {:else if draftMessage}
            <div class="field-message" role="status">{draftMessage}</div>
          {/if}
          <button type="submit" class="ghost-btn save-btn">save correction</button>
        </form>
      {/if}
    </section>

    <section class="quality" aria-labelledby="plan-quality-heading">
      <div class="section-heading">
        <h3 id="plan-quality-heading">path quality</h3>
        <span>{Math.round(plan.quality.path_coverage * 100)}% covered</span>
      </div>
      <div class="quality-grid">
        <div><span>mean confidence</span><strong>{Math.round(plan.quality.mean_confidence * 100)}%</strong></div>
        <div><span>lowest confidence</span><strong>{Math.round(plan.quality.min_confidence * 100)}%</strong></div>
        <div><span>observed</span><strong>{plan.quality.observed_keyframes}</strong></div>
        <div><span>predicted</span><strong>{plan.quality.predicted_keyframes}</strong></div>
        <div><span>held</span><strong>{plan.quality.held_keyframes}</strong></div>
        <div><span>longest gap</span><strong>{plan.quality.max_gap_frames} frames</strong></div>
      </div>
    </section>
  </div>
</section>

<style>
  .plan-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    border: 1px solid #27272d;
    border-radius: 8px;
    background: #101013;
  }

  .plan-heading,
  .section-heading,
  .plan-meta,
  .legend,
  .timeline-scale,
  .timeline-hint,
  .form-heading {
    display: flex;
    align-items: center;
  }

  .plan-heading,
  .section-heading,
  .form-heading {
    justify-content: space-between;
    gap: 10px;
  }

  .eyebrow,
  .section-heading h3,
  .section-heading > span,
  .plan-meta,
  .timeline-scale,
  .timeline-hint,
  .legend,
  .form-heading,
  .editor-form label > span,
  .quality-grid span {
    font-size: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .eyebrow,
  .section-heading > span,
  .timeline-scale,
  .timeline-hint,
  .quality-grid span {
    color: #5b5b67;
  }

  h2 {
    color: #e2e2ea;
    font-size: 13px;
    font-weight: 500;
  }

  .section-heading h3 {
    color: #8f8f9a;
    font-size: 10px;
    font-weight: 500;
  }

  .plan-meta {
    flex-wrap: wrap;
    gap: 8px;
    color: #71717d;
  }

  .timeline-block {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 9px 10px 8px;
    border: 1px solid #24242b;
    border-radius: 6px;
    background: #0c0c0f;
  }

  .timeline-scale {
    justify-content: space-between;
  }

  .timeline {
    position: relative;
    height: 38px;
    border: 1px solid #1f1f26;
    border-radius: 4px;
    background: #111116;
    cursor: crosshair;
    outline: none;
  }

  .timeline:focus-visible {
    outline: 2px solid #6ee7b7;
    outline-offset: 2px;
  }

  .timeline-rail {
    position: absolute;
    top: 18px;
    right: 0;
    left: 0;
    height: 2px;
    background: #2c2c34;
  }

  .shot-boundary {
    position: absolute;
    top: 6px;
    bottom: 6px;
    width: 1px;
    background: #b0895b;
    opacity: 0.8;
    pointer-events: none;
  }

  .timeline-marker {
    position: absolute;
    top: 13px;
    width: 3px;
    height: 12px;
    transform: translateX(-1px);
    border-radius: 2px;
    pointer-events: none;
  }

  .source-observed,
  .legend-dot.source-observed {
    background: #6ee7b7;
  }

  .source-predicted,
  .legend-dot.source-predicted {
    background: #a5b4fc;
  }

  .source-held,
  .legend-dot.source-held {
    background: #f5c992;
  }

  .manual-marker {
    position: absolute;
    top: 12px;
    width: 8px;
    height: 14px;
    padding: 0;
    transform: translateX(-4px);
    border: 1px solid #0c0c0e;
    border-radius: 2px;
    background: #f0d29a;
    cursor: pointer;
    z-index: 2;
  }

  .manual-marker:hover,
  .manual-marker.active {
    background: #fff1c7;
    box-shadow: 0 0 0 2px #5b4732;
  }

  .timeline-hint {
    justify-content: flex-end;
    letter-spacing: 0;
    text-transform: none;
  }

  .legend {
    flex-wrap: wrap;
    gap: 10px;
    color: #71717d;
    letter-spacing: 0;
    text-transform: none;
  }

  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .legend-dot,
  .legend-cut,
  .legend-manual {
    display: inline-block;
    flex: 0 0 auto;
  }

  .legend-dot {
    width: 5px;
    height: 9px;
    border-radius: 2px;
  }

  .legend-cut {
    width: 1px;
    height: 11px;
    background: #b0895b;
  }

  .legend-manual {
    width: 6px;
    height: 10px;
    border-radius: 2px;
    background: #f0d29a;
  }

  .plan-columns {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(210px, 0.9fr);
    gap: 12px;
  }

  .corrections,
  .quality {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }

  .empty-copy {
    padding: 7px 0;
    color: #5b5b67;
    font-size: 11px;
  }

  .correction-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .correction-row {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto auto;
    align-items: center;
    gap: 7px;
    padding: 5px 6px;
    border: 1px solid #25252c;
    border-radius: 5px;
    background: #14141a;
  }

  .correction-row.active {
    border-color: #6ee7b7;
    background: #11261f;
  }

  .frame-button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    border: 0;
    background: none;
    color: #d2d2dc;
    cursor: pointer;
    font: inherit;
    text-align: left;
  }

  .frame-button span {
    color: #5b5b67;
    font-size: 9px;
  }

  .correction-values {
    overflow: hidden;
    color: #8f8f9a;
    font-size: 10px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .ghost-btn.danger {
    color: #c58a8a;
    border-color: #543333;
  }

  .editor-form {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    border: 1px solid #3a3a42;
    border-radius: 6px;
    background: #131318;
  }

  .form-heading {
    color: #c8c8d2;
    letter-spacing: 0;
    text-transform: none;
  }

  .form-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 7px;
  }

  .editor-form label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }

  .editor-form input {
    width: 100%;
    min-width: 0;
    padding: 5px 6px;
    border: 1px solid #30303a;
    border-radius: 4px;
    background: #17171c;
    color: #d6d6df;
    font: inherit;
    font-size: 11px;
  }

  .editor-form input:focus {
    outline: none;
    border-color: #6ee7b7;
  }

  .field-error,
  .field-message {
    font-size: 10px;
  }

  .field-error { color: #f5c992; }
  .field-message { color: #7ee6bd; }

  .save-btn {
    align-self: flex-start;
    color: #a7f3d0;
    border-color: #2f513f;
  }

  .quality-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1px;
    overflow: hidden;
    border: 1px solid #25252c;
    border-radius: 5px;
    background: #25252c;
  }

  .quality-grid div {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
    padding: 7px;
    background: #14141a;
  }

  .quality-grid strong {
    overflow: hidden;
    color: #c8c8d2;
    font-size: 12px;
    font-weight: 500;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  @media (max-width: 680px) {
    .plan-columns { grid-template-columns: 1fr; }
    .form-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .correction-row { grid-template-columns: auto minmax(0, 1fr) auto; }
    .correction-row .danger { grid-column: 3; }
    .correction-values { grid-column: 2 / -1; grid-row: 2; }
  }
</style>
