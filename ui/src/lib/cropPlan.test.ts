import assert from 'node:assert/strict';
import test from 'node:test';

import type { CropPlanV1, ManualKeyframe } from './cropPlan.ts';
import {
  defaultCropPlanPath,
  isCropPlanV1,
  removeManualKeyframe,
  sortManualKeyframes,
  timelinePosition,
  upsertManualKeyframe,
  validateManualKeyframe,
} from './cropPlan.ts';

const manual = (frame_index: number, cx: number): ManualKeyframe => ({
  frame_index,
  cx,
  cy: 240,
  half_size: 90,
});

const plan: CropPlanV1 = {
  schema: 'focus-lock.crop-plan',
  version: 1,
  source_fingerprint: {
    version: 1,
    algorithm: 'sha256-sampled-container-v1',
    digest: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    file_size: 128,
    sampled_bytes: 128,
  },
  video: {
    width: 1920,
    height: 1080,
    frame_count: 120,
    frame_rate_num: 30,
    frame_rate_den: 1,
    duration_ms: 4000,
  },
  output: { width: 1080, height: 1920 },
  shots: [{ frame_index: 60, confidence: 0.9, kind: 'hard_cut' }],
  tracks: [{
    track_id: 0,
    identity_id: 2,
    first_frame: 1,
    last_frame: 120,
    observation_count: 12,
    mean_confidence: 0.8,
    best_confidence: 0.95,
    quality_score: 0.84,
    occlusion_frames: 3,
    reentry_count: 0,
  }],
  keyframes: [{
    frame_index: 1,
    cx: 960,
    cy: 540,
    half_size: 90,
    confidence: 0.8,
    source: 'observed',
  }],
  manual_keyframes: [],
  quality: {
    observed_keyframes: 1,
    predicted_keyframes: 0,
    held_keyframes: 0,
    mean_confidence: 0.8,
    min_confidence: 0.8,
    path_coverage: 1,
    max_gap_frames: 0,
    shot_boundary_count: 1,
  },
};

test('manual corrections replace duplicate frames and emit sorted output', () => {
  const sorted = sortManualKeyframes([manual(80, 800), manual(20, 200), manual(80, 880)]);
  assert.deepEqual(sorted, [manual(20, 200), manual(80, 880)]);

  const upserted = upsertManualKeyframe(sorted, manual(40, 400));
  assert.deepEqual(upserted.map((keyframe) => keyframe.frame_index), [20, 40, 80]);
  assert.deepEqual(removeManualKeyframe(upserted, 40), [manual(20, 200), manual(80, 880)]);
});

test('manual correction validation rejects unsafe frame and geometry values', () => {
  assert.equal(validateManualKeyframe(manual(121, 400), 120, 1920, 1080).ok, false);
  assert.equal(validateManualKeyframe(manual(0, 400), 120, 1920, 1080).ok, false);
  assert.equal(validateManualKeyframe({ ...manual(20, 400), half_size: 0 }, 120, 1920, 1080).ok, false);
  assert.equal(validateManualKeyframe({ ...manual(20, Number.NaN) }, 120, 1920, 1080).ok, false);
  assert.equal(validateManualKeyframe(manual(120, 400), 120, 1920, 1080).ok, true);
});

test('timeline positions clamp to the sidecar frame range', () => {
  assert.equal(timelinePosition(-1, 120), 0);
  assert.equal(timelinePosition(60, 120), (59 / 119) * 100);
  assert.equal(timelinePosition(999, 120), 100);
  assert.equal(timelinePosition(10, 0), 0);
});

test('normal render derives a clear crop-plan sidecar path', () => {
  assert.equal(defaultCropPlanPath('/videos/output.mp4'), '/videos/output.crop-plan.json');
  assert.equal(defaultCropPlanPath('/videos/output'), '/videos/output.crop-plan.json');
});

test('crop plan guard accepts the Rust-shaped payload and rejects a wrong version', () => {
  assert.equal(isCropPlanV1(plan), true);
  assert.equal(isCropPlanV1({ ...plan, version: 2 }), false);
  assert.equal(isCropPlanV1({ ...plan, schema: 'other' }), false);
});

test('frame and geometry guards share the one-based normalized contract', () => {
  assert.equal(isCropPlanV1({ ...plan, keyframes: [{ ...plan.keyframes[0], frame_index: 0 }] }), false);
  assert.equal(isCropPlanV1({ ...plan, keyframes: [{ ...plan.keyframes[0], cx: 0 }] }), false);
  const normalized = validateManualKeyframe(manual(20, 0), 120, 1920, 1080);
  assert.equal(normalized.ok, true);
  if (normalized.ok) assert.equal(normalized.value.cy, 540);
});
