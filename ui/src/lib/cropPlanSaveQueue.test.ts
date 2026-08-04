import assert from 'node:assert/strict';
import test from 'node:test';

import type { CropPlanV1 } from './cropPlan.ts';
import { CropPlanSaveQueue } from './cropPlanSaveQueue.ts';

function plan(frameCount: number): CropPlanV1 {
  return {
    schema: 'focus-lock.crop-plan',
    version: 1,
    source_fingerprint: {
      version: 1,
      algorithm: 'sha256-sampled-container-v1',
      digest: '0'.repeat(64),
      file_size: 1,
      sampled_bytes: 1,
    },
    video: {
      width: 1920,
      height: 1080,
      frame_count: frameCount,
      frame_rate_num: 30,
      frame_rate_den: 1,
      duration_ms: 1_000,
    },
    output: { width: 1080, height: 1920 },
    shots: [],
    tracks: [],
    keyframes: [],
    manual_keyframes: [],
    quality: {
      observed_keyframes: 0,
      predicted_keyframes: 0,
      held_keyframes: 0,
      mean_confidence: 0,
      min_confidence: 0,
      path_coverage: 0,
      max_gap_frames: 0,
      shot_boundary_count: 0,
    },
  };
}

function nextTurn(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

test('autosaves run in revision order and preserve enqueue-time snapshots', async () => {
  const releases: Array<() => void> = [];
  const started: number[] = [];
  const queue = new CropPlanSaveQueue(async (_path, value) => {
    started.push(value.video.frame_count);
    await new Promise<void>((resolve) => releases.push(resolve));
  });
  const firstPlan = plan(10);
  const first = queue.save('/tmp/plan.json', firstPlan);
  firstPlan.video.frame_count = 99;
  const second = queue.save('/tmp/plan.json', plan(11));

  await nextTurn();
  assert.deepEqual(started, [10]);
  releases.shift()?.();
  await first;
  await nextTurn();
  assert.deepEqual(started, [10, 11]);
  releases.shift()?.();
  assert.deepEqual(await second, { revision: 2 });
});

test('flush waits for the latest queued save before render may continue', async () => {
  let release: (() => void) | undefined;
  const queue = new CropPlanSaveQueue(
    () => new Promise<void>((resolve) => { release = resolve; }),
  );
  void queue.save('/tmp/plan.json', plan(10));
  let flushed = false;
  const barrier = queue.flush().then((revision) => {
    flushed = true;
    return revision;
  });

  await nextTurn();
  assert.equal(flushed, false);
  release?.();
  assert.equal(await barrier, 1);
  assert.equal(flushed, true);
});
