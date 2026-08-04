export const CROP_PLAN_SCHEMA = 'focus-lock.crop-plan' as const;
export const CROP_PLAN_VERSION = 1 as const;
export const CROP_PLAN_OUTPUT_WIDTH = 1080 as const;
export const CROP_PLAN_OUTPUT_HEIGHT = 1920 as const;

export type PlanKeyframeSource = 'observed' | 'predicted' | 'held';

export interface VideoMetadata {
  width: number;
  height: number;
  frame_count: number;
  frame_rate_num: number;
  frame_rate_den: number;
  duration_ms: number | null;
}

export interface SourceVideoFingerprint {
  version: number;
  algorithm: 'sha256-sampled-container-v1';
  digest: string;
  file_size: number;
  sampled_bytes: number;
}

export type CropOutput = {
  width: number;
  height: number;
};

export interface ShotBoundary {
  frame_index: number;
  confidence: number;
  kind: 'hard_cut';
}

export interface TrackQuality {
  track_id: number;
  identity_id: number | null;
  first_frame: number;
  last_frame: number;
  observation_count: number;
  mean_confidence: number;
  best_confidence: number;
  quality_score: number;
  occlusion_frames: number;
  reentry_count: number;
}

export interface PlanKeyframe {
  frame_index: number;
  cx: number;
  cy: number;
  half_size: number;
  confidence: number;
  source: PlanKeyframeSource;
}

export interface ManualKeyframe {
  frame_index: number;
  cx: number;
  cy: number;
  half_size: number;
}

export interface PlanQualityMetrics {
  observed_keyframes: number;
  predicted_keyframes: number;
  held_keyframes: number;
  mean_confidence: number;
  min_confidence: number;
  path_coverage: number;
  max_gap_frames: number;
  shot_boundary_count: number;
}

export interface CropPlanV1 {
  schema: typeof CROP_PLAN_SCHEMA;
  version: typeof CROP_PLAN_VERSION;
  source_fingerprint: SourceVideoFingerprint;
  video: VideoMetadata;
  output: CropOutput;
  shots: ShotBoundary[];
  tracks: TrackQuality[];
  keyframes: PlanKeyframe[];
  manual_keyframes: ManualKeyframe[];
  quality: PlanQualityMetrics;
}

export type ManualKeyframeValidation =
  | { ok: true; value: ManualKeyframe }
  | { ok: false; message: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function isNullableNonNegativeInteger(value: unknown): value is number | null {
  return value === null || isNonNegativeInteger(value);
}

function isScore(value: unknown): value is number {
  return isFiniteNumber(value) && value >= 0 && value <= 1;
}

function isVideoMetadata(value: unknown): value is VideoMetadata {
  if (!isRecord(value)) return false;
  return (
    isNonNegativeInteger(value.width) &&
    value.width > 0 &&
    isNonNegativeInteger(value.height) &&
    value.height > 0 &&
    isNonNegativeInteger(value.frame_count) &&
    isNonNegativeInteger(value.frame_rate_num) &&
    isNonNegativeInteger(value.frame_rate_den) &&
    (value.frame_rate_den === 0 || value.frame_rate_num > 0) &&
    (value.duration_ms === null || isNonNegativeInteger(value.duration_ms))
  );
}

function isSourceVideoFingerprint(value: unknown): value is SourceVideoFingerprint {
  if (!isRecord(value)) return false;
  return (
    value.version === 1 &&
    value.algorithm === 'sha256-sampled-container-v1' &&
    typeof value.digest === 'string' &&
    /^[0-9a-f]{64}$/.test(value.digest) &&
    isNonNegativeInteger(value.file_size) &&
    isNonNegativeInteger(value.sampled_bytes) &&
    value.sampled_bytes <= value.file_size
  );
}

function isPlanKeyframeSource(value: unknown): value is PlanKeyframeSource {
  return value === 'observed' || value === 'predicted' || value === 'held';
}

function isShotBoundary(value: unknown, video: VideoMetadata): value is ShotBoundary {
  if (!isRecord(value)) return false;
  return (
    isPositiveInteger(value.frame_index) &&
    (video.frame_count === 0 || value.frame_index <= video.frame_count) &&
    isScore(value.confidence) &&
    value.kind === 'hard_cut'
  );
}

function isTrackQuality(value: unknown, video: VideoMetadata): value is TrackQuality {
  if (!isRecord(value)) return false;
  return (
    isNonNegativeInteger(value.track_id) &&
    isNullableNonNegativeInteger(value.identity_id) &&
    isPositiveInteger(value.first_frame) &&
    isPositiveInteger(value.last_frame) &&
    value.first_frame <= value.last_frame &&
    (video.frame_count === 0 || value.last_frame <= video.frame_count) &&
    isNonNegativeInteger(value.observation_count) &&
    value.observation_count > 0 &&
    isScore(value.mean_confidence) &&
    isScore(value.best_confidence) &&
    isScore(value.quality_score) &&
    isNonNegativeInteger(value.occlusion_frames) &&
    isNonNegativeInteger(value.reentry_count)
  );
}

function isPlanKeyframe(value: unknown, video: VideoMetadata): value is PlanKeyframe {
  if (!isRecord(value)) return false;
  return (
    isPositiveInteger(value.frame_index) &&
    (video.frame_count === 0 || value.frame_index <= video.frame_count) &&
    isFiniteNumber(value.cx) &&
    isFiniteNumber(value.cy) &&
    isFiniteNumber(value.half_size) &&
    value.half_size > 0 &&
    isNormalizedGeometry(value.cx, value.cy, value.half_size, video.width, video.height) &&
    isScore(value.confidence) &&
    isPlanKeyframeSource(value.source)
  );
}

function isManualKeyframe(value: unknown, video: VideoMetadata): value is ManualKeyframe {
  if (!isRecord(value)) return false;
  return (
    isPositiveInteger(value.frame_index) &&
    (video.frame_count === 0 || value.frame_index <= video.frame_count) &&
    isFiniteNumber(value.cx) &&
    isFiniteNumber(value.cy) &&
    isFiniteNumber(value.half_size) &&
    value.half_size > 0 &&
    isNormalizedGeometry(value.cx, value.cy, value.half_size, video.width, video.height)
  );
}

function isQualityMetrics(value: unknown): value is PlanQualityMetrics {
  if (!isRecord(value)) return false;
  return (
    isNonNegativeInteger(value.observed_keyframes) &&
    isNonNegativeInteger(value.predicted_keyframes) &&
    isNonNegativeInteger(value.held_keyframes) &&
    isScore(value.mean_confidence) &&
    isScore(value.min_confidence) &&
    isScore(value.path_coverage) &&
    isNonNegativeInteger(value.max_gap_frames) &&
    isNonNegativeInteger(value.shot_boundary_count)
  );
}

/** Narrow JSON received from a future sidecar/Tauri boundary before rendering it. */
export function isCropPlanV1(value: unknown): value is CropPlanV1 {
  if (!isRecord(value)) return false;
  if (value.schema !== CROP_PLAN_SCHEMA || value.version !== CROP_PLAN_VERSION) return false;
  if (
    !isVideoMetadata(value.video) ||
    !isSourceVideoFingerprint(value.source_fingerprint) ||
    !isRecord(value.output) ||
    !isQualityMetrics(value.quality)
  ) {
    return false;
  }
  const video = value.video;

  return (
    value.output.width === CROP_PLAN_OUTPUT_WIDTH &&
    value.output.height === CROP_PLAN_OUTPUT_HEIGHT &&
    Array.isArray(value.shots) &&
    value.shots.every((shot) => isShotBoundary(shot, video)) &&
    Array.isArray(value.tracks) &&
    value.tracks.every((track) => isTrackQuality(track, video)) &&
    Array.isArray(value.keyframes) &&
    value.keyframes.every((keyframe) => isPlanKeyframe(keyframe, video)) &&
    Array.isArray(value.manual_keyframes) &&
    value.manual_keyframes.every((keyframe) => isManualKeyframe(keyframe, video))
  );
}

/** Default sidecar created by the normal GUI render flow. */
export function defaultCropPlanPath(outputPath: string): string {
  const trimmed = outputPath.trim();
  if (!trimmed) return 'crop-plan.json';
  const separator = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  const dot = trimmed.lastIndexOf('.');
  const stem = dot > separator ? trimmed.slice(0, dot) : trimmed;
  return `${stem}.crop-plan.json`;
}

/** Keep manual corrections immutable, unique by frame, and sorted for sidecar output. */
export function sortManualKeyframes(keyframes: readonly ManualKeyframe[]): ManualKeyframe[] {
  const byFrame = new Map<number, ManualKeyframe>();
  for (const keyframe of keyframes) {
    byFrame.set(keyframe.frame_index, { ...keyframe });
  }

  return [...byFrame.values()].sort((left, right) => left.frame_index - right.frame_index);
}

export function validateManualKeyframe(
  keyframe: ManualKeyframe,
  frameCount: number,
  videoWidth: number,
  videoHeight: number,
): ManualKeyframeValidation {
  if (!Number.isInteger(keyframe.frame_index) || keyframe.frame_index < 1) {
    return { ok: false, message: 'frame must be a whole number from 1 onward' };
  }
  if (frameCount > 0 && keyframe.frame_index > frameCount) {
    return { ok: false, message: `frame must be between 1 and ${frameCount}` };
  }
  if (!Number.isFinite(keyframe.cx) || !Number.isFinite(keyframe.cy)) {
    return { ok: false, message: 'crop center coordinates must be finite numbers' };
  }
  if (!Number.isFinite(keyframe.half_size) || keyframe.half_size <= 0) {
    return { ok: false, message: 'half-size must be greater than 0' };
  }

  const [cx, cy, half_size] = normalizeGeometry(
    keyframe.cx,
    keyframe.cy,
    keyframe.half_size,
    videoWidth,
    videoHeight,
  );
  return { ok: true, value: { frame_index: keyframe.frame_index, cx, cy, half_size } };
}

export function upsertManualKeyframe(
  keyframes: readonly ManualKeyframe[],
  keyframe: ManualKeyframe,
): ManualKeyframe[] {
  return sortManualKeyframes([
    ...keyframes.filter((existing) => existing.frame_index !== keyframe.frame_index),
    keyframe,
  ]);
}

export function removeManualKeyframe(
  keyframes: readonly ManualKeyframe[],
  frameIndex: number,
): ManualKeyframe[] {
  return sortManualKeyframes(keyframes.filter((keyframe) => keyframe.frame_index !== frameIndex));
}

export function timelinePosition(frameIndex: number, frameCount: number): number {
  if (frameCount <= 1 || !Number.isFinite(frameIndex)) return 0;
  return Math.max(0, Math.min(100, ((frameIndex - 1) / (frameCount - 1)) * 100));
}

/** Normalize geometry with the same 9:16 crop-window math used by Rust. */
export function normalizeGeometry(
  cx: number,
  cy: number,
  halfSize: number,
  width: number,
  height: number,
): [number, number, number] {
  const widthValue = Math.max(1, width);
  const heightValue = Math.max(1, height);
  const safeHalfSize = Math.min(Math.max(1, halfSize), Math.max(widthValue, heightValue));
  const aspect = 1080 / 1920;
  const cropWidth = Math.min(Math.max(safeHalfSize * 2.5, 1080), widthValue);
  const cropHeight = Math.min(cropWidth / aspect, heightValue);
  const normalizedCropWidth = cropHeight * aspect;
  const normalizedCx = Number.isFinite(cx)
    ? Math.min(
        Math.max(cx, normalizedCropWidth / 2),
        Math.max(normalizedCropWidth / 2, widthValue - normalizedCropWidth / 2),
      )
    : widthValue / 2;
  const normalizedCy = Number.isFinite(cy)
    ? Math.min(
        Math.max(cy, cropHeight / 2),
        Math.max(cropHeight / 2, heightValue - cropHeight / 2),
      )
    : heightValue / 2;
  return [normalizedCx, normalizedCy, safeHalfSize];
}

function isNormalizedGeometry(
  cx: number,
  cy: number,
  halfSize: number,
  width: number,
  height: number,
): boolean {
  const [normalizedCx, normalizedCy, normalizedHalfSize] = normalizeGeometry(
    cx,
    cy,
    halfSize,
    width,
    height,
  );
  return (
    Math.abs(cx - normalizedCx) <= 1e-3 &&
    Math.abs(cy - normalizedCy) <= 1e-3 &&
    Math.abs(halfSize - normalizedHalfSize) <= 1e-3
  );
}

function isPositiveInteger(value: unknown): value is number {
  return isNonNegativeInteger(value) && value >= 1;
}

export function nearestPlanKeyframe(
  keyframes: readonly PlanKeyframe[],
  frameIndex: number,
): PlanKeyframe | undefined {
  return keyframes.reduce<PlanKeyframe | undefined>((nearest, candidate) => {
    if (!nearest) return candidate;
    return Math.abs(candidate.frame_index - frameIndex) < Math.abs(nearest.frame_index - frameIndex)
      ? candidate
      : nearest;
  }, undefined);
}

export function formatFrame(frameIndex: number, frameRateNum: number, frameRateDen: number): string {
  if (frameRateNum > 0 && frameRateDen > 0) {
    const seconds = (Math.max(1, frameIndex) - 1) * frameRateDen / frameRateNum;
    const minutes = Math.floor(seconds / 60);
    const remainder = (seconds % 60).toFixed(1).padStart(4, '0');
    return `${minutes}:${remainder}`;
  }
  return `frame ${frameIndex}`;
}
