export type JobStatus = 'idle' | 'running' | 'cancelling' | 'done' | 'error';
export type ScanStatus = 'idle' | 'running' | 'cancelling' | 'done' | 'error';
export type ProcessingMode = 'fast' | 'balanced' | 'quality';
export type ScanSessionStatus = 'proposed' | 'validated' | 'tracking' | 'completed' | 'failed';

export type IdentityCandidate = {
  id: number;
  confidence: number;
  observations: number;
  first_frame: number;
  last_frame: number;
  anchor_x: number;
  anchor_y: number;
  anchor_x_norm?: number | null;
  anchor_y_norm?: number | null;
  thumbnail_data_url: string;
  embedding?: number[] | null;
  body_embedding?: number[] | null;
  preview_score?: number | null;
  preview_observations?: number | null;
};

export type DuplicatePair = {
  a: number;
  b: number;
  similarity: number;
};

export type ScanResult = {
  scan_id: string;
  ok: boolean;
  message: string;
  video: string;
  sampled_frames: number;
  total_decoded_frames: number;
  proposed_count: number;
  processing_mode: string;
  expected_count?: number | null;
  rescan_performed: boolean;
  needs_review: boolean;
  rejected_embeddings: number;
  suppressed_clusters: number;
  merged_clusters: number;
  provisional_tracklets: number;
  candidates: IdentityCandidate[];
  duplicates: DuplicatePair[];
};

export type IdentityReviewResult = {
  ok: boolean;
  ready: boolean;
  blockers: string[];
  active_count: number;
  selected_identity_id?: number | null;
  selected_anchor_x?: number | null;
  selected_anchor_y?: number | null;
};

export type QueueHealth = {
  depths: {
    discovery: number;
    rescan: number;
    dlq: number;
  };
  dedupe_keys: number;
};

export type QueueActionResult = {
  accepted?: boolean;
  deduplicated?: boolean;
  queue: string;
  message_id?: string | null;
  job_id?: string | null;
  moved_to_dlq?: boolean;
  requeued?: boolean;
  cancelled?: boolean;
  attempt?: number | null;
  error?: string | null;
  depth?: number;
  remaining_depth?: number;
  processed?: boolean;
};

export type QueueWorkerStatus = {
  running: boolean;
  stop_requested: boolean;
  poll_interval_ms: number;
  max_attempts_before_dlq: number;
  processed_total: number;
  last_error?: string | null;
  recent_events: QueueWorkerEvent[];
};

export type QueueWorkerEvent = {
  at_ms: number;
  queue: string;
  message_id?: string | null;
  job_id?: string | null;
  attempt?: number | null;
  moved_to_dlq: boolean;
  requeued: boolean;
  error?: string | null;
};

export type ScanSessionEvent = {
  at_ms: number;
  action: string;
  details: string;
};

export type ReviewDuplicateResolution = { a: number; b: number; keep: number };

export type ScanSessionDetail = {
  scan_id: string;
  video: string;
  status: ScanSessionStatus;
  expected_count?: number | null;
  processing_mode: string;
  review_ready: boolean;
  selected_identity_id?: number | null;
  selected_anchor_x?: number | null;
  selected_anchor_y?: number | null;
  validated_threshold?: number | null;
  last_blockers: string[];
  candidates: IdentityCandidate[];
  duplicates: DuplicatePair[];
  excluded_identity_ids: number[];
  accepted_low_confidence_ids: number[];
  resolved_duplicates: ReviewDuplicateResolution[];
  pending_split_ids: number[];
  updated_at_ms: number;
  event_count: number;
  recent_events: ScanSessionEvent[];
};

export type ScanSessionSummary = {
  scan_id: string;
  video: string;
  status: ScanSessionStatus;
  review_ready: boolean;
  selected_identity_id?: number | null;
  pending_split_count: number;
  event_count: number;
  updated_at_ms: number;
};

export type QueryIdentityScansResult = {
  rows: ScanSessionSummary[];
  next_cursor_updated_at_ms?: number | null;
  next_cursor_scan_id?: string | null;
  offset_ignored: boolean;
};

export type QueryScanEventsResult = {
  rows: ScanSessionEvent[];
  next_cursor_event_id?: number | null;
  offset_ignored: boolean;
};

export type ScanStorageStats = {
  schema_version: number;
  session_count: number;
  event_count: number;
  db_path: string;
};

export type ScanStorageMaintenanceResult = {
  deleted_sessions: number;
  deleted_events: number;
  vacuum_ran: boolean;
  stats: ScanStorageStats;
};

export type ExportDiagnosticsResult = {
  path: string;
  bytes: number;
};

export type DiagnosticsBundleInfo = {
  file_name: string;
  path: string;
  bytes: number;
  modified_at_ms?: number | null;
};

export type ListDiagnosticsBundlesResult = {
  bundles: DiagnosticsBundleInfo[];
};

export type StorageWorkerStatus = {
  running: boolean;
  stop_requested: boolean;
  poll_interval_ms: number;
  max_session_age_ms: number;
  max_events_per_scan: number;
  vacuum: boolean;
  runs_total: number;
  last_run_ms?: number | null;
  last_error?: string | null;
};

export type ScanProgressPayload = {
  run_id: string;
  sampled_frames: number;
  total_decoded_frames: number;
  estimated_total_samples: number;
  pass_fraction: number;
  overall_fraction: number;
  phase: string;
  pass_index: number;
  pass_total: number;
};

export type ScanDonePayload = {
  run_id: string;
  ok: boolean;
  message: string;
};

export type RenderProgressPayload = {
  run_id: string;
  current: number;
  total: number;
  fraction: number;
};

export type RenderDonePayload = {
  run_id?: string | null;
  ok: boolean;
  message: string;
  output_path?: string | null;
};
