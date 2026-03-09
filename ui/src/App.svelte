<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { open, save } from '@tauri-apps/plugin-dialog';
  import { onMount, onDestroy } from 'svelte';

  // ── State ─────────────────────────────────────────────────────────────────

  type JobStatus = 'idle' | 'running' | 'cancelling' | 'done' | 'error';
  type ScanStatus = 'idle' | 'running' | 'cancelling' | 'done' | 'error';

  type IdentityCandidate = {
    id: number;
    confidence: number;
    observations: number;
    first_frame: number;
    last_frame: number;
    anchor_x: number;
    anchor_y: number;
    thumbnail_data_url: string;
  };

  type DuplicatePair = {
    a: number;
    b: number;
    similarity: number;
  };

  type ScanResult = {
    scan_id: string;
    ok: boolean;
    message: string;
    video: string;
    sampled_frames: number;
    total_decoded_frames: number;
    proposed_count: number;
    expected_count?: number;
    rescan_performed: boolean;
    needs_review: boolean;
    candidates: IdentityCandidate[];
    duplicates: DuplicatePair[];
  };

  type IdentityReviewResult = {
    ok: boolean;
    ready: boolean;
    blockers: string[];
    active_count: number;
    selected_identity_id?: number;
    selected_anchor_x?: number;
    selected_anchor_y?: number;
  };

  type QueueHealth = {
    sqs_enabled: boolean;
    queues: {
      discovery_queue: string;
      rescan_queue: string;
      tracking_start_queue: string;
      tracking_monitor_queue: string;
      dlq_queue: string;
    };
    depths: {
      discovery: number;
      rescan: number;
      tracking_start: number;
      tracking_monitor: number;
      dlq: number;
    };
    dedupe_keys: number;
  };

  type QueueActionResult = {
    accepted?: boolean;
    deduplicated?: boolean;
    queue: string;
    message_id?: string;
    job_id?: string;
    moved_to_dlq?: boolean;
    requeued?: boolean;
    attempt?: number;
    error?: string;
    depth?: number;
    remaining_depth?: number;
    processed?: boolean;
  };

  type QueueWorkerStatus = {
    running: boolean;
    stop_requested: boolean;
    poll_interval_ms: number;
    max_attempts_before_dlq: number;
    processed_total: number;
    last_error?: string;
    recent_events: QueueWorkerEvent[];
  };

  type QueueWorkerEvent = {
    at_ms: number;
    queue: string;
    message_id?: string;
    job_id?: string;
    attempt?: number;
    moved_to_dlq: boolean;
    requeued: boolean;
    error?: string;
  };

  type ScanSessionEvent = {
    at_ms: number;
    action: string;
    details: string;
  };

  type ScanSessionDetail = {
    scan_id: string;
    video: string;
    status: 'proposed' | 'validated' | 'tracking' | 'completed' | 'failed';
    expected_count?: number;
    review_ready: boolean;
    selected_identity_id?: number;
    selected_anchor_x?: number;
    selected_anchor_y?: number;
    validated_threshold?: number;
    last_blockers: string[];
    candidates: IdentityCandidate[];
    duplicates: DuplicatePair[];
    excluded_identity_ids: number[];
    accepted_low_confidence_ids: number[];
    resolved_duplicate_keys: [number, number][];
    pending_split_ids: number[];
    updated_at_ms: number;
    event_count: number;
    recent_events: ScanSessionEvent[];
  };

  type ScanSessionSummary = {
    scan_id: string;
    video: string;
    status: 'proposed' | 'validated' | 'tracking' | 'completed' | 'failed';
    review_ready: boolean;
    selected_identity_id?: number;
    pending_split_count: number;
    event_count: number;
    updated_at_ms: number;
  };

  type QueryIdentityScansResult = {
    rows: ScanSessionSummary[];
    next_cursor_updated_at_ms?: number;
    next_cursor_scan_id?: string;
    offset_ignored?: boolean;
  };

  type QueryScanEventsResult = {
    rows: ScanSessionEvent[];
    next_cursor_event_id?: number;
    offset_ignored?: boolean;
  };

  type ScanStorageStats = {
    schema_version: number;
    session_count: number;
    event_count: number;
    db_path: string;
  };

  type ScanStorageMaintenanceResult = {
    deleted_sessions: number;
    deleted_events: number;
    vacuum_ran: boolean;
    stats: ScanStorageStats;
  };

  type ExportDiagnosticsResult = {
    path: string;
    bytes: number;
  };

  type DiagnosticsBundleInfo = {
    file_name: string;
    path: string;
    bytes: number;
    modified_at_ms?: number;
    sha256?: string;
  };

  type ListDiagnosticsBundlesResult = {
    bundles: DiagnosticsBundleInfo[];
  };

  type PruneDiagnosticsBundlesResult = {
    deleted: number;
    kept: number;
  };

  type ReadDiagnosticsBundleResult = {
    path: string;
    bytes: number;
    content: string;
    truncated: boolean;
  };

  type DeleteDiagnosticsBundleResult = {
    deleted: boolean;
  };

  type VerifyDiagnosticsBundleResult = {
    path: string;
    expected_sha256?: string;
    actual_sha256: string;
    matches: boolean;
  };

  type StorageWorkerStatus = {
    running: boolean;
    stop_requested: boolean;
    poll_interval_ms: number;
    max_session_age_ms: number;
    max_events_per_scan: number;
    vacuum: boolean;
    runs_total: number;
    last_run_ms?: number;
    last_error?: string;
  };

  type ScanProgressPayload = {
    sampled_frames: number;
    total_decoded_frames: number;
  };

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

  let scanStatus: ScanStatus = $state('idle');
  let scanMessage = $state('');
  let scanErr = $state('');
  let expectedMembersInput = $state('');
  let scanCandidates = $state<IdentityCandidate[]>([]);
  let duplicatePairs = $state<DuplicatePair[]>([]);
  let selectedIdentityId = $state<number | null>(null);
  let selectedAnchorX = $state<number | null>(null);
  let selectedAnchorY = $state<number | null>(null);
  let ignoredIdentityIds = $state<number[]>([]);
  let acceptedLowConfidenceIds = $state<number[]>([]);
  let resolvedDuplicateKeys = $state<string[]>([]);
  let resolvedDuplicates = $state<{ a: number; b: number; keep: number }[]>([]);
  let pendingSplitIds = $state<number[]>([]);
  let scanId = $state('');
  let reviewReady = $state(false);
  let reviewBlockers = $state<string[]>([]);
  let reviewReqSeq = $state(0);
  let scanNeedsReview = $state(false);
  let rescanPerformed = $state(false);
  let scanSampledFrames = $state(0);
  let scanDecodedFrames = $state(0);
  let scanProgressFraction = $state(0);
  let reviewDebounceTimer: ReturnType<typeof setTimeout> | null = null;

  let queueHealth = $state<QueueHealth | null>(null);
  let queueMsg = $state('');
  let queueErr = $state('');
  let queueAttempts = $state<number[]>([]);
  let workerStatus = $state<QueueWorkerStatus | null>(null);
  let workerPollMsInput = $state('1200');
  let telemetryInterval: ReturnType<typeof setInterval> | null = null;
  let telemetryRefreshing = $state(false);
  let telemetryTick = $state(0);
  let workerEventFilter = $state<'all' | 'issues'>('all');
  let scanSessionDetail = $state<ScanSessionDetail | null>(null);
  let scanSessions = $state<ScanSessionSummary[]>([]);
  let scanEventsPage = $state<ScanSessionEvent[]>([]);
  let scanSessionsCursorUpdatedAt = $state<number | null>(null);
  let scanSessionsCursorId = $state<string | null>(null);
  let scanEventsCursorId = $state<number | null>(null);
  let scanStorageStats = $state<ScanStorageStats | null>(null);
  let scanSessionStatusFilter = $state<'all' | 'proposed' | 'validated' | 'tracking' | 'completed' | 'failed'>('all');
  let scanEventActionFilter = $state('');
  let storageWorkerStatus = $state<StorageWorkerStatus | null>(null);
  let storageWorkerPollInput = $state('300000');
  let diagnosticsBundles = $state<DiagnosticsBundleInfo[]>([]);
  let diagnosticsKeepInput = $state('20');
  let scanEventWindowMinutesInput = $state('');
  let diagnosticsPreview = $state('');
  let diagnosticsPreviewPath = $state('');
  let diagnosticsVerifyState = $state<Record<string, 'ok' | 'mismatch' | 'untracked'>>({});
  let diagnosticsVerifyDetails = $state<Record<string, string>>({});

  let showSettings = $state(false);
  let videoPreviewSrc = $state('');
  let biasPreviewSrc = $state('');
  let videoPreviewWidth = $state(138);
  let videoPreviewHeight = $state(78);
  let biasPreviewWidth = $state(96);
  let biasPreviewHeight = $state(62);
  let videoPreviewRatio = $state(16 / 9);
  let biasPreviewRatio = $state(1);
  let startedAtMs = $state<number | null>(null);
  let etaSeconds = $state<number | null>(null);

  $effect(() => {
    videoPreviewWidth = 138;
    videoPreviewHeight = 78;
    videoPreviewRatio = 16 / 9;
    videoPreviewSrc = '';
    if (videoPath) {
      clearIdentityScan();
      invoke<string>('read_thumbnail', { path: videoPath })
        .then((src) => { videoPreviewSrc = src; })
        .catch(() => { videoPreviewSrc = ''; });
    }
  });

  $effect(() => {
    biasPreviewWidth = 96;
    biasPreviewHeight = 62;
    biasPreviewRatio = 1;
    biasPreviewSrc = '';
    if (biasPath) {
      invoke<string>('read_thumbnail', { path: biasPath })
        .then((src) => { biasPreviewSrc = src; })
        .catch(() => { biasPreviewSrc = ''; });
    }
  });

  $effect(() => {
    yoloModel;
    faceModel;
    clearIdentityScan();
  });

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

    refreshQueueHealth();
    refreshWorkerStatus();
    await attachListeners();
  });

  // ── Event listeners ───────────────────────────────────────────────────────

  let unlistenProgress: (() => void) | null = null;
  let unlistenDone:     (() => void) | null = null;
  let unlistenScanProgress: (() => void) | null = null;
  let unlistenScanDone: (() => void) | null = null;

  async function attachListeners() {
    if (unlistenProgress || unlistenDone || unlistenScanProgress || unlistenScanDone) {
      return;
    }
    unlistenProgress = await listen<{ current: number; total: number; fraction: number }>(
      'fancam://progress',
      (e) => {
        curFrame  = e.payload.current;
        totFrames = e.payload.total;
        progress  = e.payload.fraction;
        updateEta(e.payload.current, e.payload.total, e.payload.fraction);
      }
    );
    unlistenDone = await listen<{ ok: boolean; message: string; output_path?: string }>(
      'fancam://done',
      (e) => {
        if (e.payload.ok) {
          status     = 'done';
          resultPath = e.payload.output_path ?? '';
          progress   = 1;
          etaSeconds = 0;
        } else {
          status = 'error';
          errMsg = e.payload.message;
          etaSeconds = null;
        }
      }
    );
    unlistenScanProgress = await listen<ScanProgressPayload>('scan://progress', (e) => {
      scanSampledFrames = e.payload.sampled_frames;
      scanDecodedFrames = e.payload.total_decoded_frames;
      const denom = Math.max(scanDecodedFrames, scanSampledFrames, 1);
      scanProgressFraction = Math.min(1, scanSampledFrames / denom);
    });
    unlistenScanDone = await listen<{ ok: boolean; message: string }>('scan://done', (e) => {
      if (e.payload.ok) {
        if (scanStatus !== 'done') {
          scanStatus = 'done';
        }
      } else {
        scanStatus = 'error';
        scanErr = e.payload.message;
      }
    });
  }

  function detachListeners() {
    unlistenProgress?.();
    unlistenDone?.();
    unlistenScanProgress?.();
    unlistenScanDone?.();
    unlistenProgress = null;
    unlistenDone     = null;
    unlistenScanProgress = null;
    unlistenScanDone = null;
  }

  onDestroy(detachListeners);

  onDestroy(() => {
    if (reviewDebounceTimer !== null) {
      clearTimeout(reviewDebounceTimer);
      reviewDebounceTimer = null;
    }
    stopTelemetryLoop();
  });

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

  function clearIdentityScan() {
    scanStatus = 'idle';
    scanMessage = '';
    scanErr = '';
    scanCandidates = [];
    duplicatePairs = [];
    selectedIdentityId = null;
    selectedAnchorX = null;
    selectedAnchorY = null;
    ignoredIdentityIds = [];
    acceptedLowConfidenceIds = [];
    resolvedDuplicateKeys = [];
    resolvedDuplicates = [];
    pendingSplitIds = [];
    scanId = '';
    reviewReady = false;
    reviewBlockers = [];
    scanEventsPage = [];
    scanEventsCursorId = null;
    scanSessionsCursorUpdatedAt = null;
    scanSessionsCursorId = null;
    scanNeedsReview = false;
    rescanPerformed = false;
    scanSampledFrames = 0;
    scanDecodedFrames = 0;
    scanProgressFraction = 0;
  }

  function duplicateKey(a: number, b: number) {
    const left = Math.min(a, b);
    const right = Math.max(a, b);
    return `${left}:${right}`;
  }

  function isIgnored(id: number) {
    return ignoredIdentityIds.includes(id);
  }

  function activeCandidates() {
    return scanCandidates.filter((c) => !isIgnored(c.id));
  }

  function unresolvedLowConfidenceCandidates() {
    return activeCandidates().filter(
      (c) => c.confidence < 0.55 && !acceptedLowConfidenceIds.includes(c.id)
    );
  }

  function unresolvedDuplicatePairs() {
    return duplicatePairs.filter((p) => {
      if (isIgnored(p.a) || isIgnored(p.b)) return false;
      return !resolvedDuplicateKeys.includes(duplicateKey(p.a, p.b));
    });
  }

  function countMismatchExists() {
    const expected = expectedMembersValue();
    if (expected === null) return false;
    return activeCandidates().length !== expected;
  }

  function reviewReasons() {
    const reasons: string[] = [];
    if (countMismatchExists()) reasons.push('member count mismatch');
    if (unresolvedDuplicatePairs().length > 0) reasons.push('duplicate identities unresolved');
    if (unresolvedLowConfidenceCandidates().length > 0) reasons.push('low-confidence identities unreviewed');
    if (pendingSplitIds.length > 0) reasons.push('split requests pending');
    return reasons;
  }

  function canRenderNow() {
    return (
      !!videoPath &&
      !!biasPath &&
      !!outputPath &&
      scanStatus === 'done' &&
      reviewReady &&
      selectedIdentityId !== null
    );
  }

  function modelSetupReady() {
    return !!yoloModel && !!faceModel;
  }

  function expectedMembersValue(): number | null {
    const trimmed = expectedMembersInput.trim();
    if (!trimmed) return null;
    const n = Number.parseInt(trimmed, 10);
    if (!Number.isFinite(n) || n <= 0) return null;
    return n;
  }

  function expectedMembersInvalid() {
    const trimmed = expectedMembersInput.trim();
    if (!trimmed) return false;
    const n = Number.parseInt(trimmed, 10);
    return !Number.isFinite(n) || n <= 0;
  }

  function selectIdentity(candidate: IdentityCandidate) {
    if (isIgnored(candidate.id)) return;
    selectedIdentityId = candidate.id;
    selectedAnchorX = candidate.anchor_x;
    selectedAnchorY = candidate.anchor_y;
  }

  function toggleIgnoreIdentity(candidate: IdentityCandidate) {
    const id = candidate.id;
    if (isIgnored(id)) {
      ignoredIdentityIds = ignoredIdentityIds.filter((x) => x !== id);
      return;
    }
    ignoredIdentityIds = [...ignoredIdentityIds, id];
    acceptedLowConfidenceIds = acceptedLowConfidenceIds.filter((x) => x !== id);
    if (selectedIdentityId === id) {
      selectedIdentityId = null;
      selectedAnchorX = null;
      selectedAnchorY = null;
    }
  }

  function toggleAcceptLowConfidence(candidate: IdentityCandidate) {
    const id = candidate.id;
    if (acceptedLowConfidenceIds.includes(id)) {
      acceptedLowConfidenceIds = acceptedLowConfidenceIds.filter((x) => x !== id);
      return;
    }
    acceptedLowConfidenceIds = [...acceptedLowConfidenceIds, id];
  }

  function toggleSplitRequest(candidate: IdentityCandidate) {
    const id = candidate.id;
    if (pendingSplitIds.includes(id)) {
      pendingSplitIds = pendingSplitIds.filter((x) => x !== id);
      return;
    }
    pendingSplitIds = [...pendingSplitIds, id];
  }

  function resolveDuplicate(pair: DuplicatePair, keep: number) {
    const drop = keep === pair.a ? pair.b : pair.a;
    resolvedDuplicateKeys = [...resolvedDuplicateKeys, duplicateKey(pair.a, pair.b)];
    resolvedDuplicates = [
      ...resolvedDuplicates,
      { a: pair.a, b: pair.b, keep }
    ];
    if (!ignoredIdentityIds.includes(drop)) {
      ignoredIdentityIds = [...ignoredIdentityIds, drop];
    }
    if (selectedIdentityId === drop) {
      selectedIdentityId = null;
      selectedAnchorX = null;
      selectedAnchorY = null;
    }
  }

  async function runIdentityScan() {
    if (!videoPath || !yoloModel || !faceModel) return;
    if (expectedMembersInvalid()) {
      scanErr = 'expected members must be a positive whole number';
      scanStatus = 'error';
      return;
    }
    scanStatus = 'running';
    scanErr = '';
    scanMessage = '';
    scanSampledFrames = 0;
    scanDecodedFrames = 0;
    scanProgressFraction = 0;
    scanCandidates = [];
    duplicatePairs = [];
    selectedIdentityId = null;
    selectedAnchorX = null;
    selectedAnchorY = null;
    ignoredIdentityIds = [];
    acceptedLowConfidenceIds = [];
    resolvedDuplicateKeys = [];
    resolvedDuplicates = [];
    pendingSplitIds = [];

    try {
      const result = await invoke<ScanResult>('scan_identities', {
        args: {
          video: videoPath,
          yolo_model: yoloModel,
          face_model: faceModel,
          expected_member_count: expectedMembersValue(),
        },
      });
      if (!result.ok) {
        scanStatus = 'error';
        scanErr = result.message;
        return;
      }
      scanStatus = 'done';
      scanId = result.scan_id;
      scanMessage = result.message;
      scanCandidates = result.candidates;
      duplicatePairs = result.duplicates;
      scanNeedsReview = result.needs_review;
      rescanPerformed = result.rescan_performed;
      scanSampledFrames = result.sampled_frames;
      scanDecodedFrames = result.total_decoded_frames;
      scanProgressFraction = 1;

      if (result.candidates.length > 0) {
        const best = result.candidates[0];
        selectIdentity(best);
      }
    } catch (e: unknown) {
      scanStatus = 'error';
      scanErr = String(e);
    }
  }

  async function cancelScan() {
    if (scanStatus !== 'running') return;
    scanStatus = 'cancelling';
    try {
      await invoke('cancel_scan');
      scanMessage = 'scan cancellation requested';
    } catch (e: unknown) {
      scanStatus = 'error';
      scanErr = String(e);
    }
  }

  async function refreshQueueHealth() {
    if (telemetryRefreshing) return;
    telemetryRefreshing = true;
    try {
      queueHealth = await invoke<QueueHealth>('queue_health');
      queueErr = '';
    } catch (e: unknown) {
      queueErr = String(e);
    } finally {
      telemetryRefreshing = false;
    }
  }

  async function peekQueueAttempts() {
    queueErr = '';
    try {
      const result = await invoke<{ attempts: number[] }>('queue_peek_discovery_attempts', {
        args: { limit: 8 },
      });
      queueAttempts = result.attempts;
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function enqueueCurrentScanJob() {
    if (!scanId || !videoPath || !yoloModel || !faceModel) return;
    queueMsg = '';
    queueErr = '';
    try {
      const result = await invoke<QueueActionResult>('enqueue_discovery_job', {
        args: {
          scan_id: scanId,
          video: videoPath,
          yolo_model: yoloModel,
          face_model: faceModel,
          expected_member_count: expectedMembersValue(),
        },
      });
      queueMsg = result.deduplicated
        ? 'discovery job deduplicated'
        : 'discovery job queued';
      await refreshQueueHealth();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function enqueueSplitRescanJob() {
    if (!scanId) return;
    queueMsg = '';
    queueErr = '';
    try {
      const result = await invoke<QueueActionResult>('enqueue_split_rescan_job', {
        args: {
          scan_id: scanId,
        },
      });
      queueMsg = result.deduplicated ? 'split rescan deduplicated' : 'split rescan queued';
      await refreshQueueHealth();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function refreshWorkerStatus() {
    try {
      workerStatus = await invoke<QueueWorkerStatus>('queue_worker_status');
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function refreshQueueTelemetry() {
    if (!showSettings) return;
    telemetryTick += 1;
    const calls: Promise<unknown>[] = [
      refreshQueueHealth(),
      refreshWorkerStatus(),
    ];
    if (telemetryTick % 2 === 0) {
      calls.push(peekQueueAttempts(), refreshScanSessionDetail());
    }
    if (telemetryTick % 3 === 0) {
      calls.push(refreshScanStorageStats(), refreshStorageWorkerStatus());
    }
    if (telemetryTick % 6 === 0) {
      calls.push(refreshDiagnosticsBundles(), refreshScanSessionsList());
    }
    await Promise.allSettled(calls);
  }

  async function refreshScanSessionDetail() {
    if (!scanId) {
      scanSessionDetail = null;
      return;
    }
    try {
      scanSessionDetail = await invoke<ScanSessionDetail>('get_identity_scan', { scan_id: scanId });
    } catch {
      scanSessionDetail = null;
    }
  }

  async function refreshScanSessionsList() {
    try {
      const result = await invoke<QueryIdentityScansResult>('query_identity_scans', {
        args: {
          limit: 20,
          status: scanSessionStatusFilter === 'all' ? null : scanSessionStatusFilter,
          cursor_updated_at_ms: null,
          cursor_scan_id: null,
        },
      });
      scanSessions = result.rows;
      scanSessionsCursorUpdatedAt = result.next_cursor_updated_at_ms ?? null;
      scanSessionsCursorId = result.next_cursor_scan_id ?? null;
    } catch {
      scanSessions = [];
      scanSessionsCursorUpdatedAt = null;
      scanSessionsCursorId = null;
    }
  }

  async function loadMoreScanSessions() {
    if (scanSessionsCursorUpdatedAt === null || !scanSessionsCursorId) return;
    try {
      const result = await invoke<QueryIdentityScansResult>('query_identity_scans', {
        args: {
          limit: 20,
          status: scanSessionStatusFilter === 'all' ? null : scanSessionStatusFilter,
          cursor_updated_at_ms: scanSessionsCursorUpdatedAt,
          cursor_scan_id: scanSessionsCursorId,
        },
      });
      if (result.rows.length === 0) {
        scanSessionsCursorUpdatedAt = null;
        scanSessionsCursorId = null;
        return;
      }
      scanSessions = [...scanSessions, ...result.rows];
      scanSessionsCursorUpdatedAt = result.next_cursor_updated_at_ms ?? null;
      scanSessionsCursorId = result.next_cursor_scan_id ?? null;
    } catch {
      // no-op
    }
  }

  async function refreshScanStorageStats() {
    try {
      scanStorageStats = await invoke<ScanStorageStats>('scan_storage_stats');
    } catch {
      scanStorageStats = null;
    }
  }

  async function refreshStorageWorkerStatus() {
    try {
      storageWorkerStatus = await invoke<StorageWorkerStatus>('storage_worker_status');
    } catch {
      storageWorkerStatus = null;
    }
  }

  async function refreshDiagnosticsBundles() {
    try {
      const result = await invoke<ListDiagnosticsBundlesResult>('list_diagnostics_bundles', {
        args: { limit: 30 },
      });
      diagnosticsBundles = result.bundles;
      const next: Record<string, 'ok' | 'mismatch' | 'untracked'> = {};
      const details: Record<string, string> = {};
      for (const bundle of result.bundles) {
        const existing = diagnosticsVerifyState[bundle.path];
        const existingDetails = diagnosticsVerifyDetails[bundle.path];
        if (existing) next[bundle.path] = existing;
        if (existingDetails) details[bundle.path] = existingDetails;
      }
      diagnosticsVerifyState = next;
      diagnosticsVerifyDetails = details;
    } catch {
      diagnosticsBundles = [];
      diagnosticsVerifyState = {};
      diagnosticsVerifyDetails = {};
    }
  }

  async function verifyDiagnosticsBundle(path: string) {
    queueErr = '';
    try {
      const result = await invoke<VerifyDiagnosticsBundleResult>('verify_diagnostics_bundle', {
        args: { path },
      });
      if (!result.expected_sha256) {
        diagnosticsVerifyState = { ...diagnosticsVerifyState, [path]: 'untracked' };
        diagnosticsVerifyDetails = { ...diagnosticsVerifyDetails, [path]: 'manifest missing' };
        queueMsg = 'bundle is not tracked in manifest';
        return;
      }
      if (result.matches) {
        diagnosticsVerifyState = { ...diagnosticsVerifyState, [path]: 'ok' };
        diagnosticsVerifyDetails = {
          ...diagnosticsVerifyDetails,
          [path]: `sha ${result.actual_sha256.slice(0, 12)}`,
        };
        queueMsg = 'bundle checksum verified';
      } else {
        diagnosticsVerifyState = { ...diagnosticsVerifyState, [path]: 'mismatch' };
        diagnosticsVerifyDetails = {
          ...diagnosticsVerifyDetails,
          [path]: `expected ${result.expected_sha256.slice(0, 8)} actual ${result.actual_sha256.slice(0, 8)}`,
        };
        queueMsg = 'bundle checksum mismatch';
      }
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function pruneDiagnosticsBundles() {
    if (!confirm('Prune diagnostics bundles? Older bundles will be deleted.')) {
      return;
    }
    queueErr = '';
    try {
      const keep = Number.parseInt(diagnosticsKeepInput, 10);
      const result = await invoke<PruneDiagnosticsBundlesResult>('prune_diagnostics_bundles', {
        args: { keep_latest: Number.isFinite(keep) && keep > 0 ? keep : 20 },
      });
      queueMsg = `diagnostics pruned: deleted ${result.deleted}, kept ${result.kept}`;
      await refreshDiagnosticsBundles();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function previewDiagnosticsBundle(path: string) {
    queueErr = '';
    try {
      const result = await invoke<ReadDiagnosticsBundleResult>('read_diagnostics_bundle', {
        args: {
          path,
          max_bytes: 32 * 1024,
        },
      });
      diagnosticsPreview = result.truncated
        ? `${result.content}\n\n...truncated (${result.bytes} bytes total)`
        : result.content;
      diagnosticsPreviewPath = result.path;
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function deleteDiagnosticsBundle(path: string) {
    if (!confirm(`Delete diagnostics bundle?\n${path}`)) {
      return;
    }
    queueErr = '';
    try {
      const result = await invoke<DeleteDiagnosticsBundleResult>('delete_diagnostics_bundle', {
        args: { path },
      });
      if (result.deleted) {
        queueMsg = 'diagnostics bundle deleted';
        if (diagnosticsPreviewPath === path) {
          diagnosticsPreview = '';
          diagnosticsPreviewPath = '';
        }
      } else {
        queueMsg = 'diagnostics bundle not found';
      }
      await refreshDiagnosticsBundles();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function runStorageMaintenance() {
    if (!confirm('Run storage maintenance now? This may delete old scan sessions and events.')) {
      return;
    }
    queueErr = '';
    try {
      const result = await invoke<ScanStorageMaintenanceResult>('run_scan_storage_maintenance', {
        args: {
          max_session_age_ms: 7 * 86_400_000,
          max_events_per_scan: 120,
          vacuum: false,
        },
      });
      scanStorageStats = result.stats;
      queueMsg = `maintenance done: removed ${result.deleted_sessions} sessions, ${result.deleted_events} events`;
      await refreshScanSessionsList();
      await refreshScanSessionDetail();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function startStorageWorker() {
    queueErr = '';
    try {
      const poll = Number.parseInt(storageWorkerPollInput, 10);
      storageWorkerStatus = await invoke<StorageWorkerStatus>('storage_worker_start', {
        args: {
          poll_interval_ms: Number.isFinite(poll) && poll > 0 ? poll : 300000,
          max_session_age_ms: 7 * 86_400_000,
          max_events_per_scan: 120,
          vacuum: false,
        },
      });
      queueMsg = 'storage worker started';
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function stopStorageWorker() {
    queueErr = '';
    try {
      storageWorkerStatus = await invoke<StorageWorkerStatus>('storage_worker_stop');
      queueMsg = 'storage worker stop requested';
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function exportDiagnosticsBundle() {
    queueErr = '';
    try {
      const result = await invoke<ExportDiagnosticsResult>('export_diagnostics_bundle', {
        args: {
          scan_id: scanId || null,
        },
      });
      queueMsg = `diagnostics written: ${result.path} (${result.bytes} bytes)`;
      await refreshDiagnosticsBundles();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function refreshScanEventsFirstPage() {
    if (!scanId) {
      scanEventsPage = [];
      return;
    }
    const mins = Number.parseInt(scanEventWindowMinutesInput, 10);
    const sinceMs = Number.isFinite(mins) && mins > 0 ? Date.now() - mins * 60 * 1000 : null;
    try {
      const result = await invoke<QueryScanEventsResult>('query_scan_events', {
        args: {
          scan_id: scanId,
          limit: 25,
          action_contains: scanEventActionFilter.trim() ? scanEventActionFilter.trim() : null,
          since_ms: sinceMs,
          cursor_event_id: null,
        },
      });
      scanEventsPage = result.rows;
      scanEventsCursorId = result.next_cursor_event_id ?? null;
    } catch {
      scanEventsPage = [];
      scanEventsCursorId = null;
    }
  }

  async function loadMoreScanEvents() {
    if (!scanId || scanEventsCursorId === null) return;
    const mins = Number.parseInt(scanEventWindowMinutesInput, 10);
    const sinceMs = Number.isFinite(mins) && mins > 0 ? Date.now() - mins * 60 * 1000 : null;
    try {
      const result = await invoke<QueryScanEventsResult>('query_scan_events', {
        args: {
          scan_id: scanId,
          limit: 25,
          action_contains: scanEventActionFilter.trim() ? scanEventActionFilter.trim() : null,
          since_ms: sinceMs,
          cursor_event_id: scanEventsCursorId,
        },
      });
      if (result.rows.length === 0) return;
      scanEventsPage = [...scanEventsPage, ...result.rows];
      scanEventsCursorId = result.next_cursor_event_id ?? null;
    } catch {
      // no-op
    }
  }

  async function loadScanSession(scanIdToLoad: string) {
    queueErr = '';
    try {
      const detail = await invoke<ScanSessionDetail>('get_identity_scan', { scan_id: scanIdToLoad });
      scanId = detail.scan_id;
      scanStatus = 'done';
      scanMessage = `loaded saved scan (${detail.status})`;
      scanErr = '';
      expectedMembersInput = detail.expected_count !== undefined && detail.expected_count !== null
        ? String(detail.expected_count)
        : '';
      scanCandidates = detail.candidates;
      duplicatePairs = detail.duplicates;
      ignoredIdentityIds = detail.excluded_identity_ids;
      acceptedLowConfidenceIds = detail.accepted_low_confidence_ids;
      resolvedDuplicateKeys = detail.resolved_duplicate_keys.map(([a, b]) => duplicateKey(a, b));
      resolvedDuplicates = detail.resolved_duplicate_keys.map(([a, b]) => ({ a, b, keep: a }));
      pendingSplitIds = detail.pending_split_ids;
      selectedIdentityId = detail.selected_identity_id ?? null;
      selectedAnchorX = detail.selected_anchor_x ?? null;
      selectedAnchorY = detail.selected_anchor_y ?? null;
      if (detail.validated_threshold !== undefined && detail.validated_threshold !== null) {
        threshold = detail.validated_threshold;
      }
      reviewReady = detail.review_ready;
      reviewBlockers = detail.last_blockers;
      scanSessionDetail = detail;
      await refreshScanEventsFirstPage();
      queueMsg = `loaded ${detail.scan_id}`;
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function cleanupOldScanSessions() {
    if (!confirm('Cleanup old scan sessions now?')) {
      return;
    }
    queueErr = '';
    try {
      const removed = await invoke<number>('cleanup_identity_scans', {
        max_age_ms: 86_400_000,
      });
      queueMsg = `cleaned ${removed} old scan sessions`;
      await refreshScanSessionDetail();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  function startTelemetryLoop() {
    if (telemetryInterval !== null) return;
    telemetryInterval = setInterval(() => {
      refreshQueueTelemetry();
    }, status === 'running' ? 4500 : 2500);
  }

  function stopTelemetryLoop() {
    if (telemetryInterval === null) return;
    clearInterval(telemetryInterval);
    telemetryInterval = null;
  }

  async function startWorker() {
    queueErr = '';
    const poll = Number.parseInt(workerPollMsInput, 10);
    try {
      workerStatus = await invoke<QueueWorkerStatus>('queue_worker_start', {
        args: {
          poll_interval_ms: Number.isFinite(poll) && poll > 0 ? poll : 1200,
          max_attempts_before_dlq: 3,
        },
      });
      queueMsg = 'worker started';
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function stopWorker() {
    queueErr = '';
    try {
      workerStatus = await invoke<QueueWorkerStatus>('queue_worker_stop');
      queueMsg = 'worker stop requested';
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function clearWorkerEvents() {
    queueErr = '';
    try {
      workerStatus = await invoke<QueueWorkerStatus>('queue_worker_clear_events');
      queueMsg = 'worker events cleared';
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  function filteredWorkerEvents() {
    const events = workerStatus?.recent_events ?? [];
    if (workerEventFilter === 'all') return events;
    return events.filter((event) => event.requeued || event.moved_to_dlq || !!event.error);
  }

  async function copyQueueSnapshot() {
    queueErr = '';
    const snapshot = {
      at_ms: Date.now(),
      scan_id: scanId,
      queue_health: queueHealth,
      worker_status: workerStatus,
      storage_worker_status: storageWorkerStatus,
      storage_stats: scanStorageStats,
      queue_attempts: queueAttempts,
      diagnostics_bundles: diagnosticsBundles,
    };
    const text = JSON.stringify(snapshot, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      queueMsg = 'diagnostic snapshot copied';
    } catch {
      queueErr = 'clipboard unavailable; open devtools and copy snapshot from console';
      console.log('focus-lock queue snapshot', snapshot);
    }
  }

  async function processNextQueuedScan() {
    queueMsg = '';
    queueErr = '';
    try {
      const result = await invoke<QueueActionResult>('process_next_discovery_job', {
        args: { max_attempts_before_dlq: 3 },
      });
      if (!result.processed) {
        queueMsg = 'no queued discovery jobs';
      } else if (result.error) {
        if (result.moved_to_dlq) {
          queueErr = `failed and moved to dlq (attempt ${result.attempt ?? 0})`;
        } else if (result.requeued) {
          queueErr = `failed and requeued (attempt ${result.attempt ?? 0})`;
        } else {
          queueErr = result.error;
        }
      } else {
        queueMsg = `processed ${result.message_id ?? 'message'}`;
      }
      await refreshQueueHealth();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function processNextQueuedRescan() {
    queueMsg = '';
    queueErr = '';
    try {
      const result = await invoke<QueueActionResult>('process_next_rescan_job', {
        args: { max_attempts_before_dlq: 3 },
      });
      if (!result.processed) {
        queueMsg = 'no queued rescan jobs';
      } else if (result.error) {
        if (result.moved_to_dlq) {
          queueErr = `rescan failed and moved to dlq (attempt ${result.attempt ?? 0})`;
        } else if (result.requeued) {
          queueErr = `rescan failed and requeued (attempt ${result.attempt ?? 0})`;
        } else {
          queueErr = result.error;
        }
      } else {
        queueMsg = `processed rescan ${result.message_id ?? 'message'}`;
      }
      await refreshQueueTelemetry();
    } catch (e: unknown) {
      queueErr = String(e);
    }
  }

  async function startJob() {
    if (!canRenderNow()) return;
    status   = 'running';
    progress = 0;
    curFrame = 0;
    etaSeconds = null;
    startedAtMs = Date.now();
    errMsg   = '';
    resultPath = '';

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
              scan_id: scanId,
              selected_identity_id: selectedIdentityId,
              target_anchor_x: selectedAnchorX,
              target_anchor_y: selectedAnchorY,
            },
          }
      );
      if (!result.ok) {
        if (result.message.toLowerCase().includes('cancel')) {
          status = 'idle';
          errMsg = '';
        } else {
          status = 'error';
          errMsg = result.message;
        }
        etaSeconds = null;
      }
    } catch (e: unknown) {
      status = 'error';
      errMsg = String(e);
      etaSeconds = null;
    }
    startedAtMs = null;
  }

  async function cancelJob() {
    if (status !== 'running') return;
    status = 'cancelling';
    try {
      await invoke('cancel_job');
      errMsg = 'cancellation requested...';
    } catch (e: unknown) {
      status = 'error';
      errMsg = String(e);
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function basename(p: string) {
    return p.split(/[\\/]/).pop() ?? p;
  }

  function pct(v: number) {
    return (v * 100).toFixed(1) + '%';
  }

  function fitPreview(sourceW: number, sourceH: number, maxW: number, maxH: number) {
    if (sourceW <= 0 || sourceH <= 0) {
      return { width: maxW, height: maxH, ratio: maxW / maxH };
    }

    const scale = Math.min(maxW / sourceW, maxH / sourceH);
    const width = Math.max(44, Math.round(sourceW * scale));
    const height = Math.max(44, Math.round(sourceH * scale));
    return { width, height, ratio: sourceW / sourceH };
  }

  function handleVideoPreviewLoad(event: Event) {
    const img = event.currentTarget as HTMLImageElement;
    const size = fitPreview(img.naturalWidth, img.naturalHeight, 156, 98);
    videoPreviewWidth = size.width;
    videoPreviewHeight = size.height;
    videoPreviewRatio = size.ratio;
  }

  function handleBiasPreviewLoad(event: Event) {
    const img = event.currentTarget as HTMLImageElement;
    const size = fitPreview(img.naturalWidth, img.naturalHeight, 132, 88);
    biasPreviewWidth = size.width;
    biasPreviewHeight = size.height;
    biasPreviewRatio = size.ratio;
  }

  function updateEta(current: number, total: number, fraction: number) {
    if (status !== 'running' || startedAtMs === null) {
      etaSeconds = null;
      return;
    }

    const elapsed = Math.max((Date.now() - startedAtMs) / 1000, 0.01);
    const ratio =
      fraction > 0
        ? fraction
        : total > 0 && current > 0
          ? current / total
          : 0;

    if (ratio <= 0 || ratio >= 1) {
      etaSeconds = null;
      return;
    }

    etaSeconds = Math.max(0, Math.round((elapsed * (1 - ratio)) / ratio));
  }

  function formatEta(seconds: number | null) {
    if (seconds === null) return 'estimating...';
    if (seconds < 60) return `${seconds}s`;

    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return secs === 0 ? `${mins}m` : `${mins}m ${secs}s`;

    const hours = Math.floor(mins / 60);
    const remMins = mins % 60;
    return remMins === 0 ? `${hours}h` : `${hours}h ${remMins}m`;
  }

  function formatClock(ms: number) {
    const date = new Date(ms);
    const hh = String(date.getHours()).padStart(2, '0');
    const mm = String(date.getMinutes()).padStart(2, '0');
    const ss = String(date.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  }

  $effect(() => {
    scanNeedsReview = scanStatus === 'done' && !reviewReady;
  });

  $effect(() => {
    if (showSettings) {
      refreshQueueTelemetry();
      refreshScanSessionsList();
      startTelemetryLoop();
    } else {
      stopTelemetryLoop();
    }
  });

  $effect(() => {
    scanId;
    if (scanId) {
      refreshScanEventsFirstPage();
    } else {
      scanEventsPage = [];
    }
  });

  $effect(() => {
    scanSessionStatusFilter;
    if (showSettings) {
      refreshScanSessionsList();
    }
  });

  $effect(() => {
    scanEventActionFilter;
    if (showSettings && scanId) {
      refreshScanEventsFirstPage();
    }
  });

  $effect(() => {
    scanEventWindowMinutesInput;
    if (showSettings && scanId) {
      refreshScanEventsFirstPage();
    }
  });

  $effect(() => {
    if (scanStatus !== 'done' || !scanId) {
      if (reviewDebounceTimer !== null) {
        clearTimeout(reviewDebounceTimer);
        reviewDebounceTimer = null;
      }
      reviewReady = false;
      reviewBlockers = [];
      return;
    }

    const reqId = reviewReqSeq + 1;
    reviewReqSeq = reqId;
    const args = {
      scan_id: scanId,
      selected_identity_id: selectedIdentityId,
      threshold,
      excluded_identity_ids: ignoredIdentityIds,
      accepted_low_confidence_ids: acceptedLowConfidenceIds,
      resolved_duplicates: resolvedDuplicates,
      pending_split_ids: pendingSplitIds,
      expected_member_count: expectedMembersValue(),
    };
    if (reviewDebounceTimer !== null) {
      clearTimeout(reviewDebounceTimer);
    }
    reviewDebounceTimer = setTimeout(() => {
      invoke<IdentityReviewResult>('validate_identity_review', { args })
        .then((result) => {
          if (reqId !== reviewReqSeq) return;
          reviewReady = result.ready;
          reviewBlockers = result.blockers;
          if (result.selected_identity_id !== undefined && result.selected_identity_id !== null) {
            selectedIdentityId = result.selected_identity_id;
            selectedAnchorX = result.selected_anchor_x ?? selectedAnchorX;
            selectedAnchorY = result.selected_anchor_y ?? selectedAnchorY;
          }
        })
        .catch((e: unknown) => {
          if (reqId !== reviewReqSeq) return;
          reviewReady = false;
          reviewBlockers = [String(e)];
        });
    }, 220);

    return () => {
      if (reviewDebounceTimer !== null) {
        clearTimeout(reviewDebounceTimer);
        reviewDebounceTimer = null;
      }
    };
  });

  $effect(() => {
    if (selectedIdentityId !== null && isIgnored(selectedIdentityId)) {
      selectedIdentityId = null;
      selectedAnchorX = null;
      selectedAnchorY = null;
    }
  });
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
    >⚙ settings</button>
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

    <div class="queue-panel">
      <h4>queue</h4>
      {#if queueHealth}
        <div class="queue-meta">
          <span>{queueHealth.sqs_enabled ? 'sqs mode enabled' : 'in-memory mode'}</span>
          <span>discovery depth {queueHealth.depths.discovery}</span>
          <span>dlq depth {queueHealth.depths.dlq}</span>
          {#if scanStorageStats}
            <span>db v{scanStorageStats.schema_version}</span>
            <span>sessions {scanStorageStats.session_count}</span>
            <span>events {scanStorageStats.event_count}</span>
          {/if}
        </div>
      {/if}
      <div class="queue-actions">
        <button class="ghost-btn" onclick={refreshQueueHealth}>refresh queue</button>
        <button class="ghost-btn" onclick={refreshScanSessionDetail}>refresh session</button>
        <button class="ghost-btn" onclick={refreshScanSessionsList}>refresh sessions</button>
        <button class="ghost-btn" onclick={cleanupOldScanSessions}>cleanup sessions</button>
        <button class="ghost-btn" onclick={runStorageMaintenance}>storage maintenance</button>
        <button class="ghost-btn" onclick={exportDiagnosticsBundle}>export diagnostics</button>
        <button class="ghost-btn" onclick={refreshDiagnosticsBundles}>refresh bundles</button>
        <input class="count-input" bind:value={diagnosticsKeepInput} placeholder="keep bundles" />
        <button class="ghost-btn" onclick={pruneDiagnosticsBundles}>prune bundles</button>
        <button class="ghost-btn" disabled={!scanId} onclick={enqueueCurrentScanJob}>enqueue discovery</button>
        <button class="ghost-btn" disabled={!scanId || pendingSplitIds.length === 0} onclick={enqueueSplitRescanJob}>enqueue split rescan</button>
        <button class="ghost-btn" onclick={processNextQueuedScan}>process next</button>
        <button class="ghost-btn" onclick={processNextQueuedRescan}>process next rescan</button>
        <button class="ghost-btn" onclick={peekQueueAttempts}>peek attempts</button>
      </div>
      {#if scanSessionDetail}
        <div class="queue-meta">
          <span>scan status {scanSessionDetail.status}</span>
          <span>review {scanSessionDetail.review_ready ? 'ready' : 'blocked'}</span>
          <span>events {scanSessionDetail.event_count}</span>
          <span>pending splits {scanSessionDetail.pending_split_ids.length}</span>
          {#if scanSessionDetail.selected_identity_id !== undefined && scanSessionDetail.selected_identity_id !== null}
            <span>target #{scanSessionDetail.selected_identity_id + 1}</span>
          {/if}
        </div>
        {#if scanSessionDetail.last_blockers.length > 0}
          <div class="scan-warning">
            blockers: {scanSessionDetail.last_blockers.join(' · ')}
          </div>
        {/if}
        <div class="queue-actions">
          <input class="count-input" bind:value={scanEventActionFilter} placeholder="event action filter" />
          <input class="count-input" bind:value={scanEventWindowMinutesInput} placeholder="window mins" />
          <button class="ghost-btn" onclick={refreshScanEventsFirstPage}>load events</button>
          <button class="ghost-btn" onclick={loadMoreScanEvents}>more events</button>
        </div>
        {#if scanEventsPage.length > 0}
          <div class="queue-event-list">
            {#each scanEventsPage as event}
              <div class="queue-event-row">
                <span>{formatClock(event.at_ms)}</span>
                <span>{event.action}</span>
                <span class="event-tag">scan</span>
                <span class="event-id">{event.details}</span>
              </div>
            {/each}
          </div>
        {/if}
      {/if}
      <div class="queue-actions">
        <input class="count-input" bind:value={storageWorkerPollInput} placeholder="storage poll ms" />
        <button class="ghost-btn" onclick={startStorageWorker}>start storage worker</button>
        <button class="ghost-btn" onclick={stopStorageWorker}>stop storage worker</button>
      </div>
      {#if storageWorkerStatus}
        <div class="queue-meta">
          <span>storage worker {storageWorkerStatus.running ? 'running' : 'stopped'}</span>
          <span>poll {storageWorkerStatus.poll_interval_ms}ms</span>
          <span>runs {storageWorkerStatus.runs_total}</span>
          {#if storageWorkerStatus.last_run_ms}
            <span>last {formatClock(storageWorkerStatus.last_run_ms)}</span>
          {/if}
          {#if storageWorkerStatus.last_error}
            <span>error: {storageWorkerStatus.last_error}</span>
          {/if}
        </div>
      {/if}
      {#if diagnosticsBundles.length > 0}
        <div class="queue-session-list">
          {#each diagnosticsBundles.slice(0, 10) as bundle}
            <div class="queue-session-row">
              <span class="session-id">{bundle.file_name}</span>
              <span>{bundle.bytes}b</span>
              <span>{bundle.modified_at_ms ? formatClock(bundle.modified_at_ms) : 'n/a'}</span>
              <span>{bundle.sha256 ? bundle.sha256.slice(0, 12) : 'no-sha'}</span>
              <span>{diagnosticsVerifyState[bundle.path] ?? '-'}</span>
              <span>{diagnosticsVerifyDetails[bundle.path] ?? ''}</span>
              <span class="bundle-path">{bundle.path}</span>
              <div class="bundle-actions">
                <button class="ghost-btn tiny" onclick={() => previewDiagnosticsBundle(bundle.path)}>preview</button>
                <button class="ghost-btn tiny" onclick={() => verifyDiagnosticsBundle(bundle.path)}>verify</button>
                <button class="ghost-btn tiny" onclick={() => deleteDiagnosticsBundle(bundle.path)}>delete</button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
      {#if diagnosticsPreview}
        <div class="queue-event-list">
          <pre class="diag-preview">{diagnosticsPreview}</pre>
        </div>
      {/if}
      {#if scanSessions.length > 0}
        <div class="queue-actions">
          <button
            class="ghost-btn tiny"
            class:active={scanSessionStatusFilter === 'all'}
            onclick={() => { scanSessionStatusFilter = 'all'; }}
          >all</button>
          <button
            class="ghost-btn tiny"
            class:active={scanSessionStatusFilter === 'proposed'}
            onclick={() => { scanSessionStatusFilter = 'proposed'; }}
          >proposed</button>
          <button
            class="ghost-btn tiny"
            class:active={scanSessionStatusFilter === 'validated'}
            onclick={() => { scanSessionStatusFilter = 'validated'; }}
          >validated</button>
          <button
            class="ghost-btn tiny"
            class:active={scanSessionStatusFilter === 'tracking'}
            onclick={() => { scanSessionStatusFilter = 'tracking'; }}
          >tracking</button>
          <button
            class="ghost-btn tiny"
            class:active={scanSessionStatusFilter === 'completed'}
            onclick={() => { scanSessionStatusFilter = 'completed'; }}
          >completed</button>
          <button
            class="ghost-btn tiny"
            class:active={scanSessionStatusFilter === 'failed'}
            onclick={() => { scanSessionStatusFilter = 'failed'; }}
          >failed</button>
        </div>
        <div class="queue-session-list">
          {#each scanSessions as session}
            <button class="queue-session-row" onclick={() => loadScanSession(session.scan_id)}>
              <span class="session-id">{session.scan_id}</span>
              <span>{session.status}</span>
              <span>{session.event_count} ev</span>
              <span>{basename(session.video)}</span>
            </button>
          {/each}
        </div>
        {#if scanSessionsCursorUpdatedAt !== null && scanSessionsCursorId}
          <div class="queue-actions">
            <button class="ghost-btn" onclick={loadMoreScanSessions}>more sessions</button>
          </div>
        {/if}
      {/if}
      <div class="queue-actions">
        <input class="count-input" bind:value={workerPollMsInput} placeholder="poll ms" />
        <button class="ghost-btn" onclick={refreshWorkerStatus}>worker status</button>
        <button class="ghost-btn" onclick={startWorker}>start worker</button>
        <button class="ghost-btn" onclick={stopWorker}>stop worker</button>
        <button class="ghost-btn" onclick={clearWorkerEvents}>clear events</button>
        <button class="ghost-btn" onclick={copyQueueSnapshot}>copy snapshot</button>
      </div>
      {#if workerStatus}
        <div class="queue-meta">
          <span>worker {workerStatus.running ? 'running' : 'stopped'}</span>
          <span>poll {workerStatus.poll_interval_ms}ms</span>
          <span>processed {workerStatus.processed_total}</span>
          {#if workerStatus.last_error}
            <span>last error: {workerStatus.last_error}</span>
          {/if}
        </div>
        <div class="queue-actions">
          <button
            class="ghost-btn"
            class:active={workerEventFilter === 'all'}
            onclick={() => { workerEventFilter = 'all'; }}
          >
            all events
          </button>
          <button
            class="ghost-btn"
            class:active={workerEventFilter === 'issues'}
            onclick={() => { workerEventFilter = 'issues'; }}
          >
            issues only
          </button>
        </div>
        {#if filteredWorkerEvents().length > 0}
          <div class="queue-event-list">
            {#each filteredWorkerEvents() as event}
              <div class="queue-event-row">
                <span>{formatClock(event.at_ms)}</span>
                <span>{event.queue}</span>
                {#if event.moved_to_dlq}
                  <span class="event-tag warn">dlq</span>
                {:else if event.requeued}
                  <span class="event-tag">retry</span>
                {:else if event.error}
                  <span class="event-tag warn">error</span>
                {:else}
                  <span class="event-tag ok">ok</span>
                {/if}
                <span class="event-id">{event.message_id ?? 'n/a'}</span>
              </div>
            {/each}
          </div>
        {/if}
      {/if}
      {#if queueAttempts.length > 0}
        <div class="queue-meta">attempts: {queueAttempts.join(', ')}</div>
      {/if}
      {#if queueMsg}
        <div class="queue-msg">{queueMsg}</div>
      {/if}
      {#if queueErr}
        <div class="scan-error">{queueErr}</div>
      {/if}
    </div>
  </section>
  {/if}

  <!-- main panel -->
  <main>

    {#if !modelSetupReady()}
      <section class="scan-warning">
        model setup required: choose YOLO and face ONNX models in settings before scanning.
      </section>
    {/if}

    <!-- inputs -->
    <section class="inputs">

      <div class="input-block" class:filled={!!videoPath}>
        <div class="input-label">video input</div>
        <div class="input-row">
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
          {#if videoPath}
            <div
              class="preview-panel"
              style="width:{videoPreviewWidth}px;height:{videoPreviewHeight}px;--preview-ratio:{videoPreviewRatio};"
            >
              {#if videoPreviewSrc}
                <img
                  class="preview-media"
                  src={videoPreviewSrc}
                  alt="Video thumbnail"
                  onload={handleVideoPreviewLoad}
                />
              {:else}
                <span class="preview-fallback">loading...</span>
              {/if}
            </div>
          {/if}
        </div>
      </div>

      <div class="input-block" class:filled={!!biasPath}>
        <div class="input-label">bias reference</div>
        <div class="input-row">
          <button class="drop-zone small" onclick={pickBias}>
            {#if biasPath}
              <span class="filename">{basename(biasPath)}</span>
            {:else}
              <span class="placeholder">click to select face image</span>
            {/if}
          </button>
          {#if biasPath}
            <div
              class="preview-panel small"
              style="width:{biasPreviewWidth}px;height:{biasPreviewHeight}px;--preview-ratio:{biasPreviewRatio};"
            >
              {#if biasPreviewSrc}
                <img
                  class="preview-media"
                  src={biasPreviewSrc}
                  alt="Bias reference preview"
                  onload={handleBiasPreviewLoad}
                />
              {:else}
                <span class="preview-fallback">loading...</span>
              {/if}
            </div>
          {/if}
        </div>
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

      <div class="input-block scan-block" class:filled={scanStatus === 'done'}>
        <div class="input-label">identity discovery</div>
        <div class="scan-controls">
          <label class="count-input-wrap" for="expected-members">
            <span>expected members</span>
            <input
              id="expected-members"
              class="count-input"
              type="number"
              min="1"
              step="1"
              bind:value={expectedMembersInput}
              placeholder="optional"
            />
          </label>
          <button
            class="ghost-btn"
            disabled={!videoPath || !yoloModel || !faceModel || scanStatus === 'running' || scanStatus === 'cancelling'}
            onclick={runIdentityScan}
          >
            {scanStatus === 'running' ? 'scanning...' : 'scan members'}
          </button>
          {#if scanStatus === 'running' || scanStatus === 'cancelling'}
            <button class="ghost-btn" onclick={cancelScan} disabled={scanStatus === 'cancelling'}>
              {scanStatus === 'cancelling' ? 'cancelling...' : 'cancel scan'}
            </button>
          {/if}
        </div>

        {#if expectedMembersInvalid()}
          <div class="scan-warning">expected members must be a positive whole number</div>
        {/if}

        {#if scanStatus === 'running' || scanStatus === 'cancelling'}
          <div class="scan-meta detail">
            <span>sampled {scanSampledFrames}</span>
            <span>decoded {scanDecodedFrames}</span>
            <span>{Math.round(scanProgressFraction * 100)}%</span>
          </div>
        {/if}

        {#if scanStatus === 'error'}
          <div class="scan-error">{scanErr}</div>
        {/if}

        {#if scanStatus === 'done'}
          <div class="scan-meta">
            <span>{scanMessage}</span>
            {#if rescanPerformed}
              <span class="tag">informed rescan used</span>
            {/if}
          {#if scanNeedsReview}
            <span class="tag warn">review required</span>
          {/if}
        </div>

          <div class="scan-meta detail">
            <span>{activeCandidates().length} active members</span>
            {#if expectedMembersValue() !== null}
              <span>expected {expectedMembersValue()}</span>
            {/if}
          </div>

          {#if unresolvedDuplicatePairs().length > 0}
            <div class="scan-warning">
              <div>resolve duplicates before rendering:</div>
              <div class="duplicate-list">
                {#each unresolvedDuplicatePairs() as pair}
                  <div class="duplicate-row">
                    <span>#{pair.a + 1} ↔ #{pair.b + 1} ({(pair.similarity * 100).toFixed(0)}%)</span>
                    <div class="duplicate-actions">
                      <button class="ghost-btn tiny" onclick={() => resolveDuplicate(pair, pair.a)}>
                        keep #{pair.a + 1}
                      </button>
                      <button class="ghost-btn tiny" onclick={() => resolveDuplicate(pair, pair.b)}>
                        keep #{pair.b + 1}
                      </button>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          {#if unresolvedLowConfidenceCandidates().length > 0}
            <div class="scan-warning">
              low-confidence cards need confirmation:
              {#each unresolvedLowConfidenceCandidates() as c, idx}
                <span>{idx === 0 ? ' ' : ', '}#{c.id + 1}</span>
              {/each}
            </div>
          {/if}

          {#if countMismatchExists()}
            <div class="scan-warning">
              active members ({activeCandidates().length}) do not match expected count ({expectedMembersValue()})
            </div>
          {/if}

          {#if pendingSplitIds.length > 0}
            <div class="scan-warning">
              split requests pending for:
              {#each pendingSplitIds as splitId, idx}
                <span>{idx === 0 ? ' ' : ', '}#{splitId + 1}</span>
              {/each}
            </div>
          {/if}

          {#if scanCandidates.length > 0}
            <div class="identity-grid">
              {#each scanCandidates as candidate}
                <div
                  class="identity-card"
                  class:selected={selectedIdentityId === candidate.id}
                  class:ignored={isIgnored(candidate.id)}
                  role="button"
                  tabindex="0"
                  onclick={() => selectIdentity(candidate)}
                  onkeydown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      selectIdentity(candidate);
                    }
                  }}
                >
                  <img src={candidate.thumbnail_data_url} alt={`Identity ${candidate.id + 1}`} />
                  <div class="identity-info">
                    <span class="identity-name">member {candidate.id + 1}</span>
                    <span>confidence {(candidate.confidence * 100).toFixed(0)}%</span>
                    <span>{candidate.observations} samples</span>
                    {#if candidate.confidence < 0.55}
                      <span class="confidence-tag">low confidence</span>
                    {/if}
                  </div>
                  <div class="identity-actions">
                    <button
                      type="button"
                      class="ghost-btn tiny"
                      onclick={(event) => {
                        event.stopPropagation();
                        toggleIgnoreIdentity(candidate);
                      }}
                    >
                      {isIgnored(candidate.id) ? 'include' : 'exclude'}
                    </button>
                    {#if candidate.confidence < 0.55}
                      <button
                        type="button"
                        class="ghost-btn tiny"
                        onclick={(event) => {
                          event.stopPropagation();
                          toggleAcceptLowConfidence(candidate);
                        }}
                      >
                        {acceptedLowConfidenceIds.includes(candidate.id) ? 'unconfirm' : 'confirm'}
                      </button>
                    {/if}
                    <button
                      type="button"
                      class="ghost-btn tiny"
                      onclick={(event) => {
                        event.stopPropagation();
                        toggleSplitRequest(candidate);
                      }}
                    >
                      {pendingSplitIds.includes(candidate.id) ? 'unsplit' : 'split'}
                    </button>
                  </div>
                </div>
              {/each}
            </div>
          {/if}

          {#if reviewReasons().length > 0}
            <div class="scan-review-list">
              {reviewReasons().join(' · ')}
            </div>
          {/if}

          {#if reviewBlockers.length > 0}
            <div class="scan-review-list">
              {reviewBlockers.join(' · ')}
            </div>
          {/if}
        {/if}
      </div>

    </section>

    <!-- divider -->
    <div class="divider"></div>

    <!-- progress / status -->
    <section class="status-panel">
      {#if status === 'idle'}
        <div class="ready-msg">
          {#if !videoPath || !biasPath || !outputPath}
            select inputs above to begin
          {:else if scanStatus !== 'done'}
            run identity scan before rendering
          {:else if scanNeedsReview}
            resolve identity scan warnings before rendering
          {:else if selectedIdentityId === null}
            select a member to track
          {:else}
            ready to render
          {/if}
        </div>

      {:else if status === 'running'}
        <div class="prog-label">
          <span>rendering</span>
          <span class="prog-pct">{pct(progress)}</span>
          <span class="prog-eta">eta {formatEta(etaSeconds)}</span>
          {#if totFrames > 0}
            <span class="prog-frames">{curFrame} / {totFrames}</span>
          {/if}
        </div>
        <div class="prog-track">
          <div class="prog-fill" style="width:{pct(progress)}"></div>
        </div>

      {:else if status === 'cancelling'}
        <div class="ready-msg">cancellation requested... waiting for render worker to stop</div>

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
      {#if status === 'running' || status === 'cancelling'}
        <button class="btn-cancel" onclick={cancelJob} disabled={status === 'cancelling'}>
          {status === 'cancelling' ? 'cancelling...' : 'cancel'}
        </button>
      {:else}
        <button
          class="btn-run"
          disabled={
            !videoPath ||
            !biasPath ||
            !outputPath ||
            scanStatus !== 'done' ||
            !reviewReady ||
            selectedIdentityId === null
          }
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
  .queue-panel {
    border-top: 1px solid #23232a;
    padding-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .queue-panel h4 {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b5b67;
  }
  .queue-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 10px;
    color: #7a7a87;
  }
  .queue-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .queue-msg {
    font-size: 11px;
    color: #7ee6bd;
  }
  .queue-event-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 160px;
    overflow: auto;
    border: 1px solid #26262d;
    border-radius: 6px;
    padding: 6px;
    background: #101015;
  }
  .queue-event-row {
    display: grid;
    grid-template-columns: 54px 66px 48px 1fr;
    gap: 8px;
    align-items: center;
    font-size: 10px;
    color: #8d8d9a;
  }
  .event-tag {
    border: 1px solid #2f513f;
    color: #7ee6bd;
    border-radius: 999px;
    padding: 1px 6px;
    width: fit-content;
  }
  .event-tag.ok {
    border-color: #2f513f;
    color: #7ee6bd;
  }
  .event-tag.warn {
    border-color: #5b4732;
    color: #f5c992;
  }
  .event-id {
    color: #676776;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .diag-preview {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 10px;
    color: #8d8d9a;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .queue-session-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 180px;
    overflow: auto;
    border: 1px solid #25252c;
    border-radius: 6px;
    padding: 6px;
    background: #101015;
  }
  .queue-session-row {
    display: grid;
    grid-template-columns: 92px 66px 56px 1fr auto;
    gap: 8px;
    font-size: 10px;
    border: 1px solid #23232b;
    border-radius: 4px;
    color: #8b8b97;
    background: #14141b;
    padding: 4px 6px;
    text-align: left;
  }
  .queue-session-row:hover {
    border-color: #3a3a45;
    color: #c8c8d2;
  }
  .session-id {
    color: #d2d2dc;
  }
  .bundle-path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #6f6f7c;
  }
  .bundle-actions {
    display: flex;
    gap: 6px;
    align-items: center;
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
  .ghost-btn.active { color: #d8d8e2; border-color: #6ee7b7; background: #143126; }

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
  .input-row {
    display: flex;
    align-items: stretch;
    gap: 10px;
  }
  .input-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4a4a55;
  }
  .drop-zone {
    flex: 1;
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

  .preview-panel {
    flex-shrink: 0;
    border: 1px solid #27272d;
    border-radius: 6px;
    background: #111114;
    overflow: hidden;
    display: grid;
    place-items: center;
    transition: width 0.18s ease, height 0.18s ease;
  }
  .preview-media {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    background: #0c0c0e;
  }
  .preview-fallback {
    display: grid;
    place-items: center;
    width: 100%;
    height: 100%;
    font-size: 10px;
    color: #4a4a55;
    letter-spacing: 0.04em;
    text-transform: lowercase;
  }

  .scan-block {
    border: 1px solid #27272d;
    background: #101013;
    border-radius: 8px;
    padding: 12px;
    gap: 10px;
  }
  .scan-controls {
    display: flex;
    align-items: end;
    gap: 10px;
  }
  .count-input-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 160px;
  }
  .count-input-wrap > span {
    font-size: 10px;
    color: #71717d;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .count-input {
    background: #17171c;
    border: 1px solid #30303a;
    border-radius: 5px;
    color: #d6d6df;
    font-family: inherit;
    font-size: 12px;
    padding: 7px 8px;
  }
  .count-input:focus {
    outline: none;
    border-color: #6ee7b7;
  }
  .scan-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #8f8f9a;
  }
  .scan-meta.detail {
    color: #6f6f7e;
  }
  .tag {
    border: 1px solid #2f513f;
    color: #7ee6bd;
    background: #0f2a21;
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 10px;
  }
  .tag.warn {
    border-color: #5b4732;
    color: #f5c992;
    background: #2e2418;
  }
  .scan-warning {
    font-size: 11px;
    color: #f5c992;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .duplicate-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .duplicate-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: #1d1811;
    border: 1px solid #473526;
    border-radius: 6px;
    padding: 6px 8px;
  }
  .duplicate-actions {
    display: flex;
    gap: 6px;
  }
  .scan-error {
    font-size: 12px;
    color: #f87171;
  }
  .identity-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(136px, 1fr));
    gap: 10px;
  }
  .identity-card {
    background: #14141a;
    border: 1px solid #2b2b33;
    border-radius: 8px;
    padding: 8px;
    color: inherit;
    cursor: pointer;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: border-color 0.15s, background 0.15s;
  }
  .identity-card:hover {
    border-color: #3c3c48;
    background: #17171f;
  }
  .identity-card.selected {
    border-color: #6ee7b7;
    background: #11261f;
  }
  .identity-card.ignored {
    border-color: #433b31;
    background: #1a1713;
    opacity: 0.7;
  }
  .identity-card img {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #22222b;
    background: #0a0a0d;
  }
  .identity-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 11px;
    color: #9a9aa7;
  }
  .identity-name {
    color: #e0e0e8;
    font-size: 12px;
  }
  .confidence-tag {
    display: inline-flex;
    width: fit-content;
    border: 1px solid #5b4732;
    color: #f5c992;
    background: #2e2418;
    border-radius: 999px;
    padding: 1px 7px;
    font-size: 10px;
  }
  .identity-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .ghost-btn.tiny {
    font-size: 10px;
    padding: 2px 8px;
  }
  .scan-review-list {
    font-size: 11px;
    color: #c8a778;
  }

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
  .prog-eta {
    font-size: 11px;
    color: #8f8f9a;
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

  @media (max-width: 680px) {
    .shell {
      padding: 0 14px;
    }
    .input-row {
      flex-direction: column;
    }
    .preview-panel {
      width: min(100%, 260px) !important;
      height: auto !important;
      aspect-ratio: var(--preview-ratio, 1);
    }
    .scan-controls {
      flex-direction: column;
      align-items: stretch;
    }
  }
</style>
