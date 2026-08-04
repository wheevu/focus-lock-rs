import { invoke } from '@tauri-apps/api/core';
import type { CropPlanV1, ManualKeyframe } from './cropPlan';

export type CommandName =
  | 'model_dir'
  | 'read_crop_plan'
  | 'write_crop_plan'
  | 'update_crop_plan_keyframe'
  | 'probe_video'
  | 'read_thumbnail'
  | 'scan_identities'
  | 'validate_identity_review'
  | 'get_identity_scan'
  | 'cleanup_identity_scans'
  | 'query_identity_scans'
  | 'query_scan_events'
  | 'scan_storage_stats'
  | 'run_scan_storage_maintenance'
  | 'export_diagnostics_bundle'
  | 'list_diagnostics_bundles'
  | 'storage_worker_start'
  | 'storage_worker_stop'
  | 'storage_worker_status'
  | 'queue_health'
  | 'queue_dlq_list'
  | 'queue_dlq_replay'
  | 'enqueue_discovery_job'
  | 'enqueue_split_rescan_job'
  | 'process_next_discovery_job'
  | 'process_next_rescan_job'
  | 'queue_peek_discovery_attempts'
  | 'queue_worker_start'
  | 'queue_worker_stop'
  | 'queue_worker_status'
  | 'queue_worker_clear_events'
  | 'cancel_job'
  | 'cancel_scan'
  | 'run_fancam';

type InvokeArgs = Record<string, unknown> | number[] | ArrayBuffer | Uint8Array;

export async function invokeCommand<T>(command: CommandName, args?: InvokeArgs): Promise<T> {
  try {
    return await invoke<T>(command, args);
  } catch (error: unknown) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(typeof error === 'string' ? error : JSON.stringify(error));
  }
}

export type CropPlanWriteResult = {
  path: string;
  bytes: number;
};

/** Load and source-check a crop sidecar through the Tauri boundary. */
export async function readCropPlan(path: string, video: string): Promise<CropPlanV1> {
  return invokeCommand<CropPlanV1>('read_crop_plan', { args: { path, video } });
}

/** Persist a validated crop sidecar through the Tauri boundary. */
export async function writeCropPlan(
  path: string,
  plan: CropPlanV1,
): Promise<CropPlanWriteResult> {
  return invokeCommand<CropPlanWriteResult>('write_crop_plan', { args: { path, plan } });
}

/** Atomically apply one manual correction to an existing sidecar. */
export async function updateCropPlanKeyframe(
  path: string,
  keyframe: ManualKeyframe,
): Promise<CropPlanV1> {
  return invokeCommand<CropPlanV1>('update_crop_plan_keyframe', {
    args: { path, keyframe },
  });
}
