import type { CropPlanV1 } from './cropPlan';

export type CropPlanWriter = (path: string, plan: CropPlanV1) => Promise<unknown>;

export type CropPlanSaveReceipt = {
  revision: number;
};

/** Serializes immutable sidecar snapshots and exposes a render-time flush barrier. */
export class CropPlanSaveQueue {
  private readonly writer: CropPlanWriter;
  private latestRevision = 0;
  private savedRevision = 0;
  private tail: Promise<void> = Promise.resolve();

  constructor(writer: CropPlanWriter) {
    this.writer = writer;
  }

  save(path: string, plan: CropPlanV1): Promise<CropPlanSaveReceipt> {
    const revision = ++this.latestRevision;
    const snapshot = structuredClone(plan);
    const operation = this.tail.catch(() => undefined).then(async () => {
      await this.writer(path, snapshot);
      this.savedRevision = revision;
    });
    this.tail = operation;
    return operation.then(() => ({ revision }));
  }

  async flush(): Promise<number> {
    await this.tail;
    return this.savedRevision;
  }

  get revision(): number {
    return this.latestRevision;
  }
}
