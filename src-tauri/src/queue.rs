use std::collections::{HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEnvelope<T> {
    pub message_id: String,
    pub message_type: String,
    pub job_id: String,
    pub idempotency_key: String,
    pub created_at_ms: u64,
    pub attempt: u32,
    pub trace_id: String,
    pub payload: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryJobPayload {
    pub scan_id: String,
    pub video: String,
    pub yolo_model: String,
    pub identity_model: String,
    pub expected_member_count: Option<u32>,
    pub processing_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescanJobPayload {
    pub scan_id: String,
    pub video: String,
    pub yolo_model: String,
    pub identity_model: String,
    pub split_identity_ids: Vec<usize>,
    pub processing_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct QueueDepths {
    pub discovery: usize,
    pub rescan: usize,
    pub dlq: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueHealth {
    pub depths: QueueDepths,
    pub dedupe_keys: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueEnqueueResult {
    pub accepted: bool,
    pub deduplicated: bool,
    pub queue: String,
    pub message_id: String,
    pub idempotency_key: String,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueProcessResult {
    pub processed: bool,
    pub queue: String,
    pub message_id: Option<String>,
    pub job_id: Option<String>,
    pub moved_to_dlq: bool,
    pub requeued: bool,
    pub attempt: Option<u32>,
    pub error: Option<String>,
    pub remaining_depth: usize,
}

#[derive(Debug, Clone)]
pub struct DiscoveryDequeued {
    pub envelope: QueueEnvelope<DiscoveryJobPayload>,
    pub raw: String,
}

#[derive(Debug, Clone)]
pub struct RescanDequeued {
    pub envelope: QueueEnvelope<RescanJobPayload>,
    pub raw: String,
}

#[derive(Default)]
pub struct QueueRuntime {
    discovery: VecDeque<String>,
    rescan: VecDeque<String>,
    dlq: VecDeque<String>,
    seen_idempotency: HashSet<String>,
}

impl QueueRuntime {
    pub fn new() -> Self {
        Self {
            discovery: VecDeque::new(),
            rescan: VecDeque::new(),
            dlq: VecDeque::new(),
            seen_idempotency: HashSet::new(),
        }
    }

    pub fn health(&self) -> QueueHealth {
        QueueHealth {
            depths: QueueDepths {
                discovery: self.discovery.len(),
                rescan: self.rescan.len(),
                dlq: self.dlq.len(),
            },
            dedupe_keys: self.seen_idempotency.len(),
        }
    }

    pub fn enqueue_discovery(
        &mut self,
        payload: DiscoveryJobPayload,
        idempotency_key: String,
    ) -> Result<QueueEnqueueResult, String> {
        if self.seen_idempotency.contains(&idempotency_key) {
            return Ok(QueueEnqueueResult {
                accepted: true,
                deduplicated: true,
                queue: "discovery".to_string(),
                message_id: "deduplicated".to_string(),
                idempotency_key,
                depth: self.discovery.len(),
            });
        }

        let created_at_ms = now_ms();
        let message_id = format!("msg-disc-{created_at_ms}");
        let job_id = format!("job-disc-{created_at_ms}");
        let trace_id = format!("trace-disc-{}", created_at_ms % 100_000);

        let envelope = QueueEnvelope {
            message_id: message_id.clone(),
            message_type: "DISCOVERY_REQUEST".to_string(),
            job_id,
            idempotency_key: idempotency_key.clone(),
            created_at_ms,
            attempt: 0,
            trace_id,
            payload,
        };

        let serialized = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize queue message: {e}"))?;
        self.discovery.push_back(serialized);
        self.seen_idempotency.insert(idempotency_key.clone());

        Ok(QueueEnqueueResult {
            accepted: true,
            deduplicated: false,
            queue: "discovery".to_string(),
            message_id,
            idempotency_key,
            depth: self.discovery.len(),
        })
    }

    pub fn enqueue_rescan(
        &mut self,
        payload: RescanJobPayload,
        idempotency_key: String,
    ) -> Result<QueueEnqueueResult, String> {
        if self.seen_idempotency.contains(&idempotency_key) {
            return Ok(QueueEnqueueResult {
                accepted: true,
                deduplicated: true,
                queue: "rescan".to_string(),
                message_id: "deduplicated".to_string(),
                idempotency_key,
                depth: self.rescan.len(),
            });
        }

        let created_at_ms = now_ms();
        let message_id = format!("msg-rescan-{created_at_ms}");
        let envelope = QueueEnvelope {
            message_id: message_id.clone(),
            message_type: "RESCAN_REQUEST".to_string(),
            job_id: format!("job-rescan-{created_at_ms}"),
            idempotency_key: idempotency_key.clone(),
            created_at_ms,
            attempt: 0,
            trace_id: format!("trace-rescan-{}", created_at_ms % 100_000),
            payload,
        };
        let serialized = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize rescan queue message: {e}"))?;
        self.rescan.push_back(serialized);
        self.seen_idempotency.insert(idempotency_key.clone());

        Ok(QueueEnqueueResult {
            accepted: true,
            deduplicated: false,
            queue: "rescan".to_string(),
            message_id,
            idempotency_key,
            depth: self.rescan.len(),
        })
    }

    pub fn requeue_discovery_retry(
        &mut self,
        mut envelope: QueueEnvelope<DiscoveryJobPayload>,
    ) -> Result<usize, String> {
        envelope.attempt = envelope.attempt.saturating_add(1);
        let serialized = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize retry discovery message: {e}"))?;
        self.discovery.push_back(serialized);
        Ok(self.discovery.len())
    }

    pub fn dequeue_discovery(&mut self) -> Result<Option<DiscoveryDequeued>, String> {
        let Some(raw) = self.discovery.pop_front() else {
            return Ok(None);
        };
        let parsed = serde_json::from_str::<QueueEnvelope<DiscoveryJobPayload>>(&raw)
            .map_err(|e| format!("failed to parse discovery queue message: {e}"))?;
        Ok(Some(DiscoveryDequeued {
            envelope: parsed,
            raw,
        }))
    }

    pub fn move_discovery_to_dlq(&mut self, raw: String) -> usize {
        self.dlq.push_back(raw);
        self.dlq.len()
    }

    pub fn dequeue_rescan(&mut self) -> Result<Option<RescanDequeued>, String> {
        let Some(raw) = self.rescan.pop_front() else {
            return Ok(None);
        };
        let parsed = serde_json::from_str::<QueueEnvelope<RescanJobPayload>>(&raw)
            .map_err(|e| format!("failed to parse rescan queue message: {e}"))?;
        Ok(Some(RescanDequeued {
            envelope: parsed,
            raw,
        }))
    }

    pub fn requeue_rescan_retry(
        &mut self,
        mut envelope: QueueEnvelope<RescanJobPayload>,
    ) -> Result<usize, String> {
        envelope.attempt = envelope.attempt.saturating_add(1);
        let serialized = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize retry rescan message: {e}"))?;
        self.rescan.push_back(serialized);
        Ok(self.rescan.len())
    }

    pub fn move_rescan_to_dlq(&mut self, raw: String) -> usize {
        self.dlq.push_back(raw);
        self.dlq.len()
    }

    pub fn peek_discovery_attempts(&self, limit: usize) -> Result<Vec<u32>, String> {
        self.discovery
            .iter()
            .take(limit.max(1))
            .map(|raw| {
                let parsed = serde_json::from_str::<QueueEnvelope<DiscoveryJobPayload>>(raw)
                    .map_err(|e| format!("failed to parse discovery queue message: {e}"))?;
                Ok(parsed.attempt)
            })
            .collect()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(scan_id: &str) -> DiscoveryJobPayload {
        DiscoveryJobPayload {
            scan_id: scan_id.to_string(),
            video: "video.mp4".to_string(),
            yolo_model: "yolo.onnx".to_string(),
            identity_model: "face.onnx".to_string(),
            expected_member_count: Some(5),
            processing_mode: Some("fast".to_string()),
        }
    }

    #[test]
    fn deduplicates_same_idempotency_key() {
        let mut q = QueueRuntime::new();
        let first = q
            .enqueue_discovery(payload("scan-1"), "idem-a".to_string())
            .expect("enqueue should work");
        assert!(first.accepted);
        assert!(!first.deduplicated);

        let second = q
            .enqueue_discovery(payload("scan-1"), "idem-a".to_string())
            .expect("second enqueue should work");
        assert!(second.accepted);
        assert!(second.deduplicated);
        assert_eq!(q.health().depths.discovery, 1);
    }

    #[test]
    fn retry_increments_attempt() {
        let mut q = QueueRuntime::new();
        q.enqueue_discovery(payload("scan-2"), "idem-b".to_string())
            .expect("enqueue should work");
        let msg = q
            .dequeue_discovery()
            .expect("dequeue should parse")
            .expect("message should exist");
        assert_eq!(msg.envelope.attempt, 0);

        q.requeue_discovery_retry(msg.envelope)
            .expect("requeue should work");
        let after = q
            .dequeue_discovery()
            .expect("dequeue should parse")
            .expect("message should exist");
        assert_eq!(after.envelope.attempt, 1);
    }

    #[test]
    fn dlq_move_increases_dlq_depth() {
        let mut q = QueueRuntime::new();
        q.enqueue_discovery(payload("scan-3"), "idem-c".to_string())
            .expect("enqueue should work");
        let msg = q
            .dequeue_discovery()
            .expect("dequeue should parse")
            .expect("message should exist");
        let dlq_depth = q.move_discovery_to_dlq(msg.raw);
        assert_eq!(dlq_depth, 1);
        assert_eq!(q.health().depths.discovery, 0);
        assert_eq!(q.health().depths.dlq, 1);
    }
}
