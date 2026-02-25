use std::collections::{HashSet, VecDeque};
use std::process::Command;
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
    pub face_model: String,
    pub expected_member_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescanJobPayload {
    pub scan_id: String,
    pub video: String,
    pub yolo_model: String,
    pub face_model: String,
    pub split_identity_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueConfig {
    pub sqs_enabled: bool,
    pub discovery_queue: String,
    pub rescan_queue: String,
    pub tracking_start_queue: String,
    pub tracking_monitor_queue: String,
    pub dlq_queue: String,
}

impl QueueConfig {
    pub fn from_env() -> Self {
        Self {
            sqs_enabled: env_bool("FOCUS_LOCK_SQS_ENABLED", false),
            discovery_queue: env_or("FOCUS_LOCK_SQS_DISCOVERY_QUEUE", "identity-discovery-queue"),
            rescan_queue: env_or("FOCUS_LOCK_SQS_RESCAN_QUEUE", "identity-rescan-queue"),
            tracking_start_queue: env_or(
                "FOCUS_LOCK_SQS_TRACKING_START_QUEUE",
                "tracking-start-queue",
            ),
            tracking_monitor_queue: env_or(
                "FOCUS_LOCK_SQS_TRACKING_MONITOR_QUEUE",
                "tracking-monitor-queue",
            ),
            dlq_queue: env_or("FOCUS_LOCK_SQS_DLQ", "identity-events-dlq"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct QueueDepths {
    pub discovery: usize,
    pub rescan: usize,
    pub tracking_start: usize,
    pub tracking_monitor: usize,
    pub dlq: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueHealth {
    pub sqs_enabled: bool,
    pub queues: QueueConfig,
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

#[derive(Debug, Clone)]
pub struct SqsReceived<T> {
    pub envelope: QueueEnvelope<T>,
    pub queue_url: String,
    pub receipt_handle: String,
}

#[derive(Default)]
pub struct QueueRuntime {
    pub config: QueueConfig,
    discovery: VecDeque<String>,
    rescan: VecDeque<String>,
    tracking_start: VecDeque<String>,
    tracking_monitor: VecDeque<String>,
    dlq: VecDeque<String>,
    seen_idempotency: HashSet<String>,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl QueueRuntime {
    pub fn new() -> Self {
        Self {
            config: QueueConfig::from_env(),
            discovery: VecDeque::new(),
            rescan: VecDeque::new(),
            tracking_start: VecDeque::new(),
            tracking_monitor: VecDeque::new(),
            dlq: VecDeque::new(),
            seen_idempotency: HashSet::new(),
        }
    }

    pub fn health(&self) -> QueueHealth {
        QueueHealth {
            sqs_enabled: self.config.sqs_enabled,
            queues: self.config.clone(),
            depths: QueueDepths {
                discovery: self.discovery.len(),
                rescan: self.rescan.len(),
                tracking_start: self.tracking_start.len(),
                tracking_monitor: self.tracking_monitor.len(),
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
        let queue_name = self.config.discovery_queue.clone();
        if self.seen_idempotency.contains(&idempotency_key) {
            return Ok(QueueEnqueueResult {
                accepted: true,
                deduplicated: true,
                queue: queue_name,
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
            queue: queue_name,
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
        let queue_name = self.config.rescan_queue.clone();
        if self.seen_idempotency.contains(&idempotency_key) {
            return Ok(QueueEnqueueResult {
                accepted: true,
                deduplicated: true,
                queue: queue_name,
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
            queue: queue_name,
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

pub async fn sqs_health(config: &QueueConfig) -> Result<QueueHealth, String> {
    let discovery = sqs_queue_depth_cli(&config.discovery_queue)?;
    let rescan = sqs_queue_depth_cli(&config.rescan_queue)?;
    let tracking_start = sqs_queue_depth_cli(&config.tracking_start_queue)?;
    let tracking_monitor = sqs_queue_depth_cli(&config.tracking_monitor_queue)?;
    let dlq = sqs_queue_depth_cli(&config.dlq_queue)?;

    Ok(QueueHealth {
        sqs_enabled: true,
        queues: config.clone(),
        depths: QueueDepths {
            discovery,
            rescan,
            tracking_start,
            tracking_monitor,
            dlq,
        },
        dedupe_keys: 0,
    })
}

pub async fn sqs_enqueue_discovery(
    config: &QueueConfig,
    payload: DiscoveryJobPayload,
    idempotency_key: String,
) -> Result<QueueEnqueueResult, String> {
    let queue_url = resolve_queue_url_cli(&config.discovery_queue)?;
    let created_at_ms = now_ms();
    let envelope = QueueEnvelope {
        message_id: format!("msg-disc-{created_at_ms}"),
        message_type: "DISCOVERY_REQUEST".to_string(),
        job_id: format!("job-disc-{created_at_ms}"),
        idempotency_key: idempotency_key.clone(),
        created_at_ms,
        attempt: 0,
        trace_id: format!("trace-disc-{}", created_at_ms % 100_000),
        payload,
    };
    let body = serde_json::to_string(&envelope)
        .map_err(|e| format!("failed to serialize discovery message: {e}"))?;

    let mut args = vec![
        "sqs".to_string(),
        "send-message".to_string(),
        "--queue-url".to_string(),
        queue_url,
        "--message-body".to_string(),
        body,
        "--output".to_string(),
        "json".to_string(),
    ];
    if config.discovery_queue.ends_with(".fifo") {
        args.extend_from_slice(&[
            "--message-group-id".to_string(),
            "focus-lock-discovery".to_string(),
            "--message-deduplication-id".to_string(),
            idempotency_key.clone(),
        ]);
    }
    let out: AwsSendMessageOutput = aws_cli_json(&args)?;
    let depth = sqs_queue_depth_cli(&config.discovery_queue)?;

    Ok(QueueEnqueueResult {
        accepted: true,
        deduplicated: false,
        queue: config.discovery_queue.clone(),
        message_id: out.message_id.unwrap_or_else(|| "unknown".to_string()),
        idempotency_key,
        depth,
    })
}

pub async fn sqs_enqueue_rescan(
    config: &QueueConfig,
    payload: RescanJobPayload,
    idempotency_key: String,
) -> Result<QueueEnqueueResult, String> {
    let queue_url = resolve_queue_url_cli(&config.rescan_queue)?;
    let created_at_ms = now_ms();
    let envelope = QueueEnvelope {
        message_id: format!("msg-rescan-{created_at_ms}"),
        message_type: "RESCAN_REQUEST".to_string(),
        job_id: format!("job-rescan-{created_at_ms}"),
        idempotency_key: idempotency_key.clone(),
        created_at_ms,
        attempt: 0,
        trace_id: format!("trace-rescan-{}", created_at_ms % 100_000),
        payload,
    };
    let body = serde_json::to_string(&envelope)
        .map_err(|e| format!("failed to serialize rescan message: {e}"))?;
    let mut args = vec![
        "sqs".to_string(),
        "send-message".to_string(),
        "--queue-url".to_string(),
        queue_url,
        "--message-body".to_string(),
        body,
        "--output".to_string(),
        "json".to_string(),
    ];
    if config.rescan_queue.ends_with(".fifo") {
        args.extend_from_slice(&[
            "--message-group-id".to_string(),
            "focus-lock-rescan".to_string(),
            "--message-deduplication-id".to_string(),
            idempotency_key.clone(),
        ]);
    }
    let out: AwsSendMessageOutput = aws_cli_json(&args)?;
    let depth = sqs_queue_depth_cli(&config.rescan_queue)?;
    Ok(QueueEnqueueResult {
        accepted: true,
        deduplicated: false,
        queue: config.rescan_queue.clone(),
        message_id: out.message_id.unwrap_or_else(|| "unknown".to_string()),
        idempotency_key,
        depth,
    })
}

pub async fn sqs_receive_discovery(
    config: &QueueConfig,
) -> Result<Option<SqsReceived<DiscoveryJobPayload>>, String> {
    let queue_url = resolve_queue_url_cli(&config.discovery_queue)?;
    let out: AwsReceiveMessageOutput = aws_cli_json(&[
        "sqs".to_string(),
        "receive-message".to_string(),
        "--queue-url".to_string(),
        queue_url.clone(),
        "--max-number-of-messages".to_string(),
        "1".to_string(),
        "--wait-time-seconds".to_string(),
        "1".to_string(),
        "--visibility-timeout".to_string(),
        "30".to_string(),
        "--output".to_string(),
        "json".to_string(),
    ])?;

    let Some(msg) = out.messages.and_then(|m| m.into_iter().next()) else {
        return Ok(None);
    };
    let body = msg.body.unwrap_or_default();
    let receipt_handle = msg
        .receipt_handle
        .ok_or("sqs message missing receipt handle")?;
    let envelope = serde_json::from_str::<QueueEnvelope<DiscoveryJobPayload>>(&body)
        .map_err(|e| format!("invalid discovery envelope body: {e}"))?;
    Ok(Some(SqsReceived {
        envelope,
        queue_url,
        receipt_handle,
    }))
}

pub async fn sqs_receive_rescan(
    config: &QueueConfig,
) -> Result<Option<SqsReceived<RescanJobPayload>>, String> {
    let queue_url = resolve_queue_url_cli(&config.rescan_queue)?;
    let out: AwsReceiveMessageOutput = aws_cli_json(&[
        "sqs".to_string(),
        "receive-message".to_string(),
        "--queue-url".to_string(),
        queue_url.clone(),
        "--max-number-of-messages".to_string(),
        "1".to_string(),
        "--wait-time-seconds".to_string(),
        "1".to_string(),
        "--visibility-timeout".to_string(),
        "30".to_string(),
        "--output".to_string(),
        "json".to_string(),
    ])?;

    let Some(msg) = out.messages.and_then(|m| m.into_iter().next()) else {
        return Ok(None);
    };
    let body = msg.body.unwrap_or_default();
    let receipt_handle = msg
        .receipt_handle
        .ok_or("sqs message missing receipt handle")?;
    let envelope = serde_json::from_str::<QueueEnvelope<RescanJobPayload>>(&body)
        .map_err(|e| format!("invalid rescan envelope body: {e}"))?;
    Ok(Some(SqsReceived {
        envelope,
        queue_url,
        receipt_handle,
    }))
}

pub async fn sqs_ack(queue_url: &str, receipt_handle: &str) -> Result<(), String> {
    let _out: serde_json::Value = aws_cli_json(&[
        "sqs".to_string(),
        "delete-message".to_string(),
        "--queue-url".to_string(),
        queue_url.to_string(),
        "--receipt-handle".to_string(),
        receipt_handle.to_string(),
        "--output".to_string(),
        "json".to_string(),
    ])?;
    Ok(())
}

pub async fn sqs_retry_or_dlq(
    config: &QueueConfig,
    queue_url: &str,
    receipt_handle: &str,
    mut envelope: QueueEnvelope<DiscoveryJobPayload>,
    max_attempts_before_dlq: u32,
) -> Result<(bool, bool), String> {
    let mut moved_to_dlq = false;
    let mut requeued = false;

    if envelope.attempt + 1 >= max_attempts_before_dlq {
        let dlq_url = resolve_queue_url_cli(&config.dlq_queue)?;
        let body = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize dlq envelope: {e}"))?;
        let _out: AwsSendMessageOutput = aws_cli_json(&[
            "sqs".to_string(),
            "send-message".to_string(),
            "--queue-url".to_string(),
            dlq_url,
            "--message-body".to_string(),
            body,
            "--output".to_string(),
            "json".to_string(),
        ])?;
        moved_to_dlq = true;
    } else {
        envelope.attempt = envelope.attempt.saturating_add(1);
        let body = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize retry envelope: {e}"))?;
        let mut args = vec![
            "sqs".to_string(),
            "send-message".to_string(),
            "--queue-url".to_string(),
            queue_url.to_string(),
            "--message-body".to_string(),
            body,
            "--output".to_string(),
            "json".to_string(),
        ];
        if config.discovery_queue.ends_with(".fifo") {
            args.extend_from_slice(&[
                "--message-group-id".to_string(),
                "focus-lock-discovery".to_string(),
                "--message-deduplication-id".to_string(),
                format!("{}-attempt-{}", envelope.idempotency_key, envelope.attempt),
            ]);
        }
        let _out: AwsSendMessageOutput = aws_cli_json(&args)?;
        requeued = true;
    }

    sqs_ack(queue_url, receipt_handle).await?;
    Ok((moved_to_dlq, requeued))
}

pub async fn sqs_retry_or_dlq_rescan(
    config: &QueueConfig,
    queue_url: &str,
    receipt_handle: &str,
    mut envelope: QueueEnvelope<RescanJobPayload>,
    max_attempts_before_dlq: u32,
) -> Result<(bool, bool), String> {
    let mut moved_to_dlq = false;
    let mut requeued = false;

    if envelope.attempt + 1 >= max_attempts_before_dlq {
        let dlq_url = resolve_queue_url_cli(&config.dlq_queue)?;
        let body = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize dlq rescan envelope: {e}"))?;
        let _out: AwsSendMessageOutput = aws_cli_json(&[
            "sqs".to_string(),
            "send-message".to_string(),
            "--queue-url".to_string(),
            dlq_url,
            "--message-body".to_string(),
            body,
            "--output".to_string(),
            "json".to_string(),
        ])?;
        moved_to_dlq = true;
    } else {
        envelope.attempt = envelope.attempt.saturating_add(1);
        let body = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize retry rescan envelope: {e}"))?;
        let mut args = vec![
            "sqs".to_string(),
            "send-message".to_string(),
            "--queue-url".to_string(),
            queue_url.to_string(),
            "--message-body".to_string(),
            body,
            "--output".to_string(),
            "json".to_string(),
        ];
        if config.rescan_queue.ends_with(".fifo") {
            args.extend_from_slice(&[
                "--message-group-id".to_string(),
                "focus-lock-rescan".to_string(),
                "--message-deduplication-id".to_string(),
                format!("{}-attempt-{}", envelope.idempotency_key, envelope.attempt),
            ]);
        }
        let _out: AwsSendMessageOutput = aws_cli_json(&args)?;
        requeued = true;
    }

    sqs_ack(queue_url, receipt_handle).await?;
    Ok((moved_to_dlq, requeued))
}

fn resolve_queue_url_cli(queue_name: &str) -> Result<String, String> {
    let out: AwsQueueUrlOutput = aws_cli_json(&[
        "sqs".to_string(),
        "get-queue-url".to_string(),
        "--queue-name".to_string(),
        queue_name.to_string(),
        "--output".to_string(),
        "json".to_string(),
    ])?;
    out.queue_url
        .ok_or_else(|| format!("sqs queue url missing for {queue_name}"))
}

fn sqs_queue_depth_cli(queue_name: &str) -> Result<usize, String> {
    let url = resolve_queue_url_cli(queue_name)?;
    let out: AwsQueueAttributesOutput = aws_cli_json(&[
        "sqs".to_string(),
        "get-queue-attributes".to_string(),
        "--queue-url".to_string(),
        url,
        "--attribute-names".to_string(),
        "ApproximateNumberOfMessages".to_string(),
        "--output".to_string(),
        "json".to_string(),
    ])?;
    let value = out
        .attributes
        .and_then(|attrs| attrs.approximate_number_of_messages)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    Ok(value)
}

fn aws_cli_json<T: for<'de> Deserialize<'de>>(args: &[String]) -> Result<T, String> {
    let output = Command::new("aws")
        .args(args)
        .output()
        .map_err(|e| format!("failed to execute aws cli: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("aws cli command failed: {stderr}"));
    }
    serde_json::from_slice::<T>(&output.stdout)
        .map_err(|e| format!("failed to parse aws cli json output: {e}"))
}

#[derive(Debug, Deserialize)]
struct AwsQueueUrlOutput {
    #[serde(rename = "QueueUrl")]
    queue_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AwsQueueAttributesOutput {
    #[serde(rename = "Attributes")]
    attributes: Option<AwsAttributes>,
}

#[derive(Debug, Deserialize)]
struct AwsAttributes {
    #[serde(rename = "ApproximateNumberOfMessages")]
    approximate_number_of_messages: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AwsSendMessageOutput {
    #[serde(rename = "MessageId")]
    message_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AwsReceiveMessageOutput {
    #[serde(rename = "Messages")]
    messages: Option<Vec<AwsReceivedMessage>>,
}

#[derive(Debug, Deserialize)]
struct AwsReceivedMessage {
    #[serde(rename = "Body")]
    body: Option<String>,
    #[serde(rename = "ReceiptHandle")]
    receipt_handle: Option<String>,
}

fn env_or(key: &str, fallback: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| fallback.to_string())
}

fn env_bool(key: &str, fallback: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => fallback,
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
            face_model: "face.onnx".to_string(),
            expected_member_count: Some(5),
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
