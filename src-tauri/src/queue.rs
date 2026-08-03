use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

const QUEUE_SCHEMA_VERSION: i64 = 1;
pub const QUEUE_DEPTH_LIMIT: usize = 1024;
const QUEUE_DEDUPE_LIMIT: usize = 1024;
const MAX_RETRY_DELAY_MS: u64 = 300_000;
static QUEUE_MESSAGE_SEQ: AtomicU64 = AtomicU64::new(1);

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
    pub discovery_in_flight: usize,
    pub rescan_in_flight: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueHealth {
    pub depths: QueueDepths,
    pub dedupe_keys: usize,
    pub capacity: usize,
    pub next_discovery_at_ms: Option<u64>,
    pub next_rescan_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueEnqueueResult {
    pub accepted: bool,
    pub deduplicated: bool,
    pub queue: String,
    pub message_id: String,
    pub idempotency_key: String,
    pub depth: usize,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueProcessResult {
    pub processed: bool,
    pub cancelled: bool,
    pub queue: String,
    pub message_id: Option<String>,
    pub job_id: Option<String>,
    pub moved_to_dlq: bool,
    pub requeued: bool,
    pub attempt: Option<u32>,
    pub error: Option<String>,
    pub remaining_depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueDlqItemSummary {
    pub message_id: String,
    pub queue: String,
    pub job_id: String,
    pub attempt: u32,
    pub created_at_ms: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueReplayResult {
    pub replayed: bool,
    pub queue: String,
    pub message_id: String,
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

/// Queue runtime backed by SQLite in production and by memory in unit tests.
///
/// The durable form is deliberately at-least-once: a claimed message remains
/// in the database until the caller acknowledges successful processing.
pub struct QueueRuntime {
    conn: Option<Connection>,
    discovery: VecDeque<String>,
    rescan: VecDeque<String>,
    dlq: VecDeque<String>,
    seen_idempotency: HashSet<String>,
    seen_idempotency_order: VecDeque<String>,
}

impl Default for QueueRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueRuntime {
    /// Create an in-memory queue. This is also useful for focused unit tests.
    pub fn new() -> Self {
        Self {
            conn: None,
            discovery: VecDeque::new(),
            rescan: VecDeque::new(),
            dlq: VecDeque::new(),
            seen_idempotency: HashSet::new(),
            seen_idempotency_order: VecDeque::new(),
        }
    }

    /// Open and recover a durable queue database.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let conn =
            Connection::open(path).map_err(|e| format!("failed to open queue database: {e}"))?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| format!("failed to enable queue WAL: {e}"))?;
        conn.pragma_update(None, "synchronous", "FULL")
            .map_err(|e| format!("failed to configure queue durability: {e}"))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| format!("failed to configure queue busy timeout: {e}"))?;
        initialize_schema(&conn)?;
        conn.execute(
            "UPDATE queue_messages
             SET state = 'ready', claimed_at_ms = NULL, available_at_ms = ?1
             WHERE state = 'in_flight'",
            params![now_ms()],
        )
        .map_err(|e| format!("failed to recover in-flight queue messages: {e}"))?;

        Ok(Self {
            conn: Some(conn),
            ..Self::new()
        })
    }

    pub fn health(&self) -> QueueHealth {
        if let Some(conn) = &self.conn {
            return durable_health(conn);
        }

        QueueHealth {
            depths: QueueDepths {
                discovery: self.discovery.len(),
                rescan: self.rescan.len(),
                dlq: self.dlq.len(),
                discovery_in_flight: 0,
                rescan_in_flight: 0,
            },
            dedupe_keys: self.seen_idempotency.len(),
            capacity: QUEUE_DEPTH_LIMIT,
            next_discovery_at_ms: None,
            next_rescan_at_ms: None,
        }
    }

    pub fn enqueue_discovery(
        &mut self,
        payload: DiscoveryJobPayload,
        idempotency_key: String,
    ) -> Result<QueueEnqueueResult, String> {
        let created_at_ms = now_ms();
        let sequence = QUEUE_MESSAGE_SEQ.fetch_add(1, Ordering::Relaxed);
        let message_id = format!("msg-disc-{created_at_ms}-{sequence}");
        let envelope = QueueEnvelope {
            message_id: message_id.clone(),
            message_type: "DISCOVERY_REQUEST".to_string(),
            job_id: format!("job-disc-{created_at_ms}-{sequence}"),
            idempotency_key,
            created_at_ms,
            attempt: 0,
            trace_id: format!("trace-disc-{created_at_ms}-{sequence}"),
            payload,
        };

        if self.conn.is_some() {
            return self.enqueue_durable("discovery", &envelope);
        }
        let serialized = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize discovery queue message: {e}"))?;
        self.enqueue_memory("discovery", serialized, envelope)
    }

    pub fn enqueue_rescan(
        &mut self,
        payload: RescanJobPayload,
        idempotency_key: String,
    ) -> Result<QueueEnqueueResult, String> {
        let created_at_ms = now_ms();
        let sequence = QUEUE_MESSAGE_SEQ.fetch_add(1, Ordering::Relaxed);
        let message_id = format!("msg-rescan-{created_at_ms}-{sequence}");
        let envelope = QueueEnvelope {
            message_id: message_id.clone(),
            message_type: "RESCAN_REQUEST".to_string(),
            job_id: format!("job-rescan-{created_at_ms}-{sequence}"),
            idempotency_key,
            created_at_ms,
            attempt: 0,
            trace_id: format!("trace-rescan-{created_at_ms}-{sequence}"),
            payload,
        };

        if self.conn.is_some() {
            return self.enqueue_durable("rescan", &envelope);
        }
        let serialized = serde_json::to_string(&envelope)
            .map_err(|e| format!("failed to serialize rescan queue message: {e}"))?;
        self.enqueue_memory("rescan", serialized, envelope)
    }

    pub fn dequeue_discovery(&mut self) -> Result<Option<DiscoveryDequeued>, String> {
        let claimed = if self.conn.is_some() {
            self.claim_durable("discovery")?
        } else {
            self.discovery.pop_front().map(|raw| (String::new(), raw))
        };
        let Some((message_id, raw)) = claimed else {
            return Ok(None);
        };
        let parsed = match serde_json::from_str::<QueueEnvelope<DiscoveryJobPayload>>(&raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                if self.conn.is_some() {
                    self.move_durable_to_dlq(&message_id, Some(&error.to_string()))?;
                } else {
                    self.dlq.push_back(raw);
                }
                return Err(format!(
                    "failed to parse discovery queue message; moved to DLQ: {error}"
                ));
            }
        };
        Ok(Some(DiscoveryDequeued {
            envelope: parsed,
            raw,
        }))
    }

    pub fn dequeue_rescan(&mut self) -> Result<Option<RescanDequeued>, String> {
        let claimed = if self.conn.is_some() {
            self.claim_durable("rescan")?
        } else {
            self.rescan.pop_front().map(|raw| (String::new(), raw))
        };
        let Some((message_id, raw)) = claimed else {
            return Ok(None);
        };
        let parsed = match serde_json::from_str::<QueueEnvelope<RescanJobPayload>>(&raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                if self.conn.is_some() {
                    self.move_durable_to_dlq(&message_id, Some(&error.to_string()))?;
                } else {
                    self.dlq.push_back(raw);
                }
                return Err(format!(
                    "failed to parse rescan queue message; moved to DLQ: {error}"
                ));
            }
        };
        Ok(Some(RescanDequeued {
            envelope: parsed,
            raw,
        }))
    }

    pub fn acknowledge(&mut self, message_id: &str) -> Result<(), String> {
        let Some(conn) = self.conn.as_mut() else {
            return Ok(());
        };
        let changed = conn
            .execute(
                "DELETE FROM queue_messages WHERE message_id = ?1 AND state = 'in_flight'",
                params![message_id],
            )
            .map_err(|e| format!("failed to acknowledge queue message: {e}"))?;
        if changed == 0 {
            return Err(format!("queue message is not in flight: {message_id}"));
        }
        Ok(())
    }

    pub fn requeue_discovery_raw(&mut self, raw: String) -> usize {
        self.requeue_discovery_raw_result(raw)
            .unwrap_or_else(|_| self.health().depths.discovery)
    }

    pub fn requeue_discovery_raw_result(&mut self, raw: String) -> Result<usize, String> {
        if self.conn.is_some() {
            let message_id = message_id_from_raw(&raw)?;
            self.requeue_durable(&message_id, &raw, None, 0)?;
            return Ok(self.health().depths.discovery);
        }
        self.discovery.push_front(raw);
        Ok(self.discovery.len())
    }

    pub fn requeue_rescan_raw(&mut self, raw: String) -> usize {
        self.requeue_rescan_raw_result(raw)
            .unwrap_or_else(|_| self.health().depths.rescan)
    }

    pub fn requeue_rescan_raw_result(&mut self, raw: String) -> Result<usize, String> {
        if self.conn.is_some() {
            let message_id = message_id_from_raw(&raw)?;
            self.requeue_durable(&message_id, &raw, None, 0)?;
            return Ok(self.health().depths.rescan);
        }
        self.rescan.push_front(raw);
        Ok(self.rescan.len())
    }

    pub fn requeue_discovery_retry(
        &mut self,
        mut envelope: QueueEnvelope<DiscoveryJobPayload>,
    ) -> Result<usize, String> {
        self.requeue_discovery_retry_with_error(&mut envelope, None)
    }

    pub fn requeue_discovery_retry_with_error(
        &mut self,
        envelope: &mut QueueEnvelope<DiscoveryJobPayload>,
        error: Option<&str>,
    ) -> Result<usize, String> {
        envelope.attempt = envelope.attempt.saturating_add(1);
        let serialized = serde_json::to_string(envelope)
            .map_err(|e| format!("failed to serialize retry discovery message: {e}"))?;
        if self.conn.is_some() {
            self.requeue_durable(&envelope.message_id, &serialized, error, envelope.attempt)?;
            return Ok(self.health().depths.discovery);
        }
        self.discovery.push_back(serialized);
        Ok(self.discovery.len())
    }

    pub fn requeue_rescan_retry(
        &mut self,
        mut envelope: QueueEnvelope<RescanJobPayload>,
    ) -> Result<usize, String> {
        self.requeue_rescan_retry_with_error(&mut envelope, None)
    }

    pub fn requeue_rescan_retry_with_error(
        &mut self,
        envelope: &mut QueueEnvelope<RescanJobPayload>,
        error: Option<&str>,
    ) -> Result<usize, String> {
        envelope.attempt = envelope.attempt.saturating_add(1);
        let serialized = serde_json::to_string(envelope)
            .map_err(|e| format!("failed to serialize retry rescan message: {e}"))?;
        if self.conn.is_some() {
            self.requeue_durable(&envelope.message_id, &serialized, error, envelope.attempt)?;
            return Ok(self.health().depths.rescan);
        }
        self.rescan.push_back(serialized);
        Ok(self.rescan.len())
    }

    pub fn move_discovery_to_dlq(&mut self, raw: String) -> usize {
        self.move_discovery_to_dlq_with_error(raw, None)
            .unwrap_or_else(|_| self.health().depths.dlq)
    }

    pub fn move_discovery_to_dlq_with_error(
        &mut self,
        raw: String,
        error: Option<&str>,
    ) -> Result<usize, String> {
        if self.conn.is_some() {
            let message_id = message_id_from_raw(&raw)?;
            self.move_durable_to_dlq(&message_id, error)?;
            return Ok(self.health().depths.dlq);
        }
        self.dlq.push_back(raw);
        Ok(self.dlq.len())
    }

    pub fn move_rescan_to_dlq(&mut self, raw: String) -> usize {
        self.move_rescan_to_dlq_with_error(raw, None)
            .unwrap_or_else(|_| self.health().depths.dlq)
    }

    pub fn move_rescan_to_dlq_with_error(
        &mut self,
        raw: String,
        error: Option<&str>,
    ) -> Result<usize, String> {
        if self.conn.is_some() {
            let message_id = message_id_from_raw(&raw)?;
            self.move_durable_to_dlq(&message_id, error)?;
            return Ok(self.health().depths.dlq);
        }
        self.dlq.push_back(raw);
        Ok(self.dlq.len())
    }

    pub fn peek_discovery_attempts(&self, limit: usize) -> Result<Vec<u32>, String> {
        if let Some(conn) = &self.conn {
            let mut statement = conn
                .prepare(
                    "SELECT attempt FROM queue_messages
                     WHERE queue = 'discovery' AND state = 'ready'
                     ORDER BY available_at_ms, created_at_ms, message_id LIMIT ?1",
                )
                .map_err(|e| format!("failed to prepare queue peek: {e}"))?;
            return statement
                .query_map(params![limit.max(1) as i64], |row| row.get::<_, i64>(0))
                .map_err(|e| format!("failed to query queue attempts: {e}"))?
                .map(|row| {
                    row.map(|attempt| attempt.max(0) as u32)
                        .map_err(|e| e.to_string())
                })
                .collect();
        }
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

    pub fn list_dlq(&self, limit: usize) -> Result<Vec<QueueDlqItemSummary>, String> {
        if let Some(conn) = &self.conn {
            let mut statement = conn
                .prepare(
                    "SELECT message_id, queue, job_id, attempt, created_at_ms, last_error
                     FROM queue_messages WHERE state = 'dlq'
                     ORDER BY created_at_ms DESC, message_id DESC LIMIT ?1",
                )
                .map_err(|e| format!("failed to prepare DLQ query: {e}"))?;
            return statement
                .query_map(params![limit.max(1) as i64], |row| {
                    Ok(QueueDlqItemSummary {
                        message_id: row.get(0)?,
                        queue: row.get(1)?,
                        job_id: row.get(2)?,
                        attempt: row.get::<_, i64>(3)?.max(0) as u32,
                        created_at_ms: row.get::<_, i64>(4)?.max(0) as u64,
                        last_error: row.get(5)?,
                    })
                })
                .map_err(|e| format!("failed to query DLQ: {e}"))?
                .map(|row| row.map_err(|e| e.to_string()))
                .collect();
        }

        let items = self
            .dlq
            .iter()
            .rev()
            .take(limit.max(1))
            .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .filter_map(|value| {
                Some(QueueDlqItemSummary {
                    message_id: value.get("message_id")?.as_str()?.to_string(),
                    queue: if value.get("message_type")?.as_str()? == "DISCOVERY_REQUEST" {
                        "discovery".to_string()
                    } else {
                        "rescan".to_string()
                    },
                    job_id: value.get("job_id")?.as_str()?.to_string(),
                    attempt: value.get("attempt")?.as_u64()? as u32,
                    created_at_ms: value.get("created_at_ms")?.as_u64()?,
                    last_error: None,
                })
            })
            .collect::<Vec<_>>();
        Ok(items)
    }

    pub fn replay_dlq(&mut self, message_id: &str) -> Result<QueueReplayResult, String> {
        if let Some(conn) = self.conn.as_mut() {
            let queue: Option<String> = conn
                .query_row(
                    "SELECT queue FROM queue_messages WHERE message_id = ?1 AND state = 'dlq'",
                    params![message_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| format!("failed to inspect DLQ message: {e}"))?;
            let Some(queue) = queue else {
                return Err("DLQ message not found".to_string());
            };
            let pending: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM queue_messages
                     WHERE queue = ?1 AND state IN ('ready', 'in_flight')",
                    params![queue],
                    |row| row.get(0),
                )
                .map_err(|e| format!("failed to count queue depth: {e}"))?;
            if pending as usize >= QUEUE_DEPTH_LIMIT {
                return Err(format!("{queue} queue is full"));
            }
            conn.execute(
                "UPDATE queue_messages
                 SET state = 'ready', attempt = 0, available_at_ms = ?2,
                     claimed_at_ms = NULL, last_error = NULL
                 WHERE message_id = ?1 AND state = 'dlq'",
                params![message_id, now_ms()],
            )
            .map_err(|e| format!("failed to replay DLQ message: {e}"))?;
            return Ok(QueueReplayResult {
                replayed: true,
                queue,
                message_id: message_id.to_string(),
            });
        }

        let index = self
            .dlq
            .iter()
            .position(|raw| message_id_from_raw(raw).ok().as_deref() == Some(message_id))
            .ok_or_else(|| "DLQ message not found".to_string())?;
        let raw = self
            .dlq
            .remove(index)
            .ok_or_else(|| "DLQ message not found".to_string())?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| format!("failed to parse DLQ message: {e}"))?;
        let queue = if value
            .get("message_type")
            .and_then(serde_json::Value::as_str)
            == Some("DISCOVERY_REQUEST")
        {
            "discovery"
        } else {
            "rescan"
        };
        let mut value = value;
        value["attempt"] = serde_json::json!(0);
        let raw = serde_json::to_string(&value).map_err(|e| e.to_string())?;
        if queue == "discovery" {
            self.discovery.push_front(raw);
        } else {
            self.rescan.push_front(raw);
        }
        Ok(QueueReplayResult {
            replayed: true,
            queue: queue.to_string(),
            message_id: message_id.to_string(),
        })
    }

    fn enqueue_memory<T: Serialize>(
        &mut self,
        queue: &str,
        serialized: String,
        envelope: QueueEnvelope<T>,
    ) -> Result<QueueEnqueueResult, String> {
        if self.seen_idempotency.contains(&envelope.idempotency_key) {
            return Ok(QueueEnqueueResult {
                accepted: true,
                deduplicated: true,
                queue: queue.to_string(),
                message_id: "deduplicated".to_string(),
                idempotency_key: envelope.idempotency_key,
                depth: self.memory_depth(queue),
                reason: None,
            });
        }
        if self.memory_depth(queue) >= QUEUE_DEPTH_LIMIT {
            return Ok(rejected_enqueue(queue, envelope.idempotency_key));
        }
        match queue {
            "discovery" => self.discovery.push_back(serialized),
            _ => self.rescan.push_back(serialized),
        }
        self.remember_idempotency(envelope.idempotency_key.clone());
        Ok(QueueEnqueueResult {
            accepted: true,
            deduplicated: false,
            queue: queue.to_string(),
            message_id: envelope.message_id,
            idempotency_key: envelope.idempotency_key,
            depth: self.memory_depth(queue),
            reason: None,
        })
    }

    fn enqueue_durable<T: Serialize>(
        &mut self,
        queue: &str,
        envelope: &QueueEnvelope<T>,
    ) -> Result<QueueEnqueueResult, String> {
        let serialized = serde_json::to_string(envelope)
            .map_err(|e| format!("failed to serialize queue message: {e}"))?;
        let conn = self.conn.as_mut().expect("durable queue connection");
        let tx = conn
            .transaction()
            .map_err(|e| format!("failed to start queue transaction: {e}"))?;
        let duplicate = tx
            .query_row(
                "SELECT 1 FROM queue_dedupe WHERE idempotency_key = ?1",
                params![envelope.idempotency_key],
                |_| Ok(()),
            )
            .optional()
            .map_err(|e| format!("failed to check queue deduplication: {e}"))?
            .is_some();
        if duplicate {
            let depth = count_pending(&tx, queue)?;
            tx.commit().map_err(|e| e.to_string())?;
            return Ok(QueueEnqueueResult {
                accepted: true,
                deduplicated: true,
                queue: queue.to_string(),
                message_id: "deduplicated".to_string(),
                idempotency_key: envelope.idempotency_key.clone(),
                depth,
                reason: None,
            });
        }

        let depth = count_pending(&tx, queue)?;
        if depth >= QUEUE_DEPTH_LIMIT {
            tx.commit().map_err(|e| e.to_string())?;
            return Ok(rejected_enqueue(queue, envelope.idempotency_key.clone()));
        }

        tx.execute(
            "INSERT INTO queue_messages
             (message_id, queue, message_type, job_id, idempotency_key,
              created_at_ms, available_at_ms, claimed_at_ms, attempt, trace_id,
              payload_json, state, last_error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, NULL, ?7, ?8, ?9, 'ready', NULL)",
            params![
                envelope.message_id,
                queue,
                envelope.message_type,
                envelope.job_id,
                envelope.idempotency_key,
                envelope.created_at_ms as i64,
                envelope.attempt as i64,
                envelope.trace_id,
                serialized,
            ],
        )
        .map_err(|e| format!("failed to insert queue message: {e}"))?;
        tx.execute(
            "INSERT INTO queue_dedupe (idempotency_key, seen_at_ms) VALUES (?1, ?2)",
            params![envelope.idempotency_key, envelope.created_at_ms as i64],
        )
        .map_err(|e| format!("failed to record queue deduplication: {e}"))?;
        tx.execute(
            "DELETE FROM queue_dedupe WHERE idempotency_key IN (
                 SELECT idempotency_key FROM queue_dedupe
                 ORDER BY seen_at_ms, idempotency_key
                 LIMIT -1 OFFSET ?1
             )",
            params![QUEUE_DEDUPE_LIMIT as i64],
        )
        .map_err(|e| format!("failed to trim queue deduplication: {e}"))?;
        tx.commit()
            .map_err(|e| format!("failed to commit queue enqueue: {e}"))?;

        Ok(QueueEnqueueResult {
            accepted: true,
            deduplicated: false,
            queue: queue.to_string(),
            message_id: envelope.message_id.clone(),
            idempotency_key: envelope.idempotency_key.clone(),
            depth: depth + 1,
            reason: None,
        })
    }

    fn claim_durable(&mut self, queue: &str) -> Result<Option<(String, String)>, String> {
        let conn = self.conn.as_mut().expect("durable queue connection");
        let tx = conn
            .transaction()
            .map_err(|e| format!("failed to start queue claim: {e}"))?;
        let row = {
            let mut statement = tx
                .prepare(
                    "SELECT message_id, payload_json FROM queue_messages
                     WHERE queue = ?1 AND state = 'ready' AND available_at_ms <= ?2
                     ORDER BY available_at_ms, created_at_ms, message_id LIMIT 1",
                )
                .map_err(|e| format!("failed to prepare queue claim: {e}"))?;
            statement
                .query_row(params![queue, now_ms() as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .optional()
                .map_err(|e| format!("failed to claim queue message: {e}"))?
        };
        let Some((message_id, raw)) = row else {
            tx.commit().map_err(|e| e.to_string())?;
            return Ok(None);
        };
        tx.execute(
            "UPDATE queue_messages SET state = 'in_flight', claimed_at_ms = ?2
             WHERE message_id = ?1 AND state = 'ready'",
            params![message_id, now_ms() as i64],
        )
        .map_err(|e| format!("failed to mark queue message in flight: {e}"))?;
        tx.commit()
            .map_err(|e| format!("failed to commit queue claim: {e}"))?;
        Ok(Some((message_id, raw)))
    }

    fn requeue_durable(
        &mut self,
        message_id: &str,
        raw: &str,
        error: Option<&str>,
        attempt: u32,
    ) -> Result<(), String> {
        let conn = self.conn.as_mut().expect("durable queue connection");
        let delay = if attempt == 0 {
            0
        } else {
            retry_delay_ms(attempt)
        };
        let changed = conn
            .execute(
                "UPDATE queue_messages
                 SET payload_json = ?2, attempt = ?3, state = 'ready',
                     available_at_ms = ?4, claimed_at_ms = NULL, last_error = COALESCE(?5, last_error)
                 WHERE message_id = ?1 AND state = 'in_flight'",
                params![message_id, raw, attempt as i64, now_ms().saturating_add(delay) as i64, error],
            )
            .map_err(|e| format!("failed to requeue queue message: {e}"))?;
        if changed == 0 {
            return Err(format!("queue message is not in flight: {message_id}"));
        }
        Ok(())
    }

    fn move_durable_to_dlq(&mut self, message_id: &str, error: Option<&str>) -> Result<(), String> {
        let conn = self.conn.as_mut().expect("durable queue connection");
        let changed = conn
            .execute(
                "UPDATE queue_messages
                 SET state = 'dlq', claimed_at_ms = NULL, last_error = COALESCE(?2, last_error)
                 WHERE message_id = ?1 AND state IN ('ready', 'in_flight')",
                params![message_id, error],
            )
            .map_err(|e| format!("failed to move queue message to DLQ: {e}"))?;
        if changed == 0 {
            return Err(format!("queue message not found: {message_id}"));
        }
        Ok(())
    }

    fn memory_depth(&self, queue: &str) -> usize {
        if queue == "discovery" {
            self.discovery.len()
        } else {
            self.rescan.len()
        }
    }

    fn remember_idempotency(&mut self, key: String) {
        if self.seen_idempotency.insert(key.clone()) {
            self.seen_idempotency_order.push_back(key);
        }
        while self.seen_idempotency_order.len() > QUEUE_DEDUPE_LIMIT {
            if let Some(oldest) = self.seen_idempotency_order.pop_front() {
                self.seen_idempotency.remove(&oldest);
            }
        }
    }
}

fn initialize_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS queue_meta (
             schema_version INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS queue_messages (
             message_id TEXT PRIMARY KEY,
             queue TEXT NOT NULL CHECK (queue IN ('discovery', 'rescan')),
             message_type TEXT NOT NULL,
             job_id TEXT NOT NULL,
             idempotency_key TEXT NOT NULL,
             created_at_ms INTEGER NOT NULL,
             available_at_ms INTEGER NOT NULL,
             claimed_at_ms INTEGER,
             attempt INTEGER NOT NULL,
             trace_id TEXT NOT NULL,
             payload_json TEXT NOT NULL,
             state TEXT NOT NULL CHECK (state IN ('ready', 'in_flight', 'dlq')),
             last_error TEXT
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_message_idempotency
             ON queue_messages(idempotency_key);
         CREATE INDEX IF NOT EXISTS idx_queue_ready
             ON queue_messages(queue, state, available_at_ms, created_at_ms);
         CREATE TABLE IF NOT EXISTS queue_dedupe (
             idempotency_key TEXT PRIMARY KEY,
             seen_at_ms INTEGER NOT NULL
         );",
    )
    .map_err(|e| format!("failed to create queue schema: {e}"))?;

    let current: Option<i64> = conn
        .query_row("SELECT schema_version FROM queue_meta LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()
        .map_err(|e| format!("failed to read queue schema version: {e}"))?;
    match current {
        None => conn
            .execute(
                "INSERT INTO queue_meta(schema_version) VALUES (?1)",
                params![QUEUE_SCHEMA_VERSION],
            )
            .map(|_| ())
            .map_err(|e| format!("failed to write queue schema version: {e}")),
        Some(version) if version == QUEUE_SCHEMA_VERSION => Ok(()),
        Some(version) if version < QUEUE_SCHEMA_VERSION => Err(format!(
            "queue database migration from schema {version} is not implemented"
        )),
        Some(version) => Err(format!(
            "queue database schema {version} is newer than supported schema {QUEUE_SCHEMA_VERSION}"
        )),
    }
}

fn durable_health(conn: &Connection) -> QueueHealth {
    let ready_discovery = count_state(conn, "discovery", "ready");
    let ready_rescan = count_state(conn, "rescan", "ready");
    let dlq = count_state(conn, "discovery", "dlq") + count_state(conn, "rescan", "dlq");
    QueueHealth {
        depths: QueueDepths {
            discovery: ready_discovery,
            rescan: ready_rescan,
            dlq,
            discovery_in_flight: count_state(conn, "discovery", "in_flight"),
            rescan_in_flight: count_state(conn, "rescan", "in_flight"),
        },
        dedupe_keys: conn
            .query_row("SELECT COUNT(*) FROM queue_dedupe", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(0)
            .max(0) as usize,
        capacity: QUEUE_DEPTH_LIMIT,
        next_discovery_at_ms: next_available(conn, "discovery"),
        next_rescan_at_ms: next_available(conn, "rescan"),
    }
}

fn count_state(conn: &Connection, queue: &str, state: &str) -> usize {
    conn.query_row(
        "SELECT COUNT(*) FROM queue_messages WHERE queue = ?1 AND state = ?2",
        params![queue, state],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0)
    .max(0) as usize
}

fn count_pending(conn: &Connection, queue: &str) -> Result<usize, String> {
    conn.query_row(
        "SELECT COUNT(*) FROM queue_messages WHERE queue = ?1 AND state IN ('ready', 'in_flight')",
        params![queue],
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count.max(0) as usize)
    .map_err(|e| format!("failed to count queue messages: {e}"))
}

fn next_available(conn: &Connection, queue: &str) -> Option<u64> {
    conn.query_row(
        "SELECT MIN(available_at_ms) FROM queue_messages WHERE queue = ?1 AND state = 'ready'",
        params![queue],
        |row| row.get::<_, Option<i64>>(0),
    )
    .ok()
    .flatten()
    .map(|value| value.max(0) as u64)
}

fn rejected_enqueue(queue: &str, idempotency_key: String) -> QueueEnqueueResult {
    QueueEnqueueResult {
        accepted: false,
        deduplicated: false,
        queue: queue.to_string(),
        message_id: String::new(),
        idempotency_key,
        depth: QUEUE_DEPTH_LIMIT,
        reason: Some("queue_full".to_string()),
    }
}

fn message_id_from_raw(raw: &str) -> Result<String, String> {
    serde_json::from_str::<serde_json::Value>(raw)
        .map_err(|e| format!("failed to parse queue message: {e}"))?
        .get("message_id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "queue message has no message_id".to_string())
}

fn retry_delay_ms(attempt: u32) -> u64 {
    let shift = attempt.saturating_sub(1).min(18);
    1_000_u64
        .saturating_mul(1_u64 << shift)
        .min(MAX_RETRY_DELAY_MS)
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
    use tempfile::TempDir;

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
            .unwrap();
        assert!(first.accepted);
        let second = q
            .enqueue_discovery(payload("scan-1"), "idem-a".to_string())
            .unwrap();
        assert!(second.deduplicated);
        assert_eq!(q.health().depths.discovery, 1);
    }

    #[test]
    fn retry_increments_attempt() {
        let mut q = QueueRuntime::new();
        q.enqueue_discovery(payload("scan-2"), "idem-b".to_string())
            .unwrap();
        let msg = q.dequeue_discovery().unwrap().unwrap();
        assert_eq!(msg.envelope.attempt, 0);
        q.requeue_discovery_retry(msg.envelope).unwrap();
        let after = q.dequeue_discovery().unwrap().unwrap();
        assert_eq!(after.envelope.attempt, 1);
    }

    #[test]
    fn malformed_messages_are_sent_to_dlq_instead_of_disappearing() {
        let mut q = QueueRuntime::new();
        q.discovery.push_back("not-json".to_string());
        let result = q.dequeue_discovery();
        assert!(result.is_err());
        assert_eq!(q.health().depths.discovery, 0);
        assert_eq!(q.health().depths.dlq, 1);
    }

    #[test]
    fn durable_queue_recovers_claimed_messages_after_restart() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("queue.db");
        let mut first = QueueRuntime::open(&path).unwrap();
        first
            .enqueue_discovery(payload("scan-3"), "idem-c".to_string())
            .unwrap();
        let claimed = first.dequeue_discovery().unwrap().unwrap();
        assert_eq!(first.health().depths.discovery_in_flight, 1);
        drop(first);

        let mut second = QueueRuntime::open(&path).unwrap();
        let recovered = second.dequeue_discovery().unwrap().unwrap();
        assert_eq!(recovered.envelope.message_id, claimed.envelope.message_id);
        second.acknowledge(&recovered.envelope.message_id).unwrap();
        assert_eq!(second.health().depths.discovery, 0);
        assert_eq!(second.health().depths.discovery_in_flight, 0);
    }

    #[test]
    fn durable_queue_applies_capacity_and_dlq_replay() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("queue.db");
        let mut q = QueueRuntime::open(&path).unwrap();
        for index in 0..QUEUE_DEPTH_LIMIT {
            let result =
                q.enqueue_discovery(payload(&format!("scan-{index}")), format!("idem-{index}"));
            assert!(result.unwrap().accepted);
        }
        let full = q
            .enqueue_discovery(payload("full"), "idem-full".to_string())
            .unwrap();
        assert!(!full.accepted);
        assert_eq!(full.reason.as_deref(), Some("queue_full"));

        let claimed = q.dequeue_discovery().unwrap().unwrap();
        q.move_discovery_to_dlq_with_error(claimed.raw, Some("test failure"))
            .unwrap();
        let listed = q.list_dlq(10).unwrap();
        assert_eq!(listed.len(), 1);
        let replayed = q.replay_dlq(&listed[0].message_id).unwrap();
        assert!(replayed.replayed);
        assert_eq!(q.health().depths.dlq, 0);
    }
}
