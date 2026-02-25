use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension, Transaction};

pub const SCHEMA_VERSION: i64 = 3;

#[derive(Debug, Clone)]
pub struct ScanSessionRow {
    pub scan_id: String,
    pub video: String,
    pub yolo_model: String,
    pub face_model: String,
    pub status: String,
    pub expected_count: Option<i64>,
    pub review_ready: bool,
    pub selected_identity_id: Option<i64>,
    pub selected_anchor_x: Option<f32>,
    pub selected_anchor_y: Option<f32>,
    pub updated_at_ms: u64,
    pub candidates_json: String,
    pub duplicates_json: String,
    pub excluded_identity_ids_json: String,
    pub accepted_low_confidence_ids_json: String,
    pub resolved_duplicate_keys_json: String,
    pub pending_split_ids_json: String,
    pub pending_split_count: i64,
    pub last_blockers_json: String,
}

#[derive(Debug, Clone)]
pub struct ScanSessionEventRow {
    pub scan_id: String,
    pub at_ms: u64,
    pub action: String,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct ScanStoreRows {
    pub next_id: u64,
    pub sessions: Vec<ScanSessionRow>,
    pub events: Vec<ScanSessionEventRow>,
}

#[derive(Debug, Clone)]
pub struct ScanSummaryQueryRow {
    pub scan_id: String,
    pub video: String,
    pub status: String,
    pub review_ready: bool,
    pub selected_identity_id: Option<i64>,
    pub pending_split_count: i64,
    pub updated_at_ms: u64,
    pub event_count: u64,
}

#[derive(Debug, Clone)]
pub struct ScanEventQueryRow {
    pub event_id: u64,
    pub at_ms: u64,
    pub action: String,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub schema_version: i64,
    pub session_count: u64,
    pub event_count: u64,
}

#[derive(Debug, Clone)]
pub struct StorageMaintenanceResult {
    pub deleted_sessions: u64,
    pub deleted_events: u64,
    pub vacuum_ran: bool,
}

pub fn scan_store_db_path() -> PathBuf {
    let mut base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    base.push(".focus-lock");
    base.push("scan_sessions.db");
    base
}

pub fn scan_store_json_path() -> PathBuf {
    let mut base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    base.push(".focus-lock");
    base.push("scan_sessions.json");
    base
}

pub fn load_scan_rows(db_path: &Path) -> Result<Option<ScanStoreRows>, String> {
    let conn = open_db(db_path)?;
    migrate(&conn)?;

    let next_id: Option<u64> = conn
        .query_row(
            "SELECT value_integer FROM app_state WHERE key = 'next_id'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed to read app_state next_id: {e}"))?;

    let Some(next_id) = next_id else {
        return Ok(None);
    };

    let mut stmt = conn
        .prepare(
            "SELECT
               scan_id, video, yolo_model, face_model, status, expected_count,
               review_ready, selected_identity_id, selected_anchor_x, selected_anchor_y,
               updated_at_ms, candidates_json, duplicates_json,
               excluded_identity_ids_json, accepted_low_confidence_ids_json,
               resolved_duplicate_keys_json, pending_split_ids_json,
               pending_split_count, last_blockers_json
             FROM scan_sessions",
        )
        .map_err(|e| format!("failed to prepare sessions query: {e}"))?;

    let sessions = stmt
        .query_map([], |row| {
            Ok(ScanSessionRow {
                scan_id: row.get(0)?,
                video: row.get(1)?,
                yolo_model: row.get(2)?,
                face_model: row.get(3)?,
                status: row.get(4)?,
                expected_count: row.get(5)?,
                review_ready: row.get::<_, i64>(6)? != 0,
                selected_identity_id: row.get(7)?,
                selected_anchor_x: row.get(8)?,
                selected_anchor_y: row.get(9)?,
                updated_at_ms: row.get(10)?,
                candidates_json: row.get(11)?,
                duplicates_json: row.get(12)?,
                excluded_identity_ids_json: row.get(13)?,
                accepted_low_confidence_ids_json: row.get(14)?,
                resolved_duplicate_keys_json: row.get(15)?,
                pending_split_ids_json: row.get(16)?,
                pending_split_count: row.get(17)?,
                last_blockers_json: row.get(18)?,
            })
        })
        .map_err(|e| format!("failed to map sessions rows: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("failed to collect sessions rows: {e}"))?;

    let mut stmt = conn
        .prepare(
            "SELECT scan_id, at_ms, action, details
             FROM scan_session_events
             ORDER BY at_ms ASC, id ASC",
        )
        .map_err(|e| format!("failed to prepare events query: {e}"))?;

    let events = stmt
        .query_map([], |row| {
            Ok(ScanSessionEventRow {
                scan_id: row.get(0)?,
                at_ms: row.get(1)?,
                action: row.get(2)?,
                details: row.get(3)?,
            })
        })
        .map_err(|e| format!("failed to map events rows: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("failed to collect events rows: {e}"))?;

    Ok(Some(ScanStoreRows {
        next_id,
        sessions,
        events,
    }))
}

pub fn save_scan_rows(db_path: &Path, rows: &ScanStoreRows) -> Result<(), String> {
    let mut conn = open_db(db_path)?;
    migrate(&conn)?;

    let tx = conn
        .transaction()
        .map_err(|e| format!("failed to start sqlite transaction: {e}"))?;
    upsert_next_id(&tx, rows.next_id)?;

    tx.execute("DELETE FROM scan_session_events", [])
        .map_err(|e| format!("failed to clear events table: {e}"))?;
    tx.execute("DELETE FROM scan_sessions", [])
        .map_err(|e| format!("failed to clear sessions table: {e}"))?;

    for session in &rows.sessions {
        tx.execute(
            "INSERT INTO scan_sessions (
               scan_id, video, yolo_model, face_model, status, expected_count,
               review_ready, selected_identity_id, selected_anchor_x, selected_anchor_y,
               updated_at_ms, candidates_json, duplicates_json,
               excluded_identity_ids_json, accepted_low_confidence_ids_json,
               resolved_duplicate_keys_json, pending_split_ids_json,
               pending_split_count, last_blockers_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                session.scan_id,
                session.video,
                session.yolo_model,
                session.face_model,
                session.status,
                session.expected_count,
                if session.review_ready { 1_i64 } else { 0_i64 },
                session.selected_identity_id,
                session.selected_anchor_x,
                session.selected_anchor_y,
                session.updated_at_ms,
                session.candidates_json,
                session.duplicates_json,
                session.excluded_identity_ids_json,
                session.accepted_low_confidence_ids_json,
                session.resolved_duplicate_keys_json,
                session.pending_split_ids_json,
                session.pending_split_count,
                session.last_blockers_json,
            ],
        )
        .map_err(|e| format!("failed to insert session row: {e}"))?;
    }

    for event in &rows.events {
        tx.execute(
            "INSERT INTO scan_session_events (scan_id, at_ms, action, details)
             VALUES (?1, ?2, ?3, ?4)",
            params![event.scan_id, event.at_ms, event.action, event.details],
        )
        .map_err(|e| format!("failed to insert event row: {e}"))?;
    }

    tx.commit()
        .map_err(|e| format!("failed to commit sqlite transaction: {e}"))
}

pub fn query_scan_summaries(
    db_path: &Path,
    limit: u32,
    offset: u32,
    status: Option<&str>,
    cursor_updated_at_ms: Option<u64>,
    cursor_scan_id: Option<&str>,
) -> Result<Vec<ScanSummaryQueryRow>, String> {
    let conn = open_db(db_path)?;
    migrate(&conn)?;
    let base = "SELECT
        s.scan_id,
        s.video,
        s.status,
        s.review_ready,
        s.selected_identity_id,
        s.pending_split_count,
        s.updated_at_ms,
        COALESCE(e.event_count, 0) AS event_count
      FROM scan_sessions s
      LEFT JOIN (
        SELECT scan_id, COUNT(*) AS event_count
        FROM scan_session_events
        GROUP BY scan_id
      ) e ON e.scan_id = s.scan_id";

    let mut out = Vec::new();
    let has_cursor = cursor_updated_at_ms.is_some() && cursor_scan_id.is_some();
    if let Some(status_filter) = status {
        if has_cursor {
            let sql = format!(
                "{} WHERE s.status = ?1
                   AND (s.updated_at_ms < ?2 OR (s.updated_at_ms = ?2 AND s.scan_id < ?3))
                   ORDER BY s.updated_at_ms DESC, s.scan_id DESC
                   LIMIT ?4 OFFSET ?5",
                base
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("failed to prepare scan summaries query: {e}"))?;
            let mapped = stmt
                .query_map(
                    params![
                        status_filter,
                        cursor_updated_at_ms.unwrap_or(0) as i64,
                        cursor_scan_id.unwrap_or_default(),
                        limit as i64,
                        offset as i64
                    ],
                    |row| {
                        Ok(ScanSummaryQueryRow {
                            scan_id: row.get(0)?,
                            video: row.get(1)?,
                            status: row.get(2)?,
                            review_ready: row.get::<_, i64>(3)? != 0,
                            selected_identity_id: row.get(4)?,
                            pending_split_count: row.get(5)?,
                            updated_at_ms: row.get(6)?,
                            event_count: row.get(7)?,
                        })
                    },
                )
                .map_err(|e| format!("failed to map scan summaries rows: {e}"))?;
            for row in mapped {
                out.push(row.map_err(|e| format!("failed to read scan summary row: {e}"))?);
            }
            return Ok(out);
        }
        let sql = format!(
            "{} WHERE s.status = ?1 ORDER BY s.updated_at_ms DESC, s.scan_id DESC LIMIT ?2 OFFSET ?3",
            base
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed to prepare scan summaries query: {e}"))?;
        let mapped = stmt
            .query_map(params![status_filter, limit as i64, offset as i64], |row| {
                Ok(ScanSummaryQueryRow {
                    scan_id: row.get(0)?,
                    video: row.get(1)?,
                    status: row.get(2)?,
                    review_ready: row.get::<_, i64>(3)? != 0,
                    selected_identity_id: row.get(4)?,
                    pending_split_count: row.get(5)?,
                    updated_at_ms: row.get(6)?,
                    event_count: row.get(7)?,
                })
            })
            .map_err(|e| format!("failed to map scan summaries rows: {e}"))?;
        for row in mapped {
            out.push(row.map_err(|e| format!("failed to read scan summary row: {e}"))?);
        }
    } else {
        if has_cursor {
            let sql = format!(
                "{} WHERE (s.updated_at_ms < ?1 OR (s.updated_at_ms = ?1 AND s.scan_id < ?2))
                   ORDER BY s.updated_at_ms DESC, s.scan_id DESC
                   LIMIT ?3 OFFSET ?4",
                base
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("failed to prepare scan summaries query: {e}"))?;
            let mapped = stmt
                .query_map(
                    params![
                        cursor_updated_at_ms.unwrap_or(0) as i64,
                        cursor_scan_id.unwrap_or_default(),
                        limit as i64,
                        offset as i64
                    ],
                    |row| {
                        Ok(ScanSummaryQueryRow {
                            scan_id: row.get(0)?,
                            video: row.get(1)?,
                            status: row.get(2)?,
                            review_ready: row.get::<_, i64>(3)? != 0,
                            selected_identity_id: row.get(4)?,
                            pending_split_count: row.get(5)?,
                            updated_at_ms: row.get(6)?,
                            event_count: row.get(7)?,
                        })
                    },
                )
                .map_err(|e| format!("failed to map scan summaries rows: {e}"))?;
            for row in mapped {
                out.push(row.map_err(|e| format!("failed to read scan summary row: {e}"))?);
            }
            return Ok(out);
        }
        let sql = format!(
            "{} ORDER BY s.updated_at_ms DESC, s.scan_id DESC LIMIT ?1 OFFSET ?2",
            base
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed to prepare scan summaries query: {e}"))?;
        let mapped = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                Ok(ScanSummaryQueryRow {
                    scan_id: row.get(0)?,
                    video: row.get(1)?,
                    status: row.get(2)?,
                    review_ready: row.get::<_, i64>(3)? != 0,
                    selected_identity_id: row.get(4)?,
                    pending_split_count: row.get(5)?,
                    updated_at_ms: row.get(6)?,
                    event_count: row.get(7)?,
                })
            })
            .map_err(|e| format!("failed to map scan summaries rows: {e}"))?;
        for row in mapped {
            out.push(row.map_err(|e| format!("failed to read scan summary row: {e}"))?);
        }
    }
    Ok(out)
}

pub fn query_scan_events(
    db_path: &Path,
    scan_id: &str,
    limit: u32,
    offset: u32,
    action_contains: Option<&str>,
    since_ms: Option<u64>,
    until_ms: Option<u64>,
    cursor_event_id: Option<u64>,
) -> Result<Vec<ScanEventQueryRow>, String> {
    let conn = open_db(db_path)?;
    migrate(&conn)?;
    let action_pattern = action_contains.map(|s| format!("%{}%", s));
    let mut stmt = conn
        .prepare(
            "SELECT id, at_ms, action, details
             FROM scan_session_events
             WHERE scan_id = ?1
               AND (?2 IS NULL OR action LIKE ?2)
               AND (?3 IS NULL OR at_ms >= ?3)
               AND (?4 IS NULL OR at_ms <= ?4)
               AND (?5 IS NULL OR id < ?5)
             ORDER BY id DESC
             LIMIT ?6 OFFSET ?7",
        )
        .map_err(|e| format!("failed to prepare scan events query: {e}"))?;
    let mapped = stmt
        .query_map(
            params![
                scan_id,
                action_pattern,
                since_ms.map(|v| v as i64),
                until_ms.map(|v| v as i64),
                cursor_event_id.map(|v| v as i64),
                limit as i64,
                offset as i64
            ],
            |row| {
                Ok(ScanEventQueryRow {
                    event_id: row.get(0)?,
                    at_ms: row.get(1)?,
                    action: row.get(2)?,
                    details: row.get(3)?,
                })
            },
        )
        .map_err(|e| format!("failed to map scan events rows: {e}"))?;
    let mut out = Vec::new();
    for row in mapped {
        out.push(row.map_err(|e| format!("failed to read scan event row: {e}"))?);
    }
    Ok(out)
}

pub fn get_storage_stats(db_path: &Path) -> Result<StorageStats, String> {
    let conn = open_db(db_path)?;
    migrate(&conn)?;
    let schema_version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| format!("failed to read user_version: {e}"))?;
    let session_count: u64 = conn
        .query_row("SELECT COUNT(*) FROM scan_sessions", [], |row| row.get(0))
        .map_err(|e| format!("failed to count scan_sessions: {e}"))?;
    let event_count: u64 = conn
        .query_row("SELECT COUNT(*) FROM scan_session_events", [], |row| {
            row.get(0)
        })
        .map_err(|e| format!("failed to count scan_session_events: {e}"))?;
    Ok(StorageStats {
        schema_version,
        session_count,
        event_count,
    })
}

pub fn run_storage_maintenance(
    db_path: &Path,
    max_session_age_ms: u64,
    max_events_per_scan: u32,
    vacuum: bool,
) -> Result<StorageMaintenanceResult, String> {
    let mut conn = open_db(db_path)?;
    migrate(&conn)?;

    let cutoff = epoch_ms().saturating_sub(max_session_age_ms) as i64;
    let tx = conn
        .transaction()
        .map_err(|e| format!("failed to start maintenance transaction: {e}"))?;

    let deleted_sessions =
        tx.execute(
            "DELETE FROM scan_sessions WHERE updated_at_ms < ?1",
            params![cutoff],
        )
        .map_err(|e| format!("failed to delete old scan sessions: {e}"))? as u64;

    let deleted_events = tx
        .execute(
            "DELETE FROM scan_session_events
             WHERE id IN (
               SELECT id FROM (
                 SELECT
                   id,
                   ROW_NUMBER() OVER (
                     PARTITION BY scan_id
                     ORDER BY at_ms DESC, id DESC
                   ) AS rn
                 FROM scan_session_events
               ) ranked
               WHERE rn > ?1
             )",
            params![max_events_per_scan as i64],
        )
        .map_err(|e| format!("failed to trim scan events: {e}"))? as u64;

    tx.commit()
        .map_err(|e| format!("failed to commit maintenance transaction: {e}"))?;

    if vacuum {
        conn.execute_batch("VACUUM")
            .map_err(|e| format!("failed to vacuum sqlite db: {e}"))?;
    }

    Ok(StorageMaintenanceResult {
        deleted_sessions,
        deleted_events,
        vacuum_ran: vacuum,
    })
}

fn upsert_next_id(tx: &Transaction<'_>, next_id: u64) -> Result<(), String> {
    tx.execute(
        "INSERT INTO app_state (key, value_integer) VALUES ('next_id', ?1)
         ON CONFLICT(key) DO UPDATE SET value_integer = excluded.value_integer",
        params![next_id],
    )
    .map_err(|e| format!("failed to upsert app_state next_id: {e}"))?;
    Ok(())
}

fn open_db(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("failed to create db dir: {e}"))?;
    }
    Connection::open(path).map_err(|e| format!("failed to open sqlite db: {e}"))
}

fn migrate(conn: &Connection) -> Result<(), String> {
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|e| format!("failed to enable foreign_keys: {e}"))?;

    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| format!("failed to read sqlite user_version: {e}"))?;

    if version < 1 {
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS app_state (
               key TEXT PRIMARY KEY,
               value_integer INTEGER,
               value_text TEXT
             );
             PRAGMA user_version = 1;
             COMMIT;",
        )
        .map_err(|e| format!("failed migration v1: {e}"))?;
    }

    if version < 2 {
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS scan_sessions (
               scan_id TEXT PRIMARY KEY,
               video TEXT NOT NULL,
               yolo_model TEXT NOT NULL,
               face_model TEXT NOT NULL,
               status TEXT NOT NULL,
               expected_count INTEGER,
               review_ready INTEGER NOT NULL,
               selected_identity_id INTEGER,
               selected_anchor_x REAL,
               selected_anchor_y REAL,
               updated_at_ms INTEGER NOT NULL,
               candidates_json TEXT NOT NULL,
               duplicates_json TEXT NOT NULL,
               excluded_identity_ids_json TEXT NOT NULL,
                accepted_low_confidence_ids_json TEXT NOT NULL,
                resolved_duplicate_keys_json TEXT NOT NULL,
                pending_split_ids_json TEXT NOT NULL,
                pending_split_count INTEGER NOT NULL DEFAULT 0,
                last_blockers_json TEXT NOT NULL
              );
             CREATE TABLE IF NOT EXISTS scan_session_events (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               scan_id TEXT NOT NULL,
               at_ms INTEGER NOT NULL,
               action TEXT NOT NULL,
               details TEXT NOT NULL,
               FOREIGN KEY (scan_id) REFERENCES scan_sessions(scan_id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_scan_sessions_updated_at
               ON scan_sessions(updated_at_ms DESC);
             CREATE INDEX IF NOT EXISTS idx_scan_events_scan_id_at
               ON scan_session_events(scan_id, at_ms DESC);
             PRAGMA user_version = 2;
             COMMIT;",
        )
        .map_err(|e| format!("failed migration v2: {e}"))?;
    }

    if version < 3 {
        let alter_result = conn.execute(
            "ALTER TABLE scan_sessions ADD COLUMN pending_split_count INTEGER NOT NULL DEFAULT 0",
            [],
        );
        if let Err(e) = alter_result
            && !e.to_string().contains("duplicate column name")
        {
            return Err(format!("failed migration v3 alter: {e}"));
        }
        conn.execute(
            "UPDATE scan_sessions
             SET pending_split_count = 0
             WHERE pending_split_count IS NULL",
            [],
        )
        .map_err(|e| format!("failed migration v3 backfill: {e}"))?;
        conn.execute_batch("PRAGMA user_version = 3;")
            .map_err(|e| format!("failed to set user_version v3: {e}"))?;
    }

    let final_version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| format!("failed to re-read sqlite user_version: {e}"))?;
    if final_version != SCHEMA_VERSION {
        return Err(format!(
            "unexpected schema version: expected {}, got {}",
            SCHEMA_VERSION, final_version
        ));
    }
    Ok(())
}

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("focus-lock-test-{name}-{}.db", epoch_ms()));
        p
    }

    #[test]
    fn migration_creates_expected_schema_version() {
        let path = temp_db_path("schema-version");
        let stats = get_storage_stats(&path).expect("stats should load after migration");
        assert_eq!(stats.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn maintenance_runs_without_error() {
        let path = temp_db_path("maintenance");
        let rows = ScanStoreRows {
            next_id: 1,
            sessions: Vec::new(),
            events: Vec::new(),
        };
        save_scan_rows(&path, &rows).expect("save should work");
        let result =
            run_storage_maintenance(&path, 86_400_000, 50, false).expect("maintenance should work");
        assert_eq!(result.deleted_sessions, 0);
    }

    #[test]
    fn query_and_maintenance_trim_work() {
        let path = temp_db_path("query-maintenance");
        let now = epoch_ms();
        let rows = ScanStoreRows {
            next_id: 3,
            sessions: vec![
                ScanSessionRow {
                    scan_id: "scan-1".to_string(),
                    video: "a.mp4".to_string(),
                    yolo_model: "y.onnx".to_string(),
                    face_model: "f.onnx".to_string(),
                    status: "proposed".to_string(),
                    expected_count: Some(5),
                    review_ready: false,
                    selected_identity_id: None,
                    selected_anchor_x: None,
                    selected_anchor_y: None,
                    updated_at_ms: now.saturating_sub(10_000),
                    candidates_json: "[]".to_string(),
                    duplicates_json: "[]".to_string(),
                    excluded_identity_ids_json: "[]".to_string(),
                    accepted_low_confidence_ids_json: "[]".to_string(),
                    resolved_duplicate_keys_json: "[]".to_string(),
                    pending_split_ids_json: "[]".to_string(),
                    pending_split_count: 0,
                    last_blockers_json: "[]".to_string(),
                },
                ScanSessionRow {
                    scan_id: "scan-2".to_string(),
                    video: "b.mp4".to_string(),
                    yolo_model: "y.onnx".to_string(),
                    face_model: "f.onnx".to_string(),
                    status: "validated".to_string(),
                    expected_count: Some(5),
                    review_ready: true,
                    selected_identity_id: Some(2),
                    selected_anchor_x: Some(12.3),
                    selected_anchor_y: Some(22.1),
                    updated_at_ms: now,
                    candidates_json: "[]".to_string(),
                    duplicates_json: "[]".to_string(),
                    excluded_identity_ids_json: "[]".to_string(),
                    accepted_low_confidence_ids_json: "[]".to_string(),
                    resolved_duplicate_keys_json: "[]".to_string(),
                    pending_split_ids_json: "[1,2]".to_string(),
                    pending_split_count: 2,
                    last_blockers_json: "[]".to_string(),
                },
            ],
            events: vec![
                ScanSessionEventRow {
                    scan_id: "scan-2".to_string(),
                    at_ms: now.saturating_sub(2),
                    action: "a".to_string(),
                    details: "d1".to_string(),
                },
                ScanSessionEventRow {
                    scan_id: "scan-2".to_string(),
                    at_ms: now.saturating_sub(1),
                    action: "b".to_string(),
                    details: "d2".to_string(),
                },
                ScanSessionEventRow {
                    scan_id: "scan-2".to_string(),
                    at_ms: now,
                    action: "c".to_string(),
                    details: "d3".to_string(),
                },
            ],
        };
        save_scan_rows(&path, &rows).expect("save should work");

        let all = query_scan_summaries(&path, 10, 0, None, None, None).expect("query summaries");
        assert_eq!(all.len(), 2);
        let first_page =
            query_scan_summaries(&path, 1, 0, None, None, None).expect("first page summaries");
        assert_eq!(first_page.len(), 1);
        let second_page = query_scan_summaries(
            &path,
            10,
            0,
            None,
            Some(first_page[0].updated_at_ms),
            Some(&first_page[0].scan_id),
        )
        .expect("second page summaries");
        assert_eq!(second_page.len(), 1);
        assert_eq!(second_page[0].scan_id, "scan-1");

        let only_validated = query_scan_summaries(&path, 10, 0, Some("validated"), None, None)
            .expect("filtered query");
        assert_eq!(only_validated.len(), 1);
        assert_eq!(only_validated[0].scan_id, "scan-2");
        assert_eq!(only_validated[0].event_count, 3);

        let page = query_scan_events(&path, "scan-2", 2, 0, None, None, None, None)
            .expect("query events page");
        assert_eq!(page.len(), 2);

        let filtered = query_scan_events(&path, "scan-2", 10, 0, Some("b"), None, None, None)
            .expect("filtered events");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].action, "b");

        let since_filtered = query_scan_events(
            &path,
            "scan-2",
            10,
            0,
            None,
            Some(now.saturating_sub(1)),
            None,
            None,
        )
        .expect("since filtered events");
        assert_eq!(since_filtered.len(), 2);

        let until_filtered = query_scan_events(
            &path,
            "scan-2",
            10,
            0,
            None,
            None,
            Some(now.saturating_sub(1)),
            None,
        )
        .expect("until filtered events");
        assert_eq!(until_filtered.len(), 2);

        let cursor_filtered = query_scan_events(
            &path,
            "scan-2",
            10,
            0,
            None,
            None,
            None,
            Some(page[page.len() - 1].event_id),
        )
        .expect("cursor events");
        assert!(!cursor_filtered.is_empty());

        let maintenance =
            run_storage_maintenance(&path, 2_000, 2, false).expect("maintenance should succeed");
        assert_eq!(maintenance.deleted_sessions, 1);
        assert!(maintenance.deleted_events >= 1);
    }
}
