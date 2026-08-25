//! # langchart-memory-redb
//!
//! Embedded [`MemoryAdapter`] backed by [redb](https://github.com/cberner/redb).
//!
//! Supports `Keyword` (substring scan), `Key` (exact lookup), and
//! `Semantic` (falls back to keyword with a warning — vector search requires a
//! dedicated adapter such as `langchart-memory-qdrant`).
//!
//! All records are scoped and stored as JSON blobs. The store is suitable for
//! single-process embedded use.
//!
//! ## Usage
//!
//! ```text
//! let mem = RedbMemoryAdapter::open("./langchart-memory.redb")?;
//! ```

use async_trait::async_trait;
use langchart_adapters::memory::{
    MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope,
    QueryMode,
};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tracing::warn;
use ulid::Ulid;

// Table: MemoryId (str) → JSON StoredRecord (str)
const MEMORY: TableDefinition<&str, &str> = TableDefinition::new("memory");

// ── Internal record ───────────────────────────────────────────────────────────

/// What gets written to disk (id + record together for easy round-trip).
#[derive(Serialize, Deserialize)]
struct StoredRecord {
    id: String,
    record: MemoryRecord,
}

// ── Adapter ───────────────────────────────────────────────────────────────────

/// A [`MemoryAdapter`] backed by an embedded redb database.
#[derive(Clone)]
pub struct RedbMemoryAdapter {
    db: Arc<Mutex<Database>>,
}

impl RedbMemoryAdapter {
    /// Open or create a redb database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let db = Database::create(path.as_ref()).map_err(|e| StoreError::Open(e.to_string()))?;

        // Ensure table exists.
        let tx = db
            .begin_write()
            .map_err(|e| StoreError::Open(e.to_string()))?;
        tx.open_table(MEMORY)
            .map_err(|e| StoreError::Open(e.to_string()))?;
        tx.commit().map_err(|e| StoreError::Open(e.to_string()))?;

        Ok(Self {
            db: Arc::new(Mutex::new(db)),
        })
    }

    fn scope_key(scope: &MemoryScope) -> String {
        match scope {
            MemoryScope::Run(id) => format!("run:{}", id.0),
            MemoryScope::Workflow(id) => format!("workflow:{}", id.0),
            MemoryScope::Agent(id) => format!("agent:{}", id.0),
            MemoryScope::Global => "global".into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("failed to open memory store: {0}")]
    Open(String),
}

// ── MemoryAdapter impl ────────────────────────────────────────────────────────

#[async_trait]
impl MemoryAdapter for RedbMemoryAdapter {
    async fn store(&self, record: MemoryRecord) -> Result<MemoryId, MemoryError> {
        let id = MemoryId(Ulid::generate().to_string());
        let stored = StoredRecord {
            id: id.0.clone(),
            record,
        };
        let value =
            serde_json::to_string(&stored).map_err(|e| MemoryError::Store(e.to_string()))?;

        let db = self.db.lock().unwrap();
        let tx = db
            .begin_write()
            .map_err(|e| MemoryError::Store(e.to_string()))?;
        {
            let mut table = tx
                .open_table(MEMORY)
                .map_err(|e| MemoryError::Store(e.to_string()))?;
            table
                .insert(id.0.as_str(), value.as_str())
                .map_err(|e| MemoryError::Store(e.to_string()))?;
        }
        tx.commit().map_err(|e| MemoryError::Store(e.to_string()))?;

        Ok(id)
    }

    async fn search(&self, query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
        let scope_prefix = Self::scope_key(&query.scope);

        // For Semantic queries, fall back to keyword with a warning.
        let search_text = match &query.mode {
            QueryMode::Keyword { text } => Some(text.to_lowercase()),
            QueryMode::Semantic { text } => {
                warn!("RedbMemoryAdapter: Semantic search not supported; falling back to keyword");
                Some(text.to_lowercase())
            }
            QueryMode::Key { key } => {
                // Delegate to get-by-key scan.
                return self.search_by_key(&scope_prefix, key, query.limit).await;
            }
        };

        let db = self.db.lock().unwrap();
        let tx = db
            .begin_read()
            .map_err(|e| MemoryError::Store(e.to_string()))?;
        let table = tx
            .open_table(MEMORY)
            .map_err(|e| MemoryError::Store(e.to_string()))?;

        let mut results = Vec::new();
        for entry in table
            .iter()
            .map_err(|e| MemoryError::Store(e.to_string()))?
        {
            let (_, value_guard) = entry.map_err(|e| MemoryError::Store(e.to_string()))?;
            let stored: StoredRecord = match serde_json::from_str(value_guard.value()) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Filter by scope.
            if Self::scope_key(&stored.record.scope) != scope_prefix {
                continue;
            }

            // Filter by text (substring match on content).
            if let Some(ref text) = search_text
                && !stored.record.content.to_lowercase().contains(text.as_str())
            {
                continue;
            }

            results.push(MemoryResult {
                id: MemoryId(stored.id),
                record: stored.record,
                score: None,
            });

            if results.len() >= query.limit as usize {
                break;
            }
        }

        Ok(results)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        let db = self.db.lock().unwrap();
        let tx = db
            .begin_read()
            .map_err(|e| MemoryError::Store(e.to_string()))?;
        let table = tx
            .open_table(MEMORY)
            .map_err(|e| MemoryError::Store(e.to_string()))?;

        match table
            .get(id.0.as_str())
            .map_err(|e| MemoryError::Store(e.to_string()))?
        {
            Some(guard) => {
                let stored: StoredRecord = serde_json::from_str(guard.value())
                    .map_err(|e| MemoryError::Store(e.to_string()))?;
                Ok(Some(stored.record))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &MemoryId) -> Result<(), MemoryError> {
        let db = self.db.lock().unwrap();
        let tx = db
            .begin_write()
            .map_err(|e| MemoryError::Store(e.to_string()))?;
        {
            let mut table = tx
                .open_table(MEMORY)
                .map_err(|e| MemoryError::Store(e.to_string()))?;
            table
                .remove(id.0.as_str())
                .map_err(|e| MemoryError::Store(e.to_string()))?;
        }
        tx.commit().map_err(|e| MemoryError::Store(e.to_string()))?;
        Ok(())
    }
}

impl RedbMemoryAdapter {
    async fn search_by_key(
        &self,
        scope_prefix: &str,
        key: &str,
        limit: u32,
    ) -> Result<Vec<MemoryResult>, MemoryError> {
        let db = self.db.lock().unwrap();
        let tx = db
            .begin_read()
            .map_err(|e| MemoryError::Store(e.to_string()))?;
        let table = tx
            .open_table(MEMORY)
            .map_err(|e| MemoryError::Store(e.to_string()))?;

        let mut results = Vec::new();
        for entry in table
            .iter()
            .map_err(|e| MemoryError::Store(e.to_string()))?
        {
            let (_, value_guard) = entry.map_err(|e| MemoryError::Store(e.to_string()))?;
            let stored: StoredRecord = match serde_json::from_str(value_guard.value()) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if Self::scope_key(&stored.record.scope) != scope_prefix {
                continue;
            }
            if stored.record.key.as_deref() == Some(key) {
                results.push(MemoryResult {
                    id: MemoryId(stored.id),
                    record: stored.record,
                    score: Some(1.0),
                });
                if results.len() >= limit as usize {
                    break;
                }
            }
        }
        Ok(results)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::memory::{MemoryScope, QueryMode};
    use langchart_model::id::RunId;
    use tempfile::NamedTempFile;

    fn run_scope() -> MemoryScope {
        MemoryScope::Run(RunId::new("test-run"))
    }

    fn make_record(content: &str) -> MemoryRecord {
        MemoryRecord {
            scope: run_scope(),
            key: None,
            content: content.into(),
            embedding: None,
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn store_and_get_round_trip() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        let id = mem.store(make_record("hello world")).await.unwrap();
        let rec = mem.get(&id).await.unwrap();
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().content, "hello world");
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        let result = mem.get(&MemoryId("no-such-id".into())).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn delete_removes_record() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        let id = mem.store(make_record("to be deleted")).await.unwrap();
        mem.delete(&id).await.unwrap();
        assert!(mem.get(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn keyword_search_finds_matches() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        mem.store(make_record("the quick brown fox")).await.unwrap();
        mem.store(make_record("lazy dog")).await.unwrap();
        mem.store(make_record("another quick entry")).await.unwrap();

        let results = mem
            .search(MemoryQuery {
                scope: run_scope(),
                mode: QueryMode::Keyword {
                    text: "quick".into(),
                },
                limit: 10,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.record.content.contains("quick")));
    }

    #[tokio::test]
    async fn keyword_search_scope_isolation() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        // Store in run scope.
        mem.store(make_record("run-scoped data")).await.unwrap();
        // Store in global scope.
        mem.store(MemoryRecord {
            scope: MemoryScope::Global,
            key: None,
            content: "global data".into(),
            embedding: None,
            metadata: serde_json::Value::Null,
        })
        .await
        .unwrap();

        // Query only run scope — should not see global.
        let results = mem
            .search(MemoryQuery {
                scope: run_scope(),
                mode: QueryMode::Keyword {
                    text: "data".into(),
                },
                limit: 10,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "run-scoped data");
    }

    #[tokio::test]
    async fn key_search_exact_match() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        mem.store(MemoryRecord {
            scope: run_scope(),
            key: Some("fact:capital".into()),
            content: "Paris".into(),
            embedding: None,
            metadata: serde_json::Value::Null,
        })
        .await
        .unwrap();
        mem.store(make_record("noise")).await.unwrap();

        let results = mem
            .search(MemoryQuery {
                scope: run_scope(),
                mode: QueryMode::Key {
                    key: "fact:capital".into(),
                },
                limit: 10,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "Paris");
        assert_eq!(results[0].score, Some(1.0));
    }

    #[tokio::test]
    async fn search_respects_limit() {
        let tmp = NamedTempFile::new().unwrap();
        let mem = RedbMemoryAdapter::open(tmp.path()).unwrap();

        for i in 0..10 {
            mem.store(make_record(&format!("item {i}"))).await.unwrap();
        }

        let results = mem
            .search(MemoryQuery {
                scope: run_scope(),
                mode: QueryMode::Keyword {
                    text: "item".into(),
                },
                limit: 3,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
    }
}
