//! Adapter traits for all external integrations.
//!
//! This module is the bridge between the langchart engine and the outside world.

use async_trait::async_trait;
use langchart_model::id::{ArtifactId, ArtifactVersion, ProposalId};
use serde::{Deserialize, Serialize};

/// A chunk of artifact content returned by the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactContent {
    pub id: ArtifactId,
    pub version: ArtifactVersion,
    /// Raw bytes of the artifact. Interpretation is host-application-specific.
    pub bytes: Vec<u8>,
    /// MIME type hint (e.g. `"text/markdown"`, `"application/json"`).
    pub content_type: String,
}

/// A request to create or modify an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactProposal {
    pub id: ArtifactId,
    /// The artifact version this proposal was based on.
    pub base_version: ArtifactVersion,
    /// The proposed new content.
    pub content: Vec<u8>,
    pub content_type: String,
    /// Human-readable rationale for the change.
    pub rationale: String,
}

/// Summary of a proposal returned by `list_proposals`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalSummary {
    pub proposal_id: ProposalId,
    pub artifact_id: ArtifactId,
    pub base_version: ArtifactVersion,
    pub rationale: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("artifact `{0}` not found")]
    NotFound(ArtifactId),
    #[error("version conflict: expected `{expected}`, found `{actual}`")]
    VersionConflict {
        expected: ArtifactVersion,
        actual: ArtifactVersion,
    },
    #[error("proposal `{proposal_id}` does not belong to artifact `{artifact_id}`")]
    ProposalArtifactMismatch {
        proposal_id: ProposalId,
        artifact_id: ArtifactId,
    },
    #[error("artifact store error: {0}")]
    Store(String),
}

/// Manages versioned artifacts: reads, proposals, and commits.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn read(
        &self,
        id: &ArtifactId,
        version: Option<&ArtifactVersion>,
    ) -> Result<ArtifactContent, ArtifactError>;

    async fn propose(&self, proposal: ArtifactProposal) -> Result<ProposalId, ArtifactError>;

    async fn commit(
        &self,
        artifact_id: &ArtifactId,
        proposal_id: &ProposalId,
        expected_base: &ArtifactVersion,
    ) -> Result<ArtifactVersion, ArtifactError>;

    async fn list_proposals(
        &self,
        artifact_id: &ArtifactId,
    ) -> Result<Vec<ProposalSummary>, ArtifactError>;
}
