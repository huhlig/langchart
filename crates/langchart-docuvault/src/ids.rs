//! ArtifactId / ArtifactVersion encoding for vault entities, and the VaultRef schema.
//!
//! ## ArtifactId URI format
//!
//! ```text
//! vault://<vault_id>/file/<file_id>
//! vault://<vault_id>/file/<file_id>/section/<section_id>
//! vault://<vault_id>/file/<file_id>/attachment/<attachment_path>
//! ```
//!
//! A `vault_id` here is the string form of a docuvault `VaultId` ULID
//! (e.g. `"01HX…"`).  `file_id`, `section_id` are similarly formatted ULIDs.
//! `attachment_path` is a URL-encoded vault-relative path segment.
//!
//! ## ArtifactVersion tag format
//!
//! ```text
//! commit:<commit_id_hex>
//! checkpoint:<checkpoint_id_hex>
//! session:<session_id>@<operation_id_hex>
//! ```
//!
//! These strings are stored opaquely inside `langchart_model::id::ArtifactVersion`
//! and parsed by [`VaultVersionRef::parse`] when the adapter needs to resolve them.

use percent_encoding::{AsciiSet, CONTROLS, percent_decode_str, utf8_percent_encode};
use serde::{Deserialize, Serialize};

const ATTACHMENT_PATH_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b':')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

fn decode_attachment_path(encoded: &str) -> Result<String, String> {
    let bytes = encoded.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return Err(format!(
                    "attachment path contains malformed percent encoding: `{encoded}`"
                ));
            }
            index += 3;
        } else {
            index += 1;
        }
    }

    percent_decode_str(encoded)
        .decode_utf8()
        .map(|decoded| decoded.into_owned())
        .map_err(|_| format!("attachment path is not valid UTF-8: `{encoded}`"))
}

// ── ArtifactId URI ─────────────────────────────────────────────────────────────

/// A parsed vault entity reference derived from a `langchart` `ArtifactId` URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultEntityRef {
    pub vault_id: String,
    pub entity: VaultEntity,
}

/// The entity kind within a vault referenced by an `ArtifactId` URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaultEntity {
    /// A Markdown or attachment file addressed by `FileId`.
    File { file_id: String },
    /// A single section within a Markdown file.
    Section { file_id: String, section_id: String },
    /// A binary attachment file, identified by its vault-relative path.
    Attachment {
        file_id: String,
        attachment_path: String,
    },
}

impl VaultEntityRef {
    /// Parse a `vault://…` URI string into a [`VaultEntityRef`].
    ///
    /// # Errors
    ///
    /// Returns a human-readable string on any structural violation.
    pub fn parse(uri: &str) -> Result<Self, String> {
        let rest = uri
            .strip_prefix("vault://")
            .ok_or_else(|| format!("ArtifactId must begin with `vault://`, got: `{uri}`"))?;

        // Split into at most 5 segments: vault_id / kind / file_id [ / sub-kind / sub-id ]
        let mut parts = rest.splitn(5, '/');
        let vault_id = parts.next().ok_or("missing vault_id")?.to_owned();
        let kind = parts.next().ok_or("missing entity kind")?;

        match kind {
            "file" => {
                let file_id = parts.next().ok_or("missing file_id")?.to_owned();
                match parts.next() {
                    None => Ok(VaultEntityRef {
                        vault_id,
                        entity: VaultEntity::File { file_id },
                    }),
                    Some("section") => {
                        let section_id = parts
                            .next()
                            .ok_or("missing section_id after /section/")?
                            .to_owned();
                        Ok(VaultEntityRef {
                            vault_id,
                            entity: VaultEntity::Section {
                                file_id,
                                section_id,
                            },
                        })
                    }
                    Some("attachment") => {
                        let attachment_path = decode_attachment_path(
                            parts
                                .next()
                                .ok_or("missing attachment path after /attachment/")?,
                        )?;
                        Ok(VaultEntityRef {
                            vault_id,
                            entity: VaultEntity::Attachment {
                                file_id,
                                attachment_path,
                            },
                        })
                    }
                    Some(other) => Err(format!("unknown entity sub-kind: `{other}`")),
                }
            }
            other => Err(format!("unknown entity kind: `{other}`")),
        }
    }

    /// Encode this ref back into a `vault://…` URI string.
    pub fn to_uri(&self) -> String {
        match &self.entity {
            VaultEntity::File { file_id } => {
                format!("vault://{}/file/{}", self.vault_id, file_id)
            }
            VaultEntity::Section {
                file_id,
                section_id,
            } => {
                format!(
                    "vault://{}/file/{}/section/{}",
                    self.vault_id, file_id, section_id
                )
            }
            VaultEntity::Attachment {
                file_id,
                attachment_path,
            } => {
                format!(
                    "vault://{}/file/{}/attachment/{}",
                    self.vault_id,
                    file_id,
                    utf8_percent_encode(attachment_path, ATTACHMENT_PATH_ENCODE_SET)
                )
            }
        }
    }
}

// ── ArtifactVersion tag ────────────────────────────────────────────────────────

/// A parsed vault version reference derived from a `langchart` `ArtifactVersion` tag.
///
/// These are stored as opaque strings inside `ArtifactVersion`; the adapter
/// parses them when it needs to resolve a version to a docuvault `CommitRef`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaultVersionRef {
    /// A published commit, identified by its content-addressed commit ID hex string.
    Commit { commit_id: String },
    /// A persisted entity checkpoint, identified by its content-addressed checkpoint ID.
    Checkpoint { checkpoint_id: String },
    /// A session cursor position: session ID + the operation cursor at the time.
    Session {
        session_id: String,
        operation_id: String,
    },
}

impl VaultVersionRef {
    /// Parse a version tag string into a [`VaultVersionRef`].
    pub fn parse(tag: &str) -> Result<Self, String> {
        if let Some(hex) = tag.strip_prefix("commit:") {
            return Ok(VaultVersionRef::Commit {
                commit_id: hex.to_owned(),
            });
        }
        if let Some(hex) = tag.strip_prefix("checkpoint:") {
            return Ok(VaultVersionRef::Checkpoint {
                checkpoint_id: hex.to_owned(),
            });
        }
        if let Some(rest) = tag.strip_prefix("session:") {
            let (session_id, operation_id) = rest.split_once('@').ok_or_else(|| {
                format!("malformed session version tag (expected `session:<id>@<op>`): `{tag}`")
            })?;
            return Ok(VaultVersionRef::Session {
                session_id: session_id.to_owned(),
                operation_id: operation_id.to_owned(),
            });
        }
        Err(format!(
            "unknown version tag prefix (expected `commit:`, `checkpoint:`, or `session:`): `{tag}`"
        ))
    }

    /// Encode this version ref back into its canonical tag string.
    pub fn to_tag(&self) -> String {
        match self {
            VaultVersionRef::Commit { commit_id } => format!("commit:{commit_id}"),
            VaultVersionRef::Checkpoint { checkpoint_id } => {
                format!("checkpoint:{checkpoint_id}")
            }
            VaultVersionRef::Session {
                session_id,
                operation_id,
            } => {
                format!("session:{session_id}@{operation_id}")
            }
        }
    }
}

// ── VaultRef schema ────────────────────────────────────────────────────────────

/// A structured reference to a vault entity stored in workflow data.
///
/// Workflows SHOULD store vault references using this type rather than raw
/// URI strings so that the context resolver can reconstruct `ArtifactId`s
/// deterministically and the observability layer can record exact version pins.
///
/// All fields are optional except `vault_id`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VaultRef {
    /// Vault instance identifier (docuvault `VaultId` string).
    pub vault_id: String,
    /// The published commit this reference was resolved against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<String>,
    /// Stable file identifier (docuvault `FileId` string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// Stable section identifier (docuvault `SectionId` string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_id: Option<String>,
    /// Vault-relative path when this reference identifies a binary attachment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachment_path: Option<String>,
    /// SHA-256 hex content hash of the section body at time of reference.
    /// Used by the observability layer for exact replay identification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_content_hash: Option<String>,
    /// The proposal that was pending against this entity, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_id: Option<String>,
}

impl VaultRef {
    /// Build the `ArtifactId` URI for this reference, if enough fields are present.
    ///
    /// Returns `None` when `file_id` is absent or when both `section_id` and
    /// `attachment_path` are present.
    pub fn to_artifact_id_uri(&self) -> Option<String> {
        let file_id = self.file_id.as_deref()?;
        let entity = match (&self.section_id, &self.attachment_path) {
            (Some(_), Some(_)) => return None,
            (Some(section_id), None) => VaultEntity::Section {
                file_id: file_id.to_owned(),
                section_id: section_id.clone(),
            },
            (None, Some(attachment_path)) => VaultEntity::Attachment {
                file_id: file_id.to_owned(),
                attachment_path: attachment_path.clone(),
            },
            (None, None) => VaultEntity::File {
                file_id: file_id.to_owned(),
            },
        };
        let entity = VaultEntityRef {
            vault_id: self.vault_id.clone(),
            entity,
        };
        Some(entity.to_uri())
    }

    /// Build the `ArtifactVersion` tag for this reference, if a commit is known.
    pub fn to_artifact_version_tag(&self) -> Option<String> {
        self.commit_id.as_deref().map(|id| {
            VaultVersionRef::Commit {
                commit_id: id.to_owned(),
            }
            .to_tag()
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ArtifactId URI round-trips ─────────────────────────────────────────────

    #[test]
    fn file_uri_round_trips() {
        let original = "vault://vault01/file/file01";
        let parsed = VaultEntityRef::parse(original).unwrap();
        assert_eq!(parsed.vault_id, "vault01");
        assert_eq!(
            parsed.entity,
            VaultEntity::File {
                file_id: "file01".into()
            }
        );
        assert_eq!(parsed.to_uri(), original);
    }

    #[test]
    fn section_uri_round_trips() {
        let original = "vault://vault01/file/file01/section/sect01";
        let parsed = VaultEntityRef::parse(original).unwrap();
        assert_eq!(
            parsed.entity,
            VaultEntity::Section {
                file_id: "file01".into(),
                section_id: "sect01".into()
            }
        );
        assert_eq!(parsed.to_uri(), original);
    }

    #[test]
    fn attachment_uri_round_trips() {
        let original = "vault://vault01/file/file01/attachment/assets%2Fdiagram.png";
        let parsed = VaultEntityRef::parse(original).unwrap();
        assert_eq!(
            parsed.entity,
            VaultEntity::Attachment {
                file_id: "file01".into(),
                attachment_path: "assets/diagram.png".into()
            }
        );
        assert_eq!(parsed.to_uri(), original);
    }

    #[test]
    fn attachment_uri_encodes_reserved_and_unicode_characters() {
        let reference = VaultEntityRef {
            vault_id: "vault01".into(),
            entity: VaultEntity::Attachment {
                file_id: "file01".into(),
                attachment_path: "assets/my % diagram-雪.bin".into(),
            },
        };

        let uri = reference.to_uri();
        assert_eq!(
            uri,
            "vault://vault01/file/file01/attachment/assets%2Fmy%20%25%20diagram-%E9%9B%AA.bin"
        );
        assert_eq!(VaultEntityRef::parse(&uri).unwrap(), reference);
    }

    #[test]
    fn attachment_uri_rejects_malformed_percent_encoding() {
        assert!(
            VaultEntityRef::parse("vault://vault01/file/file01/attachment/assets%2Gfile.bin")
                .is_err()
        );
        assert!(
            VaultEntityRef::parse("vault://vault01/file/file01/attachment/assets%FFfile.bin")
                .is_err()
        );
    }

    #[test]
    fn unknown_scheme_returns_error() {
        assert!(VaultEntityRef::parse("http://example.com/file/f1").is_err());
    }

    #[test]
    fn missing_file_id_returns_error() {
        assert!(VaultEntityRef::parse("vault://vault01/file").is_err());
    }

    #[test]
    fn unknown_sub_kind_returns_error() {
        assert!(VaultEntityRef::parse("vault://vault01/file/f1/blob/b1").is_err());
    }

    // ── ArtifactVersion prefix tags ────────────────────────────────────────────

    #[test]
    fn commit_tag_round_trips() {
        let tag = "commit:deadbeef01234";
        let parsed = VaultVersionRef::parse(tag).unwrap();
        assert_eq!(
            parsed,
            VaultVersionRef::Commit {
                commit_id: "deadbeef01234".into()
            }
        );
        assert_eq!(parsed.to_tag(), tag);
    }

    #[test]
    fn checkpoint_tag_round_trips() {
        let tag = "checkpoint:aabbcc";
        let parsed = VaultVersionRef::parse(tag).unwrap();
        assert_eq!(
            parsed,
            VaultVersionRef::Checkpoint {
                checkpoint_id: "aabbcc".into()
            }
        );
        assert_eq!(parsed.to_tag(), tag);
    }

    #[test]
    fn session_tag_round_trips() {
        let tag = "session:sess01@op99";
        let parsed = VaultVersionRef::parse(tag).unwrap();
        assert_eq!(
            parsed,
            VaultVersionRef::Session {
                session_id: "sess01".into(),
                operation_id: "op99".into()
            }
        );
        assert_eq!(parsed.to_tag(), tag);
    }

    #[test]
    fn session_tag_missing_at_sign_returns_error() {
        assert!(VaultVersionRef::parse("session:noop").is_err());
    }

    #[test]
    fn unknown_version_prefix_returns_error() {
        assert!(VaultVersionRef::parse("snapshot:abc").is_err());
    }

    // ── VaultRef JSON serialization ────────────────────────────────────────────

    #[test]
    fn vault_ref_serializes_to_expected_shape() {
        let vr = VaultRef {
            vault_id: "vault01".into(),
            commit_id: Some("c1".into()),
            file_id: Some("f1".into()),
            section_id: Some("s1".into()),
            attachment_path: None,
            section_content_hash: Some("sha256:aabb".into()),
            proposal_id: None,
        };
        let json = serde_json::to_value(&vr).unwrap();
        assert_eq!(json["vault_id"], "vault01");
        assert_eq!(json["section_id"], "s1");
        // proposal_id is None → must be absent (skip_serializing_if)
        assert!(json.get("proposal_id").is_none());
    }

    #[test]
    fn vault_ref_deserializes_with_missing_optional_fields() {
        let json = serde_json::json!({ "vault_id": "vault01" });
        let vr: VaultRef = serde_json::from_value(json).unwrap();
        assert_eq!(vr.vault_id, "vault01");
        assert!(vr.commit_id.is_none());
        assert!(vr.file_id.is_none());
    }

    #[test]
    fn vault_ref_to_artifact_id_uri_needs_file_id() {
        let no_file = VaultRef {
            vault_id: "v1".into(),
            ..Default::default()
        };
        assert!(no_file.to_artifact_id_uri().is_none());

        let with_file = VaultRef {
            vault_id: "v1".into(),
            file_id: Some("f1".into()),
            ..Default::default()
        };
        assert_eq!(
            with_file.to_artifact_id_uri().unwrap(),
            "vault://v1/file/f1"
        );

        let with_section = VaultRef {
            vault_id: "v1".into(),
            file_id: Some("f1".into()),
            section_id: Some("s1".into()),
            ..Default::default()
        };
        assert_eq!(
            with_section.to_artifact_id_uri().unwrap(),
            "vault://v1/file/f1/section/s1"
        );

        let attachment: VaultRef = serde_json::from_value(serde_json::json!({
            "vault_id": "v1",
            "file_id": "f1",
            "attachment_path": "assets/my % diagram-雪.bin"
        }))
        .unwrap();
        assert_eq!(
            attachment.to_artifact_id_uri().unwrap(),
            "vault://v1/file/f1/attachment/assets%2Fmy%20%25%20diagram-%E9%9B%AA.bin"
        );

        let ambiguous: VaultRef = serde_json::from_value(serde_json::json!({
            "vault_id": "v1",
            "file_id": "f1",
            "section_id": "s1",
            "attachment_path": "asset.bin"
        }))
        .unwrap();
        assert!(
            ambiguous.to_artifact_id_uri().is_none(),
            "a reference cannot identify both a section and an attachment"
        );
    }
}
