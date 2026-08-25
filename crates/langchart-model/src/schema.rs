//! Schema versioning and forward-compatible document loading.
//!
//! The schema version embedded in a workflow document determines which fields
//! are understood by this version of the library. Unknown fields in a
//! forward-compatible minor version are preserved in `_unknown`.

use crate::error::LoadError;
use semver::{Version, VersionReq};

/// The library's current supported schema version.
pub const LIBRARY_SCHEMA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The schema version requirement this library accepts.
/// Major version must match; minor and patch may be newer (forward-compatible reads).
const ACCEPTED_SCHEMA_REQ: &str = "^1";

/// Check whether a document's declared `schema_version` is accepted by
/// this library. Returns `Ok(())` if compatible, `Err(LoadError::UnsupportedVersion)`
/// if the major version differs or the string is not valid semver.
pub fn check_schema_version(declared: &str) -> Result<(), LoadError> {
    let version = Version::parse(declared).map_err(|_| LoadError::UnsupportedVersion {
        found: declared.to_owned(),
        expected: ACCEPTED_SCHEMA_REQ.to_owned(),
    })?;

    let req = VersionReq::parse(ACCEPTED_SCHEMA_REQ).expect("hardcoded req is valid");

    if !req.matches(&version) {
        return Err(LoadError::UnsupportedVersion {
            found: declared.to_owned(),
            expected: ACCEPTED_SCHEMA_REQ.to_owned(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_version_accepted() {
        assert!(check_schema_version("1.0.0").is_ok());
    }

    #[test]
    fn newer_minor_accepted() {
        assert!(check_schema_version("1.5.0").is_ok());
    }

    #[test]
    fn different_major_rejected() {
        assert!(check_schema_version("2.0.0").is_err());
    }

    #[test]
    fn invalid_string_rejected() {
        assert!(check_schema_version("not-semver").is_err());
    }
}
